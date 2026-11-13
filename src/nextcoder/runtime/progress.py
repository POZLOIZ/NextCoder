"""进展检测：识别反复空修改、文件状态回退与长时间不验证。

检测只读取消息历史里的工具调用和结果，不额外调用模型。判定分三级：

- nudge:  只提醒，继续执行；
- replan: 注入重新规划要求，并短暂阻止对同一文件的编辑；
- stop:   重新规划后仍无新文件状态/测试结果时提前结束。

回退本身可能是正常修正，因此单次回退不触发；同一状态第 3 次出现才要求
重新规划，第 4 次且期间没有新证据才停止。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from langchain_core.messages import AIMessage, ToolMessage

EDIT_TOOLS = {"apply_patch", "write_file"}
HASH_RE = re.compile(r"content_hash=([0-9a-f]{40,64})")
STATUS_RE = re.compile(r"status=(\w+)")
NOOP_RE = re.compile(
    r"old_text\s*与\s*new_text\s*相同|old_text.*new_text.*(?:same|identical)",
    re.IGNORECASE,
)
TEST_RE = re.compile(r"\b(pytest|py\.test|bin/test|runtests|tox|nox)\b")


@dataclass(frozen=True)
class LoopLimits:
    noop_replan_limit: int = 2
    rollback_replan_count: int = 3
    rollback_stop_count: int = 4
    edit_gap_nudge: int = 8
    exact_replan_limit: int = 5
    exact_stop_count: int = 6
    block_calls: int = 1
    stop_after_calls: int = 8


@dataclass(frozen=True)
class EditEvent:
    index: int
    path: str
    successful: bool
    post_hash: str | None
    noop: bool


@dataclass
class Timeline:
    edits: list[EditEvent] = field(default_factory=list)
    tests: list[int] = field(default_factory=list)
    exact_streak: int = 0
    states: dict[str, list[str]] = field(default_factory=dict)
    rollback_path: str = ""
    rollback_hash: str = ""
    rollback_count: int = 0
    rollback_at: int | None = None
    noop_streak: int = 0
    noop_path: str = ""
    exact_signature: str = ""


@dataclass(frozen=True)
class LoopDecision:
    kind: str | None = None
    message: str = ""
    # 需要写回检查点的字段；空值表示清除。
    replan_key: str = ""
    replan_at: int = 0
    replan_path: str = ""
    replan_signature: str = ""
    replan_evidence: tuple[int, int] = (0, 0)
    block_until: int = 0
    gap_nudged: bool = False
    gap_test_count: int = 0
    end_run: bool = False


def call_signature(call: Mapping[str, Any]) -> str:
    return json.dumps(
        {"name": call.get("name", ""), "args": call.get("args") or {}},
        ensure_ascii=False,
        sort_keys=True,
    )


def calls_signature(calls: list[Mapping[str, Any]]) -> str:
    """一组并行工具的签名；顺序也参与比较，A+B 与 B+A 视为不同组合。"""
    return json.dumps(
        [call_signature(call) for call in calls],
        ensure_ascii=False,
    )


def matches_repeated_call(call: Mapping[str, Any], blob: str) -> bool:
    """判断单个调用是否属于被判定为重复的那组调用。"""
    if not blob:
        return False
    try:
        items = json.loads(blob)
    except (TypeError, ValueError):
        return False
    return isinstance(items, list) and call_signature(call) in items


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_timeline(messages: list) -> Timeline:
    timeline = Timeline()
    pending: dict[str, tuple[str, dict]] = {}
    signatures: list[str] = []

    for index, message in enumerate(messages):
        if isinstance(message, AIMessage):
            calls = list(getattr(message, "tool_calls", None) or [])
            if calls:
                signatures.append(
                    json.dumps(
                        [call_signature(call) for call in calls],
                        ensure_ascii=False,
                    )
                )
            for call in calls:
                pending[call.get("id") or ""] = (
                    call.get("name") or "",
                    call.get("args") or {},
                )
            continue
        if not isinstance(message, ToolMessage):
            continue
        name, args = pending.pop(message.tool_call_id or "", ("", {}))
        text = str(message.content or "")
        if name in EDIT_TOOLS:
            status = STATUS_RE.search(text)
            successful = bool(status and status.group(1) == "completed")
            post = HASH_RE.search(text)
            path = str(args.get("path") or "")
            event = EditEvent(
                index=index,
                path=path,
                successful=successful,
                post_hash=post.group(1) if post else None,
                noop=bool(NOOP_RE.search(text)),
            )
            timeline.edits.append(event)
            if event.post_hash and path:
                seq = timeline.states.setdefault(path, [])
                if not seq or seq[-1] != event.post_hash:
                    seq.append(event.post_hash)
            if event.noop:
                timeline.noop_streak += 1
                timeline.noop_path = path
            else:
                timeline.noop_streak = 0
                timeline.noop_path = ""
        elif name == "run_command":
            command = str(args.get("command") or "")
            if TEST_RE.search(command):
                timeline.tests.append(index)
                # 新的测试结果会打断“连续空修改”；它本身就是新证据。
                timeline.noop_streak = 0
                timeline.noop_path = ""

    # 末尾连续相同的工具调用（与 runtime/hooks.py 的重复调用 hook 同口径）。
    streak = 0
    previous = None
    for signature in reversed(signatures):
        if previous is None:
            previous, streak = signature, 1
        elif signature == previous:
            streak += 1
        else:
            break
    timeline.exact_streak = streak
    timeline.exact_signature = previous or ""

    counts: dict[tuple[str, str], int] = {}
    for event in timeline.edits:
        if not event.post_hash:
            continue
        key = (event.path, event.post_hash)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > timeline.rollback_count:
            timeline.rollback_count = counts[key]
            timeline.rollback_path = event.path
            timeline.rollback_hash = event.post_hash
    if timeline.rollback_count:
        seen = 0
        for event in timeline.edits:
            if (event.path == timeline.rollback_path
                    and event.post_hash == timeline.rollback_hash):
                seen += 1
                if seen == timeline.rollback_count:
                    timeline.rollback_at = event.index
    return timeline


def _edits_since_test(timeline: Timeline) -> int:
    last_test = max(timeline.tests, default=-1)
    return sum(1 for event in timeline.edits if event.index > last_test)


def _evidence(timeline: Timeline) -> tuple[int, int]:
    hashes = {
        (event.path, event.post_hash)
        for event in timeline.edits
        if event.post_hash
    }
    return len(hashes), len(timeline.tests)


def _state_sequence(timeline: Timeline, path: str, limit: int = 8) -> str:
    states = timeline.states.get(path) or []
    tail = states[-limit:]
    return " -> ".join(state[:8] for state in tail)


def inspect_loop(
    messages: list,
    *,
    model_calls: int,
    state: Mapping[str, Any],
    limits: LoopLimits | None = None,
) -> LoopDecision:
    limits = limits or LoopLimits()
    timeline = build_timeline(messages)
    evidence = _evidence(timeline)
    gap = _edits_since_test(timeline)

    replan_key = str(state.get("loop_replan_key") or "")
    replan_at = _int(state.get("loop_replan_at"))
    replan_path = str(state.get("loop_replan_path") or "")
    replan_signature = str(state.get("loop_replan_signature") or "")
    replan_evidence = tuple(state.get("loop_replan_evidence") or (0, 0))
    block_until = _int(state.get("loop_block_until"))
    gap_nudged = bool(state.get("loop_gap_nudged"))
    gap_test_count = _int(state.get("loop_gap_test_count"))

    # 提醒之后出现了新的测试结果，说明编辑-验证节奏已经恢复；允许下次再提醒。
    if len(timeline.tests) > gap_test_count:
        gap_nudged = False

    # 出现新的文件状态或新的测试结果，说明重新规划后确实换了做法，解除旧标记。
    if replan_key and evidence != tuple(replan_evidence):
        replan_key = ""
        replan_at = 0
        replan_path = ""
        replan_signature = ""
        replan_evidence = (0, 0)
        block_until = 0

    after_replan = (
        bool(replan_key)
        and model_calls - replan_at >= limits.stop_after_calls
        and evidence == tuple(replan_evidence)
    )
    if after_replan:
        if (timeline.rollback_count >= limits.rollback_stop_count
                and timeline.rollback_path == replan_path):
            return LoopDecision(
                kind="stop",
                end_run=True,
                message=(
                    "[运行停止：检测到循环] 文件 "
                    f"{timeline.rollback_path} 第 {timeline.rollback_count} 次"
                    f"回到状态 {timeline.rollback_hash[:8]}；重新规划后"
                    f"{model_calls - replan_at} 次调用内没有新的文件状态或测试结果。"
                    "保留当前补丁和失败证据，提前结束。"
                ),
                replan_key=replan_key,
                replan_at=replan_at,
                replan_path=replan_path,
                replan_signature=replan_signature,
                replan_evidence=tuple(replan_evidence),
                block_until=block_until,
                gap_nudged=gap_nudged,
                gap_test_count=gap_test_count,
            )
        if (timeline.noop_streak >= limits.noop_replan_limit + 1
                and timeline.noop_path == replan_path):
            return LoopDecision(
                kind="stop",
                end_run=True,
                message=(
                    "[运行停止：检测到循环] 文件 "
                    f"{timeline.noop_path} 在重新规划后仍连续 "
                    f"{timeline.noop_streak} 次提交空修改；"
                    "期间没有新的文件状态或测试结果。保留当前补丁，提前结束。"
                ),
                replan_key=replan_key,
                replan_at=replan_at,
                replan_path=replan_path,
                replan_signature=replan_signature,
                replan_evidence=tuple(replan_evidence),
                block_until=block_until,
                gap_nudged=gap_nudged,
                gap_test_count=gap_test_count,
            )
        if timeline.exact_streak >= limits.exact_stop_count:
            return LoopDecision(
                kind="stop",
                end_run=True,
                message=(
                    "[运行停止：检测到循环] 重新规划后仍连续 "
                    f"{timeline.exact_streak} 次发起相同调用，"
                    "且没有新的文件状态或测试结果。保留当前补丁，提前结束。"
                ),
                replan_key=replan_key,
                replan_at=replan_at,
                replan_path=replan_path,
                replan_signature=replan_signature,
                replan_evidence=tuple(replan_evidence),
                block_until=block_until,
                gap_nudged=gap_nudged,
                gap_test_count=gap_test_count,
            )

    if timeline.rollback_count >= limits.rollback_replan_count:
        key = f"rollback:{timeline.rollback_path}:{timeline.rollback_hash[:12]}"
        if key != replan_key:
            sequence = _state_sequence(timeline, timeline.rollback_path)
            return LoopDecision(
                kind="replan",
                message=(
                    "[循环干预] 检测到同一文件反复回到旧状态，继续提交同类补丁"
                    "不会产生新信息。\n"
                    f"文件：{timeline.rollback_path}\n"
                    f"状态序列（旧到新，前 8 位）：{sequence}\n"
                    f"状态 {timeline.rollback_hash[:8]} 已出现 "
                    f"{timeline.rollback_count} 次；最近 {gap} 次编辑没有运行测试。\n"
                    "先停止修改这个文件，回答并执行：\n"
                    "1. 当前失败的具体原因是什么（引用错误或测试输出）；\n"
                    "2. 已经尝试过哪些修改，各自为什么失败；\n"
                    "3. 下一次修改与之前有什么不同。\n"
                    f"接下来 {limits.block_calls} 次对该文件的编辑会被拒绝。"
                ),
                replan_key=key,
                replan_at=model_calls,
                replan_path=timeline.rollback_path,
                replan_evidence=evidence,
                block_until=model_calls + limits.block_calls,
                gap_nudged=gap_nudged,
                gap_test_count=gap_test_count,
            )

    if timeline.noop_streak >= limits.noop_replan_limit:
        key = f"noop:{timeline.noop_path}"
        if key != replan_key:
            return LoopDecision(
                kind="replan",
                message=(
                    "[循环干预] 连续 "
                    f"{timeline.noop_streak} 次提交了 old_text 与 new_text "
                    "相同的空修改，文件没有发生变化。\n"
                    f"文件：{timeline.noop_path}\n"
                    "先 read_lines 读取当前文件，确认目标内容是否已经存在；"
                    "如果已经存在，直接运行验证；如果不存在，给出与当前内容"
                    "不同的修改。\n"
                    f"接下来 {limits.block_calls} 次对该文件的编辑会被拒绝。"
                ),
                replan_key=key,
                replan_at=model_calls,
                replan_path=timeline.noop_path,
                replan_evidence=evidence,
                block_until=model_calls + limits.block_calls,
                gap_nudged=gap_nudged,
                gap_test_count=gap_test_count,
            )

    if timeline.exact_streak >= limits.exact_replan_limit:
        digest = hashlib.sha1(
            timeline.exact_signature.encode("utf-8")).hexdigest()[:12]
        key = f"exact:{digest}"
        if key != replan_key:
            return LoopDecision(
                kind="replan",
                message=(
                    "[循环干预] 你已经连续 "
                    f"{timeline.exact_streak} 次发起完全相同的调用：\n"
                    f"{timeline.exact_signature[:300]}\n"
                    "继续重复不会产生新信息。先停止重复，回答并执行：\n"
                    "1. 这个调用想确认什么；\n"
                    "2. 上一次结果是否已经回答了它；\n"
                    "3. 下一步换成 read_lines、run_command 或 apply_patch 中的哪一种。\n"
                    f"接下来 {limits.block_calls} 次完全相同的调用会被拒绝。"
                ),
                replan_key=key,
                replan_at=model_calls,
                replan_path="",
                replan_signature=timeline.exact_signature,
                replan_evidence=evidence,
                block_until=model_calls + limits.block_calls,
                gap_nudged=gap_nudged,
                gap_test_count=gap_test_count,
            )

    if gap >= limits.edit_gap_nudge and not gap_nudged:
        return LoopDecision(
            kind="nudge",
            message=(
                f"[进展提醒] 已经连续 {gap} 次编辑没有运行测试。"
                "如果修改已经成形，先运行被改模块的相关测试确认没有回归；"
                "如果还在试探，说明当前缺什么证据，避免继续盲改。"
            ),
            replan_key=replan_key,
            replan_at=replan_at,
            replan_path=replan_path,
            replan_signature=replan_signature,
            replan_evidence=tuple(replan_evidence),
            block_until=block_until,
            gap_nudged=True,
            gap_test_count=len(timeline.tests),
        )

    return LoopDecision(
        replan_key=replan_key,
        replan_at=replan_at,
        replan_path=replan_path,
        replan_signature=replan_signature,
        replan_evidence=tuple(replan_evidence),
        block_until=block_until,
        gap_nudged=gap_nudged,
        gap_test_count=gap_test_count,
    )
