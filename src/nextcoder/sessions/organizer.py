"""后台记忆整理 agent：合并重复、停用过时、补提取短会话里的事实。

触发时机：会话第一轮（session init），或同一会话每累计 N 轮用户请求
（按用户请求轮数计，不是工具轮数）。整理在后台线程执行，不阻塞主循环；
失败只写状态文件，不改任何记忆。

提取覆盖的是“没等到整理就被放下的会话”：少于 N 轮的会话永远不会触发
自动整理，它们的请求和结论就没人看过。所以提取每次只看最旧、且还没
审阅过的 3 个会话，把它们尚未审阅的完整轮次读一遍；审阅过的会话跳过，
不重复处理。手工条目与用户停用状态照旧受保护；提取不改写已有条目。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from nextcoder.llm import build_chat_llm
from nextcoder.sessions.memory import KIND_LABELS, ProjectMemory
from nextcoder.sessions.rollout import read_events
from nextcoder.settings import Settings

# 同一进程内最多一个整理线程在跑；重入直接跳过，不排队。
_ORGANIZER_LOCK = threading.Lock()

MAX_ACTIONS = 5
MAX_EXTRACTS = 3
# 每次补提取最多看 3 个旧会话，每个会话最多看 5 轮。
MAX_SESSIONS = 3
MAX_TURNS_PER_SESSION = 5
# 每轮带工具调用与命令输出尾部；预算按轮、按会话、按整次整理三层限制。
MAX_OPS_PER_TURN = 8
MAX_TURN_EVIDENCE = 3500
MAX_SESSION_EVIDENCE = 9000
MAX_HISTORY_EVIDENCE = 24000
VERIFY_HINTS = (
    "pytest", "unittest", "tox", "nox", "npm test", "yarn test",
    "pnpm test", "go test", "cargo test", "make test", "ruff", "mypy",
    "flake8", "eslint", "black --check", "git diff --check",
)


class MemoryOrganizer:
    """一次模型调用完成三类受限操作：合并、停用、从旧会话补提取。"""

    def __init__(self, settings: Settings, llm=None, history: str = "") -> None:
        self._settings = settings
        # 与主循环同一入口/会话头/备援策略，行为可从配置解释。
        self._llm = llm or build_chat_llm(settings)
        self._history = history

    def organize(self, memory: ProjectMemory) -> dict:
        entries = memory.select(limit=1000)
        # 没有记忆但有旧会话待审阅：仍要跑，提取不依赖已有条目。
        if len(entries) < 2 and not self._history:
            return {"status": "skipped", "reason": "没有可整理的记忆，也没有待提取的旧会话"}
        response = self._llm.invoke(_organizer_prompt(entries, self._history))
        text = _content_text(getattr(response, "content", ""))
        plan = _extract_json(text)
        merges = [item for item in plan.get("merges", [])
                  if isinstance(item, dict)][:MAX_ACTIONS]
        disables = [item for item in plan.get("disables", [])
                    if isinstance(item, dict)][:MAX_ACTIONS]
        extracts = [item for item in plan.get("extracts", [])
                    if isinstance(item, dict)][:MAX_EXTRACTS]
        applied = _apply_plan(memory, merges, disables, extracts)
        applied["status"] = "applied"
        applied["usage"] = _usage(getattr(response, "usage_metadata", None))
        return applied


def collect_pending_history(memory: ProjectMemory, settings: Settings,
                            task_dir: Path,
                            max_sessions: int = MAX_SESSIONS,
                            max_turns: int = MAX_TURNS_PER_SESSION
                            ) -> tuple[str, list[tuple[str, int, int]]]:
    """取最旧、尚未审阅过的会话轮次，作为提取新记忆的输入。

    返回 (文本, 进度表)。进度表每项是 (会话 id, 审阅到第几轮, 总轮数)，
    整理成功后由调用方写回 memory_sessions；失败则不写，下次重看。
    没有任何待审阅内容时返回空文本。

    每轮不只给请求和收尾，还从会话的 operations.sqlite 里取工具调用、
    命令参数、退出码和输出尾部；测试/检查命令的输出优先保留。
    """
    if max_sessions <= 0 or max_turns <= 0:
        return "", []
    root = task_dir / settings.runs_root / "sessions"
    if not root.is_dir():
        return "", []
    blocks: list[str] = []
    progress: list[tuple[str, int, int]] = []
    total_chars = 0
    for session_dir in sorted(root.iterdir()):  # 目录名以时间开头，旧的在前
        rollout = session_dir / "rollout.jsonl"
        if not rollout.is_file():
            continue
        turns = _complete_turns(rollout)
        done = memory.session_processed_turns(session_dir.name)
        fresh = turns[done:]
        if not fresh:
            continue
        header = f"## 会话 {session_dir.name}（第 {done + 1} 轮起）"
        turn_blocks: list[str] = []
        session_chars = 0
        for offset, turn in enumerate(fresh[:max_turns], start=done + 1):
            block = _format_turn(offset, turn, session_dir)
            cost = len(block) + 1
            if turn_blocks and (
                    total_chars + session_chars + cost > MAX_HISTORY_EVIDENCE
                    or session_chars + cost > MAX_SESSION_EVIDENCE):
                break
            turn_blocks.append(block)
            session_chars += cost
        if not turn_blocks:
            break
        blocks.append(header)
        blocks.extend(turn_blocks)
        total_chars += session_chars + len(header) + 1
        progress.append((session_dir.name, done + len(turn_blocks), len(turns)))
        if len(progress) >= max_sessions or total_chars >= MAX_HISTORY_EVIDENCE:
            break
    if not blocks:
        return "", []
    header = ("以下是还没有人整理过的旧会话轮次，最旧的在前。每轮带用户请求、"
              "模型收尾、工具调用和命令输出尾部；提取只能从这里取材，"
              "evidence 必须引用原文、命令或输出。")
    return "\n".join([header, *blocks]), progress


def _complete_turns(rollout: Path) -> list[dict]:
    """把 rollout 事件按轮次归并，只返回已经收尾（有 turn_end）的轮次。"""
    turns: list[dict] = []
    current: dict | None = None
    for event in read_events(rollout):
        etype = event.get("type")
        payload = event.get("payload") or {}
        if etype == "user_request":
            current = {"request": str(payload.get("request") or "")[:500],
                       "wrap": "", "end": None,
                       "turn_id": str(event.get("turn_id") or "")}
        elif etype == "assistant" and current is not None:
            content = str(payload.get("content") or "")
            if content.strip():
                current["wrap"] = content[:500]
        elif etype == "turn_end":
            if current is not None:
                current["end"] = payload
                if not current.get("turn_id"):
                    current["turn_id"] = str(event.get("turn_id") or "")
                turns.append(current)
            current = None
    return turns


def _format_turn(index: int, turn: dict, session_dir: Path) -> str:
    end = turn.get("end") or {}
    operations = _load_operations(session_dir, str(turn.get("turn_id") or ""))
    changes = "有" if end.get("has_changes") else "无"
    lines = [f"### 第 {index} 轮",
             f"请求：{turn.get('request', '')}",
             f"结果：stop={end.get('stop_reason')} calls={end.get('model_calls')} "
             f"改动={changes}"]
    files = sorted({name for op in operations for name in op["related_files"]})
    if files:
        lines.append("改动文件：" + "、".join(files[:8]))
    if turn.get("wrap"):
        lines.append(f"模型收尾：{turn['wrap']}")
    budget = max(0, MAX_TURN_EVIDENCE - len("\n".join(lines)))
    op_lines = _format_operations(operations, budget)
    if op_lines:
        lines.append("工具调用：")
        lines.extend(op_lines)
    else:
        lines.append("工具调用：（没有记录）")
    text = "\n".join(lines)
    if len(text) > MAX_TURN_EVIDENCE:
        text = text[:MAX_TURN_EVIDENCE] + "\n…（本轮证据过长，已截断）"
    return text


def _load_operations(session_dir: Path, turn_id: str) -> list[dict]:
    """从会话的操作审计库取这一轮的工具调用；库不存在时返回空表。"""
    db = session_dir / "operations.sqlite"
    if not turn_id or not db.is_file():
        return []
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return []
    try:
        rows = conn.execute(
            "SELECT tool, args_json, status, exit_code, error, log_path,"
            " result_text, related_files_json FROM operations"
            " WHERE turn_id=? ORDER BY created_at", (turn_id,)).fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        conn.close()
    operations: list[dict] = []
    for row in rows:
        try:
            args = json.loads(row["args_json"] or "{}")
        except json.JSONDecodeError:
            args = {}
        try:
            files = json.loads(row["related_files_json"] or "[]")
        except json.JSONDecodeError:
            files = []
        operations.append({
            "tool": str(row["tool"] or ""),
            "args": args if isinstance(args, dict) else {},
            "status": str(row["status"] or ""),
            "exit_code": row["exit_code"],
            "error": row["error"],
            "log_path": row["log_path"],
            "result_text": row["result_text"],
            "related_files": [str(name) for name in files if name],
            "session_dir": str(session_dir),
        })
    return operations


def _format_operations(operations: list[dict], budget: int) -> list[str]:
    """按“失败 > 验证命令 > 写文件 > 其他命令 > 读查”排序，装进预算。"""
    if not operations or budget <= 0:
        return []
    ranked = sorted(enumerate(operations),
                    key=lambda item: (_operation_priority(item[1]), item[0]))
    lines: list[str] = []
    used = 0
    for _, operation in ranked[:MAX_OPS_PER_TURN]:
        summary = _operation_summary(operation)
        output = _operation_output(operation)
        block = summary if not output else f"{summary}\n  输出：{output}"
        if used + len(block) + 1 > budget:
            if not lines:
                lines.append(block[:budget])
            break
        lines.append(block)
        used += len(block) + 1
    return lines


def _operation_priority(operation: dict) -> int:
    if _operation_bad(operation):
        return 0
    if _operation_verify(operation):
        return 1
    if operation["tool"] in {"apply_patch", "write_file", "edit_file"}:
        return 2
    if operation["tool"] == "run_command":
        return 3
    return 4


def _operation_bad(operation: dict) -> bool:
    return bool(
        operation.get("error")
        or operation.get("status") in {"failed", "timeout", "cancelled", "unknown"}
        or (operation.get("exit_code") not in (None, 0))
    )


def _operation_verify(operation: dict) -> bool:
    if operation["tool"] != "run_command":
        return False
    command = str(operation["args"].get("command") or "").lower()
    return any(hint in command for hint in VERIFY_HINTS)


def _operation_summary(operation: dict) -> str:
    tool = operation["tool"]
    args = operation["args"]
    target = ""
    if tool == "run_command":
        target = str(args.get("command") or "")
    elif tool in {"read_file", "write_file", "apply_patch", "edit_file", "delete_path"}:
        target = str(args.get("path") or "")
    elif tool in {"search", "grep", "find", "glob"}:
        target = str(args.get("query") or args.get("pattern") or args.get("path") or "")
    elif tool in {"memory_read", "memory_write", "memory_forget"}:
        target = str(args.get("key") or args.get("query") or "")
    if not target and operation["related_files"]:
        target = "、".join(operation["related_files"][:3])
    target = " ".join(target.split())[:300]
    bits = [f"- {tool} {operation['status']}"]
    if operation.get("exit_code") is not None:
        bits.append(f"exit={operation['exit_code']}")
    if operation.get("error"):
        bits.append(f"错误={str(operation['error'])[:200]}")
    line = " ".join(bits)
    return f"{line}：{target}" if target else line


def _operation_output(operation: dict) -> str:
    limit = 1200 if _operation_verify(operation) else (
        800 if _operation_bad(operation) else 300)
    text = str(operation.get("result_text") or "").strip()
    if not text and operation.get("log_path"):
        path = Path(str(operation["log_path"]))
        if not path.is_absolute():
            path = Path(str(operation.get("session_dir") or "")) / path
        text = _tail(path, limit * 4).strip()
    if not text:
        return ""
    if len(text) > limit:
        text = "…" + text[-limit:]
    return text


def _tail(path: Path, chars: int) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > chars:
                handle.seek(size - chars)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _organizer_prompt(entries: list, history: str = "") -> str:
    lines = [
        "你在整理一个编程助手的项目记忆库。下面是当前全部有效记忆，"
        "请只做三类操作：",
        "1. merges：把同一主题的重复或碎片条目合并成一条。"
        "from=被合并条目的 key，into=保留条目的 key，value=合并后的完整正文，"
        "kind=七种类型之一。",
        "2. disables：停用明显过时、互相矛盾或已无价值的条目，"
        "reason 用一句话说明。",
        "3. extracts：从下方旧会话轮次里提取值得跨会话保留的新事实，"
        "限于项目约定、环境处理、测试/检查命令和已验证的修复经验；"
        "topic=简短主题，body=完整正文。evidence 必须引用请求原文、实际执行的"
        "命令或输出尾部，不能只写“收尾里提到”；没有命令或输出支撑的猜测"
        "不要提取。",
        "与现有条目同主题的不要提取；这些旧会话审阅过后不会再提供，"
        "只提取确实有复用价值的。",
        "没有值得做的操作就返回空数组。规则：merges/disables 各最多 5 条、"
        "extracts 最多 3 条；标记 [manual] 的条目绝不能出现在输出里；"
        "正文改写时保留关键事实；只输出 JSON，格式：",
        '{"merges": [{"from": "...", "into": "...", "value": "...", "kind": "..."}],'
        ' "disables": [{"key": "...", "reason": "..."}],'
        ' "extracts": [{"topic": "...", "kind": "...", "body": "...",'
        ' "evidence": "..."}]}',
        "",
        "当前记忆：",
    ]
    for entry in entries:
        manual = "[manual] " if entry.meta.get("manual") else ""
        lines.append(f"- {entry.key} [{entry.kind}] {manual}conf={entry.confidence:.2f}"
                     f" source={entry.source}\n  {entry.value}")
    if not entries:
        lines.append("（暂无记忆）")
    if history:
        lines.extend(["", "旧会话轮次：", history])
    else:
        lines.extend(["", "旧会话轮次：（无，extracts 必须为空）"])
    return "\n".join(lines)


def _apply_plan(memory: ProjectMemory, merges: list[dict],
                disables: list[dict], extracts: list[dict] | None = None) -> dict:
    merged = disabled = extracted = skipped = 0
    source = f"organizer:{int(time.time())}"
    for item in merges:
        src, dst = str(item.get("from") or ""), str(item.get("into") or "")
        value = " ".join(str(item.get("value") or "").split())
        if not src or not dst or src == dst or not value:
            skipped += 1
            continue
        src_row, dst_row = memory.get(src), memory.get(dst)
        if src_row is None or dst_row is None:
            skipped += 1
            continue
        # manual 条目两边都不能碰：既不被吞并，也不被改写。
        if src_row.meta.get("manual") or dst_row.meta.get("manual"):
            skipped += 1
            continue
        kind = str(item.get("kind") or dst_row.kind)
        if kind not in KIND_LABELS:
            kind = dst_row.kind
        memory.remember(dst, value, kind=kind, source=source, confidence=0.8)
        memory.invalidate(src, f"整理合并到 {dst}", source=source)
        merged += 1
    for item in disables:
        key = str(item.get("key") or "")
        row = memory.get(key) if key else None
        if row is None or row.meta.get("manual"):
            skipped += 1
            continue
        reason = " ".join(str(item.get("reason") or "整理判定过时").split())[:200]
        if memory.invalidate(key, reason, source=source):
            disabled += 1
        else:
            skipped += 1
    for item in extracts or []:
        topic = " ".join(str(item.get("topic") or "").split())[:200]
        body = " ".join(str(item.get("body") or "").split())[:2000]
        if not topic or not body:
            skipped += 1
            continue
        # 已有 key 一律跳过：既不重复，也不借整理复活用户停用的条目。
        if memory.get(topic) is not None:
            skipped += 1
            continue
        kind = str(item.get("kind") or "project_convention")
        if kind not in KIND_LABELS:
            kind = "project_convention"
        evidence = " ".join(str(item.get("evidence") or "").split())[:500]
        memory.remember(topic, body, kind=kind, source=source, confidence=0.6,
                        meta={"evidence": evidence, "extracted_by": "organizer"})
        extracted += 1
    return {"merged": merged, "disabled": disabled,
            "extracted": extracted, "skipped": skipped}


def should_organize(turn_count: int, interval: int) -> bool:
    """会话第一轮整理一次；之后每累计 interval 轮用户请求再整理。"""
    if turn_count <= 0 or interval <= 0:
        return False
    return turn_count == 1 or turn_count % interval == 0


def spawn_organizer(settings: Settings, task_dir: Path) -> bool:
    """后台线程执行整理；返回是否实际启动。已在线程里时直接拒绝。"""
    if not settings.memory_enabled or not settings.memory_organizer_enabled:
        return False
    if not _ORGANIZER_LOCK.acquire(blocking=False):
        return False

    def _run() -> None:
        try:
            result = organize_project_memory(settings, task_dir)
            _write_status(task_dir, settings, result)
        except Exception as exc:  # noqa: BLE001 - 后台失败不影响主循环
            _write_status(task_dir, settings,
                          {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            _ORGANIZER_LOCK.release()

    threading.Thread(target=_run, name="memory-organizer", daemon=True).start()
    return True


def organize_project_memory(settings: Settings, task_dir: Path,
                            llm=None) -> dict:
    """独立打开记忆库执行整理；线程安全，主循环不受影响。"""
    memory = ProjectMemory(settings, task_dir)
    try:
        history, pending = collect_pending_history(memory, settings, task_dir)
        result = MemoryOrganizer(settings, llm=llm, history=history).organize(memory)
        # 只有模型调用和写入都成功才记进度；失败则下次重看这些轮次。
        if result.get("status") == "applied" and pending:
            for session_id, processed, total in pending:
                memory.mark_session_processed(session_id, processed, total)
            result["sessions_reviewed"] = [session_id for session_id, _, _ in pending]
        return result
    finally:
        memory.close()


def _write_status(task_dir: Path, settings: Settings, result: dict) -> None:
    path = task_dir / settings.runs_root / "organizer-last.json"
    payload = dict(result)
    payload["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    except OSError:
        pass  # 状态文件写不进去不影响整理本身


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part) for part in content)
    return str(content or "")


def _extract_json(text: str) -> dict:
    """从模型输出中抠出 JSON 对象；解析失败返回空 dict（apply 层会全跳过）。"""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _usage(metadata) -> dict[str, int]:
    if not isinstance(metadata, dict):
        return {}
    return {key: int(metadata.get(key, 0))
            for key in ("input_tokens", "output_tokens", "total_tokens")}
