"""会话管理：同一任务目录下多轮共用检查点、基线、操作记录与日志。"""

import json
import re
import signal
import sys
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from nextcoder.llm import endpoint_snapshot
from nextcoder.runtime.agent import AgentRunner
from nextcoder.runtime.delegation import DelegationManager
from nextcoder.runtime.events import TurnFinished, TurnStarted, emit
from nextcoder.runtime.instructions import load_agents_file
from nextcoder.runtime.patch import copy_baseline, git_diff_patch
from nextcoder.runtime.provenance import runtime_provenance
from nextcoder.runtime.skills import SkillCatalog, discover_skills
from nextcoder.sessions.memory import MemorySnapshot, ProjectMemory
from nextcoder.sessions.organizer import should_organize, spawn_organizer
from nextcoder.sessions.rollout import RolloutWriter
from nextcoder.sessions.store import OperationStore
from nextcoder.settings import Settings
from nextcoder.tools.base import ToolContext


@dataclass
class Session:
    session_id: str
    task_dir: Path
    root: Path
    baseline_dir: Path
    operations_db: Path
    checkpoints_db: Path
    # 用第一条用户请求生成的可读标题；旧会话没有时保持为空。
    title: str = ""
    created_at: str = ""
    # 同一会话内复用；进程恢复时由 load_session 重新加载。
    memory_snapshot: MemorySnapshot | None = None
    # 会话开始时冻结的稳定前缀内容；中途变化下一会话才生效。
    agents_text: str = ""
    skills_catalog: SkillCatalog | None = None
    # allow_session 批准的动作/目标；只存在进程内，不写入磁盘。
    permission_approvals: set[tuple[str, str]] = field(default_factory=set)
    # 会话级子 agent 管理器；后台子任务可以跨轮继续。
    delegation_manager: DelegationManager | None = None


@dataclass
class TurnOutcome:
    run_dir: Path
    final_state: dict
    stop_reason: str
    store: OperationStore
    patch: str
    session_id: str
    turn_id: str

    @property
    def has_changes(self) -> bool:
        return bool(self.patch.strip())


def sessions_root(settings: Settings, task_dir: Path) -> Path:
    return task_dir / settings.runs_root / "sessions"


_TITLE_LIMIT = 40
_TITLE_MARKDOWN = re.compile(r"^[\s#>*_`\-•]+")


def make_session_title(request: str, limit: int = _TITLE_LIMIT) -> str:
    """把第一条用户请求压成一行可读标题，不做模型调用。

    去掉开头常见的 Markdown 标记，优先在第一句结束处截断；过长时保留
    前 40 个字符并加省略号。标题只用于会话列表，不进入模型请求。
    """
    # 第一条非空行通常就是用户要做什么；多行细节不塞进标题。
    first_line = ""
    for line in str(request or "").splitlines():
        if line.strip():
            first_line = line
            break
    text = " ".join(first_line.split())
    text = _TITLE_MARKDOWN.sub("", text).strip()
    if not text:
        return "新会话"
    for mark in ("。", "！", "？", "；", ". ", "! ", "? "):
        index = text.find(mark)
        if 8 <= index <= limit:
            text = text[:index]
            break
    if len(text) > limit:
        text = text[:limit].rstrip("，,。.;；:：-_ ") + "…"
    return text or "新会话"


def _write_session_metadata(session: Session, **updates) -> None:
    """合并写回 session.json；失败不影响本轮运行。"""
    path = session.root / "session.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.update({key: value for key, value in updates.items() if value is not None})
    try:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def create_session(settings: Settings, task_dir: Path) -> Session:
    task_dir = task_dir.resolve()
    session_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    root = sessions_root(settings, task_dir) / session_id
    root.mkdir(parents=True, exist_ok=False)
    baseline = root / "baseline"
    copy_baseline(task_dir, baseline)
    metadata = {
        "session_id": session_id,
        "task_dir": str(task_dir),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (root / "session.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    RolloutWriter(root / "rollout.jsonl", session_id=session_id).append(
        "session_start", {"task_dir": str(task_dir), "created_at": metadata["created_at"]})
    return Session(
        session_id=session_id,
        task_dir=task_dir,
        root=root,
        baseline_dir=baseline,
        operations_db=root / "operations.sqlite",
        checkpoints_db=root / "checkpoints.sqlite",
        created_at=metadata["created_at"],
        memory_snapshot=_load_memory_snapshot(settings, task_dir),
        agents_text=load_agents_file(task_dir, settings),
        skills_catalog=discover_skills(task_dir, settings),
    )


def load_session(settings: Settings, task_dir: Path, session_id: str) -> Session:
    if not session_id or "/" in session_id or session_id in {".", ".."}:
        raise ValueError(f"非法 session id：{session_id}")
    task_dir = task_dir.resolve()
    root = sessions_root(settings, task_dir) / session_id
    metadata_path = root / "session.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"会话不存在：{session_id}")
    baseline = root / "baseline"
    if not baseline.is_dir():
        copy_baseline(task_dir, baseline)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    return Session(
        session_id=session_id,
        task_dir=task_dir,
        root=root,
        baseline_dir=baseline,
        operations_db=root / "operations.sqlite",
        checkpoints_db=root / "checkpoints.sqlite",
        title=str(metadata.get("title") or ""),
        created_at=str(metadata.get("created_at") or ""),
        memory_snapshot=_load_memory_snapshot(settings, task_dir),
        agents_text=load_agents_file(task_dir, settings),
        skills_catalog=discover_skills(task_dir, settings),
    )


def _load_memory_snapshot(settings: Settings, task_dir: Path) -> MemorySnapshot | None:
    if not settings.memory_enabled:
        return None
    memory = ProjectMemory(settings, task_dir)
    try:
        return memory.snapshot()
    finally:
        memory.close()


def _plan_summary(context: ToolContext) -> dict:
    """报告里的任务清单状态；未启用或未建清单时 enabled/revision 如实记录。"""
    plan = context.plan
    if plan is None:
        return {"enabled": False}
    current = plan.load()
    return {
        "enabled": True,
        "revision": current.revision,
        "total": len(current.tasks),
        "open": len(current.open_items()),
    }


def _cache_summary(usage: dict, call_usages: list[dict] | None = None) -> dict:
    """缓存命中摘要；区分首次冷启动与后续暖命中。

    第一次请求没有可复用的前缀，命中率天然是 0%。整轮总命中率会被这一次
    拉低，因此额外给出排除首次调用后的暖命中率；TUI 用这个值避免把正常的
    冷启动显示成“缓存失效”。usage 本身保留原样。
    """
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "cached_tokens": 0, "hit_rate": 0.0}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    cached_tokens = int(usage.get("cached_tokens", 0) or 0)
    hit_rate = cached_tokens / input_tokens if input_tokens > 0 else 0.0
    calls = [
        item for item in (call_usages or [])
        if isinstance(item, dict) and int(item.get("input_tokens", 0) or 0) > 0
    ]
    result = {
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "hit_rate": round(hit_rate, 4),
        "calls": len(calls),
    }
    if not calls:
        return result
    first = calls[0]
    first_input = int(first.get("input_tokens", 0) or 0)
    first_cached = int(first.get("cached_tokens", 0) or 0)
    warm = calls[1:]
    warm_input = sum(int(item.get("input_tokens", 0) or 0) for item in warm)
    warm_cached = sum(int(item.get("cached_tokens", 0) or 0) for item in warm)
    result.update({
        "first_call_input_tokens": first_input,
        "first_call_cached_tokens": first_cached,
        "first_call_hit_rate": round(first_cached / first_input, 4)
        if first_input else 0.0,
        "warm_input_tokens": warm_input,
        "warm_cached_tokens": warm_cached,
        "warm_hit_rate": round(warm_cached / warm_input, 4)
        if warm_input else None,
        "cold_start": first_cached == 0,
        "by_call": [
            {
                "input_tokens": int(item.get("input_tokens", 0) or 0),
                "cached_tokens": int(item.get("cached_tokens", 0) or 0),
            }
            for item in calls
        ],
    })
    return result


def list_sessions(settings: Settings, task_dir: Path) -> list[dict]:
    root = sessions_root(settings, task_dir.resolve())
    if not root.is_dir():
        return []
    sessions = []
    for entry in sorted(root.iterdir(), reverse=True):
        metadata_path = entry / "session.json"
        if not metadata_path.is_file():
            continue
        try:
            sessions.append(json.loads(metadata_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return sessions


def run_turn(settings: Settings, session: Session, request: str,
             system_prompt: str | None = None,
             approval_callback: Callable[[object], str] | None = None,
             event_handler: Callable[[object], None] | None = None,
             streaming: bool = False,
             cancel_event: threading.Event | None = None) -> TurnOutcome:
    """在会话中执行一轮：模型循环、工具执行、补丁与报告落盘。

    system_prompt 为 None 时用 benchmark 版；chat 传交互版。只在会话
    第一轮生效，后续轮从检查点历史继承。approval_callback 只为交互模式
    提供；None 时软权限请求仍按拒绝处理。
    """
    turn_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    turn_dir = session.root / "turns" / turn_id
    turn_dir.mkdir(parents=True, exist_ok=True)
    store = OperationStore(session.operations_db)
    store.start_run(turn_id, str(session.task_dir), request)
    # 后台记忆整理：会话第一轮与每累计 N 轮用户请求触发，不阻塞本轮。
    if should_organize(store.count_runs(), settings.memory_organizer_turn_interval):
        spawn_organizer(settings, session.task_dir)
    memory = ProjectMemory(settings, session.task_dir) if settings.memory_enabled else None
    memory_since = time.time()
    snapshot = session.memory_snapshot
    if memory is not None and snapshot is None:
        snapshot = memory.snapshot()
        session.memory_snapshot = snapshot
    rollout = RolloutWriter(session.root / "rollout.jsonl",
                            session_id=session.session_id)
    if not session.title:
        # 标题只在第一条用户请求时生成一次，之后保持稳定，避免会话列表跳动。
        session.title = make_session_title(request)
        _write_session_metadata(session, title=session.title)
    rollout.append("user_request", {"request": request[:2000]}, turn_id=turn_id)
    emit(event_handler, TurnStarted(session.session_id, turn_id, request))
    context = ToolContext(
        task_dir=session.task_dir,
        run_dir=turn_dir,
        baseline_dir=session.baseline_dir,
        store=store,
        settings=settings,
        checkpoints_db=session.checkpoints_db,
        file_states_file=session.root / "file_states.json",
        turn_id=turn_id,
        memory=memory,
        memory_snapshot=snapshot,
        agents_text=session.agents_text,
        skills_catalog=session.skills_catalog,
        approval_callback=approval_callback,
        permission_approvals=session.permission_approvals,
        event_handler=event_handler,
        rollout=rollout,
    )
    if cancel_event is not None:
        context.cancel_event = cancel_event
    runner: AgentRunner | None = None
    delegation: DelegationManager | None = None
    if settings.delegation_enabled:
        delegation = session.delegation_manager
        if delegation is None:
            delegation = DelegationManager(settings, context, session.session_id)
            session.delegation_manager = delegation
        else:
            delegation.update_parent_context(context)
        context.delegation_manager = delegation
        context.allow_delegation = True
    previous_handler = _install_cancel(context)
    try:
        runner = AgentRunner(settings, context, session_id=session.session_id,
                             system_prompt=system_prompt, streaming=streaming)
        if delegation is not None:
            # 子 agent 复用父会话已冻结的稳定前缀，再追加角色说明。
            # 测试替身可能没有该属性；真实 runner 一定有。
            delegation.parent_stable_prefix = getattr(
                runner, "_stable_system_prompt", "")
        final_state = runner.run(request, session.session_id)
        stop_reason = runner.stop_reason or final_state.get("stop_reason") or "model_finished"
    except KeyboardInterrupt:
        final_state = _failure_state(runner, "用户取消")
        stop_reason = "用户取消"
    except Exception as exc:  # noqa: BLE001 - 失败也必须落盘并导出
        detail = traceback.format_exc()[-2000:]
        stop_reason = f"error: {exc}\n{detail}"
        final_state = _failure_state(runner, f"error: {exc}")
    finally:
        if runner is not None and hasattr(runner, "close_mcp"):
            runner.close_mcp()
        _restore_cancel(previous_handler)

    # 稳定前缀在会话开始时冻结：摘要后的记忆整理仍然落盘，但本会话不再
    # 刷新系统消息里的索引；下一会话才会看到新快照。
    patch = git_diff_patch(session.task_dir, session.baseline_dir)
    (turn_dir / "final.patch").write_text(patch, encoding="utf-8")
    usage = final_state.get("usage", {})
    usage_delta = runner.usage_delta if runner is not None else {}
    memory_events: list[dict] = []
    if memory is not None:
        memory_events = memory.events_since(memory_since)
    snapshot = context.memory_snapshot
    report = {
        "session_id": session.session_id,
        "turn_id": turn_id,
        "request": request,
        "stop_reason": stop_reason,
        "model_calls": final_state.get("model_calls", 0),
        "request_tokens": final_state.get("request_tokens", 0),
        "usage": usage,
        "usage_delta": usage_delta,
        "cache": _cache_summary(
            usage, getattr(runner, "call_usages", None) if runner is not None else None),
        "has_changes": bool(patch.strip()),
        "summary": {
            "calls": getattr(runner, "summary_calls", 0) if runner is not None else 0,
            "usage": getattr(runner, "summary_usage", {}) if runner is not None else {},
            "latency_seconds": round(
                getattr(runner, "summary_latency_seconds", 0.0)
                if runner is not None else 0.0, 2),
            "errors": list(getattr(runner, "summary_errors", []))
            if runner is not None else [],
        },
        "recovery": [asdict(note) for note in
                     (getattr(runner, "recovery_notes", []) if runner is not None else [])],
        "compactions": [asdict(event) for event in
                        (getattr(runner, "compaction_events", [])
                         if runner is not None else [])],
        "project_memory": {
            "used": list(getattr(runner, "memory_used", [])) if runner is not None else [],
            "snapshot_chars": len(snapshot.text) if snapshot is not None else 0,
            "events": memory_events,
        },
        "plan": _plan_summary(context),
        "delegations": delegation.summary() if delegation is not None else [],
        "mcp": {
            "tools": getattr(runner, "mcp_tool_names", []) if runner is not None else [],
            "errors": list(getattr(runner, "mcp_errors", [])) if runner is not None else [],
        },
        "hooks": {
            "errors": list(getattr(runner, "hook_errors", []))
            if runner is not None else [],
        },
        "permission_denials": sum(
            1 for op in store.all_operations(turn_id=turn_id)
            if op.error and "[权限拒绝]" in (op.error or "")),
        "llm": {
            "config": endpoint_snapshot(settings),
            "last_endpoint": getattr(runner, "last_endpoint", "")
            if runner is not None else "",
        },
        "provenance": runtime_provenance(session.task_dir, settings),
        "operations": [
            {
                "call_id": op.call_id,
                "tool": op.tool,
                "status": op.status,
                "exit_code": op.exit_code,
                "error": op.error,
                "log_path": op.log_path,
                "related_files": op.related_files,
            }
            for op in store.all_operations(turn_id=turn_id)
        ],
    }
    (turn_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    store.finish_run(
        turn_id,
        "completed" if stop_reason == "model_finished" else "stopped",
        stop_reason,
        report["model_calls"],
        usage,
    )
    rollout.append("turn_end", {
        "stop_reason": stop_reason,
        "model_calls": report["model_calls"],
        "has_changes": bool(patch.strip()),
        "report": f"turns/{turn_id}/report.json",
    }, turn_id=turn_id)
    rollout.close()
    store.close()
    if memory is not None:
        memory.close()
    emit(event_handler, TurnFinished(
        turn_id=turn_id, stop_reason=stop_reason,
        model_calls=report["model_calls"], usage=usage,
        cache=report["cache"]))
    return TurnOutcome(
        run_dir=turn_dir,
        final_state=final_state,
        stop_reason=stop_reason,
        store=store,
        patch=patch,
        session_id=session.session_id,
        turn_id=turn_id,
    )


def compact_session(settings: Settings, session: Session) -> dict:
    """主动压缩已有会话；与自动压缩共用摘要、保存和记忆整理流程。"""
    compact_dir = (session.root / "compactions"
                   / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}")
    compact_dir.mkdir(parents=True, exist_ok=True)
    store = OperationStore(session.operations_db)
    memory = ProjectMemory(settings, session.task_dir) if settings.memory_enabled else None
    memory_since = time.time()
    context = ToolContext(
        task_dir=session.task_dir,
        run_dir=compact_dir,
        baseline_dir=session.baseline_dir,
        store=store,
        settings=settings,
        checkpoints_db=session.checkpoints_db,
        file_states_file=session.root / "file_states.json",
        turn_id=compact_dir.name,
        memory=memory,
        memory_snapshot=session.memory_snapshot,
        agents_text=session.agents_text,
        skills_catalog=session.skills_catalog,
    )
    try:
        runner = AgentRunner(settings, context, session_id=session.session_id)
        result = runner.compact(session.session_id)
    finally:
        if hasattr(runner, "close_mcp"):
            runner.close_mcp()
        store.close()
    memory_events = memory.events_since(memory_since) if memory is not None else []
    if memory is not None:
        memory.close()
    RolloutWriter(session.root / "rollout.jsonl",
                  session_id=session.session_id).append("compact", {
        "summary_id": result.get("summary_id"),
        "covered_count": result.get("covered_count"),
        "before_tokens": result.get("before_tokens"),
        "after_tokens": result.get("after_tokens"),
    }, turn_id=compact_dir.name)
    result["session_id"] = session.session_id
    result["compaction_dir"] = str(compact_dir)
    result["summary_stats"] = {
        "calls": getattr(runner, "summary_calls", 0),
        "usage": getattr(runner, "summary_usage", {}),
        "latency_seconds": round(getattr(runner, "summary_latency_seconds", 0.0), 2),
        "errors": list(getattr(runner, "summary_errors", [])),
    }
    result["project_memory"] = {
        "used": list(getattr(runner, "memory_used", [])),
        "snapshot_chars": len(context.memory_snapshot.text)
        if context.memory_snapshot is not None else 0,
        "events": memory_events,
    }
    result["provenance"] = runtime_provenance(session.task_dir, settings)
    (compact_dir / "compaction.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _failure_state(runner: AgentRunner | None, stop_reason: str) -> dict:
    """异常/取消时用检查点里的已完成调用和用量，而不是清零。"""
    state = getattr(runner, "last_state", None) or {}
    request_tokens = (getattr(runner, "last_request_tokens", 0)
                      or state.get("request_tokens", 0))
    return {
        "messages": state.get("messages") or [],
        "model_calls": state.get("model_calls") or 0,
        "usage": state.get("usage") or {},
        "request_tokens": request_tokens,
        "stop_reason": stop_reason,
    }


def _install_cancel(context: ToolContext):
    """第一次 Ctrl-C 设置取消标记（命令工具据此终止进程组），第二次强制中断。"""
    if threading.current_thread() is not threading.main_thread():
        return None
    previous = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        if context.cancel_event.is_set():
            raise KeyboardInterrupt
        context.cancel_event.set()
        print("\n[已请求停止：正在终止当前命令，完成后会导出改动]", file=sys.stderr, flush=True)

    signal.signal(signal.SIGINT, handler)
    return previous


def _restore_cancel(previous) -> None:
    if previous is not None:
        signal.signal(signal.SIGINT, previous)
