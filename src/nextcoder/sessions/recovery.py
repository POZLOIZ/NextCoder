"""崩溃后的操作核对：确认已经发生的副作用，但不自动重放。

进程可能在“副作用已经发生、完成记录尚未写入”之间退出。恢复时只根据当前
文件内容判断已知编辑操作是否已经生效；无法核对的通用命令保持 unknown，
交给调用方核对日志或文件。所有迁移都是单向的，重复恢复不会再次修改文件。
"""

from dataclasses import dataclass
from pathlib import Path

from nextcoder.sessions.store import Operation, OperationStore
from nextcoder.tools.base import hash_content

READ_ONLY_TOOLS = {"list_files", "search", "read_lines", "read_log", "view_diff"}
WRITE_TOOLS = {"write_file", "apply_patch"}


@dataclass
class RecoveryNote:
    call_id: str
    tool: str
    previous_status: str
    status: str
    message: str
    log_path: str | None = None


def recover_started_operations(store: OperationStore, task_dir: Path) -> list[RecoveryNote]:
    """把 prepared/started 操作收敛到可解释的终态，返回恢复说明。"""
    notes: list[RecoveryNote] = []
    for operation in store.all_operations():
        if operation.status == "prepared":
            message = (f"[恢复] 操作 {operation.call_id}（{operation.tool}）尚未开始，"
                       "没有执行副作用；可重新发起。")
            store.transition(operation.call_id, "cancelled", error=message,
                             result_text=message)
            notes.append(RecoveryNote(operation.call_id, operation.tool, "prepared",
                                      "cancelled", message))
            continue
        if operation.status != "started":
            continue
        note = _recover_one(store, task_dir, operation)
        notes.append(note)
    return notes


def _recover_one(store: OperationStore, task_dir: Path,
                 operation: Operation) -> RecoveryNote:
    tool = operation.tool
    if tool in WRITE_TOOLS:
        return _recover_edit(store, task_dir, operation)
    if tool == "run_command":
        log_path = _find_command_log(task_dir, operation.call_id)
        command = str(operation.args.get("command", ""))
        message = (
            f"[恢复] 操作 {operation.call_id}（run_command）在中断前可能已经执行，"
            "但没有可靠的完成记录；为避免重复副作用，不自动重放。"
            f"命令：{command!r}。请核对日志或目标文件后决定下一步。"
        )
        store.transition(operation.call_id, "unknown", error=message,
                         log_path=log_path, result_text=message)
        return RecoveryNote(operation.call_id, tool, "started", "unknown", message,
                            log_path)
    if tool in READ_ONLY_TOOLS:
        message = (f"[恢复] 操作 {operation.call_id}（{tool}）在中断前未完成；"
                   "只读操作没有副作用，可重新执行。")
        store.transition(operation.call_id, "failed", error=message, result_text=message)
        return RecoveryNote(operation.call_id, tool, "started", "failed", message)
    message = (f"[恢复] 操作 {operation.call_id}（{tool}）状态未知；"
               "没有可靠结果，不自动重放。")
    store.transition(operation.call_id, "unknown", error=message, result_text=message)
    return RecoveryNote(operation.call_id, tool, "started", "unknown", message)


def _recover_edit(store: OperationStore, task_dir: Path,
                  operation: Operation) -> RecoveryNote:
    relative = str(operation.args.get("path", ""))
    target = (task_dir / relative).resolve()
    if not target.is_relative_to(task_dir.resolve()):
        message = f"[恢复] 操作 {operation.call_id} 的路径越出任务目录，未核对。"
        store.transition(operation.call_id, "unknown", error=message, result_text=message)
        return RecoveryNote(operation.call_id, operation.tool, "started", "unknown", message)

    current = target.read_text(encoding="utf-8") if target.is_file() else None
    expected_hash = operation.args.get("expected_hash")
    current_hash = hash_content(current) if current is not None else None

    if operation.tool == "write_file":
        intended = operation.args.get("content", "")
        if current == intended:
            message = _completed_message(operation, relative, expected_hash, current)
            store.transition(operation.call_id, "completed",
                             related_files=[relative], error=None,
                             result_text=message)
            return RecoveryNote(operation.call_id, operation.tool, "started",
                                "completed", message)
        if expected_hash is not None and current_hash == expected_hash:
            message = (f"[恢复] 操作 {operation.call_id}（write_file）中断前未确认生效；"
                       f"{relative} 仍是操作前内容，未自动重写。")
            store.transition(operation.call_id, "unknown", error=message,
                             result_text=message)
            return RecoveryNote(operation.call_id, operation.tool, "started",
                                "unknown", message)
        message = (f"[恢复] 操作 {operation.call_id}（write_file）与预期不符："
                   f"{relative} 已被外部修改（当前 content_hash={current_hash}）；"
                   "为避免覆盖，未写入。")
        store.transition(operation.call_id, "unknown", error=message,
                         result_text=message)
        return RecoveryNote(operation.call_id, operation.tool, "started", "unknown", message)

    if operation.tool == "apply_patch":
        old_text = str(operation.args.get("old_text", ""))
        new_text = str(operation.args.get("new_text", ""))
        replace_all = bool(operation.args.get("replace_all", False))
        applied = False
        pre_hash = expected_hash
        if current is not None and new_text and new_text in current:
            reverted = (current.replace(new_text, old_text) if replace_all
                        else current.replace(new_text, old_text, 1))
            pre_hash = hash_content(reverted)
            applied = expected_hash is not None and pre_hash == expected_hash
        if applied:
            message = _completed_message(operation, relative, pre_hash, current)
            store.transition(operation.call_id, "completed",
                             related_files=[relative], error=None,
                             result_text=message)
            return RecoveryNote(operation.call_id, operation.tool, "started",
                                "completed", message)
        if expected_hash is not None and current_hash == expected_hash:
            message = (f"[恢复] 操作 {operation.call_id}（apply_patch）中断前未确认生效；"
                       f"{relative} 仍是操作前内容，未自动重试。")
            store.transition(operation.call_id, "unknown", error=message,
                             result_text=message)
            return RecoveryNote(operation.call_id, operation.tool, "started",
                                "unknown", message)
        message = (f"[恢复] 操作 {operation.call_id}（apply_patch）与预期不符："
                   f"{relative} 的内容无法还原为操作前状态（当前 content_hash={current_hash}）；"
                   "为避免覆盖外部修改，未写入。")
        store.transition(operation.call_id, "unknown", error=message,
                         result_text=message)
        return RecoveryNote(operation.call_id, operation.tool, "started", "unknown", message)

    message = (f"[恢复] 操作 {operation.call_id}（{operation.tool}）状态未知；"
               "不自动重放。")
    store.transition(operation.call_id, "unknown", error=message, result_text=message)
    return RecoveryNote(operation.call_id, operation.tool, "started", "unknown", message)


def _completed_message(operation: Operation, relative: str, pre_hash: str | None,
                       current: str) -> str:
    return (f"[恢复] 操作 {operation.call_id}（{operation.tool}）核对文件后确认已生效："
            f"{relative}（pre_hash={pre_hash}，content_hash={hash_content(current)}）。"
            "未重复写入。")


def _find_command_log(task_dir: Path, call_id: str) -> str | None:
    """在任务目录的运行产物中找该命令的日志；找不到也不影响 unknown 判定。"""
    root = task_dir / ".nextcoder"
    if not root.is_dir():
        return None
    for path in root.rglob(f"{call_id}-command.log"):
        return str(path)
    return None
