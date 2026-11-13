"""NC-09…NC-11 运行机制验收：直接驱动工具与进程，不依赖模型碰巧生成调用。

这些场景检验崩溃恢复和进程终止的工程行为，不计入编程任务成功率。
"""

import json
import os
import shutil
import signal
import tempfile
import threading
import time
from pathlib import Path

from nextcoder.runtime.provenance import runtime_provenance
from nextcoder.sessions.recovery import recover_started_operations
from nextcoder.sessions.store import OperationStore
from nextcoder.settings import Settings
from nextcoder.tools.base import ToolContext, hash_content
from nextcoder.tools.commands import run_command
from nextcoder.tools.edits import write_file

RUNTIME_SCENARIOS = {"NC-09", "NC-10", "NC-11"}


def run_runtime_scenario(package_dir: Path, task: dict, out_root: Path,
                         settings: Settings | None = None) -> dict:
    settings = settings or Settings(llm_base_url="", llm_model="")
    task_id = task["task_id"]
    started = time.monotonic()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    result = {"task_id": task_id, "status": "failed", "checks": [], "error": None,
              "provenance": runtime_provenance(package_dir, settings)}
    try:
        if task_id == "NC-09":
            checks = _nc09(package_dir)
        elif task_id == "NC-10":
            checks = _nc10(package_dir)
        elif task_id == "NC-11":
            checks = _nc11()
        else:
            raise ValueError(f"未知运行场景：{task_id}")
        result["checks"] = checks
        result["status"] = "passed" if all(c["passed"] for c in checks) else "failed"
    except Exception as exc:  # noqa: BLE001 - 场景失败要记录而不是中断
        result["status"] = "blocked"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["wall_seconds"] = round(time.monotonic() - started, 1)
        _write_result(out_root, task_id, result, stamp)
    return result


def _check(name: str, passed: bool, detail: str = "") -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _settings() -> Settings:
    return Settings(llm_base_url="", llm_model="")


def _context(task_dir: Path, run_dir: Path, store: OperationStore,
             turn_id: str) -> ToolContext:
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline = run_dir / "baseline"
    baseline.mkdir(exist_ok=True)
    return ToolContext(task_dir=task_dir, run_dir=run_dir, baseline_dir=baseline,
                       store=store, settings=_settings(), turn_id=turn_id)


def _write_result(out_root: Path, task_id: str, result: dict, stamp: str) -> None:
    run_dir = out_root / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")


# ---- NC-09：文件已写好、完成记录未保存 ----


def _nc09(package_dir: Path) -> list[dict]:
    checks: list[dict] = []
    work = Path(tempfile.mkdtemp(prefix="nc09-", dir="/tmp"))
    try:
        task_dir = work / "task"
        task_dir.mkdir()
        fixture = package_dir / "fixtures" / "operation-note.txt"
        initial = fixture.read_text(encoding="utf-8")
        target = task_dir / "operation-note.txt"
        target.write_text(initial, encoding="utf-8")
        intended = "start\ndone\n"

        # 先核验正常路径：同一工具调用应完成并留下正确内容。
        normal_root = work / "normal"
        normal_root.mkdir()
        normal_store = OperationStore(normal_root / "operations.sqlite")
        normal_context = _context(task_dir, normal_root / "t1", normal_store, "t1")
        normal_result = write_file(normal_context, "normal-call", target.name, intended,
                                   expected_hash=hash_content(initial))
        normal_store.close()
        checks.append(_check(
            "normal_write_path",
            normal_result.status == "completed" and target.read_text(encoding="utf-8") == intended,
            f"status={normal_result.status}"))

        # 重置到故障起点。
        target.write_text(initial, encoding="utf-8")
        session_root = task_dir / ".nextcoder" / "sessions" / "s"
        session_root.mkdir(parents=True)
        db = session_root / "operations.sqlite"
        store = OperationStore(db)
        context = _context(task_dir, session_root / "turns" / "t1", store, "t1")
        store.prepare("write_file", {
            "path": target.name, "content": intended,
            "expected_hash": hash_content(initial),
        }, call_id="nc09-call", turn_id="t1")
        store.transition("nc09-call", "started")
        # 副作用发生，但故意不写 completed，然后关闭连接模拟进程退出。
        written = write_file(context, "nc09-call", target.name, intended,
                             expected_hash=hash_content(initial))
        assert written.status == "completed"
        store.close()

        # 恢复：同一操作 ID 应通过核对文件收敛为 completed。
        reopened = OperationStore(db)
        _context(task_dir, session_root / "turns" / "t2", reopened, "t2")
        notes = recover_started_operations(reopened, task_dir)
        operation = reopened.get("nc09-call")
        content = target.read_text(encoding="utf-8")
        checks.append(_check(
            "recovered_content_exact", content == intended,
            f"content={content!r}"))
        checks.append(_check(
            "same_call_id_completed",
            operation is not None and operation.status == "completed",
            f"status={operation.status if operation else None}"))
        checks.append(_check(
            "hash_traceable",
            operation is not None and "pre_hash=" in (operation.result_text or "")
            and "content_hash=" in (operation.result_text or ""),
            (operation.result_text if operation else "")[:200]))
        checks.append(_check(
            "recovery_note_previous_started",
            len(notes) == 1 and notes[0].previous_status == "started",
            f"notes={len(notes)}"))

        # 再恢复一次：不产生额外修改，也不再迁移状态。
        before_hash = hash_content(content)
        second = recover_started_operations(reopened, task_dir)
        checks.append(_check(
            "idempotent_recovery",
            second == [] and hash_content(target.read_text(encoding="utf-8")) == before_hash,
            f"notes={len(second)}"))
        reopened.close()

        # 冲突分支：写入后文件被外部改成其他内容，恢复只能报告冲突。
        conflict = task_dir / "conflict.txt"
        conflict.write_text(initial, encoding="utf-8")
        conflict_db = session_root / "conflict.sqlite"
        conflict_store = OperationStore(conflict_db)
        conflict_context = _context(task_dir, session_root / "turns" / "c1",
                                    conflict_store, "c1")
        conflict_store.prepare("write_file", {
            "path": conflict.name, "content": intended,
            "expected_hash": hash_content(initial),
        }, call_id="nc09-conflict", turn_id="c1")
        conflict_store.transition("nc09-conflict", "started")
        write_file(conflict_context, "nc09-conflict", conflict.name, intended,
                   expected_hash=hash_content(initial))
        conflict.write_text("external\n", encoding="utf-8")
        conflict_store.close()

        conflict_reopened = OperationStore(conflict_db)
        recover_started_operations(conflict_reopened, task_dir)
        conflict_operation = conflict_reopened.get("nc09-conflict")
        checks.append(_check(
            "external_change_reports_conflict",
            conflict_operation is not None
            and conflict_operation.status == "unknown"
            and "外部修改" in (conflict_operation.error or "")
            and conflict.read_text(encoding="utf-8") == "external\n",
            f"status={conflict_operation.status if conflict_operation else None}"))
        conflict_reopened.close()
        return checks
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---- NC-10：命令结果未知时不盲目重跑 ----


def _nc10(package_dir: Path) -> list[dict]:
    checks: list[dict] = []
    work = Path(tempfile.mkdtemp(prefix="nc10-", dir="/tmp"))
    try:
        task_dir = work / "task"
        task_dir.mkdir()
        events = task_dir / "events.log"
        events.write_text("", encoding="utf-8")
        command = "echo operation-X >> events.log"
        session_root = task_dir / ".nextcoder" / "sessions" / "s"
        session_root.mkdir(parents=True)
        db = session_root / "operations.sqlite"

        # 正常路径：命令只执行一次。
        normal_store = OperationStore(session_root / "normal.sqlite")
        normal_context = _context(task_dir, session_root / "turns" / "n", normal_store, "n")
        normal = run_command(normal_context, "normal-call", command)
        normal_store.close()
        checks.append(_check(
            "normal_command_path",
            normal.status == "completed" and events.read_text(encoding="utf-8").count(
                "operation-X") == 1,
            f"status={normal.status}"))
        events.write_text("", encoding="utf-8")

        # 故障路径：追加已经发生，但完成记录未保存。
        store = OperationStore(db)
        context = _context(task_dir, session_root / "turns" / "t1", store, "t1")
        store.prepare("run_command", {"command": command},
                      call_id="nc10-call", turn_id="t1")
        store.transition("nc10-call", "started")
        executed = run_command(context, "nc10-call", command)
        assert executed.status == "completed"
        assert events.read_text(encoding="utf-8").count("operation-X") == 1
        store.close()

        reopened = OperationStore(db)
        context2 = _context(task_dir, session_root / "turns" / "t2", reopened, "t2")
        notes = recover_started_operations(reopened, task_dir)
        operation = reopened.get("nc10-call")
        checks.append(_check(
            "unknown_not_success",
            operation is not None and operation.status == "unknown",
            f"status={operation.status if operation else None}"))
        checks.append(_check(
            "command_retained",
            operation is not None and operation.args.get("command") == command
            and len(notes) == 1 and notes[0].previous_status == "started",
            f"args={operation.args if operation else None}"))
        checks.append(_check(
            "log_located",
            operation is not None and bool(operation.log_path)
            and Path(operation.log_path).is_file(),
            f"log_path={operation.log_path if operation else None}"))
        checks.append(_check(
            "recovery_explains_no_replay",
            "不自动重放" in (operation.result_text or ""),
            (operation.result_text or "")[:200]))

        # 同一 call_id 再次出现时不执行；日志仍只有一次。
        from nextcoder.tools.registry import ToolBox
        box = ToolBox(context2)
        text = box.execute("run_command", {"command": command}, call_id="nc10-call")
        checks.append(_check(
            "same_call_id_not_replayed",
            "不自动重放" in text
            and events.read_text(encoding="utf-8").count("operation-X") == 1,
            text[:200]))
        recover_started_operations(reopened, task_dir)
        checks.append(_check(
            "repeat_recovery_no_side_effect",
            events.read_text(encoding="utf-8").count("operation-X") == 1,
            f"count={events.read_text(encoding='utf-8').count('operation-X')}"))
        reopened.close()
        return checks
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---- NC-11：取消/超时后父子进程都停止 ----


_HEARTBEAT_SCRIPT = """\
import os
import time

pid = os.fork()
if pid == 0:
    with open("heartbeat.txt", "a", buffering=1) as beat:
        while True:
            beat.write(f"{os.getpid()} {time.time()}\\n")
            beat.flush()
            time.sleep(0.1)
else:
    with open("pids.txt", "w", buffering=1) as pids:
        pids.write(f"{os.getpid()} {pid}\\n")
        pids.flush()
    os.waitpid(pid, 0)
"""


def _process_alive(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    if not stat.is_file():
        return False
    try:
        fields = stat.read_text(encoding="utf-8").split()
    except OSError:
        return False
    return len(fields) > 2 and fields[2] != "Z"


def _wait_for_heartbeats(task_dir: Path, minimum: int = 3,
                         timeout: float = 8.0) -> tuple[int, int]:
    pids_file = task_dir / "pids.txt"
    beat_file = task_dir / "heartbeat.txt"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pids_file.is_file() and beat_file.is_file():
            lines = beat_file.read_text(encoding="utf-8").splitlines()
            if len(lines) >= minimum:
                parent, child = pids_file.read_text(encoding="utf-8").split()
                return int(parent), int(child)
        time.sleep(0.05)
    raise RuntimeError("心跳进程未在期限内启动")


def _heartbeat_count(task_dir: Path) -> int:
    path = task_dir / "heartbeat.txt"
    return len(path.read_text(encoding="utf-8").splitlines()) if path.is_file() else 0


def _cleanup_group(parent_pid: int | None) -> None:
    if parent_pid is None:
        return
    try:
        os.killpg(parent_pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _nc11() -> list[dict]:
    checks: list[dict] = []
    work = Path(tempfile.mkdtemp(prefix="nc11-", dir="/tmp"))
    try:
        task_dir = work / "task"
        task_dir.mkdir()
        (task_dir / "heartbeat.py").write_text(_HEARTBEAT_SCRIPT, encoding="utf-8")
        session_root = task_dir / ".nextcoder" / "sessions" / "s"
        session_root.mkdir(parents=True)

        # 取消路径。
        cancel_db = session_root / "cancel.sqlite"
        cancel_store = OperationStore(cancel_db)
        cancel_context = _context(task_dir, session_root / "turns" / "cancel",
                                  cancel_store, "cancel")
        result_holder: dict = {}

        def run_cancel() -> None:
            result_holder["result"] = run_command(
                cancel_context, "nc11-cancel", "python heartbeat.py", timeout=30)

        thread = threading.Thread(target=run_cancel)
        thread.start()
        parent = child = None
        try:
            parent, child = _wait_for_heartbeats(task_dir)
            cancel_context.cancel_event.set()
            thread.join(timeout=8)
            result = result_holder.get("result")
            checks.append(_check(
                "cancel_status",
                result is not None and result.status == "cancelled",
                f"status={result.status if result else None}"))

            grace_deadline = time.monotonic() + 3
            while time.monotonic() < grace_deadline and (
                    _process_alive(parent) or _process_alive(child)):
                time.sleep(0.05)
            checks.append(_check(
                "cancel_parent_and_child_exit",
                not _process_alive(parent) and not _process_alive(child),
                f"parent_alive={_process_alive(parent)} child_alive={_process_alive(child)}"))

            count_after_grace = _heartbeat_count(task_dir)
            time.sleep(1)
            checks.append(_check(
                "cancel_no_new_heartbeat",
                _heartbeat_count(task_dir) == count_after_grace,
                f"before={count_after_grace} after={_heartbeat_count(task_dir)}"))
        finally:
            _cleanup_group(parent)
            thread.join(timeout=2)
            cancel_store.close()

        # 超时路径：不能只设置界面状态。
        (task_dir / "heartbeat.txt").unlink(missing_ok=True)
        (task_dir / "pids.txt").unlink(missing_ok=True)
        timeout_db = session_root / "timeout.sqlite"
        timeout_store = OperationStore(timeout_db)
        timeout_context = _context(task_dir, session_root / "turns" / "timeout",
                                   timeout_store, "timeout")
        timeout_result = run_command(timeout_context, "nc11-timeout",
                                     "python heartbeat.py", timeout=2)
        checks.append(_check(
            "timeout_status",
            timeout_result.status == "timeout",
            f"status={timeout_result.status}"))
        timeout_parent = timeout_child = None
        try:
            pids = (task_dir / "pids.txt").read_text(encoding="utf-8").split()
            timeout_parent, timeout_child = int(pids[0]), int(pids[1])
        except (FileNotFoundError, ValueError, IndexError):
            pass
        try:
            grace_deadline = time.monotonic() + 3
            while time.monotonic() < grace_deadline and (
                    timeout_parent is not None and (
                        _process_alive(timeout_parent) or _process_alive(timeout_child))):
                time.sleep(0.05)
            checks.append(_check(
                "timeout_parent_and_child_exit",
                timeout_parent is not None
                and not _process_alive(timeout_parent)
                and not _process_alive(timeout_child),
                f"parent={timeout_parent} child={timeout_child}"))
            count_after_timeout = _heartbeat_count(task_dir)
            time.sleep(1)
            checks.append(_check(
                "timeout_no_new_heartbeat",
                _heartbeat_count(task_dir) == count_after_timeout,
                f"before={count_after_timeout} after={_heartbeat_count(task_dir)}"))
        finally:
            _cleanup_group(timeout_parent)
            timeout_store.close()
        return checks
    finally:
        shutil.rmtree(work, ignore_errors=True)
