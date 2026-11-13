"""恢复与产物回读测试：崩溃后核对已知编辑、未知命令不重放、跨轮读日志。"""

import pytest

from nextcoder.sessions.recovery import recover_started_operations
from nextcoder.sessions.store import OperationStore
from nextcoder.settings import Settings
from nextcoder.tools.base import ToolContext, hash_content
from nextcoder.tools.commands import run_command
from nextcoder.tools.files import read_log
from nextcoder.tools.registry import ToolBox


@pytest.fixture
def context(tmp_path):
    task_dir = tmp_path / "task"
    run_dir = task_dir / ".nextcoder" / "runs" / "test"
    task_dir.mkdir()
    run_dir.mkdir(parents=True)
    store = OperationStore(run_dir / "operations.sqlite")
    ctx = ToolContext(
        task_dir=task_dir, run_dir=run_dir, baseline_dir=tmp_path / "baseline",
        store=store, settings=Settings(llm_base_url="", llm_model=""),
        turn_id="turn-1",
    )
    ctx.baseline_dir.mkdir()
    yield ctx
    store.close()


def _start_edit(context, tool, args, call_id="call-1"):
    context.store.prepare(tool, args, call_id=call_id, turn_id="turn-1")
    context.store.transition(call_id, "started")


class TestEditRecovery:
    def test_write_applied_before_crash_is_confirmed(self, context):
        target = context.task_dir / "note.txt"
        pre = "start\n"
        target.write_text(pre, encoding="utf-8")
        intended = "start\ndone\n"
        _start_edit(context, "write_file", {
            "path": "note.txt", "content": intended,
            "expected_hash": hash_content(pre),
        })
        # 模拟“写入已经生效、完成记录尚未保存”后进程退出。
        target.write_text(intended, encoding="utf-8")

        notes = recover_started_operations(context.store, context.task_dir)
        operation = context.store.get("call-1")
        assert operation.status == "completed"
        assert "已生效" in (operation.result_text or "")
        assert f"pre_hash={hash_content(pre)}" in (operation.result_text or "")
        assert f"content_hash={hash_content(intended)}" in (operation.result_text or "")
        assert target.read_text(encoding="utf-8") == intended
        assert notes[0].status == "completed"

        # 再恢复一次不产生任何修改。
        assert recover_started_operations(context.store, context.task_dir) == []
        assert target.read_text(encoding="utf-8") == intended

    def test_external_change_reports_conflict_without_overwrite(self, context):
        target = context.task_dir / "note.txt"
        target.write_text("external\n", encoding="utf-8")
        _start_edit(context, "write_file", {
            "path": "note.txt", "content": "start\ndone\n",
            "expected_hash": hash_content("start\n"),
        })

        recover_started_operations(context.store, context.task_dir)
        operation = context.store.get("call-1")
        assert operation.status == "unknown"
        assert "外部修改" in (operation.error or "")
        assert target.read_text(encoding="utf-8") == "external\n"

    def test_unchanged_file_is_not_rewritten(self, context):
        target = context.task_dir / "note.txt"
        target.write_text("start\n", encoding="utf-8")
        _start_edit(context, "write_file", {
            "path": "note.txt", "content": "start\ndone\n",
            "expected_hash": hash_content("start\n"),
        })

        recover_started_operations(context.store, context.task_dir)
        operation = context.store.get("call-1")
        assert operation.status == "unknown"
        assert "未自动重写" in (operation.error or "")
        assert target.read_text(encoding="utf-8") == "start\n"

    def test_apply_patch_applied_is_confirmed(self, context):
        target = context.task_dir / "note.txt"
        pre = "start\n"
        post = "start\ndone\n"
        _start_edit(context, "apply_patch", {
            "path": "note.txt", "old_text": pre, "new_text": post,
            "expected_hash": hash_content(pre), "replace_all": False,
        })
        target.write_text(post, encoding="utf-8")

        recover_started_operations(context.store, context.task_dir)
        operation = context.store.get("call-1")
        assert operation.status == "completed"
        assert f"pre_hash={hash_content(pre)}" in (operation.result_text or "")
        assert target.read_text(encoding="utf-8") == post


class TestCommandRecovery:
    def test_unknown_command_is_not_replayed(self, context):
        log = context.task_dir / "events.log"
        command = f"echo operation-X >> {log.name}"
        _start_edit(context, "run_command", {"command": command}, call_id="call-cmd")
        result = run_command(context, "call-cmd", command)
        assert result.status == "completed"
        assert log.read_text(encoding="utf-8").count("operation-X") == 1

        notes = recover_started_operations(context.store, context.task_dir)
        operation = context.store.get("call-cmd")
        assert operation.status == "unknown"
        assert operation.log_path
        assert "不自动重放" in (operation.result_text or "")
        assert notes[0].status == "unknown"

        # 同一 call_id 再次出现时只返回恢复说明，不执行命令。
        box = ToolBox(context)
        text = box.execute("run_command", {"command": command}, call_id="call-cmd")
        assert "不自动重放" in text
        assert log.read_text(encoding="utf-8").count("operation-X") == 1

        # 重复恢复不会迁移终态，也不会产生新动作。
        recover_started_operations(context.store, context.task_dir)
        assert log.read_text(encoding="utf-8").count("operation-X") == 1

    def test_prepared_operation_is_cancelled(self, context):
        context.store.prepare("run_command", {"command": "touch never.txt"},
                              call_id="call-prep", turn_id="turn-1")
        recover_started_operations(context.store, context.task_dir)
        assert context.store.get("call-prep").status == "cancelled"
        assert not (context.task_dir / "never.txt").exists()


class TestCrossTurnLogRead:
    def test_read_log_from_previous_turn_by_name_and_path(self, tmp_path):
        task_dir = tmp_path / "task"
        session_root = task_dir / ".nextcoder" / "sessions" / "s"
        turn_one = session_root / "turns" / "t1"
        turn_two = session_root / "turns" / "t2"
        turn_one.mkdir(parents=True)
        turn_two.mkdir(parents=True)
        store = OperationStore(session_root / "operations.sqlite")
        settings = Settings(llm_base_url="", llm_model="")
        first = ToolContext(
            task_dir=task_dir, run_dir=turn_one, baseline_dir=tmp_path / "baseline",
            store=store, settings=settings, turn_id="t1",
        )
        second = ToolContext(
            task_dir=task_dir, run_dir=turn_two, baseline_dir=tmp_path / "baseline",
            store=store, settings=settings, turn_id="t2",
        )
        original = first.write_log("call-old-command.log", "alpha\nbeta\n")
        store.prepare("run_command", {"command": "echo"}, call_id="call-old", turn_id="t1")
        store.transition("call-old", "started")
        store.transition("call-old", "completed", log_path=str(original))
        try:
            by_name = read_log(second, "c1", "call-old-command.log")
            assert by_name.status == "completed"
            assert "alpha" in by_name.output_excerpt
            by_path = read_log(second, "c2", str(original))
            assert by_path.status == "completed"
            assert "beta" in by_path.output_excerpt
        finally:
            store.close()

    def test_read_log_rejects_path_outside_task(self, context, tmp_path):
        outside = tmp_path / "outside.log"
        outside.write_text("secret\n", encoding="utf-8")
        result = read_log(context, "c1", str(outside))
        assert result.status == "failed"
        assert "不在任务目录内" in (result.error or "")
