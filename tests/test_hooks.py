"""hooks 总线：声明式配置、执行顺序、block/note/reminder 与内置反循环提醒。"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nextcoder.cli import app
from nextcoder.runtime.events import HookTriggered
from nextcoder.runtime.hooks import (
    BaseHook,
    HookConfigError,
    HookContext,
    HookOutcome,
    build_hook_bus,
    load_config_hooks,
)
from nextcoder.sessions.store import OperationStore
from nextcoder.settings import Settings
from nextcoder.tools.base import ToolContext, ToolResult
from nextcoder.tools.registry import ToolBox


def _settings(**kwargs) -> Settings:
    return Settings(llm_base_url="", llm_model="", **kwargs)


def _write_hooks(path: Path, entries: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"hooks": entries}, ensure_ascii=False),
                    encoding="utf-8")
    return path


class _NoteHook(BaseHook):
    def __init__(self, name: str, text: str, priority: int = 0) -> None:
        self.name = name
        self.text = text
        self.priority = priority

    def run(self, context: HookContext) -> HookOutcome:
        return HookOutcome(notes=[f"{self.name}:{self.text}"])


class _RewriteHook(BaseHook):
    name = "test.rewrite"
    priority = 10

    def run(self, context: HookContext) -> HookOutcome:
        return HookOutcome(args={**context.args, "command": "echo rewritten"})


class TestConfigHooks:
    def test_load_valid_hooks(self, tmp_path):
        path = _write_hooks(tmp_path / "hooks.json", [
            {"name": "no-push", "event": "before_tool_use", "tool": "run_command",
             "pattern": r"git\s+push", "action": "block", "reason": "不要推送"},
            {"name": "pytest-note", "event": "after_tool_use",
             "tool": ["run_command"], "action": "note", "text": "记录测试结果"},
        ])
        hooks = load_config_hooks(path)
        assert [hook.name for hook in hooks] == ["no-push", "pytest-note"]

    def test_missing_file_is_empty(self, tmp_path):
        assert load_config_hooks(tmp_path / "missing.json") == []

    def test_missing_hooks_key_is_rejected(self, tmp_path):
        path = tmp_path / "hooks.json"
        path.write_text('{"hook": []}', encoding="utf-8")
        with pytest.raises(HookConfigError, match="hooks"):
            load_config_hooks(path)

    def test_invalid_event_rejected(self, tmp_path):
        path = _write_hooks(tmp_path / "hooks.json", [
            {"name": "bad", "event": "session_start", "action": "note", "text": "x"}])
        with pytest.raises(HookConfigError, match="event"):
            load_config_hooks(path)

    def test_block_only_allowed_before(self, tmp_path):
        path = _write_hooks(tmp_path / "hooks.json", [
            {"name": "bad", "event": "after_tool_use", "action": "block",
             "reason": "x"}])
        with pytest.raises(HookConfigError, match="after_tool_use"):
            load_config_hooks(path)

    def test_invalid_regex_rejected(self, tmp_path):
        path = _write_hooks(tmp_path / "hooks.json", [
            {"name": "bad", "event": "before_tool_use", "action": "note",
             "text": "x", "pattern": "("}])
        with pytest.raises(HookConfigError, match="正则"):
            load_config_hooks(path)

    def test_reminder_and_note_actions(self, tmp_path):
        _write_hooks(tmp_path / ".nextcoder" / "hooks.json", [
            {"name": "remind", "event": "before_tool_use", "tool": "run_command",
             "pattern": "pytest", "action": "reminder", "text": "先跑目标测试"},
        ])
        bus = build_hook_bus(_settings(), tmp_path)
        bus.before_tool_use("run_command", "c1", {"command": "pytest -q"}, tmp_path)
        assert bus.collect_reminders() == ["先跑目标测试"]

    def test_broken_config_is_reported_not_fatal(self, tmp_path):
        path = tmp_path / ".nextcoder" / "hooks.json"
        path.parent.mkdir(parents=True)
        path.write_text("{not json", encoding="utf-8")
        bus = build_hook_bus(_settings(), tmp_path)
        assert bus.errors and "hooks" in bus.errors[0]

    def test_disabled_hooks_skip_config_but_keep_builtins(self, tmp_path):
        _write_hooks(tmp_path / ".nextcoder" / "hooks.json", [
            {"name": "no-push", "event": "before_tool_use", "tool": "run_command",
             "pattern": "git push", "action": "block", "reason": "不要推送"},
        ])
        bus = build_hook_bus(_settings(hooks_enabled=False, repeat_warn_limit=1),
                             tmp_path)
        outcome = bus.before_tool_use(
            "run_command", "c1", {"command": "git push"}, tmp_path)
        assert not outcome.blocked
        # 内置重复提醒不受 hooks_enabled 影响。
        assert any("重复探索提醒" in item for item in bus.collect_reminders())

    def test_cli_lists_hooks(self, tmp_path):
        path = _write_hooks(tmp_path / "hooks.json", [
            {"name": "no-push", "event": "before_tool_use", "tool": "run_command",
             "pattern": "git push", "action": "block", "reason": "不要推送"},
        ])
        result = CliRunner().invoke(
            app, ["hooks", "--task-dir", str(tmp_path), "--config", str(path)])
        assert result.exit_code == 0, result.output
        assert "no-push" in result.output
        assert "before_tool_use" in result.output
        assert "重复调用提醒" in result.output

    def test_cli_reports_invalid_hooks(self, tmp_path):
        path = tmp_path / "hooks.json"
        path.write_text("{broken", encoding="utf-8")
        result = CliRunner().invoke(
            app, ["hooks", "--task-dir", str(tmp_path), "--config", str(path)])
        assert result.exit_code == 2
        assert "hooks" in result.output


class TestHookOrderingAndDecisions:
    def test_priority_order_and_notes(self, tmp_path):
        from nextcoder.runtime.hooks import HookBus

        bus = HookBus([_NoteHook("late", "b", 20), _NoteHook("early", "a", 10)])
        outcome = bus.before_tool_use("list_files", "c1", {"path": "."}, tmp_path)
        assert outcome.notes == ["early:a", "late:b"]

    def test_block_stops_later_before_hooks(self, tmp_path):
        from nextcoder.runtime.hooks import HookBus

        class BlockHook(BaseHook):
            name = "block"
            priority = 1

            def run(self, context):
                return HookOutcome(blocked=True, reason="stop")

        calls = []

        class AfterBlockHook(BaseHook):
            name = "after-block"
            priority = 2

            def run(self, context):
                calls.append(True)
                return HookOutcome(notes=["should not run"])

        bus = HookBus([BlockHook(), AfterBlockHook()])
        outcome = bus.before_tool_use("run_command", "c1", {"command": "x"}, tmp_path)
        assert outcome.blocked and outcome.reason == "stop"
        assert calls == []

    def test_args_override_is_visible_to_next_hook(self, tmp_path):
        from nextcoder.runtime.hooks import HookBus

        seen = []

        class ObserveHook(BaseHook):
            name = "observe"
            priority = 20

            def run(self, context):
                seen.append(context.args["command"])
                return None

        bus = HookBus([_RewriteHook(), ObserveHook()])
        outcome = bus.before_tool_use(
            "run_command", "c1", {"command": "original"}, tmp_path)
        assert seen == ["echo rewritten"]
        assert outcome.args == {"command": "echo rewritten"}

    def test_hook_triggered_event_is_emitted(self, tmp_path):
        from nextcoder.runtime.hooks import HookBus

        events = []
        bus = HookBus([_NoteHook("note-hook", "hello")],
                      event_handler=events.append, turn_id="t1")
        bus.before_tool_use("list_files", "c1", {"path": "."}, tmp_path)
        assert len(events) == 1
        assert isinstance(events[0], HookTriggered)
        assert events[0].hook == "note-hook"
        assert events[0].action == "note"

    def test_hook_exception_is_recorded(self, tmp_path):
        from nextcoder.runtime.hooks import HookBus

        class BrokenHook(BaseHook):
            name = "broken"

            def run(self, context):
                raise RuntimeError("boom")

        bus = HookBus([BrokenHook()])
        outcome = bus.before_tool_use("list_files", "c1", {"path": "."}, tmp_path)
        assert not outcome.blocked
        assert bus.errors and "boom" in bus.errors[0]


class TestBuiltinHooks:
    def test_repeated_calls_remind_without_blocking(self, tmp_path):
        bus = build_hook_bus(_settings(repeat_warn_limit=2), tmp_path)
        bus.before_tool_use("list_files", "c1", {"path": "."}, tmp_path)
        assert bus.collect_reminders() == []
        outcome = bus.before_tool_use("list_files", "c2", {"path": "."}, tmp_path)
        assert not outcome.blocked
        reminders = bus.collect_reminders()
        assert any("重复探索提醒" in item for item in reminders)
        # 队列只取一次，不会在后续请求里重复注入。
        assert bus.collect_reminders() == []

    def test_repeat_warning_restarts_after_different_call(self, tmp_path):
        bus = build_hook_bus(_settings(repeat_warn_limit=2), tmp_path)
        bus.before_tool_use("list_files", "c1", {"path": "."}, tmp_path)
        bus.before_tool_use("list_files", "c2", {"path": "."}, tmp_path)
        assert any("重复探索提醒" in item for item in bus.collect_reminders())
        # 换成不同调用后，连续计数和提醒节流都应重新开始。
        bus.before_tool_use("search", "c3", {"pattern": "a"}, tmp_path)
        assert bus.collect_reminders() == []
        bus.before_tool_use("search", "c4", {"pattern": "a"}, tmp_path)
        assert any("重复探索提醒" in item for item in bus.collect_reminders())

    def test_parallel_call_group_counts_once(self, tmp_path):
        from nextcoder.runtime.progress import calls_signature

        bus = build_hook_bus(_settings(repeat_warn_limit=2), tmp_path)
        calls = [
            {"name": "list_files", "args": {"path": "."}},
            {"name": "search", "args": {"pattern": "a"}},
        ]
        signature = calls_signature(calls)
        for round_index in range(2):
            for index, call in enumerate(calls):
                bus.before_tool_use(
                    call["name"], f"c{round_index}-{index}", call["args"], tmp_path,
                    group_signature=signature, group_index=index,
                    group_size=len(calls))
        reminders = bus.collect_reminders()
        assert len(reminders) == 1
        assert "并行调用组合" in reminders[0]
        assert "2 个调用" in reminders[0]

    def test_seeded_parallel_group_continues_counting(self, tmp_path):
        from langchain_core.messages import AIMessage

        from nextcoder.runtime.progress import calls_signature

        calls = [
            {"name": "list_files", "args": {"path": "."}, "id": "a"},
            {"name": "search", "args": {"pattern": "a"}, "id": "b"},
        ]
        bus = build_hook_bus(_settings(repeat_warn_limit=3), tmp_path)
        bus.seed([AIMessage(content="", tool_calls=calls),
                  AIMessage(content="", tool_calls=calls)])
        signature = calls_signature(calls)
        bus.before_tool_use(
            "list_files", "c1", calls[0]["args"], tmp_path,
            group_signature=signature, group_index=0, group_size=2)
        assert any("重复探索提醒" in item for item in bus.collect_reminders())

    def test_search_counter_resets_after_successful_edit(self, tmp_path):
        bus = build_hook_bus(_settings(search_warn_limit=2), tmp_path)
        bus.before_tool_use("search", "c1", {"pattern": "a"}, tmp_path)
        bus.before_tool_use("run_command", "c2", {"command": "grep -rn b ."}, tmp_path)
        assert any("搜索提醒" in item for item in bus.collect_reminders())
        bus.after_tool_use(
            "apply_patch", "c3", {"path": "a.py"}, tmp_path,
            ToolResult("c3", "apply_patch", "completed", output_excerpt="ok"))
        bus.before_tool_use("search", "c4", {"pattern": "c"}, tmp_path)
        assert bus.collect_reminders() == []
        # 复位后要重新累计到阈值才提醒；若 after 没被调用，这里不会再有提醒。
        bus.before_tool_use("search", "c5", {"pattern": "d"}, tmp_path)
        assert any("搜索提醒" in item for item in bus.collect_reminders())

    def test_failed_edit_does_not_reset_search_counter(self, tmp_path):
        bus = build_hook_bus(_settings(search_warn_limit=1), tmp_path)
        bus.before_tool_use("search", "c1", {"pattern": "a"}, tmp_path)
        bus.collect_reminders()
        bus.after_tool_use(
            "apply_patch", "c2", {"path": "a.py"}, tmp_path,
            ToolResult("c2", "apply_patch", "failed", error="bad hash"))
        bus.before_tool_use("search", "c3", {"pattern": "b"}, tmp_path)
        assert any("搜索提醒" in item for item in bus.collect_reminders())

    def test_empty_command_output_reminder(self, tmp_path):
        bus = build_hook_bus(_settings(empty_output_warn_limit=2), tmp_path)
        empty = ToolResult("c1", "run_command", "completed", output_excerpt="")
        bus.after_tool_use("run_command", "c1", {"command": "true"}, tmp_path, empty)
        assert bus.collect_reminders() == []
        bus.after_tool_use("run_command", "c2", {"command": ":"}, tmp_path, empty)
        assert any("反死循环" in item for item in bus.collect_reminders())

    def test_nonempty_output_resets_empty_streak(self, tmp_path):
        bus = build_hook_bus(_settings(empty_output_warn_limit=2), tmp_path)
        empty = ToolResult("c1", "run_command", "completed", output_excerpt="")
        output = ToolResult("c2", "run_command", "completed", output_excerpt="hi")
        bus.after_tool_use("run_command", "c1", {"command": "true"}, tmp_path, empty)
        bus.after_tool_use("run_command", "c2", {"command": "echo hi"}, tmp_path, output)
        bus.after_tool_use("run_command", "c3", {"command": ":"}, tmp_path, empty)
        assert bus.collect_reminders() == []

    def test_failed_command_does_not_count_as_empty_output(self, tmp_path):
        bus = build_hook_bus(_settings(empty_output_warn_limit=1), tmp_path)
        failed = ToolResult(
            "c1", "run_command", "failed",
            output_excerpt="", error="[权限拒绝] 越界写入")
        bus.after_tool_use(
            "run_command", "c1", {"command": "touch x"}, tmp_path, failed)
        assert bus.collect_reminders() == []

    def test_non_command_output_resets_empty_streak(self, tmp_path):
        bus = build_hook_bus(_settings(empty_output_warn_limit=2), tmp_path)
        empty = ToolResult("c1", "run_command", "completed", output_excerpt="")
        read = ToolResult("c2", "read_lines", "completed", output_excerpt="hello")
        bus.after_tool_use("run_command", "c1", {"command": "true"}, tmp_path, empty)
        bus.after_tool_use("read_lines", "c2", {"path": "a.py"}, tmp_path, read)
        bus.after_tool_use("run_command", "c3", {"command": ":"}, tmp_path, empty)
        assert bus.collect_reminders() == []


class TestToolBoxIntegration:
    def _context(self, tmp_path, **settings_kwargs):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        run_dir = task_dir / ".nextcoder" / "runs" / "test"
        run_dir.mkdir(parents=True)
        store = OperationStore(run_dir / "operations.sqlite")
        context = ToolContext(
            task_dir=task_dir, run_dir=run_dir,
            baseline_dir=tmp_path / "baseline", store=store,
            settings=_settings(permission_mode="bypass", **settings_kwargs))
        context.baseline_dir.mkdir()
        return context, store

    def test_config_block_prevents_side_effect(self, tmp_path):
        context, store = self._context(tmp_path)
        _write_hooks(context.task_dir / ".nextcoder" / "hooks.json", [
            {"name": "no-touch", "event": "before_tool_use",
             "tool": "run_command", "pattern": "touch blocked.txt",
             "action": "block", "reason": "测试阻止"},
        ])
        bus = build_hook_bus(context.settings, context.task_dir,
                             turn_id="t1", event_handler=context.event_handler)
        box = ToolBox(context, hooks=bus)
        try:
            text = box.execute(
                "run_command", {"command": "touch blocked.txt"}, call_id="c1")
            assert "[钩子阻止]" in text
            assert "测试阻止" in text
            assert not (context.task_dir / "blocked.txt").exists()
            assert store.get("c1").status == "failed"
        finally:
            store.close()

    def test_args_override_changes_executed_command(self, tmp_path):
        context, store = self._context(tmp_path)
        from nextcoder.runtime.hooks import HookBus

        bus = HookBus([_RewriteHook()], turn_id="t1")
        box = ToolBox(context, hooks=bus)
        try:
            text = box.execute("run_command",
                               {"command": "touch should-not-exist.txt"},
                               call_id="c1")
            assert "status=completed" in text
            assert "rewritten" in text
            assert not (context.task_dir / "should-not-exist.txt").exists()
            assert store.get("c1").args["command"] == "echo rewritten"
        finally:
            store.close()

    def test_after_note_is_visible_to_model(self, tmp_path):
        context, store = self._context(tmp_path)
        _write_hooks(context.task_dir / ".nextcoder" / "hooks.json", [
            {"name": "note", "event": "after_tool_use", "tool": "run_command",
             "action": "note", "text": "记得记录测试结果"},
        ])
        bus = build_hook_bus(context.settings, context.task_dir, turn_id="t1")
        box = ToolBox(context, hooks=bus)
        try:
            text = box.execute("run_command", {"command": "echo hi"}, call_id="c1")
            assert "记得记录测试结果" in text
        finally:
            store.close()
