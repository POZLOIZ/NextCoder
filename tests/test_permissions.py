"""权限判定测试：底线、模式默认策略、规则文件与 ToolBox 执行点。"""

import json

import pytest

from nextcoder.runtime.permissions import (
    EXECUTE_SHELL,
    MCP_TOOL,
    READ_PATH,
    WRITE_PATH,
    PermissionGate,
    action_for_call,
    load_rules,
)
from nextcoder.sessions.rollout import RolloutWriter, read_events
from nextcoder.sessions.store import OperationStore
from nextcoder.settings import Settings
from nextcoder.tools.base import ToolContext
from nextcoder.tools.registry import ToolBox

MODES = ("manual", "workspace", "auto", "bypass")


def _gate(tmp_path, mode="workspace", rules=None) -> PermissionGate:
    settings = Settings(permission_mode=mode, llm_base_url="http://localhost",
                        llm_model="fake", llm_api_key="test")
    if rules is not None:
        (tmp_path / ".nextcoder").mkdir(exist_ok=True)
        (tmp_path / ".nextcoder" / "permissions.json").write_text(
            json.dumps(rules, ensure_ascii=False), encoding="utf-8")
    return PermissionGate.from_settings(settings, tmp_path)


def _context(tmp_path, **kwargs) -> ToolContext:
    task_dir = tmp_path / "task"
    run_dir = task_dir / ".nextcoder" / "runs" / "test"
    task_dir.mkdir(exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline = run_dir / "baseline"
    baseline.mkdir(exist_ok=True)
    store = OperationStore(run_dir / "operations.sqlite")
    return ToolContext(
        task_dir=task_dir, run_dir=run_dir, baseline_dir=baseline, store=store,
        settings=Settings(llm_base_url="http://localhost", llm_model="fake",
                          llm_api_key="test", **kwargs),
        turn_id="turn-1")


class TestIrreversibleFloor:
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("command", [
        "rm -rf /", "rm -rf /*", "rm -fr /", "sudo apt install x", "su root",
        "mkfs.ext4 /dev/sda1", "dd if=/dev/zero of=/dev/sda",
        "chmod -R 777 /", ":(){ :|:& };:",
    ])
    def test_floor_blocks_dangerous_commands_in_all_modes(self, tmp_path, mode, command):
        gate = _gate(tmp_path, mode=mode)
        assert not gate.check(EXECUTE_SHELL, command).allowed

    @pytest.mark.parametrize("command", [
        "rm -rf build/", "rm -rf src/old", "git checkout -- file.py",
        "pip install pytest", "chmod +x run.sh", "cat /etc/hostname",
    ])
    def test_floor_allows_scoped_commands(self, tmp_path, command):
        gate = _gate(tmp_path, mode="bypass")
        assert gate.check(EXECUTE_SHELL, command).allowed, command


class TestModeDefaults:
    def test_manual_asks_for_write_shell_and_mcp(self, tmp_path):
        gate = _gate(tmp_path, mode="manual")
        assert gate.check(READ_PATH, "src/app.py").allowed
        assert gate.check(WRITE_PATH, "src/app.py", confirmable=True).ask
        assert gate.check(EXECUTE_SHELL, "pytest -q", confirmable=True).ask
        assert gate.check(MCP_TOOL, "fetch:get", confirmable=True).ask

    def test_workspace_allows_edits_but_asks_for_commands(self, tmp_path):
        gate = _gate(tmp_path, mode="workspace")
        assert gate.check(WRITE_PATH, "src/app.py").allowed
        assert gate.check(EXECUTE_SHELL, "pytest -q", confirmable=True).ask
        assert not gate.check(EXECUTE_SHELL, "pytest -q").allowed
        assert gate.check(MCP_TOOL, "fetch:get", confirmable=True).ask

    def test_auto_allows_shell_only_when_sandboxed(self, tmp_path):
        gate = _gate(tmp_path, mode="auto")
        assert gate.check(EXECUTE_SHELL, "pytest -q", sandboxed=True).allowed
        decision = gate.check(EXECUTE_SHELL, "pytest -q", confirmable=True)
        assert not decision.allowed and decision.ask
        assert "沙箱" in decision.reason
        assert gate.check(WRITE_PATH, "src/app.py").allowed
        assert gate.check(MCP_TOOL, "fetch:get", confirmable=True).ask

    def test_workspace_does_not_ask_for_compound_shell_rule(self, tmp_path):
        # 命令统一询问，不再用“组合命令”当作判断条件；allow 规则仍优先。
        gate = _gate(tmp_path, mode="workspace")
        decision = gate.check(EXECUTE_SHELL, "a && b", confirmable=True)
        assert decision.ask
        assert gate.check(EXECUTE_SHELL, "pytest -q | tail -3").allowed is False

    def test_workspace_denies_sensitive_paths(self, tmp_path):
        gate = _gate(tmp_path, mode="workspace")
        for relative in (".env", "config/.env", "id_rsa.pem", "secrets.key",
                         ".git/HEAD"):
            assert not gate.check(READ_PATH, relative).allowed, relative
            assert not gate.check(WRITE_PATH, relative).allowed, relative

    def test_workspace_allows_normal_paths(self, tmp_path):
        gate = _gate(tmp_path, mode="workspace")
        assert gate.check(READ_PATH, "src/app.py").allowed
        assert gate.check(WRITE_PATH, "src/app.py").allowed

    def test_workspace_denies_outside_root_write(self, tmp_path):
        decision = _gate(tmp_path, mode="workspace").check(
            WRITE_PATH, "/etc/passwd")
        assert not decision.allowed
        assert "任务目录" in decision.reason

    def test_auto_mode_runs_command_when_process_is_already_isolated(
            self, tmp_path, monkeypatch):
        from nextcoder.tools import commands

        monkeypatch.setattr(commands, "running_in_container", lambda: True)
        context = _context(tmp_path, permission_mode="auto")
        box = ToolBox(context)
        try:
            text = box.execute("run_command", {"command": "printf auto-ok"},
                               call_id="call-auto")
        finally:
            context.store.close()
        assert "auto-ok" in text
        assert "[权限拒绝]" not in text

    def test_auto_mode_asks_when_no_isolation_is_available(self, tmp_path):
        context = _context(tmp_path, permission_mode="auto")
        box = ToolBox(context)
        try:
            text = box.execute("run_command", {"command": "printf blocked"},
                               call_id="call-auto-deny")
        finally:
            context.store.close()
        assert "[权限拒绝]" in text
        assert "沙箱" in text

    def test_bypass_allows_sensitive_paths_but_keeps_floor(self, tmp_path):
        gate = _gate(tmp_path, mode="bypass")
        assert gate.check(READ_PATH, ".env").allowed
        assert gate.check(EXECUTE_SHELL, "pytest -q | tail -5").allowed
        assert gate.check(MCP_TOOL, "fetch:get").allowed
        assert not gate.check(EXECUTE_SHELL, "sudo rm x").allowed

    def test_legacy_modes_migrate(self, tmp_path):
        assert _gate(tmp_path, mode="standard").mode == "workspace"
        assert _gate(tmp_path, mode="aggressive").mode == "bypass"


class TestRulesFile:
    def test_deny_rule_beats_allow_rule(self, tmp_path):
        gate = _gate(tmp_path, mode="bypass", rules={
            "allow": [{"scope": "path", "pattern": "src/*"}],
            "deny": [{"scope": "path", "pattern": "src/secret.py"}],
        })
        assert gate.check(WRITE_PATH, "src/main.py").allowed
        assert not gate.check(WRITE_PATH, "src/secret.py").allowed

    def test_command_prefix_rule(self, tmp_path):
        gate = _gate(tmp_path, mode="bypass", rules={
            "deny": [{"scope": "command_prefix", "pattern": "git push"}],
        })
        assert not gate.check(EXECUTE_SHELL, "git push origin main").allowed
        assert gate.check(EXECUTE_SHELL, "git status").allowed

    def test_allow_rule_lifts_workspace_shell_ask(self, tmp_path):
        gate = _gate(tmp_path, mode="workspace", rules={
            "allow": [{"scope": "command_prefix", "pattern": "pytest"}],
        })
        assert gate.check(EXECUTE_SHELL, "pytest -q | tail -3").allowed

    def test_mcp_tool_rule_scope(self, tmp_path):
        gate = _gate(tmp_path, mode="workspace", rules={
            "allow": [{"scope": "mcp_tool", "pattern": "fetch:*"}],
        })
        assert gate.check(MCP_TOOL, "fetch:get").allowed
        assert not gate.check(MCP_TOOL, "db:query").allowed

    def test_missing_or_broken_rules_file_is_ignored(self, tmp_path):
        assert load_rules(tmp_path / ".nextcoder" / "permissions.json").allow == ()
        broken = tmp_path / "rules.json"
        broken.write_text("{not json", encoding="utf-8")
        assert load_rules(broken).deny == ()


class TestActionMapping:
    def test_builtin_tools_map_to_actions(self):
        assert action_for_call("list_files", {}) == (READ_PATH, ".")
        assert action_for_call("read_lines", {"path": "a.py"}) == (READ_PATH, "a.py")
        assert action_for_call("write_file", {"path": "a.py"}) == (WRITE_PATH, "a.py")
        assert action_for_call("run_command", {"command": "ls"}) == (EXECUTE_SHELL, "ls")
        assert action_for_call("mcp__fetch__get", {}) == (MCP_TOOL, "fetch:get")

    def test_internal_tools_have_no_action(self):
        for name in ("memory_read", "plan_create", "plan_update", "load_skill"):
            assert action_for_call(name, {}) is None


class TestToolBoxEnforcement:
    def test_denied_command_returns_prefix_and_records_operation(self, tmp_path):
        context = _context(tmp_path, permission_mode="bypass")
        box = ToolBox(context)
        text = box.execute("run_command", {"command": "sudo rm -rf /"},
                           call_id="call-deny")
        operation = context.store.get("call-deny")
        context.store.close()
        assert "[权限拒绝]" in text
        assert "提权" in text
        assert operation is not None and operation.status == "failed"
        assert "[权限拒绝]" in (operation.error or "")

    def test_denied_write_in_workspace_mode(self, tmp_path):
        context = _context(tmp_path, permission_mode="workspace")
        box = ToolBox(context)
        text = box.execute("write_file",
                           {"path": ".env", "content": "K=1\n"},
                           call_id="call-env")
        context.store.close()
        assert "[权限拒绝]" in text
        assert not (context.task_dir / ".env").exists()

    def test_internal_tools_bypass_permission_gate(self, tmp_path):
        context = _context(tmp_path, permission_mode="workspace")
        box = ToolBox(context)
        text = box.execute("plan_create",
                           {"tasks": ["第一步：复现", "第二步：修复"]},
                           call_id="call-plan")
        context.store.close()
        assert "已创建任务清单" in text
        assert "[权限拒绝]" not in text


class TestInteractiveAsk:
    def test_soft_denials_are_askable_only_when_confirmable(self, tmp_path):
        gate = _gate(tmp_path, mode="workspace")
        assert not gate.check(EXECUTE_SHELL, "a && b").ask
        assert gate.check(EXECUTE_SHELL, "a && b", confirmable=True).ask
        assert gate.check(WRITE_PATH, ".env", confirmable=True).ask
        assert gate.check(MCP_TOOL, "fetch:get", confirmable=True).ask

    def test_hard_floor_and_explicit_deny_never_ask(self, tmp_path):
        gate = _gate(tmp_path, mode="workspace", rules={
            "deny": [{"scope": "command_prefix", "pattern": "git push"}]})
        floor = gate.check(EXECUTE_SHELL, "sudo rm -rf /", confirmable=True)
        explicit = gate.check(EXECUTE_SHELL, "git push origin main",
                              confirmable=True)
        outside = gate.check(WRITE_PATH, "/etc/passwd", confirmable=True)
        assert not floor.allowed and not floor.ask
        assert not explicit.allowed and not explicit.ask
        assert not outside.allowed and not outside.ask

    def test_allow_once_executes_without_started_before_permission(self, tmp_path):
        answers = []
        context = _context(tmp_path, permission_mode="workspace")
        context.approval_callback = lambda request: answers.append(request) or "allow_once"
        box = ToolBox(context)
        text = box.execute("run_command",
                           {"command": "printf a && printf b"},
                           call_id="call-once")
        operation = context.store.get("call-once")
        context.store.close()
        assert answers and answers[0].action == EXECUTE_SHELL
        assert "completed" in text
        assert operation is not None and operation.status == "completed"

    def test_deny_marks_failed_without_started(self, tmp_path):
        context = _context(tmp_path, permission_mode="workspace")
        context.approval_callback = lambda request: "deny"
        box = ToolBox(context)
        text = box.execute("run_command",
                           {"command": "printf a && printf b"},
                           call_id="call-deny-ask")
        operation = context.store.get("call-deny-ask")
        context.store.close()
        assert "用户拒绝" in text
        assert operation is not None and operation.status == "failed"

    def test_allow_session_reuses_approval_within_context(self, tmp_path):
        calls = []

        def callback(request):
            calls.append(request)
            return "allow_session"

        context = _context(tmp_path, permission_mode="workspace")
        context.approval_callback = callback
        box = ToolBox(context)
        box.execute("run_command", {"command": "printf a && printf b"},
                    call_id="call-session-1")
        box.execute("run_command", {"command": "printf a && printf b"},
                    call_id="call-session-2")
        context.store.close()
        assert len(calls) == 1
        assert (EXECUTE_SHELL, "printf a && printf b") in context.permission_approvals

    def test_callback_exception_is_denied(self, tmp_path):
        def callback(request):
            raise RuntimeError("ui crashed")

        context = _context(tmp_path, permission_mode="workspace")
        context.approval_callback = callback
        box = ToolBox(context)
        text = box.execute("run_command",
                           {"command": "printf a && printf b"},
                           call_id="call-error")
        operation = context.store.get("call-error")
        context.store.close()
        assert "权限确认失败" in text
        assert operation is not None and operation.status == "failed"

    def test_permission_events_are_written_to_rollout(self, tmp_path):
        context = _context(tmp_path, permission_mode="workspace")
        context.rollout = RolloutWriter(
            context.run_dir / "rollout.jsonl", session_id="s")
        context.approval_callback = lambda request: "allow_once"
        box = ToolBox(context)
        box.execute("run_command", {"command": "printf a && printf b"},
                    call_id="call-events")
        context.store.close()
        context.rollout.close()
        events = read_events(context.run_dir / "rollout.jsonl")
        types = [event["type"] for event in events]
        assert types == ["permission_requested", "permission_answered"]
        assert events[1]["payload"]["answer"] == "allow_once"
