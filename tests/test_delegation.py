"""子 agent 测试：角色白名单、前后台、取消/恢复、worktree 隔离与补丁。"""

import json
import subprocess
import threading
import time
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from nextcoder.runtime import delegation as delegation_module
from nextcoder.runtime.delegation import (
    CODER_TOOLS,
    READ_TOOLS,
    ROLES,
    TEST_TOOLS,
    DelegationManager,
)
from nextcoder.runtime.skills import SkillCatalog
from nextcoder.sessions.store import OperationStore
from nextcoder.settings import Settings
from nextcoder.tools.base import ToolContext
from nextcoder.tools.registry import ToolBox


def _settings(**overrides) -> Settings:
    values = dict(
        llm_base_url="http://localhost", llm_model="fake", llm_api_key="test",
        mcp_enabled=False, memory_enabled=False, plan_enabled=False,
        delegation_model_calls_limit=7, delegation_wall_time_seconds=30,
        delegation_background_max=2, delegation_max_resumes=1,
    )
    values.update(overrides)
    return Settings(**values)


def _parent(tmp_path, **settings_kwargs):
    task_dir = tmp_path / "repo"
    task_dir.mkdir()
    (task_dir / "app.py").write_text("base\n", encoding="utf-8")
    run_dir = task_dir / ".nextcoder" / "runs" / "test"
    run_dir.mkdir(parents=True)
    baseline = run_dir / "baseline"
    baseline.mkdir()
    store = OperationStore(run_dir / "operations.sqlite")
    context = ToolContext(
        task_dir=task_dir, run_dir=run_dir, baseline_dir=baseline,
        store=store, settings=_settings(**settings_kwargs), turn_id="turn-1",
        skills_catalog=SkillCatalog())
    return context


def _install_runner(monkeypatch, run_fn):
    class StubRunner:
        instances: list = []

        def __init__(self, settings, context, session_id="", system_prompt=None,
                     stable_system_prompt=None):
            self.settings = settings
            self.context = context
            self.session_id = session_id
            self.stable_system_prompt = stable_system_prompt
            self.calls: list[tuple[str, str]] = []
            self.stop_reason = None
            StubRunner.instances.append(self)

        def run(self, request, thread_id):
            self.calls.append((request, thread_id))
            result = run_fn(self, request, thread_id)
            if isinstance(result, Exception):
                raise result
            return result

        def close_mcp(self):
            pass

    monkeypatch.setattr(delegation_module, "AgentRunner", StubRunner)
    return StubRunner


def _ok(text="完成", calls=2):
    return {
        "messages": [AIMessage(content=text)],
        "model_calls": calls,
        "usage": {"input_tokens": 100, "cached_tokens": 40},
        "stop_reason": None,
    }


def test_role_allowlists_exclude_nesting_and_plan_tools(tmp_path):
    assert "run_command" not in READ_TOOLS
    assert "write_file" not in TEST_TOOLS
    assert "write_file" in CODER_TOOLS
    for spec in ROLES.values():
        assert "delegate" not in spec.tools
        assert "delegation_status" not in spec.tools
        assert "plan_create" not in spec.tools
        assert "plan_update" not in spec.tools


def test_child_gets_parent_prefix_and_role_allowlist(tmp_path, monkeypatch):
    StubRunner = _install_runner(monkeypatch, lambda self, request, thread_id: _ok())
    context = _parent(tmp_path)
    manager = DelegationManager(_settings(), context, "session-1")
    manager.parent_stable_prefix = "父稳定前缀"

    result = manager.delegate("researcher", "定位失败路径")
    manager.close()
    context.store.close()

    assert result["status"] == "completed"
    runner = StubRunner.instances[0]
    assert runner.calls == [("定位失败路径", runner.calls[0][1])]
    assert runner.stable_system_prompt.startswith("父稳定前缀")
    assert "researcher" in runner.stable_system_prompt
    assert runner.context.allowed_tools == set(READ_TOOLS)
    assert runner.context.allow_delegation is False
    assert runner.context.allow_mcp_tools is False
    assert runner.settings.model_calls_limit == 7
    definitions = {
        tool.name for tool in ToolBox(runner.context).definitions()
    }
    assert definitions <= READ_TOOLS
    assert "delegate" not in definitions


def test_foreground_report_contains_summary_and_usage(tmp_path, monkeypatch):
    _install_runner(monkeypatch, lambda self, request, thread_id: _ok("已定位到 src/app.py"))
    context = _parent(tmp_path)
    manager = DelegationManager(_settings(), context, "session-1")

    result = manager.delegate("researcher", "定位问题")
    report = manager.summary()[0]
    manager.close()
    context.store.close()

    assert result["status"] == "completed"
    assert "已定位到 src/app.py" in result["text"]
    assert report["model_calls"] == 2
    assert report["usage"]["cached_tokens"] == 40
    assert Path(report["report"]).is_file()


def test_report_artifact_preserves_full_summary(tmp_path, monkeypatch):
    summary = "REPORT-HEAD\n" + ("细节行\n" * 600) + "REPORT-TAIL"
    _install_runner(
        monkeypatch, lambda self, request, thread_id: _ok(summary))
    context = _parent(tmp_path)
    manager = DelegationManager(
        _settings(delegation_report_chars=1200), context, "session-1")

    result = manager.delegate("researcher", "生成完整报告")
    report = manager.summary()[0]
    manager.close()
    context.store.close()

    data = json.loads(Path(report["report"]).read_text(encoding="utf-8"))
    markdown = Path(report["report_md"]).read_text(encoding="utf-8")
    assert data["schema"] == "nextcoder.subagent.report.v1"
    assert data["summary"] == summary
    assert data["task"] == "生成完整报告"
    assert data["report_markdown_path"] == report["report_md"]
    assert data["report_md"] == report["report_md"]
    assert data["report"] == report["report"]
    assert summary in markdown
    # 紧凑结果保留头尾，且完整报告入口不会被折叠切掉。
    assert "REPORT-HEAD" in result["text"]
    assert "REPORT-TAIL" in result["text"]
    assert "超出预算" in result["text"]
    assert f"完整报告：{report['report_md']}" in result["text"]


def test_report_artifact_keeps_command_and_failure_text(tmp_path, monkeypatch):
    command = "[run_command] status=completed exit_code=0\n" + ("ok\n" * 700)
    failure = "[write_file] status=failed error=denied\n" + ("bad\n" * 300)

    def run(self, request, thread_id):
        return {
            **_ok("完成"),
            "messages": [
                ToolMessage(content=command, tool_call_id="call-1"),
                ToolMessage(content=failure, tool_call_id="call-2"),
                AIMessage(content="完成"),
            ],
        }

    _install_runner(monkeypatch, run)
    context = _parent(tmp_path)
    manager = DelegationManager(_settings(), context, "session-1")

    manager.delegate("researcher", "运行检查")
    report = manager.summary()[0]
    manager.close()
    context.store.close()

    data = json.loads(Path(report["report"]).read_text(encoding="utf-8"))
    assert data["tests"] == [command]
    assert data["unresolved"] == [failure]
    assert len(data["tests"][0]) > 600


def test_failed_child_still_writes_report_artifact(tmp_path, monkeypatch):
    _install_runner(
        monkeypatch,
        lambda self, request, thread_id: RuntimeError("boom"))
    context = _parent(tmp_path)
    manager = DelegationManager(_settings(), context, "session-1")

    result = manager.delegate("researcher", "触发失败")
    report = manager.summary()[0]
    manager.close()
    context.store.close()

    data = json.loads(Path(report["report"]).read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert data["status"] == "failed"
    assert "RuntimeError" in data["error"]
    assert Path(report["report_md"]).is_file()
    assert "boom" in Path(report["report_md"]).read_text(encoding="utf-8")
    assert "boom" in result["text"]


def test_background_completion_notifies_once(tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def run(self, request, thread_id):
        started.set()
        release.wait(timeout=5)
        return _ok("后台完成")

    _install_runner(monkeypatch, run)
    context = _parent(tmp_path)
    manager = DelegationManager(_settings(), context, "session-1")
    try:
        result = manager.delegate("reviewer", "审查补丁", background=True)
        assert result["status"] == "running"
        assert started.wait(timeout=5)
        release.set()
        deadline = time.time() + 5
        while time.time() < deadline:
            notifications = manager.poll_notifications()
            if notifications:
                break
            time.sleep(0.02)
        assert notifications and "后台完成" in notifications[0]
        assert manager.poll_notifications() == []
    finally:
        manager.close()
        context.store.close()


def test_cancel_stops_running_child(tmp_path, monkeypatch):
    started = threading.Event()

    def run(self, request, thread_id):
        started.set()
        self.context.cancel_event.wait(timeout=5)
        return {
            **_ok("收到取消"),
            "stop_reason": "用户取消",
        }

    _install_runner(monkeypatch, run)
    context = _parent(tmp_path)
    manager = DelegationManager(_settings(), context, "session-1")
    try:
        manager.delegate("reviewer", "运行测试", background=True)
        assert started.wait(timeout=5)
        child_id = manager.summary()[0]["child_id"]
        assert "请求取消" in manager.cancel(child_id)
        deadline = time.time() + 5
        while time.time() < deadline:
            if manager.summary()[0]["status"] == "cancelled":
                break
            time.sleep(0.02)
        assert manager.summary()[0]["status"] == "cancelled"
    finally:
        manager.close()
        context.store.close()


def test_resume_uses_same_thread_and_respects_limit(tmp_path, monkeypatch):
    attempts = {"count": 0}

    def run(self, request, thread_id):
        attempts["count"] += 1
        return RuntimeError("总是失败")

    StubRunner = _install_runner(monkeypatch, run)
    context = _parent(tmp_path)
    manager = DelegationManager(_settings(), context, "session-1")
    try:
        first = manager.delegate("researcher", "查找线索")
        assert first["status"] == "failed"
        child_id = manager.summary()[0]["child_id"]
        first_thread = StubRunner.instances[0].calls[0][1]

        resumed = manager.resume(child_id)
        assert resumed["status"] == "failed"
        report = manager.summary()[0]
        assert report["resumes"] == 1
        assert len(StubRunner.instances) == 2
        assert StubRunner.instances[1].calls[0] == ("继续未完成的工作。", first_thread)

        again = manager.resume(child_id)
        assert again["status"] == "failed"
        assert "上限" in again["text"]
        assert len(StubRunner.instances) == 2
    finally:
        manager.close()
        context.store.close()


def _git_repo(path: Path) -> None:
    (path / ".gitignore").write_text(".nextcoder/\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=Test",
         "commit", "-q", "-m", "init"], cwd=path, check=True)


def test_worktree_isolates_changes_and_exports_patch(tmp_path, monkeypatch):
    def run(self, request, thread_id):
        (self.context.task_dir / "app.py").write_text("hello\n", encoding="utf-8")
        return _ok("已修改 app.py")

    _install_runner(monkeypatch, run)
    context = _parent(tmp_path)
    _git_repo(context.task_dir)
    manager = DelegationManager(_settings(), context, "session-1")
    try:
        result = manager.delegate("coder", "修改 app.py")
        report = manager.summary()[0]
        patch_path = Path(report["patch"])
        patch = patch_path.read_text(encoding="utf-8")
        assert result["status"] == "completed"
        assert "+hello" in patch
        assert (context.task_dir / "app.py").read_text(encoding="utf-8") == "base\n"
        assert report["changed_files"] == ["app.py"]
        assert not Path(report["workdir"]).exists()
        data = json.loads(Path(report["report"]).read_text(encoding="utf-8"))
        assert data["patch"] == str(patch_path)
        assert data["changed_files"] == ["app.py"]
        assert str(patch_path) in Path(report["report_md"]).read_text(encoding="utf-8")
    finally:
        manager.close()
        context.store.close()


def test_worktree_copies_hooks_config(tmp_path, monkeypatch):
    seen = {}

    def run(self, request, thread_id):
        hooks = self.context.task_dir / ".nextcoder" / "hooks.json"
        seen["exists"] = hooks.is_file()
        seen["text"] = hooks.read_text(encoding="utf-8") if hooks.is_file() else ""
        return _ok("已检查 hooks")

    _install_runner(monkeypatch, run)
    context = _parent(tmp_path)
    _git_repo(context.task_dir)
    hooks = context.task_dir / ".nextcoder" / "hooks.json"
    hooks.parent.mkdir(exist_ok=True)
    hooks.write_text('{"hooks": []}', encoding="utf-8")
    manager = DelegationManager(_settings(), context, "session-1")
    try:
        result = manager.delegate("coder", "检查 hooks 是否随 worktree 复制")
        assert result["status"] == "completed"
        assert seen["exists"]
        assert "hooks" in seen["text"]
    finally:
        manager.close()
        context.store.close()


def test_worktree_role_fails_outside_git(tmp_path, monkeypatch):
    _install_runner(monkeypatch, lambda self, request, thread_id: _ok())
    context = _parent(tmp_path)
    manager = DelegationManager(_settings(), context, "session-1")
    try:
        result = manager.delegate("coder", "尝试修改")
        assert result["status"] == "failed"
        assert "git 仓库" in result["text"] or "worktree" in result["text"]
    finally:
        manager.close()
        context.store.close()


def test_toolbox_blocks_tools_outside_allowlist(tmp_path):
    context = _parent(tmp_path)
    context.allowed_tools = {"read_lines"}
    box = ToolBox(context)
    try:
        text = box.execute("run_command", {"command": "echo hi"}, call_id="call-x")
        assert "[工具不可用]" in text
        assert context.store.get("call-x") is None
    finally:
        context.store.close()
