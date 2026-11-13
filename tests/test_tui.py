"""TUI 离线测试：提交、流式渲染、任务清单、权限弹窗、补全与取消。"""

import asyncio
import time
from types import SimpleNamespace

import pytest
from textual.widgets import Footer, Markdown, Static

from nextcoder.runtime.events import (
    AssistantDelta,
    PermissionAnswered,
    PermissionRequested,
    PlanUpdated,
    ToolFinished,
    ToolStarted,
    TurnFinished,
)
from nextcoder.runtime.permissions import PermissionRequest
from nextcoder.runtime.planning import PlanStore
from nextcoder.sessions import manager
from nextcoder.settings import Settings
from nextcoder.tui import app as tui_app
from nextcoder.tui.completion import (
    apply_completion,
    build_file_index,
    command_token,
    file_token,
    match_commands,
    match_files,
)
from nextcoder.tui.screens import (
    ConfigScreen,
    ConfigValues,
    PermissionModeScreen,
    WelcomeScreen,
    build_configured_settings,
    effort_from_settings,
)
from nextcoder.tui.widgets import ComposerTextArea, PermissionScreen, cache_label


def _settings() -> Settings:
    return Settings(
        llm_base_url="http://localhost", llm_model="fake", llm_api_key="test",
        memory_enabled=False, delegation_enabled=False)


def _make_app(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "a.txt").write_text("x\n", encoding="utf-8")
    settings = _settings()
    session = manager.create_session(settings, task)
    return tui_app.NextCoderApp(settings, session)


def _composer(app) -> ComposerTextArea:
    return app.query_one("#composer-area", ComposerTextArea)


async def _wait_idle(app, pilot, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.05)
        if not app._busy:
            return
    raise AssertionError("TUI 轮次没有在超时内结束")


def _outcome(tmp_path, stop_reason="model_finished"):
    return SimpleNamespace(stop_reason=stop_reason, turn_id="turn-1",
                           run_dir=tmp_path / "run")


def test_submit_streams_and_updates_plan(tmp_path, monkeypatch):
    app = _make_app(tmp_path)

    def fake_run_turn(settings, session, request, **kwargs):
        PlanStore(session.task_dir / settings.runs_root
                  / "task-plan.json").replace(["写代码"])
        handler = kwargs["event_handler"]
        handler(AssistantDelta("turn-1", "你"))
        handler(AssistantDelta("turn-1", "好"))
        handler(PlanUpdated("turn-1", 1, (
            {"id": "t1", "content": "写代码", "status": "pending"},)))
        handler(ToolStarted("turn-1", "c1", "read_lines", "a.txt"))
        handler(ToolFinished("turn-1", "c1", "read_lines", "completed", "ok"))
        handler(TurnFinished("turn-1", "model_finished", 2, {},
                             {"input_tokens": 10, "cached_tokens": 5,
                              "hit_rate": 0.5}))
        return _outcome(tmp_path)

    monkeypatch.setattr(tui_app, "run_turn", fake_run_turn)

    async def scenario():
        async with app.run_test() as pilot:
            _composer(app).set_text("打个招呼")
            await pilot.press("enter")
            await _wait_idle(app, pilot)
            assert "> 打个招呼" in app.transcript_text
            assert "你好" in app.transcript_text
            assert "read_lines" in app.transcript_text
            assert "写代码" in app.plan_text
            assert "缓存 50%" in app.status_text
            assert _composer(app).disabled is False
            # 助手消息走 Markdown 部件，而不是纯 Static。
            assert app.query(Markdown)
            # 轮次标题必须排在用户消息、助手内容和 footer 之前。
            card = app.query_one(".round-card")
            children = list(card.children)
            assert "ROUND 01" in str(children[0].render())
            assert "工作区写入" in str(children[0].render())
            kinds = [type(child).__name__ for child in children]
            assert kinds.index("Markdown") < len(kinds) - 1

    asyncio.run(scenario())


def test_plan_panel_keeps_completed_and_cancelled_marks(tmp_path, monkeypatch):
    app = _make_app(tmp_path)

    def fake_run_turn(settings, session, request, **kwargs):
        store = PlanStore(session.task_dir / settings.runs_root / "task-plan.json")
        store.replace(["a", "b", "c"])
        store.update({"t1": "completed", "t3": "cancelled"})
        handler = kwargs["event_handler"]
        handler(PlanUpdated("turn-1", 2, (
            {"id": "t1", "content": "a", "status": "completed"},
            {"id": "t2", "content": "b", "status": "pending"},
            {"id": "t3", "content": "c", "status": "cancelled"},
        )))
        handler(TurnFinished("turn-1", "model_finished", 1, {}, {}))
        return _outcome(tmp_path)

    monkeypatch.setattr(tui_app, "run_turn", fake_run_turn)

    async def scenario():
        async with app.run_test() as pilot:
            _composer(app).set_text("做个计划")
            await pilot.press("enter")
            await _wait_idle(app, pilot)
            panel = app.query_one("#plan", Static)
            rendered = "\n".join(
                panel.render_line(index).text for index in range(panel.size.height))
            assert "[x] t1 a" in rendered
            assert "[ ] t2 b" in rendered
            assert "[-] t3 c" in rendered

    asyncio.run(scenario())


def test_permission_modal_allow_once(tmp_path, monkeypatch):
    app = _make_app(tmp_path)
    captured = {}

    def fake_run_turn(settings, session, request, **kwargs):
        handler = kwargs["event_handler"]
        handler(PermissionRequested("turn-1", "write_file", "write_path",
                                    "a.txt", "敏感路径", "确认后写入"))
        answer = kwargs["approval_callback"](PermissionRequest(
            action="write_path", target="a.txt", tool="write_file",
            reason="敏感路径", hint="确认后写入"))
        captured["answer"] = answer
        handler(PermissionAnswered("turn-1", "write_file", "write_path",
                                   "a.txt", answer))
        handler(TurnFinished("turn-1", "model_finished", 1, {}, {}))
        return _outcome(tmp_path)

    monkeypatch.setattr(tui_app, "run_turn", fake_run_turn)

    async def scenario():
        async with app.run_test() as pilot:
            _composer(app).set_text("写文件")
            await pilot.press("enter")
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                await pilot.pause(0.05)
                if isinstance(app.screen, PermissionScreen):
                    break
            assert isinstance(app.screen, PermissionScreen)
            await pilot.click("#allow-once")
            await _wait_idle(app, pilot)
            assert captured["answer"] == "allow_once"
            assert app.permission_answer == "allow_once"

    asyncio.run(scenario())


def test_ctrl_c_cancels_running_turn(tmp_path, monkeypatch):
    app = _make_app(tmp_path)
    captured = {}

    def fake_run_turn(settings, session, request, **kwargs):
        event = kwargs["cancel_event"]
        captured["event"] = event
        event.wait(timeout=5)
        kwargs["event_handler"](TurnFinished("turn-1", "用户取消", 1, {}, {}))
        return _outcome(tmp_path, "用户取消")

    monkeypatch.setattr(tui_app, "run_turn", fake_run_turn)

    async def scenario():
        async with app.run_test() as pilot:
            _composer(app).set_text("长任务")
            await pilot.press("enter")
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and "event" not in captured:
                await pilot.pause(0.05)
            await pilot.press("ctrl+c")
            await _wait_idle(app, pilot)
            assert captured["event"].is_set()
            assert "用户取消" in app.activity_text

    asyncio.run(scenario())


def test_slash_command_suggestions_and_tab_accept():
    matches = match_commands("/m")
    assert [command.name for command in matches] == ["/mode", "/model"]
    assert command_token("/mode auto") is None
    assert command_token("look /mode") is None
    assert apply_completion("/mo", "/mode", command=True) == "/mode "

    async def scenario():
        settings = _settings()
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            app = tui_app.NextCoderApp(
                settings, manager.create_session(settings, task))
            async with app.run_test() as pilot:
                composer = _composer(app)
                for char in "/m":
                    await pilot.press(char)
                await pilot.pause()
                assert [value for value, _label, _kind
                        in app._suggestion_values] == ["/mode", "/model"]
                assert app._suggestion_kind == "command"
                await pilot.press("tab")
                await pilot.pause()
                assert composer.text == "/mode "
                assert composer.suggestion_active is False

    asyncio.run(scenario())


def test_file_mention_suggestions_and_tab_accept(tmp_path):
    task = tmp_path / "task"
    (task / "src").mkdir(parents=True)
    (task / "src" / "app.py").write_text("print(1)\n", encoding="utf-8")
    (task / ".github" / "workflows").mkdir(parents=True)
    (task / ".github" / "workflows" / "ci.yml").write_text(
        "name: ci\n", encoding="utf-8")
    paths = build_file_index(task)
    assert ".github/" in paths
    assert "src/app.py" in paths
    assert match_files(paths, ".g")[0] == ".github/"
    assert file_token("see @src/ap") == (4, "src/ap")
    assert file_token("mail me@host") is None
    assert apply_completion("see @src/ap", "src/app.py", command=False) == \
        "see @src/app.py "

    settings = _settings()
    app = tui_app.NextCoderApp(settings, manager.create_session(settings, task))

    async def scenario():
        async with app.run_test() as pilot:
            composer = _composer(app)
            for char in "see @src/ap":
                await pilot.press(char)
            await pilot.pause()
            assert [value for value, _label, _kind
                    in app._suggestion_values] == ["src/app.py"]
            await pilot.press("tab")
            await pilot.pause()
            assert composer.text == "see @src/app.py "

    asyncio.run(scenario())


def test_welcome_creates_session_when_started_without_one(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    app = tui_app.NextCoderApp(_settings(), task_dir=task)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, WelcomeScreen)
            assert app.session is None
            await pilot.press("n")
            await pilot.pause()
            assert app.session is not None
            assert not isinstance(app.screen, WelcomeScreen)
            assert _composer(app).disabled is False

    asyncio.run(scenario())


def test_loaded_session_replays_round_history(tmp_path):
    from nextcoder.sessions.rollout import RolloutWriter

    task = tmp_path / "task"
    task.mkdir()
    settings = _settings()
    session = manager.create_session(settings, task)
    rollout = RolloutWriter(session.root / "rollout.jsonl",
                            session_id=session.session_id)
    rollout.append("user_request", {"request": "first question"}, turn_id="t1")
    rollout.append("assistant", {"content": "## Answer\n\n- one"}, turn_id="t1")
    rollout.append("turn_end", {"stop_reason": "model_finished",
                                "model_calls": 1}, turn_id="t1")
    rollout.close()

    app = tui_app.NextCoderApp(settings, task_dir=task)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, WelcomeScreen)
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            assert app.session is not None
            assert app._round_no == 1
            assert "first question" in app.transcript_text
            assert "Answer" in app.transcript_text
            assert len(app.query(".round-card")) == 1

    asyncio.run(scenario())


def test_mode_picker_escape_returns_to_welcome(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    app = tui_app.NextCoderApp(_settings(), task_dir=task)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, WelcomeScreen)
            await pilot.press("m")
            await pilot.pause()
            assert isinstance(app.screen, PermissionModeScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, WelcomeScreen)
            assert app.settings.permission_mode == "workspace"

    asyncio.run(scenario())


def test_cache_label_distinguishes_cold_start_from_cache_failure():
    assert cache_label(None) == ""
    assert cache_label({
        "input_tokens": 100, "cached_tokens": 0, "hit_rate": 0.0, "calls": 1,
    }) == "缓存 首次"
    assert cache_label({
        "input_tokens": 100, "cached_tokens": 75, "hit_rate": 0.75,
        "calls": 3, "warm_hit_rate": 0.9375,
    }) == "缓存 75%（暖 94%）"
    assert cache_label({
        "input_tokens": 100, "cached_tokens": 75, "hit_rate": 0.75,
        "calls": 3, "warm_hit_rate": 0.9375,
    }, compact=True) == "缓存 94%（暖）"
    assert cache_label({
        "input_tokens": 100, "cached_tokens": 95, "hit_rate": 0.95,
        "calls": 2, "warm_hit_rate": 0.95,
    }) == "缓存 95%"


def test_mode_picker_highlights_current_mode(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    app = tui_app.NextCoderApp(_settings(), task_dir=task)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("m")
            await pilot.pause()
            assert isinstance(app.screen, PermissionModeScreen)
            listing = app.screen.query_one("#mode-list")
            assert listing.highlighted == 1  # 默认工作区写入，而不是永远第一项
            description = app.screen.query_one("#mode-desc", Static)
            assert "工作区写入" in str(description.render())

    asyncio.run(scenario())


def test_config_screen_reports_unusable_sandbox(tmp_path, monkeypatch):
    from nextcoder.tui import screens

    monkeypatch.setattr(
        screens, "bwrap_status",
        lambda: ("unusable", "/usr/bin/bwrap", "namespace creation denied"))
    app = tui_app.NextCoderApp(_settings(), task_dir=tmp_path)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("c")
            await pilot.pause()
            assert isinstance(app.screen, ConfigScreen)
            status = app.screen.query_one("#cfg-sandbox-status", Static)
            text = str(status.render())
            assert "无法创建命名空间" in text
            assert "namespace creation denied" in text

    asyncio.run(scenario())


def test_mode_command_switches_permission_mode(tmp_path):
    app = _make_app(tmp_path)

    async def scenario():
        async with app.run_test() as pilot:
            _composer(app).set_text("/mode auto")
            await pilot.press("enter")
            await pilot.pause()
            assert app.settings.permission_mode == "auto"
            assert "沙箱内自动" in app.transcript_text

    asyncio.run(scenario())


def test_config_merge_manual_endpoint_effort_and_sandbox():
    settings = Settings(llm_base_url="http://old", llm_model="old",
                        llm_api_key="test")
    configured = build_configured_settings(
        settings, {}, ConfigValues(
            endpoint="http://127.0.0.1:9/v1", model="new", effort="high",
            permission_mode="auto", sandbox=True,
            sandbox_network=True, sandbox_fail_closed=False))
    assert configured.llm_base_url == "http://127.0.0.1:9/v1"
    assert configured.llm_model == "new"
    assert configured.llm_thinking == "enabled"
    assert configured.llm_reasoning_effort == "high"
    assert configured.permission_mode == "auto"
    assert configured.command_sandbox is True
    assert configured.command_sandbox_network is True
    assert configured.command_sandbox_fail_closed is False
    assert configured.model_profile == ""
    assert configured.llm_provider == "custom"
    assert effort_from_settings(configured) == "high"

    with pytest.raises(ValueError):
        build_configured_settings(settings, {}, ConfigValues(
            endpoint="ftp://bad", model="x"))
    with pytest.raises(ValueError):
        build_configured_settings(settings, {}, ConfigValues(
            endpoint="http://ok", model="x", effort="turbo"))


def test_config_merge_profile_api_key_and_manual_override():
    from nextcoder.model_config import ModelProfile

    settings = Settings(llm_api_key="old", llm_provider="old")
    profile = ModelProfile(
        name="p", provider="p", base_url="http://profile/v1",
        model="profile-model", api_key_env="NEXTCODER_TEST_UNSET_KEY")
    configured = build_configured_settings(
        settings, {"p": profile}, ConfigValues(
            profile_name="p", endpoint="http://profile/v1",
            model="profile-model", effort="medium", api_key="typed-key"))
    assert configured.model_profile == "p"
    assert configured.llm_api_key.get_secret_value() == "typed-key"
    assert configured.llm_thinking == "enabled"
    assert configured.llm_reasoning_effort == "medium"

    overridden = build_configured_settings(
        settings, {"p": profile}, ConfigValues(
            profile_name="p", endpoint="http://elsewhere/v1",
            model="other", effort="off"))
    assert overridden.model_profile == ""
    assert overridden.llm_provider == "custom"
    assert overridden.llm_fallback_base_url == ""
    assert overridden.llm_base_url == "http://elsewhere/v1"


def test_header_status_and_bottom_dock_do_not_overlap(tmp_path):
    app = _make_app(tmp_path)

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            brand = app.query_one("#brand")
            status = app.query_one("#status")
            hint = app.query_one("#composer-hint")
            footer = app.query_one(Footer)
            assert brand.region.y == 0
            assert status.region.y == brand.region.y + brand.region.height
            assert hint.region.y + hint.region.height <= footer.region.y
            assert footer.region.y + footer.region.height <= 40
            assert app.query_one("#suggestions").display is False
            assert not app.status_text.startswith("◆")
            assert "工作区写入" in app.status_text
            assert "沙箱关" in app.status_text

    asyncio.run(scenario())


def test_turn_footer_scrolls_into_view_and_activity_completes(tmp_path, monkeypatch):
    app = _make_app(tmp_path)

    def fake_run_turn(settings, session, request, **kwargs):
        handler = kwargs["event_handler"]
        body = "\n\n".join(f"第 {index} 段说明，用来把 transcript 撑高。"
                           for index in range(30))
        handler(AssistantDelta("turn-1", body))
        handler(PlanUpdated("turn-1", 1, (
            {"id": "t1", "content": "检查文件", "status": "completed"},
            {"id": "t2", "content": "运行测试", "status": "in_progress"},
        )))
        handler(TurnFinished("turn-1", "model_finished", 1, {},
                             {"input_tokens": 10, "cached_tokens": 5,
                              "hit_rate": 0.5}))
        return _outcome(tmp_path)

    monkeypatch.setattr(tui_app, "run_turn", fake_run_turn)

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            _composer(app).set_text("跑一个长任务")
            await pilot.press("enter")
            await _wait_idle(app, pilot)
            await pilot.pause(0.15)
            transcript = app.query_one("#transcript")
            assert transcript.max_scroll_y > 0
            assert transcript.scroll_y == transcript.max_scroll_y
            footer = transcript.query_one(".round-footer")
            viewport_bottom = transcript.region.y + transcript.region.height
            assert footer.region.y + footer.region.height <= viewport_bottom
            assert app.activity_text == "本轮完成"

    asyncio.run(scenario())


def test_config_essentials_fit_at_120x40(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    app = tui_app.NextCoderApp(_settings(), task_dir=task)

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("c")
            await pilot.pause()
            assert isinstance(app.screen, ConfigScreen)
            body = app.screen.query_one("#config-body")
            body_bottom = body.region.y + body.region.height
            for widget_id in ("cfg-mode", "cfg-sandbox", "cfg-network",
                              "cfg-strict", "cfg-api-key"):
                widget = app.screen.query_one(f"#{widget_id}")
                assert widget.display
                assert body.region.y <= widget.region.y
                assert widget.region.y + widget.region.height <= body_bottom

    asyncio.run(scenario())
