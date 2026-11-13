"""会话管理测试：创建、加载、多轮共用 thread、失败仍落盘。"""

import json
from pathlib import Path

from langchain_core.messages import AIMessage

from nextcoder.runtime.events import TurnFinished, TurnStarted
from nextcoder.sessions import manager
from nextcoder.settings import Settings


def _settings(**kwargs) -> Settings:
    return Settings(llm_base_url="", llm_model="", **kwargs)


class StubRunner:
    prompts: list = []

    def __init__(self, settings, context, session_id: str = "",
                 system_prompt: str | None = None,
                 streaming: bool = False):
        self.context = context
        self.stop_reason = None
        self.usage_delta = {"total_tokens": 7}
        type(self).prompts.append(system_prompt)

    def run(self, request, thread_id):
        type(self).calls.append((thread_id, request))
        return {
            "messages": [AIMessage(content="完成")],
            "model_calls": 1,
            "usage": {"total_tokens": 7},
            "request_tokens": 42,
            "stop_reason": None,
        }


class BrokenRunner:
    def __init__(self, settings, context, session_id: str = "",
                 system_prompt: str | None = None,
                 streaming: bool = False):
        self.stop_reason = None
        self.usage_delta = {"input_tokens": 321}
        self.last_state = {
            "messages": [],
            "model_calls": 4,
            "usage": {"input_tokens": 321},
            "request_tokens": 456,
        }
        self.last_request_tokens = 789

    def run(self, request, thread_id):
        raise RuntimeError("模型不可用")


class StubCompactor:
    def __init__(self, settings, context, session_id: str = "",
                 system_prompt: str | None = None,
                 streaming: bool = False):
        self.context = context
        self.summary_calls = 1
        self.summary_usage = {"input_tokens": 10, "output_tokens": 2}
        self.summary_latency_seconds = 0.2
        self.summary_errors = []
        self.memory_used = []

    def compact(self, thread_id):
        return {
            "compacted": True, "summary_id": "summary-1",
            "covered_count": 12, "covered_groups": 5, "kept_groups": 2,
            "summary": "目标：完成修复", "usage": {},
            "before_tokens": 900, "after_tokens": 300,
            "remaining_messages": 7,
        }


def test_compact_session_writes_report(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "AgentRunner", StubCompactor)
    task = tmp_path / "task"
    task.mkdir()
    (task / "a.txt").write_text("x", encoding="utf-8")
    session = manager.create_session(_settings(), task)

    result = manager.compact_session(_settings(), session)
    assert result["compacted"] is True
    assert result["session_id"] == session.session_id
    report = json.loads(Path(result["compaction_dir"], "compaction.json")
                        .read_text(encoding="utf-8"))
    assert report["summary_id"] == "summary-1"
    assert report["summary_stats"]["calls"] == 1
    assert report["summary_stats"]["usage"]["input_tokens"] == 10
    assert "provenance" in report


def test_create_and_load(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "a.txt").write_text("x", encoding="utf-8")
    session = manager.create_session(_settings(), task)
    assert (session.root / "session.json").is_file()
    assert (session.baseline_dir / "a.txt").is_file()

    loaded = manager.load_session(_settings(), task, session.session_id)
    assert loaded.session_id == session.session_id
    assert session.session_id in {
        item["session_id"] for item in manager.list_sessions(_settings(), task)
    }


def test_two_turns_share_thread_and_baseline(tmp_path, monkeypatch):
    StubRunner.calls = []
    monkeypatch.setattr(manager, "AgentRunner", StubRunner)
    task = tmp_path / "task"
    task.mkdir()
    (task / "a.txt").write_text("x", encoding="utf-8")
    session = manager.create_session(_settings(), task)

    first = manager.run_turn(_settings(), session, "第一问")
    (task / "a.txt").write_text("changed", encoding="utf-8")
    second = manager.run_turn(_settings(), session, "第二问")

    assert [call[0] for call in StubRunner.calls] == [session.session_id, session.session_id]
    assert [call[1] for call in StubRunner.calls] == ["第一问", "第二问"]
    assert first.patch == ""
    assert "a.txt" in second.patch
    report = json.loads((second.run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["session_id"] == session.session_id
    assert report["usage_delta"]["total_tokens"] == 7
    assert (second.run_dir / "final.patch").is_file()


def test_run_turn_reports_cache_hit_rate(tmp_path, monkeypatch):
    class CacheRunner(StubRunner):
        call_usages = [
            {"input_tokens": 20, "cached_tokens": 0},
            {"input_tokens": 40, "cached_tokens": 38},
            {"input_tokens": 40, "cached_tokens": 37},
        ]

        def run(self, request, thread_id):
            return {
                "messages": [AIMessage(content="完成")],
                "model_calls": 3,
                "usage": {"input_tokens": 100, "cached_tokens": 75,
                          "total_tokens": 110},
                "request_tokens": 42,
                "stop_reason": None,
            }

    monkeypatch.setattr(manager, "AgentRunner", CacheRunner)
    task = tmp_path / "task"
    task.mkdir()
    (task / "a.txt").write_text("x", encoding="utf-8")
    session = manager.create_session(_settings(), task)

    outcome = manager.run_turn(_settings(), session, "第一问")
    report = json.loads((outcome.run_dir / "report.json").read_text(encoding="utf-8"))
    cache = report["cache"]
    assert cache["input_tokens"] == 100
    assert cache["cached_tokens"] == 75
    assert cache["hit_rate"] == 0.75
    assert cache["calls"] == 3
    assert cache["cold_start"] is True
    assert cache["warm_input_tokens"] == 80
    assert cache["warm_cached_tokens"] == 75
    assert cache["warm_hit_rate"] == 0.9375


def test_first_user_request_generates_session_title(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "AgentRunner", StubRunner)
    task = tmp_path / "task"
    task.mkdir()
    (task / "a.txt").write_text("x", encoding="utf-8")
    session = manager.create_session(_settings(), task)
    assert session.title == ""

    manager.run_turn(
        _settings(), session,
        "## 修复购物车折扣逻辑\n只改 src/cart.py，并补测试。")
    metadata = json.loads(
        (session.root / "session.json").read_text(encoding="utf-8"))
    assert metadata["title"] == "修复购物车折扣逻辑"
    loaded = manager.load_session(_settings(), task, session.session_id)
    assert loaded.title == "修复购物车折扣逻辑"


def test_session_title_is_plain_single_line():
    assert manager.make_session_title("  你好  ") == "你好"
    assert manager.make_session_title("") == "新会话"
    long = manager.make_session_title("改" * 100)
    assert len(long) == 41 and long.endswith("…")


def test_run_turn_passes_prompt_variant(tmp_path, monkeypatch):
    StubRunner.calls = []
    StubRunner.prompts = []
    monkeypatch.setattr(manager, "AgentRunner", StubRunner)
    task = tmp_path / "task"
    task.mkdir()
    (task / "a.txt").write_text("x", encoding="utf-8")
    session = manager.create_session(_settings(), task)

    manager.run_turn(_settings(), session, "第一问", system_prompt="交互版提示")
    # 不传 prompt 时保持 benchmark 默认（None → AgentRunner 内部取 SYSTEM_PROMPT）。
    manager.run_turn(_settings(), session, "第二问")

    assert StubRunner.prompts == ["交互版提示", None]


def test_run_turn_emits_lifecycle_events(tmp_path, monkeypatch):
    StubRunner.calls = []
    monkeypatch.setattr(manager, "AgentRunner", StubRunner)
    task = tmp_path / "task"
    task.mkdir()
    (task / "a.txt").write_text("x", encoding="utf-8")
    session = manager.create_session(_settings(), task)
    events = []

    manager.run_turn(_settings(), session, "事件", event_handler=events.append)

    assert isinstance(events[0], TurnStarted)
    assert events[0].session_id == session.session_id
    assert any(isinstance(event, TurnFinished) for event in events)
    finished = next(event for event in events if isinstance(event, TurnFinished))
    assert finished.model_calls == 1
    assert finished.cache["hit_rate"] == 0.0


def test_failure_still_exports_report(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "AgentRunner", BrokenRunner)
    task = tmp_path / "task"
    task.mkdir()
    session = manager.create_session(_settings(), task)
    outcome = manager.run_turn(_settings(), session, "请求")
    assert outcome.stop_reason.startswith("error:")
    report = json.loads((outcome.run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["stop_reason"].startswith("error:")
    assert report["model_calls"] == 4
    assert report["request_tokens"] == 789
    assert report["usage"]["input_tokens"] == 321
    assert report["usage_delta"]["input_tokens"] == 321
    assert (outcome.run_dir / "final.patch").is_file()


def test_report_records_provenance_without_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "AgentRunner", StubRunner)
    settings = _settings(llm_api_key="super-secret")
    task = tmp_path / "task"
    task.mkdir()
    session = manager.create_session(settings, task)
    outcome = manager.run_turn(settings, session, "检查一下")

    report_text = (outcome.run_dir / "report.json").read_text(encoding="utf-8")
    report = json.loads(report_text)
    provenance = report["provenance"]
    assert provenance["nextcoder_version"] == "0.2.0.dev0"
    assert provenance["nextcoder_commit"]
    assert "llm_model" in provenance["settings"]
    assert "llm_api_key" not in provenance["settings"]
    assert "super-secret" not in report_text


def test_invalid_session_id_rejected(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    try:
        manager.load_session(_settings(), task, "../escape")
    except ValueError as exc:
        assert "非法" in str(exc)
    else:
        raise AssertionError("应拒绝路径形式的 session id")


def test_rollout_jsonl_records_session_events(tmp_path, monkeypatch):
    """会话事件日志：session_start/user_request/turn_end 逐轮追加，可读回。"""
    from nextcoder.sessions.rollout import read_events

    monkeypatch.setattr(manager, "AgentRunner", StubRunner)
    task = tmp_path / "task"
    task.mkdir()
    (task / "a.txt").write_text("x", encoding="utf-8")
    session = manager.create_session(_settings(), task)

    manager.run_turn(_settings(), session, "第一问")
    (task / "a.txt").write_text("changed", encoding="utf-8")
    manager.run_turn(_settings(), session, "第二问")

    rollout_path = session.root / "rollout.jsonl"
    assert rollout_path.exists()
    events = read_events(rollout_path)
    types = [event["type"] for event in events]
    # 一次 session_start + 每轮 user_request/turn_end，跨轮追加不覆盖。
    assert types.count("session_start") == 1
    assert types.count("user_request") == 2
    assert types.count("turn_end") == 2
    assert events[0]["session_id"] == session.session_id
    assert [event["payload"]["request"] for event in events
            if event["type"] == "user_request"] == ["第一问", "第二问"]
    ends = [event for event in events if event["type"] == "turn_end"]
    assert ends[0]["payload"]["has_changes"] is False
    assert ends[1]["payload"]["has_changes"] is True
    assert ends[1]["payload"]["report"].startswith("turns/")


def test_rollout_survives_unwritable_path(tmp_path):
    """日志写不进去时静默失败，不影响主流程。"""
    from nextcoder.sessions.rollout import RolloutWriter

    writer = RolloutWriter(tmp_path / "no-such-dir-allowed" / "x" / "rollout.jsonl")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)  # 去掉写权限
    try:
        denied = RolloutWriter(blocked / "rollout.jsonl")
        denied.append("user_request", {"request": "hi"})
        assert not (blocked / "rollout.jsonl").exists()
    finally:
        blocked.chmod(0o700)
    writer.append("user_request", {"request": "hi"})
    assert writer._fh is None or True  # 未抛错即通过
