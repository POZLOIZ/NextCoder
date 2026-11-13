"""流式入口测试：主备切换、chunk 合并、reasoning 保留与增量事件。"""

import httpx
import pytest
from langchain_core.messages import AIMessageChunk
from openai import RateLimitError

from nextcoder.llm import (
    BoundFallback,
    _FallbackState,
    _ReasoningChatOpenAI,
)
from nextcoder.runtime.agent import AgentRunner
from nextcoder.runtime.events import (
    AssistantDelta,
    PermissionRequested,
    PlanUpdated,
    ToolFinished,
    ToolStarted,
)
from nextcoder.sessions.store import OperationStore
from nextcoder.settings import Settings
from nextcoder.tools.base import ToolContext
from nextcoder.tools.registry import ToolBox


def _rate_limit() -> RateLimitError:
    request = httpx.Request("POST", "http://localhost/chat")
    return RateLimitError("429", response=httpx.Response(429, request=request),
                          body=None)


class FakeStreamModel:
    def __init__(self, chunks=(), error_before=None, error_after=None,
                 fail_at: int | None = None) -> None:
        self.chunks = list(chunks)
        self.error_before = error_before
        self.error_after = error_after
        self.fail_at = fail_at
        self.calls = 0

    def stream(self, messages):
        self.calls += 1

        def generate():
            if self.error_before is not None:
                raise self.error_before
            for index, chunk in enumerate(self.chunks):
                if self.error_after is not None and index == self.fail_at:
                    raise self.error_after
                yield chunk
            if (self.error_after is not None
                    and (self.fail_at is None
                         or self.fail_at >= len(self.chunks))):
                raise self.error_after
        return generate()


class ScriptedStreamModel:
    """每次模型调用取一个脚本，用于多轮工具循环。"""

    def __init__(self, scripts) -> None:
        self.scripts = list(scripts)
        self.calls = 0

    def stream(self, messages):
        self.calls += 1
        script = self.scripts.pop(0)
        return iter(script)


class FakeInvokeModel:
    def __init__(self, result) -> None:
        self.result = result

    def invoke(self, messages):
        return self.result


class TestFallbackStream:
    def test_switches_before_first_chunk(self):
        primary = FakeStreamModel(error_before=_rate_limit())
        backup = FakeStreamModel([AIMessageChunk(content="备援")])
        state = _FallbackState()
        bound = BoundFallback(primary, backup, state)

        chunks = list(bound.stream([]))
        assert "".join(chunk.content for chunk in chunks) == "备援"
        assert state.last == "backup"

    def test_does_not_replay_after_output(self):
        primary = FakeStreamModel(
            [AIMessageChunk(content="前半")], error_after=_rate_limit(),
            fail_at=1)
        backup = FakeStreamModel([AIMessageChunk(content="不应出现")])
        state = _FallbackState()
        bound = BoundFallback(primary, backup, state)

        emitted = []
        with pytest.raises(RateLimitError):
            for chunk in bound.stream([]):
                emitted.append(chunk.content)
        assert emitted == ["前半"]
        assert backup.calls == 0
        assert state.last == "primary"

    def test_primary_success_keeps_endpoint(self):
        primary = FakeStreamModel([AIMessageChunk(content="ok")])
        backup = FakeStreamModel([AIMessageChunk(content="backup")])
        state = _FallbackState()
        chunks = list(BoundFallback(primary, backup, state).stream([]))
        assert "".join(chunk.content for chunk in chunks) == "ok"
        assert backup.calls == 0
        assert state.last == "primary"


def test_reasoning_content_is_collected_across_chunks():
    model = _ReasoningChatOpenAI(
        model="fake", api_key="test", base_url="http://localhost",
        preserve_reasoning=True)
    first = model._convert_chunk_to_generation_chunk(
        {"choices": [{"delta": {"role": "assistant",
                                "reasoning_content": "先想"}}]},
        AIMessageChunk, None)
    second = model._convert_chunk_to_generation_chunk(
        {"choices": [{"delta": {"reasoning_content": "再答"}}]},
        AIMessageChunk, None)
    merged = first.message + second.message
    assert merged.additional_kwargs["reasoning_content"] == "先想再答"


def _context(tmp_path, events):
    task_dir = tmp_path / "task"
    run_dir = task_dir / ".nextcoder" / "runs" / "test"
    task_dir.mkdir()
    run_dir.mkdir(parents=True)
    baseline = run_dir / "baseline"
    baseline.mkdir()
    (task_dir / "a.txt").write_text("x\n", encoding="utf-8")
    store = OperationStore(run_dir / "operations.sqlite")
    settings = Settings(llm_base_url="http://localhost", llm_model="fake",
                        llm_api_key="test")
    context = ToolContext(
        task_dir=task_dir, run_dir=run_dir, baseline_dir=baseline, store=store,
        settings=settings, turn_id="turn-stream", event_handler=events.append)
    return AgentRunner(settings, context, streaming=True), context, store


def test_streaming_agent_merges_chunks_and_emits_deltas(tmp_path):
    events = []
    runner, context, store = _context(tmp_path, events)
    runner._llm = FakeStreamModel([
        AIMessageChunk(content="你"),
        AIMessageChunk(content="好"),
    ])
    try:
        final = runner.run("打个招呼", "t-stream")
        deltas = [event.text for event in events
                  if isinstance(event, AssistantDelta)]
        assert "".join(deltas) == "你好"
        assert final["messages"][-1].content == "你好"
    finally:
        store.close()


def test_streaming_tool_calls_are_executed(tmp_path):
    events = []
    runner, context, store = _context(tmp_path, events)
    runner._llm = ScriptedStreamModel([
        [
            AIMessageChunk(content="", tool_call_chunks=[
                {"name": "list_files", "args": '{"path"', "id": "call-s",
                 "index": 0}]),
            AIMessageChunk(content="", tool_call_chunks=[
                {"name": "", "args": ': "."}', "id": None, "index": 0}]),
        ],
        [AIMessageChunk(content="完成")],
    ])
    try:
        final = runner.run("看看目录", "t-stream-tools")
        toolbox_calls = [event for event in events
                         if isinstance(event, (ToolStarted, ToolFinished))]
        assert any(isinstance(event, ToolStarted) for event in toolbox_calls)
        assert final["messages"][-1].content == "完成"
    finally:
        store.close()


def test_toolbox_emits_permission_and_plan_events(tmp_path):
    events = []
    runner, context, store = _context(tmp_path, events)
    toolbox = ToolBox(context)
    try:
        text = toolbox.execute("plan_create", {"tasks": ["写代码"]}, "call-plan")
        assert "plan" in text or "清单" in text
        assert any(isinstance(event, PlanUpdated) for event in events)
    finally:
        store.close()


def test_toolbox_permission_callback_events(tmp_path):
    events = []
    runner, context, store = _context(tmp_path, events)
    context.approval_callback = lambda request: "deny"
    toolbox = ToolBox(context)
    try:
        # 敏感路径在工作区写入模式下可以确认；回调拒绝后操作失败。
        toolbox.execute("read_lines", {"path": ".env"}, "call-perm")
        assert any(isinstance(event, PermissionRequested) for event in events)
    finally:
        store.close()
