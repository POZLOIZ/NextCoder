"""上下文压缩测试：工具正文裁剪、语义摘要请求、保存与复用。"""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from nextcoder.runtime.agent import AgentRunner, _estimate_tokens
from nextcoder.runtime.context import (
    apply_summary,
    compact_messages,
    prepare_context,
    preserves_latest_user,
)
from nextcoder.sessions import manager
from nextcoder.sessions.store import OperationStore
from nextcoder.sessions.summary import ModelSummarizer, SummaryOutput, SummaryStore
from nextcoder.settings import Settings
from nextcoder.tools.base import ToolContext


def _settings(**overrides) -> Settings:
    values = dict(
        llm_base_url="http://localhost", llm_model="fake", llm_api_key="test",
        application_input_token_budget=4000, output_reserve_tokens=100,
        compaction_trigger_ratio=0.5, compaction_keep_recent_groups=2,
        compaction_recent_token_ratio=0.5, compaction_stub_chars=120,
        compaction_summary_chars=900,
    )
    values.update(overrides)
    return Settings(**values)


def _summary_settings(**overrides) -> Settings:
    """低预算：第一层裁剪后仍超过低水位，必须请求语义摘要。"""
    values = dict(
        application_input_token_budget=1100,
        compaction_recent_token_ratio=0.5,
        # 这些用例专门验证摘要路径；把低水位压到 50%，避免廉价折叠直接收手。
        compaction_low_watermark_ratio=0.5,
    )
    values.update(overrides)
    return _settings(**values)


def _agent_settings(**overrides) -> Settings:
    """运行循环还要计入工具定义；预算要覆盖回退视图和摘要成功后的视图。"""
    values = dict(
        application_input_token_budget=5000,
        compaction_recent_token_ratio=0.5,
        # 这些用例只验证一次摘要；重试/收敛由专门的用例覆盖。
        compaction_summary_retry_limit=0,
    )
    values.update(overrides)
    return _settings(**values)


def _history(turns: int = 8) -> list:
    messages = [
        SystemMessage(content="系统提示"),
        HumanMessage(content="最初需求：修复解析器并跑测试"),
    ]
    for index in range(turns):
        call_id = f"call-{index}"
        messages.append(AIMessage(content="", tool_calls=[{
            "name": "run_command", "args": {"command": f"echo {index}"},
            "id": call_id,
        }]))
        messages.append(ToolMessage(
            content=(f"[run_command] status=completed changed=files/{index}.py "
                     f"full_output=/logs/{call_id}-command.log\n"
                     + ("输出内容" * 200)),
            tool_call_id=call_id,
        ))
    return messages


def _estimate_messages(messages: list) -> int:
    return sum(_estimate_tokens(str(getattr(message, "content", "")))
               for message in messages)


def _is_stub(message) -> bool:
    return isinstance(message, AIMessage) and "[早期工具摘要]" in str(message.content)


def _assert_no_orphan_tools(messages: list) -> None:
    pending: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage):
            pending = {call["id"] for call in (message.tool_calls or [])}
        elif isinstance(message, ToolMessage):
            assert message.tool_call_id in pending


def test_tool_stubs_fold_old_tools_and_keep_recent():
    messages = _history()
    view, event = compact_messages(messages, _settings(), _estimate_messages)

    assert event.triggered
    assert event.method == "tool_stubs"
    assert event.after_tokens < event.before_tokens
    assert event.compacted_groups >= 1
    assert view[0].type == "system"
    assert view[1].type == "human"
    assert "最初需求" in view[1].content

    stubs = [message for message in view if _is_stub(message)]
    assert stubs, "较早的工具组应折叠成摘要"
    first_stub = stubs[0].content
    assert "call-0" in first_stub
    assert "/logs/call-0-command.log" in first_stub
    assert "status=completed" in first_stub
    assert "输出内容" not in first_stub

    # 最近两组工具结果保留完整正文（可回读的摘要仍然指向日志）。
    text = "\n".join(str(message.content) for message in view)
    assert "call-7" in text and "call-6" in text
    _assert_no_orphan_tools(view)


def test_baseline_mode_summarizes_instead_of_stopping_at_tool_stubs():
    """关闭第一层后，折叠视图即使已低于触发线，也要请求模型摘要。"""
    messages = _history()
    prepared = prepare_context(
        messages, _settings(compaction_stub_first=False), _estimate_messages)

    assert prepared.event.triggered
    assert prepared.summary_request is not None
    assert prepared.fallback_view is not None
    assert prepared.event.after_tokens < prepared.event.before_tokens
    _assert_no_orphan_tools(prepared.fallback_view)


def test_no_compaction_under_threshold():
    messages = _history(turns=1)
    prepared = prepare_context(
        messages, _settings(application_input_token_budget=100000),
        _estimate_messages)
    assert not prepared.event.triggered
    assert prepared.view is messages
    assert prepared.summary_request is None


def test_requested_summary_covers_complete_groups_only():
    messages = _history()
    prepared = prepare_context(messages, _summary_settings(), _estimate_messages)

    assert prepared.event.triggered
    assert prepared.summary_request is not None
    request = prepared.summary_request
    assert request.covered_groups >= 1
    assert request.cutoff > request.old_covered_count
    assert request.cutoff % 2 == 0, "摘要边界落在工具组之间"
    assert not isinstance(messages[request.cutoff], ToolMessage)
    assert any("输出内容" in str(message.content)
               for message in request.messages)
    # 请求事件报告的是第一层回退视图，而不是“已覆盖 0 组”。
    assert prepared.event.compacted_groups >= 1
    assert prepared.event.covered_groups == 0
    assert prepared.fallback_view is not None


def test_summary_does_not_cover_latest_user_request():
    messages = _history(turns=4) + [HumanMessage(content="新的要求：补一个回归测试")]
    prepared = prepare_context(messages, _summary_settings(), _estimate_messages)

    assert prepared.summary_request is not None
    cutoff = prepared.summary_request.cutoff
    last_user = max(index for index, message in enumerate(messages)
                    if isinstance(message, HumanMessage))
    assert cutoff <= last_user
    assert preserves_latest_user(messages, cutoff)
    assert all("新的要求" not in str(message.content)
               for message in prepared.summary_request.messages)


def test_apply_summary_builds_model_view():
    messages = _history()
    prepared = prepare_context(messages, _summary_settings(), _estimate_messages)
    request = prepared.summary_request
    assert request is not None

    view, event = apply_summary(
        messages, _summary_settings(), _estimate_messages,
        "目标：修复解析器\n已验证：pytest 通过", request,
        summary_id="summary-1", summary_chars=20)

    assert event.method == "model_summary"
    assert event.covered_groups == request.covered_groups
    assert event.summary_id == "summary-1"
    assert event.before_tokens == request.before_tokens
    assert view[0].type == "system"
    assert view[1].type == "human"
    assert isinstance(view[2], AIMessage)
    assert "[任务摘要 v2]" in view[2].content
    assert "目标：修复解析器" in view[2].content
    assert "输出内容" not in "\n".join(str(message.content) for message in view[2:3])
    assert messages[request.cutoff:] == view[3:]
    _assert_no_orphan_tools(view)


def test_apply_summary_reports_when_still_above_trigger():
    messages = _history()
    prepared = prepare_context(messages, _summary_settings(), _estimate_messages)
    request = prepared.summary_request
    assert request is not None

    _view, event = apply_summary(
        messages, _summary_settings(), _estimate_messages,
        "摘要" * 2000, request, summary_id="summary-2", summary_chars=4000)

    assert event.after_tokens > 500
    assert "仍高于触发线" in event.reason


def test_saved_summary_is_reused_even_under_trigger():
    messages = _history()
    prepared = prepare_context(messages, _summary_settings(), _estimate_messages)
    request = prepared.summary_request
    assert request is not None

    reused = prepare_context(
        messages, _summary_settings(), _estimate_messages,
        summary_text="目标：修复解析器", covered_count=request.cutoff)

    assert reused.summary_request is None
    assert not reused.event.triggered
    assert reused.event.method == "summary_reuse"
    assert reused.event.summary_reused
    assert reused.event.covered_groups == request.covered_groups
    assert "[任务摘要 v2]" in reused.view[2].content


def test_previous_stub_view_is_reused_without_new_event():
    """同一检查点全量历史变大时，不能每轮都重新记一次 tool_stubs 事件。"""
    messages = _history()
    settings = _settings()
    first = prepare_context(messages, settings, _estimate_messages)
    assert first.event.triggered
    assert first.event.method == "tool_stubs"

    reused = prepare_context(
        messages, settings, _estimate_messages,
        prior_request_tokens=first.event.after_tokens,
        prior_message_count=len(messages))

    assert not reused.event.triggered
    assert reused.event.method == "tool_stubs_reuse"
    assert reused.view is not messages
    assert any(_is_stub(message) for message in reused.view)
    _assert_no_orphan_tools(reused.view)


def test_invalid_summary_coverage_falls_back_to_stubs():
    messages = _history()
    # 覆盖点落在 ToolMessage 上：必须拒绝，而不是切出孤立工具结果。
    prepared = prepare_context(
        messages, _summary_settings(), _estimate_messages,
        summary_text="目标：修复解析器", covered_count=3)

    assert prepared.event.method == "tool_stubs"
    assert not prepared.event.summary_reused
    assert prepared.view is not None
    _assert_no_orphan_tools(prepared.view)


def test_summary_extension_keeps_old_summary_and_new_range():
    messages = _history()
    # 旧摘要只覆盖前三组；后面仍有完整工具组，必须请求合并式扩展。
    old_cutoff = 8
    extended = prepare_context(
        messages, _summary_settings(), _estimate_messages,
        summary_text="旧摘要", covered_count=old_cutoff)

    assert extended.summary_request is not None
    request = extended.summary_request
    assert request.old_summary == "旧摘要"
    assert request.old_covered_count == old_cutoff
    assert request.cutoff > old_cutoff
    assert request.messages == messages[old_cutoff:request.cutoff]
    assert extended.event.summary_reused
    assert extended.event.covered_groups == 3
    assert extended.fallback_view is not None


def test_force_compaction_below_trigger_keeps_recent_working_set():
    messages = _history(turns=12)
    settings = _agent_settings(application_input_token_budget=100000)
    prepared = prepare_context(
        messages, settings, _estimate_messages, force=True)

    assert prepared.summary_request is not None
    request = prepared.summary_request
    assert request.covered_groups < 12, "主动压缩也必须保留最近工具组"
    assert request.cutoff % 2 == 0
    assert preserves_latest_user(messages, request.cutoff)
    # 未超过触发线也要生成摘要，但事件解释为主动压缩。
    assert "主动压缩" in prepared.event.reason


def test_force_compaction_is_noop_without_foldable_groups():
    messages = _history(turns=2)
    settings = _agent_settings(application_input_token_budget=100000)
    prepared = prepare_context(
        messages, settings, _estimate_messages, force=True)
    assert prepared.summary_request is None
    assert prepared.event.method == "tool_stubs"


class FakeLLM:
    def __init__(self) -> None:
        self.seen: list[list] = []

    def invoke(self, messages):
        self.seen.append(list(messages))
        return AIMessage(
            content="完成",
            usage_metadata={"input_tokens": 100, "output_tokens": 5,
                            "total_tokens": 105},
        )


class FakeSummarizer:
    def __init__(self, summary: str = "目标：修复解析器\n已验证：pytest 通过") -> None:
        self.summary = summary
        self.calls: list[tuple[str, list, str]] = []

    def summarize(self, old_summary, messages, task: str = ""):
        self.calls.append((old_summary, list(messages), task))
        return SummaryOutput(
            summary=self.summary,
            memory_suggestions=[],
            usage={"input_tokens": 40, "output_tokens": 10, "total_tokens": 50},
        )


class FailingSummarizer:
    def summarize(self, old_summary, messages, task: str = ""):
        raise RuntimeError("summary model unavailable")


def _runner(tmp_path, settings):
    task_dir = tmp_path / "task"
    run_dir = task_dir / ".nextcoder" / "runs" / "test"
    task_dir.mkdir()
    run_dir.mkdir(parents=True)
    baseline = run_dir / "baseline"
    baseline.mkdir()
    from nextcoder.sessions.store import OperationStore
    from nextcoder.tools.base import ToolContext

    store = OperationStore(run_dir / "operations.sqlite")
    context = ToolContext(task_dir=task_dir, run_dir=run_dir, baseline_dir=baseline,
                          store=store, settings=settings, turn_id="turn-1")
    return AgentRunner(settings, context), context, store


def _state(messages: list) -> dict:
    return {"messages": messages, "model_calls": 0, "usage": {},
            "request_tokens": 0, "stop_reason": None}


def test_model_node_saves_summary_and_uses_it(tmp_path):
    settings = _agent_settings()
    runner, context, store = _runner(tmp_path, settings)
    fake = FakeLLM()
    summarizer = FakeSummarizer()
    runner._llm = fake
    runner._summarizer = summarizer
    runner._thread_id = "session-1"
    messages = _history()
    expected = prepare_context(
        messages, settings, runner._estimate_request_tokens).summary_request
    assert expected is not None

    result = runner._model_node(_state(messages))
    store.close()

    assert summarizer.calls, "超过预算时应调用摘要模型"
    assert summarizer.calls[0][2] == "最初需求：修复解析器并跑测试", (
        "首条用户要求不在摘要消息范围里，必须单独传给摘要模型")
    assert runner.summary_calls == 1
    record = SummaryStore(context.summary_path).latest("session-1")
    assert record is not None
    assert record.covered_count == expected.cutoff
    assert record.summary.startswith("目标：修复解析器")

    sent = fake.seen[0]
    assert any(isinstance(message, AIMessage)
               and "[任务摘要 v2]" in str(message.content) for message in sent)
    event = runner.compaction_events[-1]
    assert event.method == "model_summary"
    assert event.summary_id == record.summary_id
    assert event.covered_groups >= 1

    # 摘要调用的 usage 计入本轮总量和单独统计。
    assert result["usage"]["input_tokens"] == 140
    assert result["usage"]["output_tokens"] == 15
    assert runner.summary_usage["input_tokens"] == 40
    assert not runner.summary_errors


class _PromptLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(
            content=json.dumps({
                "summary": "目标：修复库存 low_stock 并跑测试",
                "memory_suggestions": [],
            }, ensure_ascii=False),
            usage_metadata={"input_tokens": 20, "output_tokens": 8,
                            "total_tokens": 28},
        )


def test_model_summarizer_prompt_keeps_original_task():
    """摘要输入不含首条用户要求时，提示词必须单独带上任务目标。"""
    summarizer = ModelSummarizer(_agent_settings(), llm=_PromptLLM())
    prompt = summarizer._llm.prompts
    output = summarizer.summarize(
        "", [HumanMessage(content="继续实现")],
        task="修复库存 low_stock 并跑测试")

    assert prompt and "修复库存 low_stock" in prompt[0]
    assert "不能把目标写成" in prompt[0]
    assert output.summary.startswith("目标：修复库存")


def test_model_node_falls_back_when_summary_fails(tmp_path):
    settings = _agent_settings()
    runner, context, store = _runner(tmp_path, settings)
    fake = FakeLLM()
    runner._llm = fake
    runner._summarizer = FailingSummarizer()
    runner._thread_id = "session-1"

    result = runner._model_node(_state(_history()))
    store.close()

    assert runner.summary_calls == 1
    assert runner.summary_errors and "summary model unavailable" in runner.summary_errors[0]
    assert SummaryStore(context.summary_path).latest("session-1") is None
    assert result["usage"]["input_tokens"] == 100
    sent_text = "\n".join(str(message.content) for message in fake.seen[0])
    assert "[任务摘要 v2]" not in sent_text
    assert "call-0" in sent_text
    assert runner.compaction_events[-1].triggered
    assert runner.compaction_events[-1].covered_groups == 0


def test_model_node_reuses_persisted_summary(tmp_path):
    settings = _agent_settings()
    runner, context, store = _runner(tmp_path, settings)
    fake = FakeLLM()
    runner._llm = fake
    runner._summarizer = FailingSummarizer()
    messages = _history()

    summary_store = SummaryStore(context.summary_path)
    record = summary_store.save(
        "session-1", 16, "目标：修复解析器", model="fake",
        usage={"input_tokens": 40, "output_tokens": 10})
    loaded = runner._load_summary("session-1", messages)
    assert loaded is not None and loaded.summary_id == record.summary_id
    runner._summary_record = loaded
    runner._thread_id = "session-1"

    runner._model_node(_state(messages))
    store.close()

    sent_text = "\n".join(str(message.content) for message in fake.seen[0])
    assert "[任务摘要 v2]" in sent_text
    assert "目标：修复解析器" in sent_text
    assert "call-0" not in sent_text, "摘要覆盖的早期工具组不应再发送完整正文"
    assert "call-7" in sent_text, "最近工具组保持完整"
    assert runner.summary_calls == 0, "已保存摘要不应再次调用摘要模型"
    assert runner.compaction_events[-1].method == "summary_reuse"


def test_model_node_rejects_summary_covering_latest_user(tmp_path):
    settings = _agent_settings()
    runner, context, store = _runner(tmp_path, settings)
    messages = _history(turns=4) + [HumanMessage(content="新的要求")]
    SummaryStore(context.summary_path).save(
        "session-1", len(messages), "错误摘要", model="fake")

    assert runner._load_summary("session-1", messages) is None
    assert runner.summary_errors
    store.close()


def test_active_compact_saves_summary_and_is_idempotent(tmp_path):
    settings = _agent_settings(application_input_token_budget=100000)
    task = tmp_path / "task"
    task.mkdir()
    session = manager.create_session(settings, task)
    store = OperationStore(session.operations_db)
    context = ToolContext(
        task_dir=task, run_dir=session.root / "compact-source",
        baseline_dir=session.baseline_dir, store=store, settings=settings,
        checkpoints_db=session.checkpoints_db, turn_id="compact-source")
    runner = AgentRunner(settings, context)
    summarizer = FakeSummarizer("目标：修复解析器\n已验证：pytest 通过")
    runner._summarizer = summarizer
    messages = _history(turns=12)

    config = {"configurable": {"thread_id": session.session_id}}
    with SqliteSaver.from_conn_string(str(session.checkpoints_db)) as saver:
        graph = runner._build_graph(saver)
        graph.update_state(config, {"messages": messages}, as_node="model")

    result = runner.compact(session.session_id)
    assert result["compacted"] is True
    assert result["covered_count"] > 2
    assert result["after_tokens"] < result["before_tokens"]
    assert len(summarizer.calls) == 1
    record = SummaryStore(context.summary_path).latest(session.session_id)
    assert record is not None
    assert record.covered_count == result["covered_count"]

    # 再次主动压缩时，旧摘要已覆盖全部可折叠组，不应重复调用摘要模型。
    second = runner.compact(session.session_id)
    assert second["compacted"] is False
    assert len(summarizer.calls) == 1
    store.close()


def test_active_compact_refuses_pending_tool_call(tmp_path):
    settings = _agent_settings(application_input_token_budget=100000)
    task = tmp_path / "task"
    task.mkdir()
    session = manager.create_session(settings, task)
    store = OperationStore(session.operations_db)
    context = ToolContext(
        task_dir=task, run_dir=session.root / "compact-source",
        baseline_dir=session.baseline_dir, store=store, settings=settings,
        checkpoints_db=session.checkpoints_db, turn_id="compact-source")
    runner = AgentRunner(settings, context)
    messages = _history(turns=3) + [AIMessage(content="", tool_calls=[{
        "name": "read_lines", "args": {"path": "x.py"}, "id": "pending-call",
    }])]
    config = {"configurable": {"thread_id": session.session_id}}
    with SqliteSaver.from_conn_string(str(session.checkpoints_db)) as saver:
        graph = runner._build_graph(saver)
        graph.update_state(config, {"messages": messages}, as_node="model")

    result = runner.compact(session.session_id)
    assert result["compacted"] is False
    assert "尚未结束" in result["reason"]
    store.close()


def test_tool_fold_below_low_watermark_stops_without_summary():
    """折叠后不超过 70% 低水位：廉价折叠已经够用，不再调用摘要。"""
    messages = _history()
    prepared = prepare_context(messages, _settings(), _estimate_messages)

    assert prepared.event.triggered
    assert prepared.event.method == "tool_stubs"
    assert prepared.summary_request is None
    assert prepared.event.low_watermark == 2730
    assert prepared.event.after_tokens <= prepared.event.low_watermark
    assert "低水位" in prepared.event.reason


def test_tool_fold_above_low_watermark_requests_summary():
    """折叠后仍高于 70% 低水位：即使低于 90% 触发线也要摘要。"""
    messages = _history()
    settings = _settings(compaction_low_watermark_ratio=0.25)
    prepared = prepare_context(messages, settings, _estimate_messages)

    assert prepared.event.triggered
    assert prepared.summary_request is not None
    assert prepared.event.low_watermark == 975
    assert prepared.event.after_tokens > prepared.event.low_watermark
    assert "低水位" in prepared.event.reason


def test_summary_retry_converges_with_smaller_recent_window(tmp_path):
    settings = _agent_settings(
        application_input_token_budget=10204,
        compaction_trigger_ratio=0.12,
        compaction_low_watermark_ratio=0.05,
        compaction_recent_token_ratio=0.16,
        compaction_retry_recent_ratio=0.08,
        compaction_summary_retry_limit=1,
        compaction_keep_recent_groups=4,
    )
    runner, context, store = _runner(tmp_path, settings)
    runner._estimate_request_tokens = _estimate_messages
    runner._llm = FakeLLM()
    runner._summarizer = FakeSummarizer()
    runner._thread_id = "session-retry"
    messages = _history(turns=12)

    runner._model_node(_state(messages))
    store.close()

    assert runner.summary_calls == 2, "第一次摘要仍超线时应重试一次"
    first, second = runner.compaction_events[-2], runner.compaction_events[-1]
    assert first.summary_attempts == 1
    assert second.summary_attempts == 2
    assert second.after_tokens < first.after_tokens
    assert second.retry_reason
    record = SummaryStore(context.summary_path).latest("session-retry")
    assert record is not None
    assert record.covered_count == 2 + 2 * second.covered_groups


def test_summary_retry_stops_when_not_converging(tmp_path):
    class GrowingSummarizer:
        def __init__(self) -> None:
            self.calls = 0

        def summarize(self, old_summary, messages, task: str = ""):
            self.calls += 1
            size = 20 if self.calls == 1 else 4000
            return SummaryOutput(
                summary="摘" * size,
                memory_suggestions=[],
                usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            )

    settings = _agent_settings(
        application_input_token_budget=10204,
        compaction_trigger_ratio=0.12,
        compaction_low_watermark_ratio=0.05,
        compaction_recent_token_ratio=0.16,
        compaction_retry_recent_ratio=0.08,
        compaction_summary_retry_limit=1,
        compaction_keep_recent_groups=4,
    )
    runner, context, store = _runner(tmp_path, settings)
    runner._estimate_request_tokens = _estimate_messages
    runner._llm = FakeLLM()
    summarizer = GrowingSummarizer()
    runner._summarizer = summarizer
    runner._thread_id = "session-no-converge"

    runner._model_node(_state(_history(turns=12)))
    store.close()

    assert summarizer.calls == 2, "不收敛时不能无限重试"
    event = runner.compaction_events[-1]
    assert "未收敛" in event.reason
