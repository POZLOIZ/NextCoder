"""运行循环离线测试：消息角色、检查点、请求预算与操作复用。不依赖模型 API。"""

import sqlite3

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from nextcoder.runtime.agent import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_INTERACTIVE,
    AgentRunner,
    _normalize_usage,
)
from nextcoder.sessions.store import OperationStore
from nextcoder.settings import Settings
from nextcoder.tools.base import ToolContext, hash_content
from nextcoder.tools.registry import ToolBox


class FakeLLM:
    """按脚本回放模型响应，并记录每次收到的消息；不联网。"""

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.seen: list[list] = []

    def invoke(self, messages):
        self.seen.append(list(messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _make_runner(tmp_path, **settings_kwargs):
    # 这些测试关注运行循环本身；默认放行命令，避免权限询问掩盖被测行为。
    settings_kwargs.setdefault("permission_mode", "bypass")
    task_dir = tmp_path / "task"
    run_dir = task_dir / ".nextcoder" / "runs" / "test"
    task_dir.mkdir()
    run_dir.mkdir(parents=True)
    baseline = run_dir / "baseline"
    baseline.mkdir()
    store = OperationStore(run_dir / "operations.sqlite")
    context = ToolContext(
        task_dir=task_dir, run_dir=run_dir, baseline_dir=baseline, store=store,
        settings=Settings(llm_base_url="http://localhost", llm_model="fake",
                          llm_api_key="test", **settings_kwargs),
        turn_id="turn-1",
    )
    return AgentRunner(context.settings, context), context, store


def _checkpoint_count(run_dir) -> int:
    db = sqlite3.connect(run_dir / "checkpoints.sqlite")
    try:
        return db.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
    finally:
        db.close()


def _tool_call(call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "list_files", "args": {"path": "."}, "id": call_id}],
    )


@pytest.fixture
def context(tmp_path):
    runner, ctx, store = _make_runner(tmp_path)
    yield ctx
    store.close()


class TestCheckpointer:
    def test_checkpoints_persisted(self, tmp_path):
        """v0.1 曾只打开 saver 未传给 compile()，检查点表为空——本测试防回归。"""
        runner, context, store = _make_runner(tmp_path)
        runner._llm = FakeLLM([AIMessage(content="完成")])
        final = runner.run("需求", "t-checkpoint")
        count = _checkpoint_count(context.run_dir)
        store.close()
        assert final["stop_reason"] is None
        assert count >= 1

    def test_user_request_is_human_message(self, tmp_path):
        runner, context, store = _make_runner(tmp_path)
        fake = FakeLLM([AIMessage(content="完成")])
        runner._llm = fake
        runner.run("需求", "t-roles")
        store.close()
        assert isinstance(fake.seen[0][0], SystemMessage)
        assert isinstance(fake.seen[0][1], HumanMessage)
        assert fake.seen[0][1].content == "需求"

    def test_second_turn_reuses_history(self, tmp_path):
        runner, context, store = _make_runner(tmp_path)
        fake = FakeLLM([AIMessage(content="第一轮完成"), AIMessage(content="第二轮完成")])
        runner._llm = fake
        runner.run("第一问", "t-resume")
        runner.run("第二问", "t-resume")
        store.close()
        second = fake.seen[1]
        assert sum(isinstance(message, SystemMessage) for message in second) == 1
        assert isinstance(second[-1], HumanMessage)
        assert second[-1].content == "第二问"
        assert any(getattr(message, "content", "") == "第一轮完成" for message in second)

    def test_pending_tool_call_from_crash_is_completed(self, tmp_path):
        """模拟模型返回 tool_call 后进程退出：下一轮先补 ToolMessage 再继续。"""
        runner, context, store = _make_runner(tmp_path)
        config = {"configurable": {"thread_id": "t-crash"}}
        crashed = AIMessage(content="", tool_calls=[
            {"name": "list_files", "args": {"path": "."}, "id": "call-crash"}])
        with SqliteSaver.from_conn_string(str(context.checkpoints_path)) as saver:
            graph = runner._build_graph(saver)
            graph.update_state(config, {
                "messages": [SystemMessage(content=SYSTEM_PROMPT),
                             HumanMessage(content="原需求"), crashed],
                "model_calls": 1, "usage": {}, "request_tokens": 0,
                "stop_reason": None,
            }, as_node="model")

        fake = FakeLLM([AIMessage(content="已恢复并继续")])
        runner._llm = fake
        final = runner.run("继续", "t-crash")
        recovered = [message for message in fake.seen[0] if isinstance(message, ToolMessage)]
        operation = store.get("call-crash")
        store.close()

        assert any(message.tool_call_id == "call-crash" for message in recovered)
        assert fake.seen[0][-1].content == "继续"
        assert operation is not None and operation.status == "completed"
        assert final["stop_reason"] is None

    def test_exception_keeps_completed_call_metrics(self, tmp_path):
        """接口中途失败时，已完成的模型调用与用量仍要出现在报告里。"""
        runner, context, store = _make_runner(tmp_path)
        runner._llm = FakeLLM([_tool_call(), RuntimeError("endpoint returned None")])
        with pytest.raises(RuntimeError):
            runner.run("需求", "t-error")
        state = runner.last_state
        store.close()
        assert state is not None
        assert state["model_calls"] == 1
        assert runner.last_request_tokens > 0
        assert isinstance(state["messages"][-1], ToolMessage)


class TestRequestBudget:
    def test_budget_stops_when_full_request_exceeds(self, tmp_path):
        runner, context, store = _make_runner(tmp_path)
        initial = runner._estimate_request_tokens(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content="需求")])
        runner._settings.application_input_token_budget = (
            initial + runner._settings.output_reserve_tokens + 1)
        runner._llm = FakeLLM([_tool_call(), AIMessage(content="完成")])
        final = runner.run("需求", "t-budget")
        operations = [(op.tool, op.status) for op in store.all_operations()]
        store.close()
        assert operations == [("list_files", "completed")]
        assert final["model_calls"] == 1
        assert "本次请求上下文估算" in (final["stop_reason"] or "")

    def test_budget_not_hit_run_completes(self, tmp_path):
        runner, context, store = _make_runner(
            tmp_path, application_input_token_budget=64000)
        runner._llm = FakeLLM([_tool_call(), AIMessage(content="完成")])
        final = runner.run("需求", "t-normal")
        store.close()
        assert final["stop_reason"] is None
        assert final["request_tokens"] > 0
        assert final["model_calls"] == 2

    def test_cancel_stops_before_model_call(self, tmp_path):
        runner, context, store = _make_runner(tmp_path)
        context.cancel_event.set()
        runner._llm = FakeLLM([])
        final = runner.run("需求", "t-cancel")
        store.close()
        assert final["stop_reason"] == "用户取消"
        assert final["model_calls"] == 0


class TestStopReasons:
    def test_output_truncation_is_retried_with_smaller_steps(self, tmp_path):
        """一次输出截断只丢弃响应并提醒拆小步骤，不能直接结束任务。"""
        runner, context, store = _make_runner(tmp_path)
        fake = FakeLLM([
            AIMessage(content="部分", response_metadata={"finish_reason": "length"}),
            AIMessage(content="完成"),
        ])
        runner._llm = fake
        final = runner.run("需求", "t-length")
        store.close()
        assert final["stop_reason"] is None
        assert final["model_calls"] == 2
        retry_request = "\n".join(str(message.content) for message in fake.seen[1])
        assert "输出长度提醒" in retry_request
        assert "拆小" in retry_request
        assert not any(
            isinstance(message, AIMessage) and message.content == "部分"
            for message in final["messages"])

    def test_repeated_truncation_stops_after_limit(self, tmp_path):
        runner, context, store = _make_runner(tmp_path, length_retry_limit=1)
        fake = FakeLLM([
            AIMessage(content="部分", response_metadata={"finish_reason": "length"}),
            AIMessage(content="又一部分", response_metadata={"finish_reason": "length"}),
        ])
        runner._llm = fake
        final = runner.run("需求", "t-length-limit")
        store.close()
        assert final["model_calls"] == 2
        assert "连续 2 次" in (final["stop_reason"] or "")
        assert "截断" in (final["stop_reason"] or "")

    def test_truncated_tool_calls_are_not_executed(self, tmp_path):
        """截断响应可能带不完整的 tool_calls，必须丢弃而不是执行。"""
        runner, context, store = _make_runner(tmp_path)
        fake = FakeLLM([
            AIMessage(content="", tool_calls=[{
                "name": "write_file",
                "args": {"path": "broken.txt", "content": "half"},
                "id": "truncated-call",
            }], response_metadata={"finish_reason": "length"}),
            AIMessage(content="完成"),
        ])
        runner._llm = fake
        final = runner.run("需求", "t-length-tool")
        operations = store.all_operations()
        store.close()
        assert final["stop_reason"] is None
        assert operations == []
        assert not (context.task_dir / "broken.txt").exists()
        retry_request = "\n".join(str(message.content) for message in fake.seen[1])
        assert "工具调用没有执行" in retry_request

    def test_empty_response_is_reported(self, tmp_path):
        runner, context, store = _make_runner(tmp_path)
        runner._llm = FakeLLM([AIMessage(content="")])
        final = runner.run("需求", "t-empty")
        store.close()
        assert "空响应" in (final["stop_reason"] or "")


class TestUsage:
    def test_usage_delta_and_cumulative(self, tmp_path):
        runner, context, store = _make_runner(tmp_path)
        response = AIMessage(content="完成", usage_metadata={
            "input_tokens": 10, "output_tokens": 2, "total_tokens": 12})
        runner._llm = FakeLLM([response, response])
        first = runner.run("一", "t-usage")
        first_delta = dict(runner.usage_delta)
        second = runner.run("二", "t-usage")
        second_delta = dict(runner.usage_delta)
        store.close()
        assert first_delta["total_tokens"] == 12
        assert second_delta["total_tokens"] == 12
        assert second["usage"]["total_tokens"] == 24
        assert first["usage"]["total_tokens"] == 12

    def test_usage_normalizes_provider_cache_fields(self):
        deepseek = _normalize_usage({
            "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
            "prompt_cache_hit_tokens": 60,
        })
        assert deepseek == {
            "input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
            "cached_tokens": 60,
        }

        anthropic = _normalize_usage({
            "input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
            "cache_read_input_tokens": 40, "cache_creation_input_tokens": 5,
        })
        assert anthropic["cached_tokens"] == 40
        assert anthropic["cache_creation_tokens"] == 5
        assert "cache_read_input_tokens" not in anthropic

        langchain = _normalize_usage({
            "input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
            "input_token_details": {"cache_read": 30, "cache_creation": 2},
            "output_token_details": {"reasoning": 3},
        })
        assert langchain["cached_tokens"] == 30
        assert langchain["cache_creation_tokens"] == 2
        assert langchain["reasoning_tokens"] == 3
        assert "input_token_details" not in langchain
        assert langchain["input_tokens"] == 100


class TestOperationReplay:
    def test_completed_call_is_not_executed_twice(self, context):
        box = ToolBox(context)
        args = {"command": "echo once >> counter.txt"}
        first = box.execute("run_command", args, call_id="call-x")
        second = box.execute("run_command", args, call_id="call-x")
        assert first == second
        assert (context.task_dir / "counter.txt").read_text(encoding="utf-8") == "once\n"

    def test_new_call_id_executes_again(self, context):
        box = ToolBox(context)
        args = {"command": "echo once >> counter.txt"}
        box.execute("run_command", args, call_id="call-x")
        box.execute("run_command", args, call_id="call-y")
        assert (context.task_dir / "counter.txt").read_text(encoding="utf-8") == "once\nonce\n"

    def test_started_operation_is_not_replayed(self, context):
        context.store.prepare("run_command", {"command": "touch side-effect.txt"},
                              call_id="call-z")
        context.store.transition("call-z", "started")
        box = ToolBox(context)
        text = box.execute("run_command", {"command": "touch side-effect.txt"},
                           call_id="call-z")
        assert "不自动重放" in text
        assert not (context.task_dir / "side-effect.txt").exists()


class TestLoopGuards:
    def test_identical_calls_execute_and_only_add_advisory_reminder(self, tmp_path):
        """相同调用不再缓存/阻断；第三次只提醒检查是否重复探索。"""
        runner, context, store = _make_runner(tmp_path, repeat_warn_limit=3)
        responses = [
            AIMessage(content="", tool_calls=[{
                "name": "list_files", "args": {"path": "."}, "id": f"repeat-{i}"}])
            for i in range(3)
        ] + [AIMessage(content="完成")]
        fake = FakeLLM(responses)
        runner._llm = fake
        final = runner.run("需求", "t-repeat")
        operations = [(op.tool, op.status) for op in store.all_operations()]
        store.close()

        assert final["stop_reason"] is None
        assert len(fake.seen) == 4
        tool_texts = [str(message.content) for message in final["messages"]
                      if isinstance(message, ToolMessage)]
        assert all("已硬禁用" not in text for text in tool_texts)
        assert all("已阻止再次执行" not in text for text in tool_texts)
        assert operations == [("list_files", "completed")] * 3
        fourth_request = "\n".join(
            str(message.content) for message in fake.seen[3])
        assert "重复探索提醒" in fourth_request
        assert "结果不会变化" not in fourth_request
        assert "直接修改代码" not in fourth_request

    def test_search_reminder_is_advisory_and_never_blocks(self, tmp_path):
        """搜索计数只提醒方向，不硬禁用，也不要求先修改代码。"""
        runner, context, store = _make_runner(tmp_path, search_warn_limit=2)
        calls = [
            ("search", {"pattern": "alpha", "path": "."}),
            ("run_command", {"command": "grep -rn beta ."}),
            ("search", {"pattern": "gamma", "path": "."}),
            ("run_command", {"command": "find . -name '*.py'"}),
        ]
        responses = [
            AIMessage(content="", tool_calls=[{
                "name": name, "args": args, "id": f"search-{index}"}])
            for index, (name, args) in enumerate(calls)
        ] + [AIMessage(content="完成")]
        fake = FakeLLM(responses)
        runner._llm = fake
        final = runner.run("需求", "t-search-advice")
        operations = [(op.tool, op.status) for op in store.all_operations()]
        store.close()

        assert final["stop_reason"] is None
        tool_texts = [str(message.content) for message in final["messages"]
                      if isinstance(message, ToolMessage)]
        assert all("硬禁用" not in text for text in tool_texts)
        assert operations == [("search", "completed"),
                              ("run_command", "completed"),
                              ("search", "completed"),
                              ("run_command", "completed")]
        third_request = "\n".join(str(message.content) for message in fake.seen[2])
        assert "搜索提醒" in third_request
        assert "不能说明" in third_request

    def test_budget_warning_is_advisory_at_configured_ratio(self, tmp_path):
        """用掉配置比例的调用后只提醒收敛，不改变完成语义。"""
        runner, context, store = _make_runner(
            tmp_path, model_calls_limit=2, budget_warn_ratio=0.5)
        fake = FakeLLM([
            AIMessage(content="", tool_calls=[{
                "name": "list_files", "args": {"path": "."}, "id": "budget-0"}]),
            AIMessage(content="完成"),
        ])
        runner._llm = fake
        final = runner.run("需求", "t-budget")
        store.close()

        assert final["stop_reason"] is None
        second_request = "\n".join(str(message.content) for message in fake.seen[1])
        assert "预算提醒" in second_request
        assert "1/2" in second_request

    def test_five_identical_calls_are_blocked_after_replan(self, tmp_path):
        """连续 5 次相同调用后要求重新规划，并拒绝下一次相同调用。"""
        runner, context, store = _make_runner(tmp_path)
        responses = [
            AIMessage(content="", tool_calls=[{
                "name": "list_files", "args": {"path": "."}, "id": f"exact-{i}"}])
            for i in range(6)
        ] + [AIMessage(content="完成")]
        fake = FakeLLM(responses)
        runner._llm = fake
        final = runner.run("需求", "t-exact")
        operations = [(op.tool, op.status) for op in store.all_operations()]
        store.close()

        assert final["stop_reason"] is None
        assert operations == [("list_files", "completed")] * 5
        sixth_request = "\n".join(str(message.content) for message in fake.seen[5])
        assert "循环干预" in sixth_request
        tool_texts = [str(message.content) for message in final["messages"]
                      if isinstance(message, ToolMessage)]
        assert any("调用已阻止" in text for text in tool_texts)

    def test_same_command_executes_again_after_environment_change(self, tmp_path):
        """装依赖后重跑同一个命令，必须得到新结果，而不是旧缓存。"""
        # 本测试验证重执行语义；用完全放行模式避免权限确认打断命令。
        runner, context, store = _make_runner(tmp_path, permission_mode="bypass")
        command = "printf 'x\\n' >> state.txt; cat state.txt"
        calls = [
            ("run_command", {"command": command}),
            ("run_command", {"command": "printf 'dep\\n' >> dep.txt"}),
            ("run_command", {"command": command}),
        ]
        responses = [
            AIMessage(content="", tool_calls=[{
                "name": name, "args": args, "id": f"rerun-{index}"}])
            for index, (name, args) in enumerate(calls)
        ] + [AIMessage(content="完成")]
        fake = FakeLLM(responses)
        runner._llm = fake
        final = runner.run("需求", "t-rerun")
        store.close()

        assert final["stop_reason"] is None
        assert (context.task_dir / "dep.txt").read_text(encoding="utf-8") == "dep\n"
        assert (context.task_dir / "state.txt").read_text(
            encoding="utf-8") == "x\nx\n"

    def test_empty_output_streak_adds_diagnostic_reminder(self, tmp_path):
        runner, context, store = _make_runner(tmp_path, empty_output_warn_limit=3)
        commands = ["true", ":", "printf ''"]
        responses = [
            AIMessage(content="", tool_calls=[{
                "name": "run_command", "args": {"command": command},
                "id": f"empty-{index}"}])
            for index, command in enumerate(commands)
        ] + [AIMessage(content="完成")]
        fake = FakeLLM(responses)
        runner._llm = fake
        final = runner.run("需求", "t-empty")
        store.close()

        assert final["stop_reason"] is None
        fourth_request = "\n".join(str(message.content) for message in fake.seen[3])
        assert "[反死循环]" in fourth_request
        assert "没有输出" in fourth_request
        assert "2>/dev/null" in fourth_request

    def test_no_edit_streak_adds_one_advisory_reminder(self, tmp_path):
        """长时间只探索不落盘时提醒一次，但不阻断、不强制修改。"""
        runner, context, store = _make_runner(tmp_path, no_edit_warn_limit=2)
        responses = [
            AIMessage(content="", tool_calls=[{
                "name": "list_files", "args": {"path": "."},
                "id": f"explore-{index}"}])
            for index in range(2)
        ] + [AIMessage(content="完成")]
        fake = FakeLLM(responses)
        runner._llm = fake
        final = runner.run("需求", "t-no-edit")
        operations = [(op.tool, op.status) for op in store.all_operations()]
        store.close()

        assert final["stop_reason"] is None
        assert operations == [("list_files", "completed")] * 2
        third_request = "\n".join(str(message.content) for message in fake.seen[2])
        assert "长时间未修改提醒" in third_request
        assert "最小补丁" in third_request
        assert "必须修改代码" not in third_request

    def test_no_edit_reminder_resets_after_successful_edit(self, tmp_path):
        runner, context, store = _make_runner(tmp_path, no_edit_warn_limit=3)
        target = context.task_dir / "a.py"
        target.write_text("x = 1\n", encoding="utf-8")
        responses = [
            AIMessage(content="", tool_calls=[{
                "name": "list_files", "args": {"path": "."}, "id": f"explore-{i}"}])
            for i in range(2)
        ] + [
            AIMessage(content="", tool_calls=[{
                "name": "apply_patch",
                "args": {"path": "a.py", "old_text": "x = 1", "new_text": "x = 2",
                         "expected_hash": hash_content("x = 1\n")},
                "id": "edit-1"}]),
            AIMessage(content="", tool_calls=[{
                "name": "list_files", "args": {"path": "."}, "id": "explore-3"}]),
            AIMessage(content="", tool_calls=[{
                "name": "list_files", "args": {"path": "."}, "id": "explore-4"}]),
            AIMessage(content="完成"),
        ]
        fake = FakeLLM(responses)
        runner._llm = fake
        final = runner.run("需求", "t-no-edit-reset")
        store.close()

        assert final["stop_reason"] is None
        assert target.read_text(encoding="utf-8") == "x = 2\n"
        requests = "\n".join(
            str(message.content) for view in fake.seen for message in view)
        assert "长时间未修改提醒" not in requests

    def test_failed_edit_streak_adds_actionable_reminder(self, tmp_path):
        """连续失败的补丁仍会再次执行；提醒里带最后一次错误和重读建议。"""
        runner, context, store = _make_runner(tmp_path, edit_failure_warn_limit=2)
        target = context.task_dir / "a.py"
        target.write_text("def f():\n    return 1\n", encoding="utf-8")
        responses = [
            AIMessage(content="", tool_calls=[{
                "name": "apply_patch",
                "args": {"path": "a.py", "old_text": "return 2", "new_text": "return 3",
                         "expected_hash": hash_content(
                             "def f():\n    return 1\n")},
                "id": f"failed-edit-{index}"}])
            for index in range(2)
        ] + [AIMessage(content="完成")]
        fake = FakeLLM(responses)
        runner._llm = fake
        final = runner.run("需求", "t-edit-fail")
        operations = [(op.tool, op.status) for op in store.all_operations()]
        store.close()

        assert final["stop_reason"] is None
        assert operations == [("apply_patch", "failed")] * 2
        assert target.read_text(encoding="utf-8") == "def f():\n    return 1\n"
        third_request = "\n".join(str(message.content) for message in fake.seen[2])
        assert "[编辑失败提醒]" in third_request
        assert "没有匹配" in third_request
        assert "read_lines" in third_request


class TestPlanIntegration:
    def test_finish_with_open_plan_items_gets_one_reminder(self, tmp_path):
        """模型宣布完成但清单有未完成项：核对一次后放行，不无限拦截。"""
        runner, context, store = _make_runner(tmp_path)
        context.plan.replace(["复现失败", "修复根因"])
        fake = FakeLLM([
            AIMessage(content="", tool_calls=[{
                "name": "plan_update",
                "args": {"statuses": {"t1": "completed"}}, "id": "plan-upd"}]),
            AIMessage(content="完成"),
            AIMessage(content="最终总结"),
        ])
        runner._llm = fake
        final = runner.run("需求", "t-plan")
        store.close()

        assert final["stop_reason"] is None
        third_request = "\n".join(str(m.content) for m in fake.seen[2])
        assert "清单核对" in third_request
        assert final["plan_finish_warned"] is True
        assert len(fake.seen) == 3

    def test_finish_with_closed_plan_does_not_remind(self, tmp_path):
        runner, context, store = _make_runner(tmp_path)
        context.plan.replace(["复现失败"])
        context.plan.update({"t1": "completed"})
        fake = FakeLLM([AIMessage(content="完成")])
        runner._llm = fake
        final = runner.run("需求", "t-plan-closed")
        store.close()
        assert final["stop_reason"] is None
        assert len(fake.seen) == 1

    def test_plan_snapshot_and_skills_injected_into_stable_prefix_and_tail(self, tmp_path):
        skills_root = tmp_path / "extra-skills"
        skill_dir = skills_root / "testing"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: testing\ndescription: 测试规范\n---\n正文", encoding="utf-8")
        runner, context, store = _make_runner(
            tmp_path, skills_roots=[str(skills_root)])
        context.plan.replace(["复现失败"])
        # 清单有未完成项：第一次"完成"会被核对提醒拦一次，第二次才收尾。
        fake = FakeLLM([AIMessage(content="完成"), AIMessage(content="完成")])
        runner._llm = fake
        runner.run("需求", "t-inject")
        store.close()

        first = fake.seen[0]
        system_messages = [m for m in first if isinstance(m, SystemMessage)]
        assert len(system_messages) == 1
        prefix = str(system_messages[0].content)
        assert prefix == runner._stable_system_prompt
        assert "[项目技能]" in prefix
        assert "testing" in prefix
        # plan 不进稳定前缀，只作为请求尾部的临时快照。
        assert "[任务清单 v1]" not in prefix
        tail = [m for m in first if isinstance(m, HumanMessage)]
        assert any("[任务清单 v1]" in str(m.content) for m in tail)

    def test_permission_denial_reaches_model_as_tool_result(self, tmp_path):
        runner, context, store = _make_runner(tmp_path, permission_mode="bypass")
        fake = FakeLLM([
            AIMessage(content="", tool_calls=[{
                "name": "run_command",
                "args": {"command": "sudo rm -rf /"}, "id": "danger-1"}]),
            AIMessage(content="完成"),
        ])
        runner._llm = fake
        final = runner.run("需求", "t-deny")
        tool_texts = [str(m.content) for m in final["messages"]
                      if isinstance(m, ToolMessage)]
        store.close()

        assert any("[权限拒绝]" in text for text in tool_texts)
        assert final["stop_reason"] is None


class TestPromptVariants:
    def test_default_runner_uses_benchmark_prompt(self, tmp_path):
        runner, context, store = _make_runner(tmp_path)
        fake = FakeLLM([AIMessage(content="完成")])
        runner._llm = fake
        runner.run("需求", "t-prompt-default")
        store.close()
        assert fake.seen[0][0].content == SYSTEM_PROMPT

    def test_interactive_prompt_used_when_selected(self, tmp_path):
        runner, context, store = _make_runner(tmp_path)
        interactive = AgentRunner(context.settings, context,
                                  system_prompt=SYSTEM_PROMPT_INTERACTIVE)
        fake = FakeLLM([AIMessage(content="分析结论")])
        interactive._llm = fake
        interactive.run("帮我分析，先不要改代码", "t-prompt-interactive")
        store.close()
        assert fake.seen[0][0].content == SYSTEM_PROMPT_INTERACTIVE

    def test_prompt_inherited_from_history_on_later_turns(self, tmp_path):
        """会话第一轮定 prompt，后续轮从检查点历史继承，不因新入口变卦。"""
        runner, context, store = _make_runner(tmp_path)
        first = AgentRunner(context.settings, context,
                            system_prompt=SYSTEM_PROMPT_INTERACTIVE)
        fake1 = FakeLLM([AIMessage(content="好")])
        first._llm = fake1
        first.run("第一问", "t-prompt-inherit")
        second = AgentRunner(context.settings, context)  # 默认 benchmark
        fake2 = FakeLLM([AIMessage(content="完成")])
        second._llm = fake2
        second.run("第二问", "t-prompt-inherit")
        store.close()
        system_messages = [m for m in fake2.seen[0] if isinstance(m, SystemMessage)]
        assert len(system_messages) == 1
        assert system_messages[0].content == SYSTEM_PROMPT_INTERACTIVE

    def test_variants_share_tool_discipline_rules(self):
        """两版只有交付物定义与收尾措辞不同；工具纪律、反空转、权限必须一致。"""
        anchors = (
            "修改前必须先用 read_lines",
            "不要用 `2>/dev/null` 隐藏错误",
            "不要用 shell 绕过文件状态检查",
            "命令输出很长时用 read_log",
            "避免无效重复",
            "只在任务目录内操作",
            "危险或不可逆的命令",
            "plan_create 建立任务清单",
            "用 load_skill",
        )
        for anchor in anchors:
            assert anchor in SYSTEM_PROMPT, anchor
            assert anchor in SYSTEM_PROMPT_INTERACTIVE, anchor

    def test_benchmark_bias_only_in_benchmark_prompt(self):
        assert "你的最终交付物是源代码修改" in SYSTEM_PROMPT
        assert "你的最终交付物是源代码修改" not in SYSTEM_PROMPT_INTERACTIVE
        assert "用户的明确要求优先于本提示的通用流程" in SYSTEM_PROMPT_INTERACTIVE
        assert "用户的明确要求优先于" not in SYSTEM_PROMPT
