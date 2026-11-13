"""运行循环：模型决定下一步工具，运行程序负责执行、预算与状态保存。

LangGraph 负责调度与 SQLite 检查点；工具执行、预算、操作记录由本项目实现。
"""

import json
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from nextcoder.llm import build_chat_llm
from nextcoder.runtime.context import (
    CompactionEvent,
    SummaryRequest,
    apply_summary,
    is_valid_coverage,
    prepare_context,
    preserves_latest_user,
)
from nextcoder.runtime.events import AssistantDelta, Compaction, emit
from nextcoder.runtime.hooks import build_hook_bus
from nextcoder.runtime.mcp import McpManager
from nextcoder.runtime.progress import (
    LoopLimits,
    calls_signature,
    inspect_loop,
    matches_repeated_call,
)
from nextcoder.runtime.skills import catalog_text
from nextcoder.sessions.recovery import RecoveryNote, recover_started_operations
from nextcoder.sessions.summary import ModelSummarizer, SummaryRecord, SummaryStore
from nextcoder.settings import Settings
from nextcoder.tools.base import ToolContext
from nextcoder.tools.registry import ToolBox

IDENTITY_PROMPT = """你是 NextCoder，一个运行在终端里的编程 agent。NextCoder 是当前程序和产品的名字；
任务目录只是你这次工作的仓库，不是你的身份。不要把任务目录里的示例应用、README
或 AGENTS.md 描述的产品当成你自己（例如不要把 shop-cart demo 说成“我正在开发的项目”）。
当用户问“你是谁 / 你能做什么”时，介绍 NextCoder 的能力：读写与搜索代码、执行命令和
测试、维护任务清单、委派子 agent、调用 MCP 工具、使用 hooks、切换权限模式、使用
bubblewrap 沙箱，以及通过会话检查点和崩溃恢复继续工作。只有在与当前任务直接相关时
才讨论任务目录里的业务项目。"""

SYSTEM_PROMPT = IDENTITY_PROMPT + """
你是一个在任务目录中工作的编程助手。你的最终交付物是源代码修改
（补丁），不是分析报告。规则：
1. 先定位：用 search/read_lines 找到相关文件和失败路径。修改前必须先用 read_lines
   读取目标文件；结果中的 content_hash 就是该文件当前状态的哈希。
2. 复现：用最小命令确认问题。不要用 `2>/dev/null` 隐藏错误；管道命令的退出码来自
   最后一个命令，诊断时加 `set -o pipefail`，否则前面的失败会被吞掉。
3. 修改：一旦找到可能的根因，立即用 apply_patch 做最小修改；不要无限期继续读代码。
   小范围修改优先用 apply_patch：提供 old_text、new_text 和 read_lines 返回的
   content_hash。只有新建文件或整体重写时才用 write_file；已有文件必须附 content_hash。
4. 如果 apply_patch 或 write_file 被拒绝，说明文件内容已变化，必须重新读取后再改，
   不要用 shell 绕过文件状态检查。
5. 验证：修改后运行必要的检查（如测试），确认行为正确。命令输出很长时用 read_log
   回读，不要凭记忆猜测；也不要用逐行 sed 试探文件内容。
6. 完成后用 view_diff 确认改动，然后用一段话总结：做了什么修改、跑了哪些检查、
   哪些事项没有完成或未验证。
7. 避免无效重复：如果连续多次发起相同调用，先检查是否在原地打转；可以换一种查找
   方式、读取具体文件、开始修改或运行验证。环境可能已经变化（例如刚安装依赖），
   必要时重新执行同一命令获取新结果。命令连续无输出时，先检查命令是否失败，
   不要继续逐行试探。
8. 只在任务目录内操作；不要修改与需求无关的文件。用户消息里的 `@path` 表示
   引用该文件；需要内容时用 read_lines 读取，不要凭空猜测。
9. 危险或不可逆的命令（删除根目录、格式化磁盘、提权等）会被权限层直接拒绝；
   被拒绝时不要原样重试，按提示改用范围明确、可验证的命令。
10. 需求复杂时先用 plan_create 建立任务清单，用 plan_update 随进展更新状态；
    宣布完成前核对清单没有未完成项。系统提示列出项目技能时，用 load_skill
    读取完整指导后再动手。"""

# 交互入口（chat）用：交付物由用户当轮要求决定，用户只要分析/方案时不动代码。
# 工具纪律、反空转、权限等规则与 benchmark 版保持一致；改动共用规则时两份同步。
SYSTEM_PROMPT_INTERACTIVE = IDENTITY_PROMPT + """
你是一个在任务目录中工作的编程助手。按用户当轮的明确要求交付：用户只要
分析、方案、评审或解释时，交付的就是分析本身，不要修改代码；用户要求实现
或修复时，交付物是源代码修改（补丁），并推进到验证完成，不要停在分析。
用户的明确要求优先于本提示的通用流程。规则：
1. 先定位：用 search/read_lines 找到相关文件和失败路径。修改前必须先用 read_lines
   读取目标文件；结果中的 content_hash 就是该文件当前状态的哈希。
2. 复现：用最小命令确认问题。不要用 `2>/dev/null` 隐藏错误；管道命令的退出码来自
   最后一个命令，诊断时加 `set -o pipefail`，否则前面的失败会被吞掉。
3. 修改：用户要求修改时，一旦找到可能的根因，立即用 apply_patch 做最小修改；
   不要无限期继续读代码。小范围修改优先用 apply_patch：提供 old_text、new_text
   和 read_lines 返回的 content_hash。只有新建文件或整体重写时才用 write_file；
   已有文件必须附 content_hash。
4. 如果 apply_patch 或 write_file 被拒绝，说明文件内容已变化，必须重新读取后再改，
   不要用 shell 绕过文件状态检查。
5. 验证：修改后运行必要的检查（如测试），确认行为正确。命令输出很长时用 read_log
   回读，不要凭记忆猜测；也不要用逐行 sed 试探文件内容。
6. 收尾：有代码改动时用 view_diff 确认改动，然后用一段话总结：做了什么修改、
   跑了哪些检查、哪些事项没有完成或未验证；只做分析时说明结论、依据和建议
   的下一步。
7. 避免无效重复：如果连续多次发起相同调用，先检查是否在原地打转；可以换一种查找
   方式、读取具体文件、开始修改或运行验证。环境可能已经变化（例如刚安装依赖），
   必要时重新执行同一命令获取新结果。命令连续无输出时，先检查命令是否失败，
   不要继续逐行试探。
8. 只在任务目录内操作；不要修改与需求无关的文件。用户消息里的 `@path` 表示
   引用该文件；需要内容时用 read_lines 读取，不要凭空猜测。
9. 危险或不可逆的命令（删除根目录、格式化磁盘、提权等）会被权限层直接拒绝；
   被拒绝时不要原样重试，按提示改用范围明确、可验证的命令。
10. 需求复杂时先用 plan_create 建立任务清单，用 plan_update 随进展更新状态；
    宣布完成前核对清单没有未完成项。系统提示列出项目技能时，用 load_skill
    读取完整指导后再动手。"""

# 输出截断后的重试提示：丢弃被截断的响应，要求模型拆小步骤再来一次。
LENGTH_RETRY_REMINDER = """[输出长度提醒] 上一次输出因长度限制被截断，其中的工具调用没有执行。请减少解释，把工作拆小：先完成一个最小修改（例如一个 apply_patch 或一个短命令），再继续下一步；需要给出大段代码时，分多次 apply_patch。现在直接继续任务。"""
# 强制重新规划后如果只回复分析、不调用工具，运行会误判为完成；补一轮提醒。
REPLAN_CONTINUE_REMINDER = """[继续] 重新规划后不要停在分析。分析完成后必须继续调用工具（read_lines、run_command 或 view_diff）验证或推进任务；现在直接继续。"""
# 模型宣布完成但清单仍有未完成项时核对一次；只提醒一次，不强制继续执行。
PLAN_FINISH_REMINDER = """[清单核对] 你宣布了任务完成，但任务清单仍有未完成项。请用 plan_update 把已完成的事项标记为 completed、不再需要的标记为 cancelled，或继续完成剩余项；核对后再次给出交付总结。"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    model_calls: int
    usage: dict[str, int]
    # 最近一次请求的完整输入估算（系统提示 + 工具定义 + 历史 + 工具结果）。
    request_tokens: int
    # 最近一次请求发出时检查点里的消息总数；下一请求只把其后的消息算作新增，
    # 避免用检查点全量历史反复触发第一层折叠。
    request_full_count: int
    stop_reason: str | None
    # 输出截断重试：retry_model 为 True 时路由回模型节点，不执行被截断的工具调用。
    retry_model: bool
    length_retries: int
    # 连续未成功修改文件的工具调用计数；只用于一次性提醒，不阻断。
    no_edit_calls: int
    no_edit_warned: bool
    # 连续失败的编辑次数与最后一次错误；同样只用于提醒。
    edit_failures: int
    edit_warned: bool
    last_edit_error: str
    # 进展检测：最后一次强制重新规划的位置，以及被阻止编辑的文件。
    loop_replan_key: str
    loop_replan_at: int
    loop_replan_path: str
    loop_replan_signature: str
    loop_replan_evidence: tuple[int, int]
    loop_block_until: int
    loop_gap_nudged: bool
    loop_gap_test_count: int
    loop_replan_retry: int
    budget_warned: bool
    # 模型宣布完成但清单仍有未完成项时，只核对一次。
    plan_finish_warned: bool


class AgentRunner:
    def __init__(self, settings: Settings, context: ToolContext,
                 session_id: str = "", system_prompt: str | None = None,
                 stable_system_prompt: str | None = None,
                 streaming: bool = False) -> None:
        self._settings = settings
        self._context = context
        self._streaming = streaming
        # None = benchmark 版（run/eval 一致口径）；chat 传交互版。
        # 只在会话第一轮生效，后续轮从检查点历史继承同一系统消息。
        self._system_prompt = system_prompt or SYSTEM_PROMPT
        # 稳定前缀在 AgentRunner 创建时组装一次：base 规则 + AGENTS.md +
        # 记忆快照 + 技能目录。plan 不在这里，改为请求尾部的临时快照。
        # 子 agent 直接复用父会话已经冻结的前缀，再追加角色说明。
        self._stable_system_prompt = (
            stable_system_prompt if stable_system_prompt is not None
            else self._build_stable_system_prompt())
        # MCP：配置文件存在才启动子进程并注册工具；否则零开销。
        self._mcp: McpManager | None = None
        if settings.mcp_enabled:
            manager = McpManager.from_task(settings, context.task_dir)
            if manager is not None:
                manager.connect_all()
                self._mcp = manager
        self.mcp_errors = list(self._mcp.errors) if self._mcp is not None else []
        # 反重复/搜索/空输出等内置提醒与用户 hooks.json 共用同一总线。
        self._hooks = build_hook_bus(
            settings, context.task_dir, turn_id=context.turn_id,
            event_handler=context.event_handler, rollout=context.rollout)
        self._toolbox = ToolBox(context, mcp=self._mcp, hooks=self._hooks)
        definitions = self._toolbox.definitions()
        self._tool_schema_text = _tool_schema_text(definitions)
        # 会话头必须跨轮稳定：manager 传入 session_id；单次运行没有会话时用固定值。
        self._session_id = session_id
        self._llm = build_chat_llm(
            settings, session_id=session_id).bind_tools(definitions)
        self._started = time.monotonic()
        self.stop_reason: str | None = None
        self.usage_delta: dict[str, int] = {}
        # 每次主循环模型调用的归一化 usage；用于区分首次冷启动与后续暖命中。
        self.call_usages: list[dict[str, int]] = []
        # 异常中止时仍保留最近一次已写入检查点的状态与请求估算，供报告使用。
        self.last_state: dict[str, Any] | None = None
        self.last_request_tokens = 0
        self.recovery_notes: list[RecoveryNote] = []
        self.compaction_events: list[CompactionEvent] = []
        self.memory_used: list[str] = []
        self._summary_store = SummaryStore(context.summary_path)
        self._summarizer = ModelSummarizer(settings) if settings.summary_enabled else None
        self._summary_record: SummaryRecord | None = None
        self._thread_id = ""
        self.summary_usage: dict[str, int] = {}
        self.summary_calls = 0
        self.summary_latency_seconds = 0.0
        self.summary_errors = []
        # 搜索计数由 SearchBudgetHook 维护；这里只保留 hook 总线。
        self.last_endpoint = "primary"
        self._loop_limits = LoopLimits(
            noop_replan_limit=settings.loop_noop_replan_limit,
            rollback_replan_count=settings.loop_rollback_replan_count,
            rollback_stop_count=settings.loop_rollback_stop_count,
            edit_gap_nudge=settings.loop_edit_gap_nudge,
            block_calls=settings.loop_block_calls,
            stop_after_calls=settings.loop_stop_after_calls,
        )

    def run(self, request: str, thread_id: str) -> dict[str, Any]:
        # 检查点必须传给 compile() 才会真正写入：只打开 saver 不生效（v0.1 曾错）。
        self.compaction_events = []
        self.memory_used = []
        self.summary_usage = {}
        self.summary_calls = 0
        self.summary_latency_seconds = 0.0
        self.summary_errors = []
        self.last_endpoint = "primary"
        self.last_state = None
        self.last_request_tokens = 0
        self._thread_id = thread_id
        config = {"configurable": {"thread_id": thread_id}}
        with SqliteSaver.from_conn_string(str(self._context.checkpoints_path)) as checkpointer:
            graph = self._build_graph(checkpointer)
            # 先核对上一次进程留下的未决操作，再补齐检查点里缺少结果的工具调用。
            self.recovery_notes = recover_started_operations(
                self._context.store, self._context.task_dir)
            if self._context.memory is not None and self._context.memory_snapshot is None:
                # 会话入口/进程恢复时加载一次；后续请求只复用快照。
                self._context.memory_snapshot = self._context.memory.snapshot()
            if self._context.memory_snapshot is not None:
                self.memory_used = list(self._context.memory_snapshot.keys)
            before = graph.get_state(config).values or {}
            self._recover_pending_tools(graph, config, before.get("messages") or [])
            before = graph.get_state(config).values or {}
            # hook 状态按轮重置；从既有消息恢复连续重复/空输出计数，
            # 这样跨轮对话仍能识别同一个探索循环。
            self._hooks.reset()
            self._hooks.seed(before.get("messages") or [])
            self._summary_record = self._load_summary(
                thread_id, before.get("messages") or [])
            prior_usage = before.get("usage") or {}
            if before.get("messages"):
                messages = [HumanMessage(content=request)]
            else:
                messages = [SystemMessage(content=self._stable_system_prompt),
                            HumanMessage(content=request)]
            try:
                final = graph.invoke(
                    {"messages": messages, "model_calls": 0, "request_tokens": 0,
                     "stop_reason": None, "retry_model": False, "length_retries": 0,
                     "no_edit_calls": 0, "no_edit_warned": False,
                     "edit_failures": 0, "edit_warned": False, "last_edit_error": "",
                     "loop_replan_key": "", "loop_replan_at": 0,
                     "loop_replan_path": "", "loop_replan_signature": "",
                     "loop_replan_evidence": (0, 0),
                     "loop_block_until": 0, "loop_gap_nudged": False,
                     "loop_gap_test_count": 0,
                     "loop_replan_retry": 0, "budget_warned": False,
                     "plan_finish_warned": False},
                    config,
                )
            except BaseException:
                # 包括 Ctrl-C 中断：已完成的模型调用和用量已在检查点里，
                # 报告不能因为异常把它们写成 0。
                self.last_state = self._checkpoint_state(graph, config)
                if self.last_state is not None:
                    failed_usage = self.last_state.get("usage") or {}
                    keys = set(failed_usage) | set(prior_usage)
                    self.usage_delta = {
                        key: failed_usage.get(key, 0) - prior_usage.get(key, 0)
                        for key in keys
                    }
                raise
            finally:
                self.close_mcp()
        self.last_state = final
        final_usage = final.get("usage") or {}
        keys = set(final_usage) | set(prior_usage)
        self.usage_delta = {k: final_usage.get(k, 0) - prior_usage.get(k, 0) for k in keys}
        return final

    @staticmethod
    def _checkpoint_state(graph, config) -> dict[str, Any] | None:
        try:
            return graph.get_state(config).values or None
        except Exception:  # noqa: BLE001 - 读取失败不能掩盖原始异常
            return None

    def close_mcp(self) -> None:
        """结束 MCP 子进程；幂等，run() 的 finally 与 compact() 后都会调用。"""
        if self._mcp is not None:
            self._mcp.close()
            self._mcp = None

    @property
    def mcp_tool_names(self) -> list[str]:
        return sorted(self._toolbox.mcp.tools) if self._toolbox.mcp is not None else []

    @property
    def hook_errors(self) -> list[str]:
        """hooks.json 配置错误或单个 hook 运行异常；不阻断主流程。"""
        return list(self._hooks.errors)

    def _recover_pending_tools(self, graph, config, messages: list) -> None:
        """修复崩溃留下的“有 tool_call、没有 ToolMessage”状态。

        进程可能在模型返回工具调用之后、工具结果写入检查点之前退出。此时直接
        追加新的用户消息会让接口收到不完整的工具调用序列。这里用同一个操作 ID
        执行/核对缺失的调用：已完成的复用缓存，状态不明的返回“不自动重放”。
        """
        if not messages:
            return
        last_call_index = -1
        for index, message in enumerate(messages):
            if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
                last_call_index = index
        if last_call_index < 0:
            return
        last = messages[last_call_index]
        answered = {
            getattr(item, "tool_call_id", None)
            for item in messages[last_call_index + 1:]
            if isinstance(item, ToolMessage)
        }
        pending = [call for call in last.tool_calls if call.get("id") not in answered]
        if not pending:
            return
        recovered: list[ToolMessage] = []
        for call in pending:
            call_id = call.get("id")
            if not call_id:
                continue
            text = self._toolbox.execute(
                call.get("name", ""), call.get("args") or {}, call_id=call_id)
            recovered.append(ToolMessage(content=text, tool_call_id=call_id))
        if recovered:
            # as_node="tools" 让检查点把补上的结果视为工具节点的输出，后续从模型继续。
            graph.update_state(config, {"messages": recovered}, as_node="tools")

    def _build_graph(self, checkpointer):
        builder = StateGraph(AgentState)
        builder.add_node("model", self._model_node)
        builder.add_node("tools", self._tools_node)
        builder.add_edge(START, "model")
        builder.add_conditional_edges(
            "model", self._route, {"tools": "tools", "end": END, "model": "model"})
        builder.add_edge("tools", "model")
        return builder.compile(checkpointer=checkpointer)

    def _budget_left(self, state: AgentState, request_tokens: int) -> str | None:
        if self._context.cancel_event.is_set():
            return "用户取消"
        if state["model_calls"] >= self._settings.model_calls_limit:
            return f"模型调用次数达到上限 {self._settings.model_calls_limit}"
        elapsed = time.monotonic() - self._started
        if elapsed > self._settings.wall_time_seconds:
            return f"运行时间超过 {self._settings.wall_time_seconds} 秒"
        budget = self._settings.application_input_token_budget
        if request_tokens + self._settings.output_reserve_tokens > budget:
            return (f"本次请求上下文估算 {request_tokens} tokens，另预留输出 "
                    f"{self._settings.output_reserve_tokens} tokens，超过预算 {budget}")
        return None

    def _build_stable_system_prompt(self) -> str:
        """组装一次稳定系统前缀；会话中途不再改写。"""
        parts = [self._system_prompt.strip()]
        agents = (self._context.agents_text or "").strip()
        if agents:
            parts.append("[项目指令 AGENTS.md]\n" + agents)
        snapshot = self._context.memory_snapshot
        if snapshot is not None and snapshot.text:
            parts.append(snapshot.text)
        skills_block = catalog_text(self._context.skills,
                                    self._settings.skills_prompt_chars)
        if skills_block:
            parts.append(skills_block)
        return "\n\n".join(part for part in parts if part)

    def _emit(self, event: object) -> None:
        emit(self._context.event_handler, event)

    def _emit_compaction(self, event: CompactionEvent) -> None:
        self._emit(Compaction(
            turn_id=self._context.turn_id,
            before_tokens=event.before_tokens,
            after_tokens=event.after_tokens,
            method=event.method,
            reason=event.reason,
        ))

    def _stream_model(self, view: list):
        """流式调用模型并合并 chunk；约 200ms 合并一次 delta 给展示层。"""
        full = None
        buffer: list[str] = []
        last_emit = time.monotonic()
        for chunk in self._llm.stream(view):
            full = chunk if full is None else full + chunk
            text = _chunk_text(chunk)
            if text:
                buffer.append(text)
                now = time.monotonic()
                if now - last_emit >= 0.2:
                    self._emit(AssistantDelta(self._context.turn_id,
                                              "".join(buffer)))
                    buffer.clear()
                    last_emit = now
        if buffer:
            self._emit(AssistantDelta(self._context.turn_id, "".join(buffer)))
        if full is None:
            raise RuntimeError("模型流没有返回任何内容")
        return full

    def _model_node(self, state: AgentState) -> dict:
        messages = state["messages"]
        snapshot = self._context.memory_snapshot
        self.memory_used = list(snapshot.keys) if snapshot is not None else []
        # 稳定前缀（base 规则、AGENTS.md、记忆快照、技能目录）在会话开始时
        # 已冻结；这里不再改写第一条 SystemMessage。plan 快照在压缩后作为
        # 临时尾部消息追加，避免 plan 更新破坏前缀缓存。

        prepared = prepare_context(
            messages, self._settings, self._estimate_request_tokens,
            summary_text=self._summary_record.summary if self._summary_record else None,
            covered_count=self._summary_record.covered_count if self._summary_record else 0,
            prior_request_tokens=state.get("request_tokens", 0),
            prior_message_count=state.get("request_full_count", 0),
        )
        view = prepared.view
        merged = dict(state.get("usage", {}))
        summary_used = False
        if prepared.summary_request is not None:
            compacted_view = self._summary_chain(
                prepared.summary_request, messages, merged)
            if compacted_view is not None:
                view = compacted_view
                summary_used = True
        if not summary_used and prepared.event.triggered:
            # 摘要失败或无需摘要时保留回退视图事件，方便报告解释实际发送内容。
            self.compaction_events.append(prepared.event)
            self._emit_compaction(prepared.event)

        # 内置反重复/搜索/空输出提醒与声明式 hook 都排队在这里；
        # 每轮模型请求前取走，避免把同一提醒重复写进历史。
        reminders: list[str] = self._hooks.collect_reminders()

        # 编辑失败与“只探索不落盘”都只提醒一次：重复追加同类提醒本身会
        # 消耗上下文，而模型更需要的是具体错误和下一步动作。
        warned: dict[str, bool] = {}
        edit_failures = int(state.get("edit_failures", 0))
        if (self._settings.edit_failure_warn_limit > 0
                and edit_failures >= self._settings.edit_failure_warn_limit
                and not state.get("edit_warned")):
            detail = _tool_error_text(str(state.get("last_edit_error") or ""))
            reminders.append(
                f"[编辑失败提醒] 最近连续 {edit_failures} 次 apply_patch/write_file "
                f"失败。{('最后一次错误：' + detail + '。') if detail else ''}"
                "不要重复提交同一补丁；先 read_lines 读取当前文件，核对 old_text "
                "的准确原文和 content_hash，或把修改拆得更小。")
            warned["edit_warned"] = True
        no_edit_calls = int(state.get("no_edit_calls", 0))
        if (self._settings.no_edit_warn_limit > 0
                and no_edit_calls >= self._settings.no_edit_warn_limit
                and not state.get("no_edit_warned")):
            reminders.append(
                f"[长时间未修改提醒] 最近 {no_edit_calls} 次工具调用没有成功修改"
                "任何文件。如果已经定位到根因，可以提交最小补丁并运行验证；"
                "如果还在定位，请说明还缺什么信息，换一种更直接的查找方式。")
            warned["no_edit_warned"] = True
        if (self._settings.budget_warn_ratio > 0
                and not state.get("budget_warned")):
            threshold = max(
                1,
                int(self._settings.model_calls_limit
                    * self._settings.budget_warn_ratio),
            )
            if int(state["model_calls"]) >= threshold:
                reminders.append(
                    f"[预算提醒] 已使用 {state['model_calls']}/"
                    f"{self._settings.model_calls_limit} 次模型调用。"
                    "停止扩展探索：优先运行目标测试确认当前方向；"
                    "如果已经定位根因，直接提交最小补丁并用 view_diff 检查。"
                    "剩余调用不足以继续大范围尝试。")
                warned["budget_warned"] = True
        if reminders:
            view = [*view, HumanMessage(content="\n\n".join(reminders))]

        loop = inspect_loop(
            messages,
            model_calls=int(state["model_calls"]),
            state=state,
            limits=self._loop_limits,
        )
        loop_updates = {
            "loop_replan_key": loop.replan_key,
            "loop_replan_at": loop.replan_at,
            "loop_replan_path": loop.replan_path,
            "loop_replan_signature": loop.replan_signature,
            "loop_replan_evidence": loop.replan_evidence,
            "loop_block_until": loop.block_until,
            "loop_gap_nudged": loop.gap_nudged,
            "loop_gap_test_count": loop.gap_test_count,
        }
        previous_key = str(state.get("loop_replan_key") or "")
        replan_retry = int(state.get("loop_replan_retry", 0))
        if loop.kind == "replan" or loop.replan_key != previous_key:
            replan_retry = 0
        loop_updates["loop_replan_retry"] = replan_retry
        if loop.end_run:
            self.stop_reason = loop.message
            return {"messages": [AIMessage(content=loop.message)],
                    "stop_reason": loop.message,
                    "model_calls": state["model_calls"],
                    "request_tokens": state.get("request_tokens", 0),
                    "request_full_count": len(state["messages"]),
                    "retry_model": False, **warned, **loop_updates}
        if loop.message:
            view = [*view, HumanMessage(content=loop.message)]

        # plan 快照只挂在本轮请求尾部：plan 更新不会改稳定前缀，也不会写进
        # 检查点。压缩发生在它之前，因此快照不会被摘要吃掉。
        plan_block = (self._context.plan.snapshot_text()
                      if self._context.plan is not None else "")
        if plan_block:
            view = [*view, HumanMessage(content=(
                "[运行时注入的任务清单快照，不是用户的新请求]\n" + plan_block))]

        # 后台子 agent 的完成/失败通知只注入一次；同样不写检查点。
        manager = self._context.delegation_manager
        if manager is not None and self._context.allow_delegation:
            notifications = manager.poll_notifications()
            if notifications:
                view = [*view, HumanMessage(content=(
                    "[运行时注入的子 agent 通知，不是用户的新请求]\n"
                    + "\n\n".join(notifications)))]

        history_count = len(state["messages"])
        request_tokens = self._estimate_request_tokens(view)
        self.last_request_tokens = request_tokens
        reason = self._budget_left(state, request_tokens)
        if reason is not None:
            self.stop_reason = reason
            return {"stop_reason": reason, "request_tokens": request_tokens,
                    "request_full_count": history_count,
                    "retry_model": False, **warned, **loop_updates,
                    "messages": [AIMessage(content=f"[运行停止：{reason}]")]}
        if self._streaming and self._context.event_handler is not None:
            response = self._stream_model(view)
        else:
            response = self._llm.invoke(view)
        self.last_endpoint = getattr(self._llm, "last_endpoint", "primary")
        call_usage = _normalize_usage(getattr(response, "usage_metadata", None))
        if call_usage:
            self.call_usages.append(call_usage)
        _add_usage(merged, call_usage)
        if self._context.rollout is not None:
            self._context.rollout.append("assistant", {
                "content": _content_excerpt(response),
                "tool_calls": [
                    {"id": call.get("id"), "name": call.get("name"),
                     "args": _args_excerpt(call.get("args"))}
                    for call in (getattr(response, "tool_calls", None) or [])
                ],
            }, turn_id=self._context.turn_id)
        if _is_length_truncated(response):
            retries = int(state.get("length_retries", 0))
            if (self._settings.length_retry_limit > 0
                    and retries < self._settings.length_retry_limit):
                # 被截断的响应不写入历史：它可能带有未执行的 tool_calls，
                # 直接追加会让下一次接口调用出现“工具调用没有结果”的非法序列。
                # 只追加提醒，让模型用更小的输出重新完成这一步。
                self.stop_reason = None
                return {"messages": [HumanMessage(content=LENGTH_RETRY_REMINDER)],
                        "model_calls": state["model_calls"] + 1,
                        "usage": merged, "request_tokens": request_tokens,
                        "request_full_count": history_count,
                        "stop_reason": None, "retry_model": True,
                        "length_retries": retries + 1,
                        **warned, **loop_updates}
            reason = (f"模型输出连续 {retries + 1} 次被截断"
                      f"（finish_reason=length）")
            self.stop_reason = reason
            return {"messages": [AIMessage(content=f"[运行停止：{reason}]")],
                    "model_calls": state["model_calls"] + 1,
                    "usage": merged, "request_tokens": request_tokens,
                    "request_full_count": history_count,
                    "stop_reason": reason, "retry_model": False,
                    "length_retries": retries + 1,
                    **warned, **loop_updates}
        if (loop.kind == "replan"
                and not getattr(response, "tool_calls", None)
                and _response_stop_reason(response) is None
                and int(state.get("loop_replan_retry", 0)) < 1):
            # 重新规划后模型可能只回复分析；不能把它当成任务完成而结束。
            return {"messages": [response, HumanMessage(content=REPLAN_CONTINUE_REMINDER)],
                    "model_calls": state["model_calls"] + 1,
                    "usage": merged, "request_tokens": request_tokens,
                    "request_full_count": history_count,
                    "stop_reason": None, "retry_model": True,
                    "length_retries": 0,
                    **warned,
                    **{**loop_updates, "loop_replan_retry": 1}}
        stop_reason = _response_stop_reason(response)
        if (stop_reason is None and not getattr(response, "tool_calls", None)
                and self._context.plan is not None
                and not state.get("plan_finish_warned", False)
                and self._context.plan.has_open_items()):
            # 模型宣布完成但清单有未完成项：核对一次（提醒后不再拦）。
            return {"messages": [response, HumanMessage(content=PLAN_FINISH_REMINDER)],
                    "model_calls": state["model_calls"] + 1,
                    "usage": merged, "request_tokens": request_tokens,
                    "request_full_count": history_count,
                    "stop_reason": None, "retry_model": True,
                    "length_retries": 0, "plan_finish_warned": True,
                    **warned, **loop_updates}
        if stop_reason is not None:
            self.stop_reason = stop_reason
        return {"messages": [response], "model_calls": state["model_calls"] + 1,
                "usage": merged, "request_tokens": request_tokens,
                "request_full_count": history_count,
                "stop_reason": stop_reason, "retry_model": False,
                "length_retries": 0, **warned, **loop_updates}

    def compact(self, thread_id: str) -> dict:
        """主动压缩：与自动压缩共用摘要生成、保存、记忆整理流程。

        只读取检查点，不追加用户消息，也不执行工具。当前工具调用必须已经
        结束（检查点里没有未配对调用），否则拒绝压缩，避免切断调用关系。
        """
        self._thread_id = thread_id
        config = {"configurable": {"thread_id": thread_id}}
        with SqliteSaver.from_conn_string(
                str(self._context.checkpoints_path)) as checkpointer:
            graph = self._build_graph(checkpointer)
            messages = (graph.get_state(config).values or {}).get("messages") or []
        if not messages:
            return {"compacted": False, "reason": "会话还没有可压缩的历史"}
        if _has_pending_tool_call(messages):
            return {"compacted": False,
                    "reason": "最近一次工具调用尚未结束，不能主动压缩"}

        self._summary_record = self._load_summary(thread_id, messages)
        prepared = prepare_context(
            messages, self._settings, self._estimate_request_tokens,
            summary_text=self._summary_record.summary if self._summary_record else None,
            covered_count=self._summary_record.covered_count
            if self._summary_record else 0,
            force=True,
        )
        if prepared.summary_request is None:
            return {"compacted": False, "reason": prepared.event.reason}
        request = prepared.summary_request
        merged: dict[str, int] = {}
        if not self._try_summary(request, merged):
            return {"compacted": False, "reason": "摘要生成或保存失败",
                    "errors": list(self.summary_errors)}
        assert self._summary_record is not None
        view, event = apply_summary(
            messages, self._settings, self._estimate_request_tokens,
            self._summary_record.summary, request,
            summary_id=self._summary_record.summary_id,
            summary_chars=len(self._summary_record.summary),
        )
        return {
            "compacted": True,
            "summary_id": self._summary_record.summary_id,
            "covered_count": self._summary_record.covered_count,
            "covered_groups": event.covered_groups,
            "kept_groups": event.kept_groups,
            "summary": self._summary_record.summary,
            "usage": dict(self.summary_usage),
            "before_tokens": event.before_tokens,
            "after_tokens": event.after_tokens,
            "remaining_messages": len(view),
            "memory_used": list(self.memory_used),
        }

    def _load_summary(self, thread_id: str, messages: list) -> SummaryRecord | None:
        record = self._summary_store.latest(thread_id)
        if record is None:
            return None
        if (record.covered_count <= 0 or record.covered_count > len(messages)
                or not is_valid_coverage(messages, record.covered_count)
                or not preserves_latest_user(messages, record.covered_count)):
            self.summary_errors.append(
                f"忽略不匹配的摘要 {record.summary_id}：覆盖 {record.covered_count}，"
                f"当前消息 {len(messages)}")
            return None
        return record

    def _summary_chain(self, request: SummaryRequest, messages: list,
                       merged: dict[str, int]) -> list | None:
        """生成摘要；仍高于触发线时最多重试一次，并检查是否真的收敛。

        第一次摘要保留可用输入的 16%；如果结果仍高于 90% 触发线，第二次用
        8% 的近期预算覆盖更多旧内容。第二次没有更小就停止，不无限重试。
        返回值是最终请求视图；None 表示没有可用的摘要视图。
        """
        limit = max(0, int(self._settings.compaction_summary_retry_limit))
        attempt = 0
        last_view: list | None = None
        previous_tokens = 0
        while True:
            attempt += 1
            if not self._try_summary(request, merged):
                return last_view
            assert self._summary_record is not None
            try:
                view, event = apply_summary(
                    messages, self._settings, self._estimate_request_tokens,
                    self._summary_record.summary, request,
                    summary_id=self._summary_record.summary_id,
                    summary_chars=len(self._summary_record.summary),
                )
            except Exception as exc:  # noqa: BLE001
                self.summary_errors.append(
                    f"摘要视图组装失败：{type(exc).__name__}: {exc}")
                return last_view
            event.summary_attempts = attempt
            if attempt > 1:
                event.retry_reason = (
                    f"第 {attempt} 次摘要：{previous_tokens} -> "
                    f"{event.after_tokens} tokens")
            self.compaction_events.append(event)
            self._emit_compaction(event)
            last_view = view

            budget = (self._settings.application_input_token_budget
                      - self._settings.output_reserve_tokens)
            trigger = max(1, int(
                budget * self._settings.compaction_trigger_ratio))
            if event.after_tokens <= trigger:
                return view
            if attempt > 1 and event.after_tokens >= previous_tokens:
                event.reason += (
                    f"；摘要重试未收敛（{previous_tokens} -> "
                    f"{event.after_tokens}），停止重试")
                return view
            if attempt > limit:
                event.reason += (
                    f"；已重试 {attempt - 1} 次仍未低于触发线 "
                    f"{trigger}，按总预算继续")
                return view

            previous_tokens = event.after_tokens
            retry = prepare_context(
                messages, self._settings, self._estimate_request_tokens,
                summary_text=self._summary_record.summary,
                covered_count=self._summary_record.covered_count,
                force=True,
                recent_ratio=self._settings.compaction_retry_recent_ratio,
            )
            if retry.summary_request is None:
                event.reason += "；没有可继续摘要的范围，停止重试"
                return view
            request = retry.summary_request

    def _try_summary(self, request: SummaryRequest,
                     merged: dict[str, int]) -> bool:
        if self._summarizer is None:
            self.summary_errors.append("摘要功能已关闭，保留回退视图")
            return False
        started = time.monotonic()
        self.summary_calls += 1
        try:
            output = self._summarizer.summarize(
                request.old_summary, request.messages, task=request.task)
            record = self._summary_store.save(
                self._thread_id or "unknown",
                request.cutoff,
                output.summary,
                model=self._settings.llm_model,
                usage=output.usage,
                parent_id=self._summary_record.summary_id if self._summary_record else None,
            )
        except Exception as exc:  # noqa: BLE001 - 摘要失败不应丢掉旧状态
            self.summary_errors.append(f"{type(exc).__name__}: {exc}")
            return False
        finally:
            self.summary_latency_seconds += time.monotonic() - started
        self._summary_record = record
        _add_usage(merged, output.usage)
        _add_usage(self.summary_usage, output.usage)
        self._consolidate_memory(record, output, request)
        return True

    def _consolidate_memory(self, record: SummaryRecord, output, request) -> None:
        """记忆整理与摘要分开保存；记忆失败不阻止摘要生效。"""
        if self._context.memory is None:
            return
        try:
            self._context.memory.consolidate(
                summary_id=record.summary_id,
                covered_count=record.covered_count,
                suggestions=output.memory_suggestions,
            )
        except Exception as exc:  # noqa: BLE001
            self.summary_errors.append(f"记忆整理失败：{type(exc).__name__}: {exc}")

    def _tools_node(self, state: AgentState) -> dict:
        messages: list[ToolMessage] = []
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", None) or [])
        # 一组并行调用共用一个签名；重复检测按“整组”计数，而不是组内每个调用。
        group_signature = calls_signature(calls)
        no_edit_calls = int(state.get("no_edit_calls", 0))
        edit_failures = int(state.get("edit_failures", 0))
        last_edit_error = str(state.get("last_edit_error") or "")
        edit_succeeded = False
        for index, call in enumerate(calls):
            if self._context.cancel_event.is_set():
                # 保持 tool_call 与 ToolMessage 成对，避免恢复时出现孤立调用。
                messages.extend(
                    ToolMessage(content="[未执行：用户取消]", tool_call_id=pending["id"])
                    for pending in calls[index:]
                )
                self.stop_reason = "用户取消"
                self._emit_tool_events(calls, messages)
                return {"messages": messages, "stop_reason": self.stop_reason}
            # 每次都执行新的调用。即使参数与之前完全相同，环境也可能已经
            # 变化（装依赖、改配置、生成文件），不能把旧结果当成新结果返回。
            # 崩溃恢复的幂等由 ToolBox.execute 按 call_id 处理，不在这里做缓存。
            is_edit = call.get("name") in {"apply_patch", "write_file"}
            blocked_path = str(state.get("loop_replan_path") or "")
            blocked_signature = str(state.get("loop_replan_signature") or "")
            block_until = int(state.get("loop_block_until") or 0)
            if (blocked_signature
                    and matches_repeated_call(call, blocked_signature)
                    and int(state.get("model_calls") or 0) <= block_until):
                # 完全相同的调用已连续多次；重新规划后先拒绝一次，迫使模型换动作。
                text = (
                    "[调用已阻止] 检测到连续重复同一调用并要求重新规划："
                    "这次完全相同的调用不会执行。请按干预要求说明原因，"
                    "并换用 read_lines、run_command 或 apply_patch。"
                )
                no_edit_calls += 1
                messages.append(ToolMessage(content=text, tool_call_id=call["id"]))
                continue
            if (is_edit and blocked_path
                    and str((call.get("args") or {}).get("path") or "") == blocked_path
                    and int(state.get("model_calls") or 0) <= block_until):
                # 重新规划后短暂拒绝同一文件的编辑，防止“说换方案、实际仍改原处”。
                text = (
                    "[编辑已阻止] 检测到循环后要求先重新规划："
                    f"{blocked_path} 暂时不能修改。先说明失败原因、已尝试的"
                    "方案和新的修改路径，或运行测试获取新证据。"
                )
                no_edit_calls += 1
                messages.append(ToolMessage(content=text, tool_call_id=call["id"]))
                continue
            text = self._toolbox.execute(
                call["name"], call["args"], call_id=call.get("id"),
                group_signature=group_signature, group_index=index,
                group_size=len(calls))
            if is_edit and "status=completed" in text:
                no_edit_calls = 0
                edit_failures = 0
                last_edit_error = ""
                edit_succeeded = True
            else:
                no_edit_calls += 1
                if is_edit:
                    edit_failures += 1
                    last_edit_error = _tool_error_text(text)
            messages.append(ToolMessage(content=text, tool_call_id=call["id"]))
        self._emit_tool_events(calls, messages)
        updates = {
            "messages": messages,
            "no_edit_calls": no_edit_calls,
            "edit_failures": edit_failures,
            "last_edit_error": last_edit_error,
        }
        if edit_succeeded:
            # 成功修改后重新武装两类提醒，供下一次长时间探索使用。
            updates["no_edit_warned"] = False
            updates["edit_warned"] = False
        return updates

    def _emit_tool_events(self, calls: list, messages: list[ToolMessage]) -> None:
        """把本轮工具结果写进 rollout 事件日志；未启用时零开销。"""
        if self._context.rollout is None or not messages:
            return
        names = {call.get("id"): call.get("name") for call in calls}
        for message in messages:
            self._context.rollout.append("tool_result", {
                "call_id": message.tool_call_id,
                "tool": names.get(message.tool_call_id, ""),
                "text_head": str(message.content or "")[:400],
            }, turn_id=self._context.turn_id)

    def _route(self, state: AgentState) -> str:
        if state.get("stop_reason"):
            return "end"
        if state.get("retry_model"):
            # 输出被截断后只重试模型；截断响应里的 tool_calls 没有执行。
            return "model"
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "end"

    def _estimate_request_tokens(self, messages: list) -> int:
        total = _estimate_tokens(self._tool_schema_text)
        for message in messages:
            total += _estimate_tokens(_message_text(message))
        return total + 512


def _has_pending_tool_call(messages: list) -> bool:
    """检查点最后一条工具调用是否缺少结果；主动压缩不能切断这对消息。"""
    last_call_index = -1
    for index, message in enumerate(messages):
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            last_call_index = index
    if last_call_index < 0:
        return False
    answered = {
        getattr(item, "tool_call_id", None)
        for item in messages[last_call_index + 1:]
        if isinstance(item, ToolMessage)
    }
    return any(call.get("id") not in answered
               for call in messages[last_call_index].tool_calls)


def _add_usage(target: dict[str, int], usage) -> None:
    for key, value in _normalize_usage(usage).items():
        target[key] = target.get(key, 0) + value


def _normalize_usage(usage) -> dict[str, int]:
    """把不同 provider 的 usage 归一化后累加，避免 input/prompt 双计。

    兼容 OpenAI/DeepSeek 的 prompt_tokens/prompt_cache_hit_tokens、
    LangChain 的 input_tokens、Anthropic 的 cache_read_input_tokens，以及
    prompt_tokens_details.cached_tokens 这类嵌套字段。
    """
    if not isinstance(usage, dict):
        return {}
    result: dict[str, int] = {}

    def add(key: str, value) -> None:
        if isinstance(value, int) and not isinstance(value, bool):
            result[key] = result.get(key, 0) + value

    def nested(*keys: str):
        current = usage
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    input_tokens = usage.get("input_tokens")
    if not isinstance(input_tokens, int):
        input_tokens = usage.get("prompt_tokens")
    add("input_tokens", input_tokens)
    output_tokens = usage.get("output_tokens")
    if not isinstance(output_tokens, int):
        output_tokens = usage.get("completion_tokens")
    add("output_tokens", output_tokens)
    add("total_tokens", usage.get("total_tokens"))

    cached = usage.get("cached_tokens")
    if not isinstance(cached, int):
        cached = usage.get("prompt_cache_hit_tokens")
    if not isinstance(cached, int):
        cached = usage.get("cache_read_input_tokens")
    if not isinstance(cached, int):
        cached = nested("prompt_tokens_details", "cached_tokens")
    if not isinstance(cached, int):
        cached = nested("input_token_details", "cache_read")
    add("cached_tokens", cached)

    cache_creation = usage.get("cache_creation_tokens")
    if not isinstance(cache_creation, int):
        cache_creation = usage.get("cache_creation_input_tokens")
    if not isinstance(cache_creation, int):
        cache_creation = nested("input_token_details", "cache_creation")
    add("cache_creation_tokens", cache_creation)

    reasoning = usage.get("reasoning_tokens")
    if not isinstance(reasoning, int):
        reasoning = nested("completion_tokens_details", "reasoning_tokens")
    if not isinstance(reasoning, int):
        reasoning = nested("output_token_details", "reasoning")
    add("reasoning_tokens", reasoning)

    # 保留 provider 特有的其它整数字段，但不重复计入已归一化的字段。
    consumed = {
        "input_tokens", "prompt_tokens", "output_tokens", "completion_tokens",
        "total_tokens", "cached_tokens", "prompt_cache_hit_tokens",
        "cache_read_input_tokens", "prompt_tokens_details",
        "cache_creation_tokens", "cache_creation_input_tokens",
        "reasoning_tokens", "completion_tokens_details", "output_token_details",
        "input_token_details",
    }
    for key, value in usage.items():
        if key not in consumed:
            add(str(key), value)
    return result


def _tool_schema_text(definitions: list) -> str:
    parts: list[str] = []
    for tool in definitions:
        parts.append(tool.name)
        parts.append(tool.description or "")
        schema = getattr(tool, "args_schema", None)
        if schema is not None and hasattr(schema, "model_json_schema"):
            parts.append(json.dumps(schema.model_json_schema(), ensure_ascii=False))
    return "\n".join(parts)


def _message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, list):
        content = json.dumps(content, ensure_ascii=False, default=str)
    parts = [str(content or "")]
    for call in getattr(message, "tool_calls", None) or []:
        parts.append(json.dumps(call, ensure_ascii=False, default=str))
    return "\n".join(parts)


def _is_length_truncated(response) -> bool:
    metadata = getattr(response, "response_metadata", None) or {}
    return metadata.get("finish_reason") == "length"


def _content_excerpt(response) -> str:
    """rollout 事件里的模型回复摘录：足够回看，不复制全文。"""
    content = getattr(response, "content", "")
    if isinstance(content, list):
        content = json.dumps(content, ensure_ascii=False, default=str)
    return str(content or "")[:600]


def _chunk_text(chunk) -> str:
    """提取流式 chunk 的可见文本；兼容字符串与 content blocks。"""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _args_excerpt(args) -> str:
    text = json.dumps(args, ensure_ascii=False, default=str)
    return text if len(text) <= 400 else text[:400] + "…"


def _response_stop_reason(response) -> str | None:
    if not getattr(response, "tool_calls", None) and not str(response.content or "").strip():
        return "模型返回空响应"
    return None


def _tool_error_text(text: str) -> str:
    """从工具结果里取出 error 行，供提醒使用；避免把整段输出带进上下文。"""
    marker = "\nerror: "
    if marker in text:
        text = text.split(marker, 1)[1]
    return text.strip().replace("\n", " ")[:400]


def _estimate_tokens(text: str) -> int:
    """应用注入内容的 token 估算：中文约 1 字符/token、英文约 4 字符/token，
    取 //2 作中间估计。这是估算，不是服务端精确用量。"""
    return (len(text) + 1) // 2
