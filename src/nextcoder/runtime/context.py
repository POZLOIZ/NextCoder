"""上下文准备：裁剪旧工具正文，并在需要时请求模型生成任务摘要。

第一层只影响“发给模型的本轮视图”：旧工具正文被替换为带调用 ID、状态、
改动文件和日志路径的占位；检查点仍保留完整历史，完整工具结果仍在操作表和
日志里。第一层仍装不下时，本模块给出需要摘要的消息范围和回退视图，由运行
循环调用模型生成任务摘要、持久化后再切换。摘要失败时不静默丢弃历史。
"""

import re
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from nextcoder.settings import Settings

_FULL_OUTPUT = re.compile(r"full_output=(\S+)")
_CHANGED = re.compile(r"changed=(\S+)")
_STATUS = re.compile(r"status=(\w+)")


@dataclass
class CompactionEvent:
    triggered: bool
    before_tokens: int
    after_tokens: int
    compacted_groups: int
    kept_groups: int
    reason: str
    method: str = "none"  # tool_stubs / model_summary / summary_reuse
    covered_groups: int = 0
    summary_chars: int = 0
    summary_id: str = ""
    summary_reused: bool = False
    low_watermark: int = 0
    summary_attempts: int = 0
    retry_reason: str = ""


@dataclass
class SummaryRequest:
    """一次待生成的语义摘要；范围是完整工具组边界。"""

    old_summary: str
    old_covered_count: int
    cutoff: int
    messages: list
    covered_groups: int
    before_tokens: int
    fallback_tokens: int
    # 首条用户要求不在被摘要的消息范围里，必须单独带给摘要模型，
    # 否则新增过程从工具调用开始时摘要会丢失任务目标。
    task: str = ""


@dataclass
class PreparedContext:
    view: list
    event: CompactionEvent
    summary_request: SummaryRequest | None = None
    before_tokens: int = 0
    fallback_view: list | None = None


def prepare_context(messages: list, settings: Settings, estimate, *,
                    summary_text: str | None = None,
                    covered_count: int = 0,
                    prior_request_tokens: int = 0,
                    prior_message_count: int = 0,
                    force: bool = False,
                    recent_ratio: float | None = None) -> PreparedContext:
    """按预算准备模型视图；需要新摘要时返回 SummaryRequest。

    触发线按“上一次请求实际视图 + 上次请求之后新增的消息”估算，而不是按
    检查点全量历史估算。后者会在每轮把同一批旧消息重新折叠一次，产生大量
    tool_stubs 事件；前者只在视图真的接近预算时才压缩。

    已有成功摘要始终优先复用，即使当前估算低于触发线，避免下一次请求又
    退回完整历史。只有摘要仍超过触发线、且存在尚未覆盖的完整工具组时，才
    请求生成新摘要。近期工具组按 token 软预算保留。
    force=True 供主动压缩使用：即使未到触发线也生成摘要，但仍保留最近工作集
    和最后一条用户要求。
    recent_ratio 只用于摘要重试：第二次摘要把近期预算降得更低，覆盖更多旧内容。
    """
    full_tokens = estimate(messages)
    if prior_request_tokens > 0 and 0 < prior_message_count <= len(messages):
        effective_before = prior_request_tokens + estimate(
            messages[prior_message_count:])
    else:
        effective_before = full_tokens
    budget = settings.application_input_token_budget - settings.output_reserve_tokens
    trigger = max(1, int(budget * settings.compaction_trigger_ratio))
    low = max(1, int(budget * settings.compaction_low_watermark_ratio))

    system, first_user, groups, group_starts = _split(messages)
    base_start = len(system) + (1 if first_user is not None else 0)
    fallback_compacted = 0
    fallback_kept = 0

    valid_summary = bool(
        summary_text and covered_count > base_start
        and covered_count <= len(messages)
        and is_valid_coverage(messages, covered_count)
        and preserves_latest_user(messages, covered_count)
    )
    if valid_summary:
        view = build_summary_view(messages, summary_text or "", covered_count)
        tokens = estimate(view)
        # 摘要视图就是实际请求视图，触发判断和事件都应以它为准。
        before = tokens
        if tokens <= trigger and not force:
            return PreparedContext(
                view,
                CompactionEvent(
                    False, before, tokens, 0,
                    _group_count_after(groups, group_starts, covered_count),
                    "复用已保存的任务摘要", method="summary_reuse",
                    covered_groups=_group_count_before(group_starts, covered_count),
                    summary_chars=len(summary_text or ""), summary_reused=True,
                ),
                before_tokens=before,
            )
        fallback_view = view
        fallback_tokens = tokens
        fallback_compacted = _group_count_before(group_starts, covered_count)
        fallback_kept = _group_count_after(groups, group_starts, covered_count)
    else:
        summary_text = None
        covered_count = 0
        before = effective_before
        # 上一轮已经裁剪过、这次新增内容还没把视图推过触发线：复用同一套
        # 裁剪视图，但不再记一条新的压缩事件。
        reuse_stubs = (prior_request_tokens > 0 and full_tokens > trigger)
        if before <= trigger and not force:
            if not reuse_stubs or not groups:
                return PreparedContext(
                    messages,
                    CompactionEvent(False, before, before, 0, 0, "未接近预算"),
                    before_tokens=before,
                )
            tool_indices = [index for index, group in enumerate(groups)
                            if not isinstance(group[0], HumanMessage)]
            pending = {index for index in tool_indices
                       if _is_pending_group(groups[index])}
            recent_budget = max(
                1, int(budget * (
                    recent_ratio if recent_ratio is not None
                    else settings.compaction_recent_token_ratio)))
            keep_indices = _select_recent(
                groups, tool_indices, recent_budget,
                settings.compaction_keep_recent_groups, pending)
            stub_view = _build_stub_view(
                system, first_user, groups, keep_indices,
                settings.compaction_stub_chars)
            stub_tokens = estimate(stub_view)
            compacted = len(
                [index for index in tool_indices if index not in keep_indices])
            if settings.compaction_stub_first:
                return PreparedContext(
                    stub_view,
                    CompactionEvent(
                        False, before, stub_tokens, compacted, len(keep_indices),
                        "复用上一轮的裁剪视图；新增内容尚未推过触发线",
                        method="tool_stubs_reuse", covered_groups=compacted,
                    ),
                    before_tokens=before,
                )
            # 基线模式：裁剪视图只作为摘要失败时的回退，不把它当终点；
            # 继续往下请求模型摘要。
            fallback_view = stub_view
            fallback_tokens = stub_tokens
            fallback_compacted = compacted
            fallback_kept = len(keep_indices)
        if not groups:
            return PreparedContext(
                messages,
                CompactionEvent(False, before, before, 0, 0, "没有可折叠的历史"),
                before_tokens=before,
            )
        tool_indices = [index for index, group in enumerate(groups)
                        if not isinstance(group[0], HumanMessage)]
        pending = {index for index in tool_indices if _is_pending_group(groups[index])}
        recent_budget = max(1, int(budget * (
            recent_ratio if recent_ratio is not None
            else settings.compaction_recent_token_ratio)))
        keep_indices = _select_recent(groups, tool_indices, recent_budget,
                                      settings.compaction_keep_recent_groups, pending)
        stub_view = _build_stub_view(system, first_user, groups, keep_indices,
                                     settings.compaction_stub_chars)
        stub_tokens = estimate(stub_view)
        compacted = len([index for index in tool_indices if index not in keep_indices])
        if (not force and settings.compaction_stub_first
                and stub_tokens <= low):
            return PreparedContext(
                stub_view,
                CompactionEvent(
                    True, before, stub_tokens, compacted, len(keep_indices),
                    (f"输入估算 {before} 超过触发阈值 {trigger}；旧工具正文已裁剪到 "
                     f"{stub_tokens}，不高于低水位 {low}，不再摘要"),
                    method="tool_stubs", covered_groups=compacted,
                    low_watermark=low,
                ),
                before_tokens=before,
            )
        fallback_view = stub_view
        fallback_tokens = stub_tokens
        fallback_compacted = compacted
        fallback_kept = len(keep_indices)

    # 第二层：模型摘要。只覆盖完整工具组，保留最近工作集和最后一条用户要求。
    request = _summary_request(
        messages, settings, groups, group_starts, base_start,
        summary_text, covered_count, before, fallback_tokens,
        recent_ratio=recent_ratio,
    )
    if before > trigger:
        pressure = f"输入估算 {before} 超过触发阈值 {trigger}；"
        if settings.compaction_stub_first and fallback_tokens > low:
            pressure += f"折叠后 {fallback_tokens} 仍高于低水位 {low}；"
    else:
        pressure = "主动压缩；"
    if request is None:
        event = CompactionEvent(
            True, before, fallback_tokens,
            fallback_compacted,
            fallback_kept,
            (f"{pressure}已有摘要覆盖可折叠范围，但当前请求仍为 {fallback_tokens}"),
            method="summary_reuse" if valid_summary else "tool_stubs",
            covered_groups=_group_count_before(group_starts, covered_count)
            if valid_summary else 0,
            summary_chars=len(summary_text or ""),
            summary_reused=valid_summary,
            low_watermark=low,
        )
        return PreparedContext(
            fallback_view, event, before_tokens=before, fallback_view=fallback_view)

    event = CompactionEvent(
        True, before, fallback_tokens,
        fallback_compacted,
        fallback_kept,
        f"{pressure}需要模型生成任务摘要",
        method="summary_reuse" if valid_summary else "tool_stubs",
        covered_groups=_group_count_before(group_starts, covered_count)
        if valid_summary else 0,
        summary_chars=len(summary_text or ""),
        summary_reused=valid_summary,
        low_watermark=low,
    )
    return PreparedContext(
        fallback_view, event, summary_request=request,
        before_tokens=before, fallback_view=fallback_view)


def apply_summary(messages: list, settings: Settings, estimate, summary_text: str,
                  request: SummaryRequest, *, summary_id: str = "",
                  summary_chars: int | None = None) -> tuple[list, CompactionEvent]:
    """摘要已持久化后构建请求视图；调用方必须先保存成功再调用本函数。"""
    view = build_summary_view(messages, summary_text, request.cutoff)
    tokens = estimate(view)
    _, _, groups, group_starts = _split(messages)
    kept = _group_count_after(groups, group_starts, request.cutoff)
    budget = settings.application_input_token_budget - settings.output_reserve_tokens
    trigger = max(1, int(budget * settings.compaction_trigger_ratio))
    low = max(1, int(budget * settings.compaction_low_watermark_ratio))
    event = CompactionEvent(
        True, request.before_tokens, tokens, request.covered_groups, kept,
        "已生成并保存模型任务摘要",
        method="model_summary", covered_groups=request.covered_groups,
        summary_chars=summary_chars if summary_chars is not None else len(summary_text),
        summary_id=summary_id, low_watermark=low,
    )
    # 摘要后仍用同一条触发线复核；不引入第二个目标。仍超出时如实记录，
    # 请求能否发出由总窗口（含输出预留）的硬预算决定。
    if tokens > trigger:
        event.reason += f"；摘要后仍高于触发线 {trigger}，按总预算继续"
    return view, event


def compact_messages(messages: list, settings: Settings, estimate) -> tuple[list, CompactionEvent]:
    """兼容入口：只做第一层裁剪；需要摘要时返回回退视图和提示事件。

    运行循环应使用 prepare_context/apply_summary，以便真正调用模型摘要。
    """
    prepared = prepare_context(messages, settings, estimate)
    if prepared.summary_request is not None:
        prepared.event.reason += "；仍需模型语义摘要"
    return prepared.view, prepared.event


def build_summary_view(messages: list, summary_text: str, covered_count: int) -> list:
    """按“系统 + 首条用户要求 + 摘要 + 未覆盖消息”构建请求视图。"""
    system, first_user, _groups, _starts = _split(messages)
    result = list(system)
    if first_user is not None:
        result.append(first_user)
    result.append(AIMessage(content="[任务摘要 v2]\n" + summary_text))
    result.extend(messages[covered_count:])
    return result


def _summary_request(messages: list, settings: Settings, groups: list[list],
                     group_starts: list[int], base_start: int,
                     summary_text: str | None, covered_count: int,
                     before: int, fallback_tokens: int,
                     *, recent_ratio: float | None = None) -> SummaryRequest | None:
    tool_indices = [index for index, group in enumerate(groups)
                    if not isinstance(group[0], HumanMessage)]
    if not tool_indices:
        return None
    pending = {index for index in tool_indices if _is_pending_group(groups[index])}
    budget = settings.application_input_token_budget - settings.output_reserve_tokens
    recent_budget = max(1, int(budget * (
        recent_ratio if recent_ratio is not None
        else settings.compaction_recent_token_ratio)))
    keep = _select_recent(groups, tool_indices, recent_budget,
                          settings.compaction_keep_recent_groups, pending)
    cutoff_group = min(keep) if keep else len(groups)
    # 最后一条用户要求必须保持完整，不能折进摘要。
    human_indices = [index for index, group in enumerate(groups)
                     if isinstance(group[0], HumanMessage)]
    if human_indices:
        cutoff_group = min(cutoff_group, human_indices[-1])
    if cutoff_group <= 0:
        return None
    cutoff = base_start + sum(len(group) for group in groups[:cutoff_group])
    old_start = max(covered_count if summary_text else base_start, base_start)
    if cutoff <= old_start:
        return None
    _system, first_user, _groups, _starts = _split(messages)
    return SummaryRequest(
        old_summary=summary_text or "",
        old_covered_count=old_start,
        cutoff=cutoff,
        messages=messages[old_start:cutoff],
        covered_groups=cutoff_group,
        before_tokens=before,
        fallback_tokens=fallback_tokens,
        task=_message_text(first_user) if first_user is not None else "",
    )


def _split(messages: list) -> tuple[list, HumanMessage | None, list[list], list[int]]:
    """拆出首条系统消息、首条用户消息和后续完整消息组。

    带 tool_calls 的 AIMessage 与其 ToolMessage 视为一组；组边界用于摘要和
    覆盖范围，绝不能留下没有结果的工具调用。
    """
    system: list = []
    first_user: HumanMessage | None = None
    groups: list[list] = []
    starts: list[int] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if isinstance(message, SystemMessage) and not system and first_user is None:
            system.append(message)
            index += 1
            continue
        if isinstance(message, HumanMessage) and first_user is None:
            first_user = message
            index += 1
            continue
        start = index
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            group = [message]
            index += 1
            while index < len(messages) and isinstance(messages[index], ToolMessage):
                group.append(messages[index])
                index += 1
            groups.append(group)
            starts.append(start)
            continue
        groups.append([message])
        starts.append(start)
        index += 1
    return system, first_user, groups, starts


def is_valid_coverage(messages: list, covered: int) -> bool:
    """覆盖点必须落在完整消息组之间；不能从 ToolMessage 中间切开。"""
    if covered < 0 or covered > len(messages):
        return False
    if covered == len(messages):
        return True
    return not isinstance(messages[covered], ToolMessage)


def preserves_latest_user(messages: list, covered: int) -> bool:
    """摘要不能吃掉最新用户要求；首条用户要求由视图单独保留。"""
    human_indices = [index for index, message in enumerate(messages)
                     if isinstance(message, HumanMessage)]
    if not human_indices:
        return True
    # 首条用户要求始终原样放在摘要之前，即使覆盖点越过了它也不受影响。
    if human_indices[-1] == human_indices[0]:
        return True
    return covered <= human_indices[-1]


def _select_recent(groups: list[list], tool_indices: list[int], recent_budget: int,
                   max_groups: int, pending: set[int]) -> set[int]:
    """从最近往前保留工具组，直到达到 token 软预算或条数上限。"""
    keep: set[int] = set(pending)
    used = sum(_group_tokens(groups[index], len(groups)) for index in pending)
    for index in reversed(tool_indices):
        if index in keep:
            continue
        if len(keep) >= max(max_groups, len(pending)):
            break
        cost = _group_tokens(groups[index], len(groups))
        if keep and used + cost > recent_budget:
            break
        keep.add(index)
        used += cost
    if tool_indices and not keep:
        keep.add(tool_indices[-1])
    return keep


def _group_tokens(group: list, _total: int) -> int:
    return sum((len(_message_text(message)) + 1) // 2 for message in group)


def _build_stub_view(system: list, first_user: HumanMessage | None,
                     groups: list[list], keep: set[int], stub_chars: int) -> list:
    result = list(system)
    if first_user is not None:
        result.append(first_user)
    for index, group in enumerate(groups):
        if index in keep or isinstance(group[0], HumanMessage):
            result.extend(group)
            continue
        stub = _stub_for_group(group, stub_chars)
        if stub is not None:
            result.append(stub)
    return result


def _stub_for_group(group: list, stub_chars: int) -> AIMessage | None:
    first = group[0]
    if isinstance(first, AIMessage) and getattr(first, "tool_calls", None):
        lines = []
        for call in first.tool_calls:
            call_id = call.get("id", "?")
            name = call.get("name", "?")
            result = next((item for item in group[1:]
                           if isinstance(item, ToolMessage)
                           and item.tool_call_id == call_id), None)
            status = _status(result)
            output = _match(_FULL_OUTPUT, result)
            changed = _match(_CHANGED, result)
            detail = f"{call_id} {name} status={status}"
            if changed:
                detail += f" changed={changed}"
            if output:
                detail += f" full_output={output}"
            lines.append(detail)
        if not lines:
            return None
        text = "[早期工具摘要] " + "；".join(lines)
        return AIMessage(content=_clip(text, max(stub_chars, 80)))
    if isinstance(first, AIMessage):
        text = _message_text(first)
        if not text.strip():
            return None
        return AIMessage(content="[早期助手摘要] " + _clip(text, stub_chars))
    if isinstance(first, HumanMessage):
        return HumanMessage(content=first.content)
    text = _message_text(first)
    return AIMessage(content="[早期消息摘要] " + _clip(text, stub_chars)) if text else None


def _is_pending_group(group: list) -> bool:
    if not isinstance(group[0], AIMessage) or not getattr(group[0], "tool_calls", None):
        return False
    answered = {getattr(item, "tool_call_id", None) for item in group[1:]
                if isinstance(item, ToolMessage)}
    return any(call.get("id") not in answered for call in group[0].tool_calls)


def _group_count_before(starts: list[int], covered: int) -> int:
    return sum(1 for start in starts if start < covered)


def _group_count_after(groups: list[list], starts: list[int], covered: int) -> int:
    return sum(1 for index, start in enumerate(starts) if start >= covered)


def _status(message: ToolMessage | None) -> str:
    if message is None:
        return "missing"
    match = _STATUS.search(str(message.content or ""))
    return match.group(1) if match else "unknown"


def _match(pattern: re.Pattern, message: ToolMessage | None) -> str | None:
    if message is None:
        return None
    match = pattern.search(str(message.content or ""))
    return match.group(1) if match else None


def _message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, list):
        import json
        return json.dumps(content, ensure_ascii=False, default=str)
    return str(content or "")


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"
