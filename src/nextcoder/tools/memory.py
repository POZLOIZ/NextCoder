"""记忆工具：按需读取、写入和停用项目记忆。"""

from nextcoder.tools.base import ToolContext, ToolResult


def memory_list(context: ToolContext, call_id: str, query: str = "") -> ToolResult:
    memory = context.memory
    if memory is None:
        return ToolResult(call_id, "memory_list", "failed", error="项目记忆未启用")
    needle = " ".join(str(query or "").split()).lower()
    rows = []
    for item in memory.select(limit=1000):
        text = f"{item.key} {item.kind} {item.value}".lower()
        if needle and needle not in text:
            continue
        rows.append(f"- {item.key} [{item.kind}] {item.value[:240]}")
    if not rows:
        body = "（没有匹配的记忆）"
    else:
        body = "项目记忆主题：\n" + "\n".join(rows[:50])
        if len(rows) > 50:
            body += f"\n（另有 {len(rows) - 50} 条，缩小 query 再查）"
    return ToolResult(call_id, "memory_list", "completed", output_excerpt=body)


def memory_read(context: ToolContext, call_id: str, topic: str) -> ToolResult:
    memory = context.memory
    if memory is None:
        return ToolResult(call_id, "memory_read", "failed", error="项目记忆未启用")
    item = memory.get(topic)
    if item is None:
        return ToolResult(call_id, "memory_read", "failed", error=f"记忆不存在：{topic}")
    status = "有效" if item.invalidated_at is None else (
        f"停用：{item.invalid_reason or '未说明'}")
    body = (
        f"主题：{item.key}\n类型：{item.kind}\n状态：{status}\n"
        f"来源：{item.source}\n置信度：{item.confidence:.2f}\n正文：\n{item.value}"
    )
    return ToolResult(call_id, "memory_read", "completed", output_excerpt=body)


def memory_write(context: ToolContext, call_id: str, topic: str, content: str,
                 kind: str = "project_convention") -> ToolResult:
    memory = context.memory
    if memory is None:
        return ToolResult(call_id, "memory_write", "failed", error="项目记忆未启用")
    try:
        key = memory.remember(
            topic, content, kind=kind,
            source=f"agent:{context.turn_id}:memory_write",
            confidence=0.8,
            meta={"turn_id": context.turn_id},
        )
    except ValueError as exc:
        return ToolResult(call_id, "memory_write", "failed", error=str(exc))
    item = memory.get(key)
    if item is None:
        status = "未写入"
    elif item.invalidated_at is not None:
        status = "未覆盖（条目已停用，需手工恢复）"
    else:
        status = "已保存"
    return ToolResult(
        call_id, "memory_write", "completed", output_excerpt=f"{status}：{key}")


def memory_forget(context: ToolContext, call_id: str, topic: str) -> ToolResult:
    memory = context.memory
    if memory is None:
        return ToolResult(call_id, "memory_forget", "failed", error="项目记忆未启用")
    if memory.get(topic) is None:
        return ToolResult(call_id, "memory_forget", "failed", error=f"记忆不存在：{topic}")
    disabled = memory.disable(topic, "Agent 按用户要求停用")
    if not disabled:
        return ToolResult(
            call_id, "memory_forget", "completed",
            output_excerpt=f"已是停用状态：{topic}")
    return ToolResult(
        call_id, "memory_forget", "completed", output_excerpt=f"已停用：{topic}")
