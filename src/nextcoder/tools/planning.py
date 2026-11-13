"""任务清单工具：创建与更新计划；快照由运行循环注入模型视图。"""

from nextcoder.runtime.events import PlanUpdated, emit
from nextcoder.runtime.planning import PlanStore, TaskPlan
from nextcoder.tools.base import ToolContext, ToolResult


def _record(context: ToolContext, action: str, plan: TaskPlan) -> None:
    """清单文件是事实源；rollout 只追加可读事件，供回放与审计。"""
    tasks = tuple(
        {"id": task.id, "content": task.content, "status": task.status}
        for task in plan.tasks
    )
    emit(context.event_handler,
         PlanUpdated(context.turn_id, plan.revision, tasks))
    if context.rollout is None:
        return
    context.rollout.append("plan_updated", {
        "action": action,
        "revision": plan.revision,
        "tasks": list(tasks),
    }, turn_id=context.turn_id)


def plan_create(context: ToolContext, call_id: str, tasks: list[str]) -> ToolResult:
    store: PlanStore | None = context.plan
    if store is None:
        return ToolResult(call_id, "plan_create", "failed", error="任务清单未启用")
    try:
        plan = store.replace([str(item) for item in tasks])
    except ValueError as exc:
        return ToolResult(call_id, "plan_create", "failed", error=str(exc))
    _record(context, "create", plan)
    return ToolResult(
        call_id, "plan_create", "completed",
        output_excerpt=(
            f"已创建任务清单 v{plan.revision}，共 {len(plan.tasks)} 项"
            f"（id 为 t1…t{len(plan.tasks)}）。用 plan_update 更新状态。"))


def plan_update(context: ToolContext, call_id: str,
                statuses: dict[str, str]) -> ToolResult:
    store: PlanStore | None = context.plan
    if store is None:
        return ToolResult(call_id, "plan_update", "failed", error="任务清单未启用")
    try:
        plan = store.update({str(key): str(value) for key, value in statuses.items()})
    except ValueError as exc:
        return ToolResult(call_id, "plan_update", "failed", error=str(exc))
    _record(context, "update", plan)
    open_count = len(plan.open_items())
    summary = f"已更新任务清单 v{plan.revision}；未完成 {open_count} 项"
    if open_count:
        remaining = "、".join(task.id for task in plan.open_items())
        summary += f"（{remaining}）"
    return ToolResult(call_id, "plan_update", "completed", output_excerpt=summary + "。")
