"""任务清单：任务目录级的小型计划状态，同一任务的所有会话共享。

清单保存在 <task_dir>/.nextcoder/task-plan.json；每次修改 revision + 1。
运行循环把快照注入模型视图，模型宣布完成时核对是否还有未完成项。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

PLAN_STATUSES = ("pending", "in_progress", "completed", "cancelled")
MAX_PLAN_TASKS = 30
_STATUS_MARKS = {"pending": "[ ]", "in_progress": "[~]",
                 "completed": "[x]", "cancelled": "[-]"}


@dataclass
class PlanTask:
    id: str
    content: str
    status: str = "pending"


@dataclass
class TaskPlan:
    revision: int = 0
    tasks: list[PlanTask] = field(default_factory=list)

    def open_items(self) -> list[PlanTask]:
        return [task for task in self.tasks if task.status in {"pending", "in_progress"}]


class PlanStore:
    """清单文件的薄封装：无状态对象，每次操作读写 JSON。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> TaskPlan:
        if not self.path.exists():
            return TaskPlan()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            tasks = [
                PlanTask(id=str(item["id"]), content=str(item["content"]),
                         status=str(item["status"]))
                for item in data.get("tasks", [])
                if isinstance(item, dict)
            ]
            revision = int(data.get("revision") or 0)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            # 损坏的清单不能阻断运行：当作空清单重新开始。
            return TaskPlan()
        return TaskPlan(revision=revision,
                        tasks=[task for task in tasks
                               if task.status in PLAN_STATUSES and task.content])

    def replace(self, contents: list[str]) -> TaskPlan:
        """整体重建清单；同序号且内容相同的旧项保留原状态。"""
        cleaned = [str(item).strip() for item in contents]
        if not cleaned or any(not item for item in cleaned):
            raise ValueError("任务清单不能为空，且每一项都必须有内容")
        if len(cleaned) > MAX_PLAN_TASKS:
            raise ValueError(f"任务清单最多 {MAX_PLAN_TASKS} 项，请合并或精简")
        previous = self.load()
        tasks: list[PlanTask] = []
        for index, content in enumerate(cleaned, start=1):
            task_id = f"t{index}"
            old = next((item for item in previous.tasks
                        if item.id == task_id and item.content == content), None)
            status = old.status if old is not None else "pending"
            tasks.append(PlanTask(task_id, content, status))
        plan = TaskPlan(revision=previous.revision + 1, tasks=tasks)
        self._save(plan)
        return plan

    def update(self, statuses: dict[str, str]) -> TaskPlan:
        plan = self.load()
        if not statuses:
            raise ValueError("statuses 不能为空")
        for status in statuses.values():
            if status not in PLAN_STATUSES:
                raise ValueError(f"未知状态：{status}（可选 {', '.join(PLAN_STATUSES)}）")
        known = {task.id for task in plan.tasks}
        unknown = [task_id for task_id in statuses if task_id not in known]
        if unknown:
            raise ValueError(f"未找到任务项：{', '.join(unknown)}")
        for task in plan.tasks:
            if task.id in statuses:
                task.status = statuses[task.id]
        plan.revision += 1
        self._save(plan)
        return plan

    def snapshot_text(self) -> str:
        """注入模型视图的清单快照；空清单返回空串（不注入）。"""
        plan = self.load()
        if not plan.tasks:
            return ""
        lines = [f"[任务清单 v{plan.revision}]"]
        lines.extend(
            f"- {_STATUS_MARKS[task.status]} {task.id}：{task.content}"
            for task in plan.tasks)
        return "\n".join(lines)

    def has_open_items(self) -> bool:
        return bool(self.load().open_items())

    def _save(self, plan: TaskPlan) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "revision": plan.revision,
            "tasks": [
                {"id": task.id, "content": task.content, "status": task.status}
                for task in plan.tasks
            ],
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
