"""运行事件协议：runtime 只发事件，CLI/TUI 决定怎么展示。

事件处理器可能运行在子 agent 的工作线程或 TUI 的 worker 线程里；所有
发送点都必须容错，展示层异常不能影响运行循环本身。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class TurnStarted:
    session_id: str
    turn_id: str
    request: str


@dataclass(frozen=True)
class AssistantDelta:
    turn_id: str
    text: str


@dataclass(frozen=True)
class ToolStarted:
    turn_id: str
    call_id: str
    tool: str
    detail: str = ""


@dataclass(frozen=True)
class ToolFinished:
    turn_id: str
    call_id: str
    tool: str
    status: str
    text: str = ""


@dataclass(frozen=True)
class PlanUpdated:
    turn_id: str
    revision: int
    tasks: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class Compaction:
    turn_id: str
    before_tokens: int
    after_tokens: int
    method: str
    reason: str = ""


@dataclass(frozen=True)
class PermissionRequested:
    turn_id: str
    tool: str
    action: str
    target: str
    reason: str = ""
    hint: str = ""


@dataclass(frozen=True)
class PermissionAnswered:
    turn_id: str
    tool: str
    action: str
    target: str
    answer: str


@dataclass(frozen=True)
class ChildUpdated:
    turn_id: str
    child_id: str
    role: str
    status: str
    detail: str = ""


@dataclass(frozen=True)
class HookTriggered:
    turn_id: str
    hook: str
    event: str
    tool: str
    action: str
    detail: str = ""


@dataclass(frozen=True)
class TurnFinished:
    turn_id: str
    stop_reason: str
    model_calls: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)


def emit(handler: Callable[[object], None] | None, event: object) -> None:
    """安全发送：展示层失败只丢弃该事件，不影响运行。"""
    if handler is None:
        return
    try:
        handler(event)
    except Exception:  # noqa: BLE001 - UI 不应改变运行结果
        return
