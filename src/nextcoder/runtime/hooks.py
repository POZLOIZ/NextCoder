"""工具生命周期 hooks：before_tool_use / after_tool_use。

hook 是运行循环和工具之间的可插拔检查点，不修改工具实现本身。执行顺序：

1. 参数校验、操作记录 prepare；
2. `before_tool_use`：按优先级执行。可以阻止调用、替换参数、追加提示，
   或排队一条提醒（下一次模型请求时注入）；参数替换会同步到审计记录；
3. 发送 ToolStarted（显示实际执行的参数）、权限判定与实际执行；
4. `after_tool_use`：总会在执行后执行（包括被权限拒绝/钩子阻止的调用），
   可以追加提示或排队提醒，但不能改变已发生的副作用。

内置的反重复调用、搜索预算、连续空输出提醒也注册在同一总线里，因此它们
和用户 hook 的语义完全一致：只提醒、不缓存结果、不伪造输出。配置 hook 写在
`<task_dir>/.nextcoder/hooks.json`，只支持声明式动作，不执行任意脚本。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from nextcoder.runtime.events import HookTriggered, emit
from nextcoder.runtime.progress import call_signature, calls_signature
from nextcoder.settings import Settings

if TYPE_CHECKING:
    from nextcoder.tools.base import ToolResult


class HookConfigError(ValueError):
    """hooks.json 缺失字段、未知事件或非法正则。"""


@dataclass(frozen=True)
class HookContext:
    event: str
    turn_id: str
    tool: str
    call_id: str
    args: dict[str, Any]
    task_dir: Path
    result: "ToolResult | None" = None
    # 模型一次返回多个并行调用时，用整组签名判断“重复调用组合”；
    # 组内只有第一个调用更新计数，避免一组 N 个调用被算成 N 次重复。
    group_signature: str = ""
    group_index: int = 0
    group_size: int = 1


@dataclass
class HookOutcome:
    blocked: bool = False
    reason: str = ""
    args: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    reminders: list[str] = field(default_factory=list)


class BaseHook:
    name = "hook"
    event = "before_tool_use"
    priority = 0

    def run(self, context: HookContext) -> HookOutcome | None:
        raise NotImplementedError


_SEARCH_COMMAND = re.compile(
    r"(?:^|[;&|]\s*)"
    r"(?:git\s+(?:grep|log|show)|grep|rg|ag|ack|find|ls|tree|sed)\b"
)
_EDIT_TOOLS = {"apply_patch", "write_file"}


def is_search_call(tool: str, args: dict[str, Any]) -> bool:
    """搜索类动作共享一个阶段计数：search/list_files/grep/rg/find 等。"""
    if tool in {"search", "list_files"}:
        return True
    if tool != "run_command":
        return False
    return bool(_SEARCH_COMMAND.search(str(args.get("command", ""))))


def _call_detail(tool: str, args: dict[str, Any]) -> str:
    detail = args.get("command") or args.get("path") or args.get("pattern") or ""
    detail = str(detail).replace("\n", " ")
    return f"{tool}({detail[:160]})" if detail else tool


class RepeatedCallHook(BaseHook):
    """连续相同调用达到阈值时排队一条提醒；不阻止、不复用旧结果。"""

    name = "builtin.repeated_call"
    event = "before_tool_use"
    priority = 900

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self.reset()

    def reset(self) -> None:
        self._last = ""
        self._streak = 0
        self._warned_at = 0

    def seed(self, messages: list) -> None:
        """从历史消息恢复连续计数；跨轮对话仍能识别重复探索。"""
        groups: list[str] = []
        latest: list = []
        for message in reversed(messages):
            if not isinstance(message, AIMessage):
                continue
            calls = list(getattr(message, "tool_calls", None) or [])
            if not calls:
                continue
            signature = calls_signature(calls)
            if not groups:
                groups.append(signature)
                latest = calls
            elif signature == groups[-1]:
                groups.append(signature)
            else:
                break
        if not latest:
            return
        self._streak = len(groups)
        # 单个调用沿用单调用签名；并行组用整组签名，二者不能混用。
        self._last = (
            call_signature(latest[0]) if len(latest) == 1 else groups[0])

    def run(self, context: HookContext) -> HookOutcome | None:
        if self._limit <= 0:
            return None
        if context.group_size > 1 and context.group_index != 0:
            # 整组已在第一个调用时计过一次，组内其余调用不重复累计。
            return None
        signature = (
            context.group_signature
            if context.group_size > 1 and context.group_signature
            else call_signature({"name": context.tool, "args": context.args})
        )
        if signature == self._last:
            self._streak += 1
        else:
            # 换了一个调用就开启新一轮连续计数；提醒节流也必须清零，
            # 否则新序列要等到“上一次提醒点 + limit”才会再次提醒。
            self._last, self._streak = signature, 1
            self._warned_at = 0
        if self._streak < self._limit:
            return None
        # 达到阈值后每再累计 limit 次提醒一次，避免每个调用都重复同一句。
        if self._warned_at and self._streak - self._warned_at < self._limit:
            return None
        self._warned_at = self._streak
        if context.group_size > 1:
            subject = "完全相同的并行调用组合"
            detail = f"{context.group_size} 个调用"
        else:
            subject = "完全相同的调用"
            detail = _call_detail(context.tool, context.args)
        return HookOutcome(reminders=[
            f"[重复探索提醒] 你已经连续 {self._streak} 次发起{subject}："
            f"{detail}。请检查这是否是重复探索："
            "如果已有足够信息，考虑换一种方法、读取具体文件、修改代码或运行验证；"
            "如果环境可能已经变化，也可以重新执行同一命令获取新结果。"
        ])


class SearchBudgetHook(BaseHook):
    """自上次成功修改文件以来的搜索次数提醒；成功编辑后由 after hook 复位。"""

    name = "builtin.search_budget"
    event = "before_tool_use"
    priority = 910

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self.reset()

    def reset(self) -> None:
        self._count = 0
        self._warned_at = 0

    def run(self, context: HookContext) -> HookOutcome | None:
        if self._limit <= 0 or not is_search_call(context.tool, context.args):
            return None
        self._count += 1
        if self._count < self._limit:
            return None
        if self._warned_at and self._count - self._warned_at < self._limit:
            return None
        self._warned_at = self._count
        return HookOutcome(reminders=[
            f"[搜索提醒] 自上次修改文件以来已执行 {self._count} 次搜索/查找。"
            "搜索次数本身不能说明信息已经足够：如果仍在定位问题，可以继续搜索"
            "或换一种查找方式；如果已有足够信息，建议开始修改并运行验证。"
        ])

    def after(self, context: HookContext) -> HookOutcome | None:
        if context.tool in _EDIT_TOOLS and context.result is not None \
                and context.result.status == "completed":
            self.reset()
        return None


class EmptyOutputHook(BaseHook):
    """连续多个命令零输出时提醒检查 stderr/管道退出码。"""

    name = "builtin.empty_output"
    event = "after_tool_use"
    priority = 920

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self.reset()

    def reset(self) -> None:
        self._streak = 0
        self._warned_at = 0

    def seed(self, messages: list) -> None:
        count = 0
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                break
            if not isinstance(message, ToolMessage):
                continue
            text = str(message.content or "")
            if "输出为空" in text or "（无输出）" in text:
                count += 1
            else:
                break
        self._streak = count

    def run(self, context: HookContext) -> HookOutcome | None:
        if context.result is None:
            return None
        if context.tool != "run_command":
            # 任何有正文的非命令结果都打断“连续空命令”序列。
            if context.result.output_excerpt.strip():
                self._streak = 0
                self._warned_at = 0
            return None
        if context.result.status != "completed" or context.result.error:
            # 失败、权限拒绝、钩子阻止、沙箱 fail-closed 都不是“空输出循环”；
            # 它们本身已有错误信息，不应再排一条反死循环提醒。
            self._streak = 0
            self._warned_at = 0
            return None
        if context.result.output_excerpt.strip():
            self._streak = 0
            self._warned_at = 0
            return None
        self._streak += 1
        if self._limit <= 0 or self._streak < self._limit:
            return None
        if self._warned_at and self._streak - self._warned_at < self._limit:
            return None
        self._warned_at = self._streak
        return HookOutcome(reminders=[
            f"[反死循环] 最近 {self._streak} 个命令都没有输出。停止逐行试探；"
            "先检查命令本身是否失败（去掉 2>/dev/null，必要时加 "
            "`set -o pipefail`），或改用 read_lines/read_log 读取文件。"
        ])


class ConfigHook(BaseHook):
    """hooks.json 里的声明式 hook：匹配工具/正则后 block/note/reminder。"""

    def __init__(self, name: str, event: str, tools: frozenset[str] | None,
                 pattern: re.Pattern[str] | None, action: str, text: str,
                 priority: int) -> None:
        self.name = name
        self.event = event
        self.tools = tools
        self.pattern = pattern
        self.action = action
        self.text = text
        self.priority = priority

    def run(self, context: HookContext) -> HookOutcome | None:
        if self.event != context.event:
            return None
        if self.tools is not None and context.tool not in self.tools:
            return None
        if self.pattern is not None:
            subject = _subject(context)
            if self.pattern.search(subject) is None:
                return None
        if self.action == "block":
            return HookOutcome(blocked=True, reason=self.text)
        if self.action == "note":
            return HookOutcome(notes=[self.text])
        return HookOutcome(reminders=[self.text])


def _subject(context: HookContext) -> str:
    """匹配串：命令/路径等参数 + JSON 参数；after 还包含结果正文。"""
    parts = [
        str(context.args.get("command", "")),
        str(context.args.get("path", "")),
        str(context.args.get("pattern", "")),
        json.dumps(context.args, ensure_ascii=False, sort_keys=True),
    ]
    if context.result is not None:
        parts.extend([
            str(context.result.output_excerpt or ""),
            str(context.result.error or ""),
        ])
    return "\n".join(parts)


def load_config_hooks(path: Path) -> list[ConfigHook]:
    """读取 `.nextcoder/hooks.json`；文件不存在时返回空表。"""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HookConfigError(f"无法读取 hooks 配置 {path}：{exc}") from exc
    entries = data.get("hooks") if isinstance(data, dict) else data
    if isinstance(data, dict) and "hooks" not in data:
        raise HookConfigError(
            f"{path} 必须是 {{\"hooks\": [...]}} 或 hook 数组")
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise HookConfigError(f"{path} 的 hooks 必须是数组")
    hooks: list[ConfigHook] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise HookConfigError(f"{path} 第 {index} 条 hook 不是对象")
        name = str(entry.get("name") or f"config-{index}").strip()
        event = str(entry.get("event") or "").strip()
        if event not in {"before_tool_use", "after_tool_use"}:
            raise HookConfigError(
                f"hook {name} 的 event 必须是 before_tool_use 或 after_tool_use")
        action = str(entry.get("action") or "").strip()
        if action not in {"block", "note", "reminder"}:
            raise HookConfigError(
                f"hook {name} 的 action 必须是 block/note/reminder")
        if action == "block" and event != "before_tool_use":
            raise HookConfigError(f"hook {name} 不能在 after_tool_use 中 block")
        text = str(entry.get("reason") if action == "block" else entry.get("text"))
        if not text or text == "None":
            key = "reason" if action == "block" else "text"
            raise HookConfigError(f"hook {name} 缺少 {key}")
        raw_tools = entry.get("tool")
        if isinstance(raw_tools, str):
            tools: frozenset[str] | None = frozenset({raw_tools})
        elif isinstance(raw_tools, list) and raw_tools:
            tools = frozenset(str(item) for item in raw_tools)
        elif raw_tools is None:
            tools = None
        else:
            raise HookConfigError(f"hook {name} 的 tool 必须是字符串或数组")
        pattern = None
        raw_pattern = entry.get("pattern")
        if raw_pattern:
            try:
                pattern = re.compile(str(raw_pattern))
            except re.error as exc:
                raise HookConfigError(
                    f"hook {name} 的正则无效：{exc}") from exc
        try:
            priority = int(entry.get("priority", 100))
        except (TypeError, ValueError) as exc:
            raise HookConfigError(f"hook {name} 的 priority 必须是整数") from exc
        hooks.append(ConfigHook(name, event, tools, pattern, action, text, priority))
    return hooks


class HookBus:
    """一次运行内的 hook 总线；提醒队列在每次模型请求前被取走。"""

    def __init__(self, hooks: Iterable[BaseHook], *,
                 turn_id: str = "",
                 event_handler: Callable[[object], None] | None = None,
                 rollout=None,
                 errors: list[str] | None = None) -> None:
        self._hooks = sorted(hooks, key=lambda hook: hook.priority)
        self._turn_id = turn_id
        self._event_handler = event_handler
        self._rollout = rollout
        self._reminders: list[str] = []
        self.errors = errors or []

    def reset(self) -> None:
        self._reminders.clear()
        for hook in self._hooks:
            reset = getattr(hook, "reset", None)
            if callable(reset):
                reset()

    def seed(self, messages: list) -> None:
        for hook in self._hooks:
            seed = getattr(hook, "seed", None)
            if callable(seed):
                seed(messages)

    def collect_reminders(self) -> list[str]:
        reminders, self._reminders = self._reminders, []
        return reminders

    def before_tool_use(self, tool: str, call_id: str, args: dict[str, Any],
                        task_dir: Path, *, group_signature: str = "",
                        group_index: int = 0,
                        group_size: int = 1) -> HookOutcome:
        outcome = HookOutcome()
        current = dict(args)
        changed = False
        for hook in self._hooks:
            if hook.event != "before_tool_use":
                continue
            try:
                result = hook.run(HookContext(
                    "before_tool_use", self._turn_id, tool, call_id, current,
                    task_dir, group_signature=group_signature,
                    group_index=group_index, group_size=group_size))
            except Exception as exc:  # noqa: BLE001 - 单个 hook 失败不能中断工具
                self.errors.append(f"{hook.name}: {type(exc).__name__}: {exc}")
                continue
            if result is None:
                continue
            if not (result.blocked or result.args is not None
                    or result.notes or result.reminders):
                continue
            outcome.notes.extend(result.notes)
            outcome.reminders.extend(result.reminders)
            if result.args is not None:
                current = dict(result.args)
                changed = True
            self._record(hook, "before_tool_use", tool, result)
            if result.blocked:
                outcome.blocked = True
                outcome.reason = result.reason
                break
        if changed:
            outcome.args = current
        self._reminders.extend(outcome.reminders)
        return outcome

    def after_tool_use(self, tool: str, call_id: str, args: dict[str, Any],
                       task_dir: Path, result: "ToolResult") -> HookOutcome:
        outcome = HookOutcome()
        for hook in self._hooks:
            context = HookContext(
                "after_tool_use", self._turn_id, tool, call_id, args, task_dir,
                result=result)
            # after 事件 hook 走 run()；before 事件 hook 可以额外实现 after()
            # 观察结果（例如搜索预算 hook 在成功编辑后复位计数）。
            handler = hook.run if hook.event == "after_tool_use" else \
                getattr(hook, "after", None)
            if not callable(handler):
                continue
            try:
                hook_result = handler(context)
            except Exception as exc:  # noqa: BLE001 - 单个 hook 失败不能中断工具
                self.errors.append(f"{hook.name}: {type(exc).__name__}: {exc}")
                continue
            if hook_result is None or not (
                    hook_result.notes or hook_result.reminders):
                continue
            outcome.notes.extend(hook_result.notes)
            outcome.reminders.extend(hook_result.reminders)
            self._record(hook, "after_tool_use", tool, hook_result)
        self._reminders.extend(outcome.reminders)
        return outcome

    def _record(self, hook: BaseHook, event: str, tool: str,
                outcome: HookOutcome) -> None:
        actions = []
        if outcome.blocked:
            actions.append("block")
        if outcome.args is not None:
            actions.append("args")
        if outcome.notes:
            actions.append("note")
        if outcome.reminders:
            actions.append("reminder")
        detail = outcome.reason or (
            outcome.notes[0] if outcome.notes else
            outcome.reminders[0] if outcome.reminders else "参数已修改")
        emit(self._event_handler, HookTriggered(
            self._turn_id, hook.name, event, tool, "+".join(actions),
            detail[:200]))
        if self._rollout is not None:
            self._rollout.append("hook_triggered", {
                "hook": hook.name,
                "event": event,
                "tool": tool,
                "actions": actions,
                "detail": detail[:400],
            }, turn_id=self._turn_id)


def build_hook_bus(settings: Settings, task_dir: Path, *,
                   turn_id: str = "",
                   event_handler: Callable[[object], None] | None = None,
                   rollout=None) -> HookBus:
    """内置 hook 始终注册；hooks_enabled 只控制声明式 hooks.json。"""
    hooks: list[BaseHook] = [
        RepeatedCallHook(settings.repeat_warn_limit),
        SearchBudgetHook(settings.search_warn_limit),
        EmptyOutputHook(settings.empty_output_warn_limit),
    ]
    errors: list[str] = []
    if settings.hooks_enabled:
        path = Path(settings.hooks_file)
        if not path.is_absolute():
            path = task_dir / path
        try:
            hooks.extend(load_config_hooks(path))
        except HookConfigError as exc:
            errors.append(str(exc))
    return HookBus(hooks, turn_id=turn_id, event_handler=event_handler,
                   rollout=rollout, errors=errors)
