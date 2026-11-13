"""TUI 小部件：多行输入框、轮次卡片、工具行与权限弹窗。"""

from __future__ import annotations

import time
from typing import Callable

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Markdown, Static, TextArea

from nextcoder.runtime.events import (
    ToolFinished,
    ToolStarted,
    TurnFinished,
)
from nextcoder.runtime.permissions import PermissionRequest

_STATUS_STYLE = {
    "completed": ("✓", "#7bba55"),
    "failed": ("×", "#c85f5f"),
    "cancelled": ("×", "#b28443"),
}


class ComposerTextArea(TextArea):
    """Enter 发送、Shift+Enter 换行；补全可见时用 Enter/Tab 接受候选。

    没有补全且输入仍是单行时，上/下键浏览历史；多行输入保留正常光标移动。
    """

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class SuggestionMove(Message):
        def __init__(self, delta: int) -> None:
            super().__init__()
            self.delta = delta

    class SuggestionAccept(Message):
        pass

    class SuggestionDismiss(Message):
        pass

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.suggestion_active = False
        self.submit_history: list[str] = []
        self._history_index: int | None = None
        self._draft = ""

    async def _on_key(self, event) -> None:
        key = event.key
        if key == "enter":
            event.prevent_default()
            event.stop()
            if self.suggestion_active:
                self.post_message(self.SuggestionAccept())
            else:
                self.post_message(self.Submitted(self.text))
            return
        if key == "shift+enter":
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        if key == "tab" and self.suggestion_active:
            event.prevent_default()
            event.stop()
            self.post_message(self.SuggestionAccept())
            return
        if key in {"up", "down"} and self.suggestion_active:
            event.prevent_default()
            event.stop()
            self.post_message(self.SuggestionMove(-1 if key == "up" else 1))
            return
        if key == "escape" and self.suggestion_active:
            event.prevent_default()
            event.stop()
            self.post_message(self.SuggestionDismiss())
            return
        if key in {"up", "down"} and "\n" not in self.text:
            event.prevent_default()
            event.stop()
            self._history_move(-1 if key == "up" else 1)
            return
        await super()._on_key(event)

    def _history_move(self, delta: int) -> None:
        if not self.submit_history:
            return
        if self._history_index is None:
            if delta > 0:
                return
            self._draft = self.text
            self._history_index = len(self.submit_history) - 1
        else:
            index = self._history_index + delta
            if index >= len(self.submit_history):
                self._history_index = None
                self.set_text(self._draft)
                return
            self._history_index = max(0, index)
        self.set_text(self.submit_history[self._history_index])

    def add_history(self, text: str) -> None:
        value = text.strip()
        if not value:
            return
        if not self.submit_history or self.submit_history[-1] != value:
            self.submit_history.append(value)
        self._history_index = None
        self._draft = ""

    def set_text(self, value: str) -> None:
        self.text = value
        lines = value.split("\n")
        self.move_cursor((len(lines) - 1, len(lines[-1])))


class RoundCard(Vertical):
    """一轮用户请求的视觉容器；左侧竖线把多轮清楚分开。

    不把 header 作为构造参数传给 Vertical：子组件在卡片挂载完成前排队，
    会和紧随其后的用户消息抢顺序，导致轮次标题跑到消息下面。先挂卡片，
    再按 header、user、assistant、footer 的顺序 mount。
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(classes="round-card", **kwargs)


def cache_label(cache: dict | None, *, compact: bool = False) -> str:
    """把缓存统计转成一句人话。

    第一轮第一次请求天然没有前缀可命中；这时显示“首次”而不是刺眼的 0%。
    同一轮后续调用应接近全命中，因此总命中率低于 90% 但暖命中率高于 90%
    时，把暖命中率也显示出来，避免把一次冷启动误读成缓存失效。
    """
    if not isinstance(cache, dict) or not cache.get("input_tokens"):
        return ""
    calls = _as_int(cache.get("calls"))
    hit = cache.get("hit_rate")
    warm = cache.get("warm_hit_rate")
    if calls <= 0:
        # 旧报告或测试替身没有逐次调用明细，按总命中率显示。
        return f"缓存 {hit:.0%}" if isinstance(hit, (int, float)) else ""
    if calls == 1:
        if _as_int(cache.get("cached_tokens")) == 0:
            return "缓存 首次"
        if isinstance(hit, (int, float)):
            return f"缓存 {hit:.0%}"
        return ""
    if not isinstance(hit, (int, float)):
        return ""
    if hit >= 0.9:
        return f"缓存 {hit:.0%}"
    if isinstance(warm, (int, float)):
        if compact:
            return f"缓存 {warm:.0%}（暖）" if warm >= 0.9 else f"缓存 {hit:.0%}"
        return f"缓存 {hit:.0%}（暖 {warm:.0%}）"
    return f"缓存 {hit:.0%}"


class RoundView:
    """把一轮事件落到卡片里；同时维护可导出的纯文本。"""

    def __init__(self, transcript: VerticalScroll, number: int, model: str,
                 mode: str, on_plain: Callable[[str], None],
                 timestamp: str | None = None) -> None:
        self.transcript = transcript
        self.number = number
        self.model = model
        self.mode = mode
        self.on_plain = on_plain
        self.started = time.monotonic()
        self.timestamp = timestamp or time.strftime("%H:%M:%S")
        self._assistant_text = ""
        self._markdown: Markdown | None = None
        self._last_markdown_update = 0.0
        self._tools: dict[str, tuple[Static, float, str, str]] = {}
        self.header = Static(self._header_text("●", "运行中"),
                             classes="round-header")
        self.card = RoundCard()
        transcript.mount(self.card)
        self.card.mount(self.header)

    def _header_text(self, icon: str, status: str,
                     status_style: str = "#8a8f98") -> Text:
        text = Text()
        text.append(f"{icon} ROUND {self.number:02d}", style="bold #7bba55")
        text.append(f"  {self.timestamp}", style="#6e6d72")
        text.append(f"  ·  {self.model}", style="#e5c07b")
        text.append(f"  ·  {self.mode}", style="#50b7c2")
        text.append(f"  ·  {status}", style=status_style)
        return text

    def add_user(self, request: str, display: str | None = None) -> None:
        shown = display if display is not None else request
        text = Text()
        text.append("› ", style="bold #50b7c2")
        text.append(shown, style="bold #d7dae0")
        self.card.mount(Static(text, classes="user-message"))
        self.on_plain(f"> {request}")

    def add_delta(self, text: str) -> None:
        self._assistant_text += text
        if self._markdown is None:
            # 用当前累积文本创建，而不是先建空 Markdown 再 update：历史回放
            # 时 update 可能发生在 widget 挂载前，空文档会覆盖掉这次更新。
            self._markdown = Markdown(self._assistant_text, open_links=False,
                                      classes="assistant-message")
            self.card.mount(self._markdown)
            self._last_markdown_update = time.monotonic()
            return
        now = time.monotonic()
        if now - self._last_markdown_update >= 0.12 or "\n" in text:
            self._refresh_markdown()
            self._last_markdown_update = now

    def _refresh_markdown(self) -> None:
        if self._markdown is None:
            return
        try:
            self._markdown.update(self._assistant_text)
        except Exception:  # noqa: BLE001 - 不完整 Markdown 不应中断一轮
            return

    def finalize_stream(self) -> None:
        if self._markdown is None:
            return
        self._refresh_markdown()
        if self._assistant_text.strip():
            self.on_plain(self._assistant_text.strip())
        self._markdown = None
        self._assistant_text = ""

    def add_tool_started(self, event: ToolStarted) -> None:
        self.finalize_stream()
        detail = _short(event.detail, 90)
        text = Text()
        text.append("  … ", style="#b28443")
        text.append(event.tool, style="bold #e5c07b")
        if detail:
            text.append(f"  {detail}", style="#8a8f98")
        widget = Static(text, classes="tool-line")
        self.card.mount(widget)
        self._tools[event.call_id] = (widget, time.monotonic(), event.tool, detail)
        self.on_plain(f"[工具] {event.tool} {detail}".rstrip())

    def add_tool_finished(self, event: ToolFinished) -> None:
        started = self._tools.pop(event.call_id, None)
        detail = _short(event.text, 90)
        if started is None:
            self.add_tool_started(ToolStarted(
                turn_id=event.turn_id, call_id=event.call_id,
                tool=event.tool, detail=detail))
            started = self._tools.pop(event.call_id, None)
        elapsed_ms = 0
        tool = event.tool
        if started is not None:
            widget, began, tool, old_detail = started
            elapsed_ms = int((time.monotonic() - began) * 1000)
            detail = detail or old_detail
        else:
            widget = Static("", classes="tool-line")
            self.card.mount(widget)
        icon, style = _STATUS_STYLE.get(event.status, ("•", "#8a8f98"))
        text = Text()
        text.append(f"  {icon} ", style=style)
        text.append(tool, style="bold #cfd1d6")
        if detail:
            text.append(f"  {detail}", style="#8a8f98")
        if elapsed_ms:
            text.append(f"  {elapsed_ms}ms", style="#5f636d")
        widget.update(text)
        self.on_plain(f"  {icon} {tool} {detail}".rstrip())

    def add_note(self, text: str, css_class: str = "system-note") -> None:
        self.card.mount(Static(Text(text), classes=css_class))
        self.on_plain(text)

    def finish(self, event: TurnFinished,
               elapsed: float | None = None) -> None:
        self.finalize_stream()
        cancelled = "取消" in event.stop_reason
        failed = event.stop_reason.startswith("error") or "失败" in event.stop_reason
        if failed:
            icon, label, style = "!", "失败", "#c85f5f"
        elif cancelled:
            icon, label, style = "×", "已取消", "#b28443"
        else:
            icon, label, style = "✓", "完成", "#7bba55"
        self.header.update(self._header_text(icon, label, style))

        usage = event.usage or {}
        input_tokens = _as_int(usage.get("input_tokens"))
        output_tokens = _as_int(usage.get("output_tokens"))
        cached = _as_int(usage.get("cached_tokens"))
        parts = [f"{icon} {label}", f"{event.model_calls} 次模型调用"]
        if input_tokens or output_tokens:
            parts.append(f"{_tokens(input_tokens)} in / {_tokens(output_tokens)} out")
        label = cache_label(event.cache)
        if label:
            parts.append(label)
        elif cached:
            parts.append(f"缓存 {_tokens(cached)}")
        if elapsed is not None:
            parts.append(f"{elapsed:.1f}s")
        footer = "  ·  ".join(parts)
        self.card.mount(Static(Text("  " + footer, style=style),
                               classes="round-footer"))
        self.on_plain(f"  {footer}")


def _short(text: str, limit: int) -> str:
    value = " ".join((text or "").split())
    return value[:limit]


def _as_int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _tokens(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return str(value)


class PermissionScreen(ModalScreen[str]):
    """阻塞式权限确认；返回 allow_once / allow_session / deny。"""

    BINDINGS = [
        ("escape", "deny", "拒绝"),
        ("a", "allow_once", "允许一次"),
        ("s", "allow_session", "本会话允许"),
        ("d", "deny", "拒绝"),
    ]

    def __init__(self, request: PermissionRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        request = self.request
        text = Text()
        text.append("● 权限确认\n", style="bold #e5c07b")
        text.append("工具  ", style="#8a8f98")
        text.append(f"{request.tool}\n", style="bold #d7dae0")
        if request.action and request.action != request.tool:
            text.append("动作  ", style="#8a8f98")
            text.append(f"{request.action}\n", style="#d7dae0")
        text.append("目标  ", style="#8a8f98")
        text.append(f"{request.target}\n", style="#50b7c2")
        if request.reason:
            text.append("原因  ", style="#8a8f98")
            text.append(f"{request.reason}\n", style="#b28443")
        if request.hint:
            text.append("建议  ", style="#8a8f98")
            text.append(f"{request.hint}\n", style="#8a8f98")
        text.append("\nEsc 拒绝 · A 允许一次 · S 本会话允许", style="#8a8f98")
        with Vertical(id="permission-dialog"):
            yield Static(text, id="permission-text")
            with Horizontal(id="permission-buttons"):
                yield Button("✓ 允许一次", id="allow-once", variant="primary")
                yield Button("● 本会话允许", id="allow-session")
                yield Button("× 拒绝", id="deny", variant="error")

    def on_mount(self) -> None:
        # 默认焦点放在拒绝上，避免误按 Enter 执行敏感操作。
        self.query_one("#deny", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        answers = {
            "allow-once": "allow_once",
            "allow-session": "allow_session",
            "deny": "deny",
        }
        self.dismiss(answers.get(event.button.id or "", "deny"))

    def action_allow_once(self) -> None:
        self.dismiss("allow_once")

    def action_allow_session(self) -> None:
        self.dismiss("allow_session")

    def action_deny(self) -> None:
        self.dismiss("deny")
