"""Textual 交互界面：启动页、配置、Markdown 流式输出与权限控制。

TUI 只是事件消费者：模型循环、工具执行、压缩和子 agent 都在 runtime 中；
这里负责渲染，并通过回调把权限答案送回工具层。
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future
from pathlib import Path

from rich.table import Table as RichTable
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Footer, Static

from nextcoder.runtime.agent import SYSTEM_PROMPT_INTERACTIVE
from nextcoder.runtime.events import (
    AssistantDelta,
    ChildUpdated,
    Compaction,
    HookTriggered,
    PermissionAnswered,
    PermissionRequested,
    PlanUpdated,
    ToolFinished,
    ToolStarted,
    TurnFinished,
)
from nextcoder.runtime.planning import PlanStore
from nextcoder.sessions.manager import (
    Session,
    compact_session,
    create_session,
    list_sessions,
    load_session,
    make_session_title,
    run_turn,
    sessions_root,
)
from nextcoder.sessions.rollout import read_events
from nextcoder.settings import Settings
from nextcoder.tools.commands import bwrap_status
from nextcoder.tui.completion import (
    apply_completion,
    build_file_index,
    command_token,
    file_token,
    match_commands,
    match_files,
)
from nextcoder.tui.screens import (
    MODE_INFO,
    MODE_STYLES,
    ConfigScreen,
    HelpScreen,
    PermissionModeScreen,
    SessionPickerScreen,
    SessionSummary,
    WelcomeScreen,
)
from nextcoder.tui.widgets import (
    ComposerTextArea,
    PermissionScreen,
    RoundView,
    cache_label,
)

_STATUS_MARKS = {
    "pending": "[ ]", "in_progress": "[~]",
    "completed": "[x]", "cancelled": "[-]",
}
_STOP_LABELS = {
    "model_finished": "完成",
    "cancelled": "已取消",
    "max_model_calls": "达到调用上限",
    "max_wall_time": "达到时间上限",
}


def _stop_label(reason: str) -> str:
    """把运行时 stop_reason 转成界面用语；已经是中文的原因原样保留。"""
    value = reason or "model_finished"
    if value in _STOP_LABELS:
        return _STOP_LABELS[value]
    if value.startswith("error"):
        return "失败"
    return value


_MODE_MARKS = {name: info[0] for name, info in MODE_INFO.items()}
_MODE_NAMES = {name: info[1] for name, info in MODE_INFO.items()}

_COMPOSER_HINT = (
    "Enter 发送 · Shift+Enter 换行 · / 命令 · @ 文件 · ↑↓ 历史 · Ctrl-C 停止/退出")
_SPINNER = "·•●"


class NextCoderApp(App):
    """一次会话的完整交互界面；同一 task_dir 的后续会话可复用进程。"""

    TITLE = "NextCoder"
    CSS_PATH = "theme.tcss"
    # Footer 已经承载了主要快捷键；Textual 的 palette 入口会把底部挤出屏幕。
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("ctrl+c", "cancel", "停止/退出", priority=True),
        Binding("ctrl+q", "quit", "退出", priority=True),
        Binding("f1", "help", "帮助"),
        Binding("f2", "config", "配置"),
        Binding("f3", "mode", "权限模式"),
        Binding("f4", "sessions", "会话"),
        Binding("ctrl+n", "new_session", "新会话"),
    ]

    def __init__(self, settings: Settings, session: Session | None = None,
                 task_dir: Path | None = None) -> None:
        super().__init__()
        if session is None and task_dir is None:
            raise ValueError("NextCoderApp 需要 session 或 task_dir")
        self.settings = settings
        self.session = session
        self.task_dir = Path(session.task_dir if session is not None
                             else task_dir).resolve()
        self.transcript_text = ""
        self.export_text = ""
        self.plan_text = ""
        self.activity_text = ""
        self.status_text = ""
        self.permission_answer = ""
        self.last_cache: dict = {}
        self.last_usage: dict = {}
        self.last_model_calls = 0
        self._busy = False
        self._compacting = False
        self._session_opening = False
        self._cancel_event: threading.Event | None = None
        self._permission_future: Future | None = None
        self._round: RoundView | None = None
        self._round_no = 0
        self._round_finished = False
        self._run_dir: Path | None = None
        self._file_index: list[str] | None = None
        self._file_index_at = 0.0
        self._suggestion_values: list[tuple[str, str, str]] = []
        self._suggestion_index = 0
        self._suggestion_kind = ""
        self._plan_visible = True
        self._spinner_index = 0
        self._shutting_down = False
        self.theme = "textual-dark"

    # ---- 布局 ----

    def compose(self) -> ComposeResult:
        brand = Text()
        brand.append("◆ NextCoder", style="bold #7bba55")
        # 同一边的 dock 兄弟会互相覆盖：品牌栏和状态栏、输入区和 Footer
        # 都必须各自放进一个 dock 容器，否则会有一行被压掉。
        with Vertical(id="header-dock"):
            yield Static(brand, id="brand")
            yield Static(Text(""), id="status")
        yield VerticalScroll(id="transcript")
        yield Static(Text(""), id="plan")
        yield Static(Text(""), id="activity")
        with Vertical(id="bottom-dock"):
            yield Static(Text(""), id="suggestions")
            with Vertical(id="composer"):
                yield ComposerTextArea(id="composer-area", soft_wrap=True)
                yield Static(Text(_COMPOSER_HINT, style="#5f636d"),
                             id="composer-hint")
            yield Footer()

    def on_mount(self) -> None:
        self.set_interval(0.12, self._tick_spinner)
        if self.session is not None:
            self._activate_session(self.session, initial=True)
        else:
            self._set_composer_enabled(False, "选择或新建会话后即可输入")
            self._refresh_status()
            self._refresh_plan()
            self._set_activity("选择会话")
            self._open_welcome()

    def on_unmount(self) -> None:
        self._shutting_down = True
        try:
            if self._cancel_event is not None:
                self._cancel_event.set()
        except Exception:  # noqa: BLE001 - 退出清理不能阻塞终端恢复
            pass
        try:
            if self._permission_future is not None:
                if not self._permission_future.done():
                    self._permission_future.set_result("deny")
                self._permission_future = None
        except Exception:  # noqa: BLE001
            pass
        self._close_delegation()

    # ---- 启动页与会话 ----

    def _open_welcome(self) -> None:
        if self._shutting_down:
            return
        self.push_screen(
            WelcomeScreen(self.settings, self.task_dir, self._recent_sessions()),
            self._on_welcome_result)

    def _on_welcome_result(self, result: dict | None) -> None:
        if self._shutting_down or not result:
            return
        action = result.get("action")
        if action == "new":
            self._new_session(from_welcome=True)
        elif action == "load":
            self._load_session(result.get("session_id", ""), from_welcome=True)
        elif action == "config":
            self._open_config(reopen_welcome=True)
        elif action == "mode":
            self._open_mode(reopen_welcome=True)
        elif action == "help":
            self.push_screen(HelpScreen(), lambda _result: self._open_welcome())
        elif action == "quit":
            self.action_quit()

    def _recent_sessions(self) -> list[SessionSummary]:
        summaries: list[SessionSummary] = []
        try:
            metadata = list_sessions(self.settings, self.task_dir)
        except Exception:  # noqa: BLE001 - 启动页不应因坏会话退出
            return summaries
        root = sessions_root(self.settings, self.task_dir)
        for item in metadata[:20]:
            session_id = str(item.get("session_id") or "")
            if not session_id:
                continue
            preview = self._session_preview(root / session_id / "rollout.jsonl")
            title = str(item.get("title") or "") or (
                make_session_title(preview) if preview else "")
            if not title:
                # 还没发过消息的空会话不占启动页位置。
                continue
            created = str(item.get("created_at") or "")
            created = created.replace("T", " ")[5:16] if created else ""
            summaries.append(SessionSummary(
                session_id=session_id,
                created_at=created,
                title=title,
                preview=preview,
            ))
            if len(summaries) >= 8:
                break
        return summaries

    @staticmethod
    def _session_preview(path: Path) -> str:
        try:
            with path.open(encoding="utf-8", errors="replace") as handle:
                for _, line in zip(range(30), handle):
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "user_request":
                        request = str((event.get("payload") or {}).get("request") or "")
                        return " ".join(request.split())[:48]
        except OSError:
            return ""
        return ""

    def _new_session(self, from_welcome: bool = False) -> None:
        if self._busy:
            self._set_activity("当前轮还在运行；先停止或等待结束")
            return
        if self._session_opening:
            self._set_activity("正在打开另一个会话…")
            return
        self._session_opening = True
        self._set_activity("正在创建会话…")
        self._create_worker(from_welcome)

    @work(thread=True)
    def _create_worker(self, from_welcome: bool) -> None:
        """基线复制可能耗时，放到工作线程，避免启动页看起来卡死。"""
        try:
            session = create_session(self.settings, self.task_dir)
        except Exception as exc:  # noqa: BLE001 - 展示失败原因
            self._call_ui(self._session_open_failed,
                          f"新建会话失败：{type(exc).__name__}: {exc}",
                          from_welcome)
            return
        self._call_ui(self._session_ready, session, from_welcome, False)

    def _load_session(self, session_id: str, from_welcome: bool = False) -> None:
        if self._busy:
            self._set_activity("当前轮还在运行；先停止或等待结束")
            return
        if self.session is not None and session_id == self.session.session_id:
            self._append_system("已经是当前会话", "system-note")
            return
        if self._session_opening:
            self._set_activity("正在打开另一个会话…")
            return
        self._session_opening = True
        self._set_activity("正在打开会话…")
        self._load_worker(session_id, from_welcome)

    @work(thread=True)
    def _load_worker(self, session_id: str, from_welcome: bool) -> None:
        try:
            session = load_session(self.settings, self.task_dir, session_id)
        except Exception as exc:  # noqa: BLE001
            self._call_ui(self._session_open_failed,
                          f"打开会话失败：{type(exc).__name__}: {exc}",
                          from_welcome)
            return
        self._call_ui(self._session_ready, session, from_welcome, True)

    def _session_open_failed(self, message: str, from_welcome: bool) -> None:
        self._session_opening = False
        self._append_system(message, "error-line")
        self._set_activity("会话打开失败")
        if from_welcome and self.session is None:
            self._open_welcome()

    def _session_ready(self, session: Session, from_welcome: bool,
                       show_plan: bool) -> None:
        self._session_opening = False
        self._close_delegation()
        self._activate_session(session, show_plan=show_plan)
        self._set_activity("会话已就绪")

    def _activate_session(self, session: Session, initial: bool = False,
                          show_plan: bool = True) -> None:
        self.session = session
        self.task_dir = Path(session.task_dir).resolve()
        self._round = None
        self._round_no = 0
        self._round_finished = False
        self.transcript_text = ""
        self.export_text = ""
        self.last_cache = {}
        self.last_usage = {}
        self.last_model_calls = 0
        self._file_index = None
        try:
            self.query_one("#transcript", VerticalScroll).remove_children()
        except Exception:  # noqa: BLE001 - 首次挂载前可能还没有 transcript
            pass
        self._set_composer_enabled(True, _COMPOSER_HINT)
        self._banner()
        self._replay_history(session)
        if show_plan:
            self._refresh_plan()
        else:
            # 任务清单是 task_dir 级别的状态；新会话先隐藏，等本轮真的产生
            # plan_updated 再显示，避免把上一个会话的旧计划当成当前计划。
            self._render_plan((), 0)
            self._plan_visible = False
            try:
                self.query_one("#plan", Static).display = False
            except Exception:  # noqa: BLE001
                pass
        self._refresh_status()
        self._set_activity("就绪 · /help 查看命令 · @ 引用文件")
        if not initial:
            self._scroll_transcript()
        self.query_one("#composer-area", ComposerTextArea).focus()

    def _close_delegation(self) -> None:
        session = self.session
        if session is None or session.delegation_manager is None:
            return
        try:
            session.delegation_manager.close()
        except Exception:  # noqa: BLE001 - 清理失败不影响切换
            pass
        session.delegation_manager = None

    # ---- 输入、补全与命令 ----

    @on(ComposerTextArea.Changed)
    def _on_composer_changed(self, event: ComposerTextArea.Changed) -> None:
        self._update_suggestions(event.text_area.text)

    @on(ComposerTextArea.Submitted)
    def _on_composer_submitted(self, event: ComposerTextArea.Submitted) -> None:
        request = event.text.strip()
        if not request:
            return
        composer = self.query_one("#composer-area", ComposerTextArea)
        composer.add_history(request)
        composer.set_text("")
        self._hide_suggestions()
        if self._busy:
            self._set_activity("当前轮还在运行；等待结束或按 Ctrl-C 停止")
            return
        if self.session is None:
            self._set_activity("请先选择或新建会话")
            self._open_welcome()
            return
        if request.startswith("/"):
            self._run_command(request)
            return
        self._start_turn(request)

    @on(ComposerTextArea.SuggestionMove)
    def _on_suggestion_move(self, event: ComposerTextArea.SuggestionMove) -> None:
        if not self._suggestion_values:
            return
        count = len(self._suggestion_values)
        self._suggestion_index = (self._suggestion_index + event.delta) % count
        self._render_suggestions()

    @on(ComposerTextArea.SuggestionAccept)
    def _on_suggestion_accept(self, event: ComposerTextArea.SuggestionAccept) -> None:
        self._accept_suggestion()

    @on(ComposerTextArea.SuggestionDismiss)
    def _on_suggestion_dismiss(self, event: ComposerTextArea.SuggestionDismiss) -> None:
        self._hide_suggestions()

    def _update_suggestions(self, text: str) -> None:
        composer = self.query_one("#composer-area", ComposerTextArea)
        if self._busy or self.session is None:
            self._hide_suggestions()
            return
        if command_token(text) is not None:
            commands = match_commands(text)
            # usage 与 description 用制表符分隔，渲染时放进两列表格；
            # 直接用空格对齐会被 Rich 的换行逻辑折叠，长命令会挤掉说明。
            self._suggestion_values = [
                (command.name, f"{command.usage}\t{command.description}",
                 "command")
                for command in commands
            ]
            self._suggestion_kind = "command"
        else:
            token = file_token(text)
            if token is None:
                self._hide_suggestions()
                return
            prefix = token[1]
            paths = self._file_paths()
            matches = match_files(paths, prefix)
            self._suggestion_values = [
                (path, f"@{path}", "file") for path in matches
            ]
            self._suggestion_kind = "file"
        if not self._suggestion_values:
            self._hide_suggestions()
            return
        self._suggestion_index = 0
        composer.suggestion_active = True
        self._render_suggestions()

    def _render_suggestions(self) -> None:
        widget = self.query_one("#suggestions", Static)
        if not self._suggestion_values:
            widget.display = False
            widget.update(Text(""))
            self.query_one("#composer-area", ComposerTextArea).suggestion_active = False
            return
        widget.display = True
        table = RichTable.grid(padding=(0, 1), expand=False)
        table.add_column(width=1, no_wrap=True)
        table.add_column(no_wrap=True)
        table.add_column(no_wrap=True, overflow="ellipsis")
        for index, (_value, label, kind) in enumerate(self._suggestion_values):
            selected = index == self._suggestion_index
            if kind == "command":
                usage, _, description = label.partition("\t")
            else:
                usage, description = label, ""
            style = "bold #7bba55" if selected else (
                "#50b7c2" if kind == "command" else "#8a8f98")
            table.add_row(
                Text("›" if selected else " ", style=style),
                Text(usage, style=style),
                Text(description, style=(
                    "bold #7bba55" if selected else "#5f636d")),
            )
        widget.update(table)

    def _accept_suggestion(self) -> None:
        if not self._suggestion_values:
            return
        value, _label, kind = self._suggestion_values[self._suggestion_index]
        composer = self.query_one("#composer-area", ComposerTextArea)
        new_text = apply_completion(
            composer.text, value, command=kind == "command")
        composer.set_text(new_text)
        self._hide_suggestions()
        composer.focus()

    def _hide_suggestions(self) -> None:
        self._suggestion_values = []
        self._suggestion_index = 0
        self._suggestion_kind = ""
        try:
            widget = self.query_one("#suggestions", Static)
            widget.update(Text(""))
            widget.display = False
            self.query_one("#composer-area", ComposerTextArea).suggestion_active = False
        except Exception:  # noqa: BLE001 - 退出过程中小部件可能已卸载
            pass

    def _file_paths(self) -> list[str]:
        now = time.monotonic()
        if self._file_index is None or now - self._file_index_at > 5:
            self._file_index = build_file_index(self.task_dir)
            self._file_index_at = now
        return self._file_index

    def _run_command(self, text: str) -> None:
        name, _, argument = text.partition(" ")
        command = name.casefold()
        argument = argument.strip()
        if command == "/help":
            self.push_screen(HelpScreen())
        elif command == "/new":
            self._new_session()
        elif command == "/sessions":
            self._open_sessions()
        elif command in {"/config", "/model"}:
            self._open_config()
        elif command == "/mode":
            if not argument:
                self._open_mode()
            elif argument in MODE_INFO:
                self._set_mode(argument)
            else:
                self._append_system(
                    "用法：/mode manual|workspace|auto|bypass —— "
                    "逐项确认 / 工作区写入 / 沙箱内自动 / 完全放行",
                    "error-line")
        elif command == "/sandbox":
            self._sandbox_command(argument)
        elif command == "/plan":
            self._toggle_plan()
        elif command == "/compact":
            self._compact()
        elif command == "/export":
            self._export(argument)
        elif command == "/clear":
            self._clear_transcript()
        elif command == "/status":
            self._show_status()
        elif command == "/quit":
            self.action_quit()
        else:
            self._append_system(
                f"未知命令：{name}。输入 /help 查看可用命令。", "error-line")

    # ---- 配置与权限模式 ----

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_config(self) -> None:
        from_welcome = isinstance(self.screen, WelcomeScreen)
        if from_welcome:
            self.pop_screen()
        self._open_config(reopen_welcome=from_welcome)

    def _open_config(self, reopen_welcome: bool = False) -> None:
        if self._busy:
            self._set_activity("当前轮还在运行；先停止或等待结束再改配置")
            return

        def done(result: Settings | None) -> None:
            if result is not None:
                self._apply_settings(result)
            if reopen_welcome:
                self._open_welcome()

        self.push_screen(ConfigScreen(self.settings), done)

    def _apply_settings(self, settings: Settings) -> None:
        self.settings = settings
        self._refresh_status()
        mode = _MODE_NAMES.get(settings.permission_mode,
                               settings.permission_mode)
        sandbox = "开" if settings.command_sandbox else "关"
        note = (f"配置已更新：{settings.model_profile or 'custom'} / "
                f"{settings.llm_model} · 权限 {mode} · 沙箱 {sandbox} · "
                f"下一轮生效")
        self._append_system(note, "success-line")

    def action_mode(self) -> None:
        self._open_mode()

    def _open_mode(self, reopen_welcome: bool = False) -> None:
        if self._busy:
            self._set_activity("当前轮还在运行；先停止或等待结束")
            return

        def done(mode: str | None) -> None:
            if mode:
                self._set_mode(mode)
            if reopen_welcome:
                self._open_welcome()

        self.push_screen(
            PermissionModeScreen(self.settings.permission_mode), done)

    def _set_mode(self, mode: str) -> None:
        if mode not in MODE_INFO:
            self._append_system(f"未知权限模式：{mode}", "error-line")
            return
        self.settings = self.settings.model_copy(
            update={"permission_mode": mode})
        self._refresh_status()
        info = MODE_INFO[mode]
        self._append_system(
            f"权限模式已切换为 {info[1]} · 下一轮生效",
            "success-line")

    def _sandbox_command(self, argument: str) -> None:
        parts = argument.split()
        if not parts or parts[0] == "status":
            self._show_sandbox()
            return
        action = parts[0].casefold()
        if action in {"on", "off"}:
            enabled = action == "on"
            self._apply_settings(self.settings.model_copy(
                update={"command_sandbox": enabled}))
            state, _path, detail = bwrap_status()
            if enabled and state != "ok":
                reason = "未找到 bwrap" if state == "missing" else detail
                self._append_system(
                    f"警告：沙箱不可用（{reason}）；开启 fail-closed 时命令会被拒绝。",
                    "error-line")
            return
        if action in {"network", "strict"} and len(parts) > 1:
            enabled = parts[1].casefold() == "on"
            key = ("command_sandbox_network" if action == "network"
                   else "command_sandbox_fail_closed")
            self._apply_settings(self.settings.model_copy(update={key: enabled}))
            return
        self._append_system(
            "用法：/sandbox [on|off|network on|off|strict on|off|status]",
            "error-line")

    def _show_sandbox(self) -> None:
        settings = self.settings
        state, path, detail = bwrap_status()
        bwrap_line = path or "未找到"
        if state == "ok":
            bwrap_line += "（可用）"
        elif state == "unusable":
            bwrap_line += f"（不可用：{detail}）"
        lines = [
            "沙箱状态",
            f"  总开关      {'on' if settings.command_sandbox else 'off'}",
            f"  fail-closed {'on' if settings.command_sandbox_fail_closed else 'off'}",
            f"  网络        {'允许' if settings.command_sandbox_network else '禁止'}",
            f"  bwrap       {bwrap_line}",
            f"  writable    {settings.sandbox_writable_roots or '任务目录'}",
            f"  readonly    {settings.sandbox_readonly_roots or '（未配置）'}",
            "  切换：/sandbox on|off · /sandbox network on|off",
        ]
        self._append_system("\n".join(lines), "system-note")

    def _show_status(self) -> None:
        session = self.session
        if session is None:
            self._append_system("当前没有会话", "error-line")
            return
        plan = PlanStore(self.task_dir / self.settings.runs_root
                         / "task-plan.json").load()
        thinking = self.settings.llm_thinking
        if self.settings.llm_reasoning_effort:
            thinking += f" / {self.settings.llm_reasoning_effort}"
        lines = [
            "运行状态",
            f"  session     {session.session_id}",
            f"  task_dir    {self.task_dir}",
            f"  profile     {self.settings.model_profile or 'custom'}",
            f"  model       {self.settings.llm_model}",
            f"  endpoint    {self.settings.llm_base_url}",
            f"  thinking    {thinking}",
            f"  permission  {_MODE_NAMES.get(self.settings.permission_mode, self.settings.permission_mode)}",
            f"  sandbox     {'on' if self.settings.command_sandbox else 'off'}"
            f" · network {'on' if self.settings.command_sandbox_network else 'off'}"
            f" · fail-closed {'on' if self.settings.command_sandbox_fail_closed else 'off'}",
            f"  plan        v{plan.revision} · {len(plan.tasks)} 项",
            f"  last turn   {self.last_model_calls} 次模型调用 · "
            f"{self.last_usage.get('input_tokens', 0)} in / "
            f"{self.last_usage.get('output_tokens', 0)} out",
        ]
        self._append_system("\n".join(lines), "system-note")

    # ---- 会话列表 ----

    def action_sessions(self) -> None:
        self._open_sessions()

    def _open_sessions(self) -> None:
        if self._busy:
            self._set_activity("当前轮还在运行；先停止或等待结束")
            return
        self.push_screen(SessionPickerScreen(self._recent_sessions()),
                         self._on_session_result)

    def _on_session_result(self, result: dict | None) -> None:
        if not result:
            return
        action = result.get("action")
        if action == "new":
            self._new_session()
        elif action == "load":
            self._load_session(result.get("session_id", ""))

    def action_new_session(self) -> None:
        self._new_session()

    # ---- 一轮运行 ----

    def _start_turn(self, request: str) -> None:
        if self.session is None:
            return
        self._busy = True
        self._compacting = False
        self._cancel_event = threading.Event()
        self._round_no += 1
        model = (f"{self.settings.model_profile}/{self.settings.llm_model}"
                 if self.settings.model_profile else self.settings.llm_model)
        self._round = RoundView(
            self.query_one("#transcript", VerticalScroll),
            self._round_no, model,
            _MODE_NAMES.get(self.settings.permission_mode,
                            self.settings.permission_mode),
            self._append_plain)
        self._round.add_user(request)
        self._round_finished = False
        self._set_composer_enabled(False, "运行中 · Ctrl-C 停止")
        self._set_activity("模型生成中…")
        self._refresh_status()
        self._scroll_transcript()
        self._turn_worker(request, self._cancel_event)

    @work(thread=True)
    def _turn_worker(self, request: str, cancel_event: threading.Event) -> None:
        try:
            outcome = run_turn(
                self.settings, self.session, request,
                system_prompt=SYSTEM_PROMPT_INTERACTIVE,
                approval_callback=self._permission_callback,
                event_handler=self._event_from_worker,
                streaming=True,
                cancel_event=cancel_event,
            )
            self._call_ui(self._finish_turn, outcome)
        except Exception as exc:  # noqa: BLE001 - 失败必须显示而不是退出
            self._call_ui(self._fail_turn, f"{type(exc).__name__}: {exc}")

    def _call_ui(self, callback, *args) -> None:
        if self._shutting_down:
            return
        try:
            self.call_from_thread(callback, *args)
        except RuntimeError:
            return

    def _event_from_worker(self, event: object) -> None:
        self._call_ui(self._handle_event, event)

    def _handle_event(self, event: object) -> None:
        round_view = self._round
        if isinstance(event, AssistantDelta):
            if round_view is not None:
                round_view.add_delta(event.text)
        elif isinstance(event, ToolStarted):
            if round_view is not None:
                round_view.add_tool_started(event)
            self._set_activity(f"执行工具：{event.tool}")
        elif isinstance(event, ToolFinished):
            if round_view is not None:
                round_view.add_tool_finished(event)
            self._set_activity(f"工具完成：{event.tool}")
        elif isinstance(event, PlanUpdated):
            self._render_plan(event.tasks, event.revision)
            self._set_activity(f"任务清单 v{event.revision}")
        elif isinstance(event, Compaction):
            note = (f"上下文压缩：{event.before_tokens} → {event.after_tokens} "
                    f"tokens（{event.method}）")
            if round_view is not None:
                round_view.add_note(note)
            else:
                self._append_system(note, "system-note")
            self._set_activity(note)
        elif isinstance(event, PermissionRequested):
            if round_view is not None:
                round_view.add_note(
                    f"● 等待权限：{event.tool} {event.action} {event.target}",
                    "system-note")
            self._set_activity(f"等待权限确认：{event.tool} {event.action}")
        elif isinstance(event, PermissionAnswered):
            if round_view is not None:
                round_view.add_note(f"权限结果：{event.answer}", "system-note")
            self._set_activity(f"权限结果：{event.answer}")
        elif isinstance(event, HookTriggered):
            if round_view is not None:
                round_view.add_note(
                    f"» hook {event.hook} {event.action} · {event.tool}",
                    "system-note")
            self._set_activity(f"hook：{event.hook}（{event.action}）")
        elif isinstance(event, ChildUpdated):
            if round_view is not None:
                round_view.add_note(
                    f"» 子 agent {event.role}（{event.child_id}）：{event.status}",
                    "system-note")
            self._set_activity(
                f"子 agent {event.role}（{event.child_id}）：{event.status}")
            self._refresh_status()
        elif isinstance(event, TurnFinished):
            self.last_model_calls = event.model_calls
            self.last_cache = dict(event.cache)
            self.last_usage = dict(event.usage)
            if round_view is not None:
                round_view.finish(
                    event, elapsed=time.monotonic() - round_view.started)
                self._round_finished = True
            self._refresh_status()

    def _finish_turn(self, outcome) -> None:
        if self._round is not None and not self._round_finished:
            state = getattr(outcome, "final_state", {}) or {}
            self._round.finish(TurnFinished(
                turn_id=getattr(outcome, "turn_id", ""),
                stop_reason=getattr(outcome, "stop_reason", "model_finished"),
                model_calls=int(state.get("model_calls", 0) or 0),
                usage=state.get("usage", {}) or {},
                cache=self.last_cache,
            ), elapsed=time.monotonic() - self._round.started)
            self._round_finished = True
        self._busy = False
        self._compacting = False
        self._cancel_event = None
        self._run_dir = getattr(outcome, "run_dir", None)
        self._round = None
        self._file_index = None
        stop_reason = getattr(outcome, "stop_reason", "model_finished")
        self._set_activity(f"本轮{_stop_label(stop_reason)}")
        self._set_composer_enabled(True, _COMPOSER_HINT)
        self._refresh_status()
        self._scroll_transcript()

    def _fail_turn(self, message: str) -> None:
        if self._round is not None and not self._round_finished:
            self._round.finalize_stream()
            self._round.add_note(f"运行失败：{message}", "error-line")
        self._busy = False
        self._compacting = False
        self._cancel_event = None
        self._round = None
        self._file_index = None
        self._append_system(f"运行失败：{message}", "error-line")
        self._set_activity("运行失败，可修改输入后重试")
        self._set_composer_enabled(True, _COMPOSER_HINT)
        self._refresh_status()
        self._scroll_transcript()

    # ---- 压缩 ----

    def _compact(self) -> None:
        if self._busy:
            self._set_activity("当前轮还在运行；先停止或等待结束")
            return
        if self.session is None:
            self._append_system("当前没有会话", "error-line")
            return
        self._busy = True
        self._compacting = True
        self._cancel_event = None
        self._set_composer_enabled(False, "压缩上下文中…")
        self._set_activity("压缩上下文中…")
        self._refresh_status()
        self._compact_worker()

    @work(thread=True)
    def _compact_worker(self) -> None:
        try:
            result = compact_session(self.settings, self.session)
            self._call_ui(self._compact_finished, result)
        except Exception as exc:  # noqa: BLE001
            self._call_ui(self._compact_failed, f"{type(exc).__name__}: {exc}")

    def _compact_finished(self, result: dict) -> None:
        self._busy = False
        self._compacting = False
        before = result.get("before_tokens", "?")
        after = result.get("after_tokens", "?")
        self._append_system(
            f"压缩完成：{before} → {after} tokens · "
            f"覆盖 {result.get('covered_count', '?')} 条消息",
            "success-line")
        self._set_activity("压缩完成")
        self._set_composer_enabled(True, _COMPOSER_HINT)
        self._refresh_status()

    def _compact_failed(self, message: str) -> None:
        self._busy = False
        self._compacting = False
        self._append_system(f"压缩失败：{message}", "error-line")
        self._set_activity("压缩失败")
        self._set_composer_enabled(True, _COMPOSER_HINT)
        self._refresh_status()

    # ---- 权限 ----

    def _permission_callback(self, request) -> str:
        if self._shutting_down:
            return "deny"
        future: Future = Future()
        self._call_ui(self._push_permission, request, future)
        try:
            answer = future.result(timeout=600)
        except Exception:  # noqa: BLE001 - 超时/退出一律拒绝
            return "deny"
        self.permission_answer = str(answer)
        return self.permission_answer

    def _push_permission(self, request, future: Future) -> None:
        if self._shutting_down:
            future.set_result("deny")
            return
        self._permission_future = future
        self._set_activity(f"等待权限确认：{request.tool} {request.action}")

        def done(result: str | None) -> None:
            self._permission_future = None
            if not future.done():
                future.set_result(result or "deny")

        self.push_screen(PermissionScreen(request), done)

    # ---- 取消与退出 ----

    def action_cancel(self) -> None:
        if self._busy:
            if self._cancel_event is None:
                self._set_activity("压缩中，请稍候；完成后可继续")
                return
            if not self._cancel_event.is_set():
                self._cancel_event.set()
                self._set_activity("已请求停止；当前命令结束后收尾")
            if self._permission_future is not None:
                if not self._permission_future.done():
                    self._permission_future.set_result("deny")
                self._permission_future = None
                if isinstance(self.screen, PermissionScreen):
                    self.pop_screen()
            return
        if len(self.screen_stack) > 1:
            if isinstance(self.screen, WelcomeScreen):
                self.action_quit()
            else:
                self.pop_screen()
            return
        self.action_quit()

    def action_quit(self) -> None:
        self._shutting_down = True
        if self._cancel_event is not None:
            self._cancel_event.set()
        if self._permission_future is not None:
            if not self._permission_future.done():
                self._permission_future.set_result("deny")
            self._permission_future = None
        self.exit()

    # ---- 渲染 ----

    def _banner(self) -> None:
        mode = _MODE_NAMES.get(self.settings.permission_mode,
                               self.settings.permission_mode)
        text = Text()
        text.append("◆ NextCoder 已就绪\n", style="bold #7bba55")
        text.append("任务目录  ", style="#6e6d72")
        text.append(f"{self.task_dir}\n", style="#cfd1d6")
        text.append("模型      ", style="#6e6d72")
        text.append(f"{self.settings.model_profile or 'custom'} / "
                    f"{self.settings.llm_model}\n", style="#e5c07b")
        text.append("权限      ", style="#6e6d72")
        text.append(mode, style="#50b7c2")
        text.append("    沙箱  ", style="#6e6d72")
        text.append("开" if self.settings.command_sandbox else "关",
                    style="#7bba55" if self.settings.command_sandbox
                    else "#8a8f98")
        text.append("\n输入 /help 查看命令；输入 @ 引用文件；Shift+Enter 换行。",
                    style="#5f636d")
        widget = Static(text, classes="welcome-card")
        self.query_one("#transcript", VerticalScroll).mount(widget)

    def _replay_history(self, session: Session) -> None:
        """从 rollout 还原历史轮次，切换/恢复会话时不再只剩一条启动横幅。

        rollout 的 assistant 事件是内容摘要（工具调用只保留参数摘要），足以
        还原对话；工具状态不重放，避免把历史调用伪装成当前会话的执行结果。
        """
        events = read_events(session.root / "rollout.jsonl")
        if not events:
            return
        current: RoundView | None = None
        turn_id = ""
        for event in events:
            kind = event.get("type")
            payload = event.get("payload") or {}
            if kind == "user_request":
                self._round_no += 1
                current = RoundView(
                    self.query_one("#transcript", VerticalScroll),
                    self._round_no,
                    self.settings.llm_model or "model",
                    _MODE_NAMES.get(self.settings.permission_mode,
                                    self.settings.permission_mode),
                    self._append_plain,
                    timestamp=str(event.get("ts") or "")[11:19] or None,
                )
                turn_id = str(event.get("turn_id") or "")
                current.add_user(str(payload.get("request") or ""))
            elif kind == "assistant" and current is not None:
                content = str(payload.get("content") or "")
                if content:
                    current.add_delta(content)
                current.finalize_stream()
            elif kind == "turn_end" and current is not None:
                try:
                    model_calls = int(payload.get("model_calls") or 0)
                except (TypeError, ValueError):
                    model_calls = 0
                current.finish(TurnFinished(
                    turn_id=turn_id,
                    stop_reason=str(payload.get("stop_reason") or "model_finished"),
                    model_calls=model_calls,
                ))
                current = None
        if current is not None:
            current.finalize_stream()
        self._round = None

    def _append_plain(self, text: str) -> None:
        self.transcript_text += text + "\n"

    def _append_system(self, text: str, css_class: str = "system-note") -> None:
        self.transcript_text += text + "\n"
        widget = Static(Text(text), classes=css_class)
        try:
            self.query_one("#transcript", VerticalScroll).mount(widget)
            self._scroll_transcript()
        except Exception:  # noqa: BLE001
            return

    def _clear_transcript(self) -> None:
        try:
            self.query_one("#transcript", VerticalScroll).remove_children()
        except Exception:  # noqa: BLE001
            return
        self.transcript_text = ""
        self.export_text = ""
        self._round = None
        self._round_no = 0
        self._banner()
        self._append_system("屏幕已清空；会话记录和检查点仍然保留。",
                            "system-note")

    def _toggle_plan(self) -> None:
        self._plan_visible = not self._plan_visible
        self.query_one("#plan", Static).display = self._plan_visible
        self._set_activity("任务清单已" + ("显示" if self._plan_visible else "隐藏"))

    def _export(self, argument: str) -> None:
        if self.session is None:
            self._append_system("当前没有会话", "error-line")
            return
        if argument:
            target = Path(argument).expanduser()
            if not target.is_absolute():
                target = self.task_dir / target
        else:
            target = (self.task_dir / self.settings.runs_root / "exports"
                      / f"{self.session.session_id}-"
                        f"{time.strftime('%Y%m%d-%H%M%S')}.md")
        target = target.resolve()
        try:
            target.relative_to(self.task_dir)
        except ValueError:
            self._append_system("导出路径必须在任务目录内", "error-line")
            return
        header = (
            f"# NextCoder 对话导出\n\n"
            f"- session: `{self.session.session_id}`\n"
            f"- model: `{self.settings.llm_model}`\n"
            f"- task: `{self.task_dir}`\n"
            f"- exported: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(header + self.transcript_text, encoding="utf-8")
        except OSError as exc:
            self._append_system(f"导出失败：{exc}", "error-line")
            return
        self._append_system(f"已导出：{target.relative_to(self.task_dir)}",
                            "success-line")

    def _set_composer_enabled(self, enabled: bool, hint: str) -> None:
        try:
            composer = self.query_one("#composer-area", ComposerTextArea)
            composer.disabled = not enabled
            self.query_one("#composer-hint", Static).update(
                Text(hint, style="#5f636d"))
            if enabled:
                composer.focus()
        except Exception:  # noqa: BLE001
            return

    def _scroll_transcript(self) -> None:
        # 新 footer、任务清单和 Markdown 最终高度都要等布局完成后才算出来。
        # 只调度一次 call_after_refresh 仍可能早于 mount 触发的下一帧，
        # 再补一个短定时器，确保 plan/footer 撑高后仍停在真正的底部。
        try:
            self.call_after_refresh(self._scroll_transcript_now)
            self.set_timer(0.05, self._scroll_transcript_now)
        except Exception:  # noqa: BLE001 - 退出过程中可能已没有活动应用
            return

    def _scroll_transcript_now(self) -> None:
        try:
            self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)
        except Exception:  # noqa: BLE001
            return

    def _set_activity(self, text: str) -> None:
        self.activity_text = text
        self._render_activity()

    def _render_activity(self) -> None:
        try:
            widget = self.query_one("#activity", Static)
        except Exception:  # noqa: BLE001
            return
        if self._busy:
            frame = _SPINNER[self._spinner_index % len(_SPINNER)]
            text = Text()
            text.append(f"{frame} ", style="#7bba55")
            text.append(self.activity_text, style="#b28443")
            widget.update(text)
        else:
            widget.update(Text("● " + self.activity_text, style="#6e6d72"))

    def _tick_spinner(self) -> None:
        if not self._busy:
            return
        self._spinner_index += 1
        self._render_activity()

    def _refresh_status(self) -> None:
        # 品牌栏已经显示产品名，状态栏从模型开始，避免同一行重复。
        parts: list[tuple[str, str]] = []
        if self.session is None:
            parts.append(("未选择会话", "#b28443"))
            parts.append(("F2 配置", "#5f636d"))
        else:
            profile = self.settings.model_profile or "custom"
            model = self.settings.llm_model or "未配置"
            parts.append((f"{profile}/{model}", "#e5c07b"))
            mode = self.settings.permission_mode
            mode_style = MODE_STYLES.get(mode, "#8a8f98")
            parts.append((f"{_MODE_MARKS.get(mode, '●')} "
                          f"{_MODE_NAMES.get(mode, mode)}", mode_style))
            parts.append((self._sandbox_label(), self._sandbox_style()))
            title = self.session.title or "新会话"
            parts.append((f"会话 {title[:14]}", "#50b7c2"))
            parts.append(("运行中" if self._busy else "就绪",
                          "#b28443" if self._busy else "#7bba55"))
            if self.last_model_calls:
                parts.append((f"调用 {self.last_model_calls}", "#8a8f98"))
            label = cache_label(self.last_cache, compact=True)
            if label:
                parts.append((label, "#7bba55"))
        self.status_text = " · ".join(part for part, _style in parts)
        text = Text()
        for index, (part, style) in enumerate(parts):
            if index:
                text.append("  ·  ", style="#30343d")
            text.append(part, style=style)
        try:
            self.query_one("#status", Static).update(text)
        except Exception:  # noqa: BLE001
            pass

    def _sandbox_label(self) -> str:
        settings = self.settings
        if not settings.command_sandbox:
            return "□ 沙箱关"
        state, _path, _detail = bwrap_status()
        if state == "missing":
            return "! 沙箱缺失"
        if state == "unusable":
            return "! 沙箱不可用"
        if settings.command_sandbox_network:
            return "■ 沙箱开·联网"
        return "■ 沙箱开"

    def _sandbox_style(self) -> str:
        if not self.settings.command_sandbox:
            return "#5f636d"
        state, _path, _detail = bwrap_status()
        if state != "ok":
            return "#c85f5f"
        return "#7bba55"

    def _refresh_plan(self) -> None:
        if self.session is None:
            self.plan_text = "任务清单\n（选择会话后显示）"
            try:
                self.query_one("#plan", Static).update(Text(self.plan_text))
            except Exception:  # noqa: BLE001
                pass
            return
        store = PlanStore(
            self.session.task_dir / self.settings.runs_root / "task-plan.json")
        try:
            plan = store.load()
        except Exception:  # noqa: BLE001 - 坏 plan 不应阻塞界面
            self._render_plan((), 0)
            return
        tasks = tuple(
            {"id": task.id, "content": task.content, "status": task.status}
            for task in plan.tasks)
        self._render_plan(tasks, plan.revision)

    def _render_plan(self, tasks, revision: int = 0) -> None:
        text = Text()
        text.append(f"▌ 任务清单 v{revision}", style="bold #7bba55")
        if not tasks:
            text.append("\n（暂无任务；让模型先列计划）", style="#5f636d")
            self.plan_text = f"任务清单 v{revision}\n（暂无任务）"
        else:
            lines = [f"任务清单 v{revision}"]
            for task in tasks:
                status = task.get("status", "pending")
                mark = _STATUS_MARKS.get(status, "[ ]")
                line = f"{mark} {task.get('id', '')} {task.get('content', '')}"
                lines.append(line)
                style = {
                    "completed": "#7bba55",
                    "in_progress": "#e5c07b",
                    "cancelled": "#5f636d",
                }.get(status, "#cfd1d6")
                text.append("\n" + line, style=style)
            self.plan_text = "\n".join(lines)
        try:
            widget = self.query_one("#plan", Static)
            widget.update(text)
            if tasks:
                # 新会话的旧计划面板默认隐藏；本轮产生 plan_updated 后重新显示。
                self._plan_visible = True
                widget.display = True
                self._scroll_transcript()
        except Exception:  # noqa: BLE001
            pass
