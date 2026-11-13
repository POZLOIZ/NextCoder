"""启动页、会话选择、配置、帮助与权限模式选择屏幕。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import SecretStr
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Input,
    Label,
    OptionList,
    Select,
    Static,
    Switch,
)
from textual.widgets.option_list import Option

from nextcoder.model_config import (
    ModelConfigError,
    ModelProfile,
    apply_model_profile,
    load_model_profiles,
)
from nextcoder.settings import Settings
from nextcoder.tools.commands import bwrap_status
from nextcoder.tui.completion import COMMANDS
from nextcoder.tui.welcome_art import welcome_renderable

MODE_INFO: dict[str, tuple[str, str, str]] = {
    "manual": (
        "●", "逐项确认",
        "每次改文件、删文件、运行命令或调用外部工具，都先问你。"
        "读取项目文件不打断。",
    ),
    "workspace": (
        "■", "工作区写入",
        "任务目录里的读取和修改自动放行；运行命令、调用外部工具仍然先问你。",
    ),
    "auto": (
        "▲", "沙箱内自动",
        "任务目录里的读写自动放行；命令在沙箱或容器里自动运行，"
        "离开隔离环境前先问你。",
    ),
    "bypass": (
        "◆", "完全放行",
        "除不可逆命令外全部放行。只在你完全信任的隔离环境里使用。",
    ),
}

MODE_STYLES: dict[str, str] = {
    "manual": "#50b7c2",
    "workspace": "#7bba55",
    "auto": "#e5c07b",
    "bypass": "#c85f5f",
}

MANUAL_PROFILE = "__manual__"

EFFORT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("off · 关闭 thinking", "off"),
    ("on · 开启 thinking（默认强度）", "on"),
    ("omit · 不发送 thinking 字段", "omit"),
    ("low · 低推理强度", "low"),
    ("medium · 中推理强度", "medium"),
    ("high · 高推理强度", "high"),
)

_EFFORT_TO_SETTINGS: dict[str, tuple[str, str]] = {
    "off": ("disabled", ""),
    "on": ("enabled", ""),
    "omit": ("omit", ""),
    "low": ("enabled", "low"),
    "medium": ("enabled", "medium"),
    "high": ("enabled", "high"),
}


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    created_at: str
    title: str = ""
    preview: str = ""


@dataclass(frozen=True)
class ConfigValues:
    profile_name: str = ""
    endpoint: str = ""
    model: str = ""
    effort: str = "off"
    permission_mode: str = "workspace"
    sandbox: bool = False
    sandbox_network: bool = False
    sandbox_fail_closed: bool = True
    api_key: str = ""


def effort_from_settings(settings: Settings) -> str:
    if settings.llm_thinking == "disabled":
        return "off"
    if settings.llm_thinking == "omit":
        return "omit"
    if settings.llm_reasoning_effort in {"low", "medium", "high"}:
        return settings.llm_reasoning_effort
    return "on"


def build_configured_settings(
        settings: Settings, profiles: dict[str, ModelProfile],
        values: ConfigValues) -> Settings:
    """把配置屏幕的值合并成新的 Settings；不写文件，不改进程环境。"""
    endpoint = values.endpoint.strip()
    model = values.model.strip()
    if not endpoint or not model:
        raise ValueError("endpoint 和模型名都不能为空")
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError("endpoint 必须以 http:// 或 https:// 开头")
    if values.permission_mode not in MODE_INFO:
        raise ValueError(f"未知权限模式：{values.permission_mode}")
    if values.effort not in _EFFORT_TO_SETTINGS:
        raise ValueError(f"未知推理强度：{values.effort}")

    base = settings
    applied_profile = False
    profile: ModelProfile | None = None
    if values.profile_name:
        profile = profiles.get(values.profile_name)
        if profile is None:
            raise ValueError(f"找不到模型 profile：{values.profile_name}")

    # 先比较用户填写的 endpoint 与 profile 是否一致；手动改过 endpoint 时
    # 不应再要求 profile 的密钥，否则用户无法只改地址而不提供 profile key。
    if profile is not None:
        manual_endpoint = (
            endpoint != profile.base_url or model != profile.model)
    else:
        manual_endpoint = (
            endpoint != base.llm_base_url or model != base.llm_model)
    if profile is not None and not manual_endpoint:
        base = apply_model_profile(
            settings, profile, api_key=values.api_key.strip() or None)
        applied_profile = True

    thinking, reasoning_effort = _EFFORT_TO_SETTINGS[values.effort]
    updates: dict = {
        "llm_base_url": endpoint,
        "llm_model": model,
        "llm_thinking": thinking,
        "llm_reasoning_effort": reasoning_effort,
        "permission_mode": values.permission_mode,
        "command_sandbox": values.sandbox,
        "command_sandbox_network": values.sandbox_network,
        "command_sandbox_fail_closed": values.sandbox_fail_closed,
    }
    if not applied_profile:
        updates["model_profile"] = ""
    if manual_endpoint:
        # 手动改过 endpoint 后不能再沿用 profile 的备援入口，否则 429 会把请求
        # 发到一个与当前密钥不匹配的地址。
        updates.update({
            "model_profile": "",
            "llm_provider": "custom",
            "llm_fallback_base_url": "",
            "llm_fallback_model": "",
            "llm_fallback_api_key": SecretStr(""),
        })
    if values.api_key.strip():
        updates["llm_api_key"] = SecretStr(values.api_key.strip())
    return base.model_copy(update=updates)


def _display_path(path: Path) -> str:
    """启动页只显示一个短路径；完整路径仍在会话和报告里。"""
    try:
        relative = path.resolve().relative_to(Path.home())
        text = "~" if str(relative) == "." else f"~/{relative}"
    except ValueError:
        text = str(path)
    if len(text) > 44:
        text = f"…{text[-43:]}"
    return text


class WelcomeScreen(ModalScreen[dict]):
    """启动页：新建、最近会话、配置、权限与退出。"""

    BINDINGS = [
        ("n", "new_session", "新建会话"),
        ("c", "config", "配置"),
        ("m", "mode", "权限模式"),
        ("f2", "config", "配置"),
        ("f3", "mode", "权限模式"),
        ("question_mark", "help", "帮助"),
        ("q", "quit_app", "退出"),
    ]

    def __init__(self, settings: Settings, task_dir: Path,
                 sessions: list[SessionSummary]) -> None:
        super().__init__()
        self.settings = settings
        self.task_dir = task_dir
        self.sessions = sessions

    def compose(self) -> ComposeResult:
        mode = MODE_INFO.get(self.settings.permission_mode)
        options: list[Option] = [
            Option(Text("＋  新建会话", style="bold #7bba55"), id="new"),
        ]
        for summary in self.sessions[:3]:
            title = summary.title or summary.preview or "未命名会话"
            prompt = Text()
            prompt.append("›  ", style="#50b7c2")
            prompt.append(title, style="#d7dae0")
            if summary.created_at:
                prompt.append(f"  {summary.created_at}", style="#8a8f98")
            options.append(Option(prompt, id=f"load:{summary.session_id}"))
        if not self.sessions:
            options.append(Option(
                Text("   暂无历史会话", style="#8a8f98"), disabled=True))
        options.extend([
            Option(Text("≡  配置", style="#e5c07b"), id="config"),
            Option(Text(f"{mode[0] if mode else '●'}  权限模式",
                        style=MODE_STYLES.get(
                            self.settings.permission_mode, "#50b7c2")),
                   id="mode"),
            Option(Text("×  退出", style="#c85f5f"), id="quit"),
        ])

        mode_label = mode[1] if mode else self.settings.permission_mode
        sandbox = "开" if self.settings.command_sandbox else "关"
        meta = Text(justify="center")
        meta.append(f"{self.settings.model_profile or 'custom'} / "
                    f"{self.settings.llm_model or '未配置'}", style="#e5c07b")
        meta.append("   ·   ", style="#30343d")
        meta.append(mode_label,
                    style=MODE_STYLES.get(
                        self.settings.permission_mode, "#50b7c2"))
        meta.append("   ·   ", style="#30343d")
        meta.append(f"沙箱{sandbox}",
                    style="#7bba55" if self.settings.command_sandbox
                    else "#8a8f98")
        with Vertical(id="welcome-dialog"):
            yield Static(
                welcome_renderable(self.app.size.width, self.app.size.height),
                id="welcome-logo", markup=False)
            yield Static(meta, id="welcome-meta")
            yield OptionList(*options, id="welcome-list")
            yield Static(Text(
                "↑↓ 选择 · Enter 打开 · ? 帮助 · Q 退出",
                style="#8a8f98"), id="welcome-keys")

    def on_mount(self) -> None:
        try:
            dialog = self.query_one("#welcome-dialog", Vertical)
            dialog.border_subtitle = _display_path(self.task_dir)
        except NoMatches:
            pass
        self._sync_logo()
        listing = self.query_one("#welcome-list", OptionList)
        listing.highlighted = 0
        listing.focus()

    def on_resize(self) -> None:
        self._sync_logo()

    def _sync_logo(self) -> None:
        try:
            logo = self.query_one("#welcome-logo", Static)
        except NoMatches:
            return
        logo.update(
            welcome_renderable(self.app.size.width, self.app.size.height))

    def on_option_list_option_selected(
            self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(self._result(event.option.id or ""))

    def _result(self, option_id: str) -> dict:
        if option_id.startswith("load:"):
            return {"action": "load", "session_id": option_id[5:]}
        if option_id == "config":
            return {"action": "config"}
        if option_id == "mode":
            return {"action": "mode"}
        if option_id == "help":
            return {"action": "help"}
        if option_id == "quit":
            return {"action": "quit"}
        return {"action": "new"}

    def action_new_session(self) -> None:
        self.dismiss({"action": "new"})

    def action_config(self) -> None:
        self.dismiss({"action": "config"})

    def action_mode(self) -> None:
        self.dismiss({"action": "mode"})

    def action_help(self) -> None:
        self.dismiss({"action": "help"})

    def action_quit_app(self) -> None:
        self.dismiss({"action": "quit"})


class SessionPickerScreen(ModalScreen[dict | None]):
    """运行中切换会话；返回 {"action": "new"}、{"action": "load", ...} 或 None。"""

    BINDINGS = [
        ("escape", "cancel", "取消"),
        ("n", "new_session", "新建会话"),
    ]

    def __init__(self, sessions: list[SessionSummary]) -> None:
        super().__init__()
        self.sessions = sessions

    def compose(self) -> ComposeResult:
        options: list[Option] = [
            Option(Text("＋  新建会话", style="bold #7bba55"), id="new")]
        for summary in self.sessions[:10]:
            prompt = Text()
            prompt.append("›  ", style="#50b7c2")
            prompt.append(summary.title or summary.preview or "未命名会话",
                          style="#d7dae0")
            if summary.created_at:
                prompt.append(f"  {summary.created_at}", style="#8a8f98")
            options.append(Option(prompt, id=f"load:{summary.session_id}"))
        if not self.sessions:
            options.append(Option(
                Text("   暂无历史会话", style="#8a8f98"), disabled=True))
        options.append(Option(Text("×  取消", style="#8a8f98"), id="cancel"))
        with Vertical(id="session-dialog"):
            yield Static(Text("› 切换会话", style="bold #7bba55"),
                         classes="dialog-title")
            yield OptionList(*options, id="session-list")
            yield Static(Text("↑↓ 选择 · Enter 打开 · N 新建 · Esc 取消",
                              style="#5f636d"), classes="dialog-hint")

    def on_mount(self) -> None:
        listing = self.query_one("#session-list", OptionList)
        listing.highlighted = 0
        listing.focus()

    def on_option_list_option_selected(
            self, event: OptionList.OptionSelected) -> None:
        event.stop()
        option_id = event.option.id or "cancel"
        if option_id == "cancel":
            self.dismiss(None)
        elif option_id.startswith("load:"):
            self.dismiss({"action": "load", "session_id": option_id[5:]})
        else:
            self.dismiss({"action": "new"})

    def action_new_session(self) -> None:
        self.dismiss({"action": "new"})

    def action_cancel(self) -> None:
        self.dismiss(None)


class PermissionModeScreen(ModalScreen[str | None]):
    """权限模式选择；返回模式名或 None。"""

    BINDINGS = [
        ("escape", "cancel", "取消"),
        ("e", "manual", "逐项确认"),
        ("w", "workspace", "工作区写入"),
        ("a", "auto", "沙箱内自动"),
        ("b", "bypass", "完全放行"),
    ]

    def __init__(self, current: str = "workspace") -> None:
        super().__init__()
        self.current = current if current in MODE_INFO else "workspace"

    def compose(self) -> ComposeResult:
        options: list[Option] = []
        for name, (icon, label, _description) in MODE_INFO.items():
            # OptionList 每项只按单行渲染；说明放在列表下方随高亮更新。
            prompt = Text()
            prompt.append(f"{icon}  {label}", style="bold #e5c07b")
            if name == self.current:
                prompt.append("  当前", style="#7bba55")
            options.append(Option(prompt, id=name))
        with Vertical(id="mode-dialog"):
            yield Static(Text("● 权限模式", style="bold #7bba55"),
                         classes="dialog-title")
            yield OptionList(*options, id="mode-list")
            yield Static(self._description(self.current), id="mode-desc")
            yield Static(Text(
                "↑↓ 选择 · Enter 应用 · E 逐项确认 · W 工作区 · "
                "A 沙箱内自动 · B 完全放行 · Esc 取消",
                style="#8a8f98"), classes="dialog-hint")

    def on_mount(self) -> None:
        listing = self.query_one("#mode-list", OptionList)
        # 高亮当前模式，而不是永远落在第一项；否则说明文字和高亮行会对不上。
        listing.highlighted = list(MODE_INFO).index(self.current)
        listing.focus()

    def on_option_list_option_highlighted(
            self, event: OptionList.OptionHighlighted) -> None:
        mode = event.option.id or "workspace"
        self.query_one("#mode-desc", Static).update(self._description(mode))

    @staticmethod
    def _description(mode: str) -> Text:
        info = MODE_INFO.get(mode)
        if info is None:
            return Text("")
        text = Text()
        text.append(f"{info[0]} {info[1]}：", style="bold #e5c07b")
        text.append(info[2], style="#8a8f98")
        return text

    def on_option_list_option_selected(
            self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(event.option.id)

    def action_manual(self) -> None:
        self.dismiss("manual")

    def action_workspace(self) -> None:
        self.dismiss("workspace")

    def action_auto(self) -> None:
        self.dismiss("auto")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_bypass(self) -> None:
        self.dismiss("bypass")


class HelpScreen(ModalScreen[None]):
    """帮助、快捷键、命令、权限与沙箱说明。"""

    BINDINGS = [
        ("escape", "close", "关闭"),
        ("q", "close", "关闭"),
        ("enter", "close", "关闭"),
    ]

    def compose(self) -> ComposeResult:
        text = Text()
        text.append("◆ NextCoder\n", style="bold #7bba55")
        text.append(
            "在仓库里工作的编程助手：读写代码、执行命令与测试、维护任务清单、"
            "委派子 agent、调用 MCP、使用 hooks、权限模式与 bubblewrap 沙箱。\n"
            "任务目录只是当前工作仓库，不是 NextCoder 本身。\n\n",
            style="#cfd1d6")
        text.append("快捷键\n", style="bold #e5c07b")
        for key, description in (
            ("Enter", "发送；补全可见时接受候选"),
            ("Shift+Enter", "换行"),
            ("↑ / ↓", "浏览历史；补全可见时选择候选"),
            ("Tab", "接受 / 命令或 @文件 候选"),
            ("Esc", "关闭补全或弹窗"),
            ("Ctrl-C", "停止当前轮；空闲时退出"),
            ("Ctrl-Q", "退出"),
            ("F1 / F2 / F3 / F4", "帮助 / 配置 / 权限模式 / 会话"),
            ("Ctrl-N", "新建会话"),
        ):
            text.append(f"  {key:<20}", style="#50b7c2")
            text.append(f"{description}\n", style="#cfd1d6")
        text.append("\n斜杠命令\n", style="bold #e5c07b")
        for command in COMMANDS:
            text.append(f"  {command.usage:<42}", style="#50b7c2")
            text.append(f"{command.description}\n", style="#cfd1d6")
        text.append("\n@ 文件提及\n", style="bold #e5c07b")
        text.append(
            "输入 @ 后显示任务目录内的文件候选；Tab/Enter 接受。"
            "插入的 @path 只是给模型的引用提示，模型仍会用 read_lines 读取。\n\n",
            style="#cfd1d6")
        text.append("权限模式\n", style="bold #e5c07b")
        for name, (icon, label, description) in MODE_INFO.items():
            text.append(f"  {icon} {label} ({name})\n", style="#50b7c2")
            text.append(f"    {description}\n", style="#8a8f98")
        text.append("\n沙箱\n", style="bold #e5c07b")
        text.append(
            "command_sandbox 使用 bubblewrap：默认断网、只写任务目录、"
            "环境变量按白名单透传。开启后命令在沙箱内执行，策略失败时默认拒绝"
            "（fail-closed）。可在 /config 或 /sandbox 中切换。\n",
            style="#cfd1d6")
        with Vertical(id="help-dialog"):
            yield Static(Text("? 帮助", style="bold #7bba55"),
                         classes="dialog-title")
            with VerticalScroll(id="help-body"):
                yield Static(text)
            yield Static(Text("Esc / Q / Enter 关闭", style="#5f636d"),
                         classes="dialog-hint")

    def action_close(self) -> None:
        self.dismiss(None)


class ConfigScreen(ModalScreen[Settings | None]):
    """运行配置：profile、endpoint、模型、effort、权限模式与沙箱。"""

    BINDINGS = [
        ("escape", "cancel", "取消"),
        ("ctrl+s", "save", "保存"),
    ]

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self._profiles: dict[str, ModelProfile] = {}
        self._profile_error = ""
        try:
            self._profiles = load_model_profiles(
                Path(settings.model_config_file).expanduser())
        except ModelConfigError as exc:
            self._profile_error = str(exc)

    def compose(self) -> ComposeResult:
        profile_options: list[tuple[str, str]] = [
            ("手动 / 当前 endpoint", MANUAL_PROFILE)]
        profile_options.extend(
            (f"{name} · {profile.model}", name)
            for name, profile in self._profiles.items())
        current_profile = (self.settings.model_profile
                           if self.settings.model_profile in self._profiles
                           else MANUAL_PROFILE)
        mode = (self.settings.permission_mode
                if self.settings.permission_mode in MODE_INFO else "workspace")
        effort = effort_from_settings(self.settings)
        sandbox_state, sandbox_path, sandbox_detail = bwrap_status()

        with Vertical(id="config-dialog"):
            yield Static(Text("≡ 运行配置", style="bold #7bba55"),
                         classes="dialog-title")
            with VerticalScroll(id="config-body"):
                # 40 行终端里配置页必须一屏放下常用项：地址和模型名并排，
                # effort 和权限模式并排，三个沙箱开关压成一行三列。
                with Horizontal(classes="form-row"):
                    with Vertical(classes="form-field profile-field"):
                        yield Label("模型 profile", classes="field-label")
                        yield Select(profile_options, value=current_profile,
                                     allow_blank=False, id="cfg-profile")
                    with Vertical(classes="form-field endpoint-field"):
                        yield Label("Endpoint（OpenAI 兼容）",
                                    classes="field-label")
                        yield Input(value=self.settings.llm_base_url,
                                    id="cfg-endpoint")
                    with Vertical(classes="form-field model-field"):
                        yield Label("模型名", classes="field-label")
                        yield Input(value=self.settings.llm_model,
                                    id="cfg-model")
                with Horizontal(classes="form-row"):
                    with Vertical(classes="form-field"):
                        yield Label("思考强度 / effort", classes="field-label")
                        yield Select(list(EFFORT_OPTIONS), value=effort,
                                     allow_blank=False, id="cfg-effort")
                    with Vertical(classes="form-field"):
                        yield Label("权限模式", classes="field-label")
                        yield Select(
                            [(f"{info[1]} · {name}", name)
                             for name, info in MODE_INFO.items()],
                            value=mode, allow_blank=False, id="cfg-mode")
                yield Static(Text(self._mode_description(mode)),
                             id="cfg-mode-desc")
                yield Label("命令沙箱（下一轮生效）", classes="field-label")
                with Horizontal(classes="switch-grid"):
                    with Vertical(classes="switch-cell"):
                        yield Switch(self.settings.command_sandbox,
                                     id="cfg-sandbox")
                        yield Label("启用 bwrap")
                    with Vertical(classes="switch-cell"):
                        yield Switch(self.settings.command_sandbox_network,
                                     id="cfg-network")
                        yield Label("允许网络")
                    with Vertical(classes="switch-cell"):
                        yield Switch(self.settings.command_sandbox_fail_closed,
                                     id="cfg-strict")
                        yield Label("fail-closed")
                status = Text()
                if sandbox_state == "ok":
                    status.append(f"✓ 沙箱可用（{sandbox_path}）。", style="#7bba55")
                elif sandbox_state == "missing":
                    status.append(
                        "× 未找到 bwrap；启用沙箱后命令会因 fail-closed 被拒绝。",
                        style="#c85f5f")
                else:
                    status.append(
                        "× 已找到 bwrap，但当前环境无法创建命名空间；"
                        "启用沙箱后命令会被拒绝。\n",
                        style="#c85f5f")
                    status.append(sandbox_detail, style="#8a8f98")
                if self._profile_error:
                    status.append(f"\n! profile 配置读取失败：{self._profile_error}",
                                  style="#b28443")
                yield Static(status, id="cfg-sandbox-status")
                with Horizontal(classes="form-row"):
                    with Vertical(classes="form-field"):
                        yield Label(
                            "API Key（留空保持当前值；只存在内存，不落盘）",
                            classes="field-label")
                        yield Input(password=True, id="cfg-api-key")
                yield Static(Text(""), id="cfg-status")
            with Horizontal(id="config-buttons"):
                yield Button("保存并应用", id="cfg-save", variant="primary")
                yield Button("取消", id="cfg-cancel")
            yield Static(Text("Tab 切换字段 · Ctrl-S 保存 · Esc 取消",
                              style="#5f636d"), classes="dialog-hint")

    def on_mount(self) -> None:
        self.query_one("#cfg-profile", Select).focus()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "cfg-profile":
            name = event.value if isinstance(event.value, str) else MANUAL_PROFILE
            if name == MANUAL_PROFILE:
                self._set_status("手动 endpoint：请确认密钥属于该入口。", "#8a8f98")
                return
            profile = self._profiles.get(name)
            if profile is not None:
                self.query_one("#cfg-endpoint", Input).value = profile.base_url
                self.query_one("#cfg-model", Input).value = profile.model
                if profile.thinking == "omit":
                    self.query_one("#cfg-effort", Select).value = "omit"
                elif profile.thinking == "disabled":
                    self.query_one("#cfg-effort", Select).value = "off"
                else:
                    self.query_one("#cfg-effort", Select).value = "on"
                self._set_status(
                    f"已选择 profile {name}；保存后下一轮生效。", "#7bba55")
        elif event.select.id == "cfg-mode":
            mode = event.value if isinstance(event.value, str) else "workspace"
            self.query_one("#cfg-mode-desc", Static).update(
                Text(self._mode_description(mode)))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cfg-cancel":
            self.action_cancel()
        elif event.button.id == "cfg-save":
            self.action_save()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_save(self) -> None:
        profile_name = self._select_value("cfg-profile")
        if profile_name == MANUAL_PROFILE:
            profile_name = ""
        values = ConfigValues(
            profile_name=profile_name,
            endpoint=self.query_one("#cfg-endpoint", Input).value,
            model=self.query_one("#cfg-model", Input).value,
            effort=self._select_value("cfg-effort") or "off",
            permission_mode=self._select_value("cfg-mode") or "workspace",
            sandbox=self.query_one("#cfg-sandbox", Switch).value,
            sandbox_network=self.query_one("#cfg-network", Switch).value,
            sandbox_fail_closed=self.query_one("#cfg-strict", Switch).value,
            api_key=self.query_one("#cfg-api-key", Input).value,
        )
        try:
            configured = build_configured_settings(
                self.settings, self._profiles, values)
        except ValueError as exc:
            self._set_status(str(exc), "#c85f5f")
            return
        self.dismiss(configured)

    def _select_value(self, widget_id: str) -> str:
        value = self.query_one(f"#{widget_id}", Select).value
        return value if isinstance(value, str) else ""

    def _set_status(self, message: str, style: str) -> None:
        self.query_one("#cfg-status", Static).update(Text(message, style=style))

    @staticmethod
    def _mode_description(mode: str) -> str:
        info = MODE_INFO.get(mode)
        if info is None:
            return ""
        return f"{info[0]} {info[1]}：{info[2]}"
