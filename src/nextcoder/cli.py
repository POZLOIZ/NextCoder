import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

import typer
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from nextcoder import __version__

app = typer.Typer(help="NextCoder：在现有仓库中工作的命令行编程助手。")
memory_app = typer.Typer(help="查看、更正或停用任务级项目记忆。")
app.add_typer(memory_app, name="memory")

# app callback 写入；命令通过 _settings() 统一解析模型 profile。
_ACTIVE_PROFILE = ""


@app.callback(invoke_without_command=True)
def entry(
    ctx: typer.Context,
    profile: str = typer.Option(
        "", "--profile", envvar="NEXTCODER_MODEL_PROFILE",
        help="config/models.local.toml 中 models 数组的条目名称"),
    version: bool = typer.Option(
        False, "--version", is_eager=True, help="显示版本并退出"),
) -> None:
    if version:
        typer.echo(f"NextCoder {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()
    global _ACTIVE_PROFILE
    _ACTIVE_PROFILE = profile


def _settings(require_model: bool = True):
    """构造 Settings 并按需应用模型 profile；配置错误直接给出可读信息。

    `memory`/`sessions` 等只读写本地状态的命令传 False：它们不应该因为
    模型密钥未配置就无法工作。
    """
    from nextcoder.model_config import ModelConfigError, resolve_settings
    from nextcoder.settings import Settings as _AppSettings

    settings = _AppSettings()
    if not require_model:
        return settings
    try:
        return resolve_settings(settings, profile_name=_ACTIVE_PROFILE)
    except ModelConfigError as exc:
        typer.echo(f"模型配置错误：{exc}", err=True)
        raise typer.Exit(2) from exc


class ProbeState(TypedDict):
    value: int


def check_checkpoint() -> None:
    builder = StateGraph(ProbeState)
    builder.add_node("increment", lambda state: {"value": state["value"] + 1})
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    config = {"configurable": {"thread_id": "environment-check"}}
    with tempfile.TemporaryDirectory(prefix="nextcoder-env-") as directory:
        path = str(Path(directory) / "checkpoints.sqlite")
        with SqliteSaver.from_conn_string(path) as saver:
            graph = builder.compile(checkpointer=saver)
            assert graph.invoke({"value": 0}, config)["value"] == 1
        with SqliteSaver.from_conn_string(path) as saver:
            reopened = builder.compile(checkpointer=saver)
            assert reopened.get_state(config).values["value"] == 1


@app.command()
def doctor(sandbox: bool = typer.Option(False, help="另外检查可选的 bubblewrap")) -> None:
    """检查依赖、SQLite 检查点、命令工具及可选沙箱。"""
    failures = 0

    def check(label, operation):
        nonlocal failures
        try:
            operation()
            typer.echo(f"通过：{label}")
        except Exception as exc:
            failures += 1
            typer.echo(f"失败：{label}（{type(exc).__name__}）")

    def check_venv():
        if sys.prefix == sys.base_prefix:
            raise RuntimeError("请使用项目虚拟环境")

    check("独立 Python 虚拟环境", check_venv)
    for module in ("langchain", "langchain_openai", "langgraph", "mcp", "tiktoken"):
        check(f"依赖 {module}", lambda module=module: importlib.import_module(module))
    check("LangGraph 执行与 SQLite 检查点重新读取", check_checkpoint)

    def require_command(name):
        if shutil.which(name) is None:
            raise FileNotFoundError(name)

    for command in ("git", "bash"):
        check(f"命令 {command}", lambda command=command: require_command(command))

    try:
        require_command("rg")
        typer.echo("通过：命令 rg（搜索加速）")
    except FileNotFoundError:
        typer.echo("提示：未安装 rg，search 会退回内置扫描（较慢）")

    if sandbox:

        def check_bwrap():
            require_command("bwrap")
            subprocess.run(
                [
                    "bwrap",
                    "--unshare-all",
                    "--die-with-parent",
                    "--ro-bind",
                    "/",
                    "/",
                    "--proc",
                    "/proc",
                    "--dev",
                    "/dev",
                    "--tmpfs",
                    "/tmp",
                    "/usr/bin/true",
                ],
                check=True,
                timeout=10,
                capture_output=True,
            )

        check("bubblewrap 启动与命名空间创建", check_bwrap)
    else:
        typer.echo("未检查：bubblewrap（添加 --sandbox 可检查）")
    typer.echo("本机环境检查完成；共用测试模型由工作区 .env 提供。")
    raise typer.Exit(1 if failures else 0)


@app.command("models")
def models(
    config: Path = typer.Option(
        None, "--config", help="模型配置文件；默认读取 Settings.model_config_file"),
) -> None:
    """列出模型 profile；只显示地址、模型名和密钥环境变量名。"""
    from nextcoder.model_config import ModelConfigError, load_model_profiles
    from nextcoder.settings import Settings as _AppSettings

    path = config or _AppSettings().model_config_file
    try:
        profiles = load_model_profiles(path)
    except ModelConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    if not profiles:
        typer.echo(f"（{path} 中没有 profile）")
        return
    typer.echo(f"配置文件：{path}")
    for profile in profiles.values():
        typer.echo(
            f"{profile.name}  [{profile.provider or profile.protocol}] "
            f"{profile.model}  {profile.base_url}")
        if profile.api_key_env:
            typer.echo(f"  密钥环境变量：{profile.api_key_env}")


@app.command("hooks")
def hooks(
    task_dir: Path = typer.Option(Path("."), "--task-dir", help="任务代码目录"),
    config: Path = typer.Option(
        None, "--config", help="hook 配置文件；默认读取任务目录下的 hooks_file"),
) -> None:
    """列出内置 hook 和任务配置的 hook，并校验 hooks.json。"""
    from nextcoder.runtime.hooks import HookConfigError, load_config_hooks
    from nextcoder.settings import Settings as _AppSettings

    settings = _AppSettings()
    path = config or (task_dir / settings.hooks_file)
    typer.echo(
        "内置：重复调用提醒 "
        f"{settings.repeat_warn_limit} 次、搜索提醒 {settings.search_warn_limit} 次、"
        f"连续空输出提醒 {settings.empty_output_warn_limit} 次"
        "（0 表示关闭；均由 hook 总线执行）")
    try:
        entries = load_config_hooks(path)
    except HookConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    if not entries:
        typer.echo(f"（{path} 中没有额外 hook）")
        return
    typer.echo(f"配置文件：{path}")
    for hook in sorted(entries, key=lambda item: item.priority):
        tools = "、".join(sorted(hook.tools)) if hook.tools else "全部工具"
        pattern = f"  pattern={hook.pattern.pattern}" if hook.pattern else ""
        typer.echo(
            f"{hook.name}  {hook.event}  {hook.action}  {tools}"
            f"{pattern}  priority={hook.priority}")


@app.command()
def run(
    task_dir: Path = typer.Option(..., help="任务代码目录（会被修改）"),
    request: str = typer.Option(..., help="用户需求"),
    model_calls: int = typer.Option(None, help="本轮模型调用上限（默认用配置）"),
    session: str = typer.Option(None, help="继续已有会话的 session id"),
) -> None:
    """在任务目录中运行一轮助手，输出补丁与运行记录。"""
    from nextcoder.sessions.manager import create_session, load_session, run_turn

    settings = _settings()
    if model_calls is not None:
        settings.model_calls_limit = model_calls
    if not task_dir.is_dir():
        typer.echo(f"任务目录不存在：{task_dir}", err=True)
        raise typer.Exit(2)
    task_dir = task_dir.resolve()
    try:
        active = load_session(settings, task_dir, session) if session else \
            create_session(settings, task_dir)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    outcome = run_turn(settings, active, request)
    if active.delegation_manager is not None:
        active.delegation_manager.close()
    typer.echo(f"会话：{active.session_id}")
    typer.echo(f"本轮：{outcome.run_dir}")
    typer.echo(f"停止原因：{outcome.stop_reason}")
    suffix = "" if outcome.has_changes else "（未产生改动）"
    typer.echo(f"补丁：{outcome.run_dir / 'final.patch'}{suffix}")


@app.command()
def chat(
    task_dir: Path = typer.Option(..., help="任务代码目录（会被修改）"),
    session: str = typer.Option(None, help="继续已有会话的 session id"),
    model_calls: int = typer.Option(None, help="每轮模型调用上限（默认用配置）"),
    stream: bool = typer.Option(False, "--stream", help="流式显示模型输出"),
) -> None:
    """进入交互模式：提出需求后可以继续追问、停止或继续。"""
    from nextcoder.runtime.agent import SYSTEM_PROMPT_INTERACTIVE
    from nextcoder.sessions.manager import create_session, load_session, run_turn

    settings = _settings()
    if model_calls is not None:
        settings.model_calls_limit = model_calls
    if not task_dir.is_dir():
        typer.echo(f"任务目录不存在：{task_dir}", err=True)
        raise typer.Exit(2)
    task_dir = task_dir.resolve()
    try:
        active = load_session(settings, task_dir, session) if session else \
            create_session(settings, task_dir)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"会话：{active.session_id}（输入 exit 结束，Ctrl-C 停止当前轮）")
    while True:
        try:
            request = typer.prompt("you")
        except (EOFError, KeyboardInterrupt, typer.Abort):
            break
        if request.strip().lower() in {"exit", "quit", "/exit", "/quit"}:
            break
        if not request.strip():
            continue
        printer = _StreamPrinter() if stream else None
        outcome = run_turn(
            settings, active, request,
            system_prompt=SYSTEM_PROMPT_INTERACTIVE,
            approval_callback=_ask_permission,
            event_handler=printer.handle if printer is not None else None,
            streaming=stream,
        )
        if printer is not None and printer.assistant_text:
            typer.echo("")
        else:
            answer = _last_assistant_message(outcome.final_state)
            typer.echo(f"\nassistant> {answer}\n")
        typer.echo(f"[{outcome.stop_reason} · 本轮 {outcome.run_dir}]")
    if active.delegation_manager is not None:
        active.delegation_manager.close()


@app.command()
def tui(
    task_dir: Path = typer.Option(..., help="任务代码目录（会被修改）"),
    session: str = typer.Option(None, help="继续已有会话的 session id"),
    model_calls: int = typer.Option(None, help="每轮模型调用上限（默认用配置）"),
) -> None:
    """启动 Textual 交互界面（需要 uv sync --extra tui）。"""
    try:
        from nextcoder.tui.app import NextCoderApp
    except ImportError as exc:
        typer.echo("未安装 Textual；请运行 uv sync --extra tui", err=True)
        raise typer.Exit(1) from exc
    from nextcoder.sessions.manager import load_session

    settings = _settings()
    if model_calls is not None:
        settings.model_calls_limit = model_calls
    if not task_dir.is_dir():
        typer.echo(f"任务目录不存在：{task_dir}", err=True)
        raise typer.Exit(2)
    task_dir = task_dir.resolve()
    if session:
        try:
            active = load_session(settings, task_dir, session)
        except (ValueError, FileNotFoundError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(2) from exc
        app = NextCoderApp(settings, active)
    else:
        # 不预建会话：先显示启动页，让用户选择历史会话或新建。
        app = NextCoderApp(settings, task_dir=task_dir)
    app.run()
    active = app.session
    if active is not None and active.delegation_manager is not None:
        active.delegation_manager.close()


@app.command()
def compact(
    task_dir: Path = typer.Option(..., help="任务代码目录"),
    session: str = typer.Option(..., help="要主动压缩的会话 id"),
) -> None:
    """主动压缩已有会话：生成并保存任务摘要，后续请求复用。"""
    from nextcoder.sessions.manager import compact_session, load_session

    if not task_dir.is_dir():
        typer.echo(f"任务目录不存在：{task_dir}", err=True)
        raise typer.Exit(2)
    settings = _settings()
    try:
        active = load_session(settings, task_dir.resolve(), session)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    result = compact_session(settings, active)
    typer.echo(f"会话：{active.session_id}")
    typer.echo(f"记录：{result['compaction_dir']}")
    if not result["compacted"]:
        typer.echo(f"未压缩：{result['reason']}")
        if result.get("summary_stats", {}).get("errors"):
            for error in result["summary_stats"]["errors"]:
                typer.echo(f"错误：{error}", err=True)
        raise typer.Exit(1 if result.get("summary_stats", {}).get("errors") else 0)
    typer.echo(f"摘要：{result['summary_id']}（覆盖到第 {result['covered_count']} 条消息）")
    typer.echo(f"估算：{result['before_tokens']} → {result['after_tokens']} tokens；"
               f"覆盖 {result['covered_groups']} 组，保留 {result['kept_groups']} 组")
    typer.echo("摘要正文：")
    typer.echo(result["summary"])


@app.command()
def sessions(task_dir: Path = typer.Option(..., help="任务代码目录")) -> None:
    """列出任务目录下已有的会话。"""
    from nextcoder.sessions.manager import list_sessions

    found = list_sessions(_settings(require_model=False), task_dir)
    if not found:
        typer.echo("（没有会话）")
        return
    for item in found:
        typer.echo(f"{item.get('session_id')}  {item.get('created_at', '')}")


@memory_app.command("list")
def memory_list(
    task_dir: Path = typer.Option(..., help="任务代码目录"),
    include_invalid: bool = typer.Option(False, "--all", help="包含已停用条目"),
) -> None:
    """查看项目记忆。"""
    from nextcoder.sessions.memory import ProjectMemory

    if not task_dir.is_dir():
        typer.echo(f"任务目录不存在：{task_dir}", err=True)
        raise typer.Exit(2)
    memory = ProjectMemory(_settings(require_model=False), task_dir)
    items = memory.list_all() if include_invalid else memory.select(limit=1000)
    memory.close()
    if not items:
        typer.echo("（没有匹配的记忆）")
        return
    for item in items:
        if item.invalidated_at is None:
            status = "有效"
        else:
            status = f"停用（{item.invalid_reason or '未说明'}）"
        typer.echo(f"{item.key}  [{item.kind}]  {status}  "
                   f"confidence={item.confidence:.2f}")
        typer.echo(f"  {item.value}")


@memory_app.command("disable")
def memory_disable(
    task_dir: Path = typer.Option(..., help="任务代码目录"),
    key: str = typer.Option(..., help="记忆 key，先用 memory list 查看"),
    reason: str = typer.Option("用户手动停用", help="停用原因"),
) -> None:
    """停用一条记忆；停用后不再注入提示。"""
    from nextcoder.sessions.memory import ProjectMemory

    if not task_dir.is_dir():
        typer.echo(f"任务目录不存在：{task_dir}", err=True)
        raise typer.Exit(2)
    memory = ProjectMemory(_settings(require_model=False), task_dir)
    disabled = memory.disable(key, reason)
    memory.close()
    if not disabled:
        typer.echo(f"未停用：{key} 不存在或已经停用", err=True)
        raise typer.Exit(1)
    typer.echo(f"已停用：{key}")


@memory_app.command("set")
def memory_set(
    task_dir: Path = typer.Option(..., help="任务代码目录"),
    key: str = typer.Option(..., help="记忆 key；已有条目会被更正，否则新建"),
    kind: str = typer.Option(..., help="记忆类型，如 user_preference/test_command"),
    value: str = typer.Option(..., help="更正后的内容"),
    confidence: float = typer.Option(1.0, min=0.0, max=1.0),
) -> None:
    """手工新增或更正一条记忆；自动提取不会覆盖它。"""
    from nextcoder.sessions.memory import ProjectMemory

    if not task_dir.is_dir():
        typer.echo(f"任务目录不存在：{task_dir}", err=True)
        raise typer.Exit(2)
    memory = ProjectMemory(_settings(require_model=False), task_dir)
    memory.set_manual(key, kind, value, confidence=confidence)
    memory.close()
    typer.echo(f"已保存：{key}")


@app.command()
def eval(
    suite: Path = typer.Option(..., help="任务包 tasks.json 路径"),
    tasks: str = typer.Option("", help="逗号分隔的 task_id；空为全部已实现验收的题"),
    out: Path = typer.Option(Path("evaluation/runs"), help="结果输出目录"),
) -> None:
    """运行任务包评测：准备环境、运行助手、独立验收补丁或运行场景。"""
    from nextcoder.evalrunner import GRADERS, run_graded_task
    from nextcoder.runtime_scenarios import RUNTIME_SCENARIOS, run_runtime_scenario

    package_dir = suite.parent
    all_tasks = json.loads(suite.read_text(encoding="utf-8"))["tasks"]
    wanted = {t.strip() for t in tasks.split(",") if t.strip()}
    runnable = set(GRADERS) | RUNTIME_SCENARIOS
    selected = [t for t in all_tasks
                if (not wanted or t["task_id"] in wanted) and t["task_id"] in runnable]
    if not selected:
        typer.echo("没有匹配且已实现验收的任务", err=True)
        raise typer.Exit(2)
    settings = _settings()
    for task in selected:
        print(f"运行 {task['task_id']} …", flush=True)
        if task["task_id"] in RUNTIME_SCENARIOS:
            result = run_runtime_scenario(package_dir, task, out, settings)
        else:
            result = run_graded_task(settings, package_dir, task, out)
        typer.echo(f"{result['task_id']}: {result['status']}"
                   + (f"（{result['error']}）" if result["error"] else ""))
        for check in result["checks"]:
            typer.echo(f"  {'通过' if check['passed'] else '未过'} {check['name']}"
                       + (f"：{check.get('detail')}" if not check["passed"] else ""))


def _last_assistant_message(final_state: dict) -> str:
    for message in reversed(final_state.get("messages") or []):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, list):
                return json.dumps(content, ensure_ascii=False)
            return str(content)
    return "（本轮没有助手消息）"


class _StreamPrinter:
    """chat --stream 的轻量渲染：模型增量直接输出，工具事件单独成行。"""

    def __init__(self) -> None:
        self.assistant_text = ""
        self._inline = False

    def handle(self, event) -> None:
        from nextcoder.runtime.events import (
            AssistantDelta,
            ChildUpdated,
            Compaction,
            HookTriggered,
            PlanUpdated,
            ToolFinished,
            ToolStarted,
        )

        if isinstance(event, AssistantDelta):
            self.assistant_text += event.text
            typer.echo(event.text, nl=False)
            self._inline = True
        elif isinstance(event, ToolStarted):
            self._newline()
            detail = f" {event.detail}" if event.detail else ""
            typer.echo(f"[工具] {event.tool}{detail}")
        elif isinstance(event, ToolFinished):
            typer.echo(f"  {event.status} {event.tool}")
        elif isinstance(event, PlanUpdated):
            self._newline()
            typer.echo(f"[任务清单] v{event.revision}，共 {len(event.tasks)} 项")
        elif isinstance(event, Compaction):
            self._newline()
            typer.echo(f"[压缩] {event.before_tokens} → {event.after_tokens} tokens")
        elif isinstance(event, ChildUpdated):
            self._newline()
            typer.echo(f"[子 agent] {event.role} {event.status}")
        elif isinstance(event, HookTriggered):
            self._newline()
            detail = f" · {event.detail}" if event.detail else ""
            typer.echo(f"[hook] {event.hook} {event.action} {event.tool}{detail}")

    def _newline(self) -> None:
        if self._inline:
            typer.echo("")
            self._inline = False


def _ask_permission(request) -> str:
    """交互模式下的权限确认；EOF/Ctrl-C 一律按 deny 处理。"""
    typer.echo("")
    typer.echo(f"[权限确认] {request.tool} · {request.action}")
    typer.echo(f"  目标：{request.target}")
    if request.reason:
        typer.echo(f"  原因：{request.reason}")
    if request.hint:
        typer.echo(f"  建议：{request.hint}")
    try:
        answer = typer.prompt("  是否允许？[o]nce/[s]ession/[d]eny", default="d")
    except (EOFError, KeyboardInterrupt, typer.Abort):
        return "deny"
    normalized = answer.strip().lower()
    if normalized in {"o", "once", "y", "yes"}:
        return "allow_once"
    if normalized in {"s", "session"}:
        return "allow_session"
    return "deny"
