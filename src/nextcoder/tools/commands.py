"""命令执行：输出边产生边落盘、有限摘录、超时/取消终止整个进程组。"""

import os
import shlex
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from nextcoder.settings import Settings
from nextcoder.tools.base import ToolContext, ToolResult

_SECRET_ENV_SUFFIX = "_API_KEY"
_SECRET_ENV_NAMES = {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
# docker exec 自身失败（容器不存在、daemon 不可达）时的退出码。
_DOCKER_ERROR_EXIT = 125
# 容器内 timeout 终止命令后的退出码；与 OOM 等 137 区分开。
_CONTAINER_TIMEOUT_EXIT = 124
# 宿主机只比容器内超时多等一会儿：真正的终止由容器内 timeout 完成，
# 这样即使 docker CLI 被杀，容器里也不会留下跑不完的进程。
_CONTAINER_TIMEOUT_GRACE_SECONDS = 30

# TUI 每次刷新状态都探测 bwrap 太贵；缓存 30 秒即可反映安装/权限变化。
_BWRAP_STATUS_TTL = 30.0
_BWRAP_STATUS_CACHE: tuple[float, tuple[str, str, str]] | None = None


def bwrap_status(timeout: float = 3.0) -> tuple[str, str, str]:
    """返回 (state, path, detail)，state 为 ok/missing/unusable。

    仅检查 `which` 会误判：容器里常有 bwrap 但内核/外部 seccomp 不允许创建
    命名空间。这里实际启动一次最小沙箱，UI 才能如实显示“可用/不可用”。
    """
    global _BWRAP_STATUS_CACHE
    now = time.monotonic()
    if (_BWRAP_STATUS_CACHE is not None
            and now - _BWRAP_STATUS_CACHE[0] < _BWRAP_STATUS_TTL):
        return _BWRAP_STATUS_CACHE[1]
    found = shutil.which("bwrap")
    if found is None:
        result = ("missing", "", "未找到 bwrap 可执行文件")
    else:
        path = str(Path(found).resolve())
        try:
            completed = subprocess.run(
                [
                    path, "--unshare-all", "--die-with-parent",
                    "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
                    "--tmpfs", "/tmp", "/usr/bin/true",
                ],
                timeout=timeout,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            result = ("unusable", path, f"bwrap 探测失败：{exc}")
        else:
            if completed.returncode == 0:
                result = ("ok", path, "")
            else:
                detail = " ".join(
                    (completed.stderr or completed.stdout or "").split())
                result = ("unusable", path,
                          detail[:300] or f"bwrap 返回 {completed.returncode}")
    _BWRAP_STATUS_CACHE = (now, result)
    return result


def running_in_container() -> bool:
    """当前进程是否已经在一个容器里；用于判断命令是否已经与宿主隔离。"""
    return Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()


def _secret_free_env() -> dict[str, str]:
    """子进程/docker CLI 不继承 Agent 密钥与 NEXTCODER_* 配置。"""
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if upper.startswith("NEXTCODER_"):
            continue
        if upper.endswith(_SECRET_ENV_SUFFIX) or upper in _SECRET_ENV_NAMES:
            continue
        env[key] = value
    return env


def _child_env(settings: Settings) -> dict[str, str]:
    """宿主机命令环境；任务自有虚拟环境的 bin 应优先于宿主环境。"""
    env = _secret_free_env()
    if settings.command_path_prepend:
        current = env.get("PATH", "")
        env["PATH"] = settings.command_path_prepend + (
            os.pathsep + current if current else "")
    return env


def _container_argv(settings: Settings, command: str, limit: int) -> list[str]:
    """把命令包进 docker exec；超时由容器内 timeout 兜底。"""
    activate = settings.command_container_activate.strip()
    prefix = f"{activate} && " if activate else ""
    inner = (
        f"{prefix}cd {shlex.quote(settings.command_container_workdir)} && "
        f"timeout --signal=KILL {limit}s bash -c {shlex.quote(command)}"
    )
    return ["docker", "exec", settings.command_container, "bash", "-lc", inner]


def _sandbox_env(settings: Settings) -> dict[str, str]:
    """沙箱内环境：默认清空，只保留最低限度和显式透传项。

    `sandbox_env_passthrough` 是用户显式选择，但仍拒绝 `*_API_KEY` 和
    `NEXTCODER_*`，避免一次配置失误把模型密钥带进命令进程。
    """
    env = {
        "PATH": os.environ.get("PATH") or "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "LANG": os.environ.get("LANG") or "C.UTF-8",
    }
    for name in settings.sandbox_env_passthrough:
        if not name:
            continue
        upper = name.upper()
        if upper.startswith("NEXTCODER_"):
            continue
        if upper.endswith(_SECRET_ENV_SUFFIX) or upper in _SECRET_ENV_NAMES:
            continue
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    return env


def _sandbox_argv(settings: Settings, task_dir: Path, command: str,
                  limit: int) -> list[str] | None:
    """宿主机命令的 bubblewrap 包装；bwrap 不存在时返回 None。

    隔离原则：
    - 不绑定真实 `$HOME`，任务目录是唯一默认可写位置；
    - 清空环境变量，只注入 PATH/HOME/TMPDIR/LANG 与显式透传项；
    - 默认 `--unshare-net`，需要联网时显式设置 `command_sandbox_network`；
    - `--new-session` 防止通过控制终端逃逸，`--die-with-parent` 随父进程退出。
    是否允许在缺少 bwrap 时裸跑由调用方按 `command_sandbox_fail_closed` 决定。
    """
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        return None
    # 必须使用解析后的绝对路径：Popen 会用 _child_env() 的 PATH 再次查找，
    # 而该 PATH 可能包含 command_path_prepend（任务目录内可写）。若仍传
    # "bwrap"，模型可以在该目录放一个假 bwrap，从而绕过沙箱。
    bwrap_path = Path(bwrap).resolve()
    task = Path(task_dir).resolve()
    try:
        bwrap_path.relative_to(task)
    except ValueError:
        pass
    else:
        # 不接受任务目录内的 bwrap；任务目录是模型可写区域。
        return None
    argv = [
        str(bwrap_path),
        "--clearenv",
        "--new-session",
        "--die-with-parent",
        "--unshare-pid",
        "--unshare-ipc",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
    ]
    if not settings.command_sandbox_network:
        argv.append("--unshare-net")
    for root in ("/usr", "/etc", "/bin", "/lib", "/lib64", "/sbin"):
        argv += ["--ro-bind-try", root, root]
    for root in settings.sandbox_readonly_roots:
        expanded = str(Path(root).expanduser())
        argv += ["--ro-bind-try", expanded, expanded]
    argv += ["--bind", str(task), str(task)]
    for root in settings.sandbox_writable_roots:
        expanded = str(Path(root).expanduser())
        argv += ["--bind-try", expanded, expanded]
    for name, value in _sandbox_env(settings).items():
        argv += ["--setenv", name, value]
    argv += [
        "bash", "-c",
        f"cd {shlex.quote(str(task))} && "
        f"timeout --signal=KILL {limit}s bash -c {shlex.quote(command)}",
    ]
    return argv


def run_command(context: ToolContext, call_id: str, command: str,
                timeout: int | None = None) -> ToolResult:
    started = time.monotonic()
    limit = timeout or context.settings.command_timeout_seconds
    log_name = f"{call_id}-command.log"
    log_path = context.logs_dir / log_name
    context.logs_dir.mkdir(parents=True, exist_ok=True)
    in_container = bool(context.settings.command_container)
    unsandboxed_note: str | None = None
    try:
        # 独立进程组：超时/取消时能终止全部子进程。
        if in_container:
            process = subprocess.Popen(
                _container_argv(context.settings, command, limit),
                cwd=context.task_dir,
                env=_secret_free_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        else:
            argv = ["bash", "-c", command]
            if context.settings.command_sandbox:
                argv = _sandbox_argv(context.settings, context.task_dir, command, limit)
                if argv is None:
                    if context.settings.command_sandbox_fail_closed:
                        # fail-closed：沙箱不可用时不执行命令，避免静默裸跑。
                        message = (
                            "[沙箱] bwrap 不可用，命令未执行"
                            "（command_sandbox_fail_closed=true）")
                        log_path.write_text(message + "\n", encoding="utf-8")
                        return ToolResult(
                            call_id, "run_command", "failed",
                            artifact_path=str(log_path),
                            duration_seconds=time.monotonic() - started,
                            error="沙箱不可用：未找到 bwrap",
                            note=message,
                        )
                    # 只有显式关闭 fail-closed 才允许裸跑，并明确告知未沙箱化。
                    unsandboxed_note = "[沙箱] bwrap 不可用，本次命令未沙箱化执行"
                    argv = ["bash", "-c", command]
            process = subprocess.Popen(
                argv,
                cwd=context.task_dir,
                env=_child_env(context.settings),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
    except OSError as exc:
        return ToolResult(call_id, "run_command", "failed", error=f"无法启动命令：{exc}")

    chunks: list[str] = []
    lock = threading.Lock()
    state = {"bytes": 0, "truncated": False}
    max_bytes = context.settings.max_output_bytes

    def reader() -> None:
        with log_path.open("w", encoding="utf-8") as log:
            while True:
                try:
                    chunk = process.stdout.read(65536) if process.stdout else ""
                except (ValueError, OSError):
                    break
                if not chunk:
                    break
                data = chunk.encode("utf-8")
                with lock:
                    remaining = max_bytes - state["bytes"]
                    if remaining <= 0:
                        state["truncated"] = True
                        continue
                    piece = data[:remaining].decode("utf-8", "ignore")
                    chunks.append(piece)
                    log.write(piece)
                    log.flush()
                    state["bytes"] += len(piece.encode("utf-8"))
                    if len(data) > remaining:
                        state["truncated"] = True
            if state["truncated"]:
                log.write("\n[输出超过存储上限，后续内容已截断]\n")

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    deadline = time.monotonic() + limit
    if in_container:
        deadline += _CONTAINER_TIMEOUT_GRACE_SECONDS
    status = "completed"
    error: str | None = None
    while process.poll() is None:
        if context.cancel_event.is_set():
            _kill_process_group(process)
            status = "cancelled"
            error = "用户取消"
            break
        if time.monotonic() >= deadline:
            _kill_process_group(process)
            status = "timeout"
            error = f"timeout after {limit}s"
            break
        time.sleep(0.05)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
    reader_thread.join(timeout=5)

    output = "".join(chunks)
    if status == "completed":
        if process.returncode == 0:
            status = "completed"
        elif in_container and process.returncode == _CONTAINER_TIMEOUT_EXIT:
            # 容器内 timeout 已终止命令；docker exec 把 124 原样带回。
            status = "timeout"
            error = f"timeout after {limit}s"
            output += f"\n[超时：容器内命令超过 {limit} 秒被终止]"
        else:
            status = "failed"
            error = f"exit code {process.returncode}"
            if in_container and process.returncode == _DOCKER_ERROR_EXIT:
                error = ("docker exec 失败：容器不可用或命令无法启动"
                         f"（exit {_DOCKER_ERROR_EXIT}）")
            elif in_container and process.returncode == 137:
                error = "exit code 137（容器内命令被 SIGKILL，可能是 OOM 或外部终止）"
    elif status == "cancelled":
        output += "\n[已取消：命令及其子进程已终止]"
    elif status == "timeout":
        output += f"\n[超时：命令超过 {limit} 秒被终止，子进程一并终止]"
    if state["truncated"]:
        output += "\n[输出已按存储上限截断，完整内容未保留]"
    if output:
        note = f"完整输出：{log_path}" + ("（已截断）" if state["truncated"] else "")
    else:
        # 空输出必须明确告诉模型，否则它会把“没有输出”误认为“输出被省略”，
        # 继续逐行试探同一个命令。管道还会把前面命令的失败藏在最后一个命令
        # 的退出码后面，尤其是模型自己加了 2>/dev/null 时。
        note = "输出为空（命令没有打印任何内容）"
        if "2>/dev/null" in command:
            note += ("；注意：命令隐藏了 stderr，管道前的命令可能已经失败。"
                     "去掉 2>/dev/null，必要时加 `set -o pipefail` 后重跑。")
        elif "|" in command:
            note += "；管道命令的退出码来自最后一个命令，诊断时建议加 `set -o pipefail`。"
    if unsandboxed_note is not None:
        note = f"{note}\n{unsandboxed_note}" if note else unsandboxed_note
    return ToolResult(
        call_id, "run_command", status,
        output_excerpt=output,
        artifact_path=str(log_path),
        exit_code=process.returncode,
        duration_seconds=time.monotonic() - started,
        error=error,
        note=note,
    )


def _kill_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()
