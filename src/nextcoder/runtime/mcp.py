"""MCP 客户端：官方 SDK 的 stdio / Streamable HTTP 传输，后台事件循环线程供同步调用。

配置文件（默认 <task_dir>/.nextcoder/mcp.json）不存在时零开销：不启动
任何连接、不注册任何工具。单个服务器启动失败不影响其余服务器，失败
原因记录在 McpManager.errors。工具命名为 mcp__<server>__<tool>。
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import re
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from nextcoder.settings import Settings
from nextcoder.tools.base import ToolError
from nextcoder.tools.commands import _secret_free_env

MCP_PREFIX = "mcp__"
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_CALL_TIMEOUT_MS = 30_000
_ENV_PATTERN = re.compile(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class McpServerConfig:
    name: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    transport: Literal["stdio", "http"] = "stdio"
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    timeout_ms: int = DEFAULT_CALL_TIMEOUT_MS
    # fnmatch 模式列表；None 表示不过滤（接受服务器全部工具）。
    allowed_tools: list[str] | None = None


def expand_env_value(value: str) -> str:
    """把 {env:VAR} 展开为环境变量；未定义时替换为空串。"""
    return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)


def load_mcp_config(path: Path) -> dict[str, McpServerConfig]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    servers: dict[str, McpServerConfig] = {}
    for name, spec in (data.get("servers") or {}).items():
        if not isinstance(spec, dict):
            continue
        transport = str(spec.get("transport") or "stdio").strip().lower()
        if transport not in {"stdio", "http"}:
            continue
        command = str(spec.get("command") or "").strip()
        url = str(spec.get("url") or "").strip()
        if transport == "stdio" and not command:
            continue
        if transport == "http" and not url:
            continue
        raw_env = spec.get("env") or {}
        raw_headers = spec.get("headers") or {}
        servers[str(name)] = McpServerConfig(
            name=str(name),
            command=command,
            args=[str(arg) for arg in spec.get("args") or []],
            env={str(key): expand_env_value(str(value))
                 for key, value in raw_env.items()},
            transport=transport,  # type: ignore[arg-type]
            url=url,
            headers={str(key): expand_env_value(str(value))
                     for key, value in raw_headers.items()},
            enabled=bool(spec.get("enabled", True)),
            timeout_ms=int(spec.get("timeout_ms") or DEFAULT_CALL_TIMEOUT_MS),
            allowed_tools=([str(pattern) for pattern in spec.get("allowed_tools") or []]
                           or None),
        )
    return servers


def tool_full_name(server: str, tool: str) -> str:
    return f"{MCP_PREFIX}{server}__{tool}"


def split_tool_name(name: str) -> tuple[str, str] | None:
    parts = name.split("__")
    if len(parts) == 3 and parts[0] == MCP_PREFIX.rstrip("__"):
        return parts[1], parts[2]
    return None


def render_mcp_result(result) -> str:
    """把 MCP CallToolResult 渲染为给模型的文本（text 块 + structuredContent）。"""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        parts.append(str(text) if text else f"[{getattr(block, 'type', 'content')}]")
    structured = getattr(result, "structuredContent", None)
    if structured:
        parts.append("structured: " + json.dumps(
            structured, ensure_ascii=False, default=str))
    return "\n".join(parts)


@asynccontextmanager
async def _connect_streams(config: McpServerConfig):
    """按 transport 打开读写流；HTTP 客户端由本函数管理生命周期。"""
    if config.transport == "http":
        client = create_mcp_http_client(
            headers=config.headers or None,
            timeout=max(config.timeout_ms, 1000) / 1000)
        async with client:
            async with streamable_http_client(
                    config.url, http_client=client,
                    terminate_on_close=False) as streams:
                # SDK 2.3 的不同小版本分别返回二元组/三元组；多出的 session id
                # 不参与 ClientSession 初始化。
                yield streams[0], streams[1]
        return
    # 子进程环境沿用命令工具的密钥过滤；配置 env 里的 {env:VAR} 已展开。
    merged_env = {**_secret_free_env(), **config.env}
    params = StdioServerParameters(
        command=config.command, args=config.args, env=merged_env)
    async with stdio_client(params) as (read, write):
        yield read, write


class _ServerHandle:
    """单个 MCP 服务器：后台线程跑事件循环，保持 stdio 会话常开。"""

    def __init__(self, config: McpServerConfig) -> None:
        self.config = config
        self.loop: asyncio.AbstractEventLoop | None = None
        self.session: ClientSession | None = None
        self._stop: asyncio.Event | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self.start_error: str | None = None

    def start(self) -> bool:
        self._thread = threading.Thread(
            target=self._run, name=f"mcp-{self.config.name}", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=DEFAULT_CONNECT_TIMEOUT):
            self.start_error = f"连接超时（>{DEFAULT_CONNECT_TIMEOUT:.0f}s）"
        if self.start_error is not None:
            self.close()
            return False
        return True

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self.loop = loop
        asyncio.set_event_loop(loop)
        stop = asyncio.Event()
        self._stop = stop
        try:
            loop.run_until_complete(self._serve(stop))
        except Exception as exc:  # noqa: BLE001 - 启动失败记录原因，不影响其他服务器
            if self.start_error is None:
                self.start_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._ready.set()
            try:
                loop.close()
            except Exception:  # noqa: BLE001 - 关闭残留循环的异常无需上抛
                pass

    async def _serve(self, stop: asyncio.Event) -> None:
        async with _connect_streams(self.config) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.session = session
                self._ready.set()
                await stop.wait()
        self.session = None

    def submit(self, coroutine, timeout_s: float):
        loop = self.loop
        if loop is None or not loop.is_running():
            raise ToolError(f"MCP 服务器 {self.config.name} 未连接")
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result(
            timeout=timeout_s)

    def close(self, timeout: float = 5.0) -> None:
        stop = self._stop
        loop = self.loop
        if stop is not None and loop is not None and loop.is_running():
            loop.call_soon_threadsafe(stop.set)
        if self._thread is not None:
            self._thread.join(timeout=timeout)


class McpManager:
    """已连接 MCP 服务器的同步门面；connect_all 后 tools 提供工具元数据。"""

    def __init__(self, config_path: Path) -> None:
        self.config_path = Path(config_path)
        self.servers = load_mcp_config(self.config_path)
        self.handles: dict[str, _ServerHandle] = {}
        # full name → {server, tool, description, inputSchema}
        self.tools: dict[str, dict] = {}
        self.errors: list[str] = []

    @classmethod
    def from_task(cls, settings: Settings, task_dir: Path) -> "McpManager | None":
        """配置文件存在才创建；否则返回 None（评测任务零开销）。"""
        path = Path(task_dir) / settings.mcp_config_file
        if not path.is_file():
            return None
        return cls(path)

    def connect_all(self) -> None:
        for config in self.servers.values():
            if not config.enabled:
                continue
            handle = _ServerHandle(config)
            if not handle.start():
                self.errors.append(f"{config.name}: {handle.start_error}")
                continue
            self.handles[config.name] = handle
            try:
                listing = handle.submit(
                    _list_tools_coro(handle), DEFAULT_CONNECT_TIMEOUT)
            except Exception as exc:  # noqa: BLE001 - 单服务器失败不影响其余
                self.errors.append(
                    f"{config.name}: 列出工具失败 {type(exc).__name__}: {exc}")
                continue
            for tool in getattr(listing, "tools", None) or []:
                if config.allowed_tools and not any(
                        fnmatch.fnmatch(tool.name, pattern)
                        for pattern in config.allowed_tools):
                    continue
                # mcp 2.x 字段为 snake_case（input_schema）；兼容旧 camelCase。
                input_schema = (getattr(tool, "input_schema", None)
                                or getattr(tool, "inputSchema", None)
                                or {"type": "object", "properties": {}})
                self.tools[tool_full_name(config.name, tool.name)] = {
                    "server": config.name,
                    "tool": tool.name,
                    "description": tool.description or "",
                    "inputSchema": input_schema,
                }

    def call_tool(self, server: str, tool: str, args: dict):
        handle = self.handles.get(server)
        if handle is None:
            raise ToolError(f"MCP 服务器未连接：{server}")
        timeout_s = max(handle.config.timeout_ms, 1000) / 1000
        try:
            return handle.submit(
                _call_tool_coro(handle, tool, args), timeout_s)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - 超时/协议错误统一转为可读信息
            raise ToolError(
                f"MCP 调用失败 {server}::{tool}：{type(exc).__name__}: {exc}") from exc

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


async def _list_tools_coro(handle: _ServerHandle):
    assert handle.session is not None
    return await handle.session.list_tools()


async def _call_tool_coro(handle: _ServerHandle, tool: str, args: dict):
    assert handle.session is not None
    return await handle.session.call_tool(tool, dict(args))
