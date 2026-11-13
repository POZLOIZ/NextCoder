"""MCP 测试：命名/渲染纯函数、配置展开、传输选择，以及 stdio 真实集成。"""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from nextcoder.runtime import mcp as mcp_module
from nextcoder.runtime.mcp import (
    McpManager,
    McpServerConfig,
    _connect_streams,
    expand_env_value,
    load_mcp_config,
    render_mcp_result,
    split_tool_name,
    tool_full_name,
)
from nextcoder.sessions.store import OperationStore
from nextcoder.settings import Settings
from nextcoder.tools.base import ToolContext
from nextcoder.tools.registry import ToolBox, schema_to_args_model

# mcp 2.x：FastMCP 更名为 MCPServer。
_SERVER_SCRIPT = """
from mcp.server.mcpserver import MCPServer

server = MCPServer(name="demo")

def add(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b

server.add_tool(add)
server.run()
"""


def _settings(**kwargs) -> Settings:
    return Settings(llm_base_url="http://localhost", llm_model="fake",
                    llm_api_key="test", **kwargs)


def _write_config(tmp_path, servers: dict) -> Path:
    path = tmp_path / ".nextcoder" / "mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"servers": servers}, ensure_ascii=False),
                    encoding="utf-8")
    return path


class TestPureHelpers:
    def test_tool_naming_roundtrip(self):
        assert tool_full_name("fetch", "get") == "mcp__fetch__get"
        assert split_tool_name("mcp__fetch__get") == ("fetch", "get")
        assert split_tool_name("not_mcp") is None

    def test_render_text_and_structured(self):
        class Block:
            type = "text"
            text = "hello"

        class Result:
            content = [Block()]
            structuredContent = {"value": 3}

        rendered = render_mcp_result(Result())
        assert "hello" in rendered
        assert '"value": 3' in rendered

    def test_env_expansion(self, monkeypatch):
        monkeypatch.setenv("NC_TEST_TOKEN", "secret-value")
        assert expand_env_value("Bearer {env:NC_TEST_TOKEN}") == "Bearer secret-value"
        assert expand_env_value("{env:NC_UNDEFINED}") == ""

    def test_config_load_skips_invalid_and_missing(self, tmp_path):
        assert load_mcp_config(tmp_path / ".nextcoder" / "mcp.json") == {}
        path = _write_config(tmp_path, {
            "good": {"command": "python"},
            "no-command": {"args": ["x"]},
            "disabled": {"command": "python", "enabled": False},
            "no-url": {"transport": "http"},
            "bad-transport": {"transport": "websocket", "command": "python"},
        })
        config = load_mcp_config(path)
        assert set(config) == {"good", "disabled"}
        assert not config["disabled"].enabled

    def test_http_config_parses_transport_headers_and_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NC_TEST_TOKEN", "secret-value")
        path = _write_config(tmp_path, {
            "remote": {
                "transport": "http",
                "url": "https://example.test/mcp",
                "headers": {"Authorization": "Bearer {env:NC_TEST_TOKEN}"},
                "timeout_ms": 5000,
            },
        })
        config = load_mcp_config(path)["remote"]
        assert config.transport == "http"
        assert config.url == "https://example.test/mcp"
        assert config.headers["Authorization"] == "Bearer secret-value"
        assert config.timeout_ms == 5000
        assert config.command == ""


class TestTransportSelection:
    def test_stdio_uses_command_and_args(self, tmp_path, monkeypatch):
        seen = []

        @asynccontextmanager
        async def fake_stdio(params):
            seen.append(params)
            yield ("read", "write")

        monkeypatch.setattr(mcp_module, "stdio_client", fake_stdio)

        async def run():
            config = McpServerConfig(name="local", command="python",
                                     args=["server.py"])
            async with _connect_streams(config) as streams:
                assert streams == ("read", "write")

        asyncio.run(run())
        assert seen[0].command == "python"
        assert seen[0].args == ["server.py"]

    def test_http_passes_headers_and_adapts_streams(self, monkeypatch):
        client_kwargs = {}
        stream_calls = []

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc_info):
                return False

        def fake_create(**kwargs):
            client_kwargs.update(kwargs)
            return FakeClient()

        @asynccontextmanager
        async def fake_streamable(url, *, http_client, terminate_on_close):
            stream_calls.append((url, http_client, terminate_on_close))
            # 兼容 SDK 2.3 的三元组；_connect_streams 只取前两个。
            yield ("read", "write", "session-id")

        monkeypatch.setattr(mcp_module, "create_mcp_http_client", fake_create)
        monkeypatch.setattr(mcp_module, "streamable_http_client", fake_streamable)

        async def run():
            config = McpServerConfig(
                name="remote", transport="http", url="https://example.test/mcp",
                headers={"Authorization": "Bearer token"}, timeout_ms=4000)
            async with _connect_streams(config) as streams:
                assert streams == ("read", "write")

        asyncio.run(run())
        assert client_kwargs["headers"] == {"Authorization": "Bearer token"}
        assert client_kwargs["timeout"] == 4.0
        assert stream_calls[0][0] == "https://example.test/mcp"
        assert stream_calls[0][2] is False


class TestSchemaModel:
    def test_json_schema_to_pydantic(self):
        model = schema_to_args_model("mcp__demo__add", {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "left"},
                "b": {"type": "integer"},
                "note": {"type": "string"},
            },
            "required": ["a", "b"],
        })
        instance = model(a=1, b=2)  # type: ignore[call-arg]
        assert instance.a == 1 and instance.b == 2
        assert instance.note is None


class TestStdioIntegration:
    def test_manager_missing_config_is_none(self, tmp_path):
        assert McpManager.from_task(_settings(), tmp_path / "task-missing") is None

    def test_connect_list_call_and_toolbox_roundtrip(self, tmp_path):
        script = tmp_path / "demo_server.py"
        script.write_text(_SERVER_SCRIPT, encoding="utf-8")
        config_path = _write_config(tmp_path, {
            "demo": {"command": sys.executable, "args": [str(script)]},
        })
        task_dir = tmp_path / "task"
        run_dir = task_dir / ".nextcoder" / "runs" / "test"
        task_dir.mkdir()
        run_dir.mkdir(parents=True)
        baseline = run_dir / "baseline"
        baseline.mkdir()
        store = OperationStore(run_dir / "operations.sqlite")
        context = ToolContext(
            task_dir=task_dir, run_dir=run_dir, baseline_dir=baseline, store=store,
            settings=_settings(permission_mode="bypass"), turn_id="turn-1")
        manager = McpManager(config_path)
        try:
            manager.connect_all()
            assert manager.errors == []
            assert list(manager.tools) == ["mcp__demo__add"]
            assert manager.tools["mcp__demo__add"]["server"] == "demo"

            box = ToolBox(context, mcp=manager)
            names = [tool.name for tool in box.definitions()]
            assert "mcp__demo__add" in names

            text = box.execute("mcp__demo__add", {"a": 20, "b": 22},
                               call_id="mcp-call-1")
            operation = store.get("mcp-call-1")
            assert text.startswith("[mcp__demo__add] status=completed")
            assert "42" in text
            assert operation is not None and operation.status == "completed"
        finally:
            manager.close()
            store.close()

    def test_allowed_tools_filter(self, tmp_path):
        script = tmp_path / "demo_server.py"
        script.write_text(_SERVER_SCRIPT, encoding="utf-8")
        config_path = _write_config(tmp_path, {
            "demo": {"command": sys.executable, "args": [str(script)],
                     "allowed_tools": ["missing_*"]},
        })
        manager = McpManager(config_path)
        try:
            manager.connect_all()
            assert manager.tools == {}
        finally:
            manager.close()

    def test_broken_server_records_error_without_raising(self, tmp_path):
        config_path = _write_config(tmp_path, {
            "broken": {"command": sys.executable,
                       "args": ["-c", "import sys; sys.exit(3)"]},
        })
        manager = McpManager(config_path)
        try:
            manager.connect_all()
            assert manager.tools == {}
            assert manager.errors and "broken" in manager.errors[0]
            with pytest.raises(Exception, match="未连接"):
                manager.call_tool("broken", "anything", {})
        finally:
            manager.close()
