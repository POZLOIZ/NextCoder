"""斜杠命令与 `@file` 提及的补全数据源。

这里只做纯文本匹配，不依赖 Textual；界面层负责把候选渲染成提示条。
文件索引按任务目录缓存，目录遍历有数量上限，避免大仓库拖慢输入。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

_SKIP_DIRS = {
    ".git", ".nextcoder", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    "target", ".idea", ".vscode", "site-packages",
}
_FILE_TOKEN = re.compile(r"(?:^|[\s(])@([^\s@]*)$")
MAX_INDEX_ENTRIES = 4000


@dataclass(frozen=True)
class SlashCommand:
    name: str
    usage: str
    description: str
    aliases: tuple[str, ...] = ()

    @property
    def token(self) -> str:
        return self.name[1:]


COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("/help", "/help", "帮助、快捷键与命令说明"),
    SlashCommand("/new", "/new", "新建一个会话"),
    SlashCommand("/sessions", "/sessions", "切换最近的会话"),
    SlashCommand("/config", "/config", "配置 endpoint、模型、effort、权限与沙箱"),
    SlashCommand("/model", "/model", "切换模型 profile / endpoint"),
    SlashCommand("/mode", "/mode [manual|workspace|auto|bypass]",
                 "逐项确认 / 工作区写入 / 沙箱内自动 / 完全放行"),
    SlashCommand("/sandbox", "/sandbox [on|off|network on|off|status]",
                 "查看或切换 bubblewrap 沙箱"),
    SlashCommand("/plan", "/plan", "显示或隐藏任务清单面板"),
    SlashCommand("/compact", "/compact", "压缩当前会话上下文"),
    SlashCommand("/export", "/export [path]", "把当前对话导出为 Markdown"),
    SlashCommand("/clear", "/clear", "清空屏幕（不删除会话记录）"),
    SlashCommand("/status", "/status", "显示当前会话、模型与运行状态"),
    SlashCommand("/quit", "/quit", "退出 NextCoder"),
)


def command_token(text: str) -> str | None:
    """返回正在输入的 `/xxx` 片段；出现空格或换行后不再提示。"""
    if "\n" in text or any(ch.isspace() for ch in text):
        return None
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    return stripped


def match_commands(text: str, limit: int = 8) -> list[SlashCommand]:
    token = command_token(text)
    if token is None:
        return []
    query = token[1:].casefold()
    matched = [
        command for command in COMMANDS
        if not query
        or command.token.casefold().startswith(query)
        or any(alias.casefold().startswith(query) for alias in command.aliases)
    ]
    matched.sort(key=lambda command: (
        not command.token.casefold().startswith(query), command.name))
    return matched[:limit]


def file_token(text: str) -> tuple[int, str] | None:
    """返回 (替换起点, 已输入路径)；只匹配光标前最后一个 `@token`。

    Composer 目前把光标视为文本末尾；`@` 必须位于行首或空白/括号之后，
    避免把邮箱地址误判成文件提及。
    """
    match = _FILE_TOKEN.search(text)
    if match is None:
        return None
    prefix = match.group(1)
    return match.start(1) - 1, prefix


def build_file_index(task_dir: Path, limit: int = MAX_INDEX_ENTRIES) -> list[str]:
    """遍历任务目录，生成相对路径列表；目录带结尾 `/`。"""
    root = Path(task_dir)
    entries: list[str] = []
    try:
        walker = os.walk(root)
        for current, dirs, files in walker:
            dirs[:] = sorted(
                name for name in dirs
                if name not in _SKIP_DIRS and name != ".git")
            base = Path(current)
            for name in dirs:
                entries.append((base / name).relative_to(root).as_posix() + "/")
                if len(entries) >= limit:
                    return sorted(entries)
            for name in sorted(files):
                if name.endswith((".pyc", ".pyo")):
                    continue
                entries.append((base / name).relative_to(root).as_posix())
                if len(entries) >= limit:
                    return sorted(entries)
    except OSError:
        return sorted(entries)
    return sorted(entries)


def _rank(path: str) -> tuple[int, bool, str]:
    """浅层优先；同层目录优先于文件，便于先进入目录再选文件。"""
    depth = path.rstrip("/").count("/")
    return (depth, not path.endswith("/"), path.casefold())


def match_files(paths: list[str], prefix: str, limit: int = 8) -> list[str]:
    query = prefix.replace("\\", "/")
    if query.startswith("./"):
        # 只去掉显式的 "./"；不能 lstrip("./")，否则 .github/.env 会被误伤。
        query = query[2:]
    query = query.casefold()
    if not query:
        # 空查询优先展示顶层条目，避免一上来就是深层路径。
        return sorted(paths, key=_rank)[:limit]
    starts = [path for path in paths if path.casefold().startswith(query)]
    contains = [
        path for path in paths
        if path not in starts and query in path.casefold()
    ]
    ranked = sorted(starts, key=_rank)
    ranked.extend(sorted(contains, key=_rank))
    return ranked[:limit]


def apply_completion(text: str, replacement: str, *, command: bool) -> str:
    """把当前输入中的 `/token` 或 `@token` 替换成完整候选。"""
    if command:
        return replacement + " "
    token = file_token(text)
    if token is None:
        return text
    start, _prefix = token
    return text[:start] + "@" + replacement + " "
