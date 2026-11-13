"""项目指令文件：会话开始时读取仓库根的 AGENTS.md。

内容只作为项目约定放进稳定系统前缀，不做环境变量展开、不执行任何命令。
会话中途修改文件不会改变当前会话的前缀；下一会话才生效。
"""

from __future__ import annotations

from pathlib import Path

from nextcoder.settings import Settings


def load_agents_file(task_dir: Path, settings: Settings) -> str:
    """读取并限长；文件不存在、不可读或为空时静默返回空串。"""
    if not settings.agents_enabled:
        return ""
    name = settings.agents_file.strip()
    if not name:
        return ""
    root = Path(task_dir).resolve()
    try:
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""
    text = text.strip()
    if not text:
        return ""
    limit = max(0, int(settings.agents_max_chars))
    if limit and len(text) > limit:
        text = text[:limit].rstrip() + "\n\n[AGENTS.md 已截断]"
    return text
