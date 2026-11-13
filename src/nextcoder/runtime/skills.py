"""技能发现：SKILL.md 目录注入与按名加载。

项目技能目录 <task_dir>/.nextcoder/skills 优先于 settings.skills_roots 与
~/.nextcoder/skills；同名技能以靠前来源为准。目录注入受字符预算限制，
完整指导通过 load_skill 按需读取。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from nextcoder.settings import Settings

SKILL_FILE = "SKILL.md"
_MAX_DESC_CHARS = 240
_HEADING = re.compile(r"^#{1,6}\s+(.+)")


@dataclass(frozen=True)
class SkillInfo:
    name: str
    description: str
    path: Path


@dataclass(frozen=True)
class SkillCatalog:
    skills: tuple[SkillInfo, ...] = ()

    def find(self, name: str) -> SkillInfo | None:
        return next((skill for skill in self.skills if skill.name == name), None)


def parse_skill_markdown(text: str) -> tuple[str, str]:
    """解析 frontmatter 的 name/description；缺 name 时回退首个标题。"""
    meta: dict[str, str] = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            body = parts[2].lstrip("\n")
            for line in parts[1].splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    meta[key.strip().lower()] = value.strip()
    name = meta.get("name", "")
    description = meta.get("description", "")
    if not name:
        for line in body.splitlines():
            match = _HEADING.match(line.strip())
            if match:
                name = match.group(1).strip()
                break
    return name, description


def discover_skills(task_dir: Path, settings: Settings) -> SkillCatalog:
    if not settings.skills_enabled:
        return SkillCatalog()
    roots = [
        Path(task_dir) / settings.runs_root / "skills",
        *(Path(root).expanduser() for root in settings.skills_roots),
        Path.home() / ".nextcoder" / "skills",
    ]
    found: dict[str, SkillInfo] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            skill_md = entry / SKILL_FILE
            if not (entry.is_dir() and skill_md.is_file()):
                continue
            try:
                text = skill_md.read_text(encoding="utf-8")
            except OSError:
                continue
            name, description = parse_skill_markdown(text)
            # 名字会进入工具调用参数与提示，禁止路径形态，防目录逃逸。
            if not name or "/" in name or "\\" in name or name in {".", ".."}:
                continue
            found.setdefault(name, SkillInfo(name, description, skill_md))
    return SkillCatalog(tuple(found.values()))


def catalog_text(catalog: SkillCatalog, budget_chars: int) -> str:
    """技能目录注入文本；无技能或预算为零时返回空串（零开销）。"""
    if not catalog.skills or budget_chars <= 0:
        return ""
    header = "[项目技能] 调用 load_skill(name) 获取完整指导："
    lines = [header]
    used = len(header)
    shown = 0
    for skill in catalog.skills:
        description = skill.description or ""
        entry = f"- {skill.name}：{description[:_MAX_DESC_CHARS]}"
        if used + len(entry) > budget_chars:
            break
        lines.append(entry)
        used += len(entry)
        shown += 1
    if shown < len(catalog.skills):
        lines.append(f"（另有 {len(catalog.skills) - shown} 个技能未列出，"
                     "总数见 load_skill 的失败提示）")
    return "\n".join(lines)
