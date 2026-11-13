"""技能工具：按名加载 SKILL.md 全文；目录注入由运行循环完成。"""

from nextcoder.runtime.skills import SkillCatalog
from nextcoder.tools.base import ToolContext, ToolResult


def load_skill(context: ToolContext, call_id: str, name: str) -> ToolResult:
    catalog: SkillCatalog = context.skills
    skill = catalog.find(str(name))
    if skill is None:
        names = ", ".join(item.name for item in catalog.skills) or "（无）"
        return ToolResult(
            call_id, "load_skill", "failed",
            error=f"技能不存在：{name}；可用技能：{names}")
    try:
        body = skill.path.read_text(encoding="utf-8")
    except OSError as exc:
        return ToolResult(call_id, "load_skill", "failed", error=f"读取失败：{exc}")
    return ToolResult(call_id, "load_skill", "completed", output_excerpt=body)
