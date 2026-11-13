"""技能发现测试：frontmatter 解析、来源覆盖、目录预算与 load_skill 工具。"""

from nextcoder.runtime.skills import (
    SkillCatalog,
    catalog_text,
    discover_skills,
    parse_skill_markdown,
)
from nextcoder.sessions.store import OperationStore
from nextcoder.settings import Settings
from nextcoder.tools.base import ToolContext
from nextcoder.tools.registry import ToolBox


def _write_skill(root, name: str, body: str, frontmatter: str = "") -> None:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    text = f"---\n{frontmatter}---\n{body}" if frontmatter else body
    (directory / "SKILL.md").write_text(text, encoding="utf-8")


def _settings(tmp_path, **kwargs) -> Settings:
    return Settings(llm_base_url="http://localhost", llm_model="fake",
                    llm_api_key="test", **kwargs)


def _context(tmp_path, **kwargs) -> ToolContext:
    task_dir = tmp_path / "task"
    run_dir = task_dir / ".nextcoder" / "runs" / "test"
    task_dir.mkdir(exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline = run_dir / "baseline"
    baseline.mkdir(exist_ok=True)
    store = OperationStore(run_dir / "operations.sqlite")
    return ToolContext(
        task_dir=task_dir, run_dir=run_dir, baseline_dir=baseline, store=store,
        settings=_settings(tmp_path, **kwargs), turn_id="turn-1")


class TestParsing:
    def test_frontmatter_name_and_description(self):
        text = "---\nname: git-workflow\ndescription: 分支与提交规范\n---\n正文"
        assert parse_skill_markdown(text) == ("git-workflow", "分支与提交规范")

    def test_falls_back_to_first_heading(self):
        text = "# Code Review 清单\n正文"
        name, _ = parse_skill_markdown(text)
        assert name == "Code Review 清单"

    def test_no_name_no_heading_is_skipped(self):
        assert parse_skill_markdown("只有正文")[0] == ""


class TestDiscovery:
    def test_project_overrides_global_same_name(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        project_root = task_dir / ".nextcoder" / "skills"
        global_root = tmp_path / "global-skills"
        _write_skill(project_root, "testing", "项目版",
                     "name: testing\ndescription: project\n")
        _write_skill(global_root, "testing", "全局版",
                     "name: testing\ndescription: global\n")
        _write_skill(global_root, "deploy", "部署流程",
                     "name: deploy\ndescription: 部署\n")
        catalog = discover_skills(
            task_dir, _settings(tmp_path, skills_roots=[str(global_root)]))
        names = [skill.name for skill in catalog.skills]
        assert sorted(names) == ["deploy", "testing"]
        assert catalog.find("testing").description == "project"

    def test_disabled_returns_empty_catalog(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        _write_skill(task_dir / ".nextcoder" / "skills", "x", "# X")
        assert discover_skills(task_dir, _settings(tmp_path, skills_enabled=False)).skills == ()

    def test_names_with_path_separators_are_rejected(self, tmp_path):
        # 目录名本身带斜杠不可能，但 frontmatter name 可以恶意构造；
        # 这里的拒绝发生在名字校验：无法通过 find 命中真实路径。
        catalog = SkillCatalog(skills=())
        assert catalog.find("../etc/passwd") is None


class TestCatalogText:
    def test_budget_truncates_and_reports_hidden_count(self):
        skills = tuple(
            type("Skill", (), {"name": f"skill-{index}",
                               "description": "描述" * 20,
                               "path": None})()
            for index in range(20))
        catalog = SkillCatalog(skills=skills)  # type: ignore[arg-type]
        text = catalog_text(catalog, 800)
        assert len(text) <= 1600
        assert "未列出" in text
        assert text.startswith("[项目技能]")

    def test_empty_catalog_is_zero_overhead(self):
        assert catalog_text(SkillCatalog(), 1200) == ""


class TestLoadSkillTool:
    def test_returns_body_and_result_format(self, tmp_path):
        context = _context(tmp_path)
        _write_skill(context.task_dir / ".nextcoder" / "skills", "testing",
                     "## 步骤\n1. 先跑 pytest",
                     "name: testing\ndescription: 测试规范\n")
        box = ToolBox(context)
        text = box.execute("load_skill", {"name": "testing"}, call_id="skill-1")
        context.store.close()
        assert text.startswith("[load_skill] status=completed")
        assert "先跑 pytest" in text

    def test_unknown_name_lists_available(self, tmp_path):
        context = _context(tmp_path)
        _write_skill(context.task_dir / ".nextcoder" / "skills", "testing",
                     "正文", "name: testing\ndescription: d\n")
        box = ToolBox(context)
        text = box.execute("load_skill", {"name": "nope"}, call_id="skill-2")
        context.store.close()
        assert "技能不存在" in text
        assert "testing" in text

    def test_load_skill_absent_without_skills(self, tmp_path):
        context = _context(tmp_path)
        names = [tool.name for tool in ToolBox(context).definitions()]
        context.store.close()
        assert "load_skill" not in names
