"""AGENTS.md 项目指令：读取、截断、路径约束与会话冻结。"""

from pathlib import Path

from nextcoder.runtime.instructions import load_agents_file
from nextcoder.sessions import manager
from nextcoder.settings import Settings


def _settings(**overrides) -> Settings:
    values = dict(llm_base_url="", llm_model="", agents_max_chars=8000)
    values.update(overrides)
    return Settings(**values)


def test_loads_root_agents_file(tmp_path):
    (tmp_path / "AGENTS.md").write_text("使用 uv 管理依赖。\n", encoding="utf-8")
    assert load_agents_file(tmp_path, _settings()) == "使用 uv 管理依赖。"


def test_missing_or_empty_file_is_ignored(tmp_path):
    assert load_agents_file(tmp_path, _settings()) == ""
    (tmp_path / "AGENTS.md").write_text("   \n", encoding="utf-8")
    assert load_agents_file(tmp_path, _settings()) == ""


def test_disabled_and_custom_name(tmp_path):
    (tmp_path / "AGENTS.md").write_text("默认", encoding="utf-8")
    (tmp_path / "TEAM.md").write_text("团队规则", encoding="utf-8")
    assert load_agents_file(tmp_path, _settings(agents_enabled=False)) == ""
    assert load_agents_file(tmp_path, _settings(agents_file="TEAM.md")) == "团队规则"


def test_long_file_is_truncated_with_marker(tmp_path):
    (tmp_path / "AGENTS.md").write_text("x" * 50, encoding="utf-8")
    text = load_agents_file(tmp_path, _settings(agents_max_chars=10))
    assert text.startswith("x" * 10)
    assert "[AGENTS.md 已截断]" in text


def test_agents_file_cannot_escape_task_dir(tmp_path):
    secret = tmp_path.parent / "secret.md"
    secret.write_text("secret", encoding="utf-8")
    task = tmp_path / "task"
    task.mkdir()
    assert load_agents_file(task, _settings(agents_file="../secret.md")) == ""
    assert load_agents_file(task, _settings(agents_file=str(secret))) == ""


def test_session_freezes_agents_text_until_reload(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "AGENTS.md").write_text("第一版", encoding="utf-8")
    settings = _settings()
    session = manager.create_session(settings, task)
    assert session.agents_text == "第一版"

    (task / "AGENTS.md").write_text("第二版", encoding="utf-8")
    assert session.agents_text == "第一版"

    loaded = manager.load_session(settings, task, session.session_id)
    assert loaded.agents_text == "第二版"


def test_skills_catalog_is_frozen_at_session_start(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    settings = _settings(skills_roots=[str(tmp_path / "skills")])
    skill = Path(tmp_path / "skills" / "testing")
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: testing\ndescription: 测试规范\n---\n正文", encoding="utf-8")

    session = manager.create_session(settings, task)
    assert [item.name for item in session.skills_catalog.skills] == ["testing"]

    extra = Path(tmp_path / "skills" / "extra")
    extra.mkdir()
    (extra / "SKILL.md").write_text(
        "---\nname: extra\ndescription: 新增技能\n---\n正文", encoding="utf-8")
    assert [item.name for item in session.skills_catalog.skills] == ["testing"]

    loaded = manager.load_session(settings, task, session.session_id)
    assert {item.name for item in loaded.skills_catalog.skills} == {"testing", "extra"}
