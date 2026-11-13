"""项目记忆测试：快照读取、主题更新、手工保护、压缩整理与记忆工具。"""

import json
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from nextcoder.runtime.agent import AgentRunner
from nextcoder.sessions import manager
from nextcoder.sessions import memory as memory_module
from nextcoder.sessions.memory import ProjectMemory
from nextcoder.sessions.store import OperationStore
from nextcoder.settings import Settings
from nextcoder.tools.base import ToolContext
from nextcoder.tools.memory import memory_forget, memory_list, memory_read, memory_write


def _settings(**overrides) -> Settings:
    values = dict(
        llm_base_url="http://localhost", llm_model="fake", llm_api_key="test",
        memory_prompt_limit=8, memory_snapshot_chars=1600,
        application_input_token_budget=100000,
    )
    values.update(overrides)
    return Settings(**values)


def _memory(tmp_path, **overrides) -> ProjectMemory:
    task = tmp_path / "repo"
    task.mkdir(exist_ok=True)
    return ProjectMemory(_settings(**overrides), task)


def _context(tmp_path, memory: ProjectMemory) -> ToolContext:
    task = memory._task_dir
    run_dir = task / ".nextcoder" / "runs" / "test"
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline = run_dir / "baseline"
    baseline.mkdir(exist_ok=True)
    return ToolContext(
        task_dir=task, run_dir=run_dir, baseline_dir=baseline,
        store=OperationStore(run_dir / "operations.sqlite"),
        settings=_settings(), turn_id="turn-1", memory=memory,
        memory_snapshot=memory.snapshot(),
    )


def test_remember_updates_same_topic_instead_of_adding_conflict(tmp_path):
    memory = _memory(tmp_path)
    key = memory.remember("test-cmd", "python -m pytest -q", kind="test_command",
                          source="agent:turn-1", confidence=0.7)
    memory.remember("test-cmd", "uv run pytest -q", kind="test_command",
                    source="agent:turn-2", confidence=0.8)

    items = [item for item in memory.select() if item.kind == "test_command"]
    assert len(items) == 1
    assert items[0].key == key
    assert items[0].value == "uv run pytest -q"
    assert any(event["action"] == "replaced" for event in memory.events_since(0))
    memory.close()


def test_snapshot_injects_index_and_points_to_detail_files(tmp_path):
    memory = _memory(tmp_path, memory_prompt_limit=3, memory_snapshot_chars=300)
    for index in range(6):
        memory.remember(f"topic-{index}", f"正文 {index}", kind="project_convention",
                        source=f"agent:{index}")

    snapshot = memory.snapshot()
    assert len(snapshot.entries) == 3
    assert snapshot.keys == [item.key for item in snapshot.entries]
    assert snapshot.text.startswith("# 项目记忆")
    assert ".nextcoder/memory/" in snapshot.text, "索引必须给出详情文件路径"
    assert "topic-" in snapshot.text
    memory.close()


def test_manual_memory_is_not_overwritten_by_summary(tmp_path):
    memory = _memory(tmp_path)
    memory.set_manual("tests/run-mode", "project_convention", "必须用 uv run pytest")

    memory.remember("tests/run-mode", "摘要建议改成 pytest", kind="project_convention",
                    source="summary:summary-1")
    item = memory.get("tests/run-mode")
    assert item is not None
    assert item.value == "必须用 uv run pytest"
    assert item.meta.get("manual") is True
    memory.close()


def test_manual_disable_is_not_revived_by_automatic_paths(tmp_path):
    memory = _memory(tmp_path)
    memory.set_manual("tests/run-mode", "project_convention", "必须用 uv run pytest")
    assert memory.disable("tests/run-mode")

    memory.remember("tests/run-mode", "摘要建议改成 pytest", kind="project_convention",
                    source="summary:summary-2")
    memory.remember("tests/run-mode", "Agent 重新写入", kind="project_convention",
                    source="agent:turn-9")
    item = memory.get("tests/run-mode")
    assert item is not None
    assert item.invalidated_at is not None
    assert memory.select() == []
    memory.close()


def test_consolidation_applies_once_per_summary(tmp_path):
    memory = _memory(tmp_path)
    suggestions = [{
        "topic": "test-cmd", "kind": "test_command",
        "body": "uv run pytest -q", "evidence": "两次会话都使用该命令",
    }]

    first = memory.consolidate(
        summary_id="summary-1", covered_count=12, suggestions=suggestions)
    assert first["status"] == "applied"
    assert [item.key for item in memory.select()] == ["test-cmd"]
    created = [event for event in memory.events_since(0)
               if event["action"] == "created" and event["key"] == "test-cmd"]
    assert len(created) == 1

    second = memory.consolidate(
        summary_id="summary-1", covered_count=12, suggestions=suggestions)
    assert second["summary_id"] == "summary-1"
    assert second["status"] == "applied"
    created = [event for event in memory.events_since(0)
               if event["action"] == "created" and event["key"] == "test-cmd"]
    assert len(created) == 1, "同一摘要不能重复应用记忆建议"
    memory.close()


def test_consolidation_with_no_suggestions_is_recorded_as_empty(tmp_path):
    memory = _memory(tmp_path)
    result = memory.consolidate(summary_id="summary-empty", covered_count=4,
                                suggestions=[])
    assert result["status"] == "empty"
    assert memory.select() == []
    memory.close()


def test_memory_never_expires_by_time(tmp_path):
    # TTL 已按设计移除：时间流逝不改变条目可见性，停用只能显式触发。
    task = tmp_path / "repo"
    task.mkdir()
    clock = [1_700_000_000.0]
    memory = ProjectMemory(_settings(), task, clock=lambda: clock[0])
    memory.remember("test-cmd", "pytest -q", kind="test_command",
                    source="agent:turn-1")
    clock[0] += 365 * 86400
    assert [item.key for item in memory.select()] == ["test-cmd"]
    assert not hasattr(memory, "expire")
    memory.close()


def test_memory_md_file_is_rewritten_on_every_change(tmp_path):
    memory = _memory(tmp_path)
    memory.remember("test-cmd", "pytest -q", kind="test_command",
                    source="agent:turn-1")
    path = memory.memory_file_path()
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "test-cmd" in content and "pytest -q" in content

    memory.remember("lint-cmd", "ruff check .", kind="lint_command",
                    source="agent:turn-2")
    content = path.read_text(encoding="utf-8")
    assert "test-cmd" in content and "lint-cmd" in content

    # 排序：test_command 优先于 lint_command。
    assert content.index("test-cmd") < content.index("lint-cmd")

    memory.disable("test-cmd")
    content = path.read_text(encoding="utf-8")
    assert "test-cmd" not in content and "lint-cmd" in content
    memory.close()


def test_memory_detail_files_written_and_pruned(tmp_path):
    memory = _memory(tmp_path)
    body = "pytest -q " + "长正文 " * 40
    memory.remember("test-cmd", body, kind="test_command", source="agent:turn-1")
    memory.remember("lint-cmd", "ruff check .", kind="lint_command",
                    source="agent:turn-2")

    detail_dir = memory.memory_dir()
    names = sorted(path.name for path in detail_dir.glob("*.md"))
    assert len(names) == 2
    # 详情文件保存完整正文与来源，索引只留 80 字摘要。
    test_detail = next((detail_dir / name).read_text(encoding="utf-8")
                       for name in names if "test" in name)
    assert body[100:] in test_detail
    assert "agent:turn-1" in test_detail
    index_text = memory.memory_file_path().read_text(encoding="utf-8")
    assert body[100:] not in index_text

    # 停用条目：索引行与详情文件同时消失；手工塞入的孤儿文件也会被清理。
    orphan = detail_dir / "手工遗留.md"
    orphan.write_text("旧文件", encoding="utf-8")
    memory.disable("test-cmd")
    names = {path.name for path in detail_dir.glob("*.md")}
    assert not any("test-cmd" in name for name in names)
    assert "手工遗留.md" not in names
    assert memory.get("lint-cmd") is not None
    memory.close()


def test_memory_tools_list_read_write_and_forget(tmp_path):
    memory = _memory(tmp_path)
    context = _context(tmp_path, memory)

    written = memory_write(context, "call-1", "tests/run-mode",
                           "必须用 uv run pytest", kind="project_convention")
    assert written.status == "completed"
    assert "已保存" in written.output_excerpt

    listed = memory_list(context, "call-2", "uv run")
    assert "tests/run-mode" in listed.output_excerpt

    read = memory_read(context, "call-3", "tests/run-mode")
    assert read.status == "completed"
    assert "必须用 uv run pytest" in read.output_excerpt
    assert "agent:turn-1:memory_write" in read.output_excerpt

    forgotten = memory_forget(context, "call-4", "tests/run-mode")
    assert "已停用" in forgotten.output_excerpt
    assert memory.select() == []

    # 停用后再次写入不能复活（手工停用保护优先）。
    again = memory_write(context, "call-5", "tests/run-mode", "另一个值")
    assert "未覆盖" in again.output_excerpt
    assert memory.get("tests/run-mode").invalidated_at is not None
    memory.close()


def test_disabled_memory_is_not_in_snapshot(tmp_path):
    memory = _memory(tmp_path)
    memory.remember("test-cmd", "pytest -q", kind="test_command",
                    source="agent:turn-1")
    memory.disable("test-cmd")
    assert memory.snapshot().text == ""
    memory.close()


def test_automatic_extraction_interfaces_are_removed(tmp_path):
    # 第一版设计明确不做逐命令/逐用户消息自动提取，也不按文件清单刷新。
    for name in ("observe_operation", "observe_operations", "observe_user_message",
                 "refresh_project", "classify_command"):
        assert not hasattr(memory_module, name)
    memory = _memory(tmp_path)
    context = _context(tmp_path, memory)
    memory_list(context, "call-1")
    memory.close()


def test_session_start_loads_memory_snapshot(tmp_path):
    task = tmp_path / "repo"
    task.mkdir()
    settings = _settings()
    memory = ProjectMemory(settings, task)
    memory.remember("test-cmd", "pytest -q", kind="test_command",
                    source="agent:turn-1")
    memory.close()

    session = manager.create_session(settings, task)
    assert session.memory_snapshot is not None
    assert "test-cmd" in session.memory_snapshot.keys
    assert "pytest -q" in session.memory_snapshot.text


def test_snapshot_is_injected_into_stable_system_prefix(tmp_path):
    settings = _settings()
    memory = _memory(tmp_path)
    memory.remember("test-cmd", "python -m pytest -q", kind="test_command",
                    source="agent:turn-1")
    context = _context(tmp_path, memory)
    context.memory_snapshot = memory.snapshot()
    runner = AgentRunner(settings, context)
    runner._llm = _FakeLLM()

    runner.run("继续", "t-memory")

    sent = runner._llm.seen[0]
    assert isinstance(sent[0], SystemMessage)
    assert "# 项目记忆" in sent[0].content
    assert "python -m pytest -q" in sent[0].content
    assert "test-cmd" in runner.memory_used
    memory.close()
    context.store.close()


def test_memory_cli_lists_sets_and_disables(tmp_path):
    from typer.testing import CliRunner

    from nextcoder.cli import app

    task = tmp_path / "repo"
    task.mkdir()
    runner = CliRunner()
    key = "tests/run-mode"

    result = runner.invoke(app, [
        "memory", "set", "--task-dir", str(task), "--key", key,
        "--kind", "project_convention", "--value", "以后使用 uv run pytest",
    ])
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["memory", "list", "--task-dir", str(task)])
    assert result.exit_code == 0, result.output
    assert key in result.output and "以后使用 uv run pytest" in result.output

    result = runner.invoke(app, [
        "memory", "disable", "--task-dir", str(task), "--key", key,
    ])
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["memory", "list", "--task-dir", str(task)])
    assert key not in result.output
    result = runner.invoke(app, ["memory", "list", "--task-dir", str(task), "--all"])
    assert key in result.output and "停用" in result.output


class _FakeLLM:
    def __init__(self) -> None:
        self.seen: list[list] = []

    def invoke(self, messages):
        self.seen.append(list(messages))
        return AIMessage(content="完成")


class _PlanLLM:
    """返回固定整理计划的假模型；也可以让它抛错测失败路径。"""

    def __init__(self, payload) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def invoke(self, prompt):
        if isinstance(self.payload, Exception):
            raise self.payload
        self.prompts.append(prompt)
        return AIMessage(
            content=json.dumps(self.payload, ensure_ascii=False),
            usage_metadata={"input_tokens": 30, "output_tokens": 10,
                            "total_tokens": 40},
        )


def test_organizer_merges_duplicates_and_disables_stale(tmp_path):
    from nextcoder.sessions.organizer import MemoryOrganizer

    memory = _memory(tmp_path)
    memory.remember("test-cmd", "pytest -q", kind="test_command",
                    source="agent:turn-1")
    memory.remember("tests/run", "uv run pytest -q -x", kind="test_command",
                    source="agent:turn-2")
    memory.remember("old-approach", "以前手动改 JSON 配置", kind="project_convention",
                    source="agent:turn-1")

    plan = {
        "merges": [{
            "from": "tests/run", "into": "test-cmd",
            "value": "统一用 uv run pytest -q 运行测试", "kind": "test_command",
        }],
        "disables": [{"key": "old-approach", "reason": "已被配置脚本取代"}],
    }
    result = MemoryOrganizer(_settings(), llm=_PlanLLM(plan)).organize(memory)

    assert result["status"] == "applied"
    assert result["merged"] == 1 and result["disabled"] == 1
    merged = memory.get("test-cmd")
    assert merged.value == "统一用 uv run pytest -q 运行测试"
    assert memory.get("tests/run").invalidated_at is not None
    assert memory.get("old-approach").invalidated_at is not None
    assert [item.key for item in memory.select()] == ["test-cmd"]
    # memory.md 随整理自动重写。
    content = memory.memory_file_path().read_text(encoding="utf-8")
    assert "统一用 uv run pytest -q" in content
    assert "old-approach" not in content
    memory.close()


def test_organizer_never_touches_manual_entries(tmp_path):
    from nextcoder.sessions.organizer import MemoryOrganizer

    memory = _memory(tmp_path)
    memory.set_manual("manual-rule", "project_convention", "必须走 review 流程")
    memory.remember("auto-note", "模型自己记的杂项", kind="project_convention",
                    source="agent:turn-1")

    plan = {
        "merges": [{
            "from": "manual-rule", "into": "auto-note",
            "value": "合并后吞掉手工条目", "kind": "project_convention",
        }],
        "disables": [{"key": "manual-rule", "reason": "试图停用手工条目"}],
    }
    result = MemoryOrganizer(_settings(), llm=_PlanLLM(plan)).organize(memory)

    assert result["skipped"] >= 2
    manual = memory.get("manual-rule")
    assert manual.invalidated_at is None
    assert manual.value == "必须走 review 流程"
    assert memory.get("auto-note").value == "模型自己记的杂项"
    memory.close()


def test_organizer_failure_and_sparse_memory_change_nothing(tmp_path):
    from nextcoder.sessions.organizer import MemoryOrganizer

    memory = _memory(tmp_path)
    # 只有一条记忆：直接跳过，不发模型调用。
    memory.remember("only-one", "pytest -q", kind="test_command",
                    source="agent:turn-1")
    llm = _PlanLLM(RuntimeError("endpoint down"))
    assert MemoryOrganizer(_settings(), llm=llm).organize(memory)["status"] == "skipped"
    assert not llm.prompts

    memory.remember("second", "ruff check .", kind="lint_command",
                    source="agent:turn-2")
    # 模型调用抛错：整理失败，记忆原样。
    try:
        MemoryOrganizer(_settings(), llm=llm).organize(memory)
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    assert [item.key for item in memory.select()] == ["only-one", "second"]
    memory.close()


def test_organizer_trigger_schedule_and_background_spawn(tmp_path):
    from nextcoder.sessions import organizer as organizer_module

    assert organizer_module.should_organize(1, 10)
    for count in (2, 5, 9, 11, 19):
        assert not organizer_module.should_organize(count, 10)
    assert organizer_module.should_organize(10, 10)
    assert organizer_module.should_organize(20, 10)
    assert not organizer_module.should_organize(0, 10)

    # 后台触发：第 1、11 轮各整理一次；线程池单飞，重入跳过。
    settings = _settings(memory_organizer_turn_interval=10)
    task = tmp_path / "repo"
    task.mkdir()
    calls = []
    original = organizer_module.organize_project_memory

    def _stub(st, td, llm=None):
        time.sleep(0.3)  # 让线程在整理期间持有锁，验证重入跳过
        calls.append(str(td))
        return {"status": "applied", "merged": 0, "disabled": 0, "skipped": 0}

    organizer_module.organize_project_memory = _stub
    try:
        assert organizer_module.spawn_organizer(settings, task)
        assert not organizer_module.spawn_organizer(settings, task), "重入必须跳过"
        status_file = task / ".nextcoder" / "organizer-last.json"
        for _ in range(300):
            if status_file.exists():
                break
            time.sleep(0.02)
        assert calls == [str(task)]
        assert status_file.exists()
        assert json.loads(status_file.read_text(encoding="utf-8"))["status"] == "applied"
    finally:
        organizer_module.organize_project_memory = original
        for _ in range(300):
            if not organizer_module._ORGANIZER_LOCK.locked():
                break
            time.sleep(0.02)


def test_organizer_extracts_from_history_with_limits(tmp_path):
    from nextcoder.sessions.organizer import MemoryOrganizer

    memory = _memory(tmp_path)
    memory.remember("test-cmd", "pytest -q", kind="test_command",
                    source="agent:turn-1")
    memory.remember("dup-topic", "已有条目", kind="project_convention",
                    source="agent:turn-2")
    memory.remember("dead-topic", "已停用条目", kind="project_convention",
                    source="agent:turn-3")
    memory.disable("dead-topic")

    plan = {
        "extracts": [
            {"topic": "构建命令", "kind": "environment",
             "body": "本仓库构建用 uv run build",
             "evidence": "模型收尾：已用 uv run build 完成构建并验证"},
            {"topic": "test-cmd", "kind": "test_command",
             "body": "与现有条目同主题，必须跳过", "evidence": "重复"},
            {"topic": "dead-topic", "kind": "project_convention",
             "body": "停用条目不能借整理复活", "evidence": "复活"},
            {"topic": "", "kind": "environment", "body": "空主题", "evidence": ""},
        ],
    }
    llm = _PlanLLM(plan)
    history = "## s1/t1\n请求：构建\n结果：stop=model_finished"
    result = MemoryOrganizer(_settings(), llm=llm, history=history).organize(memory)

    # 第 4 条（空主题）在进入 apply 前就被 MAX_EXTRACTS=3 截断，不计 skipped。
    assert result["extracted"] == 1 and result["skipped"] == 2
    created = memory.get("构建命令")
    assert created is not None and created.value == "本仓库构建用 uv run build"
    assert created.source.startswith("organizer:")
    assert created.meta.get("evidence")
    assert created.meta.get("extracted_by") == "organizer"
    # 同 key 不覆盖、停用条目不复活。
    assert memory.get("test-cmd").value == "pytest -q"
    assert memory.get("dead-topic").invalidated_at is not None
    # 提示词带旧会话段落与提取规则。
    assert "旧会话轮次" in llm.prompts[0]
    assert "extracts" in llm.prompts[0]
    assert "请求：构建" in llm.prompts[0]
    memory.close()


def test_organizer_without_history_forbids_extracts_in_prompt(tmp_path):
    from nextcoder.sessions.organizer import MemoryOrganizer

    memory = _memory(tmp_path)
    memory.remember("a", "第一条", kind="test_command", source="agent:1")
    memory.remember("b", "第二条", kind="lint_command", source="agent:2")

    llm = _PlanLLM({"merges": [], "disables": [], "extracts": []})
    MemoryOrganizer(_settings(), llm=llm).organize(memory)
    assert "（无，extracts 必须为空）" in llm.prompts[0]
    memory.close()


def _write_session(task_path, session_id: str, turns: int) -> None:
    """造一条只有完整轮次的会话 rollout。"""
    from nextcoder.sessions.rollout import RolloutWriter

    session_dir = task_path / ".nextcoder" / "sessions" / session_id
    session_dir.mkdir(parents=True)
    roll = RolloutWriter(session_dir / "rollout.jsonl", session_id=session_id)
    roll.append("session_start", {"task_dir": str(task_path)})
    for index in range(1, turns + 1):
        roll.append("user_request", {"request": f"请求 {session_id} 第 {index} 轮"},
                    turn_id=f"t{index}")
        roll.append("assistant", {"content": f"收尾 {session_id} 第 {index} 轮"},
                    turn_id=f"t{index}")
        roll.append("turn_end", {"stop_reason": "model_finished", "model_calls": 5,
                                 "has_changes": True}, turn_id=f"t{index}")
    roll.close()


def test_extraction_takes_oldest_unprocessed_sessions(tmp_path):
    from nextcoder.sessions.organizer import collect_pending_history

    task = tmp_path / "repo"
    task.mkdir()
    memory = ProjectMemory(_settings(), task)
    # 旧会话：a 两轮、b 一轮、c 没有完整轮次、d 一轮、e 一轮。
    _write_session(task, "20260101-000000-a", turns=2)
    _write_session(task, "20260102-000000-b", turns=1)
    _write_session(task, "20260103-000000-c", turns=0)
    _write_session(task, "20260104-000000-d", turns=1)
    _write_session(task, "20260105-000000-e", turns=1)

    text, pending = collect_pending_history(memory, _settings(), task)
    assert [item[0] for item in pending] == [
        "20260101-000000-a", "20260102-000000-b", "20260104-000000-d"]
    assert pending[0] == ("20260101-000000-a", 2, 2)
    assert "20260101-000000-a" in text and "20260105-000000-e" not in text
    assert text.index("20260101-000000-a") < text.index("20260102-000000-b")

    # 记过进度的会话不再出现；同一目录下次接着处理 e。
    for session_id, processed, total in pending:
        memory.mark_session_processed(session_id, processed, total)
    text, pending = collect_pending_history(memory, _settings(), task)
    assert [item[0] for item in pending] == ["20260105-000000-e"]
    memory.close()


def test_extraction_history_includes_tool_outputs(tmp_path):
    from nextcoder.sessions.organizer import collect_pending_history
    from nextcoder.sessions.store import OperationStore

    task = tmp_path / "repo"
    task.mkdir()
    session_id = "20260101-000000-a"
    _write_session(task, session_id, turns=1)
    session_dir = task / ".nextcoder" / "sessions" / session_id
    log_dir = session_dir / "turns" / "t1" / "logs"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "cmd.log"
    log_path.write_text("collected 3 items\n\n2 failed, 1 passed\n", encoding="utf-8")

    store = OperationStore(session_dir / "operations.sqlite")
    call_id = store.prepare("run_command",
                            {"command": "uv run pytest tests/test_x.py"},
                            turn_id="t1")
    store.transition(call_id, "started")
    store.transition(call_id, "completed", exit_code=2,
                     log_path=str(log_path), result_text="2 failed, 1 passed")
    store.close()

    memory = ProjectMemory(_settings(), task)
    text, _ = collect_pending_history(memory, _settings(), task)
    assert "uv run pytest tests/test_x.py" in text
    assert "exit=2" in text
    assert "2 failed, 1 passed" in text
    memory.close()


def test_extraction_progress_only_moves_forward(tmp_path):
    task = tmp_path / "repo"
    task.mkdir()
    memory = ProjectMemory(_settings(), task)
    memory.mark_session_processed("s1", 3, 5)
    memory.mark_session_processed("s1", 1, 5)  # 回退写入被忽略
    assert memory.session_processed_turns("s1") == 3
    memory.close()


def test_organize_project_memory_marks_reviewed_sessions(tmp_path):
    from nextcoder.sessions import organizer as organizer_module

    task = tmp_path / "repo"
    task.mkdir()
    _write_session(task, "20260101-000000-a", turns=1)
    _write_session(task, "20260102-000000-b", turns=1)

    plan = {"extracts": [{"topic": "构建命令", "kind": "environment",
                          "body": "用 uv build", "evidence": "收尾里提到"}]}
    result = organizer_module.organize_project_memory(
        _settings(), task, llm=_PlanLLM(plan))
    assert result["status"] == "applied"
    assert result["sessions_reviewed"] == ["20260101-000000-a", "20260102-000000-b"]

    memory = ProjectMemory(_settings(), task)
    assert memory.session_processed_turns("20260101-000000-a") == 1
    assert memory.get("构建命令") is not None
    # 已审阅的会话第二次不再送入模型。
    llm = _PlanLLM({"extracts": []})
    organizer_module.organize_project_memory(_settings(), task, llm=llm)
    assert llm.prompts == []  # 没有旧会话可看，记忆只有一条 → 直接跳过
    memory.close()


def test_organizer_failure_does_not_mark_sessions(tmp_path):
    from nextcoder.sessions import organizer as organizer_module

    task = tmp_path / "repo"
    task.mkdir()
    _write_session(task, "20260101-000000-a", turns=1)
    llm = _PlanLLM(RuntimeError("endpoint down"))
    try:
        organizer_module.organize_project_memory(_settings(), task, llm=llm)
    except RuntimeError:
        pass
    memory = ProjectMemory(_settings(), task)
    assert memory.session_processed_turns("20260101-000000-a") == 0
    memory.close()


def test_model_node_emits_rollout_event(tmp_path):
    from nextcoder.sessions.rollout import RolloutWriter, read_events

    settings = _settings()
    memory = _memory(tmp_path)
    context = _context(tmp_path, memory)
    rollout = RolloutWriter(context.run_dir / "rollout.jsonl", session_id="sess-1")
    context.rollout = rollout
    context.turn_id = "turn-9"
    runner = AgentRunner(settings, context)
    runner._llm = _FakeLLM()

    state = {
        "messages": [SystemMessage(content="系统提示"), HumanMessage(content="继续")],
        "model_calls": 0, "usage": {}, "request_tokens": 0, "stop_reason": None,
    }
    runner._model_node(state)

    events = read_events(rollout.path)
    assert [event["type"] for event in events] == ["assistant"]
    assert events[0]["session_id"] == "sess-1"
    assert events[0]["turn_id"] == "turn-9"
    assert events[0]["payload"]["content"] == "完成"
    memory.close()
    context.store.close()
