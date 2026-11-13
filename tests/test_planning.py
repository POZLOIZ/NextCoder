"""任务清单测试：PlanStore 状态机与 plan 工具的 ToolBox 闭环。"""

import pytest

from nextcoder.runtime.planning import MAX_PLAN_TASKS, PlanStore
from nextcoder.sessions.store import OperationStore
from nextcoder.settings import Settings
from nextcoder.tools.base import ToolContext
from nextcoder.tools.registry import ToolBox


@pytest.fixture
def store(tmp_path):
    return PlanStore(tmp_path / ".nextcoder" / "task-plan.json")


@pytest.fixture
def context(tmp_path):
    task_dir = tmp_path / "task"
    run_dir = task_dir / ".nextcoder" / "runs" / "test"
    task_dir.mkdir()
    run_dir.mkdir(parents=True)
    baseline = run_dir / "baseline"
    baseline.mkdir()
    operation_store = OperationStore(run_dir / "operations.sqlite")
    context = ToolContext(
        task_dir=task_dir, run_dir=run_dir, baseline_dir=baseline,
        store=operation_store,
        settings=Settings(llm_base_url="http://localhost", llm_model="fake",
                          llm_api_key="test"),
        turn_id="turn-1")
    yield context
    operation_store.close()


class TestPlanStore:
    def test_replace_assigns_ids_and_revision(self, store):
        plan = store.replace(["复现失败", "修复根因", "补回归测试"])
        assert plan.revision == 1
        assert [task.id for task in plan.tasks] == ["t1", "t2", "t3"]
        assert all(task.status == "pending" for task in plan.tasks)

    def test_snapshot_text_marks_statuses(self, store):
        store.replace(["复现失败", "修复根因"])
        store.update({"t1": "completed", "t2": "in_progress"})
        text = store.snapshot_text()
        assert "[任务清单 v2]" in text
        assert "- [x] t1" in text
        assert "- [~] t2" in text

    def test_update_validates_status_and_ids(self, store):
        store.replace(["复现失败"])
        with pytest.raises(ValueError, match="未知状态"):
            store.update({"t1": "done"})
        with pytest.raises(ValueError, match="未找到任务项"):
            store.update({"t9": "completed"})
        assert store.load().revision == 1

    def test_replace_keeps_status_of_unchanged_items(self, store):
        store.replace(["复现失败", "修复根因"])
        store.update({"t1": "completed"})
        plan = store.replace(["复现失败", "修复根因", "补测试"])
        assert plan.revision == 3
        assert plan.tasks[0].status == "completed"
        assert plan.tasks[2].status == "pending"

    def test_replace_rejects_empty_and_oversize(self, store):
        with pytest.raises(ValueError, match="不能为空"):
            store.replace(["  "])
        with pytest.raises(ValueError, match="最多"):
            store.replace([f"任务{i}" for i in range(MAX_PLAN_TASKS + 1)])

    def test_corrupt_file_recovers_as_empty_plan(self, store):
        store.path.parent.mkdir(parents=True, exist_ok=True)
        store.path.write_text("{broken", encoding="utf-8")
        assert store.load().tasks == []
        plan = store.replace(["重新开始"])
        assert plan.revision == 1

    def test_has_open_items(self, store):
        store.replace(["a", "b"])
        assert store.has_open_items()
        store.update({"t1": "completed", "t2": "cancelled"})
        assert not store.has_open_items()
        assert store.snapshot_text()


class TestPlanTools:
    def test_toolbox_roundtrip_and_result_format(self, context):
        box = ToolBox(context)
        created = box.execute("plan_create", {"tasks": ["复现失败", "修复根因"]},
                              call_id="plan-1")
        updated = box.execute(
            "plan_update", {"statuses": {"t1": "completed"}}, call_id="plan-2")
        assert created.startswith("[plan_create] status=completed")
        assert "已创建任务清单 v1" in created
        assert "未完成 1 项（t2）" in updated

    def test_plan_disabled_returns_friendly_failure(self, tmp_path):
        task_dir = tmp_path / "task"
        run_dir = task_dir / ".nextcoder" / "runs" / "test"
        task_dir.mkdir()
        run_dir.mkdir(parents=True)
        baseline = run_dir / "baseline"
        baseline.mkdir()
        operation_store = OperationStore(run_dir / "operations.sqlite")
        context = ToolContext(
            task_dir=task_dir, run_dir=run_dir, baseline_dir=baseline,
            store=operation_store,
            settings=Settings(llm_base_url="http://localhost", llm_model="fake",
                              llm_api_key="test", plan_enabled=False),
            turn_id="turn-1")
        box = ToolBox(context)
        text = box.execute("plan_create", {"tasks": ["x"]}, call_id="plan-off")
        operation_store.close()
        assert "任务清单未启用" in text

    def test_plan_tools_absent_from_definitions_when_disabled(self, context):
        context.settings.plan_enabled = False
        # plan 是 cached_property，需要绕过缓存直接验证定义逻辑。
        context.__dict__.pop("plan", None)
        names = [tool.name for tool in ToolBox(context).definitions()]
        assert "plan_create" not in names and "plan_update" not in names
        assert "list_files" in names

    def test_plan_tools_present_by_default(self, context):
        names = [tool.name for tool in ToolBox(context).definitions()]
        assert "plan_create" in names and "plan_update" in names
