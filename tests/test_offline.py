"""离线单元测试：操作状态机、写入校验、diff 与命令超时。不依赖模型 API。"""

import io
import subprocess
import time
from pathlib import Path

import pytest

from nextcoder.runtime.patch import copy_tree_filtered, git_diff_patch
from nextcoder.sessions.store import OperationStore
from nextcoder.settings import Settings
from nextcoder.tools.base import ToolContext, ToolResult, hash_content
from nextcoder.tools.commands import run_command
from nextcoder.tools.edits import apply_patch, view_diff, write_file
from nextcoder.tools.files import list_files, read_lines, search


@pytest.fixture
def context(tmp_path):
    task_dir = tmp_path / "task"
    run_dir = task_dir / ".nextcoder" / "runs" / "test"
    task_dir.mkdir()
    run_dir.mkdir(parents=True)
    store = OperationStore(run_dir / "operations.sqlite")
    context = ToolContext(
        task_dir=task_dir, run_dir=run_dir, baseline_dir=tmp_path / "baseline",
        store=store, settings=Settings(llm_base_url="", llm_model=""),
    )
    context.baseline_dir.mkdir()
    yield context
    store.close()


class TestOperationStore:
    def test_lifecycle(self, context):
        call_id = context.store.prepare("write_file", {"path": "a.py"})
        op = context.store.get(call_id)
        assert op.status == "prepared" and op.tool == "write_file"
        context.store.transition(call_id, "started")
        context.store.transition(call_id, "completed", related_files=["a.py"], exit_code=0,
                                 result_text="done")
        op = context.store.get(call_id)
        assert op.status == "completed"
        assert op.related_files == ["a.py"]
        assert op.exit_code == 0
        assert op.result_text == "done"

    def test_stable_call_id(self, context):
        first = context.store.prepare("read_lines", {"path": "a.py"}, call_id="call-abc")
        second = context.store.prepare("read_lines", {"path": "a.py"}, call_id="call-abc")
        assert first == second == "call-abc"

    def test_unknown_status_rejected(self, context):
        call_id = context.store.prepare("run_command", {})
        with pytest.raises(ValueError):
            context.store.transition(call_id, "finished")

    def test_terminal_status_cannot_restart(self, context):
        call_id = context.store.prepare("run_command", {})
        context.store.transition(call_id, "started")
        context.store.transition(call_id, "completed")
        with pytest.raises(ValueError):
            context.store.transition(call_id, "started")

    def test_missing_operation(self, context):
        with pytest.raises(KeyError):
            context.store.transition("no-such-id", "started")

    def test_run_record(self, context):
        context.store.start_run("r1", "/tmp", "请求")
        context.store.finish_run("r1", "stopped", "budget", 3, {"input_tokens": 100})
        rows = context.store.all_operations()
        assert rows == []


class TestWriteGuards:
    def test_existing_file_requires_hash(self, context):
        target = context.task_dir / "a.txt"
        target.write_text("原始内容", encoding="utf-8")
        result = write_file(context, "c1", "a.txt", "新内容")
        assert result.status == "failed"
        assert "content_hash" in result.error

    def test_stale_hash_rejected(self, context):
        target = context.task_dir / "a.txt"
        target.write_text("当前内容", encoding="utf-8")
        result = write_file(context, "c1", "a.txt", "新内容",
                            expected_hash=hash_content("旧内容"))
        assert result.status == "failed"
        assert "外部修改" in result.error

    def test_read_returns_hash_and_write_succeeds(self, context):
        target = context.task_dir / "a.txt"
        target.write_text("第一版", encoding="utf-8")
        read = read_lines(context, "c1", "a.txt")
        assert read.status == "completed"
        assert read.content_hash == hash_content("第一版")
        assert f"content_hash={read.content_hash}" in read.as_text(context.settings)
        result = write_file(context, "c2", "a.txt", "第二版",
                            expected_hash=read.content_hash)
        assert result.status == "completed"
        assert result.content_hash == hash_content("第二版")
        assert target.read_text(encoding="utf-8") == "第二版"

    def test_new_file_without_hash(self, context):
        result = write_file(context, "c1", "new.txt", "内容")
        assert result.status == "completed"

    def test_path_escape_rejected(self, context):
        result = write_file(context, "c1", "../outside.txt", "内容")
        assert result.status == "failed"


class TestApplyPatch:
    def test_patch_after_read(self, context):
        target = context.task_dir / "a.py"
        target.write_text("x = 1\ny = 2\n", encoding="utf-8")
        read = read_lines(context, "c1", "a.py")
        result = apply_patch(context, "c2", "a.py", "x = 1", "x = 10",
                             expected_hash=read.content_hash)
        assert result.status == "completed", result.error
        assert target.read_text(encoding="utf-8") == "x = 10\ny = 2\n"
        assert result.content_hash == hash_content(target.read_text(encoding="utf-8"))

    def test_patch_requires_hash(self, context):
        (context.task_dir / "a.py").write_text("x = 1\n", encoding="utf-8")
        result = apply_patch(context, "c1", "a.py", "x = 1", "x = 2")
        assert result.status == "failed"
        assert "content_hash" in result.error

    def test_patch_stale_hash_rejected(self, context):
        target = context.task_dir / "a.py"
        target.write_text("x = 1\n", encoding="utf-8")
        result = apply_patch(context, "c1", "a.py", "x = 1", "x = 2",
                             expected_hash=hash_content("旧内容"))
        assert result.status == "failed"
        assert "外部修改" in result.error
        assert "字符" in result.error

    def test_patch_noop_is_explicit(self, context):
        """old_text == new_text 时明确说明文件不会变化，避免反复空补丁。"""
        (context.task_dir / "a.py").write_text("x = 1\n", encoding="utf-8")
        read = read_lines(context, "c1", "a.py")
        result = apply_patch(context, "c2", "a.py", "x = 1", "x = 1",
                             expected_hash=read.content_hash)
        assert result.status == "failed"
        assert "文件不会被修改" in result.error
        assert "运行验证" in result.error

    def test_patch_missing_text(self, context):
        (context.task_dir / "a.py").write_text("x = 1\n", encoding="utf-8")
        read = read_lines(context, "c1", "a.py")
        result = apply_patch(context, "c2", "a.py", "z = 9", "z = 10",
                             expected_hash=read.content_hash)
        assert result.status == "failed"
        assert "没有匹配" in result.error

    def test_patch_missing_text_hints_whitespace_mismatch(self, context):
        """未匹配时给出缩进/空白线索，而不是让模型盲目重试。"""
        (context.task_dir / "a.py").write_text("def f():\n    return 1\n",
                                               encoding="utf-8")
        read = read_lines(context, "c1", "a.py")
        result = apply_patch(context, "c2", "a.py", "\treturn 1\n", "\treturn 2\n",
                             expected_hash=read.content_hash)
        assert result.status == "failed"
        assert "去掉 old_text 首尾空白后可以唯一匹配" in result.error

    def test_patch_missing_text_hints_nearest_line(self, context):
        (context.task_dir / "a.py").write_text("def f():\n    return 1\n",
                                               encoding="utf-8")
        read = read_lines(context, "c1", "a.py")
        result = apply_patch(context, "c2", "a.py", "def f():\n    return 2",
                             "def f():\n    return 3",
                             expected_hash=read.content_hash)
        assert result.status == "failed"
        assert "最接近第 1 行" in result.error

    def test_patch_ambiguous_requires_replace_all(self, context):
        target = context.task_dir / "a.py"
        target.write_text("x = 1\nx = 1\n", encoding="utf-8")
        read = read_lines(context, "c1", "a.py")
        ambiguous = apply_patch(context, "c2", "a.py", "x = 1", "x = 2",
                                expected_hash=read.content_hash)
        assert ambiguous.status == "failed"
        assert "2 处" in ambiguous.error
        replaced = apply_patch(context, "c3", "a.py", "x = 1", "x = 2",
                               expected_hash=read.content_hash, replace_all=True)
        assert replaced.status == "completed"
        assert target.read_text(encoding="utf-8") == "x = 2\nx = 2\n"


class TestDiff:
    def test_diff_shows_changes(self, context):
        (context.task_dir / "a.txt").write_text("新内容\n", encoding="utf-8")
        (context.baseline_dir / "a.txt").write_text("旧内容\n", encoding="utf-8")
        result = view_diff(context, "c1")
        assert result.status == "completed"
        assert result.changed_files == ["a.txt"]

    def test_no_changes(self, context):
        (context.task_dir / "a.txt").write_text("同样\n", encoding="utf-8")
        (context.baseline_dir / "a.txt").write_text("同样\n", encoding="utf-8")
        result = view_diff(context, "c1")
        assert result.changed_files == []

    def test_baseline_under_nextcoder_dir_is_not_ignored(self, tmp_path):
        """基线位于 .nextcoder/ 下时，不能把基线文件本身当成忽略项。"""
        task_dir = tmp_path / "task"
        session_root = task_dir / ".nextcoder" / "sessions" / "s"
        run_dir = session_root / "turns" / "t"
        baseline = session_root / "baseline"
        run_dir.mkdir(parents=True)
        baseline.mkdir(parents=True)
        (task_dir / "same.txt").write_text("同样\n", encoding="utf-8")
        (baseline / "same.txt").write_text("同样\n", encoding="utf-8")
        (task_dir / "new.txt").write_text("新增\n", encoding="utf-8")
        store = OperationStore(run_dir / "operations.sqlite")
        context = ToolContext(
            task_dir=task_dir, run_dir=run_dir, baseline_dir=baseline,
            store=store, settings=Settings(llm_base_url="", llm_model=""),
        )
        try:
            result = view_diff(context, "c1")
        finally:
            store.close()
        assert result.changed_files == ["new.txt"]


class TestPatchExport:
    def test_patch_applies_to_clean_copy(self, tmp_path):
        baseline = tmp_path / "baseline"
        current = tmp_path / "current"
        baseline.mkdir()
        (baseline / "mod.py").write_text("x = 1\n", encoding="utf-8")
        (baseline / "gone.txt").write_text("old\n", encoding="utf-8")
        copy_tree_filtered(baseline, current)
        (current / "mod.py").write_text("x = 2\n", encoding="utf-8")
        (current / "gone.txt").unlink()
        (current / "新增.txt").write_text("中文\n", encoding="utf-8")

        patch = git_diff_patch(current, baseline)
        assert patch.startswith("diff --git a/")
        assert "a/mod.py" in patch and "b/新增.txt" in patch
        assert "a/current/" not in patch and "b/baseline/" not in patch
        assert "diff --git a/新增.txt b/新增.txt" in patch

        verify = tmp_path / "verify"
        copy_tree_filtered(baseline, verify)
        applied = subprocess.run(["git", "apply", "--check", "-"], input=patch,
                                 cwd=verify, capture_output=True, text=True)
        assert applied.returncode == 0, applied.stderr

    def test_ignored_directories_are_pruned_before_walking(self, tmp_path, monkeypatch):
        """基线复制不能进入 .git/.venv；否则大仓库创建会话会长时间卡住。"""
        from nextcoder.runtime import patch as patch_module

        baseline = tmp_path / "baseline"
        baseline.mkdir()
        for ignored in (".git", ".venv", "node_modules"):
            (baseline / ignored / "deep").mkdir(parents=True)
            (baseline / ignored / "deep" / "noise.bin").write_bytes(b"x")
        (baseline / "src").mkdir()
        (baseline / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")

        visited: list[str] = []
        real_walk = patch_module.os.walk

        def tracking_walk(top, *args, **kwargs):
            for root, directories, files in real_walk(top, *args, **kwargs):
                visited.append(Path(root).relative_to(baseline).as_posix())
                yield root, directories, files

        monkeypatch.setattr(patch_module.os, "walk", tracking_walk)
        copy_tree_filtered(baseline, tmp_path / "copy")

        assert "src" in visited
        assert not any(name.startswith((".git", ".venv", "node_modules"))
                       for name in visited)
        assert (tmp_path / "copy" / "src" / "app.py").is_file()
        assert not (tmp_path / "copy" / ".git").exists()

    def test_symlink_not_expanded_into_duplicate_file(self, tmp_path):
        baseline = tmp_path / "baseline"
        current = tmp_path / "current"
        baseline.mkdir()
        (baseline / "README.md").write_text("old\n", encoding="utf-8")
        (baseline / "docs").mkdir()
        (baseline / "docs" / "index.md").symlink_to("../README.md")
        copy_tree_filtered(baseline, current)
        assert (current / "docs" / "index.md").is_symlink()

        (current / "README.md").write_text("new\n", encoding="utf-8")
        patch = git_diff_patch(current, baseline)
        assert "README.md" in patch
        assert "docs/index.md" not in patch

        verify = tmp_path / "verify"
        copy_tree_filtered(baseline, verify)
        applied = subprocess.run(["git", "apply", "--check", "-"], input=patch,
                                 cwd=verify, capture_output=True, text=True)
        assert applied.returncode == 0, applied.stderr

    def test_no_changes_is_empty(self, tmp_path):
        baseline = tmp_path / "baseline"
        current = tmp_path / "current"
        baseline.mkdir()
        (baseline / "a.py").write_text("x = 1\n", encoding="utf-8")
        copy_tree_filtered(baseline, current)
        assert git_diff_patch(current, baseline) == ""

    def test_tool_caches_are_excluded_from_patch(self, tmp_path):
        """跑测试留下的缓存不能进入交付补丁：sqlfluff 曾带入 137 个 .hypothesis 文件。"""
        baseline = tmp_path / "baseline"
        current = tmp_path / "current"
        baseline.mkdir()
        (baseline / "mod.py").write_text("x = 1\n", encoding="utf-8")
        copy_tree_filtered(baseline, current)
        (current / "mod.py").write_text("x = 2\n", encoding="utf-8")
        for cache in (".hypothesis", ".mypy_cache", ".pytest_cache"):
            (current / cache).mkdir()
            (current / cache / "state").write_text("noise\n", encoding="utf-8")

        patch = git_diff_patch(current, baseline)
        assert "a/mod.py" in patch
        assert ".hypothesis" not in patch
        assert ".mypy_cache" not in patch
        assert ".pytest_cache" not in patch


    def test_non_utf8_content_does_not_break_patch_export(self, tmp_path):
        """matplotlib 测试产物含非 UTF-8 字节，补丁导出不能因此中断。"""
        baseline = tmp_path / "baseline"
        current = tmp_path / "current"
        baseline.mkdir()
        (baseline / "data.txt").write_bytes(b"caf\xe9\n")
        copy_tree_filtered(baseline, current)
        (current / "data.txt").write_bytes(b"caf\xe9 updated\n")

        patch = git_diff_patch(current, baseline)
        assert patch
        assert "data.txt" in patch

        verify = tmp_path / "verify"
        copy_tree_filtered(baseline, verify)
        applied = subprocess.run(["git", "apply", "--check", "-"], input=patch,
                                 cwd=verify, capture_output=True, text=True)
        assert applied.returncode == 0, applied.stderr


class TestCommands:
    def test_exit_code_and_output(self, context):
        result = run_command(context, "c1", "echo hello")
        assert result.status == "completed"
        assert result.exit_code == 0
        assert "hello" in result.output_excerpt
        assert result.artifact_path

    def test_nonzero_exit_is_failed(self, context):
        result = run_command(context, "c1", "exit 3")
        assert result.status == "failed"
        assert result.exit_code == 3

    def test_empty_output_is_explicit(self, context):
        result = run_command(context, "c1", "true")
        assert result.status == "completed"
        assert result.output_excerpt == ""
        assert "输出为空" in (result.note or "")
        assert "输出为空" in result.as_text(context.settings)

    def test_empty_pipeline_flags_hidden_stderr(self, context):
        # 管道退出码来自 cat，前面的 exit 7 被 2>/dev/null 吞掉；工具必须提示。
        result = run_command(
            context, "c1", "sh -c 'exit 7' 2>/dev/null | cat")
        assert result.status == "completed"
        assert result.exit_code == 0
        assert "隐藏了 stderr" in (result.note or "")
        assert "pipefail" in (result.note or "")

    def test_timeout_kills_group(self, context):
        started = time.monotonic()
        result = run_command(context, "c1", "sleep 30", timeout=1)
        assert time.monotonic() - started < 10
        assert result.status == "timeout"
        assert result.error and "timeout" in result.error

    def test_child_killed_too(self, context):
        result = run_command(context, "c1", "bash -c 'sleep 60 & wait'", timeout=1)
        assert result.status == "timeout"

    def test_cancel_stops_process(self, context):
        context.cancel_event.set()
        started = time.monotonic()
        result = run_command(context, "c1", "sleep 30", timeout=30)
        assert time.monotonic() - started < 5
        assert result.status == "cancelled"
        assert "取消" in result.output_excerpt

    def test_agent_secrets_not_inherited(self, context, monkeypatch):
        monkeypatch.setenv("NEXTCODER_LLM_API_KEY", "secret-value-xyz")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-value-abc")
        result = run_command(context, "c1", "env")
        assert "secret-value-xyz" not in result.output_excerpt
        assert "secret-value-abc" not in result.output_excerpt
        log = context.logs_dir / "c1-command.log"
        assert "secret-value-xyz" not in log.read_text(encoding="utf-8")

    def test_task_command_path_prepended(self, context, tmp_path):
        bin_dir = tmp_path / "task-bin"
        bin_dir.mkdir()
        tool = bin_dir / "task-python"
        tool.write_text("#!/bin/sh\necho task-env-ok\n", encoding="utf-8")
        tool.chmod(0o755)
        context.settings.command_path_prepend = str(bin_dir)
        result = run_command(context, "c1", "task-python")
        assert result.exit_code == 0
        assert "task-env-ok" in result.output_excerpt

    def test_docker_exec_wraps_command(self, context, monkeypatch):
        captured = {}

        class FakeProcess:
            returncode = 0

            def __init__(self, argv, **kwargs):
                captured["argv"] = argv
                captured["kwargs"] = kwargs
                self.stdout = io.StringIO("")

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

        monkeypatch.setattr("nextcoder.tools.commands.subprocess.Popen", FakeProcess)
        context.settings.command_container = "swe-eval-test"
        context.settings.command_container_workdir = "/testbed"
        context.settings.command_container_activate = (
            "source /opt/miniconda3/bin/activate && conda activate testbed")
        context.settings.command_path_prepend = "/host/env/bin"
        result = run_command(context, "c1", "python -c 'print(1)'", timeout=7)
        assert result.status == "completed"
        argv = captured["argv"]
        assert argv[:4] == ["docker", "exec", "swe-eval-test", "bash"]
        inner = argv[-1]
        assert "conda activate testbed" in inner
        assert "cd /testbed" in inner
        assert "timeout --signal=KILL 7s" in inner
        assert "python -c" in inner and "print(1)" in inner
        # 宿主机 PATH 前缀不能带进容器。
        assert "/host/env/bin" not in captured["kwargs"]["env"]["PATH"]

    def test_docker_exec_timeout_exit_is_timeout(self, context, monkeypatch):
        class FakeProcess:
            returncode = 124

            def __init__(self, argv, **kwargs):
                self.stdout = io.StringIO("")

            def poll(self):
                return 124

            def wait(self, timeout=None):
                return 124

        monkeypatch.setattr("nextcoder.tools.commands.subprocess.Popen", FakeProcess)
        context.settings.command_container = "swe-eval-test"
        result = run_command(context, "c1", "sleep 999", timeout=5)
        assert result.status == "timeout"
        assert result.error and "timeout" in result.error
        assert "超时" in result.output_excerpt

    def test_docker_exec_failure_is_reported(self, context, monkeypatch):
        class FakeProcess:
            returncode = 125

            def __init__(self, argv, **kwargs):
                self.stdout = io.StringIO(
                    "Error response from daemon: No such container\n")

            def poll(self):
                return 125

            def wait(self, timeout=None):
                return 125

        monkeypatch.setattr("nextcoder.tools.commands.subprocess.Popen", FakeProcess)
        context.settings.command_container = "missing-container"
        result = run_command(context, "c1", "true")
        assert result.status == "failed"
        assert result.error and "docker exec" in result.error
        assert "No such container" in result.output_excerpt

    def test_output_storage_cap(self, context):
        context.settings.max_output_bytes = 1000
        result = run_command(context, "c1", "python -c \"print('x' * 5000)\"")
        assert result.status == "completed"
        assert "截断" in result.output_excerpt
        assert "已截断" in (result.note or "")


class TestFiles:
    def test_list_files(self, context):
        (context.task_dir / "sub").mkdir()
        (context.task_dir / "sub" / "b.py").write_text("x=1", encoding="utf-8")
        (context.task_dir / "a.py").write_text("y=2", encoding="utf-8")
        result = list_files(context, "c1")
        assert "a.py" in result.output_excerpt and "sub/b.py" in result.output_excerpt

    def test_read_lines_paged_and_note_survives(self, context):
        (context.task_dir / "big.txt").write_text(
            "\n".join(f"line{i}" for i in range(1, 501)), encoding="utf-8")
        result = read_lines(context, "c1", "big.txt")
        assert "1: line1" in result.output_excerpt
        assert "start=201" in result.note
        assert result.content_hash

    def test_search_is_literal(self, context):
        (context.task_dir / "a.txt").write_text("a.b\naxb\n", encoding="utf-8")
        result = search(context, "c1", "a.b")
        assert "a.txt:1:a.b" in result.output_excerpt
        assert "axb" not in result.output_excerpt

    def test_search_pagination_note(self, context):
        (context.task_dir / "many.txt").write_text(
            "\n".join(f"hit {i}" for i in range(150)), encoding="utf-8")
        result = search(context, "c1", "hit")
        assert "offset=100" in (result.note or "")

    def test_search_falls_back_when_rg_missing(self, context, monkeypatch):
        monkeypatch.setattr("nextcoder.tools.files.shutil.which", lambda name: None)
        (context.task_dir / "a.txt").write_text("a.b\naxb\n", encoding="utf-8")
        ignored = context.task_dir / "node_modules"
        ignored.mkdir()
        (ignored / "skip.txt").write_text("a.b\n", encoding="utf-8")
        result = search(context, "c1", "a.b")
        assert result.status == "completed"
        assert "a.txt:1:a.b" in result.output_excerpt
        assert "axb" not in result.output_excerpt
        assert "node_modules" not in result.output_excerpt


def test_file_read_tools_bypass_excerpt_limit():
    settings = Settings(llm_base_url="", llm_model="", output_excerpt_chars=50)
    body = "x" * 500
    for tool in ("read_lines", "read_log"):
        result = ToolResult("c1", tool, "completed", output_excerpt=body)
        text = result.as_text(settings)
        assert body in text, f"{tool} 的按行窗口不能被字符摘录截断"
        assert "正文截断" not in text

    # 命令等无界输出仍按字符摘录，并提示完整产物位置。
    result = ToolResult("c2", "run_command", "completed", output_excerpt=body,
                        artifact_path="/logs/run.log")
    text = result.as_text(settings)
    assert "正文截断" in text
    assert "/logs/run.log" in text


class TestSandboxArgv:
    def _settings(self, **kwargs) -> Settings:
        return Settings(llm_base_url="", llm_model="", **kwargs)

    def test_argv_shape_with_writable_roots(self, tmp_path, monkeypatch):
        from nextcoder.tools import commands as commands_module
        from nextcoder.tools.commands import _sandbox_argv

        # argv 形状是纯函数测试；不依赖本机真的装了 bwrap。
        monkeypatch.setattr(commands_module.shutil, "which",
                            lambda name: "/usr/bin/bwrap")
        extra = tmp_path / "cache"
        extra.mkdir()
        readonly = tmp_path / "data"
        readonly.mkdir()
        argv = _sandbox_argv(
            self._settings(
                sandbox_writable_roots=[str(extra)],
                sandbox_readonly_roots=[str(readonly)]),
            tmp_path, "pytest -q", 120)
        joined = " ".join(argv)
        assert argv[0] == "/usr/bin/bwrap"
        assert "--ro-bind-try /usr /usr" in joined
        assert f"--bind {tmp_path.resolve()} {tmp_path.resolve()}" in joined
        assert f"--bind-try {extra} {extra}" in joined
        assert f"--ro-bind-try {readonly} {readonly}" in joined
        assert "--tmpfs /tmp" in joined
        assert "--clearenv" in joined
        assert "--new-session" in joined
        assert "--die-with-parent" in joined
        assert "--unshare-pid" in joined and "--unshare-ipc" in joined
        assert "--unshare-net" in joined
        assert "--setenv HOME /tmp" in joined
        assert "timeout --signal=KILL 120s" in joined
        # 不再把真实 home 整个只读挂进沙箱。
        assert f"--ro-bind-try {Path.home()} {Path.home()}" not in joined

    def test_network_and_env_controls(self, tmp_path, monkeypatch):
        from nextcoder.tools import commands as commands_module
        from nextcoder.tools.commands import _sandbox_argv

        monkeypatch.setattr(commands_module.shutil, "which",
                            lambda name: "/usr/bin/bwrap")
        monkeypatch.setenv("SAFE_TOKEN", "abc")
        monkeypatch.setenv("OTHER_API_KEY", "should-not-leak")
        argv = _sandbox_argv(
            self._settings(
                command_sandbox_network=True,
                sandbox_env_passthrough=[
                    "SAFE_TOKEN", "OTHER_API_KEY", "NEXTCODER_LLM_API_KEY"]),
            tmp_path, "env", 30)
        joined = " ".join(argv)
        assert "--unshare-net" not in argv
        assert "--setenv SAFE_TOKEN abc" in joined
        assert "should-not-leak" not in joined
        assert "OTHER_API_KEY" not in joined
        assert "NEXTCODER_LLM_API_KEY" not in joined

    def test_bwrap_missing_returns_none(self, tmp_path, monkeypatch):
        from nextcoder.tools import commands as commands_module

        monkeypatch.setattr(commands_module.shutil, "which", lambda name: None)
        argv = commands_module._sandbox_argv(
            self._settings(), tmp_path, "echo hi", 30)
        assert argv is None

    def test_bwrap_inside_task_dir_is_rejected(self, tmp_path, monkeypatch):
        from nextcoder.tools import commands as commands_module

        fake = tmp_path / "bwrap"
        fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake.chmod(0o755)
        monkeypatch.setattr(commands_module.shutil, "which",
                            lambda name: str(fake))
        assert commands_module._sandbox_argv(
            self._settings(), tmp_path, "echo hi", 30) is None

    def test_sandbox_uses_absolute_bwrap_path(self, tmp_path, monkeypatch):
        """Popen 会用 command_path_prepend 重新查 PATH，不能把裸名字交给它。"""
        from nextcoder.tools import commands as commands_module

        fake_dir = tmp_path / "task-bin"
        fake_dir.mkdir()
        (fake_dir / "bwrap").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        (fake_dir / "bwrap").chmod(0o755)
        monkeypatch.setattr(commands_module.shutil, "which",
                            lambda name: "/usr/bin/bwrap")
        argv = commands_module._sandbox_argv(
            self._settings(command_path_prepend=str(fake_dir)),
            tmp_path, "echo hi", 30)
        assert argv is not None
        assert argv[0] == "/usr/bin/bwrap"

    def test_sandboxed_command_runs_and_sees_task_dir(self, tmp_path):
        """bwrap 存在时走真实沙箱；环境中没有 bwrap 时跳过（fail-open 路径已单测）。"""
        import shutil as _shutil

        if _shutil.which("bwrap") is None:
            pytest.skip("本机没有 bwrap，跳过真实沙箱执行")
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        run_dir = task_dir / ".nextcoder" / "runs" / "test"
        run_dir.mkdir(parents=True)
        store = OperationStore(run_dir / "operations.sqlite")
        context = ToolContext(
            task_dir=task_dir, run_dir=run_dir, baseline_dir=tmp_path / "baseline",
            store=store,
            settings=self._settings(command_sandbox=True))
        context.baseline_dir.mkdir()
        try:
            result = run_command(context, "c1", "pwd > where.txt && cat where.txt")
            text = result.as_text(context.settings)
            assert "status=completed" in text, text
            assert str(task_dir) in (context.task_dir / "where.txt").read_text(
                encoding="utf-8")
            assert "未沙箱化" not in text
        finally:
            store.close()

    def test_sandbox_requested_but_missing_fails_closed(self, tmp_path, monkeypatch):
        from nextcoder.tools import commands as commands_module

        monkeypatch.setattr(commands_module.shutil, "which", lambda name: None)
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        run_dir = task_dir / ".nextcoder" / "runs" / "test"
        run_dir.mkdir(parents=True)
        store = OperationStore(run_dir / "operations.sqlite")
        context = ToolContext(
            task_dir=task_dir, run_dir=run_dir, baseline_dir=tmp_path / "baseline",
            store=store,
            settings=self._settings(command_sandbox=True,
                                    permission_mode="bypass"))
        context.baseline_dir.mkdir()
        try:
            result = run_command(context, "c1", "touch should_not_exist")
            text = result.as_text(context.settings)
            assert result.status == "failed"
            assert "bwrap 不可用" in text
            assert "未执行" in text
            assert not (task_dir / "should_not_exist").exists()
        finally:
            store.close()

    def test_sandbox_missing_can_fail_open_when_explicitly_configured(
            self, tmp_path, monkeypatch):
        from nextcoder.tools import commands as commands_module

        monkeypatch.setattr(commands_module.shutil, "which", lambda name: None)
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        run_dir = task_dir / ".nextcoder" / "runs" / "test"
        run_dir.mkdir(parents=True)
        store = OperationStore(run_dir / "operations.sqlite")
        context = ToolContext(
            task_dir=task_dir, run_dir=run_dir, baseline_dir=tmp_path / "baseline",
            store=store,
            settings=self._settings(
                command_sandbox=True,
                command_sandbox_fail_closed=False,
                permission_mode="bypass"))
        context.baseline_dir.mkdir()
        try:
            result = run_command(context, "c1", "echo open")
            text = result.as_text(context.settings)
            assert result.status == "completed"
            assert "[沙箱] bwrap 不可用" in text
            assert "未沙箱化" in text
        finally:
            store.close()


class TestBwrapStatus:
    """TUI 状态栏/配置页依赖真实探测，不能把“装了 bwrap”误报成“可用”。"""

    def _stub_run(self, monkeypatch, returncode: int,
                  stderr: str = "") -> None:
        from nextcoder.tools import commands as commands_module

        class Completed:
            pass

        completed = Completed()
        completed.returncode = returncode
        completed.stderr = stderr
        completed.stdout = ""
        monkeypatch.setattr(
            commands_module.subprocess, "run", lambda *args, **kwargs: completed)

    def test_missing(self, monkeypatch):
        from nextcoder.tools import commands as commands_module

        monkeypatch.setattr(commands_module, "_BWRAP_STATUS_CACHE", None)
        monkeypatch.setattr(commands_module.shutil, "which", lambda name: None)
        assert commands_module.bwrap_status() == (
            "missing", "", "未找到 bwrap 可执行文件")

    def test_unusable_when_namespace_creation_is_denied(self, monkeypatch):
        from nextcoder.tools import commands as commands_module

        monkeypatch.setattr(commands_module, "_BWRAP_STATUS_CACHE", None)
        monkeypatch.setattr(commands_module.shutil, "which",
                            lambda name: "/usr/bin/bwrap")
        self._stub_run(
            monkeypatch, 1,
            "bwrap: No permissions to create a new namespace\n")
        state, path, detail = commands_module.bwrap_status()
        assert state == "unusable"
        assert path == "/usr/bin/bwrap"
        assert "No permissions" in detail

    def test_ok_and_cached(self, monkeypatch):
        from nextcoder.tools import commands as commands_module

        monkeypatch.setattr(commands_module, "_BWRAP_STATUS_CACHE", None)
        monkeypatch.setattr(commands_module.shutil, "which",
                            lambda name: "/usr/bin/bwrap")
        calls = {"n": 0}

        def run(*args, **kwargs):
            calls["n"] += 1
            class Completed:
                returncode = 0
                stdout = ""
                stderr = ""
            return Completed()

        monkeypatch.setattr(commands_module.subprocess, "run", run)
        assert commands_module.bwrap_status() == ("ok", "/usr/bin/bwrap", "")
        # 30 秒缓存：状态栏每轮刷新不应反复启动探测进程。
        assert commands_module.bwrap_status() == ("ok", "/usr/bin/bwrap", "")
        assert calls["n"] == 1
