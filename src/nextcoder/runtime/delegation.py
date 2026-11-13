"""子 agent：角色白名单、worktree 隔离、后台执行与检查点恢复。

父 agent 只通过 delegate 工具派发独立任务。子 agent 拥有全新的消息历史
和预算，但复用父会话已经冻结的稳定前缀（base 规则 + AGENTS.md + 记忆
快照 + 技能目录），因此不会改变父请求的前缀缓存。

只读角色（researcher/reviewer）直接在任务目录工作；tester/coder 在
git worktree 中工作，结束后导出 changes.patch 并清理 worktree。子 agent
禁止继续委派，工具白名单同时约束 definitions 与 execute。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nextcoder.runtime.agent import AgentRunner
from nextcoder.runtime.events import ChildUpdated, emit
from nextcoder.runtime.patch import copy_baseline
from nextcoder.sessions.memory import MemorySnapshot, ProjectMemory
from nextcoder.sessions.rollout import RolloutWriter
from nextcoder.sessions.store import OperationStore
from nextcoder.settings import Settings
from nextcoder.tools.base import ToolContext

READ_TOOLS = {
    "list_files", "search", "read_lines", "read_log", "view_diff",
    "memory_list", "memory_read", "load_skill",
}
TEST_TOOLS = READ_TOOLS | {"run_command"}
CODER_TOOLS = TEST_TOOLS | {"write_file", "apply_patch", "memory_write",
                            "memory_forget"}


@dataclass(frozen=True)
class RoleSpec:
    name: str
    tools: frozenset[str]
    worktree: bool
    allow_mcp: bool
    instruction: str


ROLES: dict[str, RoleSpec] = {
    "researcher": RoleSpec(
        "researcher", frozenset(READ_TOOLS), False, False,
        "负责定位相关文件、阅读实现与收集证据；不能修改文件或执行命令。"),
    "reviewer": RoleSpec(
        "reviewer", frozenset(READ_TOOLS), False, False,
        "负责独立审查方案或补丁的正确性、风险与遗漏；不能修改文件。"),
    "tester": RoleSpec(
        "tester", frozenset(TEST_TOOLS), True, False,
        "负责在隔离 worktree 中运行测试、复现失败并报告命令与输出；可以执行命令，"
        "但不能直接编辑文件。"),
    "coder": RoleSpec(
        "coder", frozenset(CODER_TOOLS), True, True,
        "负责在隔离 worktree 中完成代码修改并运行目标测试；可以使用文件工具、"
        "命令、记忆写入和已启用的 MCP 工具。"),
}


@dataclass
class ChildTask:
    child_id: str
    role: str
    task: str
    background: bool
    child_dir: Path
    workdir: Path
    worktree: bool
    status: str = "queued"
    stop_reason: str = ""
    error: str = ""
    summary: str = ""
    changed_files: list[str] = field(default_factory=list)
    model_calls: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    patch_path: str = ""
    report_path: str = ""
    report_markdown_path: str = ""
    report_text: str = ""
    resumes: int = 0
    cancel_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    def as_dict(self) -> dict[str, Any]:
        """父轮次报告里的紧凑视图；完整内容在子 agent 的 report.md。"""
        return self._as_dict(compact=True)

    def as_full_dict(self) -> dict[str, Any]:
        """子 agent 的 report.json；不截断任务、摘要或错误。"""
        return self._as_dict(compact=False)

    def _as_dict(self, *, compact: bool) -> dict[str, Any]:
        return {
            "child_id": self.child_id,
            "role": self.role,
            "task": self.task[:1000] if compact else self.task,
            "background": self.background,
            "status": self.status,
            "workdir": str(self.workdir),
            "worktree": self.worktree,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "summary": self.summary[:1000] if compact else self.summary,
            "changed_files": list(self.changed_files),
            "model_calls": self.model_calls,
            "usage": dict(self.usage),
            "patch": self.patch_path,
            "report": self.report_path,
            "report_md": self.report_markdown_path,
            "resumes": self.resumes,
        }


class DelegationManager:
    """父会话持有的子 agent 生命周期管理器；线程安全。"""

    def __init__(self, settings: Settings, parent_context: ToolContext,
                 parent_session_id: str = "") -> None:
        self.settings = settings
        self.parent_session_id = parent_session_id
        self._parent = parent_context
        # 由 sessions.manager 在 AgentRunner 创建后填入；子 agent 复用它。
        self.parent_stable_prefix = ""
        self._lock = threading.RLock()
        self._children: dict[str, ChildTask] = {}
        self._notifications: list[str] = []
        self._background_slots = threading.BoundedSemaphore(
            max(1, int(settings.delegation_background_max)))
        session_root = (parent_context.checkpoints_db.parent
                        if parent_context.checkpoints_db is not None
                        else parent_context.run_dir)
        self.root = session_root / "subagents"
        self.root.mkdir(parents=True, exist_ok=True)
        self._worktree_root = (
            Path(settings.delegation_worktree_root).expanduser()
            if settings.delegation_worktree_root
            else Path(tempfile.mkdtemp(prefix="nextcoder-worktrees-")))

    # ---- 工具入口 ----

    def update_parent_context(self, context: ToolContext) -> None:
        """会话管理每轮重建 ToolContext；子 agent 管理器跟随最新父上下文。"""
        self._parent = context

    def delegate(self, role: str, task: str, *,
                 background: bool = False) -> dict[str, Any]:
        spec = ROLES.get(role)
        if spec is None:
            return {"status": "failed", "text": f"未知子 agent 角色：{role}",
                    "error": f"未知角色 {role}"}
        task = task.strip()
        if not task:
            return {"status": "failed", "text": "子任务不能为空",
                    "error": "空任务"}
        child = self._create_child(spec, task, background)
        if not background:
            self._execute(child)
            return self._tool_result(child)
        child.thread = threading.Thread(
            target=self._run_background, args=(child,),
            name=f"delegate-{child.child_id}", daemon=True)
        child.thread.start()
        with self._lock:
            self._children[child.child_id] = child
        return {
            "status": "running",
            "text": (f"已在后台启动子 agent {child.child_id}"
                     f"（{child.role}）。继续当前工作；完成通知会自动注入。"),
        }

    def status_text(self, child_id: str) -> str:
        child = self._get(child_id)
        if child is None:
            return f"未找到子 agent：{child_id}"
        with self._lock:
            lines = [
                f"子 agent {child.child_id}（{child.role}）状态：{child.status}",
                f"任务：{child.task[:400]}",
            ]
            if child.stop_reason:
                lines.append(f"停止原因：{child.stop_reason}")
            if child.error:
                lines.append(f"错误：{child.error[:400]}")
            if child.summary:
                lines.append(f"最近摘要：{child.summary[:600]}")
            if child.changed_files:
                lines.append("改动文件：" + ", ".join(child.changed_files[:30]))
            if child.report_markdown_path:
                lines.append(
                    f"完整报告：{child.report_markdown_path}（可用 read_log 回读）")
            if child.report_path:
                lines.append(f"结构化结果：{child.report_path}")
            if child.patch_path:
                lines.append(f"补丁：{child.patch_path}")
            return "\n".join(lines)

    def cancel(self, child_id: str) -> str:
        child = self._get(child_id)
        if child is None:
            return f"未找到子 agent：{child_id}"
        with self._lock:
            if child.status in {"completed", "failed", "cancelled"}:
                return f"子 agent {child_id} 已结束（{child.status}），无需取消。"
            child.cancel_event.set()
            child.status = "cancelling"
        return f"已请求取消子 agent {child_id}；工具与模型循环会在下一个检查点停止。"

    def resume(self, child_id: str) -> dict[str, Any]:
        child = self._get(child_id)
        if child is None:
            return {"status": "failed", "text": f"未找到子 agent：{child_id}",
                    "error": "未找到子任务"}
        if child.status == "completed":
            return {"status": "completed",
                    "text": f"子 agent {child_id} 已经完成，无需恢复。"}
        if child.status in {"running", "queued", "cancelling"}:
            return {"status": "running",
                    "text": f"子 agent {child_id} 仍在运行（{child.status}）。"}
        if child.resumes >= self.settings.delegation_max_resumes:
            return {"status": "failed",
                    "text": f"子 agent {child_id} 已达到恢复上限"
                            f"（{self.settings.delegation_max_resumes} 次）。",
                    "error": "恢复次数已达上限"}
        child.resumes += 1
        child.cancel_event = threading.Event()
        child.error = ""
        self._execute(child, resume=True)
        return self._tool_result(child)

    def poll_notifications(self) -> list[str]:
        with self._lock:
            items = list(self._notifications)
            self._notifications.clear()
            return items

    def summary(self) -> list[dict[str, Any]]:
        with self._lock:
            return [child.as_dict() for child in self._children.values()]

    def close(self) -> None:
        """取消所有子任务并等待短暂收尾；不删除已导出的报告与补丁。"""
        with self._lock:
            children = list(self._children.values())
        for child in children:
            child.cancel_event.set()
        for child in children:
            if child.thread is not None:
                child.thread.join(timeout=5)

    # ---- 子任务执行 ----

    def _create_child(self, spec: RoleSpec, task: str,
                      background: bool) -> ChildTask:
        child_id = f"{spec.name}-{time.strftime('%H%M%S')}-{uuid.uuid4().hex[:6]}"
        child_dir = self.root / child_id
        child_dir.mkdir(parents=True, exist_ok=True)
        if spec.worktree:
            workdir = self._worktree_path(child_id)
        else:
            workdir = self._parent.task_dir
        with self._lock:
            child = ChildTask(
                child_id=child_id, role=spec.name, task=task,
                background=background, child_dir=child_dir, workdir=workdir,
                worktree=spec.worktree, status="queued")
            self._children[child_id] = child
        return child

    def _run_background(self, child: ChildTask) -> None:
        with self._background_slots:
            if child.cancel_event.is_set():
                with self._lock:
                    child.status = "cancelled"
                self._notify(child)
                return
            self._execute(child)
        self._notify(child)

    def _execute(self, child: ChildTask, *, resume: bool = False) -> None:
        spec = ROLES[child.role]
        with self._lock:
            child.status = "running"
        self._emit_child(child)
        store: OperationStore | None = None
        memory: ProjectMemory | None = None
        rollout: RolloutWriter | None = None
        runner: AgentRunner | None = None
        messages: list = []
        try:
            if spec.worktree:
                self._prepare_worktree(child)
            baseline = child.child_dir / "baseline"
            if not baseline.exists():
                copy_baseline(child.workdir, baseline)

            child_settings = self.settings.model_copy(update={
                "model_calls_limit": self.settings.delegation_model_calls_limit,
                "wall_time_seconds": self.settings.delegation_wall_time_seconds,
                "mcp_enabled": self.settings.mcp_enabled and spec.allow_mcp,
            })
            store = OperationStore(child.child_dir / "operations.sqlite")
            memory = (ProjectMemory(child_settings, self._parent.task_dir)
                      if child_settings.memory_enabled else None)
            snapshot: MemorySnapshot | None = self._parent.memory_snapshot
            rollout = RolloutWriter(child.child_dir / "rollout.jsonl",
                                    session_id=child.child_id)
            context = ToolContext(
                task_dir=child.workdir,
                run_dir=child.child_dir,
                baseline_dir=baseline,
                store=store,
                settings=child_settings,
                checkpoints_db=child.child_dir / "checkpoints.sqlite",
                file_states_file=child.child_dir / "file_states.json",
                cancel_event=child.cancel_event,
                turn_id=child.child_id,
                memory=memory,
                memory_snapshot=snapshot,
                skills_catalog=self._parent.skills_catalog,
                allowed_tools=set(spec.tools),
                allow_mcp_tools=spec.allow_mcp,
                allow_delegation=False,
                rollout=rollout,
            )
            runner = AgentRunner(
                child_settings, context,
                session_id=f"{self.parent_session_id}:{child.child_id}",
                stable_system_prompt=self._child_prefix(spec))
            request = "继续未完成的工作。" if resume else child.task
            final = runner.run(request, child.child_id)
            messages = final.get("messages") or []
            with self._lock:
                child.model_calls = int(final.get("model_calls", 0))
                child.usage = dict(final.get("usage") or {})
                child.stop_reason = str(runner.stop_reason or "")
                if child.cancel_event.is_set():
                    child.status = "cancelled"
                elif runner.stop_reason:
                    child.status = "stopped"
                else:
                    child.status = "completed"
                child.summary = _last_assistant_text(messages)
        except Exception as exc:  # noqa: BLE001 - 子任务失败不能拖垮父会话
            with self._lock:
                child.status = "failed"
                child.error = f"{type(exc).__name__}: {exc}"
                child.stop_reason = child.stop_reason or "error"
                child.summary = child.summary or str(exc)
            try:
                (child.child_dir / "error.txt").write_text(
                    traceback.format_exc()[-4000:], encoding="utf-8")
            except OSError:
                pass
        finally:
            if runner is not None:
                runner.close_mcp()
            if store is not None:
                store.close()
            if memory is not None:
                memory.close()
            if rollout is not None:
                rollout.close()
            if spec.worktree:
                try:
                    self._export_and_remove_worktree(child)
                except Exception as exc:  # noqa: BLE001 - 清理失败只记录
                    with self._lock:
                        child.error = (child.error + f"；worktree 清理失败：{exc}").strip("；")
            # 报告必须在 worktree 导出后写：changes.patch 与改动文件列表此时
            # 才完整，失败/取消的子任务也会留下可回读的报告。
            try:
                self._build_report(child, messages)
            except Exception as exc:  # noqa: BLE001 - 报告失败不拖垮父会话
                with self._lock:
                    child.error = (child.error + f"；报告写入失败：{exc}").strip("；")
            self._emit_child(child)

    def _emit_child(self, child: ChildTask) -> None:
        emit(self._parent.event_handler, ChildUpdated(
            turn_id=self._parent.turn_id,
            child_id=child.child_id,
            role=child.role,
            status=child.status,
            detail=(child.error or child.summary or "")[:200],
        ))

    def _child_prefix(self, spec: RoleSpec) -> str:
        parent_prefix = self.parent_stable_prefix.strip()
        role_block = (
            f"[子 agent 角色：{spec.name}]\n{spec.instruction}\n"
            "你只处理交给你的子任务，看不到父 agent 的对话历史。"
            "使用允许的工具独立完成；结束时给出简洁报告：做了什么、"
            "改动文件、测试命令与结果、未解决问题。不要继续委派子任务。")
        return (parent_prefix + "\n\n" + role_block).strip()

    def _prepare_worktree(self, child: ChildTask) -> None:
        if child.workdir.exists():
            return
        if not _is_git_repo(self._parent.task_dir):
            raise RuntimeError("当前任务目录不是 git 仓库，无法为 tester/coder 创建 worktree")
        path = child.workdir
        path.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            ["git", "worktree", "add", "--detach", str(path), "HEAD"],
            cwd=self._parent.task_dir, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"git worktree add 失败：{completed.stderr[:500]}")
        try:
            _seed_worktree(self._parent.task_dir, path)
            self._copy_child_config(path)
            patch = Path(child.patch_path) if child.patch_path else None
            if patch is not None and patch.is_file():
                applied = subprocess.run(
                    ["git", "apply", "--binary", "--whitespace=nowarn", str(patch)],
                    cwd=path, capture_output=True, text=True)
                if applied.returncode != 0:
                    raise RuntimeError(f"恢复补丁失败：{applied.stderr[:500]}")
        except Exception:
            subprocess.run(["git", "worktree", "remove", "--force", str(path)],
                           cwd=self._parent.task_dir, capture_output=True)
            raise

    def _copy_child_config(self, workdir: Path) -> None:
        """把 MCP/权限/hooks 配置复制进 worktree；这些文件通常被 gitignore，不会进补丁。"""
        for relative in (self.settings.mcp_config_file,
                         self.settings.permissions_file,
                         self.settings.hooks_file):
            source = self._parent.task_dir / relative
            if not source.is_file():
                continue
            target = workdir / relative
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            except OSError:
                continue

    def _export_and_remove_worktree(self, child: ChildTask) -> None:
        path = child.workdir
        if not path.exists():
            return
        # 先记录改动再删除 worktree；报告在清理之后生成，仍需完整文件列表。
        changed = self._changed_files(child)
        with self._lock:
            child.changed_files = changed
        patch_path = child.child_dir / "changes.patch"
        try:
            subprocess.run(["git", "add", "-A", "-N"], cwd=path,
                           capture_output=True)
            diff = subprocess.run(
                ["git", "diff", "--binary", "HEAD"], cwd=path,
                capture_output=True)
            if diff.returncode == 0:
                patch_path.write_bytes(diff.stdout)
                with self._lock:
                    child.patch_path = str(patch_path)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(path)],
                           cwd=self._parent.task_dir, capture_output=True)
            subprocess.run(["git", "worktree", "prune"], cwd=self._parent.task_dir,
                           capture_output=True)

    def _build_report(self, child: ChildTask, messages: list) -> None:
        changed = list(child.changed_files)
        if not changed:
            changed = self._changed_files(child)
        commands = _command_results(messages)
        errors = _failed_tool_lines(messages)
        report_path = child.child_dir / "report.json"
        markdown_path = child.child_dir / "report.md"
        report = {
            "schema": "nextcoder.subagent.report.v1",
            **child.as_full_dict(),
            "status": child.status,
            "changed_files": changed,
            # 完整保留最近一次子会话里已有的命令结果与失败信息；工具本身
            # 已按 output_excerpt_chars 限过单条长度，报告层不再二次截断。
            "tests": commands,
            "unresolved": errors,
            "report": str(report_path),
            "report_md": str(markdown_path),
            "report_path": str(report_path),
            "report_markdown_path": str(markdown_path),
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown = self._report_markdown(child, changed, commands, errors)
        markdown_path.write_text(markdown, encoding="utf-8")
        with self._lock:
            child.changed_files = changed
            child.report_path = str(report_path)
            child.report_markdown_path = str(markdown_path)
            child.report_text = self._compact_text(child, commands, errors)

    def _report_markdown(self, child: ChildTask, changed: list[str],
                         commands: list[str], errors: list[str]) -> str:
        """完整报告：不截断最终回复、任务与失败信息，供父 agent read_log 回读。"""
        usage = ", ".join(
            f"{key}={value}" for key, value in sorted(child.usage.items()))
        lines = [
            f"# 子 agent 报告：{child.child_id}",
            "",
            "## 元数据",
            "",
            f"- 角色：{child.role}",
            f"- 状态：{child.status}",
            f"- 停止原因：{child.stop_reason or '（无）'}",
            f"- 模型调用：{child.model_calls}",
            f"- Token 用量：{usage or '（无）'}",
            f"- worktree：{'是' if child.worktree else '否'}",
            f"- 工作目录：{child.workdir}",
            f"- 结构化结果：{child.child_dir / 'report.json'}",
        ]
        if child.patch_path:
            lines.append(f"- 补丁：{child.patch_path}")
        lines += [
            "",
            "## 任务",
            "",
            child.task or "（无）",
            "",
            "## 最终报告",
            "",
            child.summary or "（子 agent 没有返回最终文本）",
            "",
            f"## 改动文件（{len(changed)}）",
            "",
        ]
        lines.extend(f"- {path}" for path in changed)
        if not changed:
            lines.append("（无）")
        lines += ["", f"## 测试与命令（{len(commands)}）", ""]
        if commands:
            for index, command in enumerate(commands, start=1):
                lines += [f"### 命令 {index}", "", command, ""]
        else:
            lines += ["（无）", ""]
        lines += ["", f"## 未解决/失败（{len(errors)}）", ""]
        if errors:
            for index, error in enumerate(errors, start=1):
                lines += [f"### 失败 {index}", "", error, ""]
        else:
            lines.append("（无）")
        if child.error:
            lines += ["", "## 运行错误", "", child.error]
        return "\n".join(lines).rstrip() + "\n"

    def _compact_text(self, child: ChildTask, commands: list[str],
                      errors: list[str]) -> str:
        """父模型看到的紧凑报告：元数据与产物路径固定保留，正文按预算折叠。"""
        header = [
            f"子 agent {child.child_id}（{child.role}）状态：{child.status}",
            f"任务：{_one_line(child.task, 240)}",
        ]
        if child.changed_files:
            shown = child.changed_files[:30]
            suffix = (f" 等 {len(child.changed_files)} 个"
                      if len(child.changed_files) > len(shown) else "")
            header.append("改动文件：" + ", ".join(shown) + suffix)
        # 这些路径是完整报告的唯一入口，必须在任何正文折叠之外。
        artifacts: list[str] = []
        if child.report_markdown_path:
            artifacts.append(
                f"完整报告：{child.report_markdown_path}（可用 read_log 回读）")
        if child.report_path:
            artifacts.append(f"结构化结果：{child.report_path}")
        if child.patch_path:
            artifacts.append(f"补丁：{child.patch_path}")
        if child.error and child.status == "failed":
            artifacts.append(f"错误：{_one_line(child.error, 300)}")

        body: list[str] = []
        if child.summary:
            body += ["## 最终报告", child.summary]
        if commands:
            body += ["## 最近命令结果", *commands[-3:]]
        if errors:
            body += ["## 未解决/失败", *errors[-5:]]
        prefix = "\n".join(header + artifacts)
        body_text = "\n".join(body).strip()
        limit = max(1200, int(self.settings.delegation_report_chars))
        if not body_text:
            return prefix
        # 给固定元数据和折叠标记留出空间，正文只使用剩余预算。
        marker_hint = "\n…（正文超出预算，已折叠；完整内容见上方完整报告路径）"
        body_budget = max(200, limit - len(prefix) - len(marker_hint) - 2)
        clipped = _head_tail(
            body_text, body_budget,
            marker=marker_hint.strip(),
        )
        return prefix + "\n" + clipped

    def _changed_files(self, child: ChildTask) -> list[str]:
        if not child.worktree or not child.workdir.exists():
            return []
        completed = subprocess.run(
            ["git", "status", "--porcelain"], cwd=child.workdir,
            capture_output=True, text=True)
        if completed.returncode != 0:
            return []
        files: list[str] = []
        for line in completed.stdout.splitlines():
            value = line[3:].strip()
            if " -> " in value:
                value = value.split(" -> ", 1)[1]
            if value:
                files.append(value)
        return files

    def _tool_result(self, child: ChildTask) -> dict[str, Any]:
        with self._lock:
            text = getattr(child, "report_text", "") or self.status_text(child.child_id)
            return {
                "status": child.status,
                "text": text,
                "error": child.error or None,
            }

    def _notify(self, child: ChildTask) -> None:
        message = (
            f"子 agent {child.child_id}（{child.role}）已结束：{child.status}。"
            f"{_one_line(child.summary, 300)}")
        if child.report_markdown_path:
            message += f"\n完整报告：{child.report_markdown_path}"
        if child.report_path:
            message += f"\n结构化结果：{child.report_path}"
        with self._lock:
            self._notifications.append(message)

    def _get(self, child_id: str) -> ChildTask | None:
        with self._lock:
            return self._children.get(child_id)

    def _worktree_path(self, child_id: str) -> Path:
        return self._worktree_root / child_id


def _is_git_repo(path: Path) -> bool:
    completed = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"], cwd=path,
        capture_output=True, text=True)
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def _seed_worktree(source: Path, target: Path) -> None:
    """把父目录未提交的改动与未跟踪文件复制到 worktree，保证起点一致。"""
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary"], cwd=source, capture_output=True)
    if diff.returncode != 0:
        raise RuntimeError(
            f"读取父目录 diff 失败：{diff.stderr.decode('utf-8', 'replace')[:500]}")
    if diff.stdout.strip():
        applied = subprocess.run(
            ["git", "apply", "--binary", "--whitespace=nowarn", "-"],
            cwd=target, input=diff.stdout, capture_output=True)
        if applied.returncode != 0:
            raise RuntimeError(
                f"复制父目录改动失败：{applied.stderr.decode('utf-8', 'replace')[:500]}")
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=source, capture_output=True)
    if untracked.returncode != 0:
        return
    for raw in untracked.stdout.split(b"\0"):
        if not raw:
            continue
        relative = raw.decode("utf-8", "replace")
        if relative.startswith(".nextcoder/"):
            continue
        src = source / relative
        dst = target / relative
        try:
            if src.is_symlink():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(src.readlink())
            elif src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        except OSError:
            continue


def _last_assistant_text(messages: list) -> str:
    from langchain_core.messages import AIMessage
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, list):
                try:
                    return json.dumps(content, ensure_ascii=False)
                except TypeError:
                    return str(content)
            if content:
                return str(content)
    return ""


def _command_results(messages: list) -> list[str]:
    """保留完整工具结果字符串；工具层已按 output_excerpt_chars 限制单条长度。"""
    results: list[str] = []
    for message in messages:
        content = str(getattr(message, "content", ""))
        if content.startswith("[run_command] status="):
            results.append(content)
    return results


def _failed_tool_lines(messages: list) -> list[str]:
    errors: list[str] = []
    for message in messages:
        content = str(getattr(message, "content", ""))
        if "status=failed" in content or "status=unknown" in content:
            errors.append(content)
    return errors


def _one_line(text: str, limit: int) -> str:
    """单行元数据摘录；只用于任务/错误概览，不用于报告正文。"""
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "…"


def _head_tail(text: str, limit: int, *, marker: str) -> str:
    """按 dsh 工具输出的 4:1 头尾比例折叠，避免只保留开头。

    头尾都优先落在换行边界；没有换行时才在字符边界切，并在切口补省略号。
    完整正文已写入 report.md，这里只负责给父模型一个稳定、可读的摘要。
    """
    if len(text) <= limit:
        return text
    note = (f"\n…（{marker}；正文共 {len(text)} 字符，"
            "完整内容见上方完整报告路径）\n")
    available = max(80, limit - len(note))
    head_budget = max(40, available * 4 // 5)
    tail_budget = max(20, available - head_budget)
    head = text[:head_budget]
    newline = head.rfind("\n")
    if newline >= head_budget // 2:
        head = head[:newline]
    tail = text[-tail_budget:]
    newline = tail.find("\n")
    if 0 <= newline <= tail_budget // 2:
        tail = tail[newline + 1:]
    head = head.rstrip()
    tail = tail.lstrip("\n")
    if head and not head.endswith(("。", "！", "？", ".", "!", "?")):
        head += "…"
    if tail and not tail.startswith("…"):
        tail = "…" + tail
    return head + note + tail
