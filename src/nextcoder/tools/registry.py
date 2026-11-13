"""工具注册表：把 LangChain 工具定义映射到真实执行。

参数校验通过后才执行；执行经过 OperationStore 状态机：
prepare → started → completed / failed / cancelled / timeout / unknown。
调用方传入模型的 tool call id 作为操作 ID；同一 ID 已完成时直接复用结果，
中断后状态不明的调用不自动重放。所有外部副作用工具（文件/命令/MCP）在
执行前经过权限判定，拒绝以 [权限拒绝] 前缀返回给模型。
"""

import re
from typing import TYPE_CHECKING, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from nextcoder.runtime.events import (
    PermissionAnswered,
    PermissionRequested,
    ToolFinished,
    ToolStarted,
    emit,
)
from nextcoder.runtime.mcp import split_tool_name
from nextcoder.runtime.permissions import (
    EXECUTE_SHELL,
    PermissionDecision,
    PermissionRequest,
    action_for_call,
)
from nextcoder.sessions.store import TERMINAL_STATUSES
from nextcoder.tools import commands, edits, files
from nextcoder.tools import memory as memory_tools
from nextcoder.tools import planning as planning_tools
from nextcoder.tools import skills as skill_tools
from nextcoder.tools.base import ToolContext, ToolError, ToolResult

if TYPE_CHECKING:
    from nextcoder.runtime.hooks import HookBus


class ListFilesArgs(BaseModel):
    path: str = Field(default=".", description="相对任务目录的路径")


class SearchArgs(BaseModel):
    pattern: str = Field(description="搜索文本（固定字符串，按字面量匹配）")
    path: str = Field(default=".", description="搜索范围")
    offset: int = Field(default=0, ge=0, description="结果翻页偏移")


class ReadLinesArgs(BaseModel):
    path: str = Field(description="文件路径")
    start: int = Field(default=1, ge=1, description="起始行（1-based）")
    end: int | None = Field(default=None, description="结束行，默认读 200 行")


class ReadLogArgs(BaseModel):
    artifact_path: str = Field(description="工具返回的 full_output 路径")
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = None


class WriteFileArgs(BaseModel):
    path: str = Field(description="目标文件路径")
    content: str = Field(description="完整文件内容（整体覆盖）")
    expected_hash: str | None = Field(
        default=None,
        description="上次 read_lines / apply_patch 返回的 content_hash；已有文件必填",
    )


class ApplyPatchArgs(BaseModel):
    path: str = Field(description="目标文件路径")
    old_text: str = Field(description="要替换的原文，必须与当前文件完全一致")
    new_text: str = Field(description="替换后的内容")
    expected_hash: str = Field(
        description="上次 read_lines 返回的 content_hash，用于确认文件未被外部修改"
    )
    replace_all: bool = Field(default=False, description="是否替换所有匹配位置")


class RunCommandArgs(BaseModel):
    command: str = Field(description="bash 命令")
    timeout: int | None = Field(default=None, ge=1, description="秒；默认用配置上限")


class ViewDiffArgs(BaseModel):
    path: str | None = Field(default=None, description="只看某个文件；默认全部改动")


class MemoryListArgs(BaseModel):
    query: str = Field(default="", description="按主题、类型或正文过滤；空字符串列出全部")


class MemoryReadArgs(BaseModel):
    topic: str = Field(description="记忆主题键，先用 memory_list 查看")


class MemoryWriteArgs(BaseModel):
    topic: str = Field(description="稳定、简短的主题键；同一主题会更新而不是新增")
    content: str = Field(description="要长期保留的正文；只写跨会话有价值的事实")
    kind: str = Field(
        default="project_convention",
        description="user_preference/project_convention/environment/test_command/"
                    "lint_command/project_language/known_failure",
    )


class MemoryForgetArgs(BaseModel):
    topic: str = Field(description="要停用的记忆主题键")


class PlanCreateArgs(BaseModel):
    tasks: list[str] = Field(description="按执行顺序排列的任务描述列表，每项一句话")


class PlanUpdateArgs(BaseModel):
    statuses: dict[str, str] = Field(
        description="任务 id 到新状态的映射；状态可选 pending/in_progress/completed/cancelled")


class LoadSkillArgs(BaseModel):
    name: str = Field(description="技能名，见系统提示中的项目技能目录")


class DelegateArgs(BaseModel):
    role: Literal["researcher", "reviewer", "tester", "coder"] = Field(
        description="子 agent 角色：researcher/reviewer 只读，tester 可跑命令，coder 可改代码")
    task: str = Field(description="交给子 agent 的独立任务；它看不到父对话历史")
    background: bool = Field(
        default=False,
        description="true 时立即返回 child_id，父 agent 继续；完成通知稍后注入")


class DelegationChildArgs(BaseModel):
    child_id: str = Field(description="delegate 返回的子任务 id")


# MCP inputSchema 的 JSON Schema 类型到 Python 类型的映射；未知类型退化为 str。
_SCHEMA_TYPES = {"string": str, "integer": int, "number": float, "boolean": bool}


def schema_to_args_model(tool_name: str, schema: dict) -> type[BaseModel]:
    """把 MCP 工具的 inputSchema 转成 pydantic 参数模型（LangChain 需要）。"""
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: dict[str, tuple] = {}
    for prop, spec in properties.items():
        spec = spec if isinstance(spec, dict) else {}
        py_type = _SCHEMA_TYPES.get(spec.get("type"), str)
        if prop in required:
            fields[prop] = (py_type, Field(default=...,
                                           description=str(spec.get("description", ""))))
        else:
            fields[prop] = (py_type | None, Field(
                default=None, description=str(spec.get("description", ""))))
    safe_name = re.sub(r"\W", "_", tool_name)
    return create_model(f"{safe_name}Args", **fields)


class ToolBox:
    """绑定一次运行上下文的工具集；execute 是所有工具的唯一执行入口。"""

    def __init__(self, context: ToolContext, mcp=None,
                 hooks: "HookBus | None" = None) -> None:
        self.context = context
        self.mcp = mcp
        self.hooks = hooks
        self.results: list[ToolResult] = []

    def execute(self, name: str, args: dict, call_id: str | None = None, *,
                group_signature: str = "", group_index: int = 0,
                group_size: int = 1) -> str:
        store = self.context.store
        if not self._tool_allowed(name):
            # 白名单同时约束 definitions 与执行：模型伪造工具名也无法越过角色边界。
            return f"[工具不可用] 当前角色不允许调用 {name}"
        if call_id is not None:
            existing = store.get(call_id)
            if existing is not None:
                if existing.result_text is not None:
                    return existing.result_text
                if existing.status in TERMINAL_STATUSES or existing.status == "started":
                    return (
                        f"[恢复] 操作 {call_id}（{existing.tool}）之前状态为 "
                        f"{existing.status}，为避免重复副作用，本次不自动重放。"
                        "请核对当前文件或输出后再决定下一步。"
                    )
        op_id = store.prepare(name, args, call_id=call_id, turn_id=self.context.turn_id)
        event_call_id = call_id or op_id
        notes: list[str] = []
        # before hook 在权限判定和执行之前运行：可以阻止调用或替换参数。
        if self.hooks is not None:
            hook = self.hooks.before_tool_use(
                name, event_call_id, args, self.context.task_dir,
                group_signature=group_signature, group_index=group_index,
                group_size=group_size)
            notes.extend(hook.notes)
            if hook.args is not None:
                args = hook.args
                # 审计记录必须和真正执行的参数一致，否则事后回放会看到旧命令。
                store.update_args(op_id, args)
            blocked_by_hook = hook.blocked
            hook_reason = hook.reason
        else:
            blocked_by_hook, hook_reason = False, ""
        # ToolStarted 在 arg 替换之后发送，界面显示的是实际要执行的参数。
        emit(self.context.event_handler,
             ToolStarted(self.context.turn_id, event_call_id, name,
                         _event_detail(args)))
        if blocked_by_hook:
            result = ToolResult(
                op_id, name, "failed", error=f"[钩子阻止] {hook_reason}")
        else:
            # 权限判定必须在 started 之前：被拒绝的操作没有副作用，直接 prepared→failed。
            decision = self._permission_decision(name, args)
            if decision is not None and not decision.allowed:
                result = ToolResult(op_id, name, "failed",
                                    error=self._denial_text(decision))
            else:
                store.transition(op_id, "started")
                try:
                    result = self._dispatch(name, op_id, args)
                except ToolError as exc:
                    result = ToolResult(op_id, name, "failed", error=str(exc))
                except Exception as exc:  # noqa: BLE001 - 工具异常必须返回给模型而不是崩溃
                    result = ToolResult(op_id, name, "unknown",
                                        error=f"{type(exc).__name__}: {exc}")
        # after hook 总会看到最终结果（包括拒绝/阻止/异常），但不能再改变副作用。
        if self.hooks is not None:
            after = self.hooks.after_tool_use(
                name, event_call_id, args, self.context.task_dir, result)
            notes.extend(after.notes)
        text = result.as_text(self.context.settings)
        if notes:
            text += "\n" + "\n".join(notes)
        self.results.append(result)
        status = result.status if result.status in {
            "completed", "failed", "cancelled", "timeout", "unknown"
        } else "unknown"
        store.transition(
            op_id, status,
            related_files=result.changed_files or None,
            exit_code=result.exit_code,
            log_path=result.artifact_path,
            error=result.error,
            result_text=text,
        )
        emit(self.context.event_handler,
             ToolFinished(self.context.turn_id, event_call_id, name, status,
                          _event_detail({"text": text})))
        return text

    def _permission_decision(self, name: str, args: dict) -> PermissionDecision | None:
        """外部副作用工具执行前的权限判定；交互模式下可转为 ask。"""
        mapped = action_for_call(name, args)
        if mapped is None:
            return None
        key = mapped
        if key in self.context.permission_approvals:
            return PermissionDecision(True)
        confirmable = self.context.approval_callback is not None
        decision = self.context.permissions.check(
            *mapped, confirmable=confirmable,
            sandboxed=(mapped[0] == EXECUTE_SHELL
                       and self._command_is_sandboxed()))
        if decision.allowed or not decision.ask:
            return decision
        request = PermissionRequest(
            action=mapped[0], target=mapped[1], tool=name,
            reason=decision.reason, hint=decision.hint)
        self._record_permission("permission_requested", request,
                                reason=decision.reason, hint=decision.hint)
        try:
            # confirmable=True 保证回调存在；异常按拒绝处理，避免绕过审批。
            answer = str(self.context.approval_callback(request))
        except Exception as exc:  # noqa: BLE001 - 审批失败不能放行
            self._record_permission(
                "permission_answered", request, answer="error",
                detail=f"{type(exc).__name__}: {exc}")
            return PermissionDecision(
                False, f"权限确认失败（{type(exc).__name__}）",
                "重试或改用不需要确认的操作")
        self._record_permission("permission_answered", request, answer=answer)
        if answer == "allow_session":
            self.context.permission_approvals.add(key)
            return PermissionDecision(True)
        if answer == "allow_once":
            return PermissionDecision(True)
        return PermissionDecision(False, f"用户拒绝：{decision.reason}", decision.hint)

    def _command_is_sandboxed(self) -> bool:
        """命令是否已经在隔离边界内运行。

        容器执行、bwrap 可用、或当前进程本身就在容器里，三者满足其一即可
        视为“在沙箱内”。auto 模式只在这个条件成立时自动放行命令。
        """
        settings = self.context.settings
        if settings.command_container or commands.running_in_container():
            return True
        if not settings.command_sandbox:
            return False
        state, _path, _detail = commands.bwrap_status()
        return state == "ok"

    def _record_permission(self, event_type: str, request: PermissionRequest,
                           **extra) -> None:
        if event_type == "permission_requested":
            emit(self.context.event_handler, PermissionRequested(
                self.context.turn_id, request.tool, request.action,
                request.target, request.reason, request.hint))
        elif event_type == "permission_answered":
            emit(self.context.event_handler, PermissionAnswered(
                self.context.turn_id, request.tool, request.action,
                request.target, str(extra.get("answer", ""))))
        if self.context.rollout is None:
            return
        payload = {
            "tool": request.tool, "action": request.action,
            "target": request.target, **extra,
        }
        self.context.rollout.append(event_type, payload,
                                    turn_id=self.context.turn_id)

    @staticmethod
    def _denial_text(decision: PermissionDecision) -> str:
        parts = [decision.reason]
        if decision.hint:
            parts.append(decision.hint)
        return f"[权限拒绝] {'；'.join(parts)}"

    def _dispatch(self, name: str, call_id: str, args: dict) -> ToolResult:
        if name == "list_files":
            return files.list_files(self.context, call_id, **args)
        if name == "search":
            return files.search(self.context, call_id, **args)
        if name == "read_lines":
            return files.read_lines(self.context, call_id, **args)
        if name == "read_log":
            return files.read_log(self.context, call_id, **args)
        if name == "write_file":
            return edits.write_file(self.context, call_id, **args)
        if name == "apply_patch":
            return edits.apply_patch(self.context, call_id, **args)
        if name == "view_diff":
            return edits.view_diff(self.context, call_id, **args)
        if name == "run_command":
            return commands.run_command(self.context, call_id, **args)
        if name == "memory_list":
            return memory_tools.memory_list(self.context, call_id, **args)
        if name == "memory_read":
            return memory_tools.memory_read(self.context, call_id, **args)
        if name == "memory_write":
            return memory_tools.memory_write(self.context, call_id, **args)
        if name == "memory_forget":
            return memory_tools.memory_forget(self.context, call_id, **args)
        if name == "plan_create":
            return planning_tools.plan_create(self.context, call_id, **args)
        if name == "plan_update":
            return planning_tools.plan_update(self.context, call_id, **args)
        if name == "load_skill":
            return skill_tools.load_skill(self.context, call_id, **args)
        if name == "delegate":
            return self._dispatch_delegate(call_id, **args)
        if name == "delegation_status":
            return self._dispatch_delegation_status(call_id, **args)
        if name == "delegation_cancel":
            return self._dispatch_delegation_cancel(call_id, **args)
        if name == "delegation_resume":
            return self._dispatch_delegation_resume(call_id, **args)
        if name.startswith("mcp__"):
            return self._dispatch_mcp(call_id, name, args)
        raise ToolError(f"未知工具：{name}")

    def _tool_allowed(self, name: str) -> bool:
        if self.context.allowed_tools is None:
            return True
        if name in self.context.allowed_tools:
            return True
        return self.context.allow_mcp_tools and name.startswith("mcp__")

    def _dispatch_delegate(self, call_id: str, role: str, task: str,
                           background: bool = False) -> ToolResult:
        manager = self.context.delegation_manager
        if manager is None:
            raise ToolError("子 agent 未启用")
        result = manager.delegate(role, task, background=background)
        status = "completed" if result["status"] != "failed" else "failed"
        return ToolResult(call_id, "delegate", status,
                          output_excerpt=result["text"],
                          error=result.get("error"))

    def _dispatch_delegation_status(self, call_id: str,
                                    child_id: str) -> ToolResult:
        manager = self.context.delegation_manager
        if manager is None:
            raise ToolError("子 agent 未启用")
        return ToolResult(call_id, "delegation_status", "completed",
                          output_excerpt=manager.status_text(child_id))

    def _dispatch_delegation_cancel(self, call_id: str,
                                    child_id: str) -> ToolResult:
        manager = self.context.delegation_manager
        if manager is None:
            raise ToolError("子 agent 未启用")
        return ToolResult(call_id, "delegation_cancel", "completed",
                          output_excerpt=manager.cancel(child_id))

    def _dispatch_delegation_resume(self, call_id: str,
                                    child_id: str) -> ToolResult:
        manager = self.context.delegation_manager
        if manager is None:
            raise ToolError("子 agent 未启用")
        result = manager.resume(child_id)
        status = "completed" if result["status"] != "failed" else "failed"
        return ToolResult(call_id, "delegation_resume", status,
                          output_excerpt=result["text"],
                          error=result.get("error"))

    def _dispatch_mcp(self, call_id: str, name: str, args: dict) -> ToolResult:
        from nextcoder.runtime.mcp import render_mcp_result

        if self.mcp is None:
            raise ToolError(f"未知工具：{name}（MCP 未启用或未配置）")
        parsed = split_tool_name(name)
        if parsed is None:
            raise ToolError(f"非法 MCP 工具名：{name}")
        server, tool = parsed
        result = self.mcp.call_tool(server, tool, args)
        rendered = render_mcp_result(result)
        if getattr(result, "isError", False):
            return ToolResult(call_id, name, "failed", output_excerpt=rendered,
                              error="MCP 服务器返回错误")
        return ToolResult(call_id, name, "completed", output_excerpt=rendered)

    def definitions(self) -> list:
        """给模型的工具定义（StructuredTool：显式命名，与 execute 注册一致）。"""
        definitions = [
            StructuredTool.from_function(self._list_files, name="list_files",
                                         args_schema=ListFilesArgs),
            StructuredTool.from_function(self._search, name="search", args_schema=SearchArgs),
            StructuredTool.from_function(self._read_lines, name="read_lines",
                                         args_schema=ReadLinesArgs),
            StructuredTool.from_function(self._read_log, name="read_log",
                                         args_schema=ReadLogArgs),
            StructuredTool.from_function(self._write_file, name="write_file",
                                         args_schema=WriteFileArgs),
            StructuredTool.from_function(self._apply_patch, name="apply_patch",
                                         args_schema=ApplyPatchArgs),
            StructuredTool.from_function(self._run_command, name="run_command",
                                         args_schema=RunCommandArgs),
            StructuredTool.from_function(self._view_diff, name="view_diff",
                                         args_schema=ViewDiffArgs),
            StructuredTool.from_function(self._memory_list, name="memory_list",
                                         args_schema=MemoryListArgs),
            StructuredTool.from_function(self._memory_read, name="memory_read",
                                         args_schema=MemoryReadArgs),
            StructuredTool.from_function(self._memory_write, name="memory_write",
                                         args_schema=MemoryWriteArgs),
            StructuredTool.from_function(self._memory_forget, name="memory_forget",
                                         args_schema=MemoryForgetArgs),
        ]
        if self.context.settings.plan_enabled:
            definitions.append(StructuredTool.from_function(
                self._plan_create, name="plan_create", args_schema=PlanCreateArgs))
            definitions.append(StructuredTool.from_function(
                self._plan_update, name="plan_update", args_schema=PlanUpdateArgs))
        # 技能目录为空时不注入 load_skill（无技能任务零开销）。
        if self.context.skills.skills:
            definitions.append(StructuredTool.from_function(
                self._load_skill, name="load_skill", args_schema=LoadSkillArgs))
        if self.mcp is not None:
            definitions.extend(self._mcp_definitions())
        if self.context.allow_delegation and self.context.delegation_manager is not None:
            definitions.extend([
                StructuredTool.from_function(
                    self._delegate, name="delegate", args_schema=DelegateArgs),
                StructuredTool.from_function(
                    self._delegation_status, name="delegation_status",
                    args_schema=DelegationChildArgs),
                StructuredTool.from_function(
                    self._delegation_cancel, name="delegation_cancel",
                    args_schema=DelegationChildArgs),
                StructuredTool.from_function(
                    self._delegation_resume, name="delegation_resume",
                    args_schema=DelegationChildArgs),
            ])
        return [definition for definition in definitions
                if self._tool_allowed(definition.name)]

    def _mcp_definitions(self) -> list:
        definitions = []
        for full_name, info in self.mcp.tools.items():
            wrapper = self._mcp_wrapper(full_name)
            description = (info["description"] or f"MCP 工具 {full_name}")[:1024]
            definitions.append(StructuredTool.from_function(
                wrapper, name=full_name, description=description,
                args_schema=schema_to_args_model(full_name, info["inputSchema"])))
        return definitions

    def _mcp_wrapper(self, full_name: str):
        def invoke(**kwargs) -> str:
            return self.execute(full_name, kwargs)
        return invoke

    # ---- 包装函数：LangChain 会按字段名传入关键字参数 ----

    def _list_files(self, path: str = ".") -> str:
        """列出任务目录中的文件（忽略 .git、虚拟环境等），返回相对路径。"""
        return self.execute("list_files", {"path": path})

    def _search(self, pattern: str, path: str = ".", offset: int = 0) -> str:
        """按固定字符串搜索文本，返回路径、行号与匹配行；结果可翻页。"""
        return self.execute("search", {"pattern": pattern, "path": path, "offset": offset})

    def _read_lines(self, path: str, start: int = 1, end: int | None = None) -> str:
        """按行读取文件，返回“行号: 内容”与该文件的 content_hash；默认读 200 行。"""
        return self.execute("read_lines", {"path": path, "start": start, "end": end})

    def _read_log(self, artifact_path: str, start_line: int = 1,
                  end_line: int | None = None) -> str:
        """按行回读工具的完整输出日志（工具结果里 full_output 指向的文件）。"""
        return self.execute(
            "read_log", {"artifact_path": artifact_path, "start_line": start_line,
                         "end_line": end_line})

    def _write_file(self, path: str, content: str, expected_hash: str | None = None) -> str:
        """写入完整文件内容。已有文件必须先 read_lines，并附上 content_hash。"""
        return self.execute(
            "write_file", {"path": path, "content": content, "expected_hash": expected_hash})

    def _apply_patch(self, path: str, old_text: str, new_text: str,
                     expected_hash: str, replace_all: bool = False) -> str:
        """局部修改：把 old_text 替换为 new_text，并附上 read_lines 的 content_hash。"""
        return self.execute(
            "apply_patch", {"path": path, "old_text": old_text, "new_text": new_text,
                            "expected_hash": expected_hash, "replace_all": replace_all})

    def _run_command(self, command: str, timeout: int | None = None) -> str:
        """在任务目录执行 bash 命令，返回退出码与输出摘录；完整输出落日志可回读。
        stderr 会合并到输出；不要用 2>/dev/null 隐藏错误。管道命令的退出码来自
        最后一个命令，诊断时加 `set -o pipefail`。"""
        return self.execute("run_command", {"command": command, "timeout": timeout})

    def _view_diff(self, path: str | None = None) -> str:
        """查看相对任务起点的改动（unified diff）；不传 path 查看全部改动文件。"""
        return self.execute("view_diff", {"path": path})

    def _memory_list(self, query: str = "") -> str:
        """列出本项目可用的记忆主题；细节用 memory_read 按主题读取。"""
        return self.execute("memory_list", {"query": query})

    def _memory_read(self, topic: str) -> str:
        """读取一条项目记忆的完整内容、来源和状态。"""
        return self.execute("memory_read", {"topic": topic})

    def _memory_write(self, topic: str, content: str,
                      kind: str = "project_convention") -> str:
        """保存或更新一条长期记忆；同一主题会覆盖旧正文，不按整句新增事实。"""
        return self.execute(
            "memory_write", {"topic": topic, "content": content, "kind": kind})

    def _memory_forget(self, topic: str) -> str:
        """停用一条项目记忆；停用后不会自动恢复。"""
        return self.execute("memory_forget", {"topic": topic})

    def _plan_create(self, tasks: list[str]) -> str:
        """创建任务清单（整体替换旧清单，最多 30 项，id 为 t1、t2…）。

        需求包含多个步骤或多个修改点时先建清单；快照会出现在系统提示里。"""
        return self.execute("plan_create", {"tasks": tasks})

    def _plan_update(self, statuses: dict[str, str]) -> str:
        """更新任务项状态（t1→completed 等）；状态可选
        pending/in_progress/completed/cancelled。"""
        return self.execute("plan_update", {"statuses": statuses})

    def _load_skill(self, name: str) -> str:
        """读取一个项目技能的完整指导（SKILL.md 全文）。"""
        return self.execute("load_skill", {"name": name})

    def _delegate(self, role: str, task: str, background: bool = False) -> str:
        """把独立子任务交给子 agent；只读角色共享任务目录，tester/coder 用 worktree。"""
        return self.execute("delegate", {
            "role": role, "task": task, "background": background})

    def _delegation_status(self, child_id: str) -> str:
        """查看某个子 agent 的状态与最近报告。"""
        return self.execute("delegation_status", {"child_id": child_id})

    def _delegation_cancel(self, child_id: str) -> str:
        """请求取消一个正在运行或排队中的子 agent。"""
        return self.execute("delegation_cancel", {"child_id": child_id})

    def _delegation_resume(self, child_id: str) -> str:
        """用子 agent 的检查点继续未完成的工作，最多恢复一次。"""
        return self.execute("delegation_resume", {"child_id": child_id})


def _event_detail(args: dict) -> str:
    """事件里只带一行可读摘要，完整参数仍在操作记录里。"""
    for key in ("command", "path", "pattern", "task", "text", "content", "query"):
        value = args.get(key)
        if value:
            return str(value).replace("\n", " ")[:200]
    return ""

