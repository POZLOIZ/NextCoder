"""工具基础设施：统一返回结构、路径约束、日志落盘与摘录。"""

import hashlib
import json
import threading
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from nextcoder.sessions.store import OperationStore
from nextcoder.settings import Settings

if TYPE_CHECKING:
    from nextcoder.runtime.delegation import DelegationManager
    from nextcoder.runtime.permissions import PermissionGate
    from nextcoder.runtime.planning import PlanStore
    from nextcoder.runtime.skills import SkillCatalog
    from nextcoder.sessions.memory import MemorySnapshot, ProjectMemory
    from nextcoder.sessions.rollout import RolloutWriter

IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", ".nextcoder", "node_modules",
                ".pytest_cache", ".ruff_cache", ".mypy_cache", ".hypothesis",
                ".tox", ".nox", ".data", "*.egg-info", "build", "dist", ".eggs",
                "htmlcov", "result_images"}


class ToolError(Exception):
    """参数或状态错误：返回给模型，让它修正后重试。"""


@dataclass
class ToolResult:
    call_id: str
    tool: str
    status: str  # completed / failed / cancelled / timeout / unknown
    output_excerpt: str = ""
    artifact_path: str | None = None
    changed_files: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    exit_code: int | None = None
    error: str | None = None
    content_hash: str | None = None
    # 不参与正文截断的提示（翻页、总行数、存储上限等）。
    note: str | None = None

    def as_text(self, settings: Settings) -> str:
        """给模型的有限摘录；完整结果在 artifact，可继续回读。"""
        parts = [f"[{self.tool}] status={self.status}"]
        if self.exit_code is not None:
            parts.append(f"exit_code={self.exit_code}")
        if self.duration_seconds:
            parts.append(f"duration={self.duration_seconds:.1f}s")
        if self.changed_files:
            parts.append("changed=" + ",".join(self.changed_files))
        if self.artifact_path:
            parts.append(f"full_output={self.artifact_path}")
        if self.content_hash:
            parts.append(f"content_hash={self.content_hash}")
        text = " ".join(parts)
        if self.output_excerpt:
            excerpt = self.output_excerpt
            # 文件回读工具自身按行窗口限长；子 agent 报告由
            # delegation_report_chars 统一做头尾折叠。两者若再按字符截断，
            # 会切掉翻页信息或完整报告路径，模型只能反复重读。
            if self.tool not in {"read_lines", "read_log", "delegate",
                                 "delegation_status", "delegation_resume"} and \
                    len(excerpt) > settings.output_excerpt_chars:
                excerpt = excerpt[: settings.output_excerpt_chars] + (
                    f"\n…（正文截断，完整内容 {len(self.output_excerpt)} 字符，"
                    f"用 read_log 回读 {self.artifact_path}）"
                )
            text += "\n" + excerpt
        if self.note:
            text += "\n" + self.note
        elif not self.output_excerpt and not self.error:
            # 没有正文也没有错误时必须显式说明，避免模型把空结果当成“内容被省略”。
            text += "\n（无输出）"
        if self.error:
            text += f"\nerror: {self.error}"
        return text


@dataclass
class ToolContext:
    task_dir: Path
    run_dir: Path          # 本轮日志与报告目录
    baseline_dir: Path     # 任务起点的完整副本，diff 基准
    store: OperationStore
    settings: Settings
    # 会话级检查点；缺省放在本轮目录下（单次运行）。
    checkpoints_db: Path | None = None
    # 会话级文件状态登记；缺省放在本轮目录下。
    file_states_file: Path | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    turn_id: str = ""
    # 任务级项目记忆；None 表示关闭或不启用。
    memory: "ProjectMemory | None" = None
    # 会话边界加载的记忆索引；请求组装只复用快照，不在每个模型节点查库。
    memory_snapshot: "MemorySnapshot | None" = None
    # 会话开始时冻结的 AGENTS.md 与技能目录；一起进入稳定系统前缀。
    agents_text: str = ""
    skills_catalog: "SkillCatalog | None" = None
    # 权限确认回调：返回 allow_once / allow_session / deny；None 表示非交互。
    approval_callback: Callable[[object], str] | None = None
    # allow_session 批准的动作/目标；由会话持有，跨轮复用。
    permission_approvals: set[tuple[str, str]] = field(default_factory=set)
    # 运行时事件回调（TUI 流式输出、工具状态、子 agent 状态）。
    event_handler: Callable[[object], None] | None = None
    # 子 agent 禁止嵌套委派。
    allow_delegation: bool = False
    # 子 agent 工具白名单；None 表示不限制（主 agent）。
    allowed_tools: set[str] | None = None
    allow_mcp_tools: bool = False
    # 主 agent 的委派管理器；子 context 为 None。
    delegation_manager: "DelegationManager | None" = None
    # 会话事件日志（JSONL rollout）；None 表示关闭。只追加，不参与恢复。
    rollout: "RolloutWriter | None" = None

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    # ---- 惰性构建的运行时组件：按需创建，缓存在实例上 ----

    @cached_property
    def permissions(self) -> "PermissionGate":
        from nextcoder.runtime.permissions import PermissionGate
        return PermissionGate.from_settings(self.settings, self.task_dir)

    @cached_property
    def plan(self) -> "PlanStore | None":
        from nextcoder.runtime.planning import PlanStore
        if not self.settings.plan_enabled:
            return None
        return PlanStore(self.task_dir / self.settings.runs_root / "task-plan.json")

    @cached_property
    def skills(self) -> "SkillCatalog":
        # 会话开始时已冻结目录；直接构造 ToolContext 的旧路径仍然重扫。
        if self.skills_catalog is not None:
            return self.skills_catalog
        from nextcoder.runtime.skills import discover_skills
        return discover_skills(self.task_dir, self.settings)

    @property
    def checkpoints_path(self) -> Path:
        return self.checkpoints_db or (self.run_dir / "checkpoints.sqlite")

    @property
    def summary_path(self) -> Path:
        """会话级摘要库；单次运行放在本轮目录，会话运行与会话同根。"""
        return self.checkpoints_path.parent / "summaries.sqlite"

    def resolve(self, relative: str) -> Path:
        """把模型给的路径限制在任务目录内。"""
        candidate = (self.task_dir / relative).resolve()
        if not candidate.is_relative_to(self.task_dir.resolve()):
            raise ToolError(f"路径越出任务目录：{relative}")
        return candidate

    def rel(self, path: Path) -> str:
        return str(path.relative_to(self.task_dir))

    def write_log(self, name: str, content: str) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        path = self.logs_dir / name
        path.write_text(content, encoding="utf-8")
        return path

    # ---- 文件状态登记：读取与写入的哈希，供编辑前核对（外部化为 json） ----

    @property
    def file_states_path(self) -> Path:
        return self.file_states_file or (self.run_dir / "file_states.json")

    def load_file_states(self) -> dict[str, str]:
        if self.file_states_path.exists():
            return json.loads(self.file_states_path.read_text(encoding="utf-8"))
        return {}

    def save_file_state(self, relative: str, content: str) -> None:
        states = self.load_file_states()
        states[relative] = hash_content(content)
        self.file_states_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_states_path.write_text(json.dumps(states, ensure_ascii=False, indent=1),
                                         encoding="utf-8")

    def recorded_hash(self, relative: str) -> str | None:
        return self.load_file_states().get(relative)

    # ---- 产物定位：跨轮 read_log 不能只按当前轮目录拼路径 ----

    def resolve_artifact(self, artifact_path: str) -> Path:
        """把工具返回的 full_output 定位到实际文件。

        模型可能传绝对路径、相对任务目录的路径，或只传文件名。只传文件名时，
        先查当前轮，再查会话操作记录里的 log_path；同一会话后续轮次因此仍能
        回读上一轮的日志。所有结果都必须位于任务目录内。
        """
        raw = Path(artifact_path)
        candidates: list[Path] = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(self.run_dir / "logs" / raw.name)
            candidates.append(self.task_dir / raw)
        if not raw.is_absolute() or not raw.exists():
            # 跨轮回读：操作表保存了原始 log_path（绝对路径或任务目录相对路径）。
            for operation in reversed(self.store.all_operations()):
                if operation.log_path and Path(operation.log_path).name == raw.name:
                    candidates.append(Path(operation.log_path))
        root = self.task_dir.resolve()
        allowed_roots = [root, self.run_dir.resolve()]
        if self.checkpoints_db is not None:
            # 会话根同时也是子 agent 日志与报告的上级目录。
            allowed_roots.append(self.checkpoints_db.parent.resolve())
        found: list[Path] = []
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if (resolved.is_file()
                    and any(resolved.is_relative_to(base) for base in allowed_roots)
                    and resolved not in found):
                found.append(resolved)
        if not found:
            raise ToolError(f"产物不存在或不在任务目录内：{artifact_path}")
        if len(found) > 1:
            # 同名产物优先取当前轮，其次取最近一次操作；仍不唯一则要求完整路径。
            current = (self.run_dir / "logs" / raw.name).resolve()
            if current in found:
                return current
            raise ToolError(
                f"找到多个同名产物，请传完整路径：{', '.join(str(p) for p in found[:3])}"
            )
        return found[0]


def hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def is_ignored(path: Path) -> bool:
    return any(part in IGNORED_DIRS or part.endswith(".egg-info") for part in path.parts)
