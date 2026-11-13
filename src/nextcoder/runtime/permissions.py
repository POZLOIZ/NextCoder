"""权限判定：动作 + 目标 → 允许/拒绝/询问。

四种模式，从最谨慎到最宽松：

- manual（逐项确认）：每次写文件、删文件、运行命令、调用外部工具都询问；
- workspace（工作区写入）：任务目录内读写自动放行，命令和外部工具询问；
- auto（沙箱内自动）：任务目录内读写自动放行；命令只在沙箱真实可用时
  自动放行，否则询问；
- bypass（完全放行）：除不可逆命令底线外全部放行。

非交互运行没有"询问用户"的入口：需要确认的操作按拒绝处理，并把原因和
可行动建议返回给模型。交互层传入 confirmable=True 时才保留 ask 分支。
硬底线和规则文件里的显式 deny 永远只拒绝。规则文件（默认
.nextcoder/permissions.json）中 deny 优先于 allow。所有模式都保留不可逆
命令底线——bypass 也拦 rm 根目录、mkfs、dd 写设备、提权和 fork 炸弹。
文件工具无论哪种模式都不能越过任务目录，这是工具层的硬边界。
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass
from pathlib import Path

from nextcoder.settings import Settings

READ_PATH = "read_path"
WRITE_PATH = "write_path"
DELETE_PATH = "delete_path"
EXECUTE_SHELL = "execute_shell"
MCP_TOOL = "mcp_tool"

# 敏感路径：manual/workspace/auto 下读写前询问（密钥、环境配置、git 历史）。
SENSITIVE_PATTERNS = (".env", "*.env", ".env.*", "*.pem", "*.key", "*.key/*", ".git", ".git/*")

# 不可逆底线：任何模式都拒绝。模式串取自 CodeAgent gate0/1 的子集。
_FLOOR_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsudo\b|\bsu\s+(?:-\w*\s*\w*|\w+)"), "提权命令（sudo/su）"),
    (re.compile(r"\bmkfs(?:\.\w+)?\b"), "格式化文件系统（mkfs）"),
    (re.compile(r"\bdd\b[^;|&]*\bof=/dev/"), "向块设备写入（dd of=/dev/…）"),
    (re.compile(r"\brm\s+(?:-{1,2}[\w=]+\s+)+['\"]?/(?:\*?['\"]?)(?:\s|$)"),
     "删除根目录（rm … /）"),
    (re.compile(r"\bchmod\b[^;|&]*R[^;|&]*\s/\s*$"), "递归修改根目录权限"),
    (re.compile(r":\s*\(\s*\)\s*\{[^}]*\}\s*;\s*:"), "fork 炸弹"),
    (re.compile(r"\b(?:shutdown|reboot|halt|poweroff)\b|\bsystemctl\s+(?:poweroff|reboot|halt)"),
     "关机/重启"),
)

_RULE_HINT = "如需放行，请在任务目录的 .nextcoder/permissions.json 添加显式 allow 规则"


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    reason: str = ""
    hint: str = ""
    # 表示“当前模式要求先询问用户”；硬底线和显式 deny 永远为 False。
    ask: bool = False


@dataclass(frozen=True)
class PermissionRequest:
    """交给交互层确认的一次权限请求；字段只用于展示与审计。"""

    action: str
    target: str
    tool: str = ""
    reason: str = ""
    hint: str = ""


@dataclass(frozen=True)
class PermissionRules:
    """permissions.json 的 allow/deny 规则；scope 决定匹配方式。"""

    allow: tuple[tuple[str, str], ...] = ()  # (scope, pattern)
    deny: tuple[tuple[str, str], ...] = ()


def load_rules(path: Path) -> PermissionRules:
    """读取规则文件；不存在或格式错误时返回空规则（不阻断运行）。"""
    if not path.is_file():
        return PermissionRules()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return PermissionRules()
    allow = _parse_scope(data.get("allow"))
    deny = _parse_scope(data.get("deny"))
    return PermissionRules(allow=tuple(allow), deny=tuple(deny))


def _parse_scope(entries) -> list[tuple[str, str]]:
    rules: list[tuple[str, str]] = []
    if not isinstance(entries, list):
        return rules
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        scope = str(entry.get("scope") or "").strip()
        pattern = str(entry.get("pattern") or "").strip()
        if scope in {"path", "command_prefix", "mcp_tool"} and pattern:
            rules.append((scope, pattern))
    return rules


def action_for_call(name: str, args: dict) -> tuple[str, str] | None:
    """把工具调用映射为 (动作, 目标)；内部状态工具（记忆/清单/技能）返回 None。"""
    if name in {"list_files", "view_diff"}:
        return (READ_PATH, ".")
    if name == "search":
        return (READ_PATH, str(args.get("path") or "."))
    if name == "read_lines":
        return (READ_PATH, str(args.get("path") or ""))
    if name == "read_log":
        return (READ_PATH, str(args.get("artifact_path") or ""))
    if name in {"write_file", "apply_patch"}:
        return (WRITE_PATH, str(args.get("path") or ""))
    if name == "run_command":
        return (EXECUTE_SHELL, str(args.get("command") or ""))
    if name.startswith("mcp__"):
        parts = name.split("__")
        if len(parts) == 3:
            return (MCP_TOOL, f"{parts[1]}:{parts[2]}")
        return (MCP_TOOL, name)
    return None


class PermissionGate:
    """权限判定入口：底线 → deny 规则 → allow 规则 → 模式默认策略。"""

    def __init__(self, mode: str, task_dir: Path, rules: PermissionRules) -> None:
        self.mode = mode
        self.task_dir = Path(task_dir).resolve()
        self.rules = rules

    @classmethod
    def from_settings(cls, settings: Settings, task_dir: Path) -> "PermissionGate":
        return cls(
            settings.permission_mode,
            Path(task_dir).resolve(),
            load_rules(Path(task_dir) / settings.permissions_file),
        )

    def check(self, action: str, target: str, *,
              confirmable: bool = False,
              sandboxed: bool = False) -> PermissionDecision:
        if action == EXECUTE_SHELL:
            floor = _floor_violation(target)
            if floor is not None:
                return PermissionDecision(False, floor, "不可逆命令任何模式下都会被拒绝")
        for scope, pattern in self.rules.deny:
            if _rule_matches(scope, pattern, action, target):
                return PermissionDecision(
                    False, f"规则明确禁止（{scope}: {pattern}）", _RULE_HINT)
        for scope, pattern in self.rules.allow:
            if _rule_matches(scope, pattern, action, target):
                return PermissionDecision(True)
        decision = self._mode_default(action, target, sandboxed=sandboxed)
        if decision.ask and not confirmable:
            # 非交互运行没有确认回调；需要询问的操作按拒绝处理。
            return PermissionDecision(False, decision.reason, decision.hint)
        return decision

    def _mode_default(self, action: str, target: str, *,
                      sandboxed: bool = False) -> PermissionDecision:
        if self.mode == "bypass":
            return PermissionDecision(True)
        if action in {READ_PATH, WRITE_PATH, DELETE_PATH}:
            relative = self._relative(target)
            if relative is None:
                # 路径约束是工具本身也执行的硬边界，不提供“确认后越界”的选项。
                return PermissionDecision(False, f"路径越出任务目录：{target}",
                                          "只能在任务目录内操作")
            if _is_sensitive(relative):
                return PermissionDecision(
                    False, f"这是敏感文件：{relative}",
                    "确认只做必要的读写；长期放行可在 "
                    ".nextcoder/permissions.json 添加 allow 规则",
                    ask=True)
            if action == WRITE_PATH and self.mode == "manual":
                return PermissionDecision(
                    False, f"逐项确认模式：修改 {relative} 前需要你同意",
                    "需要连续修改时切换到“工作区写入”模式", ask=True)
            if action == DELETE_PATH and self.mode != "bypass":
                return PermissionDecision(
                    False, f"删除 {relative} 前需要你同意",
                    "删除不可撤销，确认后会立即执行", ask=True)
            return PermissionDecision(True)
        if action == EXECUTE_SHELL:
            if self.mode == "manual":
                return PermissionDecision(
                    False, "逐项确认模式：运行命令前需要你同意",
                    "需要自动运行测试或构建时切换到“工作区写入”或“沙箱内自动”",
                    ask=True)
            if self.mode == "workspace":
                return PermissionDecision(
                    False, "工作区写入模式：运行命令前需要你同意",
                    "想自动运行沙箱内命令时切换到“沙箱内自动”",
                    ask=True)
            # auto：只有沙箱真实可用才自动放行；否则视为越出沙箱，先询问。
            if sandboxed:
                return PermissionDecision(True)
            return PermissionDecision(
                False, "当前没有可用的沙箱，运行命令前需要你同意",
                "开启命令沙箱并确认 bwrap 可用，或在容器内运行；"
                "也可以临时切换到“完全放行”",
                ask=True)
        if action == MCP_TOOL:
            if self.mode in {"manual", "workspace", "auto"}:
                return PermissionDecision(
                    False, f"调用外部 MCP 工具前需要你同意：{target}",
                    "本会话允许后不再重复询问；也可在 permissions.json "
                    "添加 mcp_tool allow 规则",
                    ask=True)
            return PermissionDecision(True)
        return PermissionDecision(True)

    def _relative(self, target: str) -> str | None:
        """目标路径转为任务目录内相对路径；越出目录返回 None。"""
        raw = Path(target)
        candidate = raw if raw.is_absolute() else (self.task_dir / raw)
        try:
            resolved = candidate.resolve()
            relative = resolved.relative_to(self.task_dir)
        except (OSError, ValueError):
            return None
        return relative.as_posix()


def _floor_violation(command: str) -> str | None:
    for pattern, reason in _FLOOR_RULES:
        if pattern.search(command):
            return reason
    return None


def _rule_matches(scope: str, pattern: str, action: str, target: str) -> bool:
    if scope == "path" and action in {READ_PATH, WRITE_PATH, DELETE_PATH}:
        normalized = target.replace("\\", "/").lstrip("./")
        return fnmatch.fnmatch(normalized, pattern)
    if scope == "command_prefix" and action == EXECUTE_SHELL:
        return target.strip().startswith(pattern)
    if scope == "mcp_tool" and action == MCP_TOOL:
        return fnmatch.fnmatch(target, pattern)
    return False


def _is_sensitive(relative: str) -> bool:
    parts = relative.split("/")
    for pattern in SENSITIVE_PATTERNS:
        if fnmatch.fnmatch(relative, pattern) or any(
                fnmatch.fnmatch(part, pattern) for part in parts):
            return True
    return False
