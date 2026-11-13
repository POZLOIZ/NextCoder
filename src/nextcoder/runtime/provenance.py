"""运行来源：记录 NextCoder 与任务仓库的提交状态，以及非敏感配置快照。

真实模型评测和普通会话都应以提交号 + 配置解释结果；工作区未提交时明确标记
dirty，而不是把结果说成某个已发布版本。
"""

import json
import subprocess
from pathlib import Path

from nextcoder import __version__
from nextcoder.settings import Settings


def runtime_provenance(task_dir: Path, settings: Settings) -> dict:
    nextcoder_root = _git_root(Path(__file__).resolve().parent)
    return {
        "nextcoder_version": __version__,
        "nextcoder_commit": _git(nextcoder_root, "rev-parse", "HEAD"),
        "nextcoder_dirty": _is_dirty(nextcoder_root),
        "task_commit": _git(task_dir, "rev-parse", "HEAD"),
        "task_dirty": _is_dirty(task_dir),
        "settings": settings_snapshot(settings),
    }


def repository_provenance(package_dir: Path, repository_id: str) -> dict:
    """从任务包清单补上固定源码的来源提交与归档校验值。"""
    path = package_dir / "repositories.json"
    if not path.is_file():
        return {}
    try:
        repositories = json.loads(path.read_text(encoding="utf-8")).get("repositories", [])
    except (OSError, json.JSONDecodeError):
        return {}
    info = next((item for item in repositories
                 if item.get("repository_id") == repository_id), None)
    if info is None:
        return {}
    return {
        "repository_id": repository_id,
        "repository_commit": info.get("commit"),
        "repository_tag": info.get("tag"),
        "archive_sha256": info.get("archive_sha256"),
    }


def settings_snapshot(settings: Settings) -> dict:
    """只记录影响行为和成本的配置，不写 API key。"""
    keys = (
        "llm_provider", "model_profile", "llm_base_url", "llm_model",
        "llm_thinking", "model_calls_limit",
        "wall_time_seconds",
        "application_input_token_budget", "output_reserve_tokens",
        "output_excerpt_chars", "max_output_bytes", "command_timeout_seconds",
        "command_path_prepend", "command_container",
        "command_container_workdir", "command_container_activate",
        "compaction_trigger_ratio",
        "compaction_low_watermark_ratio", "compaction_summary_retry_limit",
        "compaction_retry_recent_ratio",
        "compaction_stub_first", "compaction_keep_recent_groups",
        "compaction_recent_token_ratio", "compaction_stub_chars",
        "compaction_summary_chars", "summary_enabled",
        "summary_input_token_budget", "summary_message_chars",
        "agents_enabled", "agents_file", "agents_max_chars",
        "memory_enabled", "memory_prompt_limit", "memory_snapshot_chars",
        "memory_organizer_enabled", "memory_organizer_turn_interval",
        "permission_mode", "permissions_file", "command_sandbox",
        "command_sandbox_fail_closed", "command_sandbox_network",
        "sandbox_writable_roots", "sandbox_readonly_roots",
        "sandbox_env_passthrough",
        "hooks_enabled", "hooks_file",
        "plan_enabled", "skills_enabled",
        "skills_roots", "skills_prompt_chars", "mcp_enabled", "mcp_config_file",
        "delegation_enabled", "delegation_model_calls_limit",
        "delegation_wall_time_seconds", "delegation_background_max",
        "delegation_report_chars", "delegation_max_resumes",
    )
    return {
        key: (str(value) if isinstance(value := getattr(settings, key), Path) else value)
        for key in keys
    }


def _git_root(path: Path) -> Path | None:
    root = _git(path, "rev-parse", "--show-toplevel")
    return Path(root) if root else None


def _git(path: Path | None, *args: str) -> str | None:
    if path is None:
        return None
    try:
        completed = subprocess.run(
            ["git", *args], cwd=path, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _is_dirty(path: Path | None) -> bool | None:
    """干净仓库必须返回 False；不能把“没有改动”和“无法读取 git”都记成 None。"""
    if path is None:
        return None
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"], cwd=path,
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())
