"""运行来源测试：干净仓库与 git 不可用的区别、任务来源提交、密钥不落盘。"""

import json
import subprocess

from nextcoder.runtime.provenance import (
    _is_dirty,
    repository_provenance,
    runtime_provenance,
)
from nextcoder.settings import Settings


def _settings() -> Settings:
    return Settings(
        llm_base_url="http://localhost", llm_model="fake",
        llm_api_key="super-secret",
    )


def _git(repo, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _init_repo(path) -> str:
    path.mkdir()
    _git(path, "init")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    (path / "a.txt").write_text("x", encoding="utf-8")
    _git(path, "add", "a.txt")
    _git(path, "commit", "-m", "init")
    return _git(path, "rev-parse", "HEAD")


def test_is_dirty_distinguishes_clean_repo_and_git_failure(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    assert _is_dirty(repo) is False

    (repo / "a.txt").write_text("changed", encoding="utf-8")
    assert _is_dirty(repo) is True
    assert _is_dirty(tmp_path / "not-a-repo") is None


def test_runtime_provenance_records_task_commit_without_api_key(tmp_path):
    repo = tmp_path / "repo"
    commit = _init_repo(repo)
    provenance = runtime_provenance(repo, _settings())

    assert provenance["nextcoder_version"] == "0.2.0.dev0"
    assert provenance["nextcoder_commit"]
    assert provenance["task_commit"] == commit
    assert provenance["task_dirty"] is False
    assert "llm_api_key" not in provenance["settings"]
    assert provenance["settings"]["compaction_recent_token_ratio"] == 0.16
    assert provenance["settings"]["command_path_prepend"] == ""
    assert provenance["settings"]["command_container"] == ""
    assert provenance["settings"]["command_container_workdir"] == "/testbed"
    assert provenance["settings"]["summary_enabled"] is True
    assert provenance["settings"]["summary_input_token_budget"] == 16000
    assert provenance["settings"]["memory_snapshot_chars"] == 4000
    assert "memory_ttl_days" not in provenance["settings"]
    assert "super-secret" not in json.dumps(provenance, ensure_ascii=False)


def test_repository_provenance_reads_fixed_source_commit(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "repositories.json").write_text(json.dumps({
        "repositories": [{
            "repository_id": "demo",
            "commit": "abc123",
            "tag": "v1.0.0",
            "archive_sha256": "deadbeef",
        }],
    }), encoding="utf-8")

    provenance = repository_provenance(package, "demo")
    assert provenance == {
        "repository_id": "demo",
        "repository_commit": "abc123",
        "repository_tag": "v1.0.0",
        "archive_sha256": "deadbeef",
    }
    assert repository_provenance(package, "missing") == {}
    assert repository_provenance(tmp_path, "demo") == {}
