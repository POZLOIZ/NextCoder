"""评测入口：按任务包约定准备每题环境、运行助手、执行独立验收。

环境准备与任务包 README 一致：/tmp 独立根、锁定依赖（--require-hashes）、
editable 安装、任务环境 bin 放 PATH 首位。验收在干净的另一副本上应用补丁后运行。
"""

import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from nextcoder.graders import GRADERS
from nextcoder.runtime.provenance import repository_provenance, runtime_provenance
from nextcoder.runtime.runner import run_task
from nextcoder.settings import Settings


class EvalError(RuntimeError):
    pass


def prepare_task_copy(package_dir: Path, repository_id: str, root: Path) -> Path:
    """解包固定提交的源码到独立根（避开工作区 .env 的向上搜索）。"""
    repos = json.loads((package_dir / "repositories.json").read_text(encoding="utf-8"))
    info = next(r for r in repos["repositories"] if r["repository_id"] == repository_id)
    archive = package_dir.parent / info["archive_path"] if not Path(
        info["archive_path"]).is_absolute() else Path(info["archive_path"])
    if not archive.is_file():
        archive = package_dir / "repositories" / Path(info["archive_path"]).name
    task_dir = root / info["archive_root"]
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")
    if not task_dir.is_dir():
        raise EvalError(f"解包后未找到 {task_dir}")
    return task_dir


def build_task_env(package_dir: Path, task_dir: Path, root: Path) -> Path:
    """按约定创建任务虚拟环境并 editable 安装任务副本。"""
    env_dir = root / "task-env"
    lock = package_dir / "requirements.lock"
    commands = [
        ["uv", "venv", str(env_dir), "--python", "/usr/bin/python3"],
        ["uv", "pip", "sync", "--python", str(env_dir / "bin" / "python"),
         str(lock), "--require-hashes"],
        ["uv", "pip", "install", "--python", str(env_dir / "bin" / "python"),
         "--no-build-isolation", "--no-deps", "-e", str(task_dir)],
    ]
    for command in commands:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=600,
                                   env={**os.environ, "UV_NO_PROGRESS": "1"})
        if completed.returncode != 0:
            raise EvalError(f"环境准备失败：{' '.join(command)}\n{completed.stderr[-1500:]}")
    return env_dir


def run_graded_task(settings: Settings, package_dir: Path, task: dict,
                    out_root: Path) -> dict:
    """准备环境 → 运行助手 → 在干净副本上验收补丁。"""
    task_id = task["task_id"]
    started = time.monotonic()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    result = {"task_id": task_id, "status": "failed", "checks": [], "error": None,
              "provenance": {
                  **runtime_provenance(package_dir, settings),
                  **repository_provenance(package_dir, task["repository_id"]),
              }}
    root = Path(tempfile.mkdtemp(prefix=f"nextcoder-{task_id}-", dir="/tmp"))
    try:
        task_dir = prepare_task_copy(package_dir, task["repository_id"], root)
        env_dir = build_task_env(package_dir, task_dir, root)

        budget = task.get("budget", {})
        run_settings = settings.model_copy(update={
            "model_calls_limit": budget.get("model_calls_limit",
                                            settings.model_calls_limit),
            "wall_time_seconds": budget.get("wall_time_seconds", settings.wall_time_seconds),
            "application_input_token_budget": budget.get(
                "application_input_token_budget", settings.application_input_token_budget),
            # 任务包约定：助手执行命令时必须先找到任务环境里的解释器和依赖。
            "command_path_prepend": str(env_dir / "bin"),
        })
        result["provenance"] = runtime_provenance(task_dir, run_settings)
        result["provenance"].update(
            repository_provenance(package_dir, task["repository_id"]))
        outcome = run_task(run_settings, task_dir, task["user_request"])
        patch = (outcome.run_dir / "final.patch").read_text(encoding="utf-8")
        # 保留运行产物（补丁、操作记录、完整日志），便于排查失败。
        keep_dir = out_root / stamp / "artifacts" / task_id
        shutil.copytree(outcome.run_dir, keep_dir, dirs_exist_ok=True)
        report_path = outcome.run_dir / "report.json"
        if report_path.exists():
            result["run_dir"] = str(keep_dir)

        if not patch.strip():
            result["error"] = "助手未产生改动"
            result["checks"].append(
                {"name": "patch_applies", "passed": False, "detail": "空补丁"})
            return result

        # 独立验收：干净副本 + 应用补丁；另留一份未修改基线供结构检查。
        baseline_root = root / "baseline"
        baseline_root.mkdir()
        baseline_dir = prepare_task_copy(package_dir, task["repository_id"], baseline_root)
        verify_root = root / "verify"
        verify_root.mkdir()
        verify_dir = prepare_task_copy(package_dir, task["repository_id"], verify_root)
        applied = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"],
                                 input=patch, cwd=verify_dir, capture_output=True, text=True)
        if applied.returncode != 0:
            result["error"] = f"补丁无法应用：{applied.stderr[:800]}"
            result["checks"].append({"name": "patch_applies", "passed": False})
            return result
        result["checks"].append({"name": "patch_applies", "passed": True})
        graders = GRADERS.get(task_id)
        if graders is None:
            result["status"] = "not_run"
            result["error"] = "该题验收程序待实现"
            return result
        env = {**os.environ, "PATH": f"{env_dir / 'bin'}:{os.environ['PATH']}"}
        for name, check in graders:
            passed, detail = check(verify_dir, env, baseline_dir)
            result["checks"].append({"name": name, "passed": passed, "detail": detail})
        result["status"] = "passed" if all(c["passed"] for c in result["checks"]) else "failed"
        return result
    except Exception as exc:  # noqa: BLE001 - 评测失败要记录而不是中断
        result["status"] = "blocked"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        result["wall_seconds"] = round(time.monotonic() - started, 1)
        _write_result(out_root, task_id, result, stamp)
        shutil.rmtree(root, ignore_errors=True)


def _write_result(out_root: Path, task_id: str, result: dict, stamp: str) -> None:
    run_dir = out_root / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")