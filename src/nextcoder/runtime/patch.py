"""任务副本、基线快照与补丁导出。"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from nextcoder.tools.base import is_ignored


def copy_tree_filtered(source: Path, target: Path) -> None:
    """复制目录内容，跳过 .git、虚拟环境、运行目录等忽略项。

    用 os.walk 而不是 rglob：rglob 会先进入 .git/.venv/node_modules 再逐个
    判断忽略，大型仓库光遍历就要几十秒；这里直接剪掉这些目录，复制不再
    受仓库历史体积影响。
    """
    target.mkdir(parents=True, exist_ok=True)
    for root, directories, files in os.walk(source, followlinks=False):
        root_path = Path(root)
        kept: list[str] = []
        for name in directories:
            item = root_path / name
            relative = item.relative_to(source)
            if is_ignored(relative):
                continue
            if item.is_symlink():
                # 链接到目录时保留链接本身，不跟进去复制第二份文件。
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(item.readlink())
                continue
            kept.append(name)
        directories[:] = kept

        for name in files:
            item = root_path / name
            relative = item.relative_to(source)
            if is_ignored(relative):
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if item.is_symlink():
                destination.symlink_to(item.readlink())
            else:
                shutil.copy2(item, destination)


def copy_baseline(task_dir: Path, baseline_dir: Path) -> None:
    copy_tree_filtered(task_dir, baseline_dir)


def git_diff_patch(task_dir: Path, baseline_dir: Path) -> str:
    """生成可直接 git apply 的补丁；没有改动时返回空字符串。

    在临时目录里用 baseline/ 与 current/ 两个相对目录做 --no-index diff，
    输出前缀天然是 a/ 与 b/，不再对绝对路径做字符串改写。
    """
    with tempfile.TemporaryDirectory(prefix="nextcoder-diff-") as tmp:
        root = Path(tmp)
        baseline = root / "baseline"
        current = root / "current"
        copy_tree_filtered(baseline_dir, baseline)
        copy_tree_filtered(task_dir, current)
        completed = subprocess.run(
            ["git", "-c", "core.quotepath=false", "diff", "--no-index", "--binary",
             "--", "baseline", "current"],
            cwd=root, capture_output=True,
        )
        if completed.returncode == 0:
            return ""
        if completed.returncode != 1:
            stderr = completed.stderr.decode("utf-8", "replace")
            raise RuntimeError(f"git diff 失败：{stderr[:500]}")
        try:
            patch = completed.stdout.decode("utf-8")
        except UnicodeDecodeError:
            # 仓库里可能有非 UTF-8 的文本/测试数据（matplotlib 的测试产物）。
            # 直接按 UTF-8 解码会中断整次运行，按 replace 解码又会破坏补丁。
            # 重试时把所有文件当二进制，git 会用 base85 编码内容，输出纯
            # ASCII 且可被 git apply 还原；路径也改回 quoted 形式。
            attributes = root / "attributes"
            attributes.write_text("* -diff\n", encoding="utf-8")
            completed = subprocess.run(
                ["git", "-c", "core.quotepath=true",
                 "-c", f"core.attributesFile={attributes}",
                 "diff", "--no-index", "--binary", "--", "baseline", "current"],
                cwd=root, capture_output=True,
            )
            if completed.returncode not in (0, 1):
                stderr = completed.stderr.decode("utf-8", "replace")
                raise RuntimeError(f"git diff 失败：{stderr[:500]}")
            patch = completed.stdout.decode("ascii", "replace")
    # 新增/删除文件两侧都可能出现 current/baseline 前缀，统一改成仓库相对路径。
    for prefix in ("a/baseline/", "a/current/", "b/baseline/", "b/current/"):
        patch = patch.replace(prefix, prefix[:2])
    return patch
