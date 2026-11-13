"""只读工具：列文件、搜索、按行读取、日志回读。"""

import os
import shutil
import subprocess
import time
from pathlib import Path

from nextcoder.tools.base import ToolContext, ToolError, ToolResult, hash_content, is_ignored


def list_files(context: ToolContext, call_id: str, path: str = ".") -> ToolResult:
    started = time.monotonic()
    root = context.resolve(path)
    if not root.exists():
        return ToolResult(call_id, "list_files", "failed", error=f"路径不存在：{path}")
    entries: list[str] = []
    note = None
    if root.is_file():
        entries.append(context.rel(root))
    else:
        for item in sorted(root.rglob("*")):
            if is_ignored(item.relative_to(context.task_dir)):
                continue
            if item.is_file():
                entries.append(context.rel(item))
            if len(entries) >= context.settings.list_files_limit:
                note = f"…（达到 {context.settings.list_files_limit} 条上限，结果未列全）"
                break
    output = "\n".join(entries) or "（空目录）"
    return ToolResult(call_id, "list_files", "completed", output_excerpt=output,
                      note=note, duration_seconds=time.monotonic() - started)


def _search_with_rg(target: Path, pattern: str) -> tuple[int, list[str], str]:
    # -F：按字面量匹配，与工具描述一致；避免 pattern 中的正则元字符改变语义。
    completed = subprocess.run(
        ["rg", "-F", "--line-number", "--no-heading", "--", pattern, str(target)],
        capture_output=True, text=True, timeout=30,
    )
    if completed.returncode not in (0, 1):
        raise ToolError(f"rg 退出码 {completed.returncode}：{completed.stderr[:300]}")
    return completed.returncode, completed.stdout.splitlines(), completed.stdout


def _search_builtin(context: ToolContext, target: Path, pattern: str) -> tuple[int, list[str], str]:
    """没有 rg 时的纯 Python 扫描；语义与 rg -F 对齐，跳过大文件二进制和忽略目录。"""
    if target.is_file():
        candidates = [target]
    else:
        candidates = []
        for root, dirnames, filenames in os.walk(target):
            relative_root = Path(root).relative_to(context.task_dir)
            dirnames[:] = sorted(
                name for name in dirnames if not is_ignored(relative_root / name))
            for name in sorted(filenames):
                path = Path(root) / name
                if not is_ignored(path.relative_to(context.task_dir)):
                    candidates.append(path)
    matched: list[str] = []
    for path in candidates:
        try:
            with path.open("rb") as handle:
                if b"\0" in handle.read(8192):
                    continue
            with path.open(encoding="utf-8", errors="replace") as handle:
                for number, line in enumerate(handle, start=1):
                    if pattern in line:
                        matched.append(f"{path}:{number}:{line.rstrip(chr(10) + chr(13))}")
        except OSError:
            # 与 rg 一致：单个文件读不了不影响其余结果。
            continue
    text = "\n".join(matched)
    if text:
        text += "\n"
    return (0 if matched else 1), matched, text


def search(context: ToolContext, call_id: str, pattern: str, path: str = ".",
           offset: int = 0) -> ToolResult:
    started = time.monotonic()
    if not pattern:
        return ToolResult(call_id, "search", "failed", error="缺少搜索词 pattern")
    target = context.resolve(path)
    if not target.exists():
        return ToolResult(call_id, "search", "failed", error=f"路径不存在：{path}")
    try:
        if shutil.which("rg"):
            returncode, lines, full_output = _search_with_rg(target, pattern)
        else:
            returncode, lines, full_output = _search_builtin(context, target, pattern)
    except FileNotFoundError:
        # rg 在 which 之后被移除时仍可退回内置扫描。
        returncode, lines, full_output = _search_builtin(context, target, pattern)
    except ToolError as exc:
        return ToolResult(call_id, "search", "failed", error=str(exc))
    page = lines[offset : offset + 100]
    output = "\n".join(page)
    note = None
    if len(lines) > offset + 100:
        note = f"…（共 {len(lines)} 行，用 offset={offset + 100} 翻页）"
    artifact = context.write_log(f"{call_id}-search.txt", full_output)
    return ToolResult(call_id, "search", "completed", output_excerpt=output or "（无匹配）",
                      note=note, artifact_path=str(artifact), exit_code=returncode,
                      duration_seconds=time.monotonic() - started)


def read_lines(context: ToolContext, call_id: str, path: str, start: int = 1,
               end: int | None = None) -> ToolResult:
    started = time.monotonic()

    try:
        target = context.resolve(path)
    except ToolError as exc:
        return ToolResult(call_id, "read_lines", "failed", error=str(exc))
    if not target.is_file():
        return ToolResult(call_id, "read_lines", "failed", error=f"不是文件：{path}")
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if start < 1:
        start = 1
    selected = lines[start - 1 : end if end is not None else start - 1 + 200]
    rendered = "\n".join(f"{number}: {line}" for number, line in enumerate(selected, start=start))
    total = len(lines)
    note = None
    if end is None and start - 1 + 200 < total:
        note = f"…（共 {total} 行，继续读用 start={start + 200}）"
    elif end is not None:
        note = f"…（显示第 {start}-{min(end, total)} 行，共 {total} 行）"
    context.save_file_state(context.rel(target), text)
    return ToolResult(call_id, "read_lines", "completed",
                      output_excerpt=rendered or "（空文件）",
                      content_hash=hash_content(text),
                      note=note, duration_seconds=time.monotonic() - started)


def read_log(context: ToolContext, call_id: str, artifact_path: str, start_line: int = 1,
             end_line: int | None = None) -> ToolResult:
    """按行回读完整日志/工具产物（超长输出的回读入口）。"""
    started = time.monotonic()
    try:
        path = context.resolve_artifact(artifact_path)
    except ToolError as exc:
        return ToolResult(call_id, "read_log", "failed", error=str(exc))
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    end = end_line if end_line is not None else start_line + 200
    rendered = "\n".join(f"{number}: {line}"
                         for number, line in enumerate(lines[start_line - 1 : end],
                                                       start=start_line))
    note = f"…（共 {len(lines)} 行，显示第 {start_line}-{min(end, len(lines))} 行）"
    return ToolResult(call_id, "read_log", "completed",
                      output_excerpt=rendered or "（无内容）", note=note,
                      duration_seconds=time.monotonic() - started)
