"""写入、局部补丁与改动查看：写前核对文件状态，diff 相对任务起点。"""

import difflib
import time

from nextcoder.tools.base import ToolContext, ToolError, ToolResult, hash_content


def write_file(context: ToolContext, call_id: str, path: str, content: str,
               expected_hash: str | None = None) -> ToolResult:
    """创建或覆盖文件。

    expected_hash 是上次 read_lines / apply_patch 返回的 content_hash；不一致说明
    文件已被外部改动，必须先重新读取。不传哈希时要求目标不存在（防止盲写覆盖）。
    """
    started = time.monotonic()

    try:
        target = context.resolve(path)
    except ToolError as exc:
        return ToolResult(call_id, "write_file", "failed", error=str(exc))
    relative = context.rel(target)
    current = target.read_text(encoding="utf-8") if target.exists() else None
    if current is not None:
        if expected_hash is None:
            return ToolResult(
                call_id, "write_file", "failed",
                error="文件已存在，必须先 read_lines 读取，并附上返回的 content_hash。",
            )
        if expected_hash != hash_content(current):
            return ToolResult(
                call_id, "write_file", "failed",
                error="文件内容与上次读取时不同（可能被外部修改），请重新读取后再写。",
            )
    elif expected_hash is not None and expected_hash != hash_content(""):
        return ToolResult(call_id, "write_file", "failed",
                          error="文件不存在，与给定的 expected_hash 不符。")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    context.save_file_state(relative, content)
    return ToolResult(call_id, "write_file", "completed",
                      output_excerpt=f"已写入 {relative}（{len(content)} 字符）",
                      changed_files=[relative], content_hash=hash_content(content),
                      duration_seconds=time.monotonic() - started)


def apply_patch(context: ToolContext, call_id: str, path: str, old_text: str,
                new_text: str, expected_hash: str | None = None,
                replace_all: bool = False) -> ToolResult:
    """把文件中的一段原文替换为新内容，不覆盖整文件。

    必须提供 read_lines 返回的 content_hash；old_text 必须在当前文件中恰好匹配一次
    （replace_all=True 时允许多次）。匹配不到或文件已变化时返回失败，要求重新读取。
    """
    started = time.monotonic()
    try:
        target = context.resolve(path)
    except ToolError as exc:
        return ToolResult(call_id, "apply_patch", "failed", error=str(exc))
    if not target.is_file():
        return ToolResult(call_id, "apply_patch", "failed", error=f"不是文件：{path}")
    if not old_text:
        return ToolResult(call_id, "apply_patch", "failed", error="old_text 不能为空")
    if old_text == new_text:
        # 这不是“补丁写错”，而是模型提交了空修改。明确告诉它文件不会变化，
        # 避免它把同一条空补丁反复重试几十次。
        return ToolResult(call_id, "apply_patch", "failed",
                          error="old_text 与 new_text 相同，文件不会被修改。"
                                "如果文件已经包含目标内容，请直接运行验证或继续下一步；"
                                "否则请提供不同的 new_text。")
    current = target.read_text(encoding="utf-8")
    if expected_hash is None:
        return ToolResult(call_id, "apply_patch", "failed",
                          error="必须先 read_lines 读取，并附上返回的 content_hash。")
    if expected_hash != hash_content(current):
        return ToolResult(
            call_id, "apply_patch", "failed",
            error="文件内容与上次读取时不同（可能被外部修改）："
                  f"当前 {len(current)} 字符、{current.count(chr(10)) + 1} 行。"
                  "请重新 read_lines 获取当前 content_hash 后再改。")
    occurrences = current.count(old_text)
    if occurrences == 0:
        return ToolResult(
            call_id, "apply_patch", "failed",
            error="old_text 在当前文件中没有匹配。"
                  f"{_no_match_hint(current, old_text)}。"
                  "请重新 read_lines 读取该区域后复制准确原文。")
    if occurrences > 1 and not replace_all:
        return ToolResult(call_id, "apply_patch", "failed",
                          error=f"old_text 匹配到 {occurrences} 处；请补充上下文使其唯一，"
                                "或设置 replace_all=true。")
    updated = current.replace(old_text, new_text, -1 if replace_all else 1)
    target.write_text(updated, encoding="utf-8")
    relative = context.rel(target)
    context.save_file_state(relative, updated)
    return ToolResult(call_id, "apply_patch", "completed",
                      output_excerpt=f"已修改 {relative}（{occurrences} 处匹配，"
                                     f"替换 {'全部' if replace_all else '1 处'}）",
                      changed_files=[relative], content_hash=hash_content(updated),
                      duration_seconds=time.monotonic() - started)


def _no_match_hint(current: str, old_text: str) -> str:
    """给 old_text 未匹配的情况一个可操作的线索，减少盲目重试。"""
    hints: list[str] = []
    stripped = old_text.strip()
    if stripped and stripped != old_text and current.count(stripped) == 1:
        hints.append("去掉 old_text 首尾空白后可以唯一匹配，可能是缩进或空行不一致")
    if "\r\n" in current and "\r\n" not in old_text:
        hints.append("文件使用 CRLF 换行，old_text 是 LF")
    elif "\r\n" in old_text and "\r\n" not in current:
        hints.append("old_text 使用 CRLF 换行，文件是 LF")
    current_lines = current.splitlines()
    old_lines = old_text.splitlines()
    if old_lines and old_lines[0].strip():
        first = old_lines[0].strip()
        close = difflib.get_close_matches(first, current_lines, n=1, cutoff=0.6)
        if close:
            line_no = current_lines.index(close[0]) + 1
            hints.append(
                f"old_text 第一行最接近第 {line_no} 行：{close[0].strip()[:120]}")
    if not hints:
        hints.append(f"当前文件共 {len(current_lines)} 行、{len(current)} 字符")
    return "；".join(hints)


def view_diff(context: ToolContext, call_id: str, path: str | None = None) -> ToolResult:
    """相对任务起点显示改动；不传 path 显示全部改动文件。"""
    started = time.monotonic()
    from nextcoder.tools.base import is_ignored

    changes: list[str] = []
    # 忽略项必须按“相对任务目录/基线目录”的路径判断：基线本身位于 .nextcoder/ 下，
    # 用绝对路径判断会把整个基线都当成忽略项。
    current_files = {item for item in context.task_dir.rglob("*")
                     if item.is_file()
                     and not is_ignored(item.relative_to(context.task_dir))}
    baseline_files = {item for item in context.baseline_dir.rglob("*")
                      if item.is_file()
                      and not is_ignored(item.relative_to(context.baseline_dir))}
    rel_current = {context.rel(p): p for p in current_files}
    rel_baseline = {str(p.relative_to(context.baseline_dir)): p for p in baseline_files}
    for relative in sorted(set(rel_current) | set(rel_baseline)):
        if path is not None and relative != path:
            continue
        new_text = rel_current[relative].read_text(encoding="utf-8", errors="replace") \
            if relative in rel_current else None
        old_text = rel_baseline[relative].read_text(encoding="utf-8", errors="replace") \
            if relative in rel_baseline else None
        if new_text == old_text:
            continue
        changes.append(relative)
    if not changes:
        return ToolResult(call_id, "view_diff", "completed", output_excerpt="（没有改动）",
                          duration_seconds=time.monotonic() - started)
    diffs: list[str] = []
    for relative in changes[: context.settings.max_diff_context_files]:
        old = (rel_baseline[relative].read_text(encoding="utf-8", errors="replace").splitlines()
               if relative in rel_baseline else [])
        new = (rel_current[relative].read_text(encoding="utf-8", errors="replace").splitlines()
               if relative in rel_current else [])
        diffs.extend(difflib.unified_diff(old, new, fromfile=f"a/{relative}",
                                          tofile=f"b/{relative}", lineterm=""))
    body = "\n".join(diffs)
    artifact = context.write_log(f"{call_id}-diff.patch", body)
    summary = "改动文件：\n" + "\n".join(changes)
    note = None
    if len(changes) > context.settings.max_diff_context_files:
        note = (f"…（仅显示前 {context.settings.max_diff_context_files} 个文件的 diff，"
                "完整列表见改动文件摘要）")
    return ToolResult(call_id, "view_diff", "completed",
                      output_excerpt=summary + "\n\n" + body[: context.settings.output_excerpt_chars],
                      artifact_path=str(artifact), changed_files=changes, note=note,
                      duration_seconds=time.monotonic() - started)
