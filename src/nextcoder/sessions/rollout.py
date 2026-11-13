"""会话事件日志：每个会话一份人类可读的 JSONL rollout 文件。

与 checkpoints.sqlite 的分工：SQLite 是运行时状态，负责中断恢复；
JSONL 只追加事件、只用于查看与调试，不参与恢复，避免双份状态冲突。
一行一个 JSON 对象：{"ts", "session_id", "turn_id", "type", "payload"}。
写入失败静默吞掉——日志缺几行不能影响主循环。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import IO, Any

EVENT_TYPES = {
    "session_start", "user_request", "assistant",
    "tool_result", "turn_end", "compact", "plan_updated",
    "permission_requested", "permission_answered",
}


class RolloutWriter:
    """追加式事件写入器；同一路径多次打开时续写不覆盖。"""

    def __init__(self, path: Path, session_id: str = "") -> None:
        self.path = Path(path)
        self.session_id = session_id
        self._fh: IO[str] | None = None

    def _handle(self) -> IO[str] | None:
        if self._fh is None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self.path.open("a", encoding="utf-8")
            except OSError:
                return None
        return self._fh

    def append(self, event_type: str, payload: dict[str, Any],
               turn_id: str = "") -> None:
        fh = self._handle()
        if fh is None:
            return
        event = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "session_id": self.session_id,
            "turn_id": turn_id,
            "type": event_type,
            "payload": _json_safe(payload),
        }
        try:
            fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            fh.flush()
        except OSError:
            pass  # 日志写不进去不阻塞主循环

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None


def _json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in payload.items():
        try:
            json.dumps(value)
            clean[key] = value
        except (TypeError, ValueError):
            clean[key] = str(value)
    return clean


def read_events(path: Path) -> list[dict[str, Any]]:
    """读回全部事件；坏行跳过，供调试与测试使用。"""
    events: list[dict[str, Any]] = []
    if not Path(path).exists():
        return events
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events
