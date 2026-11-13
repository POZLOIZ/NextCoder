"""操作状态外置：SQLite 操作表 + 完整日志文件。

每次工具调用使用模型的 tool call id 作为 call_id；状态机
prepared → started → completed / failed / cancelled / timeout / unknown。
记录关联文件、命令、退出码、日志路径与返回给模型的结果文本；同一 call_id
再次出现时直接复用结果，不重复执行副作用。
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS operations (
    call_id TEXT PRIMARY KEY,
    turn_id TEXT,
    tool TEXT NOT NULL,
    args_json TEXT NOT NULL,
    status TEXT NOT NULL,
    related_files_json TEXT NOT NULL DEFAULT '[]',
    exit_code INTEGER,
    log_path TEXT,
    error TEXT,
    result_text TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    task_dir TEXT NOT NULL,
    request TEXT NOT NULL,
    status TEXT NOT NULL,
    stop_reason TEXT,
    model_calls INTEGER NOT NULL DEFAULT 0,
    usage_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

VALID_STATUSES = {"prepared", "started", "completed", "failed", "cancelled", "timeout", "unknown"}

# 已进入终态的操作不再重放；started 表示副作用可能已经发生、结果未知。
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "timeout", "unknown"}
ALLOWED_TRANSITIONS = {
    # 权限拒绝发生在副作用之前：prepared 可以直接 failed，不必先 started。
    "prepared": {"started", "failed", "cancelled"},
    "started": {"completed", "failed", "cancelled", "timeout", "unknown"},
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Operation:
    call_id: str
    turn_id: str | None
    tool: str
    args: dict
    status: str
    related_files: list[str]
    exit_code: int | None
    log_path: str | None
    error: str | None
    result_text: str | None


class OperationStore:
    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(operations)")}
        if "turn_id" not in columns:
            self._conn.execute("ALTER TABLE operations ADD COLUMN turn_id TEXT")
        if "result_text" not in columns:
            self._conn.execute("ALTER TABLE operations ADD COLUMN result_text TEXT")

    def close(self) -> None:
        self._conn.close()

    # ---- 操作状态机 ----

    def prepare(self, tool: str, args: dict, call_id: str | None = None,
                turn_id: str | None = None) -> str:
        call_id = call_id or uuid.uuid4().hex[:12]
        now = _now()
        self._conn.execute(
            "INSERT OR IGNORE INTO operations"
            " (call_id, turn_id, tool, args_json, status, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, 'prepared', ?, ?)",
            (call_id, turn_id, tool, json.dumps(args, ensure_ascii=False), now, now),
        )
        self._conn.commit()
        return call_id

    def update_args(self, call_id: str, args: dict) -> None:
        """before hook 替换参数后同步审计记录，保证记录的是实际执行内容。"""
        current = self.get(call_id)
        if current is None:
            raise KeyError(f"操作不存在：{call_id}")
        if current.status != "prepared":
            raise ValueError(f"操作已开始，不能改写参数：{current.status}")
        self._conn.execute(
            "UPDATE operations SET args_json = ?, updated_at = ? WHERE call_id = ?",
            (json.dumps(args, ensure_ascii=False), _now(), call_id),
        )
        self._conn.commit()

    def transition(self, call_id: str, status: str, *, related_files: list[str] | None = None,
                   exit_code: int | None = None, log_path: str | None = None,
                   error: str | None = None, result_text: str | None = None) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"非法操作状态：{status}")
        current = self.get(call_id)
        if current is None:
            raise KeyError(f"操作不存在：{call_id}")
        allowed = ALLOWED_TRANSITIONS.get(current.status)
        if allowed is not None and status not in allowed:
            raise ValueError(f"非法状态迁移：{current.status} → {status}")
        if current.status in TERMINAL_STATUSES and status != current.status:
            raise ValueError(f"操作已结束，不能再次迁移：{current.status} → {status}")
        self._conn.execute(
            "UPDATE operations SET status = ?,"
            " related_files_json = COALESCE(?, related_files_json),"
            " exit_code = COALESCE(?, exit_code), log_path = COALESCE(?, log_path),"
            " error = ?, result_text = COALESCE(?, result_text), updated_at = ?"
            " WHERE call_id = ?",
            (
                status,
                json.dumps(related_files, ensure_ascii=False) if related_files is not None else None,
                exit_code,
                log_path,
                error,
                result_text,
                _now(),
                call_id,
            ),
        )
        self._conn.commit()

    def get(self, call_id: str) -> Operation | None:
        row = self._conn.execute(
            "SELECT * FROM operations WHERE call_id = ?", (call_id,)
        ).fetchone()
        return self._row(row)

    def all_operations(self, turn_id: str | None = None) -> list[Operation]:
        if turn_id is None:
            rows = self._conn.execute(
                "SELECT * FROM operations ORDER BY created_at, call_id"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM operations WHERE turn_id = ? ORDER BY created_at, call_id",
                (turn_id,),
            ).fetchall()
        return [self._row(row) for row in rows]

    @staticmethod
    def _row(row) -> Operation | None:
        if row is None:
            return None
        return Operation(
            call_id=row["call_id"],
            turn_id=row["turn_id"] if "turn_id" in row.keys() else None,
            tool=row["tool"],
            args=json.loads(row["args_json"]),
            status=row["status"],
            related_files=json.loads(row["related_files_json"]),
            exit_code=row["exit_code"],
            log_path=row["log_path"],
            error=row["error"],
            result_text=row["result_text"] if "result_text" in row.keys() else None,
        )

    # ---- 运行记录 ----

    def start_run(self, run_id: str, task_dir: str, request: str) -> None:
        now = _now()
        self._conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, task_dir, request, status, created_at,"
            " updated_at) VALUES (?, ?, ?, 'running', ?, ?)",
            (run_id, task_dir, request, now, now),
        )
        self._conn.commit()

    def count_runs(self) -> int:
        """本会话已开始的轮数；用于周期性后台整理的触发判定。"""
        row = self._conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        return int(row[0])

    def finish_run(self, run_id: str, status: str, stop_reason: str | None,
                   model_calls: int, usage: dict) -> None:
        self._conn.execute(
            "UPDATE runs SET status = ?, stop_reason = ?, model_calls = ?,"
            " usage_json = ?, updated_at = ? WHERE run_id = ?",
            (status, stop_reason, model_calls, json.dumps(usage, ensure_ascii=False),
             _now(), run_id),
        )
        self._conn.commit()
