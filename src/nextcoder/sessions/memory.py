"""项目记忆：保存跨会话仍有价值的事实，读取与写入分离。

写入只走三条显式路：Agent 记忆工具、语义压缩的少量候选、用户 CLI；后台
整理 agent 在会话边界合并重复、停用过时，并从近期会话历史提取新条目。
每次写入后系统自动重写可读投影：.nextcoder/memory.md 是索引（元记忆），
每条完整内容在 .nextcoder/memory/<条目>.md；会话入口注入索引，模型按需
自主读取详情文件，不需要查数据库。SQLite 仍是唯一事实来源，检查点不替代
记忆。条目不设有效期：停用只能由用户或 Agent 显式触发。
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from nextcoder.settings import Settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    key TEXT PRIMARY KEY,
    project TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL,
    value TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    invalidated_at REAL,
    invalid_reason TEXT,
    meta_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS memory_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_events_key ON memory_events(key);
CREATE INDEX IF NOT EXISTS idx_memory_events_at ON memory_events(at);
CREATE TABLE IF NOT EXISTS memory_consolidations (
    summary_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL DEFAULT '',
    covered_count INTEGER NOT NULL,
    suggestions_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL,
    error TEXT,
    created_at REAL NOT NULL,
    applied_at REAL
);
CREATE TABLE IF NOT EXISTS memory_sessions (
    session_id TEXT PRIMARY KEY,
    processed_turns INTEGER NOT NULL DEFAULT 0,
    total_turns INTEGER NOT NULL DEFAULT 0,
    processed_at REAL,
    note TEXT NOT NULL DEFAULT ''
);
"""

KIND_LABELS = {
    "user_preference": "用户偏好/更正",
    "project_convention": "项目约定",
    "environment": "环境处理",
    "test_command": "测试命令",
    "lint_command": "检查命令",
    "project_language": "项目语言",
    "known_failure": "已知问题",
}

_PRIORITY = {
    "user_preference": 0,
    "project_convention": 1,
    "environment": 2,
    "test_command": 3,
    "lint_command": 4,
    "project_language": 5,
    "known_failure": 6,
}


@dataclass
class Memory:
    key: str
    project: str
    kind: str
    value: str
    source: str
    confidence: float
    created_at: float
    updated_at: float
    invalidated_at: float | None
    invalid_reason: str | None
    meta: dict = field(default_factory=dict)

    @property
    def topic(self) -> str:
        return self.key

    @property
    def status(self) -> str:
        return "active" if self.invalidated_at is None else "disabled"


@dataclass
class MemorySnapshot:
    entries: list[Memory]
    text: str
    created_at: float

    @property
    def keys(self) -> list[str]:
        return [entry.key for entry in self.entries]


class ProjectMemory:
    """任务目录级记忆库；同一任务的所有会话共用一个 SQLite 文件。"""

    def __init__(self, settings: Settings, task_dir: Path, *,
                 clock: Callable[[], float] = time.time) -> None:
        self._settings = settings
        self._task_dir = task_dir.resolve()
        db_dir = self._task_dir / settings.runs_root
        db_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_dir / "memory.sqlite")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()
        self._clock = clock
        self.project = self._task_dir.name or _digest(str(self._task_dir))
        # 老项目可能只有数据库没有投影；打开时补一次（索引或详情目录缺一即补）。
        if not self.memory_file_path().exists() or not self.memory_dir().is_dir():
            self._sync_file()

    def _migrate(self) -> None:
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(memories)")}
        if "project" not in columns:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN project TEXT NOT NULL DEFAULT ''")
        if "meta_json" not in columns:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN meta_json TEXT NOT NULL DEFAULT '{}'")

    def close(self) -> None:
        self._conn.close()

    # ---- 写入：按主题更新，不按整句散列成冲突事实 ----
    def remember(self, topic: str, body: str, *, kind: str,
                 source: str, confidence: float = 0.7,
                 meta: dict | None = None) -> str:
        topic = _clean_topic(topic)
        body = " ".join(str(body or "").split())
        if not body:
            raise ValueError("记忆正文不能为空")
        if len(body) > 4000:
            body = body[:4000] + "…"
        return self.record(kind, body, source=source, confidence=confidence,
                           meta=meta, key=topic)

    def record(self, kind: str, value: str, *, source: str,
               confidence: float = 0.6, meta: dict | None = None,
               key: str | None = None) -> str:
        now = self._now()
        key = key or f"{kind}:{_digest(value)}"
        row = self._row(key)
        if row is not None:
            old_meta = _loads(row["meta_json"])
            manual = bool(old_meta.get("manual"))
            # 已停用条目只允许显式手工更正恢复；摘要整理、自动路径和 Agent
            # 常规写入都不能把用户停用的记忆重新注入上下文。
            if row["invalidated_at"] is not None and not source.startswith("manual"):
                return key
            if manual and source.startswith(("auto:", "summary:")):
                return key
        if source and row is not None and self._active(key) and self._observed(key, source):
            return key
        meta = dict(meta or {})
        if row is None:
            self._conn.execute(
                "INSERT INTO memories (key, project, kind, value, source, confidence,"
                " created_at, updated_at, meta_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (key, self.project, kind, value, source, confidence, now, now,
                 json.dumps(meta, ensure_ascii=False)),
            )
            action = "created"
        else:
            old_meta = _loads(row["meta_json"])
            old_meta.update(meta)
            same = row["value"] == value
            new_confidence = min(0.95, row["confidence"] + 0.1) if same else confidence
            action = "reactivated" if row["invalidated_at"] is not None else (
                "updated" if same else "replaced")
            self._conn.execute(
                "UPDATE memories SET project=?, kind=?, value=?, source=?, confidence=?,"
                " updated_at=?, invalidated_at=NULL, invalid_reason=NULL,"
                " meta_json=? WHERE key=?",
                (self.project, kind, value, source, new_confidence, now,
                 json.dumps(old_meta, ensure_ascii=False), key),
            )
        self._emit(key, action, f"{kind}={value}", source, now)
        self._conn.commit()
        self._sync_file()
        return key

    def invalidate(self, key: str, reason: str, *, source: str = "") -> bool:
        row = self._row(key)
        if row is None or row["invalidated_at"] is not None:
            return False
        if source and self._observed(key, source):
            return False
        now = self._now()
        self._conn.execute(
            "UPDATE memories SET invalidated_at=?, invalid_reason=? WHERE key=?",
            (now, reason, key),
        )
        self._emit(key, "invalidated", reason, source, now)
        self._conn.commit()
        self._sync_file()
        return True

    def set_manual(self, key: str, kind: str, value: str, *,
                   confidence: float = 1.0) -> None:
        """用户手工写入/更正一条记忆；之后摘要整理不再覆盖它。"""
        now = self._now()
        row = self._row(key)
        meta = _loads(row["meta_json"]) if row is not None else {}
        meta["manual"] = True
        if row is None:
            self._conn.execute(
                "INSERT INTO memories (key, project, kind, value, source, confidence,"
                " created_at, updated_at, meta_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (key, self.project, kind, value, "manual", confidence, now, now,
                 json.dumps(meta, ensure_ascii=False)),
            )
            action = "created"
        else:
            self._conn.execute(
                "UPDATE memories SET project=?, kind=?, value=?, source=?, confidence=?,"
                " updated_at=?, invalidated_at=NULL,"
                " invalid_reason=NULL, meta_json=? WHERE key=?",
                (self.project, kind, value, "manual", confidence, now,
                 json.dumps(meta, ensure_ascii=False), key),
            )
            action = "corrected"
        self._emit(key, action, f"{kind}={value}", "manual:set", now)
        self._conn.commit()
        self._sync_file()

    def disable(self, key: str, reason: str = "用户手动停用") -> bool:
        return self.invalidate(key, reason, source="manual:disable")

    # ---- 读取：会话边界注入整理后的全文，memory.md 随写入自动更新 ----
    def get(self, key: str) -> Memory | None:
        row = self._row(key)
        return self._to_memory(row) if row is not None else None

    def list_all(self) -> list[Memory]:
        rows = self._conn.execute(
            "SELECT * FROM memories "
            "ORDER BY (invalidated_at IS NOT NULL), updated_at DESC"
        ).fetchall()
        return [self._to_memory(row) for row in rows]

    def select(self, limit: int | None = None) -> list[Memory]:
        limit = self._settings.memory_prompt_limit if limit is None else limit
        if limit <= 0:
            return []
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE invalidated_at IS NULL "
            "ORDER BY confidence DESC, updated_at DESC"
        ).fetchall()
        memories = [self._to_memory(row) for row in rows]
        memories.sort(key=lambda item: (
            _PRIORITY.get(item.kind, 2), -item.confidence, -item.updated_at))
        return memories[:limit]

    def snapshot(self, limit: int | None = None,
                 max_chars: int | None = None) -> MemorySnapshot:
        entries = self.select(limit)
        text = render_memories(
            entries,
            max_chars=self._settings.memory_snapshot_chars if max_chars is None else max_chars,
        )
        return MemorySnapshot(entries=entries, text=text, created_at=self._now())

    def render(self, limit: int | None = None) -> str:
        return render_memories(self.select(limit))

    def memory_file_path(self) -> Path:
        """记忆索引（元记忆）；放在 .nextcoder 下避免污染任务补丁。"""
        return self._task_dir / self._settings.runs_root / "memory.md"

    def memory_dir(self) -> Path:
        """逐条详情文件目录 memory/<条目>.md；同样是写入时自动维护的投影。"""
        return self._task_dir / self._settings.runs_root / "memory"

    def events_since(self, timestamp: float) -> list[dict]:
        rows = self._conn.execute(
            "SELECT key, action, detail, source, at FROM memory_events "
            "WHERE at > ? ORDER BY id", (timestamp,)
        ).fetchall()
        return [dict(row) for row in rows]

    # ---- 会话提取进度：每条会话处理过多少完整轮次 ----
    def session_processed_turns(self, session_id: str) -> int:
        row = self._conn.execute(
            "SELECT processed_turns FROM memory_sessions WHERE session_id=?",
            (session_id,)).fetchone()
        return int(row["processed_turns"]) if row is not None else 0

    def mark_session_processed(self, session_id: str, processed_turns: int,
                               total_turns: int, note: str = "") -> None:
        """记录会话里已审阅到第几个完整轮次；只前进不后退。"""
        self._conn.execute(
            "INSERT INTO memory_sessions"
            " (session_id, processed_turns, total_turns, processed_at, note)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(session_id) DO UPDATE SET"
            " processed_turns=MAX(processed_turns, excluded.processed_turns),"
            " total_turns=excluded.total_turns,"
            " processed_at=excluded.processed_at, note=excluded.note",
            (session_id, processed_turns, total_turns, self._now(), note[:500]),
        )
        self._conn.commit()

    # ---- 语义压缩时的集中整理 ----
    def consolidate(self, *, summary_id: str, covered_count: int,
                    suggestions: list[dict], session_id: str = "") -> dict:
        """保存独立建议并应用；同一摘要只处理一次，允许零条。"""
        existing = self._conn.execute(
            "SELECT * FROM memory_consolidations WHERE summary_id=?", (summary_id,)
        ).fetchone()
        if existing is not None:
            return dict(existing)
        clean = _clean_suggestions(suggestions)
        now = self._now()
        self._conn.execute(
            "INSERT INTO memory_consolidations"
            " (summary_id, session_id, covered_count, suggestions_json, status,"
            " created_at) VALUES (?, ?, ?, ?, 'processing', ?)",
            (summary_id, session_id, covered_count,
             json.dumps(clean, ensure_ascii=False), now),
        )
        self._conn.commit()
        applied: list[str] = []
        try:
            for item in clean:
                key = self.remember(
                    item["topic"], item["body"], kind=item["kind"],
                    source=f"summary:{summary_id}", confidence=0.7,
                    meta={"evidence": item.get("evidence", ""),
                          "summary_id": summary_id,
                          "covered_count": covered_count},
                )
                applied.append(key)
        except Exception as exc:  # noqa: BLE001
            self._conn.execute(
                "UPDATE memory_consolidations SET status='failed', error=?, applied_at=?"
                " WHERE summary_id=?",
                (f"{type(exc).__name__}: {exc}"[:1000], self._now(), summary_id),
            )
            self._conn.commit()
            raise
        status = "applied" if applied else "empty"
        self._conn.execute(
            "UPDATE memory_consolidations SET status=?, applied_at=?, error=NULL"
            " WHERE summary_id=?",
            (status, self._now(), summary_id),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT * FROM memory_consolidations WHERE summary_id=?", (summary_id,)
        ).fetchone()
        return dict(row) if row is not None else {}

    # ---- 内部 ----
    def _now(self) -> float:
        return self._clock()

    def _sync_file(self) -> None:
        """重写索引与逐条详情文件；数据库仍是事实来源，文件是投影视图。"""
        entries = self.select(limit=100_000)
        detail_dir = self.memory_dir()
        detail_dir.mkdir(parents=True, exist_ok=True)
        keep = set()
        for entry in entries:
            name = _slug(entry.key)
            keep.add(name)
            (detail_dir / f"{name}.md").write_text(_detail_text(entry),
                                                   encoding="utf-8")
        for path in detail_dir.glob("*.md"):
            if path.stem not in keep:
                path.unlink()
        text = render_memories(entries)
        header = ("# 项目记忆（索引）\n\n"
                  "由系统在每次记忆写入后自动重写；每条的完整内容在 "
                  "memory/<条目>.md，模型可用 read_lines 或 memory_read 读取。"
                  "手工维护请用 `nextcoder memory set/disable`，直接编辑本目录"
                  "文件会在下次写入时被覆盖。\n\n")
        body = text if text else "（暂无记忆）\n"
        self.memory_file_path().write_text(header + body, encoding="utf-8")

    def _row(self, key: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM memories WHERE key=?", (key,)).fetchone()

    def _active(self, key: str) -> bool:
        row = self._row(key)
        return row is not None and row["invalidated_at"] is None

    def _observed(self, key: str, source: str) -> bool:
        if not source:
            return False
        return self._conn.execute(
            "SELECT 1 FROM memory_events WHERE key=? AND source=? LIMIT 1",
            (key, source),
        ).fetchone() is not None

    def _emit(self, key: str, action: str, detail: str, source: str,
              now: float | None = None) -> None:
        self._conn.execute(
            "INSERT INTO memory_events (key, action, detail, source, at) "
            "VALUES (?, ?, ?, ?, ?)",
            (key, action, detail[:500], source, self._now() if now is None else now),
        )

    def _to_memory(self, row: sqlite3.Row) -> Memory:
        return Memory(
            key=row["key"], project=row["project"], kind=row["kind"],
            value=row["value"], source=row["source"], confidence=row["confidence"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            invalidated_at=row["invalidated_at"],
            invalid_reason=row["invalid_reason"], meta=_loads(row["meta_json"]),
        )


def render_memories(memories: list[Memory], max_chars: int | None = None) -> str:
    """渲染记忆目录：像项目笔记一样分节，每条一句摘要加正文链接。

    同一函数用于会话注入与 memory.md 落盘，保证两处内容一致。正文仍在
    .nextcoder/memory/ 下，模型需要细节时自己打开。
    """
    if not memories:
        return ""
    title = "# 项目记忆"
    intro = ("本项目的长期约定和事实。每条一句摘要，正文见链接，需要细节时用 "
             "read_lines 打开；当前用户要求和文件事实优先。")
    lines = [title, "", intro]
    used = len(title) + len(intro)
    omitted = 0
    current_kind: str | None = None
    for index, memory in enumerate(memories):
        label = KIND_LABELS.get(memory.kind, memory.kind)
        value = " ".join(memory.value.split())
        brief = value[:80] + ("…" if len(value) > 80 else "")
        key = memory.key.replace("[", "\\[").replace("]", "\\]")
        link = f".nextcoder/memory/{_slug(memory.key)}.md"
        line = f"- [{key}]({link})：{brief}"
        header = f"## {label}"
        cost = len(line) + (len(header) + 2 if memory.kind != current_kind else 0)
        if max_chars is not None and used + cost > max_chars:
            omitted = len(memories) - index
            break
        if memory.kind != current_kind:
            lines.extend(["", header])
            used += len(header) + 2
            current_kind = memory.kind
        lines.append(line)
        used += len(line)
    if omitted:
        lines.append(f"\n还有 {omitted} 条没有展开，用 memory_list 查看。")
    return "\n".join(lines)


def _slug(key: str) -> str:
    """详情文件名：保留可用字符，拼短哈希保证同名人不同文件。"""
    text = re.sub(r"[^\w.-]+", "-", key, flags=re.UNICODE).strip("-.")
    return f"{text[:50]}-{_digest(key)[:8]}"


def _detail_text(entry: Memory) -> str:
    label = KIND_LABELS.get(entry.kind, entry.kind)
    lines = [
        f"# {entry.key}",
        "",
        f"- 类型：{label}（{entry.kind}）",
        f"- 置信度：{entry.confidence:.2f}",
        f"- 来源：{entry.source}",
        f"- 更新：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry.updated_at))}",
    ]
    if entry.meta.get("evidence"):
        lines.append(f"- 证据：{entry.meta['evidence']}")
    if entry.meta.get("manual"):
        lines.append("- 手工条目：整理 agent 不可合并/停用")
    lines.extend(["", "## 内容", "", entry.value, ""])
    return "\n".join(lines)


def _clean_topic(topic: str) -> str:
    text = " ".join(str(topic or "").split())
    if not text:
        raise ValueError("记忆主题不能为空")
    if len(text) > 200:
        text = text[:200]
    return text


def _clean_suggestions(suggestions: list[dict]) -> list[dict]:
    clean: list[dict] = []
    for item in (suggestions or [])[:3]:
        if not isinstance(item, dict):
            continue
        topic = " ".join(str(item.get("topic") or "").split())[:200]
        body = " ".join(str(item.get("body") or "").split())[:2000]
        if not topic or not body:
            continue
        kind = str(item.get("kind") or "project_convention")
        if kind not in KIND_LABELS:
            kind = "project_convention"
        clean.append({
            "topic": topic,
            "kind": kind,
            "body": body,
            "evidence": " ".join(str(item.get("evidence") or "").split())[:500],
        })
    return clean


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _loads(text: str) -> dict:
    try:
        value = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
