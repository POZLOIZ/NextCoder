"""模型生成的任务摘要：持久化覆盖范围，并复用已保存摘要。

摘要只用于组装模型请求，不替代检查点和日志。保存成功后才切换视图；生成或
解析失败时由运行循环保留旧摘要/回退视图，不会静默丢弃历史。
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from nextcoder.llm import build_chat_llm
from nextcoder.settings import Settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS summaries (
    summary_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    covered_count INTEGER NOT NULL,
    summary TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model TEXT NOT NULL,
    usage_json TEXT NOT NULL DEFAULT '{}',
    parent_id TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summaries_session
    ON summaries(session_id, created_at);
"""

PROMPT_VERSION = "semantic-v1"


class SummaryError(RuntimeError):
    pass


@dataclass
class SummaryRecord:
    summary_id: str
    session_id: str
    covered_count: int
    summary: str
    prompt_version: str
    model: str
    usage: dict[str, int]
    parent_id: str | None
    created_at: float


@dataclass
class SummaryOutput:
    summary: str
    memory_suggestions: list[dict] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    raw: str = ""


class SummaryStore:
    """会话级摘要库；每次操作独立连接，避免长期持有 SQLite 句柄。"""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def latest(self, session_id: str) -> SummaryRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM summaries WHERE session_id=? "
                "ORDER BY created_at DESC LIMIT 1", (session_id,)
            ).fetchone()
        return _record(row) if row is not None else None

    def get(self, summary_id: str) -> SummaryRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM summaries WHERE summary_id=?", (summary_id,)
            ).fetchone()
        return _record(row) if row is not None else None

    def save(self, session_id: str, covered_count: int, summary: str, *,
             model: str, usage: dict[str, int] | None = None,
             parent_id: str | None = None,
             prompt_version: str = PROMPT_VERSION) -> SummaryRecord:
        if covered_count <= 0:
            raise SummaryError("摘要覆盖范围必须大于 0")
        if not summary.strip():
            raise SummaryError("摘要正文为空")
        record = SummaryRecord(
            summary_id=uuid.uuid4().hex[:16],
            session_id=session_id,
            covered_count=covered_count,
            summary=summary,
            prompt_version=prompt_version,
            model=model,
            usage=dict(usage or {}),
            parent_id=parent_id,
            created_at=time.time(),
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO summaries (summary_id, session_id, covered_count, summary,"
                " prompt_version, model, usage_json, parent_id, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record.summary_id, record.session_id, record.covered_count,
                 record.summary, record.prompt_version, record.model,
                 json.dumps(record.usage, ensure_ascii=False),
                 record.parent_id, record.created_at),
            )
        return record


class ModelSummarizer:
    """只做总结、不执行工具的独立模型调用。"""

    def __init__(self, settings: Settings, llm=None) -> None:
        self._settings = settings
        # 摘要调用与主循环走同一入口/会话头/备援策略，避免摘要和主任务看到
        # 不同模型行为却无法从报告里解释。
        self._llm = llm or build_chat_llm(settings)

    def summarize(self, old_summary: str, messages: list,
                  task: str = "") -> SummaryOutput:
        if not messages:
            raise SummaryError("没有需要总结的新过程")
        process = render_process(
            messages,
            max_chars=max(2000, self._settings.summary_input_token_budget * 2),
            message_chars=self._settings.summary_message_chars,
        )
        # 首条用户要求单独传入；摘要输入里没有它，不能指望模型自己恢复目标。
        prompt = _summary_prompt(
            old_summary, process,
            _clip(task, max(800, self._settings.summary_message_chars)))
        response = self._llm.invoke(prompt)
        text = _content_text(getattr(response, "content", ""))
        payload = _extract_json(text)
        summary = payload.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise SummaryError("摘要响应缺少 summary 字段")
        summary = _clip_summary(summary.strip(), self._settings.compaction_summary_chars)
        suggestions = payload.get("memory_suggestions")
        if not isinstance(suggestions, list):
            suggestions = []
        return SummaryOutput(
            summary=summary,
            memory_suggestions=[item for item in suggestions if isinstance(item, dict)][:3],
            usage=_usage(getattr(response, "usage_metadata", None)),
            raw=text,
        )


def render_process(messages: list, *, max_chars: int, message_chars: int) -> str:
    """把新增过程压成摘要输入：保留角色、工具调用、状态和关键结果。"""
    parts: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            content = _clip(_content_text(message.content), message_chars)
            parts.append(f"[工具结果 {message.tool_call_id}]\n{content}")
            continue
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            calls = []
            for call in message.tool_calls:
                calls.append(json.dumps({
                    "id": call.get("id"),
                    "name": call.get("name"),
                    "args": call.get("args"),
                }, ensure_ascii=False, default=str))
            text = ("\n".join(calls) + "\n" + _content_text(message.content)).strip()
            parts.append("[助手工具调用]\n" + _clip(text, message_chars))
            continue
        if isinstance(message, HumanMessage):
            parts.append("[用户]\n" + _clip(_content_text(message.content), message_chars))
            continue
        if isinstance(message, SystemMessage):
            continue
        role = getattr(message, "type", "message")
        parts.append(f"[{role}]\n" + _clip(_content_text(message.content), message_chars))
    text = "\n\n".join(parts)
    if len(text) <= max_chars:
        return text
    # 新增过程很长时保留开头（目标/早期错误）和结尾（最新状态）；
    # 中间被省略的部分仍可从检查点和日志回查，由摘要明确标注。
    head = max_chars * 2 // 5
    tail = max_chars - head
    return (text[:head] + "\n…[摘要输入中段省略]…\n" + text[-tail:])


def _summary_prompt(old_summary: str, process: str, task: str = "") -> str:
    return f"""你是编程任务上下文压缩器。只总结，不执行工具，不修改文件。

把“已有摘要”和“新增过程”合并成一份后续模型可以直接接着工作的任务摘要。
必须保留目标、约束、已修改文件、实际验证结果、关键决定、待办和当前进展。
严格区分：准备做、已经修改、实际检查通过、尚未确认。不要编造未出现的命令、
文件或测试结果。过时信息要删除，不重复整个操作日志；需要细节时保留文件路径、
命令、调用 ID 和日志路径，让后续模型可以回读。

本任务最初由用户提出（摘要的“目标”必须以此为依据；即使新增过程从工具调用
开始，也不能把目标写成“未说明”）：
{task or "（未提供）"}

已有摘要：
{old_summary or "（无）"}

新增过程：
{process}

只输出一个 JSON 对象，不要 Markdown 代码围栏。格式：
{{
  "summary": "Markdown 任务摘要，建议 800-1800 字；包含：目标、约束、已修改、已验证、关键决定、待办/进行中、未确认/风险、可回读位置",
  "memory_suggestions": [
    {{
      "topic": "稳定、简短的主题键，例如 test-prep 或 tests/run-mode",
      "kind": "user_preference | project_convention | test_command | lint_command | environment | known_failure",
      "body": "一条跨会话有价值的经验；没有就给空数组",
      "evidence": "来自哪次操作/哪条用户要求"
    }}
  ]
}}

memory_suggestions 只保存用户长期偏好、反复有用的项目约定、说明未写清的环境处理
办法；普通临时约束、一次性成功命令、整个任务摘要不要写入。允许零条。"""


def _extract_json(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise SummaryError("摘要响应不是 JSON 对象")
    try:
        value = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        raise SummaryError(f"摘要 JSON 解析失败：{exc}") from exc
    if not isinstance(value, dict):
        raise SummaryError("摘要 JSON 顶层必须是对象")
    return value


def _content_text(content) -> str:
    if isinstance(content, list):
        return json.dumps(content, ensure_ascii=False, default=str)
    return str(content or "")


def _usage(metadata) -> dict[str, int]:
    if not isinstance(metadata, dict):
        return {}
    return {key: value for key, value in metadata.items()
            if isinstance(value, int) and not isinstance(value, bool)}


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def _clip_summary(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:max(200, limit - 20)].rstrip() + "\n…[摘要正文截断]"


def _record(row: sqlite3.Row) -> SummaryRecord:
    return SummaryRecord(
        summary_id=row["summary_id"],
        session_id=row["session_id"],
        covered_count=row["covered_count"],
        summary=row["summary"],
        prompt_version=row["prompt_version"],
        model=row["model"],
        usage=_loads(row["usage_json"]),
        parent_id=row["parent_id"],
        created_at=row["created_at"],
    )


def _loads(text: str) -> dict:
    try:
        value = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
