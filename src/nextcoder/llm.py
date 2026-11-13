"""模型入口：OpenCode Go 会话头、thinking 配置与可重试错误备援。

OpenCode Go 要求每个请求带稳定的 x-opencode-session；没有该头会直接 400。
DeepSeek 官方在带 tools 的请求里会返回 reasoning_content；LangChain 的 OpenAI
适配器默认丢弃该字段，因此这里用 _ReasoningChatOpenAI 在响应中补存、在下一
次请求里回传。thinking 仍按入口配置：DeepSeek 官方可设为 enabled，其他不兼容
的入口保持 disabled/omit。
默认主入口是 DeepSeek 官方；OpenCode Go 在限流、超时或连接失败时备援
（它自身 429 较频繁，不适合作为主入口）。
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from pydantic import PrivateAttr

from nextcoder.settings import Settings


class _ReasoningChatOpenAI(ChatOpenAI):
    """保留 DeepSeek 的 reasoning_content，供下一次带 tools 的请求回传。

    langchain-openai 只保留 content/tool_calls，assistant 的 reasoning_content
    会丢失；DeepSeek 在带 tools 的请求里要求把历史推理一并传回，否则模型看不到
    自己上一轮的推理。这里在响应解析后补存，并在构造请求时写回消息字典。
    """

    _preserve_reasoning: bool = PrivateAttr(default=False)

    def __init__(self, **kwargs: Any) -> None:
        preserve = bool(kwargs.pop("preserve_reasoning", False))
        super().__init__(**kwargs)
        self._preserve_reasoning = preserve

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        if not self._preserve_reasoning:
            return result
        response_dict = (
            response if isinstance(response, dict) else response.model_dump()
        )
        choices = response_dict.get("choices") or []
        for generation, choice in zip(result.generations, choices):
            message = choice.get("message") or {}
            reasoning = message.get("reasoning_content")
            if reasoning:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return result

    def _convert_chunk_to_generation_chunk(
            self, chunk, default_chunk_class, base_generation_info):
        # 流式响应里的 reasoning_content 不在标准 delta 字段里，需要逐个 chunk
        # 补存；AIMessageChunk 相加时会自动拼接该字符串。
        generation = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info)
        if not self._preserve_reasoning or generation is None:
            return generation
        choices = (chunk.get("choices") or
                   (chunk.get("chunk") or {}).get("choices") or [])
        if not choices:
            return generation
        delta = choices[0].get("delta") or {}
        reasoning = delta.get("reasoning_content")
        if reasoning:
            message = generation.message
            current = message.additional_kwargs.get("reasoning_content", "")
            message.additional_kwargs["reasoning_content"] = current + reasoning
        return generation

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if not self._preserve_reasoning:
            return payload
        try:
            sources = self._convert_input(input_).to_messages()
        except Exception:  # noqa: BLE001 - 无法还原消息时按原样发送
            return payload
        targets = payload.get("messages") or []
        if len(sources) != len(targets):
            return payload
        for source, target in zip(sources, targets):
            if not isinstance(source, AIMessage) or target.get("role") != "assistant":
                continue
            reasoning = source.additional_kwargs.get("reasoning_content")
            if reasoning is not None:
                target["reasoning_content"] = reasoning
            else:
                # 历史里可能有本次升级前产生的 assistant 消息；空串不会改变
                # 语义，但能满足接口对 reasoning_content 字段的要求。
                target.setdefault("reasoning_content", "")
        return payload


# 首个 chunk 之前可安全切换备援的错误：限流、连接失败、超时、服务端 5xx。
# 已经输出内容后再切换会重复显示，因此流式路径只在 emitted=False 时切换。
_RETRYABLE_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError,
                     InternalServerError)


def _chat_model(settings: Settings, *, base_url: str, model: str, api_key: str,
                session_id: str = "") -> ChatOpenAI:
    headers: dict[str, str] = dict(settings.llm_extra_headers)
    if settings.llm_session_header:
        value = settings.llm_session or "nextcoder"
        if session_id:
            value = f"{value}-{session_id}"
        headers[settings.llm_session_header] = value
    extra_body: dict[str, Any] | None = None
    if settings.llm_thinking in {"enabled", "disabled"}:
        extra_body = {"thinking": {"type": settings.llm_thinking}}
    return _ReasoningChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=0,
        max_tokens=settings.output_reserve_tokens,
        max_retries=1,
        # 连接超时短，读取超时按流式 chunk 间隔计算；长思考不会被总时长误杀。
        timeout=httpx.Timeout(
            connect=settings.llm_connect_timeout_seconds,
            read=settings.llm_read_timeout_seconds,
            write=30.0,
            pool=10.0,
        ),
        default_headers=headers or None,
        extra_body=extra_body,
        reasoning_effort=settings.llm_reasoning_effort or None,
        preserve_reasoning=settings.llm_thinking == "enabled",
        stream_usage=True,
    )


class _FallbackState:
    def __init__(self) -> None:
        self.last = "primary"

    def as_dict(self) -> dict[str, str]:
        return {"last": self.last}


def _stream_with_fallback(primary, backup, state: _FallbackState,
                          messages: list[BaseMessage], **kwargs: Any):
    """流式主备切换：只有首个 chunk 之前发生可重试错误才切备援。

    可重试错误包括限流、连接失败、超时和服务端 5xx。已经输出过 delta 再
    切换会重放前半段内容，用户会看到重复文本，因此中途失败直接抛出，由
    运行循环落盘失败原因。
    """
    emitted = False
    try:
        for chunk in primary.stream(messages, **kwargs):
            emitted = True
            state.last = "primary"
            yield chunk
        return
    except _RETRYABLE_ERRORS:
        if backup is None or emitted:
            raise
    state.last = "backup"
    yield from backup.stream(messages, **kwargs)


class BoundFallback:
    """绑定工具后的主备入口；只对首个 chunk 之前的可重试错误切换。"""

    def __init__(self, primary, backup, state: _FallbackState) -> None:
        self._primary = primary
        self._backup = backup
        self._state = state

    @property
    def last_endpoint(self) -> str:
        return self._state.last

    def invoke(self, messages: list[BaseMessage], **kwargs: Any):
        try:
            result = self._primary.invoke(messages, **kwargs)
            self._state.last = "primary"
            return result
        except _RETRYABLE_ERRORS:
            if self._backup is None:
                raise
            result = self._backup.invoke(messages, **kwargs)
            self._state.last = "backup"
            return result

    def stream(self, messages: list[BaseMessage], **kwargs: Any):
        return _stream_with_fallback(
            self._primary, self._backup, self._state, messages, **kwargs)


class FallbackChat:
    """未绑定工具的入口；bind_tools 后返回 BoundFallback。"""

    def __init__(self, primary, backup=None) -> None:
        self._primary = primary
        self._backup = backup
        self._state = _FallbackState()

    @property
    def last_endpoint(self) -> str:
        return self._state.last

    def bind_tools(self, definitions: list):
        primary = self._primary.bind_tools(definitions)
        backup = self._backup.bind_tools(definitions) if self._backup is not None else None
        return BoundFallback(primary, backup, self._state)

    def invoke(self, messages: list[BaseMessage], **kwargs: Any):
        try:
            result = self._primary.invoke(messages, **kwargs)
            self._state.last = "primary"
            return result
        except _RETRYABLE_ERRORS:
            if self._backup is None:
                raise
            result = self._backup.invoke(messages, **kwargs)
            self._state.last = "backup"
            return result

    def stream(self, messages: list[BaseMessage], **kwargs: Any):
        return _stream_with_fallback(
            self._primary, self._backup, self._state, messages, **kwargs)


def build_chat_llm(settings: Settings, *, session_id: str = "") -> FallbackChat:
    """主入口必填；备援三元组齐全时才启用可重试错误切换。"""
    if not (settings.llm_base_url and settings.llm_model):
        raise ValueError("NEXTCODER_LLM_BASE_URL 和 NEXTCODER_LLM_MODEL 必须配置")
    primary = _chat_model(
        settings,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key.get_secret_value(),
        session_id=session_id,
    )
    backup = None
    if (settings.llm_fallback_base_url and settings.llm_fallback_model
            and settings.llm_fallback_api_key.get_secret_value()):
        backup = _chat_model(
            settings,
            base_url=settings.llm_fallback_base_url,
            model=settings.llm_fallback_model,
            api_key=settings.llm_fallback_api_key.get_secret_value(),
            session_id=session_id,
        )
    return FallbackChat(primary, backup)


def endpoint_snapshot(settings: Settings) -> dict[str, str]:
    """血缘记录：只写入口和模型名，不写密钥。"""
    return {
        "provider": settings.llm_provider,
        "profile": settings.model_profile,
        "base_url": settings.llm_base_url,
        "model": settings.llm_model,
        "session_header": settings.llm_session_header,
        "thinking": settings.llm_thinking,
        "reasoning_effort": settings.llm_reasoning_effort,
        "fallback_base_url": settings.llm_fallback_base_url,
        "fallback_model": settings.llm_fallback_model,
    }


def signature(call: dict) -> str:
    """工具调用的稳定签名，用于重复调用检测。"""
    return json.dumps(
        {"name": call.get("name", ""), "args": call.get("args") or {}},
        ensure_ascii=False, sort_keys=True,
    )
