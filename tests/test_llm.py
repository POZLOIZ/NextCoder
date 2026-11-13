"""模型入口测试：会话头、thinking 配置与 429 备援，不联网。"""

import httpx
import pytest
from openai import RateLimitError

from nextcoder.llm import FallbackChat, build_chat_llm
from nextcoder.settings import Settings


def _settings(**overrides):
    values = dict(
        llm_base_url="https://opencode.ai/inference/go/openai/v1",
        llm_model="deepseek-v4.1-flash",
        llm_api_key="test",
        llm_session="nextcoder",
        llm_session_header="x-opencode-session",
        llm_thinking="disabled",
    )
    values.update(overrides)
    return Settings(**values)


def test_session_header_is_stable_and_scoped_to_session():
    llm = build_chat_llm(_settings(), session_id="sess-1")
    primary = llm._primary
    assert primary.default_headers == {
        "x-opencode-session": "nextcoder-sess-1"}
    assert primary.extra_body == {"thinking": {"type": "disabled"}}


def test_output_reserve_is_sent_as_max_tokens():
    llm = build_chat_llm(_settings(output_reserve_tokens=4096))
    assert llm._primary.max_tokens == 4096


def test_thinking_can_be_enabled_or_omitted():
    enabled = build_chat_llm(_settings(llm_thinking="enabled"))
    assert enabled._primary.extra_body == {"thinking": {"type": "enabled"}}
    omitted = build_chat_llm(_settings(llm_thinking="omit"))
    assert omitted._primary.extra_body is None


def test_fallback_config_is_opt_in_and_does_not_expose_keys():
    plain = build_chat_llm(_settings())
    assert plain._backup is None

    with_fallback = build_chat_llm(_settings(
        llm_fallback_base_url="https://api.deepseek.com",
        llm_fallback_model="deepseek-flash",
        llm_fallback_api_key="backup-test",
    ))
    assert with_fallback._backup is not None


class _RateLimited:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages, **kwargs):
        self.calls += 1
        request = httpx.Request("POST", "https://example.test/chat/completions")
        response = httpx.Response(429, request=request)
        raise RateLimitError("rate limited", response=response, body=None)


class _Reply:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def invoke(self, messages, **kwargs):
        self.calls += 1
        return self.text


def test_rate_limit_switches_to_backup():
    backup = _Reply("backup")
    llm = FallbackChat(_RateLimited(), backup)
    assert llm.invoke([]) == "backup"
    assert llm.last_endpoint == "backup"
    assert backup.calls == 1


def test_non_rate_limit_error_is_not_swallowed():
    class _Broken:
        def invoke(self, messages, **kwargs):
            raise ValueError("bad request")

    backup = _Reply("backup")
    llm = FallbackChat(_Broken(), backup)
    with pytest.raises(ValueError, match="bad request"):
        llm.invoke([])
    assert backup.calls == 0


def test_missing_primary_configuration_fails_clearly():
    with pytest.raises(ValueError, match="NEXTCODER_LLM_BASE_URL"):
        build_chat_llm(_settings(llm_base_url="", llm_model=""))


def test_reasoning_content_survives_tool_round_trip():
    from langchain_core.messages import HumanMessage

    from nextcoder.llm import _ReasoningChatOpenAI

    llm = _ReasoningChatOpenAI(
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
        api_key="test",
        preserve_reasoning=True,
    )
    response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek-flash",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "I will check.",
                "reasoning_content": "the tool should be used",
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    result = llm._create_chat_result(response)
    assistant = result.generations[0].message
    assert assistant.additional_kwargs["reasoning_content"] == "the tool should be used"

    payload = llm._get_request_payload(
        [HumanMessage(content="check"), assistant])
    assert payload["messages"][1]["reasoning_content"] == "the tool should be used"
