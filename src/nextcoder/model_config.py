"""模型 profile：从 TOML 选择 OpenAI 兼容入口，避免把 provider 写死在代码里。

配置文件只保存地址、模型名和密钥所在的环境变量名，不保存密钥本身。
`protocol` 目前只接受 `openai-compatible`；Anthropic/Gemini 等原生协议不在
这一层模拟，统一走它们提供的 OpenAI 兼容端点。
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import SecretStr

from nextcoder.settings import Settings

SUPPORTED_PROTOCOLS = {"openai-compatible"}


class ModelConfigError(ValueError):
    """profile 配置缺失或字段不合法。"""


@dataclass(frozen=True)
class ModelProfile:
    name: str
    protocol: str = "openai-compatible"
    provider: str = ""
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    thinking: str = ""
    session_header: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    headers: dict[str, str] = field(default_factory=dict)
    fallback_base_url: str = ""
    fallback_model: str = ""
    fallback_api_key_env: str = ""

    def snapshot(self) -> dict[str, str]:
        """只含非敏感字段，可写进日志或报告。"""
        return {
            "name": self.name,
            "provider": self.provider,
            "protocol": self.protocol,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
        }


def load_model_profiles(path: Path) -> dict[str, ModelProfile]:
    """读取 `[[models]]`；文件不存在返回空表，格式错误给出明确异常。"""
    if not path.is_file():
        return {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ModelConfigError(f"无法读取模型配置 {path}：{exc}") from exc
    entries = data.get("models") or []
    if not isinstance(entries, list):
        raise ModelConfigError(f"{path} 的 models 必须是 [[models]] 数组")
    profiles: dict[str, ModelProfile] = {}
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ModelConfigError(f"{path} 的第 {index} 个 [[models]] 不是对象")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise ModelConfigError(f"{path} 的第 {index} 个 [[models]] 缺少 name")
        if name in profiles:
            # 同名条目若静默后者覆盖前者，可能把请求发到错误端点/密钥。
            raise ModelConfigError(f"{path} 存在重复的模型 profile 名称：{name}")
        protocol = str(entry.get("protocol") or "openai-compatible").strip()
        if protocol not in SUPPORTED_PROTOCOLS:
            raise ModelConfigError(
                f"profile {name} 的 protocol={protocol!r} 不支持；"
                f"可选：{', '.join(sorted(SUPPORTED_PROTOCOLS))}")
        raw_headers = entry.get("headers") or {}
        headers = {
            str(key): _expand_secret_ref(str(value))
            for key, value in raw_headers.items()
        } if isinstance(raw_headers, dict) else {}
        profiles[name] = ModelProfile(
            name=name,
            protocol=protocol,
            provider=str(entry.get("provider") or "").strip(),
            base_url=str(entry.get("base_url") or "").strip(),
            model=str(entry.get("model") or "").strip(),
            api_key_env=str(entry.get("api_key_env") or "").strip(),
            thinking=str(entry.get("thinking") or "").strip(),
            session_header=_session_header(entry),
            context_window=_positive_int(entry.get("context_window")),
            max_output_tokens=_positive_int(entry.get("max_output_tokens")),
            headers=headers,
            fallback_base_url=str(entry.get("fallback_base_url") or "").strip(),
            fallback_model=str(entry.get("fallback_model") or "").strip(),
            fallback_api_key_env=str(
                entry.get("fallback_api_key_env") or "").strip(),
        )
    return profiles


def _session_header(entry: dict) -> str | None:
    """缺省为 None（沿用默认头）；false 表示显式关闭；true 视为配置错误。"""
    if "session_header" not in entry:
        return None
    value = entry["session_header"]
    if value is False or value is None:
        return ""
    if value is True:
        raise ModelConfigError("session_header 必须是字符串或 false")
    return str(value)


def _positive_int(value) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ModelConfigError(f"配置项必须是整数：{value!r}") from exc
    if number <= 0:
        raise ModelConfigError(f"配置项必须为正数：{value!r}")
    return number


def _expand_secret_ref(value: str) -> str:
    """headers 里的 `{env:VAR}` 展开；未定义时保留空串。

    与 api_key_env 使用同一个读取顺序（进程环境优先，其次当前目录 .env），
    避免同一个变量在密钥和 header 里表现不一致。
    """
    import re

    pattern = re.compile(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}")
    return pattern.sub(lambda match: _read_env_value(match.group(1)), value)


def _read_env_value(name: str) -> str:
    """先读进程环境，再读当前目录 .env（不修改 os.environ）。"""
    if not name:
        return ""
    value = os.environ.get(name)
    if value:
        return value
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover - python-dotenv 随 pydantic-settings 安装
        return ""
    values = dotenv_values(".env")
    return str(values.get(name) or "")


def select_profile(profiles: dict[str, ModelProfile], name: str) -> ModelProfile:
    if not profiles:
        raise ModelConfigError("模型配置里没有 [[models]] 条目")
    if name:
        profile = profiles.get(name)
        if profile is None:
            raise ModelConfigError(
                f"找不到模型 profile {name!r}；可选：{', '.join(profiles)}")
        return profile
    if "primary" in profiles:
        return profiles["primary"]
    return next(iter(profiles.values()))


def apply_model_profile(settings: Settings, profile: ModelProfile,
                        *, api_key: str | None = None) -> Settings:
    """把 profile 应用到 Settings；profile 的显式字段优先于环境默认值。

    `api_key` 用于交互式 TUI：用户在本次进程里临时填写的密钥优先于环境变量，
    不落盘、不写日志。
    """
    if not profile.base_url or not profile.model:
        raise ModelConfigError(
            f"profile {profile.name} 缺少 base_url 或 model")
    api_key = (api_key or "").strip() or _read_env_value(profile.api_key_env)
    if not api_key and not profile.api_key_env:
        # 只有未指定密钥变量名时才退回通用 NEXTCODER_LLM_API_KEY；显式指定
        # 却缺失时必须报错，避免把别家 provider 的密钥发给当前端点。
        api_key = settings.llm_api_key.get_secret_value()
    if not api_key:
        raise ModelConfigError(
            f"profile {profile.name} 的密钥环境变量 "
            f"{profile.api_key_env or 'NEXTCODER_LLM_API_KEY'} 未设置")

    updates: dict = {
        "llm_base_url": profile.base_url,
        "llm_model": profile.model,
        "llm_api_key": SecretStr(api_key),
        "llm_provider": profile.provider or profile.name,
        "model_profile": profile.name,
    }
    if profile.thinking:
        updates["llm_thinking"] = profile.thinking
    if profile.session_header is not None:
        updates["llm_session_header"] = profile.session_header
    if profile.headers:
        updates["llm_extra_headers"] = dict(profile.headers)
    if profile.context_window is not None:
        updates["application_input_token_budget"] = profile.context_window
    if profile.max_output_tokens is not None:
        updates["output_reserve_tokens"] = profile.max_output_tokens
    if profile.fallback_base_url and profile.fallback_model:
        fallback_key = _read_env_value(profile.fallback_api_key_env)
        if fallback_key:
            updates.update({
                "llm_fallback_base_url": profile.fallback_base_url,
                "llm_fallback_model": profile.fallback_model,
                "llm_fallback_api_key": SecretStr(fallback_key),
            })
    return settings.model_copy(update=updates)


def resolve_settings(settings: Settings, *, profile_name: str = "") -> Settings:
    """按需应用 profile。

    - 显式指定 profile：文件或条目缺失直接报错；
    - 未指定且 `llm_base_url`/`llm_model` 已由环境提供：保持环境配置；
    - 未指定且模型未配置：若配置文件存在，选 `primary`（否则第一条）。
    """
    path = Path(settings.model_config_file).expanduser()
    explicit = bool(profile_name or settings.model_profile)
    # 只有 base_url 和 model 都来自环境时才视为完整环境配置；只给一半时
    # 继续尝试 profile，避免静默跳过 profile 后在调用模型时才报错。
    if not explicit and settings.llm_base_url and settings.llm_model:
        return settings
    profiles = load_model_profiles(path)
    if not profiles:
        if explicit:
            raise ModelConfigError(f"模型配置文件不存在或为空：{path}")
        if settings.llm_base_url or settings.llm_model:
            raise ModelConfigError(
                "模型配置不完整：llm_base_url 和 llm_model 必须同时设置，"
                "或配置 model profile")
        return settings
    name = profile_name or settings.model_profile
    return apply_model_profile(settings, select_profile(profiles, name))
