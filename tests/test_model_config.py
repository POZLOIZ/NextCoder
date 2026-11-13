"""模型 profile：TOML 读取、选择、应用到 Settings，且不泄露密钥。"""

import json

import pytest
from typer.testing import CliRunner

from nextcoder.cli import app
from nextcoder.model_config import (
    ModelConfigError,
    apply_model_profile,
    load_model_profiles,
    resolve_settings,
    select_profile,
)
from nextcoder.settings import Settings


def _write_config(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(
        """
[[models]]
name = "primary"
provider = "deepseek"
base_url = "https://api.example.com/v1"
model = "model-a"
api_key_env = "PROFILE_KEY"
thinking = "enabled"
session_header = ""
context_window = 64000
max_output_tokens = 2048
headers = { "X-Title" = "NextCoder" }

[[models]]
name = "work"
base_url = "https://work.example.com/v1"
model = "model-b"
api_key_env = "WORK_KEY"
fallback_base_url = "https://fallback.example.com/v1"
fallback_model = "model-c"
fallback_api_key_env = "FALLBACK_KEY"
""",
        encoding="utf-8",
    )
    return path


def test_load_and_select_profiles(tmp_path):
    profiles = load_model_profiles(_write_config(tmp_path))
    assert set(profiles) == {"primary", "work"}
    primary = select_profile(profiles, "primary")
    assert primary.provider == "deepseek"
    assert primary.context_window == 64000
    assert primary.session_header == ""
    assert primary.headers == {"X-Title": "NextCoder"}
    assert select_profile(profiles, "") is primary


def test_unknown_profile_is_clear(tmp_path):
    profiles = load_model_profiles(_write_config(tmp_path))
    with pytest.raises(ModelConfigError, match="找不到模型 profile"):
        select_profile(profiles, "missing")


def test_apply_profile_overrides_runtime_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("PROFILE_KEY", "profile-secret")
    monkeypatch.setenv("WORK_KEY", "work-secret")
    monkeypatch.setenv("FALLBACK_KEY", "fallback-secret")
    profiles = load_model_profiles(_write_config(tmp_path))
    settings = Settings(llm_base_url="", llm_model="")
    applied = apply_model_profile(settings, profiles["work"])
    assert applied.llm_base_url == "https://work.example.com/v1"
    assert applied.llm_model == "model-b"
    assert applied.llm_api_key.get_secret_value() == "work-secret"
    assert applied.llm_fallback_model == "model-c"
    assert applied.llm_fallback_api_key.get_secret_value() == "fallback-secret"
    assert applied.model_profile == "work"


def test_apply_profile_uses_budget_and_thinking(tmp_path, monkeypatch):
    monkeypatch.setenv("PROFILE_KEY", "profile-secret")
    profiles = load_model_profiles(_write_config(tmp_path))
    applied = apply_model_profile(Settings(), profiles["primary"])
    assert applied.application_input_token_budget == 64000
    assert applied.output_reserve_tokens == 2048
    assert applied.llm_thinking == "enabled"
    assert applied.llm_session_header == ""


def test_profile_requires_key(tmp_path, monkeypatch):
    monkeypatch.delenv("PROFILE_KEY", raising=False)
    profiles = load_model_profiles(_write_config(tmp_path))
    with pytest.raises(ModelConfigError, match="密钥环境变量"):
        apply_model_profile(Settings(), profiles["primary"])


def test_named_key_missing_does_not_use_generic_key(tmp_path, monkeypatch):
    monkeypatch.delenv("WORK_KEY", raising=False)
    profiles = load_model_profiles(_write_config(tmp_path))
    settings = Settings(llm_api_key="generic-secret")
    with pytest.raises(ModelConfigError, match="WORK_KEY"):
        apply_model_profile(settings, profiles["work"])


def test_resolve_keeps_environment_config_when_both_set(tmp_path):
    path = _write_config(tmp_path)
    settings = Settings(
        model_config_file=path, llm_base_url="https://env", llm_model="env-model")
    resolved = resolve_settings(settings)
    assert resolved.llm_base_url == "https://env"
    assert resolved.llm_model == "env-model"


def test_partial_environment_uses_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("PROFILE_KEY", "profile-secret")
    path = _write_config(tmp_path)
    settings = Settings(
        model_config_file=path, llm_base_url="https://env", llm_model="")
    resolved = resolve_settings(settings)
    assert resolved.llm_model == "model-a"


def test_partial_environment_without_profile_is_clear(tmp_path):
    settings = Settings(
        model_config_file=tmp_path / "missing.toml",
        llm_base_url="https://env")
    with pytest.raises(ModelConfigError, match="同时"):
        resolve_settings(settings)


def test_resolve_auto_selects_primary_when_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("PROFILE_KEY", "profile-secret")
    path = _write_config(tmp_path)
    resolved = resolve_settings(Settings(model_config_file=path))
    assert resolved.llm_model == "model-a"
    assert resolved.model_profile == "primary"


def test_invalid_protocol_is_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(
        '[[models]]\nname="x"\nprotocol="anthropic"\nbase_url="u"\n'
        'model="m"\n', encoding="utf-8")
    with pytest.raises(ModelConfigError, match="protocol"):
        load_model_profiles(path)


def test_duplicate_profile_name_is_rejected(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(
        '[[models]]\nname="x"\nbase_url="u1"\nmodel="m1"\n'
        '[[models]]\nname="x"\nbase_url="u2"\nmodel="m2"\n',
        encoding="utf-8")
    with pytest.raises(ModelConfigError, match="重复"):
        load_model_profiles(path)


def test_nameless_profile_is_rejected(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(
        '[[models]]\nbase_url="u"\nmodel="m"\n', encoding="utf-8")
    with pytest.raises(ModelConfigError, match="name"):
        load_model_profiles(path)


def test_header_env_expands_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("HEADER_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("HEADER_KEY=dotenv-value\n", encoding="utf-8")
    path = tmp_path / "models.toml"
    path.write_text(
        '[[models]]\nname="x"\nbase_url="u"\nmodel="m"\n'
        'headers = { "X-Token" = "Bearer {env:HEADER_KEY}" }\n',
        encoding="utf-8")
    profile = load_model_profiles(path)["x"]
    assert profile.headers["X-Token"] == "Bearer dotenv-value"


def test_session_header_false_disables_header(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(
        '[[models]]\nname="x"\nbase_url="u"\nmodel="m"\n'
        'session_header=false\n', encoding="utf-8")
    assert load_model_profiles(path)["x"].session_header == ""


def test_models_command_lists_without_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("PROFILE_KEY", "super-secret")
    path = _write_config(tmp_path)
    result = CliRunner().invoke(app, ["models", "--config", str(path)])
    assert result.exit_code == 0, result.output
    assert "primary" in result.output
    assert "model-a" in result.output
    assert "PROFILE_KEY" in result.output
    assert "super-secret" not in result.output


def test_model_config_snapshot_has_no_key(tmp_path):
    profiles = load_model_profiles(_write_config(tmp_path))
    text = json.dumps(profiles["primary"].snapshot())
    assert "api_key_env" in text
    assert "PROFILE_KEY" in text
    assert "secret" not in text
