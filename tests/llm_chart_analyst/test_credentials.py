import pytest

from auto_stock.llm_chart_analyst.credentials import DEFAULT_MODEL, load_llm_config


def test_load_llm_config_uses_env_api_key_and_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-example")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("LLM_MAX_CALLS_PER_RUN", raising=False)

    config = load_llm_config()

    assert config.api_key == "sk-example"
    assert config.model == DEFAULT_MODEL
    assert config.max_calls_per_run == 20


def test_load_llm_config_reads_optional_overrides(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-example")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-terra")
    monkeypatch.setenv("LLM_MAX_CALLS_PER_RUN", "3")

    config = load_llm_config()

    assert config.model == "gpt-5.6-terra"
    assert config.max_calls_per_run == 3


def test_load_llm_config_raises_key_error_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(KeyError):
        load_llm_config()
