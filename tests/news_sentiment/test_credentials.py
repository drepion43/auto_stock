import pytest

from auto_stock.news_sentiment.credentials import DEFAULT_MODEL, load_llm_config


def test_load_llm_config_uses_env_api_key_and_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-example")
    monkeypatch.delenv("NEWS_SENTIMENT_OPENAI_MODEL", raising=False)

    config = load_llm_config()

    assert config.api_key == "sk-example"
    assert config.model == DEFAULT_MODEL


def test_load_llm_config_reads_model_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-example")
    monkeypatch.setenv("NEWS_SENTIMENT_OPENAI_MODEL", "gpt-5.6-terra")

    config = load_llm_config()

    assert config.model == "gpt-5.6-terra"


def test_load_llm_config_raises_key_error_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(KeyError):
        load_llm_config()


def test_load_llm_config_does_not_require_naver_credentials(monkeypatch):
    """해석 계층은 OPENAI_API_KEY만 필요하다 — NAVER_CLIENT_ID/SECRET은
    data/sources/naver_news_source.py 소관."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-example")
    monkeypatch.delenv("NAVER_CLIENT_ID", raising=False)
    monkeypatch.delenv("NAVER_CLIENT_SECRET", raising=False)

    config = load_llm_config()

    assert config.api_key == "sk-example"
