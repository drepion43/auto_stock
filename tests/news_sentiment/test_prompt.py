import inspect
from datetime import date

from auto_stock.news_sentiment.models import NewsSummary
from auto_stock.news_sentiment.prompt import SYSTEM_PROMPT, build_user_prompt, render_news_list

from .conftest import make_articles


def _summary(ticker="005930", market="KRX", as_of=date(2024, 6, 10)) -> NewsSummary:
    items = make_articles(3, ticker=ticker, market=market, start=date(2024, 6, 1))
    return NewsSummary(ticker=ticker, market=market, as_of=as_of, items=items)


def test_build_user_prompt_does_not_leak_ticker():
    summary = _summary(ticker="005930")

    prompt = build_user_prompt(summary)

    assert "005930" not in prompt


def test_build_user_prompt_does_not_leak_market():
    summary = _summary(market="KRX")

    prompt = build_user_prompt(summary)

    assert "KRX" not in prompt
    assert "NASDAQ" not in prompt


def test_build_user_prompt_does_not_leak_real_date():
    summary = _summary(as_of=date(2024, 6, 10))

    prompt = build_user_prompt(summary)

    assert "2024" not in prompt
    assert summary.as_of.isoformat() not in prompt
    for item in summary.items:
        assert item.published_at.isoformat() not in prompt


def test_build_user_prompt_does_not_leak_url():
    summary = _summary()

    prompt = build_user_prompt(summary)

    for item in summary.items:
        assert item.url not in prompt


def test_build_user_prompt_signature_has_no_action_parameter():
    """동조(sycophancy) 방어 잠금: 함수 시그니처 자체가 action을 받지 않는다."""
    signature = inspect.signature(build_user_prompt)

    assert "action" not in signature.parameters


def test_build_user_prompt_renders_relative_offsets():
    summary = _summary(as_of=date(2024, 6, 10))

    prompt = build_user_prompt(summary)

    for item in summary.items:
        offset = (summary.as_of - item.published_at).days
        assert f"D-{offset}" in prompt


def test_build_user_prompt_renders_titles_and_source():
    summary = _summary()

    prompt = build_user_prompt(summary)

    for item in summary.items:
        assert item.title in prompt
        assert item.source in prompt


def test_render_news_list_includes_source():
    items = make_articles(1, source="press.example.com")
    summary = NewsSummary(ticker="005930", market="KRX", as_of=date(2024, 6, 10), items=items)

    text = render_news_list(summary)

    assert "press.example.com" in text


def test_system_prompt_mentions_neutral_and_auxiliary_keywords():
    assert "NEUTRAL" in SYSTEM_PROMPT
    assert "보조" in SYSTEM_PROMPT


def test_system_prompt_instructs_treating_article_content_as_data_not_instructions():
    """보안 리뷰 발견사항(Stage E) — 네이버뉴스/GDELT가 색인한(사실상 공개 웹이 통제하는)
    기사 제목이 프롬프트에 그대로 삽입되므로, 그 안의 지시문처럼 보이는 문구를 따르지
    말라는 방어 지침이 필요하다."""
    assert "데이터" in SYSTEM_PROMPT
    assert "무시" in SYSTEM_PROMPT
