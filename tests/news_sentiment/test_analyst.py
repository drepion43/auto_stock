"""`analyst.py`는 `openai`를 import하지 않는다 — `NewsSentimentReader` Protocol을 만족하는
가짜(Fake) 리더로 진짜 SDK 없이 도메인 로직을 테스트한다."""

from datetime import date

from auto_stock.news_sentiment.analyst import NEWS_SENTIMENT_DISCLAIMER, analyze, to_reasons
from auto_stock.news_sentiment.schema import NewsSentimentRead

from .conftest import FakeNewsSentimentReader, make_articles, make_news_sentiment_analysis


def test_analyst_module_does_not_import_openai():
    import inspect

    import auto_stock.news_sentiment.analyst as analyst_module

    source = inspect.getsource(analyst_module)
    assert "openai" not in source


def test_analyze_returns_none_and_does_not_call_reader_when_no_articles():
    reader = FakeNewsSentimentReader()

    result = analyze(reader, [], ticker="005930", market="KRX", as_of=date(2024, 6, 10))

    assert result is None
    assert reader.calls == []


def test_analyze_fills_ticker_market_as_of_from_arguments_not_response():
    reader = FakeNewsSentimentReader()
    articles = make_articles(2, ticker="000660")

    result = analyze(reader, articles, ticker="000660", market="KRX", as_of=date(2024, 6, 10))

    assert result is not None
    assert result.ticker == "000660"
    assert result.market == "KRX"
    assert result.as_of == date(2024, 6, 10)


def test_analyze_has_no_action_parameter():
    """동조 방어 잠금: analyze는 action을 받지 않는다."""
    import inspect

    signature = inspect.signature(analyze)
    assert "action" not in signature.parameters


def test_analyze_populates_fields_from_reader_response():
    response = NewsSentimentRead(
        sentiment="NEGATIVE",
        confidence="HIGH",
        key_headline="실적 부진 우려 확산",
        rationale="시장 전망이 부정적으로 전환되었습니다.",
        caveat="추가 실적 발표 확인 필요",
    )
    reader = FakeNewsSentimentReader(response=response, model="gpt-5.6-terra")
    articles = make_articles(2)

    result = analyze(reader, articles, ticker="005930", market="KRX", as_of=date(2024, 6, 10))

    assert result.sentiment == "NEGATIVE"
    assert result.confidence == "HIGH"
    assert result.key_headline == "실적 부진 우려 확산"
    assert result.rationale == "시장 전망이 부정적으로 전환되었습니다."
    assert result.caveat == "추가 실적 발표 확인 필요"
    assert result.model == "gpt-5.6-terra"


def test_analyze_caps_items_sent_to_reader_at_max_articles_per_query():
    from auto_stock.news_sentiment.credentials import MAX_ARTICLES_PER_QUERY

    reader = FakeNewsSentimentReader()
    articles = make_articles(MAX_ARTICLES_PER_QUERY + 5)

    analyze(reader, articles, ticker="005930", market="KRX", as_of=date(2024, 6, 20))

    _, user_prompt = reader.calls[0]
    offset_lines = [line for line in user_prompt.splitlines() if line.startswith("D-")]
    assert len(offset_lines) == MAX_ARTICLES_PER_QUERY


def test_to_reasons_returns_empty_list_for_none_analysis():
    assert to_reasons(None, action="BUY") == []


def test_to_reasons_positive_sentiment_agrees_with_buy():
    analysis = make_news_sentiment_analysis(
        sentiment="POSITIVE", confidence="HIGH", key_headline="신규 투자 발표"
    )

    reasons = to_reasons(analysis, action="BUY")

    assert any("동의" in r for r in reasons)
    assert any("신규 투자 발표" in r for r in reasons)


def test_to_reasons_negative_sentiment_agrees_with_sell():
    analysis = make_news_sentiment_analysis(sentiment="NEGATIVE", confidence="MEDIUM")

    reasons = to_reasons(analysis, action="SELL")

    assert any("동의" in r for r in reasons)


def test_to_reasons_negative_sentiment_conflicts_with_buy():
    analysis = make_news_sentiment_analysis(sentiment="NEGATIVE", confidence="MEDIUM")

    reasons = to_reasons(analysis, action="BUY")

    assert any("상충" in r for r in reasons)


def test_to_reasons_positive_sentiment_conflicts_with_sell():
    analysis = make_news_sentiment_analysis(sentiment="POSITIVE", confidence="MEDIUM")

    reasons = to_reasons(analysis, action="SELL")

    assert any("상충" in r for r in reasons)


def test_to_reasons_neutral_sentiment_is_neither_agree_nor_conflict():
    analysis = make_news_sentiment_analysis(sentiment="NEUTRAL", confidence="LOW")

    reasons = to_reasons(analysis, action="BUY")

    assert not any("동의" in r for r in reasons)
    assert not any("상충" in r for r in reasons)
    assert any("중립" in r for r in reasons)


def test_to_reasons_disclaimer_is_always_last():
    for sentiment in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
        analysis = make_news_sentiment_analysis(sentiment=sentiment)
        reasons = to_reasons(analysis, action="BUY")
        assert reasons[-1] == NEWS_SENTIMENT_DISCLAIMER


def test_to_reasons_includes_caveat_line_when_present():
    analysis = make_news_sentiment_analysis(caveat="추가 실적 발표 확인 필요")

    reasons = to_reasons(analysis, action="BUY")

    assert any("단서" in r and "추가 실적 발표 확인 필요" in r for r in reasons)


def test_to_reasons_omits_caveat_line_when_none():
    analysis = make_news_sentiment_analysis(caveat=None)

    reasons = to_reasons(analysis, action="BUY")

    assert not any("단서" in r for r in reasons)


def test_to_reasons_includes_rationale():
    analysis = make_news_sentiment_analysis(
        rationale="시장이 긍정적으로 반응할 것으로 예상되는 투자 소식입니다."
    )

    reasons = to_reasons(analysis, action="BUY")

    assert any("시장이 긍정적으로 반응할 것으로 예상되는 투자 소식입니다." in r for r in reasons)
