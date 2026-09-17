"""`analyst.py`는 `openai`를 import하지 않는다 — `DisclosureReader` Protocol을 만족하는
가짜(Fake) 리더로 진짜 SDK 없이 도메인 로직을 테스트한다."""

from datetime import date

from auto_stock.news_disclosure.analyst import NEWS_DISCLAIMER, analyze, to_reasons
from auto_stock.news_disclosure.schema import DisclosureRead

from .conftest import FakeDisclosureReader, make_disclosure_analysis, make_disclosures


def test_analyst_module_does_not_import_openai():
    import inspect

    import auto_stock.news_disclosure.analyst as analyst_module

    source = inspect.getsource(analyst_module)
    assert "openai" not in source


def test_analyze_returns_none_and_does_not_call_reader_when_no_disclosures():
    reader = FakeDisclosureReader()

    result = analyze(reader, [], ticker="005930", market="KRX", as_of=date(2024, 6, 10))

    assert result is None
    assert reader.calls == []


def test_analyze_fills_ticker_market_as_of_from_arguments_not_response():
    reader = FakeDisclosureReader()
    disclosures = make_disclosures(2, ticker="000660")

    result = analyze(reader, disclosures, ticker="000660", market="KRX", as_of=date(2024, 6, 10))

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
    response = DisclosureRead(
        market_impact="NEGATIVE",
        confidence="HIGH",
        key_event="횡령·배임 혐의 발생",
        rationale="경영 리스크가 부각되었습니다.",
        caveat="검찰 수사 결과 확인 필요",
    )
    reader = FakeDisclosureReader(response=response, model="gpt-5.6-terra")
    disclosures = make_disclosures(2)

    result = analyze(reader, disclosures, ticker="005930", market="KRX", as_of=date(2024, 6, 10))

    assert result.market_impact == "NEGATIVE"
    assert result.confidence == "HIGH"
    assert result.key_event == "횡령·배임 혐의 발생"
    assert result.rationale == "경영 리스크가 부각되었습니다."
    assert result.caveat == "검찰 수사 결과 확인 필요"
    assert result.model == "gpt-5.6-terra"


def test_analyze_caps_items_sent_to_reader_at_max_disclosures_per_query():
    from auto_stock.news_disclosure.credentials import MAX_DISCLOSURES_PER_QUERY

    reader = FakeDisclosureReader()
    disclosures = make_disclosures(MAX_DISCLOSURES_PER_QUERY + 5)

    analyze(reader, disclosures, ticker="005930", market="KRX", as_of=date(2024, 6, 20))

    _, user_prompt = reader.calls[0]
    offset_lines = [line for line in user_prompt.splitlines() if line.startswith("D-")]
    assert len(offset_lines) == MAX_DISCLOSURES_PER_QUERY


def test_to_reasons_returns_empty_list_for_none_analysis():
    assert to_reasons(None, action="BUY") == []


def test_to_reasons_positive_impact_agrees_with_buy():
    analysis = make_disclosure_analysis(
        market_impact="POSITIVE", confidence="HIGH", key_event="유상증자 결정"
    )

    reasons = to_reasons(analysis, action="BUY")

    assert any("동의" in r for r in reasons)
    assert any("유상증자 결정" in r for r in reasons)


def test_to_reasons_negative_impact_agrees_with_sell():
    analysis = make_disclosure_analysis(market_impact="NEGATIVE", confidence="MEDIUM")

    reasons = to_reasons(analysis, action="SELL")

    assert any("동의" in r for r in reasons)


def test_to_reasons_negative_impact_conflicts_with_buy():
    analysis = make_disclosure_analysis(market_impact="NEGATIVE", confidence="MEDIUM")

    reasons = to_reasons(analysis, action="BUY")

    assert any("상충" in r for r in reasons)


def test_to_reasons_positive_impact_conflicts_with_sell():
    analysis = make_disclosure_analysis(market_impact="POSITIVE", confidence="MEDIUM")

    reasons = to_reasons(analysis, action="SELL")

    assert any("상충" in r for r in reasons)


def test_to_reasons_neutral_impact_is_neither_agree_nor_conflict():
    analysis = make_disclosure_analysis(market_impact="NEUTRAL", confidence="LOW")

    reasons = to_reasons(analysis, action="BUY")

    assert not any("동의" in r for r in reasons)
    assert not any("상충" in r for r in reasons)
    assert any("중립" in r for r in reasons)


def test_to_reasons_disclaimer_is_always_last():
    for impact in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
        analysis = make_disclosure_analysis(market_impact=impact)
        reasons = to_reasons(analysis, action="BUY")
        assert reasons[-1] == NEWS_DISCLAIMER


def test_to_reasons_includes_caveat_line_when_present():
    analysis = make_disclosure_analysis(caveat="검찰 수사 결과 확인 필요")

    reasons = to_reasons(analysis, action="BUY")

    assert any("단서" in r and "검찰 수사 결과 확인 필요" in r for r in reasons)


def test_to_reasons_omits_caveat_line_when_none():
    analysis = make_disclosure_analysis(caveat=None)

    reasons = to_reasons(analysis, action="BUY")

    assert not any("단서" in r for r in reasons)


def test_to_reasons_includes_rationale():
    analysis = make_disclosure_analysis(
        rationale="자금조달을 통한 사업 확장 계획이 공시되었습니다."
    )

    reasons = to_reasons(analysis, action="BUY")

    assert any("자금조달을 통한 사업 확장 계획이 공시되었습니다." in r for r in reasons)
