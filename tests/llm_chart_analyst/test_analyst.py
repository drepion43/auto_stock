"""`analyst.py`는 `openai`를 import하지 않는다 — `ChartPatternReader` Protocol을 만족하는
가짜(Fake) 리더로 진짜 SDK 없이 도메인 로직을 테스트한다."""

from auto_stock.llm_chart_analyst.analyst import LLM_DISCLAIMER, analyze, to_reasons
from auto_stock.llm_chart_analyst.schema import ChartPatternRead

from .conftest import FakeChartPatternReader, make_chart_analysis, make_records, random_walk_prices


def test_analyst_module_does_not_import_openai():
    import inspect

    import auto_stock.llm_chart_analyst.analyst as analyst_module

    source = inspect.getsource(analyst_module)
    assert "openai" not in source


def test_analyze_returns_none_and_does_not_call_reader_when_data_insufficient():
    reader = FakeChartPatternReader()
    records = make_records(random_walk_prices(10))  # below SMA60 warmup

    result = analyze(reader, records)

    assert result is None
    assert reader.calls == []


def test_analyze_fills_ticker_market_date_from_records_not_response():
    reader = FakeChartPatternReader()
    records = make_records(random_walk_prices(80), ticker="000660", market="KRX")

    result = analyze(reader, records)

    assert result is not None
    assert result.ticker == "000660"
    assert result.market == "KRX"
    assert result.date == records[-1].date


def test_analyze_has_no_action_parameter():
    """동조 방어 잠금: analyze는 action을 받지 않는다."""
    import inspect

    signature = inspect.signature(analyze)
    assert "action" not in signature.parameters


def test_analyze_populates_fields_from_reader_response():
    response = ChartPatternRead(
        direction="DOWN",
        confidence="HIGH",
        pattern_name="헤드앤숄더",
        rationale="RSI 과열 후 하락 전환.",
        caveat="거래량 확인 필요",
    )
    reader = FakeChartPatternReader(response=response, model="gpt-5.6-terra")
    records = make_records(random_walk_prices(80))

    result = analyze(reader, records)

    assert result.direction == "DOWN"
    assert result.confidence == "HIGH"
    assert result.pattern_name == "헤드앤숄더"
    assert result.rationale == "RSI 과열 후 하락 전환."
    assert result.caveat == "거래량 확인 필요"
    assert result.model == "gpt-5.6-terra"


def test_to_reasons_returns_empty_list_for_none_analysis():
    assert to_reasons(None, action="BUY") == []


def test_to_reasons_up_direction_agrees_with_buy():
    analysis = make_chart_analysis(direction="UP", confidence="HIGH", pattern_name="상승 삼각수렴")

    reasons = to_reasons(analysis, action="BUY")

    assert any("동의" in r for r in reasons)
    assert any("상승 삼각수렴" in r for r in reasons)


def test_to_reasons_down_direction_agrees_with_sell():
    analysis = make_chart_analysis(direction="DOWN", confidence="MEDIUM")

    reasons = to_reasons(analysis, action="SELL")

    assert any("동의" in r for r in reasons)


def test_to_reasons_down_direction_conflicts_with_buy():
    analysis = make_chart_analysis(direction="DOWN", confidence="MEDIUM")

    reasons = to_reasons(analysis, action="BUY")

    assert any("상충" in r for r in reasons)


def test_to_reasons_up_direction_conflicts_with_sell():
    analysis = make_chart_analysis(direction="UP", confidence="MEDIUM")

    reasons = to_reasons(analysis, action="SELL")

    assert any("상충" in r for r in reasons)


def test_to_reasons_neutral_direction_is_neither_agree_nor_conflict():
    analysis = make_chart_analysis(direction="NEUTRAL", confidence="LOW")

    reasons = to_reasons(analysis, action="BUY")

    assert not any("동의" in r for r in reasons)
    assert not any("상충" in r for r in reasons)
    assert any("중립" in r for r in reasons)


def test_to_reasons_disclaimer_is_always_last():
    for direction in ["UP", "DOWN", "NEUTRAL"]:
        analysis = make_chart_analysis(direction=direction)
        reasons = to_reasons(analysis, action="BUY")
        assert reasons[-1] == LLM_DISCLAIMER


def test_to_reasons_includes_caveat_line_when_present():
    analysis = make_chart_analysis(caveat="거래량 확인 필요")

    reasons = to_reasons(analysis, action="BUY")

    assert any("단서" in r and "거래량 확인 필요" in r for r in reasons)


def test_to_reasons_omits_caveat_line_when_none():
    analysis = make_chart_analysis(caveat=None)

    reasons = to_reasons(analysis, action="BUY")

    assert not any("단서" in r for r in reasons)


def test_to_reasons_includes_rationale():
    analysis = make_chart_analysis(rationale="RSI가 반등하고 종가가 SMA20을 상회합니다.")

    reasons = to_reasons(analysis, action="BUY")

    assert any("RSI가 반등하고 종가가 SMA20을 상회합니다." in r for r in reasons)
