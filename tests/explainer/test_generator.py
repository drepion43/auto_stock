from auto_stock.explainer.generator import generate_explanation
from auto_stock.risk_sizing.models import SizingSuggestion
from auto_stock.rule_engine.models import Candidate


def _candidate(action="BUY", ticker="005930", market="KRX", reasons=None):
    return Candidate(ticker=ticker, market=market, action=action, reasons=reasons or ["RSI(14)=25.0 과매도 구간 진입"])


def _sizing(
    ticker="005930",
    market="KRX",
    action="BUY",
    suggested_quantity=None,
    suggested_allocation_pct=None,
    stop_loss_price=None,
    take_profit_price=None,
    limit_check="NOT_APPLICABLE",
    notes=None,
):
    return SizingSuggestion(
        ticker=ticker, market=market, action=action,
        suggested_quantity=suggested_quantity, suggested_allocation_pct=suggested_allocation_pct,
        stop_loss_price=stop_loss_price, take_profit_price=take_profit_price,
        limit_check=limit_check, notes=notes or [],
    )


def test_buy_with_pass_sizing_includes_reasons_quantity_and_stop_take():
    candidate = _candidate("BUY")
    sizing = _sizing(
        suggested_quantity=50, suggested_allocation_pct=0.05,
        stop_loss_price=68000.0, take_profit_price=75000.0,
        limit_check="PASS",
    )

    explanation = generate_explanation(candidate, sizing)

    assert explanation.ticker == "005930"
    assert explanation.action == "BUY"
    assert "RSI(14)=25.0 과매도 구간 진입" in explanation.summary
    assert "50주" in explanation.summary
    assert "5.0%" in explanation.summary
    assert "68,000" in explanation.summary
    assert "75,000" in explanation.summary


def test_sell_candidate_includes_reasons_and_not_applicable_note_without_sizing_numbers():
    candidate = _candidate("SELL", reasons=["RSI(14)=82.0 과매수 구간 진입"])
    sizing = _sizing(
        action="SELL", limit_check="NOT_APPLICABLE",
        notes=["매도 후보는 기존 보유분 청산 개념이라 신규 매수 사이징 대상이 아닙니다."],
    )

    explanation = generate_explanation(candidate, sizing)

    assert "RSI(14)=82.0 과매수 구간 진입" in explanation.summary
    assert "매도 후보는" in explanation.summary
    assert "참고용 매수 제안" not in explanation.summary
    assert "ATR 기반" not in explanation.summary


def test_exceeds_max_positions_warning_note_is_included():
    candidate = _candidate("BUY")
    sizing = _sizing(
        suggested_quantity=50, suggested_allocation_pct=0.05,
        stop_loss_price=68000.0, take_profit_price=75000.0,
        limit_check="EXCEEDS_MAX_POSITIONS",
        notes=["최대 동시 보유 종목 수(10) 한도를 초과합니다."],
    )

    explanation = generate_explanation(candidate, sizing)

    assert "최대 동시 보유 종목 수(10) 한도를 초과합니다." in explanation.summary


def test_extra_reasons_are_combined_with_candidate_reasons():
    candidate = _candidate("BUY", reasons=["RSI(14)=25.0 과매도 구간 진입"])
    sizing = _sizing(limit_check="NOT_APPLICABLE", notes=["사유"])

    explanation = generate_explanation(candidate, sizing, extra_reasons=["뉴스: 실적 서프라이즈 기사 확인"])

    assert "RSI(14)=25.0 과매도 구간 진입" in explanation.summary
    assert "뉴스: 실적 서프라이즈 기사 확인" in explanation.summary
