from datetime import date, timedelta

import pytest

from auto_stock.data.models import OHLCVRecord
from auto_stock.risk_sizing.models import AccountState
from auto_stock.risk_sizing.sizing import (
    BASELINE_ATR_PCT,
    MAX_CONCURRENT_POSITIONS,
    MAX_POSITION_PCT,
    MAX_TOTAL_EXPOSURE_PCT,
    STOP_LOSS_ATR_MULT,
    TAKE_PROFIT_ATR_MULT,
    suggest_position,
)
from auto_stock.rule_engine.models import Candidate


def _records(n=20, ticker="005930", market="KRX", close=100.0):
    start = date(2026, 1, 1)
    return [
        OHLCVRecord(
            ticker=ticker, market=market, date=start + timedelta(days=i),
            open=close, high=close + 1, low=close - 1, close=close, volume=1000,
        )
        for i in range(n)
    ]


def _candidate(action="BUY", ticker="005930", market="KRX"):
    return Candidate(ticker=ticker, market=market, action=action, reasons=["테스트 근거"])


def _account(equity=10_000_000.0, held_tickers=frozenset(), total_exposure_pct=0.0):
    return AccountState(equity=equity, held_tickers=held_tickers, total_exposure_pct=total_exposure_pct)


def test_low_volatility_allocation_capped_at_max_position_pct(mocker):
    records = _records(close=100.0)
    mocker.patch("auto_stock.risk_sizing.sizing.atr", return_value=[None] * 19 + [1.0])  # atr_pct=0.01 < baseline

    result = suggest_position(_candidate("BUY"), records, _account())

    assert result.suggested_allocation_pct == pytest.approx(MAX_POSITION_PCT)
    assert result.suggested_quantity == 5000  # 10,000,000 * 0.05 / 100


def test_high_volatility_allocation_scaled_down(mocker):
    records = _records(close=100.0)
    mocker.patch("auto_stock.risk_sizing.sizing.atr", return_value=[None] * 19 + [4.0])  # atr_pct=0.04

    result = suggest_position(_candidate("BUY"), records, _account())

    expected = MAX_POSITION_PCT * BASELINE_ATR_PCT / 0.04
    assert result.suggested_allocation_pct == pytest.approx(expected)
    assert result.suggested_allocation_pct < MAX_POSITION_PCT


def test_stop_loss_and_take_profit_computed_from_atr(mocker):
    records = _records(close=100.0)
    mocker.patch("auto_stock.risk_sizing.sizing.atr", return_value=[None] * 19 + [2.0])

    result = suggest_position(_candidate("BUY"), records, _account())

    assert result.stop_loss_price == pytest.approx(100.0 - STOP_LOSS_ATR_MULT * 2.0)
    assert result.take_profit_price == pytest.approx(100.0 + TAKE_PROFIT_ATR_MULT * 2.0)


def test_exceeds_max_positions_when_new_ticker_and_already_at_cap(mocker):
    records = _records(close=100.0)
    mocker.patch("auto_stock.risk_sizing.sizing.atr", return_value=[None] * 19 + [1.0])
    held = frozenset(f"T{i}" for i in range(MAX_CONCURRENT_POSITIONS))
    account = _account(held_tickers=held, total_exposure_pct=0.1)

    result = suggest_position(_candidate("BUY", ticker="005930"), records, account)

    assert result.limit_check == "EXCEEDS_MAX_POSITIONS"


def test_exceeds_exposure_cap_when_adding_would_exceed_limit(mocker):
    records = _records(close=100.0)
    mocker.patch("auto_stock.risk_sizing.sizing.atr", return_value=[None] * 19 + [1.0])  # allocation -> 0.05
    account = _account(held_tickers=frozenset({"T1", "T2"}), total_exposure_pct=0.48)

    result = suggest_position(_candidate("BUY"), records, account)

    assert result.limit_check == "EXCEEDS_EXPOSURE_CAP"


def test_pass_when_within_all_limits(mocker):
    records = _records(close=100.0)
    mocker.patch("auto_stock.risk_sizing.sizing.atr", return_value=[None] * 19 + [1.0])
    account = _account(held_tickers=frozenset(), total_exposure_pct=0.1)

    result = suggest_position(_candidate("BUY"), records, account)

    assert result.limit_check == "PASS"


def test_sell_candidate_produces_no_sizing():
    records = _records(close=100.0)

    result = suggest_position(_candidate("SELL"), records, _account())

    assert result.suggested_quantity is None
    assert result.suggested_allocation_pct is None
    assert result.stop_loss_price is None
    assert result.take_profit_price is None
    assert result.limit_check == "NOT_APPLICABLE"
    assert result.notes


def test_missing_atr_produces_no_sizing(mocker):
    records = _records(n=2, close=100.0)
    mocker.patch("auto_stock.risk_sizing.sizing.atr", return_value=[None, None])

    result = suggest_position(_candidate("BUY"), records, _account())

    assert result.suggested_quantity is None
    assert result.suggested_allocation_pct is None
    assert result.limit_check == "NOT_APPLICABLE"
    assert result.notes


def test_non_positive_close_produces_no_sizing(mocker):
    records = _records(close=0.0)
    mocker.patch("auto_stock.risk_sizing.sizing.atr", return_value=[None] * 19 + [1.0])

    result = suggest_position(_candidate("BUY"), records, _account())

    assert result.suggested_quantity is None
    assert result.suggested_allocation_pct is None
    assert result.stop_loss_price is None
    assert result.take_profit_price is None
    assert result.limit_check == "NOT_APPLICABLE"
    assert result.notes


def test_stop_loss_price_is_clamped_above_zero_when_atr_exceeds_close(mocker):
    records = _records(close=500.0)
    mocker.patch("auto_stock.risk_sizing.sizing.atr", return_value=[None] * 19 + [400.0])  # 500 - 1.5*400 < 0

    result = suggest_position(_candidate("BUY"), records, _account())

    assert result.stop_loss_price > 0
