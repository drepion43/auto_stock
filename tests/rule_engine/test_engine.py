from datetime import date, timedelta

from auto_stock.data.models import OHLCVRecord
from auto_stock.rule_engine import engine


def _records(n=5, ticker="005930", market="KRX"):
    start = date(2026, 1, 1)
    return [
        OHLCVRecord(
            ticker=ticker, market=market, date=start + timedelta(days=i),
            open=100.0, high=100.0, low=100.0, close=100.0, volume=1000,
        )
        for i in range(n)
    ]


def test_rsi_oversold_produces_buy_candidate(mocker):
    records = _records(5)
    mocker.patch("auto_stock.rule_engine.engine.rsi", return_value=[None, None, None, None, 25.0])
    mocker.patch("auto_stock.rule_engine.engine.sma", return_value=[None] * 5)

    candidates = engine.generate_candidates(records)

    assert len(candidates) == 1
    assert candidates[0].action == "BUY"
    assert any("RSI" in reason for reason in candidates[0].reasons)


def test_rsi_overbought_produces_sell_candidate(mocker):
    records = _records(5)
    mocker.patch("auto_stock.rule_engine.engine.rsi", return_value=[None, None, None, None, 82.0])
    mocker.patch("auto_stock.rule_engine.engine.sma", return_value=[None] * 5)

    candidates = engine.generate_candidates(records)

    assert candidates[0].action == "SELL"


def test_golden_cross_produces_buy_candidate(mocker):
    records = _records(5)
    mocker.patch("auto_stock.rule_engine.engine.rsi", return_value=[None] * 5)

    def fake_sma(closes, window):
        if window == 20:
            return [None, None, None, 9.0, 11.0]  # crosses above at last step
        return [None, None, None, 10.0, 10.0]

    mocker.patch("auto_stock.rule_engine.engine.sma", side_effect=fake_sma)

    candidates = engine.generate_candidates(records)

    assert len(candidates) == 1
    assert candidates[0].action == "BUY"
    assert any("골든크로스" in reason for reason in candidates[0].reasons)


def test_dead_cross_produces_sell_candidate(mocker):
    records = _records(5)
    mocker.patch("auto_stock.rule_engine.engine.rsi", return_value=[None] * 5)

    def fake_sma(closes, window):
        if window == 20:
            return [None, None, None, 11.0, 9.0]  # crosses below at last step
        return [None, None, None, 10.0, 10.0]

    mocker.patch("auto_stock.rule_engine.engine.sma", side_effect=fake_sma)

    candidates = engine.generate_candidates(records)

    assert candidates[0].action == "SELL"
    assert any("데드크로스" in reason for reason in candidates[0].reasons)


def test_no_signal_produces_no_candidates(mocker):
    records = _records(5)
    mocker.patch("auto_stock.rule_engine.engine.rsi", return_value=[None] * 5)
    mocker.patch("auto_stock.rule_engine.engine.sma", return_value=[None] * 5)

    assert engine.generate_candidates(records) == []


def test_conflicting_signals_produce_no_candidate(mocker):
    records = _records(5)
    mocker.patch("auto_stock.rule_engine.engine.rsi", return_value=[None, None, None, None, 25.0])

    def fake_sma(closes, window):
        if window == 20:
            return [None, None, None, 11.0, 9.0]  # dead cross => SELL, conflicts with RSI BUY
        return [None, None, None, 10.0, 10.0]

    mocker.patch("auto_stock.rule_engine.engine.sma", side_effect=fake_sma)

    assert engine.generate_candidates(records) == []


def test_too_few_records_produces_no_candidates():
    assert engine.generate_candidates(_records(1)) == []


def test_empty_records_produces_no_candidates():
    assert engine.generate_candidates([]) == []
