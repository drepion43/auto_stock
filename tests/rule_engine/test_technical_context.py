from datetime import date, timedelta

from auto_stock.data.models import OHLCVRecord
from auto_stock.rule_engine import technical_context


def _records(n=5, ticker="005930", market="KOSPI"):
    start = date(2026, 1, 1)
    return [
        OHLCVRecord(
            ticker=ticker, market=market, date=start + timedelta(days=i),
            open=100.0, high=101.0, low=99.0, close=100.0, volume=1000,
        )
        for i in range(n)
    ]


def test_empty_records_returns_none():
    assert technical_context.build_technical_context([]) is None


def test_insufficient_warmup_returns_none(mocker):
    records = _records(5)
    mocker.patch("auto_stock.rule_engine.technical_context.adx", return_value=([None] * 5, [None] * 5, [None] * 5))
    mocker.patch("auto_stock.rule_engine.technical_context.bbands", return_value=([None] * 5, [None] * 5, [None] * 5, [None] * 5))
    mocker.patch("auto_stock.rule_engine.technical_context.stoch", return_value=([None] * 5, [None] * 5))
    mocker.patch("auto_stock.rule_engine.technical_context.obv", return_value=[None] * 5)
    mocker.patch("auto_stock.rule_engine.technical_context.aroon", return_value=([None] * 5, [None] * 5, [None] * 5))
    mocker.patch("auto_stock.rule_engine.technical_context.squeeze", return_value=[None] * 5)
    mocker.patch("auto_stock.rule_engine.technical_context.rsi", return_value=[None] * 5)
    mocker.patch("auto_stock.rule_engine.technical_context.rsi_divergence", return_value=[None] * 5)

    assert technical_context.build_technical_context(records) is None


def test_builds_context_when_all_indicators_available(mocker):
    records = _records(10, ticker="005930", market="KOSPI")
    mocker.patch("auto_stock.rule_engine.technical_context.adx", return_value=([None] * 9 + [25.0], [None] * 9 + [30.0], [None] * 9 + [15.0]))
    mocker.patch(
        "auto_stock.rule_engine.technical_context.bbands",
        return_value=([None] * 9 + [0.8], [None] * 9 + [5.0], [None] * 9 + [95.0], [None] * 9 + [105.0]),
    )
    mocker.patch("auto_stock.rule_engine.technical_context.stoch", return_value=([None] * 9 + [70.0], [None] * 9 + [65.0]))
    mocker.patch("auto_stock.rule_engine.technical_context.obv", return_value=[100.0, 100.0, 100.0, 100.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0])
    mocker.patch(
        "auto_stock.rule_engine.technical_context.aroon",
        return_value=([None] * 9 + [100.0], [None] * 9 + [0.0], [None] * 9 + [100.0]),
    )
    mocker.patch("auto_stock.rule_engine.technical_context.squeeze", return_value=[None] * 9 + [0.0])
    mocker.patch("auto_stock.rule_engine.technical_context.rsi", return_value=[None] * 9 + [55.0])
    mocker.patch("auto_stock.rule_engine.technical_context.rsi_divergence", return_value=[None] * 9 + ["bullish"])

    result = technical_context.build_technical_context(records)

    assert result is not None
    assert result.ticker == "005930"
    assert result.market == "KOSPI"
    assert result.adx == 25.0
    assert result.plus_di == 30.0
    assert result.minus_di == 15.0
    assert result.bb_percent_b == 0.8
    assert result.bb_bandwidth == 5.0
    assert result.stoch_k == 70.0
    assert result.stoch_d == 65.0
    assert result.obv_trend == "RISING"
    assert result.aroon_up == 100.0
    assert result.aroon_down == 0.0
    assert result.aroon_oscillator == 100.0
    assert result.squeeze_on is False
    assert result.rsi_divergence == "bullish"


def test_obv_trend_falling_when_recent_values_decline(mocker):
    records = _records(10)
    mocker.patch("auto_stock.rule_engine.technical_context.adx", return_value=([None] * 9 + [25.0], [None] * 9 + [30.0], [None] * 9 + [15.0]))
    mocker.patch(
        "auto_stock.rule_engine.technical_context.bbands",
        return_value=([None] * 9 + [0.5], [None] * 9 + [5.0], [None] * 9 + [95.0], [None] * 9 + [105.0]),
    )
    mocker.patch("auto_stock.rule_engine.technical_context.stoch", return_value=([None] * 9 + [50.0], [None] * 9 + [50.0]))
    mocker.patch("auto_stock.rule_engine.technical_context.obv", return_value=[150.0, 150.0, 150.0, 150.0, 150.0, 140.0, 130.0, 120.0, 110.0, 100.0])
    mocker.patch(
        "auto_stock.rule_engine.technical_context.aroon",
        return_value=([None] * 9 + [50.0], [None] * 9 + [50.0], [None] * 9 + [0.0]),
    )
    mocker.patch("auto_stock.rule_engine.technical_context.squeeze", return_value=[None] * 9 + [1.0])
    mocker.patch("auto_stock.rule_engine.technical_context.rsi", return_value=[None] * 9 + [50.0])
    mocker.patch("auto_stock.rule_engine.technical_context.rsi_divergence", return_value=[None] * 9 + [None])

    result = technical_context.build_technical_context(records)

    assert result.obv_trend == "FALLING"
    assert result.squeeze_on is True
    assert result.rsi_divergence is None
