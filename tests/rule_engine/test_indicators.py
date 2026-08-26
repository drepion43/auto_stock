import pytest

from auto_stock.rule_engine.indicators import atr, ema, macd, rsi, sma


def test_sma_matches_hand_computed_values():
    closes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    result = sma(closes, window=3)

    assert result[0] is None
    assert result[1] is None
    assert result[2:] == pytest.approx([2.0, 3.0, 4.0, 5.0])


def test_sma_all_none_when_series_shorter_than_window():
    result = sma([1.0, 2.0], window=3)
    assert all(v is None for v in result)


def test_rsi_is_100_for_monotonically_increasing_prices():
    closes = [100.0 + i for i in range(30)]  # strictly increasing, no losses

    result = rsi(closes, period=14)

    assert result[0] is None
    assert result[-1] == 100.0


def test_rsi_is_0_for_monotonically_decreasing_prices():
    closes = [200.0 - i for i in range(30)]  # strictly decreasing, no gains

    result = rsi(closes, period=14)

    assert result[0] is None
    assert result[-1] == 0.0


def test_ema_seeds_with_sma_then_smooths():
    closes = [1.0, 2.0, 3.0, 4.0, 5.0]

    result = ema(closes, period=3)

    assert result[0] is None
    assert result[1] is None
    assert result[2] == 2.0  # SMA(1,2,3) seed
    assert result[3] is not None and result[3] > result[2]


def test_macd_line_is_positive_for_sustained_uptrend():
    closes = [100.0 + i * 0.5 for i in range(60)]

    macd_line, signal_line, histogram = macd(closes, fast=12, slow=26, signal=9)

    assert macd_line[-1] is not None
    assert macd_line[-1] > 0
    assert signal_line[-1] is not None
    assert histogram[-1] is not None


def test_macd_line_is_negative_for_sustained_downtrend():
    closes = [200.0 - i * 0.5 for i in range(60)]

    macd_line, _signal_line, _histogram = macd(closes, fast=12, slow=26, signal=9)

    assert macd_line[-1] is not None
    assert macd_line[-1] < 0


def test_atr_is_none_when_series_shorter_than_period():
    highs = [101.0, 102.0]
    lows = [99.0, 98.0]
    closes = [100.0, 100.0]

    result = atr(highs, lows, closes, period=14)

    assert all(v is None for v in result)


def test_atr_is_higher_for_wider_high_low_ranges():
    n = 30
    calm_closes = [100.0] * n
    calm_highs = [c + 0.5 for c in calm_closes]
    calm_lows = [c - 0.5 for c in calm_closes]

    volatile_closes = [100.0] * n
    volatile_highs = [c + 5.0 for c in volatile_closes]
    volatile_lows = [c - 5.0 for c in volatile_closes]

    calm_atr = atr(calm_highs, calm_lows, calm_closes, period=14)
    volatile_atr = atr(volatile_highs, volatile_lows, volatile_closes, period=14)

    assert calm_atr[-1] is not None
    assert volatile_atr[-1] is not None
    assert volatile_atr[-1] > calm_atr[-1]
