import math

import pytest

from auto_stock.rule_engine.indicators import (
    adx,
    aroon,
    atr,
    bbands,
    ema,
    macd,
    obv,
    rsi,
    rsi_divergence,
    sma,
    squeeze,
    stoch,
    volume_ratio,
)


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


def test_volume_ratio_matches_hand_computed_values():
    volumes = [100, 100, 100, 100, 200]

    result = volume_ratio(volumes, window=3)

    assert result[:2] == [None, None]
    assert result[2] == pytest.approx(1.0)  # avg(100,100,100)=100 -> 100/100
    assert result[3] == pytest.approx(1.0)  # avg(100,100,100)=100 -> 100/100
    assert result[4] == pytest.approx(1.5)  # avg(100,100,200)=133.33 -> 200/133.33


def test_volume_ratio_is_none_when_mean_volume_is_zero():
    volumes = [0, 0, 0, 5]

    result = volume_ratio(volumes, window=3)

    assert result[2] is None


# --- technical_context용 지표 7종 (rule_engine/technical_context.py가 조립) ---


def _uptrend(n=80):
    closes = [100.0 + i * 1.5 + 3 * math.sin(i / 3) for i in range(n)]
    return [c + 1.0 for c in closes], [c - 1.0 for c in closes], closes


def test_new_indicators_are_index_aligned_with_input():
    highs, lows, closes = _uptrend()
    n = len(closes)
    volumes = [1000 + (i % 5) * 100 for i in range(n)]

    for series in (*adx(highs, lows, closes), *bbands(closes), *stoch(highs, lows, closes),
                   obv(closes, volumes), *aroon(highs, lows), squeeze(highs, lows, closes)):
        assert len(series) == n


def test_new_indicators_are_all_none_when_input_too_short():
    highs, lows, closes = [101.0] * 5, [99.0] * 5, [100.0] * 5

    for series in (*adx(highs, lows, closes), *bbands(closes), *stoch(highs, lows, closes),
                   *aroon(highs, lows), squeeze(highs, lows, closes)):
        assert all(v is None for v in series)


def test_adx_plus_di_dominates_in_uptrend():
    highs, lows, closes = _uptrend()

    adx_values, plus_di, minus_di = adx(highs, lows, closes)

    assert adx_values[-1] is not None
    assert plus_di[-1] > minus_di[-1]


def test_bbands_returns_percent_b_bandwidth_lower_upper():
    _, _, closes = _uptrend()

    percent_b, bandwidth, lower, upper = bbands(closes)

    assert lower[-1] < upper[-1]
    assert bandwidth[-1] > 0
    assert percent_b[-1] > 0.5  # 상승 추세 끝은 밴드 중심보다 위


def test_stoch_values_stay_within_0_and_100():
    highs, lows, closes = _uptrend()

    k, d = stoch(highs, lows, closes)

    assert all(0.0 <= v <= 100.0 for v in k + d if v is not None)
    assert k[-1] is not None and d[-1] is not None


def test_obv_accumulates_volume_on_up_days():
    closes = [10.0, 11.0, 12.0, 13.0, 14.0]
    volumes = [100, 100, 100, 100, 100]

    result = obv(closes, volumes)

    assert result[-1] > result[1]


def test_aroon_up_is_100_on_a_fresh_high():
    highs, lows, _ = _uptrend()
    highs = highs[:-1] + [max(highs) + 10.0]

    up, down, osc = aroon(highs, lows)

    assert up[-1] == pytest.approx(100.0)
    assert osc[-1] == pytest.approx(up[-1] - down[-1])


def test_squeeze_is_flag_valued():
    highs, lows, closes = _uptrend()

    result = squeeze(highs, lows, closes)

    assert {v for v in result if v is not None} <= {0.0, 1.0}
    assert result[-1] is not None


def test_rsi_divergence_bullish_when_price_makes_lower_low_but_rsi_does_not():
    closes = [100.0, 95.0, 98.0, 99.0, 94.0]   # 마지막이 직전 구간 최저(95)보다 낮음
    rsi_values = [50.0, 25.0, 40.0, 45.0, 32.0]  # 그런데 RSI는 그때(25)보다 높음

    result = rsi_divergence(closes, rsi_values, lookback=4)

    assert result[-1] == "bullish"


def test_rsi_divergence_bearish_when_price_makes_higher_high_but_rsi_does_not():
    closes = [100.0, 105.0, 102.0, 101.0, 106.0]
    rsi_values = [50.0, 75.0, 60.0, 55.0, 68.0]

    result = rsi_divergence(closes, rsi_values, lookback=4)

    assert result[-1] == "bearish"


def test_rsi_divergence_none_without_new_extreme_or_enough_history():
    closes = [100.0, 95.0, 98.0, 99.0, 97.0]   # 새 고점/저점 아님
    rsi_values = [50.0, 25.0, 40.0, 45.0, 42.0]

    assert rsi_divergence(closes, rsi_values, lookback=4)[-1] is None
    assert rsi_divergence(closes, rsi_values, lookback=10) == [None] * 5


def test_rsi_divergence_skips_points_with_missing_rsi():
    closes = [100.0, 95.0, 98.0, 99.0, 94.0]
    rsi_values = [None, None, None, None, 32.0]

    assert rsi_divergence(closes, rsi_values, lookback=4)[-1] is None
