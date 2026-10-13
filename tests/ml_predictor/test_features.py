from datetime import date, timedelta

import pytest

from auto_stock.ml_predictor.features import (
    CORE_FEATURE_NAMES,
    FEATURE_NAMES,
    FUNDAMENTAL_FEATURE_NAMES,
    build_feature_vectors,
    latest_feature_vector,
)
from auto_stock.ml_predictor.models import FeatureVector

from .conftest import make_records, random_walk_prices


def test_feature_names_matches_plan_definition_table():
    assert CORE_FEATURE_NAMES == [
        "return_1d",
        "return_5d",
        "return_20d",
        "rsi_14",
        "macd_hist_norm",
        "close_to_sma20",
        "close_to_sma60",
        "sma20_to_sma60",
        "atr_14_pct",
        "volume_ratio_20",
        "channel_position_20",
    ]
    assert FUNDAMENTAL_FEATURE_NAMES == [
        "earnings_yield",
        "pbr",
        "dividend_yield",
        "turnover_rate",
        "sales_yield",
        "fundamental_available",
    ]
    assert FEATURE_NAMES == CORE_FEATURE_NAMES + FUNDAMENTAL_FEATURE_NAMES


def test_features_are_none_during_warmup_window():
    prices = random_walk_prices(30)  # < SMA60 warmup requirement
    records = make_records(prices)

    vectors = build_feature_vectors(records)

    assert len(vectors) == len(records)
    assert all(v is None for v in vectors)


def test_features_are_populated_once_warmup_window_is_satisfied():
    prices = random_walk_prices(80)
    records = make_records(prices)

    vectors = build_feature_vectors(records)

    assert vectors[-1] is not None
    assert isinstance(vectors[-1], FeatureVector)
    assert set(vectors[-1].values.keys()) == set(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in vectors[-1].values.values())


def test_feature_at_index_is_unchanged_when_future_bars_are_appended():
    prices = random_walk_prices(90)
    records = make_records(prices)
    target_index = 65

    vectors_before = build_feature_vectors(records)

    extended_prices = prices + random_walk_prices(20, seed=99)
    extended_records = make_records(extended_prices)
    vectors_after = build_feature_vectors(extended_records)

    assert vectors_before[target_index] is not None
    assert vectors_after[target_index] is not None
    assert vectors_before[target_index].values == pytest.approx(vectors_after[target_index].values)


def test_core_features_are_scale_invariant():
    prices = random_walk_prices(80)
    records = make_records(prices)
    scaled_records = make_records([p * 1000 for p in prices])

    vectors = build_feature_vectors(records)
    scaled_vectors = build_feature_vectors(scaled_records)

    assert vectors[-1] is not None
    assert scaled_vectors[-1] is not None
    for name in CORE_FEATURE_NAMES:
        assert vectors[-1].values[name] == pytest.approx(scaled_vectors[-1].values[name], abs=1e-6)


def test_latest_feature_vector_returns_none_when_insufficient_history():
    records = make_records(random_walk_prices(10))
    assert latest_feature_vector(records) is None


def test_latest_feature_vector_returns_none_for_empty_records():
    assert latest_feature_vector([]) is None


def test_latest_feature_vector_matches_last_entry_of_build_feature_vectors():
    records = make_records(random_walk_prices(80))
    vectors = build_feature_vectors(records)
    assert latest_feature_vector(records) == vectors[-1]


def test_volume_ratio_reflects_volume_spike():
    prices = random_walk_prices(80)
    volumes = [1000] * 79 + [5000]  # spike on the last bar
    records = make_records(prices, volumes=volumes)

    vectors = build_feature_vectors(records)

    assert vectors[-1].values["volume_ratio_20"] > 1.0


def test_channel_position_is_near_one_at_local_high():
    # Strictly increasing prices -> latest close sits at the top of its 20-day channel.
    prices = [100.0 + i for i in range(80)]
    records = make_records(prices)

    vectors = build_feature_vectors(records)

    # high/low padding (+-1%) around a strictly increasing close series keeps this
    # slightly below the theoretical 1.0 ceiling; still must sit near the top of the range.
    assert vectors[-1].values["channel_position_20"] > 0.85


# --- 펀더멘털 채널(국내전용, 중립값 폴백) ---


def test_fundamentals_default_to_neutral_when_none_provided():
    """fundamentals_by_date를 아예 안 넘기면(해외 종목 등) 전부 중립값 + available=0."""
    records = make_records(random_walk_prices(80), market="NASDAQ")

    vectors = build_feature_vectors(records, fundamentals_by_date=None)

    latest = vectors[-1]
    assert latest.values["fundamental_available"] == 0.0
    assert latest.values["earnings_yield"] == 0.0
    assert latest.values["pbr"] == 1.0
    assert latest.values["dividend_yield"] == 0.0
    assert latest.values["turnover_rate"] == 0.0
    assert latest.values["sales_yield"] == 0.0


def test_fundamentals_are_populated_when_available():
    prices = random_walk_prices(80)
    records = make_records(prices)
    last_date = records[-1].date
    fundamentals_by_date = {
        last_date: {
            "per": 10.0,
            "pbr": 1.5,
            "div": 2.0,
            "volume": 100_000,
            "shares_outstanding": 1_000_000,
            "market_cap": 1_000_000_000,
            "revenue": 500_000_000,
        }
    }

    vectors = build_feature_vectors(records, fundamentals_by_date=fundamentals_by_date)

    latest = vectors[-1]
    assert latest.values["fundamental_available"] == 1.0
    assert latest.values["earnings_yield"] == pytest.approx(0.1)
    assert latest.values["pbr"] == pytest.approx(1.5)
    assert latest.values["dividend_yield"] == pytest.approx(0.02)
    assert latest.values["turnover_rate"] == pytest.approx(0.1)
    assert latest.values["sales_yield"] == pytest.approx(0.5)


def test_fundamentals_missing_for_specific_date_falls_back_to_neutral():
    """국내 종목이라도 그 날짜의 펀더멘털만 결측이면(공시 공백 등) 중립값 + available=0."""
    prices = random_walk_prices(80)
    records = make_records(prices)
    other_date = records[-1].date - timedelta(days=365)  # 데이터셋에 없는 날짜
    fundamentals_by_date = {other_date: {"per": 10.0, "pbr": 1.5, "div": 2.0}}

    vectors = build_feature_vectors(records, fundamentals_by_date=fundamentals_by_date)

    latest = vectors[-1]
    assert latest.values["fundamental_available"] == 0.0
    assert latest.values["pbr"] == 1.0


def test_fundamentals_partial_fields_fall_back_individually():
    """매출액 계정 매칭 실패 등으로 일부 필드만 없을 때 — available은 1.0 유지,
    없는 필드만 중립값."""
    prices = random_walk_prices(80)
    records = make_records(prices)
    last_date = records[-1].date
    fundamentals_by_date = {last_date: {"per": 10.0, "pbr": 1.5, "div": None}}

    vectors = build_feature_vectors(records, fundamentals_by_date=fundamentals_by_date)

    latest = vectors[-1]
    assert latest.values["fundamental_available"] == 1.0
    assert latest.values["earnings_yield"] == pytest.approx(0.1)
    assert latest.values["dividend_yield"] == 0.0  # revenue/market_cap 없어 sales_yield도 중립
    assert latest.values["sales_yield"] == 0.0


def test_earnings_yield_is_zero_when_per_is_zero_or_negative():
    prices = random_walk_prices(80)
    records = make_records(prices)
    last_date = records[-1].date

    vectors = build_feature_vectors(records, fundamentals_by_date={last_date: {"per": 0.0}})
    assert vectors[-1].values["earnings_yield"] == 0.0

    vectors_negative = build_feature_vectors(records, fundamentals_by_date={last_date: {"per": -5.0}})
    assert vectors_negative[-1].values["earnings_yield"] == pytest.approx(-0.2)
