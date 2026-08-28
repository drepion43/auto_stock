import pytest

from auto_stock.llm_chart_analyst.models import ChartSnapshot
from auto_stock.llm_chart_analyst.snapshot import RECENT_BARS, build_snapshot
from auto_stock.ml_predictor.features import FEATURE_NAMES

from .conftest import make_records, random_walk_prices


def test_build_snapshot_returns_none_during_warmup_window():
    records = make_records(random_walk_prices(30))  # < SMA60 warmup requirement

    assert build_snapshot(records) is None


def test_build_snapshot_returns_snapshot_once_warmup_satisfied(warmed_up_records):
    snapshot = build_snapshot(warmed_up_records)

    assert isinstance(snapshot, ChartSnapshot)
    assert len(snapshot.bars) == RECENT_BARS


def test_build_snapshot_indicator_keys_match_feature_names(warmed_up_records):
    snapshot = build_snapshot(warmed_up_records)

    assert set(snapshot.indicators.keys()) == set(FEATURE_NAMES)


def test_build_snapshot_labels_come_from_latest_record(warmed_up_records):
    snapshot = build_snapshot(warmed_up_records)
    latest = warmed_up_records[-1]

    assert snapshot.ticker == latest.ticker
    assert snapshot.market == latest.market
    assert snapshot.date == latest.date


def test_build_snapshot_bar_offsets_run_from_recent_bars_minus_one_to_zero(warmed_up_records):
    snapshot = build_snapshot(warmed_up_records)

    offsets = [bar.offset for bar in snapshot.bars]
    assert offsets == list(range(RECENT_BARS - 1, -1, -1))
    assert offsets[-1] == 0  # last row is the most recent bar


def test_build_snapshot_first_bar_close_is_normalized_to_base_index(warmed_up_records):
    snapshot = build_snapshot(warmed_up_records)

    assert snapshot.bars[0].close == pytest.approx(100.0)


def test_build_snapshot_bars_are_scale_invariant():
    prices = random_walk_prices(80)
    records = make_records(prices)
    scaled_records = make_records([p * 1000 for p in prices])

    snapshot = build_snapshot(records)
    scaled_snapshot = build_snapshot(scaled_records)

    assert snapshot is not None
    assert scaled_snapshot is not None
    for bar, scaled_bar in zip(snapshot.bars, scaled_snapshot.bars):
        assert bar.open == pytest.approx(scaled_bar.open)
        assert bar.high == pytest.approx(scaled_bar.high)
        assert bar.low == pytest.approx(scaled_bar.low)
        assert bar.close == pytest.approx(scaled_bar.close)
        assert bar.volume_ratio == pytest.approx(scaled_bar.volume_ratio)


def test_build_snapshot_indicators_are_scale_invariant():
    prices = random_walk_prices(80)
    records = make_records(prices)
    scaled_records = make_records([p * 1000 for p in prices])

    snapshot = build_snapshot(records)
    scaled_snapshot = build_snapshot(scaled_records)

    for name in FEATURE_NAMES:
        assert snapshot.indicators[name] == pytest.approx(scaled_snapshot.indicators[name], abs=1e-6)


def test_build_snapshot_is_unchanged_when_future_bars_are_appended():
    """`ml_predictor`의 인과성 테스트와 같은 패턴: 미래 봉을 추가해도 과거 시점 스냅샷은 불변."""
    prices = random_walk_prices(90)
    records = make_records(prices)
    target_len = 70

    snapshot_before = build_snapshot(records[:target_len])

    extended_prices = prices + random_walk_prices(20, seed=99)
    extended_records = make_records(extended_prices)
    snapshot_after = build_snapshot(extended_records[:target_len])

    assert snapshot_before == snapshot_after


def test_build_snapshot_returns_none_when_base_close_is_zero():
    """window[0].close(정규화 기준값)가 0이면 ZeroDivisionError 대신 None을 반환한다.

    SMA60/RSI/ATR 등은 모두 최신 시점(index -1) 위주의 후행 윈도우라 이 0값이
    latest_feature_vector 자체를 None으로 만들지는 않는다 — build_snapshot이
    스스로 이 케이스를 가드해야 한다."""
    prices = random_walk_prices(80)
    prices[80 - RECENT_BARS] = 0.0  # window[0].close가 되는 지점
    records = make_records(prices)

    assert build_snapshot(records) is None


def test_build_snapshot_respects_custom_recent_bars(warmed_up_records):
    snapshot = build_snapshot(warmed_up_records, recent_bars=10)

    assert len(snapshot.bars) == 10
    assert [bar.offset for bar in snapshot.bars] == list(range(9, -1, -1))
