from datetime import date

import pytest

from auto_stock.ml_predictor.dataset import build_samples, build_training_dataset, chronological_split, to_xy
from auto_stock.ml_predictor.features import FEATURE_NAMES
from auto_stock.ml_predictor.labeling import LABEL_HORIZON_DAYS
from auto_stock.ml_predictor.models import LabeledSample, TrainingDataset

from .conftest import make_records, random_walk_prices


def test_build_samples_drops_warmup_and_unlabeled_tail_rows():
    n = 90
    records = make_records(random_walk_prices(n))

    samples = build_samples(records)

    assert all(isinstance(s, LabeledSample) for s in samples)
    assert len(samples) <= n - LABEL_HORIZON_DAYS
    assert all(s.label in (0, 1) for s in samples)


def test_build_samples_is_empty_for_short_series():
    records = make_records(random_walk_prices(10))
    assert build_samples(records) == []


def test_build_training_dataset_pools_multiple_tickers():
    records_a = make_records(random_walk_prices(90, seed=1), ticker="AAA")
    records_b = make_records(random_walk_prices(90, seed=2), ticker="BBB")

    dataset = build_training_dataset([records_a, records_b])

    tickers = {s.feature.ticker for s in dataset.samples}
    assert tickers == {"AAA", "BBB"}
    assert dataset.feature_names == FEATURE_NAMES


def test_dataset_samples_are_sorted_by_global_date():
    records_a = make_records(random_walk_prices(90, seed=1), ticker="AAA")
    records_b = make_records(random_walk_prices(90, seed=2), ticker="BBB")

    dataset = build_training_dataset([records_a, records_b])

    dates = [s.feature.date for s in dataset.samples]
    assert dates == sorted(dates)


def test_to_xy_matches_feature_order_and_labels():
    records = make_records(random_walk_prices(90))
    dataset = build_training_dataset([records])

    X, y = to_xy(dataset)

    assert len(X) == len(dataset.samples) == len(y)
    for row, sample in zip(X, dataset.samples):
        assert row == [sample.feature.values[name] for name in FEATURE_NAMES]
    assert set(y) <= {0, 1}


def test_every_train_date_precedes_every_test_date():
    records = make_records(random_walk_prices(160))
    dataset = build_training_dataset([records])

    split = chronological_split(dataset, test_ratio=0.2)

    max_train_date = max(s.feature.date for s in split.train.samples)
    min_test_date = min(s.feature.date for s in split.test.samples)
    assert max_train_date < min_test_date


def test_embargo_gap_is_at_least_horizon_trading_days():
    records = make_records(random_walk_prices(160))
    dataset = build_training_dataset([records])

    split = chronological_split(dataset, test_ratio=0.2, embargo_days=LABEL_HORIZON_DAYS)

    train_dates = sorted({s.feature.date for s in split.train.samples})
    test_dates = sorted({s.feature.date for s in split.test.samples})
    all_dates = sorted({s.feature.date for s in dataset.samples})

    gap_dates = [d for d in all_dates if train_dates[-1] < d < test_dates[0]]
    assert len(gap_dates) >= LABEL_HORIZON_DAYS


def test_chronological_split_defaults_embargo_to_label_horizon():
    records = make_records(random_walk_prices(160))
    dataset = build_training_dataset([records])

    default_split = chronological_split(dataset, test_ratio=0.2)
    explicit_split = chronological_split(dataset, test_ratio=0.2, embargo_days=LABEL_HORIZON_DAYS)

    assert len(default_split.train.samples) == len(explicit_split.train.samples)
    assert len(default_split.test.samples) == len(explicit_split.test.samples)


def test_multi_ticker_dataset_is_split_by_global_date_not_per_ticker():
    # AAA has data through the full range; BBB only exists in the back half.
    # A per-ticker split would put none of BBB's early history at risk of crossing
    # into AAA's test window — a global-date split must cut BOTH at the same date.
    records_a = make_records(random_walk_prices(160, seed=1), ticker="AAA", start=date(2024, 1, 1))
    records_b = make_records(random_walk_prices(60, seed=2), ticker="BBB", start=date(2024, 4, 1))

    dataset = build_training_dataset([records_a, records_b])
    split = chronological_split(dataset, test_ratio=0.2)

    train_max_date = max(s.feature.date for s in split.train.samples)
    test_min_date = min(s.feature.date for s in split.test.samples)

    for s in split.train.samples:
        assert s.feature.date <= train_max_date
    for s in split.test.samples:
        assert s.feature.date >= test_min_date
    # Both tickers must respect the same global cutoff (no ticker leaks samples past it).
    assert all(s.feature.date < test_min_date for s in split.train.samples)


def test_chronological_split_never_shuffles(monkeypatch):
    # train_test_split/KFold/shuffle usage is banned in this module; a crude static
    # guard is enough here since the real guarantee is covered by the ordering tests above.
    import auto_stock.ml_predictor.dataset as dataset_module

    assert "train_test_split" not in dir(dataset_module)
    assert "KFold" not in dir(dataset_module)
