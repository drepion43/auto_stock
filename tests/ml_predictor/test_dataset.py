from datetime import date

import pytest

from auto_stock.ml_predictor.dataset import (
    build_sequence_dataset,
    build_ticker_sequence,
    chronological_split,
    collate_sequences,
    to_xy_flat,
    train_val_test_split,
)
from auto_stock.ml_predictor.features import FEATURE_NAMES
from auto_stock.ml_predictor.labeling import LABEL_HORIZON_DAYS
from auto_stock.ml_predictor.models import TickerSequence

from .conftest import make_records, random_walk_prices


def test_build_ticker_sequence_drops_warmup_and_unlabeled_tail_row():
    n = 90
    records = make_records(random_walk_prices(n))

    sequence = build_ticker_sequence(records)

    assert isinstance(sequence, TickerSequence)
    assert len(sequence.samples) <= n - LABEL_HORIZON_DAYS
    assert all(isinstance(s.label, float) for s in sequence.samples)


def test_build_ticker_sequence_is_none_for_short_series():
    records = make_records(random_walk_prices(10))
    assert build_ticker_sequence(records) is None


def test_build_sequence_dataset_groups_multiple_tickers():
    records_a = make_records(random_walk_prices(90, seed=1), ticker="AAA")
    records_b = make_records(random_walk_prices(90, seed=2), ticker="BBB")

    dataset = build_sequence_dataset([records_a, records_b])

    tickers = {seq.ticker for seq in dataset.sequences}
    assert tickers == {"AAA", "BBB"}
    assert dataset.feature_names == FEATURE_NAMES


def test_to_xy_flat_matches_feature_order_and_labels():
    records = make_records(random_walk_prices(90))
    dataset = build_sequence_dataset([records])

    X, y = to_xy_flat(dataset)

    total_samples = sum(len(seq.samples) for seq in dataset.sequences)
    assert len(X) == total_samples == len(y)
    for row in X:
        assert len(row) == len(FEATURE_NAMES)


def test_every_train_date_precedes_every_test_date():
    records = make_records(random_walk_prices(160))
    dataset = build_sequence_dataset([records])

    split = chronological_split(dataset, test_ratio=0.2)

    max_train_date = max(s.feature.date for seq in split.train.sequences for s in seq.samples)
    min_test_date = min(s.feature.date for seq in split.test.sequences for s in seq.samples)
    assert max_train_date < min_test_date


def test_embargo_gap_is_at_least_horizon_trading_days():
    records = make_records(random_walk_prices(160))
    dataset = build_sequence_dataset([records])

    split = chronological_split(dataset, test_ratio=0.2, embargo_days=LABEL_HORIZON_DAYS)

    train_dates = sorted({s.feature.date for seq in split.train.sequences for s in seq.samples})
    test_dates = sorted({s.feature.date for seq in split.test.sequences for s in seq.samples})
    all_dates = sorted({s.feature.date for seq in dataset.sequences for s in seq.samples})

    gap_dates = [d for d in all_dates if train_dates[-1] < d < test_dates[0]]
    assert len(gap_dates) >= LABEL_HORIZON_DAYS


def test_chronological_split_defaults_embargo_to_label_horizon():
    records = make_records(random_walk_prices(160))
    dataset = build_sequence_dataset([records])

    default_split = chronological_split(dataset, test_ratio=0.2)
    explicit_split = chronological_split(dataset, test_ratio=0.2, embargo_days=LABEL_HORIZON_DAYS)

    default_train_count = sum(len(seq.samples) for seq in default_split.train.sequences)
    explicit_train_count = sum(len(seq.samples) for seq in explicit_split.train.sequences)
    assert default_train_count == explicit_train_count


def test_multi_ticker_dataset_is_split_by_global_date_not_per_ticker():
    # AAA has data through the full range; BBB only exists in the back half.
    # A per-ticker split would put none of BBB's early history at risk of crossing
    # into AAA's test window — a global-date split must cut BOTH at the same date.
    records_a = make_records(random_walk_prices(160, seed=1), ticker="AAA", start=date(2024, 1, 1))
    records_b = make_records(random_walk_prices(60, seed=2), ticker="BBB", start=date(2024, 4, 1))

    dataset = build_sequence_dataset([records_a, records_b])
    split = chronological_split(dataset, test_ratio=0.2)

    train_dates = {s.feature.date for seq in split.train.sequences for s in seq.samples}
    test_min_date = min(s.feature.date for seq in split.test.sequences for s in seq.samples)

    assert all(d < test_min_date for d in train_dates)


def _dates(dataset) -> list:
    return sorted({s.feature.date for seq in dataset.sequences for s in seq.samples})


def test_train_val_test_split_orders_windows_with_embargo_gaps():
    records = make_records(random_walk_prices(300))
    dataset = build_sequence_dataset([records])

    split = train_val_test_split(dataset, val_ratio=0.1, test_ratio=0.2)

    assert split.validation is not None
    train_dates, val_dates, test_dates = _dates(split.train), _dates(split.validation), _dates(split.test)
    assert train_dates and val_dates and test_dates
    assert train_dates[-1] < val_dates[0] and val_dates[-1] < test_dates[0]
    all_dates = _dates(dataset)
    assert len([d for d in all_dates if train_dates[-1] < d < val_dates[0]]) >= LABEL_HORIZON_DAYS
    assert len([d for d in all_dates if val_dates[-1] < d < test_dates[0]]) >= LABEL_HORIZON_DAYS


def test_train_val_test_split_keeps_test_window_identical_to_two_way_split():
    # Adding a validation window must not move the test window (keeps results comparable).
    records = make_records(random_walk_prices(300))
    dataset = build_sequence_dataset([records])

    three_way = train_val_test_split(dataset, val_ratio=0.1, test_ratio=0.2)
    two_way = chronological_split(dataset, test_ratio=0.2)

    assert _dates(three_way.test) == _dates(two_way.test)


def test_two_way_split_has_no_validation():
    records = make_records(random_walk_prices(160))
    dataset = build_sequence_dataset([records])

    assert chronological_split(dataset, test_ratio=0.2).validation is None


def test_chronological_split_never_shuffles():
    # train_test_split/KFold/shuffle usage is banned in this module; a crude static
    # guard is enough here since the real guarantee is covered by the ordering tests above.
    import auto_stock.ml_predictor.dataset as dataset_module

    assert "train_test_split" not in dir(dataset_module)
    assert "KFold" not in dir(dataset_module)


# --- collate_sequences (가변 길이 배치 패딩/마스킹, §B) ---


def test_collate_sequences_pads_shorter_sequences_with_mask():
    records_short = make_records(random_walk_prices(70, seed=1), ticker="SHORT")
    records_long = make_records(random_walk_prices(90, seed=2), ticker="LONG")
    dataset = build_sequence_dataset([records_short, records_long])

    short_seq = next(seq for seq in dataset.sequences if seq.ticker == "SHORT")
    long_seq = next(seq for seq in dataset.sequences if seq.ticker == "LONG")

    X, y, mask = collate_sequences([short_seq, long_seq], dataset.feature_names)

    max_len = len(long_seq.samples)
    assert len(X) == 2
    assert all(len(row) == max_len for row in X)
    assert all(len(row) == max_len for row in y)
    assert all(len(row) == max_len for row in mask)

    short_len = len(short_seq.samples)
    assert mask[0][:short_len] == [True] * short_len
    assert mask[0][short_len:] == [False] * (max_len - short_len)
    assert mask[1] == [True] * max_len


def test_collate_sequences_empty_list_returns_empty():
    assert collate_sequences([], FEATURE_NAMES) == ([], [], [])


def test_collate_sequences_feature_row_matches_feature_order():
    records = make_records(random_walk_prices(80))
    dataset = build_sequence_dataset([records])
    seq = dataset.sequences[0]

    X, _, _ = collate_sequences([seq], dataset.feature_names)

    for t, sample in enumerate(seq.samples):
        assert X[0][t] == [sample.feature.values[name] for name in dataset.feature_names]
