import pytest

from auto_stock.ml_predictor.labeling import LABEL_HORIZON_DAYS, LABEL_THRESHOLD, forward_returns, to_label


def test_label_horizon_and_threshold_constants():
    assert LABEL_HORIZON_DAYS == 5
    assert LABEL_THRESHOLD == 0.0


def test_label_uses_exactly_close_at_t_plus_horizon():
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 110.0, 200.0]  # index0 + horizon(5) = index5 = 110.0

    result = forward_returns(closes, horizon=5)

    assert result[0] == pytest.approx(110.0 / 100.0 - 1)
    # must not accidentally read index 6 (off-by-one) for position 0
    assert result[0] != pytest.approx(200.0 / 100.0 - 1)


def test_last_horizon_rows_have_no_label():
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    horizon = 5

    result = forward_returns(closes, horizon=horizon)

    assert result[-horizon:] == [None] * horizon
    assert all(v is not None for v in result[: len(closes) - horizon])


def test_forward_returns_is_empty_safe():
    assert forward_returns([], horizon=5) == []


def test_to_label_is_1_when_forward_return_exceeds_threshold():
    assert to_label(0.01, threshold=0.0) == 1


def test_to_label_is_0_when_forward_return_at_or_below_threshold():
    assert to_label(0.0, threshold=0.0) == 0
    assert to_label(-0.01, threshold=0.0) == 0


def test_to_label_is_none_when_forward_return_is_none():
    assert to_label(None, threshold=0.0) is None
