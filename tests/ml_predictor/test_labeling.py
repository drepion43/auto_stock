import pytest

from auto_stock.ml_predictor.labeling import LABEL_HORIZON_DAYS, forward_returns


def test_label_horizon_is_one_trading_day():
    assert LABEL_HORIZON_DAYS == 1


def test_label_uses_exactly_close_at_t_plus_horizon():
    closes = [100.0, 110.0, 200.0]  # index0 + horizon(1) = index1 = 110.0

    result = forward_returns(closes, horizon=1)

    assert result[0] == pytest.approx(110.0 / 100.0 - 1)
    # must not accidentally read index 2 (off-by-one) for position 0
    assert result[0] != pytest.approx(200.0 / 100.0 - 1)


def test_last_horizon_rows_have_no_label():
    closes = [100.0, 101.0, 102.0, 103.0]
    horizon = 1

    result = forward_returns(closes, horizon=horizon)

    assert result[-horizon:] == [None] * horizon
    assert all(v is not None for v in result[: len(closes) - horizon])


def test_forward_returns_is_empty_safe():
    assert forward_returns([], horizon=1) == []


def test_forward_returns_guards_against_zero_division():
    closes = [0.0, 100.0]
    assert forward_returns(closes, horizon=1) == [None, None]
