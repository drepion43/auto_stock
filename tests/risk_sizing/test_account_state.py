import pytest

from auto_stock.risk_sizing.models import AccountState


def test_creates_valid_account_state():
    account = AccountState(equity=10_000_000.0, held_tickers=frozenset({"005930"}), total_exposure_pct=0.2)

    assert account.equity == 10_000_000.0
    assert account.total_exposure_pct == 0.2


def test_rejects_negative_equity():
    with pytest.raises(ValueError, match="equity"):
        AccountState(equity=-1.0, held_tickers=frozenset(), total_exposure_pct=0.0)


def test_rejects_total_exposure_pct_below_zero():
    with pytest.raises(ValueError, match="total_exposure_pct"):
        AccountState(equity=1.0, held_tickers=frozenset(), total_exposure_pct=-0.1)


def test_rejects_total_exposure_pct_above_one():
    with pytest.raises(ValueError, match="total_exposure_pct"):
        AccountState(equity=1.0, held_tickers=frozenset(), total_exposure_pct=1.1)
