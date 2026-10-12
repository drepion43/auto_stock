"""index_quote.py — 지수 자체 시세 조회(코스피/코스닥/코스피200/VIX/다우존스/나스닥종합/
S&P500/러셀2000). fdr.DataReader를 모킹해 실제 Yahoo Finance 호출 없이 테스트한다."""

from datetime import date

import pandas as pd
import pytest

from auto_stock.data.sources.index_quote import UnsupportedIndexError, fetch_index_quote


def _ohlcv_df(rows: list[tuple[str, float]]) -> pd.DataFrame:
    index = pd.to_datetime([d for d, _ in rows])
    closes = [c for _, c in rows]
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": [0] * len(rows)},
        index=index,
    )


def test_fetch_index_quote_returns_latest_close_and_change_pct(mocker):
    df = _ohlcv_df([("2026-01-05", 20.0), ("2026-01-06", 22.0)])
    mocker.patch("auto_stock.data.sources.index_quote.fdr.DataReader", return_value=df)

    quote = fetch_index_quote("VIX")

    assert quote.index == "VIX"
    assert quote.date == date(2026, 1, 6)
    assert quote.close == 22.0
    assert quote.change_pct == pytest.approx(10.0)


def test_fetch_index_quote_returns_none_change_pct_with_single_row(mocker):
    df = _ohlcv_df([("2026-01-06", 22.0)])
    mocker.patch("auto_stock.data.sources.index_quote.fdr.DataReader", return_value=df)

    quote = fetch_index_quote("DJI")

    assert quote.close == 22.0
    assert quote.change_pct is None


def test_fetch_index_quote_returns_none_when_no_data(mocker):
    mocker.patch("auto_stock.data.sources.index_quote.fdr.DataReader", return_value=pd.DataFrame())

    quote = fetch_index_quote("IXIC")

    assert quote is None


def test_fetch_index_quote_supports_kospi(mocker):
    df = _ohlcv_df([("2026-01-05", 2500.0), ("2026-01-06", 2550.0)])
    mocker.patch("auto_stock.data.sources.index_quote.fdr.DataReader", return_value=df)

    quote = fetch_index_quote("KS11")

    assert quote.index == "KS11"
    assert quote.close == 2550.0
    assert quote.change_pct == pytest.approx(2.0)


def test_fetch_index_quote_rejects_unsupported_index():
    with pytest.raises(UnsupportedIndexError):
        fetch_index_quote("NOT_AN_INDEX")


def test_fetch_index_quote_handles_zero_prior_close_without_zero_division(mocker):
    df = _ohlcv_df([("2026-01-05", 0.0), ("2026-01-06", 22.0)])
    mocker.patch("auto_stock.data.sources.index_quote.fdr.DataReader", return_value=df)

    quote = fetch_index_quote("RUT")

    assert quote.close == 22.0
    assert quote.change_pct is None
