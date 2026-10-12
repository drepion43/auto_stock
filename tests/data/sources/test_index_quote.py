"""index_quote.py — 지수 자체 시세 조회(코스피/코스닥/코스피200/VIX/다우존스/나스닥종합/
S&P500/러셀2000). fdr.DataReader를 모킹해 실제 Yahoo Finance 호출 없이 테스트한다."""

from datetime import date

import pandas as pd
import pytest

from auto_stock.data.sources import index_quote
from auto_stock.data.sources.index_quote import UnsupportedIndexError, fetch_index_quote


@pytest.fixture(autouse=True)
def _fixed_today(monkeypatch):
    """신선도 가드(오래된 데이터는 None)가 고정 날짜 픽스처를 막지 않도록 '오늘'을 고정한다."""
    monkeypatch.setattr(index_quote, "_today", lambda: date(2026, 1, 7))


def _krx_df(rows: list[tuple[str, float]]) -> pd.DataFrame:
    """pykrx get_index_ohlcv_by_date 형태(한글 컬럼)."""
    index = pd.to_datetime([d for d, _ in rows])
    closes = [c for _, c in rows]
    return pd.DataFrame({"시가": closes, "고가": closes, "저가": closes, "종가": closes, "거래량": [0] * len(rows)}, index=index)


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


def test_fetch_index_quote_reads_korean_indices_from_pykrx(mocker):
    """FDR은 KS11/KQ11/KS200을 KRX 소스로 보내는데 그 소스가 약 4주 늦은 데이터만 줘서(2026-10-13
    실측, 9/17까지) 최근 구간 조회가 늘 비었다 — 국내 지수는 pykrx(KRX 공식)를 주 소스로 쓴다."""
    mock_krx = mocker.patch.object(
        index_quote.stock, "get_index_ohlcv_by_date",
        return_value=_krx_df([("2026-01-05", 2500.0), ("2026-01-06", 2550.0)]),
    )
    mock_fdr = mocker.patch("auto_stock.data.sources.index_quote.fdr.DataReader")

    quote = fetch_index_quote("KS11")

    assert quote.index == "KS11"
    assert quote.close == 2550.0
    assert quote.change_pct == pytest.approx(2.0)
    assert mock_krx.call_args.args[2] == "1001"  # KRX 코스피 지수 코드
    mock_fdr.assert_not_called()


def test_kospi200_uses_its_krx_index_code(mocker):
    mock_krx = mocker.patch.object(
        index_quote.stock, "get_index_ohlcv_by_date", return_value=_krx_df([("2026-01-06", 380.0)])
    )

    assert fetch_index_quote("KS200").close == 380.0
    assert mock_krx.call_args.args[2] == "1028"


def test_korean_index_falls_back_to_yahoo_when_pykrx_fails(mocker):
    mocker.patch.object(index_quote.stock, "get_index_ohlcv_by_date", side_effect=RuntimeError("KRX down"))
    mock_fdr = mocker.patch(
        "auto_stock.data.sources.index_quote.fdr.DataReader",
        return_value=_ohlcv_df([("2026-01-05", 800.0), ("2026-01-06", 808.0)]),
    )

    quote = fetch_index_quote("KQ11")

    assert quote.close == 808.0
    assert mock_fdr.call_args.args[0] == "^KQ11"


def test_kospi200_has_no_yahoo_fallback_and_returns_none(mocker):
    mocker.patch.object(index_quote.stock, "get_index_ohlcv_by_date", return_value=pd.DataFrame())
    mock_fdr = mocker.patch("auto_stock.data.sources.index_quote.fdr.DataReader")

    assert fetch_index_quote("KS200") is None
    mock_fdr.assert_not_called()


def test_stale_data_is_treated_as_unavailable(mocker):
    """최신 데이터가 너무 오래됐으면(예: FDR KRX 소스처럼 몇 주 늦음) 그 값을 현재 시세처럼
    보여주지 않는다."""
    mocker.patch(
        "auto_stock.data.sources.index_quote.fdr.DataReader",
        return_value=_ohlcv_df([("2025-12-01", 20.0), ("2025-12-02", 21.0)]),
    )

    assert fetch_index_quote("VIX") is None


def test_fetch_index_quote_rejects_unsupported_index():
    with pytest.raises(UnsupportedIndexError):
        fetch_index_quote("NOT_AN_INDEX")


def test_fetch_index_quote_handles_zero_prior_close_without_zero_division(mocker):
    df = _ohlcv_df([("2026-01-05", 0.0), ("2026-01-06", 22.0)])
    mocker.patch("auto_stock.data.sources.index_quote.fdr.DataReader", return_value=df)

    quote = fetch_index_quote("RUT")

    assert quote.close == 22.0
    assert quote.change_pct is None
