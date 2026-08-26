from datetime import date

import pandas as pd

from auto_stock.data.sources import pykrx_source


def test_get_ticker_list_delegates_to_pykrx(mocker):
    mock_list = mocker.patch.object(
        pykrx_source.stock, "get_market_ticker_list", return_value=["005930", "000660"]
    )

    tickers = pykrx_source.get_ticker_list(date(2026, 1, 2), market="KOSPI")

    assert tickers == ["005930", "000660"]
    mock_list.assert_called_once_with("20260102", market="KOSPI")


def test_get_market_cap_returns_latest_value(mocker):
    df = pd.DataFrame(
        {"시가총액": [450_000_000_000_000], "거래량": [10_000_000]},
        index=pd.to_datetime(["2026-01-02"]),
    )
    mocker.patch.object(pykrx_source.stock, "get_market_cap", return_value=df)

    cap = pykrx_source.get_market_cap("005930", date(2026, 1, 2))

    assert cap == 450_000_000_000_000


def test_get_market_cap_returns_none_when_no_data(mocker):
    mocker.patch.object(pykrx_source.stock, "get_market_cap", return_value=pd.DataFrame())

    cap = pykrx_source.get_market_cap("000000", date(2026, 1, 2))

    assert cap is None
