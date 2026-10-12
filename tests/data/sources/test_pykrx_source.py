import json
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


def test_resolve_market_downloads_and_caches_on_miss(mocker, tmp_path):
    cache_path = tmp_path / "krx_ticker_market.json"

    def fake_ticker_list(as_of_str, market):
        return {"KOSPI": ["005930"], "KOSDAQ": ["091990"]}[market]

    mock_list = mocker.patch.object(
        pykrx_source.stock, "get_market_ticker_list", side_effect=fake_ticker_list
    )

    kospi_market = pykrx_source.resolve_market("005930", cache_path=cache_path)
    kosdaq_market = pykrx_source.resolve_market("091990", cache_path=cache_path)

    assert kospi_market == "KOSPI"
    assert kosdaq_market == "KOSDAQ"
    assert mock_list.call_count == 2  # KOSPI + KOSDAQ, cached across both resolve_market calls
    assert cache_path.exists()


def test_resolve_market_returns_none_for_unknown_ticker(mocker, tmp_path):
    cache_path = tmp_path / "krx_ticker_market.json"
    mocker.patch.object(pykrx_source.stock, "get_market_ticker_list", return_value=["005930"])

    result = pykrx_source.resolve_market("999999", cache_path=cache_path)

    assert result is None


def test_resolve_market_reads_fresh_cache_without_calling_pykrx(mocker, tmp_path):
    cache_path = tmp_path / "krx_ticker_market.json"
    cache_path.write_text(json.dumps({"005930": "KOSPI"}), encoding="utf-8")
    mock_list = mocker.patch.object(pykrx_source.stock, "get_market_ticker_list")

    result = pykrx_source.resolve_market("005930", cache_path=cache_path)

    assert result == "KOSPI"
    mock_list.assert_not_called()
