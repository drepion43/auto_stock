from datetime import date

import pandas as pd
import pytest

from auto_stock.data import service
from auto_stock.data.models import OHLCVRecord


def _record(d=date(2026, 1, 2)):
    return OHLCVRecord(
        ticker="005930", market="KRX", date=d,
        open=1.0, high=1.0, low=1.0, close=1.0, volume=1,
    )


def test_get_ohlcv_uses_cache_when_it_already_covers_range(mocker):
    cache = mocker.Mock()
    cache.covers.return_value = True
    cache.get.return_value = [_record()]
    fetch = mocker.patch("auto_stock.data.service.fdr_source.fetch_ohlcv")

    result = service.get_ohlcv(cache, "005930", date(2026, 1, 1), date(2026, 1, 6), market="KRX")

    assert result == [_record()]
    fetch.assert_not_called()
    cache.put.assert_not_called()


def test_get_ohlcv_fetches_and_caches_on_miss(mocker):
    cache = mocker.Mock()
    cache.covers.return_value = False
    cache.get.return_value = [_record()]
    fetch = mocker.patch("auto_stock.data.service.fdr_source.fetch_ohlcv", return_value=[_record()])

    result = service.get_ohlcv(cache, "005930", date(2026, 1, 1), date(2026, 1, 6), market="KRX")

    fetch.assert_called_once_with("005930", date(2026, 1, 1), date(2026, 1, 6), "KRX")
    cache.put.assert_called_once_with([_record()])
    assert result == [_record()]


def test_get_universe_krx_delegates_to_pykrx(mocker):
    get_tickers = mocker.patch(
        "auto_stock.data.service.pykrx_source.get_ticker_list", return_value=["005930"]
    )

    result = service.get_universe("KRX", as_of=date(2026, 1, 2))

    assert result == ["005930"]
    get_tickers.assert_called_once_with(date(2026, 1, 2))


def test_get_universe_nasdaq_delegates_to_fdr_listing(mocker):
    listing_df = pd.DataFrame({"Symbol": ["AAPL", "MSFT"]})
    mocker.patch("auto_stock.data.service.fdr.StockListing", return_value=listing_df)

    result = service.get_universe("NASDAQ")

    assert result == ["AAPL", "MSFT"]


def test_get_universe_raises_on_unknown_market():
    with pytest.raises(ValueError, match="market"):
        service.get_universe("NYSE")


def test_refresh_recent_fetches_and_caches_each_ticker(mocker):
    cache = mocker.Mock()
    fetch = mocker.patch("auto_stock.data.service.fdr_source.fetch_ohlcv", return_value=[_record()])

    service.refresh_recent(cache, ["005930", "000660"], market="KRX", lookback_days=3)

    assert fetch.call_count == 2
    assert cache.put.call_count == 2
