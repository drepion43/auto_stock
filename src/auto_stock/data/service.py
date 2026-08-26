from datetime import date, timedelta

import FinanceDataReader as fdr

from auto_stock.data.cache import OHLCVCache
from auto_stock.data.models import OHLCVRecord
from auto_stock.data.sources import fdr_source, pykrx_source


def get_ohlcv(cache: OHLCVCache, ticker: str, start: date, end: date, market: str) -> list[OHLCVRecord]:
    """캐시 우선 조회, 미스(구간 부족) 시 소스 API 호출 후 캐시에 적재."""
    if not cache.covers(ticker, market, start, end):
        records = fdr_source.fetch_ohlcv(ticker, start, end, market)
        cache.put(records)
    return cache.get(ticker, market, start, end)


def get_universe(market: str, as_of: date | None = None) -> list[str]:
    """전체 상장종목 리스트. KRX는 pykrx, NASDAQ은 FDR 상장목록 기반."""
    if market == "KRX":
        return pykrx_source.get_ticker_list(as_of or date.today())
    if market == "NASDAQ":
        listing = fdr.StockListing("NASDAQ")
        return listing["Symbol"].tolist()
    raise ValueError(f"unsupported market: {market!r}")


def refresh_recent(cache: OHLCVCache, tickers: list[str], market: str, lookback_days: int = 5) -> None:
    """폴링 잡이 호출하는 근접-실시간 갱신 함수."""
    end = date.today()
    start = end - timedelta(days=lookback_days)
    for ticker in tickers:
        records = fdr_source.fetch_ohlcv(ticker, start, end, market)
        cache.put(records)
