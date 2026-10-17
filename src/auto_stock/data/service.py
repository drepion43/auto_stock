from datetime import date

import FinanceDataReader as fdr

from auto_stock.data.cache import OHLCVCache
from auto_stock.data.models import OHLCVRecord
from auto_stock.data.sources import fdr_source, pykrx_source
from dotenv import load_dotenv


def get_ohlcv(cache: OHLCVCache, ticker: str, start: date, end: date, market: str) -> list[OHLCVRecord]:
    """캐시 우선 조회, 미스(구간 부족) 시 소스 API 호출 후 캐시에 적재."""
    if not cache.covers(ticker, market, start, end):
        records = fdr_source.fetch_ohlcv(ticker, start, end, market)
        cache.put(records)
    return cache.get(ticker, market, start, end)


def get_universe(market: str, as_of: date | None = None) -> list[str]:
    """원자 거래소 하나의 전체 상장종목 리스트. KOSPI/KOSDAQ은 pykrx, NASDAQ/NYSE는
    FDR 상장목록 기반. "KRX"/"US" 같은 합성 scan_key는 받지 않는다 — 호출자가
    원자 market을 명시해야 한다(`orchestrator/market_buckets.py` 참고)."""
    load_dotenv()
    if market in ("KOSPI", "KOSDAQ"):
        return pykrx_source.get_ticker_list(as_of or date.today(), market=market)
    if market in ("NASDAQ", "NYSE"):
        listing = fdr.StockListing(market)
        return listing["Symbol"].tolist()
    raise ValueError(f"unsupported market: {market!r}")
