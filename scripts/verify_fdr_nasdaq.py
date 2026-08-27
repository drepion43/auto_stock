"""Manual verification: confirms FinanceDataReader can list and fetch NASDAQ data.

Run from the project root:
    .venv/Scripts/python scripts/verify_fdr_nasdaq.py   (Windows)
    .venv/bin/python scripts/verify_fdr_nasdaq.py         (macOS/Linux)

NASDAQ has no pykrx equivalent — `data/service.py::get_universe()` and
`fdr_source.fetch_ohlcv()` both route NASDAQ through FinanceDataReader
instead (see scripts/verify_pykrx.py for the KRX/pykrx counterpart of this
check). No API key or login is required for either call.
"""

from datetime import date, timedelta

from auto_stock.data.service import get_universe
from auto_stock.data.sources.fdr_source import fetch_ohlcv

SAMPLE_TICKER = "AAPL"  # always listed, good smoke-test target

try:
    tickers = get_universe("NASDAQ")
    if not tickers:
        print("FAILED: get_universe('NASDAQ')이 빈 리스트를 반환했습니다")
    else:
        print(f"SUCCESS: NASDAQ 종목 유니버스 {len(tickers)}개 수신 (예: {tickers[:3]})")

    end = date.today()
    start = end - timedelta(days=30)
    records = fetch_ohlcv(SAMPLE_TICKER, start, end, "NASDAQ")
    if not records:
        print(f"FAILED: {SAMPLE_TICKER} OHLCV 조회 결과 없음")
    else:
        latest = records[-1]
        print(f"SUCCESS: {SAMPLE_TICKER} OHLCV {len(records)}건 수신 (최근: {latest.date} 종가 {latest.close})")
except Exception as e:
    print(f"FAILED: {e}")
