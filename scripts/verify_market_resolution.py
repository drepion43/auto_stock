"""Manual verification: confirms the new KOSPI/KOSDAQ/NASDAQ/NYSE resolver functions
(`pykrx_source.resolve_market`, `edgar_source.resolve_exchange`) work against real data —
market 세분화 작업(2026-10)의 핵심 설계 갭이었던 ticker_resolution.py의 디스앰비규에이션
로직이 실제로 올바른 거래소를 돌려주는지 확인한다.

Run from the project root so `.env`/`data/` resolve correctly:
    .venv/Scripts/python scripts/verify_market_resolution.py   (Windows)
    .venv/bin/python scripts/verify_market_resolution.py        (macOS/Linux)

Checks:
- pykrx_source.resolve_market: 005930(삼성전자, KOSPI 상장)/091990(셀트리온헬스케어,
  KOSDAQ 상장)이 올바른 서브마켓을 반환하는지.
- edgar_source.resolve_exchange: AAPL(NASDAQ 상장)/JPM(실제로는 NYSE 상장 —
  ticker_resolution.py의 한글 별칭표가 과거 NASDAQ으로 오분류하던 사례)이 올바른
  거래소를 반환하는지.

Prints only tickers/market labels — no credentials.
"""

from auto_stock.data.sources.edgar_source import resolve_exchange
from auto_stock.data.sources.pykrx_source import resolve_market

_KRX_SAMPLES = {"005930": "KOSPI", "091990": "KOSDAQ"}  # 삼성전자 / 셀트리온헬스케어
_US_SAMPLES = {"AAPL": "NASDAQ", "JPM": "NYSE"}  # Apple / JPMorgan(과거 오분류 사례)

for ticker, expected in _KRX_SAMPLES.items():
    actual = resolve_market(ticker)
    status = "SUCCESS" if actual == expected else "FAILED"
    print(f"{status}: resolve_market({ticker}) = {actual} (예상: {expected})")

for ticker, expected in _US_SAMPLES.items():
    actual = resolve_exchange(ticker)
    status = "SUCCESS" if actual == expected else "FAILED"
    print(f"{status}: resolve_exchange({ticker}) = {actual} (예상: {expected})")
