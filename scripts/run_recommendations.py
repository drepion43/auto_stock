"""MVP-0 end-to-end run: data -> rule engine -> risk sizing -> explanation -> Telegram.

Run from the project root so `.env` and `data/` resolve correctly:
    .venv/Scripts/python scripts/run_recommendations.py   (Windows)
    .venv/bin/python scripts/run_recommendations.py        (macOS/Linux)

WATCHLIST below is a small example list, not the full market universe — running
against the full listing (auto_stock.data.service.get_universe) would fire a real
Telegram message per signal across hundreds/thousands of tickers, which this
script deliberately does not do by default. Pass your own ticker list to
run_recommendation_pipeline() if you want a wider scan.

AccountState here is a placeholder: there is no broker integration yet (MVP-1
#8), so held_tickers/total_exposure_pct are always empty/zero and equity comes
from the optional ACCOUNT_EQUITY env var. Sizing suggestions are reference-only
regardless (docs/design/risk-position-sizing.md).

Prints only counts — never prints TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, or ACCOUNT_EQUITY.
"""

import os

from auto_stock.data.cache import OHLCVCache
from auto_stock.notifier.credentials import load_telegram_credentials
from auto_stock.orchestrator.pipeline import run_recommendation_pipeline
from auto_stock.risk_sizing.models import AccountState

WATCHLIST = ["005930", "000660"]  # 삼성전자, SK하이닉스 — example only
MARKET = "KRX"
DEFAULT_ACCOUNT_EQUITY = 10_000_000.0

cache = OHLCVCache("data/ohlcv.duckdb")
credentials = load_telegram_credentials()
account = AccountState(
    equity=float(os.environ.get("ACCOUNT_EQUITY", DEFAULT_ACCOUNT_EQUITY)),
    held_tickers=frozenset(),
    total_exposure_pct=0.0,
)

result = run_recommendation_pipeline(
    cache=cache, tickers=WATCHLIST, market=MARKET, account=account, credentials=credentials,
)

print(f"전송된 추천: {len(result.sent)}건")
print(f"에러: {len(result.errors)}건")
for ticker, message in result.errors:
    print(f"  - {ticker}: {message}")
