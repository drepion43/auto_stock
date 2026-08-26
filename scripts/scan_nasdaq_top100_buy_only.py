"""Scan NASDAQ top-100 (by FDR listing order) and send only BUY recommendations.

Run from the project root so `.env` and `data/` resolve correctly:
    .venv/Scripts/python scripts/scan_nasdaq_top100_buy_only.py   (Windows)
    .venv/bin/python scripts/scan_nasdaq_top100_buy_only.py        (macOS/Linux)

This is a variant of scripts/run_recommendations.py: wider scan (100 tickers
instead of a 2-stock watchlist) and BUY-only (SELL candidates are still
computed by the rule engine as usual, just skipped before sizing/explanation/
notification). Kept as a separate script rather than changing
run_recommendations.py's default, smaller, BUY+SELL behavior.

fdr.StockListing("NASDAQ") doesn't expose an explicit market-cap column, but
its rows are already ordered largest-cap-first (verified: NVDA, AAPL, MSFT at
the top) — head(100) is used as the top-100 approximation.

AccountState is a placeholder (see run_recommendations.py's docstring for why
— no broker integration yet). Prints only counts/tickers — never prints
TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, or ACCOUNT_EQUITY.
"""

import os
from datetime import date, timedelta

import FinanceDataReader as fdr

from auto_stock.data.cache import OHLCVCache
from auto_stock.data.service import get_ohlcv
from auto_stock.explainer.generator import generate_explanation
from auto_stock.notifier.credentials import load_telegram_credentials
from auto_stock.notifier.telegram_bot import send_notification
from auto_stock.orchestrator.pipeline import DEFAULT_LOOKBACK_DAYS
from auto_stock.risk_sizing.models import AccountState
from auto_stock.risk_sizing.sizing import suggest_position
from auto_stock.rule_engine.engine import generate_candidates

MARKET = "NASDAQ"
TOP_N = 100
DEFAULT_ACCOUNT_EQUITY = 10_000_000.0

listing = fdr.StockListing(MARKET)
tickers = listing["Symbol"].head(TOP_N).tolist()
print(f"대상 종목 수: {len(tickers)}")

cache = OHLCVCache("data/ohlcv.duckdb")
credentials = load_telegram_credentials()
account = AccountState(
    equity=float(os.environ.get("ACCOUNT_EQUITY", DEFAULT_ACCOUNT_EQUITY)),
    held_tickers=frozenset(),
    total_exposure_pct=0.0,
)

end = date.today()
start = end - timedelta(days=DEFAULT_LOOKBACK_DAYS)

sent = []
skipped_sell = []
errors = []

for ticker in tickers:
    try:
        records = get_ohlcv(cache, ticker, start, end, MARKET)
        for candidate in generate_candidates(records):
            if candidate.action != "BUY":
                skipped_sell.append(candidate.ticker)
                continue
            sizing = suggest_position(candidate, records, account)
            explanation = generate_explanation(candidate, sizing)
            send_notification(explanation, credentials)
            sent.append(explanation)
    except Exception as exc:
        errors.append((ticker, str(exc)))

print(f"전송된 매수 추천: {len(sent)}건")
for e in sent:
    print(f"  - {e.ticker}")
print(f"제외된 매도 신호: {len(skipped_sell)}건 -> {skipped_sell}")
print(f"에러: {len(errors)}건")
for ticker, message in errors:
    print(f"  - {ticker}: {message}")
