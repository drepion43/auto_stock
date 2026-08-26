"""End-to-end MVP-0 wiring: data -> rule engine -> risk sizing -> explanation -> notification.

Pure composition of already-built, already-tested modules (docs/design/orchestrator.md)
— no new calculation logic here. Ticker list and AccountState are supplied by the
caller (see design doc for why the orchestrator doesn't decide those itself).
"""

from datetime import date, timedelta

from auto_stock.data.cache import OHLCVCache
from auto_stock.data.service import get_ohlcv
from auto_stock.explainer.generator import generate_explanation
from auto_stock.notifier.models import TelegramCredentials
from auto_stock.notifier.telegram_bot import send_notification
from auto_stock.orchestrator.models import PipelineResult
from auto_stock.risk_sizing.models import AccountState
from auto_stock.risk_sizing.sizing import suggest_position
from auto_stock.rule_engine.engine import generate_candidates

DEFAULT_LOOKBACK_DAYS = 120  # SMA60 warmup + ATR(14), padded for weekends/holidays


def run_recommendation_pipeline(
    cache: OHLCVCache,
    tickers: list[str],
    market: str,
    account: AccountState,
    credentials: TelegramCredentials,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> PipelineResult:
    end = date.today()
    start = end - timedelta(days=lookback_days)

    sent = []
    errors = []
    for ticker in tickers:
        try:
            records = get_ohlcv(cache, ticker, start, end, market)
            for candidate in generate_candidates(records):
                sizing = suggest_position(candidate, records, account)
                explanation = generate_explanation(candidate, sizing)
                send_notification(explanation, credentials)
                sent.append(explanation)
        except Exception as exc:  # per-ticker isolation is deliberate — see design doc
            errors.append((ticker, str(exc)))

    return PipelineResult(sent=sent, errors=errors)
