"""End-to-end MVP-0 wiring: data -> rule engine -> risk sizing -> explanation -> notification.

Pure composition of already-built, already-tested modules (docs/design/orchestrator.md)
— no new calculation logic here. Ticker list and AccountState are supplied by the
caller (see design doc for why the orchestrator doesn't decide those itself).

`ml_model` is an optional boost-only signal source (ML #2, docs/design/ml-predictor-plan.md
핵심 설계 결정 2): when omitted (default None), this function is byte-for-byte identical
to the pre-ML behavior — no extra_reasons kwarg is even passed to generate_explanation.
ML failures never block notification delivery (핵심 설계 결정 5) — they're isolated in
`_ml_reasons` and only surface via `PipelineResult.errors`.
"""

from datetime import date, timedelta

from auto_stock.data.cache import OHLCVCache
from auto_stock.data.models import OHLCVRecord
from auto_stock.data.service import get_ohlcv
from auto_stock.explainer.generator import generate_explanation
from auto_stock.ml_predictor.models import ModelBundle
from auto_stock.ml_predictor.predictor import predict, to_reasons
from auto_stock.notifier.models import TelegramCredentials
from auto_stock.notifier.telegram_bot import send_notification
from auto_stock.orchestrator.models import PipelineResult
from auto_stock.risk_sizing.models import AccountState
from auto_stock.risk_sizing.sizing import suggest_position
from auto_stock.rule_engine.engine import generate_candidates

DEFAULT_LOOKBACK_DAYS = 120  # SMA60 warmup + ATR(14), padded for weekends/holidays


def _ml_reasons(
    ml_model: ModelBundle | None, records: list[OHLCVRecord], action: str
) -> tuple[list[str], str | None]:
    """절대 raise하지 않는다 — ML 실패가 추천 발송을 막아서는 안 된다(설계 결정 5)."""
    if ml_model is None:
        return [], None
    try:
        prediction = predict(ml_model, records)
        return to_reasons(prediction, action), None
    except Exception as exc:  # ML failure is isolated the same way per-ticker errors are
        return [], str(exc)


def run_recommendation_pipeline(
    cache: OHLCVCache,
    tickers: list[str],
    market: str,
    account: AccountState,
    credentials: TelegramCredentials,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ml_model: ModelBundle | None = None,
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
                if ml_model is None:
                    explanation = generate_explanation(candidate, sizing)
                else:
                    extra_reasons, ml_error = _ml_reasons(ml_model, records, candidate.action)
                    explanation = generate_explanation(candidate, sizing, extra_reasons=extra_reasons)
                    if ml_error is not None:
                        errors.append((ticker, f"ML 예측 실패: {ml_error}"))
                send_notification(explanation, credentials)
                sent.append(explanation)
        except Exception as exc:  # per-ticker isolation is deliberate — see design doc
            errors.append((ticker, str(exc)))

    return PipelineResult(sent=sent, errors=errors)
