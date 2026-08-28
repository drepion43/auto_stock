"""End-to-end MVP-0 wiring: data -> rule engine -> risk sizing -> explanation -> notification.

Pure composition of already-built, already-tested modules (docs/design/orchestrator.md)
— no new calculation logic here. Ticker list and AccountState are supplied by the
caller (see design doc for why the orchestrator doesn't decide those itself).

`ml_model`/`llm_client` are optional boost-only signal sources (ML #2, LLM #3 —
docs/design/ml-predictor-plan.md 핵심 설계 결정 2, docs/design/llm-chart-analyst-plan.md
"오케스트레이터 통합"): when both are omitted (default None), this function is
byte-for-byte identical to the pre-ML/pre-LLM behavior — no extra_reasons kwarg is even
passed to generate_explanation. The branch condition below is deliberately
`ml_model is None and llm_client is None` (not two separate `if`s) so the existing
regression tests locking the 2-positional-argument call are not broken by adding LLM.

Both failures never block notification delivery (핵심 설계 결정 5/6) — they're isolated
in `_ml_reasons`/`_llm_reasons` and only surface via `PipelineResult.errors`, independently
of each other (one failing must not suppress the other's reasons).
"""

from datetime import date, timedelta

from auto_stock.data.cache import OHLCVCache
from auto_stock.data.models import OHLCVRecord
from auto_stock.data.service import get_ohlcv
from auto_stock.explainer.generator import generate_explanation
from auto_stock.llm_chart_analyst.analyst import analyze as analyze_chart
from auto_stock.llm_chart_analyst.analyst import to_reasons as to_chart_reasons
from auto_stock.llm_chart_analyst.models import ChartPatternReader
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


def _llm_reasons(
    llm_client: ChartPatternReader | None, records: list[OHLCVRecord], action: str
) -> tuple[list[str], str | None]:
    """절대 raise하지 않는다 — LLM 실패가 추천 발송을 막아서는 안 된다(설계 결정 6)."""
    if llm_client is None:
        return [], None
    try:
        analysis = analyze_chart(llm_client, records)
        return to_chart_reasons(analysis, action), None
    except Exception as exc:  # LLM failure is isolated the same way per-ticker errors are
        return [], str(exc)


def run_recommendation_pipeline(
    cache: OHLCVCache,
    tickers: list[str],
    market: str,
    account: AccountState,
    credentials: TelegramCredentials,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ml_model: ModelBundle | None = None,
    llm_client: ChartPatternReader | None = None,
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
                if ml_model is None and llm_client is None:
                    # 회귀 잠금: 보조 신호가 하나도 없으면 2-위치인자 호출을 그대로 보존한다
                    # (tests/orchestrator/test_pipeline.py의 assert_called_once_with(ANY, ANY))
                    explanation = generate_explanation(candidate, sizing)
                else:
                    extra_reasons: list[str] = []
                    ml_r, ml_error = _ml_reasons(ml_model, records, candidate.action)
                    extra_reasons.extend(ml_r)
                    llm_r, llm_error = _llm_reasons(llm_client, records, candidate.action)
                    extra_reasons.extend(llm_r)

                    explanation = generate_explanation(candidate, sizing, extra_reasons=extra_reasons)

                    if ml_error is not None:
                        errors.append((ticker, f"ML 예측 실패: {ml_error}"))
                    if llm_error is not None:
                        errors.append((ticker, f"LLM 차트분석 실패: {llm_error}"))
                send_notification(explanation, credentials)
                sent.append(explanation)
        except Exception as exc:  # per-ticker isolation is deliberate — see design doc
            errors.append((ticker, str(exc)))

    return PipelineResult(sent=sent, errors=errors)
