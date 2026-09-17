"""전체 유니버스 배치 스캔 — "추천해줄만한 주식 있어?" 같은 전체 스캔형 질의를
`chat_agent.tools.tool_get_market_scan_recommendations`가 즉답할 수 있도록 결과를
`data.scan_cache.MarketScanCache`에 적재하기 위한 원샷 스캔 함수(`scripts/run_market_scan.py`가
호출).

`pipeline.run_recommendation_pipeline`과 데이터 흐름(get_ohlcv → generate_candidates →
suggest_position → generate_explanation, 티커별 실패 격리)은 동일하지만, 이 함수는
**텔레그램을 발송하지 않는다** — 캐시 적재 전용이라 별도 함수로 분리했다(수백 종목을
스캔하면서 매 후보마다 실제 알림을 보내면 스팸이 된다, `run_recommendations.py` docstring이
이미 경고하는 문제). 보조신호(ML/LLM차트/공시/뉴스감성)도 쓰지 않는다 — 규칙엔진+사이징만
쓰는 무료 경로다(사용자 승인, majestic-waddling-breeze.md 계획 참고)."""

from datetime import date, timedelta

from auto_stock.data.cache import OHLCVCache
from auto_stock.data.service import get_ohlcv
from auto_stock.explainer.generator import generate_explanation
from auto_stock.explainer.models import Explanation
from auto_stock.orchestrator.pipeline import DEFAULT_LOOKBACK_DAYS
from auto_stock.risk_sizing.models import AccountState
from auto_stock.risk_sizing.sizing import suggest_position
from auto_stock.rule_engine.engine import generate_candidates


def scan_market(
    cache: OHLCVCache,
    tickers: list[str],
    market: str,
    account: AccountState,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> tuple[list[Explanation], list[tuple[str, str]]]:
    end = date.today()
    start = end - timedelta(days=lookback_days)

    explanations: list[Explanation] = []
    errors: list[tuple[str, str]] = []
    for ticker in tickers:
        try:
            records = get_ohlcv(cache, ticker, start, end, market)
            for candidate in generate_candidates(records):
                sizing = suggest_position(candidate, records, account)
                explanations.append(generate_explanation(candidate, sizing))
        except Exception as exc:  # per-ticker isolation, same as run_recommendation_pipeline
            errors.append((ticker, str(exc)))

    return explanations, errors
