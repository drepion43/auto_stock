"""전체 유니버스 추천 파이프라인(①숏리스트+②심층분석+③종합)을 백그라운드로 트리거하는
조율기(recommendation-synthesis-plan.md §1/§7). `ScanCoordinator`와 동일한 "stale
체크 + in-progress 추적 + 백그라운드 스레드" 패턴을 쓰지만, `scan_market`(무료 경로)의
계약을 안 건드린다는 §2 원칙에 따라 **별도 클래스**로 분리했다 — 기존 `ScanCoordinator`를
확장하지 않는다(2026-10-05 사용자 확정).

`_fresh_universe`(`scan_coordinator.py`)는 `run_market_scan.py`도 재사용하는 모듈
레벨 함수라 여기서도 그대로 재사용한다 — 유니버스 목록 캐싱 로직을 중복시키지 않는다.

`_bind_stock_analyst_tools`(`chat_agent/tools.py`)를 재사용하려면 `ChatToolContext`가
필요한데, 그 6개 신호 tool 함수 중 `context.scan_cache`/`context.scan_coordinator`를
읽는 건 하나도 없다(그 두 필드는 `tool_get_market_scan_recommendations`만 쓴다) — 그래서
이 조율기는 `ScanCoordinator`를 몰라도 되고, 두 필드에 `None`을 넘긴다. 종목마다
`bind_tools`가 새 `ChatToolContext`(+ `cache.cursor()`)를 만드는 이유는 `run_deep_scan`이
`ThreadPoolExecutor`로 종목을 동시 처리하기 때문이다 — DuckDB 공식 스레딩 패턴(스레드마다
전용 커넥션)을 지키려면 종목별 호출 시점(해당 스레드 안)에 커서를 떠야 한다."""

import sys
import threading
from datetime import datetime, timedelta, timezone

from auto_stock.chat_agent.credentials import STOCK_ANALYST_LLM_CALL_ESTIMATE
from auto_stock.chat_agent.models import QueryBudget
from auto_stock.chat_agent.tools import ChatToolContext, _bind_stock_analyst_tools
from auto_stock.data.cache import OHLCVCache
from auto_stock.data.scan_cache import MarketScanCache
from auto_stock.llm_chart_analyst.models import ChartPatternReader
from auto_stock.ml_predictor.models import ModelBundle
from auto_stock.news_disclosure.models import DisclosureReader
from auto_stock.news_sentiment.models import NewsSentimentReader
from auto_stock.orchestrator.deep_scan import run_deep_scan
from auto_stock.orchestrator.pipeline import DEFAULT_LOOKBACK_DAYS
from auto_stock.orchestrator.scan_coordinator import _fresh_universe
from auto_stock.orchestrator.shortlist import ShortlistEntry, build_shortlist
from auto_stock.recommendation_synthesis.credentials import (
    RECOMMENDATION_SYNTHESIS_LLM_CALL_ESTIMATE,
    RECOMMENDATION_SYNTHESIS_RECURSION_LIMIT,
)
from auto_stock.recommendation_synthesis.synthesizer import run_recommendation_synthesis
from auto_stock.risk_sizing.models import AccountState

DEFAULT_STALE_AFTER = timedelta(hours=24)  # §7, 2026-10-05 확정 — 일봉 데이터라 12h는 과함
DEFAULT_SHORTLIST_CAP = 20  # §2, 2026-10-05 확정
DEFAULT_TOP_N = 5  # §4, 2026-10-05 확정
DEFAULT_MAX_WORKERS = 8  # §3, 2026-10-05 확정


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _shortlist_score(entry: ShortlistEntry) -> float:
    """§2 — 규칙엔진 후보 존재 여부 + ML 확률이 0.5에서 먼 정도를 합친 점수."""
    rule_score = 1.0 if entry.rule_candidate is not None else 0.0
    ml_score = (
        abs(entry.ml_prediction.probability_up - 0.5) if entry.ml_prediction is not None else 0.0
    )
    return rule_score + ml_score


class RecommendationCoordinator:
    def __init__(
        self,
        cache: OHLCVCache,
        scan_cache: MarketScanCache,
        ml_models: dict[str, ModelBundle | None],
        llm_client: ChartPatternReader | None,
        news_client: DisclosureReader | None,
        sentiment_client: NewsSentimentReader | None,
        account: AccountState,
        agent_model: str,
        stale_after: timedelta = DEFAULT_STALE_AFTER,
        universe_size: int = 200,
        shortlist_cap: int = DEFAULT_SHORTLIST_CAP,
        top_n: int = DEFAULT_TOP_N,
        max_workers: int = DEFAULT_MAX_WORKERS,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ) -> None:
        self._cache = cache
        self._scan_cache = scan_cache
        self._ml_models = ml_models
        self._llm_client = llm_client
        self._news_client = news_client
        self._sentiment_client = sentiment_client
        self._account = account
        self._agent_model = agent_model
        self._stale_after = stale_after
        self._universe_size = universe_size
        self._shortlist_cap = shortlist_cap
        self._top_n = top_n
        self._max_workers = max_workers
        self._lookback_days = lookback_days
        self._lock = threading.Lock()
        self._in_progress: set[str] = set()

    def ensure_fresh(self, market: str) -> threading.Thread | None:
        with self._lock:
            if market in self._in_progress:
                return None
            scanned_at, _ = self._scan_cache.get_latest_recommendations(market)
            if scanned_at is not None and _now() - scanned_at < self._stale_after:
                return None
            self._in_progress.add(market)

        thread = threading.Thread(target=self._run, args=(market,), daemon=True)
        thread.start()
        return thread

    def is_in_progress(self, market: str) -> bool:
        with self._lock:
            return market in self._in_progress

    def _budget_for(self, shortlist_size: int) -> QueryBudget:
        """§7 최악의 경우 상한 — 숏리스트 전체가 stock_analyst 재귀상한까지 다 쓰고,
        종합 agent도 자기 루프 + 재조사(재귀상한만큼)를 전부 쓰는 경우."""
        worst_case = (
            shortlist_size * STOCK_ANALYST_LLM_CALL_ESTIMATE
            + RECOMMENDATION_SYNTHESIS_LLM_CALL_ESTIMATE
            + RECOMMENDATION_SYNTHESIS_RECURSION_LIMIT * STOCK_ANALYST_LLM_CALL_ESTIMATE
        )
        return QueryBudget(max_llm_calls=worst_case)

    def _make_bind_tools(self, budget: QueryBudget):
        def bind_tools(ticker: str, market: str) -> list:
            context = ChatToolContext(
                cache=self._cache.cursor(),
                ml_models=self._ml_models,
                llm_client=self._llm_client,
                news_client=self._news_client,
                sentiment_client=self._sentiment_client,
                budget=budget,
                account=self._account,
                agent_model=self._agent_model,
                scan_cache=None,  # 6개 신호 tool 중 아무도 읽지 않음 — ScanCoordinator 몰라도 됨
                scan_coordinator=None,
            )
            return _bind_stock_analyst_tools(context, ticker, market)

        return bind_tools

    def _run(self, market: str) -> None:
        try:
            cache = self._cache.cursor()
            scan_cache = self._scan_cache.cursor()
            tickers = _fresh_universe(scan_cache, market, self._universe_size, self._stale_after)

            shortlist, build_errors = build_shortlist(
                cache, tickers, market, self._ml_models.get(market), self._lookback_days
            )
            shortlist.sort(key=_shortlist_score, reverse=True)
            shortlist = shortlist[: self._shortlist_cap]

            budget = self._budget_for(len(shortlist))
            bind_tools = self._make_bind_tools(budget)

            judgments = run_deep_scan(
                shortlist=shortlist, budget=budget, agent_model=self._agent_model,
                bind_tools=bind_tools, max_workers=self._max_workers,
            )
            synthesis = run_recommendation_synthesis(
                budget=budget, model=self._agent_model, bind_stock_analyst_tools=bind_tools,
                judgments=judgments, top_n=self._top_n,
            )

            if synthesis["available"]:
                scan_cache.put_recommendations(market, synthesis["recommendations"])
            else:
                print(
                    f"WARNING: recommendation synthesis for {market} unavailable: "
                    f"{synthesis.get('error')}",
                    file=sys.stderr,
                )
            if build_errors:
                print(
                    f"WARNING: recommendation shortlist for {market}: "
                    f"{len(build_errors)} per-ticker errors",
                    file=sys.stderr,
                )
        except Exception as exc:  # 백그라운드 스레드 — 절대 전파하지 않는다(ScanCoordinator와 동일 철학)
            print(f"WARNING: background recommendation pipeline for {market} failed: {exc}", file=sys.stderr)
        finally:
            with self._lock:
                self._in_progress.discard(market)
