"""OS 스케줄러 없이 챗봇 프로세스 자신이 배치 스캔을 트리거하는 조율기
(majestic-waddling-breeze.md "온디맨드 배치 스캔 트리거" 계획). `scripts/chat_cli.py`가
시작 시 시장별로 한 번씩 `ensure_fresh`를 선제 호출하고, `chat_agent.tools.
tool_get_market_scan_recommendations`도 질의 시점마다 자가치유 삼아 같은 메서드를
호출한다 — 두 호출부 모두 idempotent: 이미 신선하거나 이미 진행 중이면 아무 것도 안
한다.

챗봇 전용 로직(LLM, QueryBudget)이 전혀 없는 순수 오케스트레이션이라 `chat_agent/`가
아니라 `orchestrator/`에 둔다 — `market_scan.py`의 형제 모듈.

스레드 안전성은 `OHLCVCache`/`MarketScanCache`가 각자 내부 `_write_lock` + `cursor()`
(DuckDB 공식 스레딩 패턴)로 보장하므로, 이 클래스는 market별 in-progress 상태만 자체
락으로 지킨다."""

import sys
import threading
from datetime import date, datetime, timedelta, timezone

from auto_stock.data.cache import OHLCVCache
from auto_stock.data.scan_cache import MarketScanCache
from auto_stock.data.service import get_universe
from auto_stock.orchestrator.market_scan import scan_market
from auto_stock.orchestrator.pipeline import DEFAULT_LOOKBACK_DAYS
from auto_stock.risk_sizing.models import AccountState

DEFAULT_STALE_AFTER = timedelta(hours=12)


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _fresh_universe(
    scan_cache: MarketScanCache, market: str, universe_size: int, stale_after: timedelta
) -> list[str]:
    """유니버스 목록(종목 코드 리스트) 자체를 캐싱한다 — 나스닥 상장목록 조회가 실측
    약 4분 걸려서, 매 트리거마다 다시 받아오면 낭비다. `ScanCoordinator._run`뿐 아니라
    `scripts/run_market_scan.py`(스레드/코디네이터 없는 수동 실행)도 재사용할 수 있도록
    모듈 레벨 함수로 둔다."""
    fetched_at, tickers = scan_cache.get_cached_universe(market)
    if fetched_at is None or _now() - fetched_at >= stale_after:
        tickers = get_universe(market, date.today())
        scan_cache.put_universe(market, tickers)
    return tickers[:universe_size]


class ScanCoordinator:
    def __init__(
        self,
        cache: OHLCVCache,
        scan_cache: MarketScanCache,
        account: AccountState,
        stale_after: timedelta = DEFAULT_STALE_AFTER,
        universe_size: int = 200,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ) -> None:
        self._cache = cache
        self._scan_cache = scan_cache
        self._account = account
        self._stale_after = stale_after
        self._universe_size = universe_size
        self._lookback_days = lookback_days
        self._lock = threading.Lock()
        self._in_progress: set[str] = set()

    def ensure_fresh(self, market: str) -> threading.Thread | None:
        """이미 in-flight거나 신선하면 아무 것도 안 하고 None. 아니면 백그라운드 스레드를
        스폰해 반환한다(프로덕션 호출부는 fire-and-forget으로 반환값을 버리지만, 테스트가
        `.join()`으로 결정적으로 기다릴 수 있도록 반환한다). staleness 확인과 in-progress
        체크·set을 한 락 안에서 원자적으로 처리해 이중 트리거를 막는다."""
        with self._lock:
            if market in self._in_progress:
                return None
            scanned_at, _ = self._scan_cache.get_latest(market)
            if scanned_at is not None and _now() - scanned_at < self._stale_after:
                return None
            self._in_progress.add(market)

        thread = threading.Thread(target=self._run, args=(market,), daemon=True)
        thread.start()
        return thread

    def is_in_progress(self, market: str) -> bool:
        with self._lock:
            return market in self._in_progress

    def _run(self, market: str) -> None:
        try:
            cache = self._cache.cursor()
            scan_cache = self._scan_cache.cursor()
            tickers = _fresh_universe(scan_cache, market, self._universe_size, self._stale_after)
            explanations, errors = scan_market(cache, tickers, market, self._account, self._lookback_days)
            scan_cache.put_scan(market, explanations)
            if errors:
                print(
                    f"WARNING: market scan for {market}: {len(errors)} per-ticker errors",
                    file=sys.stderr,
                )
        except Exception as exc:  # 백그라운드 스레드 — 절대 전파하지 않는다(scan_market의
            # 티커별 실패격리와 동일 철학). 캐시는 stale인 채로 남아 다음 ensure_fresh
            # 호출이 자연스럽게 재시도한다.
            print(f"WARNING: background market scan for {market} failed: {exc}", file=sys.stderr)
        finally:
            with self._lock:
                self._in_progress.discard(market)
