"""KRX + NASDAQ 전체 유니버스 배치 스캔 — 챗봇의 `get_market_scan_recommendations` 도구가
즉답할 "추천해줄만한 주식 있어?"/"나스닥 추천해줄만한거 있어?" 같은 전체 스캔형 질의를
위한 캐시 적재 스크립트(majestic-waddling-breeze.md "전체 스캔형 질의 지원" 계획).
2026-09-09부로 NASDAQ도 추가됐다 — v1(2026-09-07)은 KRX만 스캔했고, `MarketScanCache`/
`scan_market`은 애초에 market 인자를 받는 범용 함수라 이 스크립트만 시장별로 순회하도록
바꾸면 됐다(캐시·오케스트레이션 계층 변경 불필요).

`run_recommendations.py`가 이미 경고하듯 전체 유니버스를 그대로 다 돌리면 수백~수천
종목에 대한 처리가 된다(NASDAQ은 특히 상장종목이 수천 개) — 이 스크립트는 텔레그램을
보내지 않으므로 스팸 문제는 없지만, `train_ml_model.py`가 검증한 것과 같은 캡 패턴
(`get_universe(MARKET)[:SCAN_UNIVERSE_SIZE]`, 기본 200)을 시장마다 동일하게 재사용해
실행 시간/API 호출량을 예측 가능하게 유지한다.

규칙엔진+사이징만 쓴다(LLM 미사용, 비용 없음) — `orchestrator/market_scan.py` 참고.

Run from the project root so `.env`/`data/` resolve correctly:
    .venv/Scripts/python scripts/run_market_scan.py   (Windows)
    .venv/bin/python scripts/run_market_scan.py        (macOS/Linux)

    MARKET_SCAN_UNIVERSE_SIZE=20 .venv/Scripts/python scripts/run_market_scan.py  (소규모 테스트)

**OS 스케줄러로 이 스크립트를 등록하지 말 것**(majestic-waddling-breeze.md "온디맨드
배치 스캔 트리거" 계획으로 전환됨, 2026-09-17) — 정상 운영에서는 `scripts/chat_cli.py`
자신이 시작 시점과 질의 시점에 `ScanCoordinator.ensure_fresh`로 스캔을 직접 트리거하므로
이 스크립트는 더 이상 필요 없다. 이 스크립트는 수동/오프라인 fallback(챗봇 없이 캐시만
미리 데워두고 싶을 때) 용도로만 남겨둔다. **`chat_cli.py`가 실행 중인 동안 이 스크립트를
같이 실행하지 말 것** — 둘 다 같은 `data/ohlcv.duckdb`/`data/market_scan.duckdb` 파일에
read-write 커넥션을 열려고 하므로 DuckDB는 두 번째로 여는 쪽에서 `duckdb.IOException`을
던진다(단일 프로세스만 파일을 열 수 있는 DuckDB의 제약). 반드시 챗봇을 먼저 종료한 뒤
실행할 것.

AccountState는 다른 스크립트와 동일한 이유로 placeholder다(브로커 연동 전, MVP-1(#8)).

Prints only counts — never prints ACCOUNT_EQUITY.
"""

import os
from datetime import timedelta

from auto_stock.data.cache import OHLCVCache
from auto_stock.data.scan_cache import MarketScanCache
from auto_stock.orchestrator.market_scan import scan_market
from auto_stock.orchestrator.scan_coordinator import DEFAULT_STALE_AFTER, _fresh_universe
from auto_stock.risk_sizing.models import AccountState

MARKETS = ["KRX", "NASDAQ"]
SCAN_UNIVERSE_SIZE = int(os.environ.get("MARKET_SCAN_UNIVERSE_SIZE", 200))
STALE_AFTER = timedelta(hours=int(os.environ.get("MARKET_SCAN_STALE_HOURS", DEFAULT_STALE_AFTER.total_seconds() // 3600)))
DEFAULT_ACCOUNT_EQUITY = 10_000_000.0

cache = OHLCVCache("data/ohlcv.duckdb")
scan_cache = MarketScanCache("data/market_scan.duckdb")
account = AccountState(
    equity=float(os.environ.get("ACCOUNT_EQUITY", DEFAULT_ACCOUNT_EQUITY)),
    held_tickers=frozenset(),
    total_exposure_pct=0.0,
)

for market in MARKETS:
    tickers = _fresh_universe(scan_cache, market, SCAN_UNIVERSE_SIZE, STALE_AFTER)
    print(f"[{market}] 유니버스: {len(tickers)}종목")

    explanations, errors = scan_market(cache=cache, tickers=tickers, market=market, account=account)
    scanned_at = scan_cache.put_scan(market, explanations)

    print(f"[{market}] 스캔 시각: {scanned_at.isoformat()}")
    print(f"[{market}] 발견된 후보: {len(explanations)}건")
    print(f"[{market}] 에러: {len(errors)}건")
    for ticker, message in errors:
        print(f"  - {ticker}: {message}")
