import threading
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from auto_stock.explainer.models import Explanation

_SCAN_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS market_scan (
    ticker VARCHAR NOT NULL,
    market VARCHAR NOT NULL,
    action VARCHAR NOT NULL,
    summary VARCHAR NOT NULL,
    PRIMARY KEY (ticker, market)
);
"""
_META_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS market_scan_meta (
    market VARCHAR PRIMARY KEY,
    scanned_at TIMESTAMP NOT NULL
);
"""
_UNIVERSE_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS universe_cache (
    market VARCHAR PRIMARY KEY,
    tickers VARCHAR[] NOT NULL,
    fetched_at TIMESTAMP NOT NULL
);
"""


class MarketScanCache:
    """전체 유니버스 배치 스캔(`orchestrator/market_scan.py`)의 최신 결과 스냅샷만 보관하는
    DuckDB 캐시(`data/cache.py`의 `OHLCVCache`와 동일한 패턴) — 이력 누적이 아니라 market당
    "가장 최근 스캔 1회분"만 유지한다(챗봇의 "지금 기준 추천" 질의 범위, 히스토리 조회는
    이번 요청 밖). `OHLCVCache`와 별도 DuckDB 파일을 쓴다 — 같은 프로세스(chat_cli.py)가
    두 캐시를 동시에 열 때 한 파일에 대한 이중 연결을 피하기 위함.

    `universe_cache` 테이블은 `data/service.py`의 `get_universe()` 결과(종목 목록 자체)를
    캐싱한다 — 나스닥 상장목록 조회가 실측 약 4분 걸려서(majestic-waddling-breeze.md
    "온디맨드 배치 스캔 트리거" 계획), 매 트리거마다 다시 받아오면 낭비다."""

    def __init__(self, db_path: str | Path):
        self._con = duckdb.connect(str(db_path))
        self._con.execute(_SCAN_TABLE_SCHEMA)
        self._con.execute(_META_TABLE_SCHEMA)
        self._con.execute(_UNIVERSE_TABLE_SCHEMA)
        self._write_lock = threading.Lock()

    def cursor(self) -> "MarketScanCache":
        """스레드 전용 커넥션(DuckDB 공식 패턴: `Connection.cursor()`)으로 동작하는 새
        인스턴스를 반환한다 — `data/cache.py`의 `OHLCVCache.cursor()`와 동일 이유·동일
        패턴(`_write_lock`은 원본과 공유, 새로 만들지 않음)."""
        clone = object.__new__(MarketScanCache)
        clone._con = self._con.cursor()
        clone._write_lock = self._write_lock
        return clone

    def put_scan(self, market: str, explanations: list[Explanation]) -> datetime:
        # scanned_at은 후보 행과 별도인 market_scan_meta에 항상 기록한다 — 후보 0건도
        # 정상적인 스캔 결과라(대부분의 날 대부분의 종목엔 신호가 없다, 실사용 중 확인),
        # scanned_at을 후보 테이블에만 묶어두면 "한 번도 스캔 안 함"과 "스캔했지만 후보
        # 없음"을 get_latest가 구분하지 못하는 버그가 된다.
        #
        # DELETE/INSERT×N/meta upsert를 하나의 트랜잭션 + 락으로 묶는다 — 코드리뷰
        # (2026-09-09)에서 발견: 개별 autocommit 문으로 실행하면 중간에 예외(제약 위반,
        # 프로세스 강제종료 등)가 나는 순간 market_scan은 일부만 지워진 채 남고
        # market_scan_meta.scanned_at은 이전 값 그대로라, 이 캐시가 애초에 없애려던
        # "scanned_at과 candidates 불일치" 버그가 다른 경로로 재발한다
        # (test_put_scan_rolls_back_on_failure_leaving_previous_batch_and_scanned_at_intact
        # 로 재현·고정). 락은 온디맨드 트리거 도입(2026-09) 이후 KRX/NASDAQ 백그라운드
        # 스레드가 같은 파일에 동시에 커밋하는 걸 막기 위해 필요해졌다.
        scanned_at = datetime.now(timezone.utc).replace(tzinfo=None)
        with self._write_lock:
            self._con.execute("BEGIN TRANSACTION")
            try:
                self._con.execute("DELETE FROM market_scan WHERE market = ?", [market])
                for e in explanations:
                    self._con.execute(
                        "INSERT INTO market_scan (ticker, market, action, summary) VALUES (?, ?, ?, ?)",
                        [e.ticker, e.market, e.action, e.summary],
                    )
                self._con.execute(
                    """
                    INSERT INTO market_scan_meta (market, scanned_at) VALUES (?, ?)
                    ON CONFLICT (market) DO UPDATE SET scanned_at = excluded.scanned_at
                    """,
                    [market, scanned_at],
                )
            except Exception:
                self._con.execute("ROLLBACK")
                raise
            else:
                self._con.execute("COMMIT")
        return scanned_at

    def put_universe(self, market: str, tickers: list[str]) -> datetime:
        """전체(캡 적용 전) 종목 목록을 저장한다 — `MARKET_SCAN_UNIVERSE_SIZE`가 바뀌어도
        캐시가 무효화되지 않도록 캡은 읽는 쪽에서 적용한다."""
        fetched_at = datetime.now(timezone.utc).replace(tzinfo=None)
        with self._write_lock:
            self._con.execute(
                """
                INSERT INTO universe_cache (market, tickers, fetched_at) VALUES (?, ?, ?)
                ON CONFLICT (market) DO UPDATE SET
                    tickers = excluded.tickers,
                    fetched_at = excluded.fetched_at
                """,
                [market, tickers, fetched_at],
            )
        return fetched_at

    def get_cached_universe(self, market: str) -> tuple[datetime | None, list[str]]:
        row = self._con.execute(
            "SELECT fetched_at, tickers FROM universe_cache WHERE market = ?", [market]
        ).fetchone()
        if row is None:
            return None, []
        return row[0], list(row[1])

    def get_latest(self, market: str) -> tuple[datetime | None, list[Explanation]]:
        meta_row = self._con.execute(
            "SELECT scanned_at FROM market_scan_meta WHERE market = ?", [market]
        ).fetchone()
        if meta_row is None:
            return None, []

        rows = self._con.execute(
            "SELECT ticker, market, action, summary FROM market_scan WHERE market = ? ORDER BY ticker",
            [market],
        ).fetchall()
        explanations = [
            Explanation(ticker=row[0], market=row[1], action=row[2], summary=row[3]) for row in rows
        ]
        return meta_row[0], explanations
