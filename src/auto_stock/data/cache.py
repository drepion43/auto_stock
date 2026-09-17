import threading
from datetime import date
from pathlib import Path

import duckdb

from auto_stock.data.models import OHLCVRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    ticker VARCHAR NOT NULL,
    market VARCHAR NOT NULL,
    date DATE NOT NULL,
    open DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    close DOUBLE NOT NULL,
    volume BIGINT NOT NULL,
    updated_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (ticker, market, date)
);
"""


class OHLCVCache:
    """DuckDB-backed local cache for OHLCV data (design: docs/design/data-collection-layer.md)."""

    def __init__(self, db_path: str | Path):
        self._con = duckdb.connect(str(db_path))
        self._con.execute(_SCHEMA)
        self._write_lock = threading.Lock()

    def cursor(self) -> "OHLCVCache":
        """스레드 전용 커넥션(DuckDB 공식 패턴: `Connection.cursor()`)으로 동작하는 새
        인스턴스를 반환한다 — 같은 DB를 가리키는 별도 커넥션이지 새 DB가 아니다
        (majestic-waddling-breeze.md "온디맨드 배치 스캔 트리거" 계획: 백그라운드 스레드가
        이 메서드로 자기 전용 커넥션을 얻어 메인 스레드와 동시에 안전하게 읽고 쓴다).
        `_write_lock`은 새로 만들지 않고 원본과 공유한다 — 그래야 원본 인스턴스와 이
        클론이 같은 파일에 동시에 쓰는 걸 막을 수 있다."""
        clone = object.__new__(OHLCVCache)
        clone._con = self._con.cursor()
        clone._write_lock = self._write_lock
        return clone

    def put(self, records: list[OHLCVRecord]) -> None:
        with self._write_lock:
            for r in records:
                self._con.execute(
                    """
                    INSERT INTO ohlcv (ticker, market, date, open, high, low, close, volume, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)
                    ON CONFLICT (ticker, market, date) DO UPDATE SET
                        open = excluded.open,
                        high = excluded.high,
                        low = excluded.low,
                        close = excluded.close,
                        volume = excluded.volume,
                        updated_at = excluded.updated_at
                    """,
                    [r.ticker, r.market, r.date, r.open, r.high, r.low, r.close, r.volume],
                )

    def get(self, ticker: str, market: str, start: date, end: date) -> list[OHLCVRecord]:
        rows = self._con.execute(
            """
            SELECT ticker, market, date, open, high, low, close, volume
            FROM ohlcv
            WHERE ticker = ? AND market = ? AND date BETWEEN ? AND ?
            ORDER BY date
            """,
            [ticker, market, start, end],
        ).fetchall()
        return [
            OHLCVRecord(
                ticker=row[0], market=row[1], date=row[2],
                open=row[3], high=row[4], low=row[5], close=row[6], volume=row[7],
            )
            for row in rows
        ]

    def covers(self, ticker: str, market: str, start: date, end: date) -> bool:
        """요청 구간을 캐시가 완전히 포함하는지 여부 (근사치: 최소/최대 날짜 비교)."""
        row = self._con.execute(
            "SELECT MIN(date), MAX(date) FROM ohlcv WHERE ticker = ? AND market = ?",
            [ticker, market],
        ).fetchone()
        min_date, max_date = row
        if min_date is None or max_date is None:
            return False
        return min_date <= start and max_date >= end
