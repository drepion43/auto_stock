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

    def put(self, records: list[OHLCVRecord]) -> None:
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
