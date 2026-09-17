from datetime import date
from threading import Thread

import pytest

from auto_stock.data.cache import OHLCVCache
from auto_stock.data.models import OHLCVRecord


@pytest.fixture
def cache(tmp_path):
    db_path = tmp_path / "cache_test.duckdb"
    return OHLCVCache(db_path)


def _record(ticker="005930", market="KRX", d=date(2026, 1, 2), close=70000.0):
    return OHLCVRecord(
        ticker=ticker, market=market, date=d,
        open=close, high=close, low=close, close=close, volume=1000,
    )


def test_put_then_get_returns_records(cache):
    cache.put([_record(d=date(2026, 1, 2)), _record(d=date(2026, 1, 5))])

    records = cache.get("005930", "KRX", date(2026, 1, 1), date(2026, 1, 6))

    assert [r.date for r in records] == [date(2026, 1, 2), date(2026, 1, 5)]


def test_get_returns_empty_list_when_no_data(cache):
    records = cache.get("005930", "KRX", date(2026, 1, 1), date(2026, 1, 6))

    assert records == []


def test_covers_is_false_when_cache_empty(cache):
    assert cache.covers("005930", "KRX", date(2026, 1, 1), date(2026, 1, 6)) is False


def test_covers_is_true_when_range_fully_contained(cache):
    cache.put([_record(d=date(2026, 1, 1)), _record(d=date(2026, 1, 6))])

    assert cache.covers("005930", "KRX", date(2026, 1, 1), date(2026, 1, 6)) is True


def test_covers_is_false_when_range_partially_missing(cache):
    cache.put([_record(d=date(2026, 1, 3))])

    assert cache.covers("005930", "KRX", date(2026, 1, 1), date(2026, 1, 6)) is False


def test_put_upserts_existing_record(cache):
    cache.put([_record(d=date(2026, 1, 2), close=70000.0)])
    cache.put([_record(d=date(2026, 1, 2), close=71000.0)])

    records = cache.get("005930", "KRX", date(2026, 1, 1), date(2026, 1, 3))

    assert len(records) == 1
    assert records[0].close == 71000.0


def test_cursor_shares_underlying_database(cache):
    """majestic-waddling-breeze.md "온디맨드 배치 스캔 트리거" 계획 — 백그라운드 스레드가
    스레드 전용 커넥션(`cursor()`)으로 쓴 데이터를 원본 인스턴스가 바로 읽을 수 있어야
    한다(DuckDB 공식 스레딩 패턴: 원본 커넥션과 cursor()는 같은 DB를 가리킨다, 새 DB가
    아니다)."""
    thread_local = cache.cursor()

    thread_local.put([_record(d=date(2026, 1, 2))])

    records = cache.get("005930", "KRX", date(2026, 1, 1), date(2026, 1, 3))
    assert len(records) == 1


def test_concurrent_put_via_cursor_loses_no_writes(cache):
    """N개 스레드가 각자 cache.cursor()로 받은 스레드 전용 커넥션으로 서로 다른 티커를
    동시에 put()해도(내부 _write_lock으로 직렬화), 유실이나 예외 없이 전부 기록돼야
    한다. sleep 없이 join()으로만 동기화되는 결정적 테스트."""
    tickers = [f"T{i:03d}" for i in range(20)]
    threads = [
        Thread(target=lambda t=ticker: cache.cursor().put([_record(ticker=t, d=date(2026, 1, 2))]))
        for ticker in tickers
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    for ticker in tickers:
        records = cache.get(ticker, "KRX", date(2026, 1, 1), date(2026, 1, 3))
        assert len(records) == 1
