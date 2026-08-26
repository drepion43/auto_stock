from datetime import date

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
