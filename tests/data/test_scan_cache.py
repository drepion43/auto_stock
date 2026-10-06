from datetime import datetime

import duckdb
import pytest

from auto_stock.data.scan_cache import MarketScanCache
from auto_stock.explainer.models import Explanation


@pytest.fixture
def cache(tmp_path):
    db_path = tmp_path / "scan_cache_test.duckdb"
    return MarketScanCache(db_path)


def _explanation(ticker="005930", market="KRX", action="BUY", summary="RSI 과매도"):
    return Explanation(ticker=ticker, market=market, action=action, summary=summary)


def test_get_latest_returns_none_and_empty_when_never_scanned(cache):
    scanned_at, explanations = cache.get_latest("KRX")

    assert scanned_at is None
    assert explanations == []


def test_put_scan_then_get_latest_round_trips(cache):
    cache.put_scan("KRX", [_explanation(ticker="005930"), _explanation(ticker="000660", action="SELL")])

    scanned_at, explanations = cache.get_latest("KRX")

    assert isinstance(scanned_at, datetime)
    assert sorted(e.ticker for e in explanations) == ["000660", "005930"]


def test_put_scan_replaces_previous_batch_entirely(cache):
    cache.put_scan("KRX", [_explanation(ticker="005930")])
    cache.put_scan("KRX", [_explanation(ticker="000660")])

    _, explanations = cache.get_latest("KRX")

    assert [e.ticker for e in explanations] == ["000660"]


def test_put_scan_with_zero_candidates_still_records_scanned_at(cache):
    """실사용 중 발견 — 후보 0건은 정상적인 스캔 결과다(대부분의 날, 대부분의 종목은
    규칙엔진 신호가 없다). scanned_at을 후보 목록과 별도로 기록하지 않으면 "한 번도
    스캔 안 함"과 "스캔했지만 후보 없음"을 구분할 수 없다."""
    cache.put_scan("KRX", [])

    scanned_at, explanations = cache.get_latest("KRX")

    assert isinstance(scanned_at, datetime)
    assert explanations == []


def test_put_scan_rolls_back_on_failure_leaving_previous_batch_and_scanned_at_intact(cache):
    """코드리뷰(2026-09-09)에서 발견 — put_scan이 DELETE/INSERT/meta upsert를 개별
    autocommit 문으로 실행하면, 중간에 예외(예: PRIMARY KEY 위반, 프로세스 강제종료)가
    나는 순간 market_scan은 일부만 지워진 채 남고 market_scan_meta의 scanned_at은 이전
    값 그대로라 "이전 배치 스냅샷"과 "이번에 기록된 시각"이 서로 안 맞는 상태가 된다 —
    이 캐시가 애초에 없애려던 "scanned_at과 candidates 불일치" 버그가 다른 경로로
    재발하는 셈이다. 하나의 트랜잭션으로 묶어 실패 시 통째로 롤백해야 한다."""
    cache.put_scan("KRX", [_explanation(ticker="005930")])
    first_scanned_at, _ = cache.get_latest("KRX")

    duplicate_ticker_batch = [_explanation(ticker="000660"), _explanation(ticker="000660")]
    with pytest.raises(duckdb.ConstraintException):
        cache.put_scan("KRX", duplicate_ticker_batch)

    scanned_at, explanations = cache.get_latest("KRX")
    assert scanned_at == first_scanned_at
    assert [e.ticker for e in explanations] == ["005930"]


def test_put_scan_scopes_by_market(cache):
    cache.put_scan("KRX", [_explanation(ticker="005930", market="KRX")])
    cache.put_scan("NASDAQ", [_explanation(ticker="AAPL", market="NASDAQ", summary="상승 추세")])

    krx_scanned_at, krx_explanations = cache.get_latest("KRX")
    nasdaq_scanned_at, nasdaq_explanations = cache.get_latest("NASDAQ")

    assert [e.ticker for e in krx_explanations] == ["005930"]
    assert [e.ticker for e in nasdaq_explanations] == ["AAPL"]
    assert krx_scanned_at is not None and nasdaq_scanned_at is not None


def test_get_cached_universe_returns_none_and_empty_when_never_fetched(cache):
    fetched_at, tickers = cache.get_cached_universe("KRX")

    assert fetched_at is None
    assert tickers == []


def test_put_universe_then_get_cached_universe_round_trips(cache):
    cache.put_universe("KRX", ["005930", "000660"])

    fetched_at, tickers = cache.get_cached_universe("KRX")

    assert isinstance(fetched_at, datetime)
    assert tickers == ["005930", "000660"]


def test_put_universe_scopes_by_market(cache):
    cache.put_universe("KRX", ["005930"])
    cache.put_universe("NASDAQ", ["AAPL", "MSFT"])

    _, krx_tickers = cache.get_cached_universe("KRX")
    _, nasdaq_tickers = cache.get_cached_universe("NASDAQ")

    assert krx_tickers == ["005930"]
    assert nasdaq_tickers == ["AAPL", "MSFT"]


def test_put_universe_replaces_previous_list(cache):
    cache.put_universe("KRX", ["005930"])
    cache.put_universe("KRX", ["000660", "005380"])

    _, tickers = cache.get_cached_universe("KRX")

    assert tickers == ["000660", "005380"]


def _recommendation(ticker="005930", market="KRX", action="BUY", rank=1, summary="RSI 과매도 + ML 상승확률 70%"):
    return {"ticker": ticker, "market": market, "action": action, "rank": rank, "summary": summary}


def test_get_latest_recommendations_returns_none_and_empty_when_never_run(cache):
    """recommendation-synthesis-plan.md §6 — scan_key는 전체시장이면 market 값 그대로,
    섹터면 합성키("SECTOR:로봇:KRX")를 재사용한다. 여기선 market 값으로만 검증."""
    scanned_at, recommendations = cache.get_latest_recommendations("KRX")

    assert scanned_at is None
    assert recommendations == []


def test_put_recommendations_then_get_latest_round_trips(cache):
    cache.put_recommendations(
        "KRX", [_recommendation(ticker="005930", rank=1), _recommendation(ticker="000660", rank=2)]
    )

    scanned_at, recommendations = cache.get_latest_recommendations("KRX")

    assert isinstance(scanned_at, datetime)
    assert [r["ticker"] for r in recommendations] == ["005930", "000660"]


def test_put_recommendations_orders_by_rank(cache):
    cache.put_recommendations(
        "KRX", [_recommendation(ticker="000660", rank=2), _recommendation(ticker="005930", rank=1)]
    )

    _, recommendations = cache.get_latest_recommendations("KRX")

    assert [r["ticker"] for r in recommendations] == ["005930", "000660"]


def test_put_recommendations_replaces_previous_batch_entirely(cache):
    cache.put_recommendations("KRX", [_recommendation(ticker="005930", rank=1)])
    cache.put_recommendations("KRX", [_recommendation(ticker="000660", rank=1)])

    _, recommendations = cache.get_latest_recommendations("KRX")

    assert [r["ticker"] for r in recommendations] == ["000660"]


def test_put_recommendations_with_empty_list_still_records_scanned_at(cache):
    cache.put_recommendations("KRX", [])

    scanned_at, recommendations = cache.get_latest_recommendations("KRX")

    assert isinstance(scanned_at, datetime)
    assert recommendations == []


def test_put_recommendations_scopes_by_scan_key(cache):
    """섹터 합성키("SECTOR:로봇:KRX")와 일반 market 키("KRX")가 서로 침범하지 않아야
    한다 — recommendation-synthesis-plan.md §6."""
    cache.put_recommendations("KRX", [_recommendation(ticker="005930")])
    cache.put_recommendations("SECTOR:로봇:KRX", [_recommendation(ticker="277810")])

    _, krx = cache.get_latest_recommendations("KRX")
    _, sector = cache.get_latest_recommendations("SECTOR:로봇:KRX")

    assert [r["ticker"] for r in krx] == ["005930"]
    assert [r["ticker"] for r in sector] == ["277810"]


def test_recommendation_cache_cursor_shares_underlying_database(cache):
    thread_local = cache.cursor()

    thread_local.put_recommendations("KRX", [_recommendation(ticker="005930")])

    scanned_at, recommendations = cache.get_latest_recommendations("KRX")
    assert scanned_at is not None
    assert [r["ticker"] for r in recommendations] == ["005930"]


def test_get_cached_sector_tickers_returns_none_and_empty_when_never_resolved(cache):
    fetched_at, tickers, confidence = cache.get_cached_sector_tickers("로봇", "KRX")

    assert fetched_at is None
    assert tickers == []
    assert confidence == []


def test_put_sector_tickers_then_get_cached_round_trips(cache):
    cache.put_sector_tickers("로봇", "KRX", ["277810", "108490"], ["confirmed", "inferred"])

    fetched_at, tickers, confidence = cache.get_cached_sector_tickers("로봇", "KRX")

    assert isinstance(fetched_at, datetime)
    assert tickers == ["277810", "108490"]
    assert confidence == ["confirmed", "inferred"]


def test_put_sector_tickers_scopes_by_query_and_market(cache):
    """같은 질의 문자열이라도 market이 다르면 별개로 캐싱돼야 한다."""
    cache.put_sector_tickers("로봇", "KRX", ["277810"], ["confirmed"])
    cache.put_sector_tickers("반도체", "KRX", ["005930"], ["confirmed"])

    _, robot_tickers, _ = cache.get_cached_sector_tickers("로봇", "KRX")
    _, semi_tickers, _ = cache.get_cached_sector_tickers("반도체", "KRX")

    assert robot_tickers == ["277810"]
    assert semi_tickers == ["005930"]


def test_put_sector_tickers_replaces_previous_result(cache):
    cache.put_sector_tickers("로봇", "KRX", ["277810"], ["confirmed"])
    cache.put_sector_tickers("로봇", "KRX", ["108490", "277810"], ["inferred", "confirmed"])

    _, tickers, confidence = cache.get_cached_sector_tickers("로봇", "KRX")

    assert tickers == ["108490", "277810"]
    assert confidence == ["inferred", "confirmed"]


def test_sector_cache_cursor_shares_underlying_database(cache):
    thread_local = cache.cursor()

    thread_local.put_sector_tickers("로봇", "KRX", ["277810"], ["confirmed"])

    fetched_at, tickers, _ = cache.get_cached_sector_tickers("로봇", "KRX")
    assert fetched_at is not None
    assert tickers == ["277810"]


def test_scan_cache_cursor_shares_underlying_database(cache):
    """majestic-waddling-breeze.md "온디맨드 배치 스캔 트리거" 계획 — 백그라운드 스레드가
    cursor()로 쓴 스캔 결과를 원본 인스턴스가 바로 읽을 수 있어야 한다."""
    thread_local = cache.cursor()

    thread_local.put_scan("KRX", [_explanation(ticker="005930")])

    scanned_at, explanations = cache.get_latest("KRX")
    assert scanned_at is not None
    assert [e.ticker for e in explanations] == ["005930"]
