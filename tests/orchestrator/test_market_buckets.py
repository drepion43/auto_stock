"""market_buckets.py — 스캔 버킷(원자/합성) → 원자 market 매핑 로직."""

from auto_stock.orchestrator.market_buckets import (
    ALL_SCAN_BUCKETS,
    bucket_markets,
    bucket_universe_by_atomic_market,
    fetch_index_constituents,
    is_index_filter_bucket,
)


def test_atomic_market_bucket_maps_to_itself():
    assert bucket_markets("KOSPI") == ["KOSPI"]
    assert bucket_markets("NYSE") == ["NYSE"]


def test_krx_bucket_maps_to_kospi_and_kosdaq():
    assert bucket_markets("KRX") == ["KOSPI", "KOSDAQ"]


def test_us_bucket_maps_to_nasdaq_and_nyse():
    assert bucket_markets("US") == ["NASDAQ", "NYSE"]


def test_index_filter_buckets_map_to_us_markets():
    assert bucket_markets("SP500") == ["NASDAQ", "NYSE"]
    assert bucket_markets("DOW") == ["NASDAQ", "NYSE"]


def test_is_index_filter_bucket():
    assert is_index_filter_bucket("SP500") is True
    assert is_index_filter_bucket("DOW") is True
    assert is_index_filter_bucket("KRX") is False
    assert is_index_filter_bucket("KOSPI") is False


def test_fetch_index_constituents_rejects_non_filter_bucket():
    import pytest

    with pytest.raises(ValueError):
        fetch_index_constituents("KRX")


def test_fetch_index_constituents_delegates_to_registered_fetcher(mocker):
    # INDEX_FILTER_BUCKETS는 모듈 로드 시점에 함수 객체 레퍼런스를 직접 캡처해
    # 저장한다 — 원본 데이터소스 모듈의 이름을 패치해도 이미 캡처된 레퍼런스는
    # 바뀌지 않으므로, 딕셔너리 엔트리 자체를 교체해 "SP500 버킷이 등록된 fetcher를
    # 호출하는지"를 검증한다.
    from auto_stock.orchestrator import market_buckets

    fake_fetch = mocker.Mock(return_value={"AAPL", "MSFT"})
    original_entry = market_buckets.INDEX_FILTER_BUCKETS["SP500"]
    market_buckets.INDEX_FILTER_BUCKETS["SP500"] = (original_entry[0], fake_fetch)
    try:
        result = fetch_index_constituents("SP500")
    finally:
        market_buckets.INDEX_FILTER_BUCKETS["SP500"] = original_entry

    assert result == {"AAPL", "MSFT"}
    fake_fetch.assert_called_once_with()


def test_all_scan_buckets_contains_eight_entries():
    assert sorted(ALL_SCAN_BUCKETS) == sorted(
        ["KOSPI", "KOSDAQ", "KRX", "NASDAQ", "NYSE", "US", "SP500", "DOW"]
    )


def test_bucket_universe_by_atomic_market_splits_composite_bucket():
    def fake_fetch_universe(market):
        return {"KOSPI": ["005930"], "KOSDAQ": ["247540"]}[market]

    result = bucket_universe_by_atomic_market("KRX", fake_fetch_universe)

    assert result == {"KOSPI": ["005930"], "KOSDAQ": ["247540"]}


def test_bucket_universe_by_atomic_market_single_atomic_market():
    def fake_fetch_universe(market):
        return ["005930"]

    result = bucket_universe_by_atomic_market("KOSPI", fake_fetch_universe)

    assert result == {"KOSPI": ["005930"]}


def test_bucket_universe_by_atomic_market_filters_index_bucket_by_constituents(mocker):
    mocker.patch(
        "auto_stock.orchestrator.market_buckets.fetch_index_constituents",
        return_value={"AAPL"},
    )

    def fake_fetch_universe(market):
        return {"NASDAQ": ["AAPL", "MSFT"], "NYSE": ["JPM"]}[market]

    result = bucket_universe_by_atomic_market("SP500", fake_fetch_universe)

    assert result == {"NASDAQ": ["AAPL"], "NYSE": []}
