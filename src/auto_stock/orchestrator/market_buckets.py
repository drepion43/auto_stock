"""백그라운드 스캔 버킷(scan_key) → 원자(atomic) market 목록 매핑.

`data/models.py::VALID_MARKETS`는 종목 레코드가 실제로 속하는 원자 거래소값
(KOSPI/KOSDAQ/NASDAQ/NYSE)만 허용한다. 하지만 사용자는 "코스피+코스닥 통합",
"미국 전체", "S&P500 중에서"처럼 여러 원자 market을 합친 단위로 추천을 묻는다 —
이 모듈은 그 합성 단위(스캔 버킷)가 어떤 원자 market들의 조합인지만 정의한다.

`MARKET_BUCKETS`에 있는 버킷은 단순 합집합(get_universe_union)으로 유니버스를
만든다. `INDEX_FILTER_BUCKETS`에 있는 버킷(S&P500/다우존스)은 합집합을 구한 뒤
그 지수의 구성종목으로 다시 필터링한다 — 구성종목 조회 함수는
`data/sources/index_constituents.py`에 있다.

이 매핑에 없는 market 문자열(즉 원자 market 자신)은 "버킷이 곧 그 market 하나"로
취급한다(`bucket_markets()` 참고) — KOSPI/KOSDAQ/NASDAQ/NYSE 단독 스캔은 별도
엔트리가 필요 없다."""

from collections.abc import Callable

from auto_stock.data.sources.index_constituents import (
    fetch_dow_constituents,
    fetch_sp500_constituents,
)

MARKET_BUCKETS: dict[str, list[str]] = {
    "KRX": ["KOSPI", "KOSDAQ"],
    "US": ["NASDAQ", "NYSE"],
}

INDEX_FILTER_BUCKETS: dict[str, tuple[list[str], Callable[[], set[str]]]] = {
    "SP500": (["NASDAQ", "NYSE"], fetch_sp500_constituents),
    "DOW": (["NASDAQ", "NYSE"], fetch_dow_constituents),
}

# 백그라운드 스캔 대상 전체 — scripts/chat_cli.py, scripts/run_market_scan.py가 순회.
ALL_SCAN_BUCKETS = ["KOSPI", "KOSDAQ", "KRX", "NASDAQ", "NYSE", "US", "SP500", "DOW"]


def bucket_markets(bucket: str) -> list[str]:
    """스캔 버킷(합성 또는 원자)이 어떤 원자 market들로 구성되는지 반환한다."""
    if bucket in MARKET_BUCKETS:
        return MARKET_BUCKETS[bucket]
    if bucket in INDEX_FILTER_BUCKETS:
        return INDEX_FILTER_BUCKETS[bucket][0]
    return [bucket]  # 원자 market 자신(KOSPI/KOSDAQ/NASDAQ/NYSE) — 버킷==그 하나


def is_index_filter_bucket(bucket: str) -> bool:
    return bucket in INDEX_FILTER_BUCKETS


def fetch_index_constituents(bucket: str) -> set[str]:
    """`bucket`이 지수 필터 버킷이 아니면 ValueError — 호출 전 is_index_filter_bucket로
    확인해야 한다."""
    if bucket not in INDEX_FILTER_BUCKETS:
        raise ValueError(f"not an index filter bucket: {bucket!r}")
    _, fetch_constituents = INDEX_FILTER_BUCKETS[bucket]
    return fetch_constituents()


def bucket_universe_by_atomic_market(
    bucket: str, fetch_universe: Callable[[str], list[str]]
) -> dict[str, list[str]]:
    """스캔 버킷(원자 또는 합성)의 유니버스를 **원자 market별로 쪼갠 채** 반환한다
    — `{"KOSPI": [...], "KOSDAQ": [...]}` 형태. 합성 버킷("KRX"=코스피+코스닥 등)의
    티커를 하나의 리스트로 뭉치면, 그 뒤 `build_shortlist`/`scan_market`처럼 "market
    하나 + tickers 리스트"를 받는 함수들이 각 티커의 실제 원자 거래소를 알 수 없게
    된다(`OHLCVRecord`가 원자 market만 허용하므로 그대로 넘기면 ValueError) — 그래서
    호출부(orchestrator 코디네이터)가 이 dict를 순회하며 원자 market마다 별도로
    `build_shortlist`/`scan_market`을 호출하고 결과를 합치는 식으로 쓴다.

    `fetch_universe(atomic_market) -> list[str]`은 호출부가 캐싱 로직(_fresh_universe
    등)을 쥔 콜백이다 — 이 함수 자신은 캐싱을 모른다.

    지수 필터 버킷(SP500/DOW)은 원자 유니버스를 구한 뒤 구성종목 집합으로 교집합
    필터링한다 — 필터링 후에도 원자 market 구분(dict의 키)은 그대로 유지된다."""
    atomic_markets = bucket_markets(bucket)
    universe_by_market = {market: fetch_universe(market) for market in atomic_markets}

    if not is_index_filter_bucket(bucket):
        return universe_by_market

    constituents = fetch_index_constituents(bucket)
    return {
        market: [t for t in tickers if t in constituents]
        for market, tickers in universe_by_market.items()
    }
