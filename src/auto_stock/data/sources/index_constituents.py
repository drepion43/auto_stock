"""지수 구성종목 조회 — S&P500/다우존스. `orchestrator/market_buckets.py`의
`INDEX_FILTER_BUCKETS`("SP500"/"DOW" 스캔 버킷)가 유니버스를 필터링할 때 쓴다.

S&P500은 FDR이 이미 지원한다(`fdr.StockListing("S&P500")`, Wikipedia
`List_of_S%26P_500_companies`를 스크래핑하는 `WikipediaStockListing` 클래스).
다우존스는 FDR에 없다(소스코드 확인 — `WikipediaStockListing`은 S&P500 URL이
클래스 내부에 하드코딩돼 있어 그대로 재사용할 수 없다) — 그래서 이 모듈이 다우존스
위키 페이지를 FDR과 동일한 패턴(`requests` + `pandas.read_html`)으로 직접
스크래핑한다. 다우존스는 30개뿐이라 스크래핑이 실패해도(위키 페이지 구조 변경 등)
영향 범위가 작다 — 실패 시 예외를 그대로 던지고, 호출부(`market_buckets.py`가
반환하는 콜백을 쓰는 orchestrator 쪽)가 그 실패를 어떻게 처리할지 결정한다(현재는
백그라운드 스캔 스레드의 "절대 전파 안 함" catch-all이 흡수한다).

둘 다 7일 TTL 로컬 캐시를 쓴다(DART/EDGAR corp_code 캐시와 동일한 "임시파일 +
os.replace" 원자적 쓰기 패턴 재사용) — Wikipedia/FDR을 매 스캔마다 다시 긁지 않는다."""

import json
import os
import time
from io import StringIO
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd
import requests

SP500_CACHE_PATH = Path("data/sp500_constituents.json")
DOW_CACHE_PATH = Path("data/dow_constituents.json")
CONSTITUENTS_REFRESH_DAYS = 7

DOW_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies"
_REQUEST_TIMEOUT_SECONDS = 30
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
)


class IndexConstituentsError(Exception):
    pass


def _is_cache_fresh(cache_path: Path, refresh_days: int) -> bool:
    if not cache_path.exists():
        return False
    age_seconds = time.time() - cache_path.stat().st_mtime
    return age_seconds < refresh_days * 86400


def _load_or_refresh(cache_path: Path, refresh_days: int, download) -> list[str]:
    if _is_cache_fresh(cache_path, refresh_days):
        return json.loads(cache_path.read_text(encoding="utf-8"))
    tickers = download()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # DART/EDGAR 캐시와 동일한 원자적 쓰기 패턴(임시파일 + os.replace).
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    tmp_path.write_text(json.dumps(tickers), encoding="utf-8")
    os.replace(tmp_path, cache_path)
    return tickers


def _download_sp500_tickers() -> list[str]:
    listing = fdr.StockListing("S&P500")
    return listing["Symbol"].tolist()


def _find_symbol_column(columns) -> str:
    for col in columns:
        if "symbol" in str(col).lower() or "ticker" in str(col).lower():
            return col
    raise IndexConstituentsError(
        f"다우존스 위키 테이블에서 종목코드 컬럼을 찾지 못했습니다 (columns={list(columns)})"
    )


def _download_dow_tickers() -> list[str]:
    try:
        response = requests.get(
            DOW_WIKIPEDIA_URL,
            headers={"User-Agent": _USER_AGENT},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        # 다른 소스 모듈과 동일 원칙 — 원본 예외를 문자열화하지 않고 타입명만 남긴다.
        raise IndexConstituentsError(f"다우존스 구성종목 조회 실패: {type(exc).__name__}") from exc

    tables = pd.read_html(StringIO(response.text), header=0)
    for table in tables:
        if any("symbol" in str(c).lower() or "ticker" in str(c).lower() for c in table.columns):
            symbol_col = _find_symbol_column(table.columns)
            symbols = table[symbol_col].astype(str).str.replace(".", "", regex=False).tolist()
            # 다우존스는 30개 — 비정상적으로 적거나 많으면 잘못된 테이블을 집은 것일 수 있다.
            if 20 <= len(symbols) <= 40:
                return symbols
    raise IndexConstituentsError("다우존스 위키 페이지에서 구성종목 테이블을 찾지 못했습니다")


def fetch_sp500_constituents() -> set[str]:
    """S&P500 구성종목 티커 집합을 반환한다. 7일 TTL 로컬 캐시."""
    tickers = _load_or_refresh(SP500_CACHE_PATH, CONSTITUENTS_REFRESH_DAYS, _download_sp500_tickers)
    return set(tickers)


def fetch_dow_constituents() -> set[str]:
    """다우존스산업평균 구성종목(30개) 티커 집합을 반환한다. 7일 TTL 로컬 캐시."""
    tickers = _load_or_refresh(DOW_CACHE_PATH, CONSTITUENTS_REFRESH_DAYS, _download_dow_tickers)
    return set(tickers)
