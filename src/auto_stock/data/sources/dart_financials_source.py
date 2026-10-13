"""DART 재무제표(단일회사 전체) API 연동 — PSR(시가총액/매출액) 계산용 매출액 조회.

`dart_source.py`와 같은 패턴을 따른다: `DART_API_KEY`는 호출 시점에 직접
`load_dotenv()`로 로드하고, 원본 예외(요청 URL에 크리덴셜 포함)는 절대 노출하지
않고 타입명만 남긴다. 매출액 계정명이 회사마다 다르다(제조업 "매출액", 일부
"영업수익", 은행 "이자수익") — 사용자 결정(간단한 폴백 리스트 방식, 완전 XBRL
account_id 표준화는 하지 않음): `_REVENUE_ACCOUNT_NAMES`에 있는 이름 중 하나와
정확히 일치하는 첫 항목을 쓰고, 못 찾으면 None(에러 아님). 금융업(은행/보험/
증권)은 DART 재무제표 API 자체가 수집 대상에서 제외하므로 이 경우도 None으로
조용히 저하된다."""

import json
import os
import time
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

FINANCIALS_URL = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"

_STATUS_OK = "000"
_STATUS_NO_DATA = "013"

_REQUEST_TIMEOUT_SECONDS = 30

# 연간 사업보고서(사업연도 전체) — 분기 보고서는 범위 밖(연간이 가장 안정적으로
# 전체 재무제표를 포함하므로 매출액 계정 매칭 신뢰도가 가장 높다).
_REPRT_CODE_ANNUAL = "11011"

# 매출액 계정명 폴백 리스트 — 순서대로 매칭 시도. 제조업은 "매출액", 유통/서비스업
# 일부는 "영업수익", 은행은 "이자수익"을 쓴다. 이 중 하나와 정확히 일치하는 첫
# 항목을 쓰고, 아무것도 못 찾으면 PSR을 None으로 둔다(에러 아님).
_REVENUE_ACCOUNT_NAMES = ["매출액", "영업수익", "이자수익"]

REVENUE_CACHE_PATH = Path("data/dart_revenue_cache.json")
REVENUE_CACHE_TTL_DAYS = 365  # 연간 재무제표라 회계연도 단위로만 바뀐다.


class DartFinancialsError(Exception):
    pass


def _load_dart_api_key() -> str:
    load_dotenv()
    return os.environ["DART_API_KEY"]


def _fetch_financials(corp_code: str, bsns_year: str, fs_div: str) -> list[dict]:
    """단일회사 전체 재무제표 조회. 결과 없으면(금융업 제외 등) 빈 리스트(예외 아님)."""
    api_key = _load_dart_api_key()
    try:
        response = requests.get(
            FINANCIALS_URL,
            params={
                "crtfc_key": api_key,
                "corp_code": corp_code,
                "bsns_year": bsns_year,
                "reprt_code": _REPRT_CODE_ANNUAL,
                "fs_div": fs_div,
            },
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        # requests/urllib3는 예외 문자열에 요청 URL(크리덴셜이 담긴 쿼리스트링 포함)을
        # 그대로 포함시킨다 — 원본 예외를 절대 노출하지 않고 타입명만 남긴다.
        raise DartFinancialsError(f"DART API 요청 실패: {type(exc).__name__}") from exc
    payload = response.json()

    status = payload.get("status")
    if status == _STATUS_NO_DATA:
        return []
    if status != _STATUS_OK:
        raise DartFinancialsError(f"DART API 오류(status={status}): {payload.get('message')}")

    return payload.get("list", [])


def _extract_revenue(items: list[dict]) -> int | None:
    """`account_nm`이 폴백 리스트 중 하나와 정확히 일치하는 첫 항목의 `thstrm_amount`를
    정수로 반환한다. 못 찾으면 None."""
    for account_name in _REVENUE_ACCOUNT_NAMES:
        for item in items:
            if item.get("account_nm") == account_name:
                amount = (item.get("thstrm_amount") or "").replace(",", "").strip()
                if not amount or amount in ("-",):
                    continue
                try:
                    return int(amount)
                except ValueError:
                    continue
    return None


def _fetch_revenue_for_year(corp_code: str, bsns_year: str) -> int | None:
    """연결(CFS) 재무제표에서 매출액을 찾고, 없으면 별도(OFS)로 1회 재시도한다 —
    금융지주사처럼 연결만 있고 별도가 없는 경우/반대 케이스를 모두 흡수한다."""
    items = _fetch_financials(corp_code, bsns_year, fs_div="CFS")
    revenue = _extract_revenue(items)
    if revenue is not None:
        return revenue

    items = _fetch_financials(corp_code, bsns_year, fs_div="OFS")
    return _extract_revenue(items)


def _accounting_year_as_of(as_of: date) -> str:
    """`as_of` 시점에 실제로 공시돼 있을 가장 최근 확정 사업연도를 반환한다.

    사업보고서는 사업연도 종료 후 통상 약 3개월 뒤에 공시된다(lookahead 방어) —
    `as_of`가 4월 1일 이전이면 전년도 사업보고서조차 아직 안 나왔을 수 있으므로
    보수적으로 전전년도까지 내려간다."""
    if as_of.month >= 4:
        return str(as_of.year - 1)
    return str(as_of.year - 2)


def _is_cache_fresh(cache_path: Path, ttl_days: int) -> bool:
    if not cache_path.exists():
        return False
    age_seconds = time.time() - cache_path.stat().st_mtime
    return age_seconds < ttl_days * 86400


def _cache_key(corp_code: str, bsns_year: str) -> str:
    return f"{corp_code}_{bsns_year}"


def _load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # 임시파일 + os.replace()로 원자적으로 쓴다(dart_source.py/pykrx_source.py와 동일 패턴).
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    tmp_path.write_text(json.dumps(cache), encoding="utf-8")
    os.replace(tmp_path, cache_path)


def fetch_latest_revenue(
    corp_code: str,
    as_of: date,
    cache_path: Path | str = REVENUE_CACHE_PATH,
    cache_ttl_days: int = REVENUE_CACHE_TTL_DAYS,
) -> int | None:
    """`as_of` 시점에 실제로 공시돼 있던 가장 최근 사업연도의 매출액(원).

    매출액 계정 자체가 없는 종목(금융지주사 등) 또는 금융업(DART 재무제표 API가
    수집 대상에서 제외)은 None을 반환한다(에러 아님) — PSR이 조용히 빠질 뿐 전체
    파이프라인을 막지 않는다. 연간 재무제표라 회계연도 단위로만 바뀌므로
    `{corp_code}_{bsns_year}` 키로 영구(TTL 1년) 캐싱한다 — 빈 결과(None)는
    캐싱하지 않는다(2026-10-13 pykrx_source.py에서 실제 발생한 "빈 결과를 캐싱하면
    일시적 실패가 영구화된다" 교훈과 동일한 이유)."""
    bsns_year = _accounting_year_as_of(as_of)
    cache_path = Path(cache_path)
    cache_key = _cache_key(corp_code, bsns_year)

    if _is_cache_fresh(cache_path, cache_ttl_days):
        cached = _load_cache(cache_path)
        if cache_key in cached:
            return cached[cache_key]

    revenue = _fetch_revenue_for_year(corp_code, bsns_year)
    if revenue is None:
        return None

    cache = _load_cache(cache_path) if cache_path.exists() else {}
    cache[cache_key] = revenue
    _save_cache(cache_path, cache)
    return revenue
