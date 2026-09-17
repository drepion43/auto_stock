"""SEC EDGAR(미국 전자공시) OpenAPI 연동 — 데이터 수집 계층(#0)의 확장.

`dart_source.py`와 같은 위치: 나스닥 상장사의 공시 데이터를 보조한다. EDGAR는 KRX
종목코드 같은 개념이 없고 자체 CIK(Central Index Key)로 종목을 식별하므로, 전체
매핑(`company_tickers.json`)을 다운로드해 로컬에 캐싱하고 종목코드로 역조회한다
(docs/design/news-disclosure-nasdaq-plan.md 핵심 설계 결정 2). DART와 달리 API 키가
아니라 SEC 정책상 필수인 `User-Agent`(연락처 포함 식별 문자열)가 필요하다(같은 문서
핵심 설계 결정 1) — 비밀은 아니지만 없으면 즉시 실패하는 계약은 동일하게 유지한다.
"""

import json
import os
import time
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

from auto_stock.data.models import DisclosureRecord

CIK_MAP_CACHE_PATH = Path("data/edgar_cik_map.json")
CIK_MAP_REFRESH_DAYS = 7
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"

_REQUEST_TIMEOUT_SECONDS = 30

_FORM_TYPE_LABELS = {
    "10-K": "연차보고서(10-K)",
    "10-K/A": "연차보고서 정정(10-K/A)",
    "10-Q": "분기보고서(10-Q)",
    "10-Q/A": "분기보고서 정정(10-Q/A)",
    "8-K": "주요사항보고(8-K)",
    "8-K/A": "주요사항보고 정정(8-K/A)",
    "S-1": "증권신고서(S-1)",
    "DEF 14A": "위임장권유서류(DEF 14A)",
    "SC 13D": "지분보고(SC 13D, 5% 이상 취득)",
    "SC 13G": "지분보고(SC 13G, 수동적 투자자)",
}
# 임원·주요주주 지분변동 신고(Form 3/4/5)는 대형주 기준 주 단위로 여러 건씩 쏟아져
# MAX_DISCLOSURES_PER_QUERY 윈도우를 노이즈로 채운다 — 기본 제외한다
# (docs/design/news-disclosure-nasdaq-plan.md 핵심 설계 결정 3).
_EXCLUDED_FORM_TYPES = frozenset({"3", "4", "5"})


class EdgarApiError(Exception):
    pass


def _load_edgar_user_agent() -> str:
    load_dotenv()
    return os.environ["SEC_EDGAR_USER_AGENT"]


def _download_cik_map() -> dict[str, dict[str, str]]:
    """`company_tickers.json`을 다운로드해 {ticker: {"cik":..., "title":...}} 매핑으로
    파싱한다. `title`(회사명)은 뉴스 감성분석(#4-뉴스) 확장에서 뉴스 검색 쿼리를 만들
    때 쓰인다(docs/design/news-sentiment-plan.md 핵심 설계 결정 2) — 이전에는 파싱만
    하고 버렸다."""
    user_agent = _load_edgar_user_agent()
    try:
        response = requests.get(
            COMPANY_TICKERS_URL,
            headers={"User-Agent": user_agent},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        # dart_source.py와 동일 원칙: 원본 예외를 문자열화하지 않고 타입명만 남긴다
        # (docs/design/news-disclosure-nasdaq-plan.md 핵심 설계 결정 7).
        raise EdgarApiError(f"EDGAR API 요청 실패: {type(exc).__name__}") from exc

    payload = response.json()
    mapping: dict[str, dict[str, str]] = {}
    for entry in payload.values():
        ticker = entry.get("ticker")
        cik = entry.get("cik_str")
        title = entry.get("title") or ""
        if ticker and cik is not None:
            mapping[ticker] = {"cik": f"{int(cik):010d}", "title": title}
    return mapping


def _is_cache_fresh(cache_path: Path, refresh_days: int) -> bool:
    if not cache_path.exists():
        return False
    age_seconds = time.time() - cache_path.stat().st_mtime
    return age_seconds < refresh_days * 86400


def _load_or_refresh_cik_map(cache_path: Path, refresh_days: int) -> dict:
    if _is_cache_fresh(cache_path, refresh_days):
        return json.loads(cache_path.read_text(encoding="utf-8"))
    mapping = _download_cik_map()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # 임시파일 + os.replace()로 원자적으로 쓴다 — dart_source.py의 캐시 쓰기가
    # 리뷰에서 지적받았던 것과 같은 문제(중간에 프로세스가 죽으면 손상된 캐시가
    # refresh_days 동안 자연 복구되지 않음)를 처음부터 방지한다.
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    tmp_path.write_text(json.dumps(mapping), encoding="utf-8")
    os.replace(tmp_path, cache_path)
    return mapping


def _cik_and_title(entry) -> tuple[str | None, str | None]:
    """캐시 항목에서 (cik, title)을 뽑는다. `entry`가 문자열이면 뉴스 감성분석 확장
    전(title이 없던 시절)에 저장된 구 스키마 캐시다 — cik만 반환하고 title은 None
    (하위호환, docs/design/news-sentiment-plan.md 핵심 설계 결정 2)."""
    if entry is None:
        return None, None
    if isinstance(entry, str):
        return entry, None
    return entry.get("cik"), entry.get("title")


def resolve_cik(
    ticker: str,
    cache_path: Path | str = CIK_MAP_CACHE_PATH,
    refresh_days: int = CIK_MAP_REFRESH_DAYS,
) -> str | None:
    """`company_tickers.json` 캐시에서 티커로 CIK(10자리 0-패딩)를 역조회한다.

    캐시가 없거나 `refresh_days`보다 오래되면 재다운로드 후 캐시를 갱신한다.
    한국 상장사 등 EDGAR에 등록되지 않은 티커는 None을 반환한다(에러 아님)."""
    mapping = _load_or_refresh_cik_map(Path(cache_path), refresh_days)
    cik, _ = _cik_and_title(mapping.get(ticker))
    return cik


def resolve_company_title(
    ticker: str,
    cache_path: Path | str = CIK_MAP_CACHE_PATH,
    refresh_days: int = CIK_MAP_REFRESH_DAYS,
) -> str | None:
    """`company_tickers.json` 캐시에서 티커로 회사명(title)을 역조회한다. 뉴스 검색
    API는 티커가 아니라 회사명 문자열로 질의해야 하므로(docs/design/
    news-sentiment-plan.md 핵심 설계 결정 2), 그 쿼리 입력을 만드는 데 쓰인다.
    `resolve_cik`와 캐시를 공유한다 — 같은 티커에 대해 두 함수를 순서대로 호출해도
    EDGAR API를 두 번 부르지 않는다."""
    mapping = _load_or_refresh_cik_map(Path(cache_path), refresh_days)
    _, title = _cik_and_title(mapping.get(ticker))
    return title


def resolve_ticker_by_name(
    name_query: str,
    cache_path: Path | str = CIK_MAP_CACHE_PATH,
    refresh_days: int = CIK_MAP_REFRESH_DAYS,
) -> list[tuple[str, str]]:
    """`company_tickers.json` 캐시에서 회사명 문자열로 티커를 역조회한다
    (`resolve_company_title`의 반대 방향) — 대화형 챗봇(#Stage B)이 사용자가 입력한
    회사명을 티커로 바꿀 때 쓴다. 새 EDGAR API 호출 없이 기존 캐시를 재사용한다.

    매칭 규칙은 `dart_source.resolve_ticker_by_name`과 동일하다: 대소문자 무시
    완전일치가 있으면 그것만 반환하고, 없으면 부분문자열 포함 여부로 폴백한다.
    이름이 없는 구 스키마(문자열) 캐시 항목은 매칭 대상에서 제외한다."""
    mapping = _load_or_refresh_cik_map(Path(cache_path), refresh_days)
    query_casefold = name_query.casefold()

    named_entries = [
        (ticker, title)
        for ticker, entry in mapping.items()
        for _, title in [_cik_and_title(entry)]
        if title
    ]

    exact_matches = [(t, n) for t, n in named_entries if n.casefold() == query_casefold]
    if exact_matches:
        return exact_matches
    return [(t, n) for t, n in named_entries if query_casefold in n.casefold()]


def _label_form_type(form: str) -> str:
    return _FORM_TYPE_LABELS.get(form, form)


def fetch_filings(cik: str, start: date, end: date) -> list[DisclosureRecord]:
    """SEC 제출이력(`submissions`) 호출. Form 3/4/5는 제외, 결과 없으면 빈 리스트(예외 아님).

    `submissions` API는 날짜 범위 파라미터를 지원하지 않고 최근 1년/1000건을 컬럼형
    배열로 돌려주므로, 기간 필터링은 이 함수 안에서 클라이언트 사이드로 수행한다."""
    user_agent = _load_edgar_user_agent()
    url = SUBMISSIONS_URL_TEMPLATE.format(cik=cik)
    try:
        response = requests.get(
            url, headers={"User-Agent": user_agent}, timeout=_REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise EdgarApiError(f"EDGAR API 요청 실패: {type(exc).__name__}") from exc

    payload = response.json()
    tickers = payload.get("tickers") or [""]
    ticker = tickers[0]
    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])

    records = []
    for form, filing_date_str in zip(forms, filing_dates):
        # 정정판(예: "4/A")은 원본 폼 코드와 정확히 일치하지 않아 단순 in-체크로는
        # 걸러지지 않는다 — "/" 앞부분만 비교해 정정판도 함께 제외한다(코드 리뷰 MEDIUM).
        if form.split("/")[0] in _EXCLUDED_FORM_TYPES:
            continue
        try:
            filed_date = date.fromisoformat(filing_date_str)
        except ValueError as exc:
            # requests 예외와 동일 원칙: 원본 예외를 문자열화하지 않고 타입명만 남긴다
            # (보안 리뷰 LOW — SEC가 예상 밖 날짜 형식을 보낼 가능성에 대한 방어).
            raise EdgarApiError(f"EDGAR 응답 파싱 실패: {type(exc).__name__}") from exc
        if not (start <= filed_date <= end):
            continue
        records.append(
            DisclosureRecord(
                corp_code=cik,
                ticker=ticker,
                report_name=_label_form_type(form),
                filed_date=filed_date,
                remark="",
            )
        )
    return records
