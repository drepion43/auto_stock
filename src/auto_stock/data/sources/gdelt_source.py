"""GDELT DOC 2.0 API 연동 — 데이터 수집 계층(#0)의 확장, 뉴스 감성분석(#4-뉴스)
미국 소스(docs/design/news-sentiment.md).

DART/EDGAR가 공시를 다루는 것과 달리, 이 모듈은 진짜 뉴스 기사를 다룬다. GDELT는
API 키가 필요 없는 무료 공개 API — 크리덴셜 로딩 자체가 없다.

**주의**: `seendate` 필드 형식(`%Y%m%dT%H%M%SZ`)은 GDELT 공식 문서·커뮤니티 자료
(예: DOC 2.0 API 날짜 파라미터가 `YYYYMMDDHHMMSS` 형식을 쓰는 것과 일관된 패턴)로
교차 확인한 값이며, 이 세션에서는 네트워크 제약(이 환경에서 api.gdeltproject.org로의
아웃바운드 연결이 모두 타임아웃됨 — WebFetch·curl 둘 다 확인)으로 실제 라이브 응답을
직접 검증하지 못했다(docs/design/news-sentiment.md Phase 0 미완료 항목). 실제
사용 전 반드시 라이브 호출로 재확인해야 한다.

**요청 제한(2026-10-12 라이브 실측)**: GDELT는 "Please limit requests to one every 5 seconds"와 함께
HTTP 429를 돌려주며, 짧게 몰아 보낸 뒤에는 수십 초 간격에도 429가 섞인다. 서브에이전트가 도구를
병렬로 부르거나 여러 종목을 연달아 분석하면 "뉴스 조회 실패"가 잦았던 원인이다. 그래서 이 모듈이
프로세스 전체의 요청 간격을 직렬로 조절하고(`_MIN_INTERVAL_SECONDS`), 429를 받으면 다음 허용 시각을
`_RATE_LIMIT_BACKOFF_SECONDS`만큼 미룬 뒤 한 번만 재시도한다.

**검색어(`company_phrase`)**: SEC 법인명("NVIDIA CORP", "APPLE INC")을 그대로 정확구문으로 검색하면
뉴스 표기("Nvidia")를 대부분 놓친다(실측 14일: "NVIDIA CORP" 27건 vs "NVIDIA" 50건 이상). 법인
접미사를 떼어 검색한다. 흔한 단어와 겹치는 이름(Apple, Meta, Oracle)은 무관한 기사가 섞일 수 있다 —
감성 해석 LLM이 관련성을 일부 거르며, 문제가 확인되면 예외 목록을 둘 것.
"""

import threading
import time
from datetime import date, datetime

import requests

from auto_stock.data.models import NewsArticle

GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_REQUEST_TIMEOUT_SECONDS = 30
_MAX_RECORDS = 50

_SEENDATE_FORMAT = "%Y%m%dT%H%M%SZ"

_MIN_INTERVAL_SECONDS = 5.5  # 공지된 "5초에 1회" + 여유
_RATE_LIMIT_BACKOFF_SECONDS = 15.0  # 429 후 실측상 5초 간격으로는 부족했다
_RATE_LIMIT_STATUS = 429

# 스레드 간 공유 상태 — 다음 요청이 허용되는 시각(monotonic). 테스트가 `_now`/`_sleep`을 바꾼다.
_throttle_lock = threading.Lock()
_next_allowed_at: float | None = None
_now = time.monotonic
_sleep = time.sleep

# 뉴스 기사에는 거의 쓰이지 않는 법인 접미사(SEC company_tickers.json 표기 기준).
_LEGAL_SUFFIXES = frozenset({
    "CORP", "CORPORATION", "INC", "INCORPORATED", "CO", "COMPANY", "COM", "LTD", "LIMITED",
    "PLC", "LLC", "LP", "HOLDINGS", "HOLDING", "GROUP", "NV", "SA", "AG", "SE", "/DE/", "/NEW/",
})


class GdeltApiError(Exception):
    pass


def company_phrase(name: str) -> str:
    """회사명 → GDELT 정확구문 검색어. 따옴표를 제거하고(구문 문법 보호) 끝의 법인 접미사를
    떼어낸다. 전부 떼면 아무것도 안 남는 이름은 원래 이름을 그대로 쓴다."""
    tokens = name.replace('"', "").split()
    kept = list(tokens)
    while kept and kept[-1].strip(".,").upper() in _LEGAL_SUFFIXES:
        kept.pop()
    phrase = " ".join(kept).rstrip(",") if kept else " ".join(tokens)
    return f'"{phrase}"'


def _wait_for_slot() -> None:
    """프로세스 전체에서 요청을 `_MIN_INTERVAL_SECONDS` 간격으로 직렬화한다(잠금을 쥔 채 기다려
    다른 스레드도 순서대로 기다리게 한다)."""
    global _next_allowed_at
    with _throttle_lock:
        now = _now()
        if _next_allowed_at is not None and now < _next_allowed_at:
            _sleep(_next_allowed_at - now)
            now = _next_allowed_at
        _next_allowed_at = now + _MIN_INTERVAL_SECONDS


def _back_off() -> None:
    global _next_allowed_at
    with _throttle_lock:
        _next_allowed_at = max(_next_allowed_at or 0.0, _now() + _RATE_LIMIT_BACKOFF_SECONDS)


def _get(params: dict) -> requests.Response:
    for attempt in range(2):
        _wait_for_slot()
        response = requests.get(GDELT_DOC_API_URL, params=params, timeout=_REQUEST_TIMEOUT_SECONDS)
        if getattr(response, "status_code", 200) == _RATE_LIMIT_STATUS and attempt == 0:
            _back_off()
            continue
        response.raise_for_status()
        return response
    raise AssertionError("unreachable")  # 두 번째 시도는 반환하거나 raise_for_status로 끝난다


def _parse_seendate(value: str) -> date:
    return datetime.strptime(value, _SEENDATE_FORMAT).date()


def search_articles(
    query: str, ticker: str, market: str, start: date, end: date
) -> list[NewsArticle]:
    """GDELT DOC 2.0 기사 검색(`mode=artlist`) 호출. 결과 없으면 빈 리스트(예외 아님).

    `query`는 호출자가 완성한 검색 질의(정확 구문 검색이 필요하면 호출자가 따옴표로
    감싸 전달 — 이 함수는 질의 구성 로직을 모른다, docs/design/news-sentiment.md
    핵심 설계 결정 3). GDELT는 `startdatetime`/`enddatetime` 파라미터를 지원하지만,
    타임존/경계 처리 불확실성을 피하기 위해 dart_source.py/edgar_source.py와 동일하게
    최신순으로 가져온 뒤 클라이언트 사이드로 기간을 필터링한다. `ticker`/`market`은
    응답에 없는 정보라 호출자가 명시적으로 넘긴 값을 그대로 각 기사에 채운다."""
    try:
        response = _get(
            {
                "query": query,
                "mode": "artlist",
                "format": "json",
                "maxrecords": _MAX_RECORDS,
                "sort": "datedesc",
            }
        )
    except requests.exceptions.RequestException as exc:
        # dart_source.py/edgar_source.py와 동일 원칙: 원본 예외를 문자열화하지 않고
        # 타입명만 남긴다.
        raise GdeltApiError(f"GDELT API 요청 실패: {type(exc).__name__}") from exc

    payload = response.json()

    articles = []
    for item in payload.get("articles", []):
        try:
            seen_date = _parse_seendate(item["seendate"])
        except (KeyError, ValueError, TypeError) as exc:
            # seendate가 JSON null이면 datetime.strptime이 ValueError가 아니라 TypeError를
            # 던진다 — naver_news_source.py의 동일 클래스 실패(pubDate가 null)와 대칭으로
            # TypeError도 잡는다(코드 리뷰 MEDIUM).
            raise GdeltApiError(f"GDELT 응답 파싱 실패: {type(exc).__name__}") from exc
        if not (start <= seen_date <= end):
            continue
        articles.append(
            NewsArticle(
                ticker=ticker,
                market=market,
                title=item.get("title") or "",
                url=item.get("url") or "",
                published_at=seen_date,
                source=item.get("domain") or "",
            )
        )
    return articles
