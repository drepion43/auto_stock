"""GDELT DOC 2.0 API 연동 — 데이터 수집 계층(#0)의 확장, 뉴스 감성분석(#4-뉴스)
미국 소스(docs/design/news-sentiment-plan.md).

DART/EDGAR가 공시를 다루는 것과 달리, 이 모듈은 진짜 뉴스 기사를 다룬다. GDELT는
API 키가 필요 없는 무료 공개 API — 크리덴셜 로딩 자체가 없다.

**주의**: `seendate` 필드 형식(`%Y%m%dT%H%M%SZ`)은 GDELT 공식 문서·커뮤니티 자료
(예: DOC 2.0 API 날짜 파라미터가 `YYYYMMDDHHMMSS` 형식을 쓰는 것과 일관된 패턴)로
교차 확인한 값이며, 이 세션에서는 네트워크 제약(이 환경에서 api.gdeltproject.org로의
아웃바운드 연결이 모두 타임아웃됨 — WebFetch·curl 둘 다 확인)으로 실제 라이브 응답을
직접 검증하지 못했다(docs/design/news-sentiment-plan.md Phase 0 미완료 항목). 실제
사용 전 반드시 라이브 호출로 재확인해야 한다.
"""

from datetime import date, datetime

import requests

from auto_stock.data.models import NewsArticle

GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_REQUEST_TIMEOUT_SECONDS = 30
_MAX_RECORDS = 50

_SEENDATE_FORMAT = "%Y%m%dT%H%M%SZ"


class GdeltApiError(Exception):
    pass


def _parse_seendate(value: str) -> date:
    return datetime.strptime(value, _SEENDATE_FORMAT).date()


def search_articles(
    query: str, ticker: str, market: str, start: date, end: date
) -> list[NewsArticle]:
    """GDELT DOC 2.0 기사 검색(`mode=artlist`) 호출. 결과 없으면 빈 리스트(예외 아님).

    `query`는 호출자가 완성한 검색 질의(정확 구문 검색이 필요하면 호출자가 따옴표로
    감싸 전달 — 이 함수는 질의 구성 로직을 모른다, docs/design/news-sentiment-plan.md
    핵심 설계 결정 3). GDELT는 `startdatetime`/`enddatetime` 파라미터를 지원하지만,
    타임존/경계 처리 불확실성을 피하기 위해 dart_source.py/edgar_source.py와 동일하게
    최신순으로 가져온 뒤 클라이언트 사이드로 기간을 필터링한다. `ticker`/`market`은
    응답에 없는 정보라 호출자가 명시적으로 넘긴 값을 그대로 각 기사에 채운다."""
    try:
        response = requests.get(
            GDELT_DOC_API_URL,
            params={
                "query": query,
                "mode": "artlist",
                "format": "json",
                "maxrecords": _MAX_RECORDS,
                "sort": "datedesc",
            },
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
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
