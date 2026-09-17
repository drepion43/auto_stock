"""네이버 뉴스 검색 API 연동 — 데이터 수집 계층(#0)의 확장, 뉴스 감성분석(#4-뉴스)
국내 소스(docs/design/news-sentiment-plan.md).

DART/EDGAR가 공시(기업이 직접 제출하는 공식 문서)를 다루는 것과 달리, 이 모듈은 진짜
뉴스 기사를 다룬다. 네이버 뉴스 검색 API는 종목코드가 아니라 회사명 문자열로 검색해야
하므로, 이 함수는 이미 해석된 회사명을 인자로 받는다 — DART의 corp_code 같은 자체
식별자 개념을 모른다(핵심 설계 결정 2).

크리덴셜은 `notifier/credentials.py`와 동일하게 필요 시점에 직접 `load_dotenv()`를
호출해 로드한다. `NAVER_CLIENT_ID`/`NAVER_CLIENT_SECRET`는 무료로 즉시 발급 가능하다.

**이용약관 주의**: 이 API는 검색 스니펫 제공이 목적이며, 상업적/자동화 대량 수집에
대한 이용약관 제약 가능성이 있다(PRD §10) — 실제 프로덕션 자동 실행(스케줄러)에 넣기
전 반드시 developers.naver.com에서 이용약관을 직접 확인해야 한다. 이 세션에서는
developers.naver.com 접근이 차단되어 재확인하지 못했다(계획 문서 Phase 0 미완료 항목).
"""

import os
import re
from datetime import date
from email.utils import parsedate_to_datetime
from html import unescape
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

from auto_stock.data.models import NewsArticle

NAVER_NEWS_SEARCH_URL = "https://naverapihub.apigw.ntruss.com/search/v1/news"
_REQUEST_TIMEOUT_SECONDS = 30
_DISPLAY_COUNT = 100  # 네이버 API의 건당 최대값

_HTML_TAG_RE = re.compile(r"<[^>]+>")


class NaverNewsApiError(Exception):
    pass


def _load_naver_credentials() -> tuple[str, str]:
    load_dotenv()
    return os.environ["NAVER_CLIENT_ID"], os.environ["NAVER_CLIENT_SECRET"]


def _clean_html(text: str) -> str:
    """네이버 응답의 title/description은 검색어를 <b> 태그로 감싸고 HTML 엔티티를
    이스케이프해서 보낸다(예: "&quot;") — 엔티티를 먼저 원문자로 되돌린 뒤 태그를
    제거한다. 순서가 중요하다: 태그 제거를 먼저 하면 `&lt;script&gt;`처럼 엔티티로
    이스케이프된 가짜 태그가 제거 단계를 통과한 뒤 언이스케이프되면서 진짜 태그로
    되살아난다(코드 리뷰 MEDIUM) — 언이스케이프를 먼저 하면 되살아난 태그도 뒤이은
    제거 단계에서 함께 걸러진다."""
    return _HTML_TAG_RE.sub("", unescape(text))


def _parse_pub_date(value: str) -> date:
    return parsedate_to_datetime(value).date()


def search_news(
    company_name: str, ticker: str, market: str, start: date, end: date
) -> list[NewsArticle]:
    """네이버 뉴스 검색(`news.json`) 호출. 결과 없으면 빈 리스트(예외 아님).

    네이버 API는 날짜 범위 파라미터를 지원하지 않으므로(정렬만 가능), 최신순으로
    최대 건수를 가져온 뒤 기간 필터링은 이 함수 안에서 클라이언트 사이드로 수행한다
    (dart_source.py/edgar_source.py와 동일 패턴). `ticker`/`market`은 응답에 없는
    정보라 호출자가 명시적으로 넘긴 값을 그대로 각 기사에 채운다."""
    client_id, client_secret = _load_naver_credentials()
    try:
        response = requests.get(
            NAVER_NEWS_SEARCH_URL,
            headers={"X-NCP-APIGW-API-KEY-ID": client_id, "X-NCP-APIGW-API-KEY": client_secret},
            params={"query": company_name, "display": _DISPLAY_COUNT, "sort": "date", "format":"json"},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        # requests/urllib3 예외 문자열에 요청 헤더/URL 정보가 담길 수 있으므로
        # dart_source.py/edgar_source.py와 동일 원칙으로 원본을 노출하지 않는다.
        raise NaverNewsApiError(f"네이버 뉴스 API 요청 실패: {type(exc).__name__}") from exc

    payload = response.json()

    articles = []
    for item in payload.get("items", []):
        try:
            published_at = _parse_pub_date(item["pubDate"])
        except (KeyError, ValueError, TypeError) as exc:
            raise NaverNewsApiError(f"네이버 뉴스 응답 파싱 실패: {type(exc).__name__}") from exc
        if not (start <= published_at <= end):
            continue
        original_link = item.get("originallink") or item.get("link") or ""
        articles.append(
            NewsArticle(
                ticker=ticker,
                market=market,
                title=_clean_html(item.get("title") or ""),
                url=item.get("link") or original_link,
                published_at=published_at,
                source=urlparse(original_link).netloc,
            )
        )
    return articles
