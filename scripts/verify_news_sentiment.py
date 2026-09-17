"""Manual verification: confirms the 뉴스 감성분석 에이전트(#4-뉴스) wiring end-to-end —
네이버(KRX)와 GDELT(NASDAQ) 둘 다(docs/design/news-sentiment-plan.md).

Run from the project root so `.env` resolves correctly:
    .venv/Scripts/python scripts/verify_news_sentiment.py                           (Windows, KRX/네이버 real calls)
    NEWS_SENTIMENT_VERIFY_MARKET=NASDAQ .venv/Scripts/python scripts/verify_news_sentiment.py  (NASDAQ/GDELT real calls)
    NEWS_SENTIMENT_VERIFY_DRY_RUN=1 .venv/Scripts/python scripts/verify_news_sentiment.py       (dry-run, 뉴스 소스만)
    .venv/bin/python scripts/verify_news_sentiment.py                                (macOS/Linux, KRX/네이버 real calls)

Two independent axes:

- `NEWS_SENTIMENT_VERIFY_MARKET` (`KRX` 기본값, 또는 `NASDAQ`): 어느 시장/소스 쌍을
  검증할지 — KRX는 네이버 뉴스 검색(`resolve_corp_name`/`search_news`, `NAVER_CLIENT_ID`/
  `NAVER_CLIENT_SECRET` 필요), NASDAQ은 GDELT(`resolve_company_title`/`search_articles`,
  크리덴셜 불필요).
- `NEWS_SENTIMENT_VERIFY_DRY_RUN=1`: 해당 시장의 뉴스 검색 함수를 실제로 호출하되
  (네이버/GDELT 둘 다 무료) OpenAI 클라이언트는 만들지 않고 렌더링된 유저 프롬프트만 출력.
- 그 외(기본값): 추가로 `OPENAI_API_KEY`가 필요. `load_llm_config()` → `max_calls_per_run=1`로
  강제한 `OpenAINewsSentimentClient` → `analyze()` 실행 후 구조화 결과 출력.

**주의**: GDELT `seendate` 필드 형식은 이 세션에서 라이브 검증하지 못했다(네트워크 제약,
docs/design/news-sentiment-plan.md Phase 0 미완료 항목) — NASDAQ 모드는 실제 실행 전
`gdelt_source.py`의 `_SEENDATE_FORMAT` 가정이 맞는지 먼저 확인할 것.

샘플 종목에 최근 기사가 없으면(`analyze()`가 `None` 반환) 정상 흐름으로 안내하고 실패
취급하지 않는다.

Prints only counts/labels/rationale text — never prints NAVER_CLIENT_ID, NAVER_CLIENT_SECRET,
or OPENAI_API_KEY.
"""

import dataclasses
import os
import sys
from datetime import date, timedelta

from auto_stock.data.sources.dart_source import resolve_corp_name
from auto_stock.data.sources.edgar_source import resolve_company_title
from auto_stock.data.sources.gdelt_source import search_articles
from auto_stock.data.sources.naver_news_source import search_news
from auto_stock.news_sentiment.analyst import analyze
from auto_stock.news_sentiment.credentials import (
    MAX_ARTICLES_PER_QUERY,
    NEWS_LOOKBACK_DAYS,
    load_llm_config,
)
from auto_stock.news_sentiment.models import NewsSummary
from auto_stock.news_sentiment.prompt import build_user_prompt

_SAMPLE_TICKERS = {"KRX": "005930", "NASDAQ": "AAPL"}  # 삼성전자 / Apple — 각 시장의 스모크 테스트 타깃


def _resolve_sample_market() -> str:
    market = os.environ.get("NEWS_SENTIMENT_VERIFY_MARKET", "KRX").upper()
    if market not in _SAMPLE_TICKERS:
        print(f"FAILED: NEWS_SENTIMENT_VERIFY_MARKET은 {sorted(_SAMPLE_TICKERS)} 중 하나여야 합니다 (입력값: {market!r})")
        sys.exit(1)
    return market


def _fetch_sample_articles(market: str) -> tuple[list, str, date]:
    ticker = _SAMPLE_TICKERS[market]
    name = resolve_corp_name(ticker) if market == "KRX" else resolve_company_title(ticker)
    if name is None:
        source = "DART" if market == "KRX" else "EDGAR"
        print(f"FAILED: {ticker}의 회사명을 {source} 매핑에서 찾을 수 없습니다")
        sys.exit(1)

    end = date.today()
    start = end - timedelta(days=NEWS_LOOKBACK_DAYS)
    articles = (
        search_news(name, ticker, market, start, end)
        if market == "KRX"
        else search_articles(f'"{name}"', ticker, market, start, end)
    )
    return articles, ticker, end


def _run_dry_run(market: str) -> None:
    source_label = "네이버뉴스" if market == "KRX" else "GDELT"
    articles, ticker, as_of = _fetch_sample_articles(market)
    print(f"SUCCESS: {source_label} 조회 완료 (OpenAI 호출 없음, 비용 0) — {len(articles)}건")
    if not articles:
        print("최근 기사 없음 — 정상 흐름(analyze()가 호출되면 None을 반환하고 OpenAI 호출 자체가 발생하지 않음)")
        return

    recent = sorted(articles, key=lambda item: item.published_at, reverse=True)
    summary = NewsSummary(ticker=ticker, market=market, as_of=as_of, items=recent[:MAX_ARTICLES_PER_QUERY])
    print()
    print("--- USER PROMPT ---")
    print(build_user_prompt(summary))


def _run_real_call(market: str) -> None:
    try:
        config = load_llm_config()
    except KeyError:
        print("FAILED: OPENAI_API_KEY가 .env에 설정되어 있지 않습니다")
        sys.exit(1)

    config = dataclasses.replace(config, max_calls_per_run=1)

    from auto_stock.news_sentiment.client import NewsSentimentError, OpenAINewsSentimentClient

    client = OpenAINewsSentimentClient(config)
    articles, ticker, as_of = _fetch_sample_articles(market)

    try:
        analysis = analyze(client, articles, ticker, market, as_of)
    except NewsSentimentError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)

    if analysis is None:
        print("최근 기사 없음 — 정상 흐름 (LLM 호출 자체가 발생하지 않았습니다)")
        return

    print("SUCCESS: 뉴스 감성분석 응답 수신")
    print(f"모델: {analysis.model}")
    print(f"sentiment: {analysis.sentiment}")
    print(f"confidence: {analysis.confidence}")
    print(f"key_headline: {analysis.key_headline}")
    print(f"rationale: {analysis.rationale}")
    print(f"caveat: {analysis.caveat}")


if __name__ == "__main__":
    if sys.stdout.encoding.lower() != "utf-8":  # Windows 콘솔 기본 cp949는 em dash(—) 등을 못 그림
        sys.stdout.reconfigure(encoding="utf-8")
    market = _resolve_sample_market()
    try:
        if os.environ.get("NEWS_SENTIMENT_VERIFY_DRY_RUN") == "1":
            _run_dry_run(market)
        else:
            _run_real_call(market)
    except KeyError as exc:
        print(f"FAILED: 크리덴셜이 .env에 설정되어 있지 않습니다 ({exc})")
        sys.exit(1)
