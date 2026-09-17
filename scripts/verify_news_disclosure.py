"""Manual verification: confirms the 뉴스/공시 분석 에이전트(#4) wiring end-to-end — DART(KRX)
and EDGAR(NASDAQ) both (docs/design/news-disclosure-nasdaq-plan.md).

Run from the project root so `.env` resolves correctly:
    .venv/Scripts/python scripts/verify_news_disclosure.py                          (Windows, KRX/DART real calls)
    NEWS_VERIFY_MARKET=NASDAQ .venv/Scripts/python scripts/verify_news_disclosure.py (NASDAQ/EDGAR real calls)
    NEWS_VERIFY_DRY_RUN=1 .venv/Scripts/python scripts/verify_news_disclosure.py     (dry-run, DART/EDGAR only)
    .venv/bin/python scripts/verify_news_disclosure.py                               (macOS/Linux, KRX/DART real calls)

Two independent axes:

- `NEWS_VERIFY_MARKET` (`KRX` 기본값, 또는 `NASDAQ`): 어느 시장/소스 쌍을 검증할지 —
  KRX는 DART(`resolve_corp_code`/`fetch_disclosures`, `DART_API_KEY` 필요), NASDAQ은
  EDGAR(`resolve_cik`/`fetch_filings`, `SEC_EDGAR_USER_AGENT` 필요, 비밀키 아님).
- `NEWS_VERIFY_DRY_RUN=1`: 해당 시장의 데이터 수집 함수를 실제로 호출하되(DART/EDGAR 둘 다
  무료) OpenAI 클라이언트는 만들지 않고 렌더링된 유저 프롬프트만 출력.
- 그 외(기본값): 추가로 `OPENAI_API_KEY`가 필요. `load_llm_config()` → `max_calls_per_run=1`로
  강제한 `OpenAIDisclosureClient` → `analyze()` 실행 후 구조화 결과 출력.

샘플 종목에 최근 공시가 없으면(`analyze()`가 `None` 반환) 정상 흐름으로 안내하고 실패
취급하지 않는다(공시가 없는 경우가 정상, docs/design/news-disclosure-plan.md 핵심 설계
결정 5).

Prints only counts/labels/rationale text — never prints DART_API_KEY, SEC_EDGAR_USER_AGENT,
or OPENAI_API_KEY.
"""

import dataclasses
import os
import sys
from datetime import date, timedelta

from auto_stock.data.sources.dart_source import fetch_disclosures, resolve_corp_code
from auto_stock.data.sources.edgar_source import fetch_filings, resolve_cik
from auto_stock.news_disclosure.analyst import analyze
from auto_stock.news_disclosure.credentials import (
    DISCLOSURE_LOOKBACK_DAYS,
    MAX_DISCLOSURES_PER_QUERY,
    load_llm_config,
)
from auto_stock.news_disclosure.models import DisclosureSummary
from auto_stock.news_disclosure.prompt import build_user_prompt

_SAMPLE_TICKERS = {"KRX": "005930", "NASDAQ": "AAPL"}  # 삼성전자 / Apple — 각 시장의 스모크 테스트 타깃
_MISSING_CREDENTIAL_ENV_VAR = {"KRX": "DART_API_KEY", "NASDAQ": "SEC_EDGAR_USER_AGENT"}


def _resolve_sample_market() -> str:
    market = os.environ.get("NEWS_VERIFY_MARKET", "KRX").upper()
    if market not in _SAMPLE_TICKERS:
        print(f"FAILED: NEWS_VERIFY_MARKET은 {sorted(_SAMPLE_TICKERS)} 중 하나여야 합니다 (입력값: {market!r})")
        sys.exit(1)
    return market


def _fetch_sample_disclosures(market: str) -> tuple[list, date]:
    ticker = _SAMPLE_TICKERS[market]
    corp_id = resolve_corp_code(ticker) if market == "KRX" else resolve_cik(ticker)
    if corp_id is None:
        source = "DART" if market == "KRX" else "EDGAR"
        print(f"FAILED: {ticker}의 {source} 식별자를 찾을 수 없습니다 (매핑 확인 필요)")
        sys.exit(1)

    end = date.today()
    start = end - timedelta(days=DISCLOSURE_LOOKBACK_DAYS)
    disclosures = fetch_disclosures(corp_id, start, end) if market == "KRX" else fetch_filings(corp_id, start, end)
    return disclosures, end


def _run_dry_run(market: str) -> None:
    source_label = "DART" if market == "KRX" else "EDGAR"
    ticker = _SAMPLE_TICKERS[market]
    disclosures, as_of = _fetch_sample_disclosures(market)
    print(f"SUCCESS: {source_label} 공시 조회 완료 (OpenAI 호출 없음, 비용 0) — {len(disclosures)}건")
    if not disclosures:
        print("최근 공시 없음 — 정상 흐름(analyze()가 호출되면 None을 반환하고 OpenAI 호출 자체가 발생하지 않음)")
        return

    recent = sorted(disclosures, key=lambda item: item.filed_date, reverse=True)
    summary = DisclosureSummary(
        ticker=ticker,
        market=market,
        as_of=as_of,
        items=recent[:MAX_DISCLOSURES_PER_QUERY],
    )
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

    from auto_stock.news_disclosure.client import NewsDisclosureError, OpenAIDisclosureClient

    client = OpenAIDisclosureClient(config)
    ticker = _SAMPLE_TICKERS[market]
    disclosures, as_of = _fetch_sample_disclosures(market)

    try:
        analysis = analyze(client, disclosures, ticker, market, as_of)
    except NewsDisclosureError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)

    if analysis is None:
        print("공시 없음 — 정상 흐름 (최근 공시가 없어 LLM 호출 자체가 발생하지 않았습니다)")
        return

    print("SUCCESS: 뉴스/공시 분석 응답 수신")
    print(f"모델: {analysis.model}")
    print(f"market_impact: {analysis.market_impact}")
    print(f"confidence: {analysis.confidence}")
    print(f"key_event: {analysis.key_event}")
    print(f"rationale: {analysis.rationale}")
    print(f"caveat: {analysis.caveat}")


if __name__ == "__main__":
    if sys.stdout.encoding.lower() != "utf-8":  # Windows 콘솔 기본 cp949는 em dash(—) 등을 못 그림
        sys.stdout.reconfigure(encoding="utf-8")
    market = _resolve_sample_market()
    try:
        if os.environ.get("NEWS_VERIFY_DRY_RUN") == "1":
            _run_dry_run(market)
        else:
            _run_real_call(market)
    except KeyError:
        missing = _MISSING_CREDENTIAL_ENV_VAR[market]
        print(f"FAILED: {missing}가 .env에 설정되어 있지 않습니다")
        sys.exit(1)
