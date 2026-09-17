"""데이터 수집 소스 개별 스모크 테스트 — DART/EDGAR/네이버/GDELT 4개 함수를 하나씩
따로 실행해서 직접 눈으로 결과를 확인하기 위한 스크립트.

기존 scripts/verify_news_disclosure.py, scripts/verify_news_sentiment.py는 시장(KRX/
NASDAQ) 분기 + LLM 해석 계층까지 포함한 end-to-end 검증인 반면, 이 스크립트는 데이터
수집 계층(#0) 4개 소스 각각의 연결성/응답 형태만 독립적으로 확인한다. OpenAI는 호출하지
않는다(비용 0).

사용법 (프로젝트 루트에서 실행해야 .env가 로드됨):
    .venv/Scripts/python.exe scripts/verify_sources_individually.py dart
    .venv/Scripts/python.exe scripts/verify_sources_individually.py edgar
    .venv/Scripts/python.exe scripts/verify_sources_individually.py naver
    .venv/Scripts/python.exe scripts/verify_sources_individually.py gdelt
    .venv/Scripts/python.exe scripts/verify_sources_individually.py all   (4개 순서대로, 하나 실패해도 계속)

필요 크리덴셜(.env): DART_API_KEY, SEC_EDGAR_USER_AGENT, NAVER_CLIENT_ID/NAVER_CLIENT_SECRET.
GDELT는 크리덴셜 불필요.
"""

import sys
from datetime import date, timedelta

_SAMPLE_TICKER_KRX = "005930"  # 삼성전자
_SAMPLE_TICKER_NASDAQ = "AAPL"
_LOOKBACK_DAYS = 14


def _print_disclosures(records: list) -> None:
    for item in records[:5]:
        print(f"  - {item.filed_date} {item.report_name}" + (f" [{item.remark}]" if item.remark else ""))
    if len(records) > 5:
        print(f"  ... 외 {len(records) - 5}건")


def _print_articles(articles: list) -> None:
    for item in articles[:5]:
        print(f"  - {item.published_at} [{item.source}] {item.title}")
    if len(articles) > 5:
        print(f"  ... 외 {len(articles) - 5}건")


def test_dart() -> None:
    from auto_stock.data.sources.dart_source import (
        DartApiError,
        fetch_disclosures,
        resolve_corp_code,
        resolve_corp_name,
    )

    print("=== DART (KRX 공시) ===")
    try:
        corp_code = resolve_corp_code(_SAMPLE_TICKER_KRX)
        corp_name = resolve_corp_name(_SAMPLE_TICKER_KRX)
        print(f"resolve_corp_code({_SAMPLE_TICKER_KRX}) = {corp_code}")
        print(f"resolve_corp_name({_SAMPLE_TICKER_KRX}) = {corp_name}")
        if corp_code is None:
            print("FAILED: corp_code를 찾지 못했습니다 (매핑 캐시 확인 필요)")
            return

        end = date.today()
        start = end - timedelta(days=_LOOKBACK_DAYS)
        disclosures = fetch_disclosures(corp_code, start, end)
        print(f"SUCCESS: 공시 {len(disclosures)}건 (최근 {_LOOKBACK_DAYS}일)")
        _print_disclosures(disclosures)
    except DartApiError as exc:
        print(f"FAILED: {exc}")
    except KeyError as exc:
        print(f"FAILED: 크리덴셜이 .env에 설정되어 있지 않습니다 ({exc})")


def test_edgar() -> None:
    from auto_stock.data.sources.edgar_source import (
        EdgarApiError,
        fetch_filings,
        resolve_cik,
        resolve_company_title,
    )

    print("=== EDGAR (NASDAQ 공시) ===")
    try:
        cik = resolve_cik(_SAMPLE_TICKER_NASDAQ)
        title = resolve_company_title(_SAMPLE_TICKER_NASDAQ)
        print(f"resolve_cik({_SAMPLE_TICKER_NASDAQ}) = {cik}")
        print(f"resolve_company_title({_SAMPLE_TICKER_NASDAQ}) = {title}")
        if cik is None:
            print("FAILED: cik를 찾지 못했습니다 (매핑 캐시 확인 필요)")
            return

        end = date.today()
        start = end - timedelta(days=_LOOKBACK_DAYS)
        filings = fetch_filings(cik, start, end)
        print(f"SUCCESS: 공시 {len(filings)}건 (최근 {_LOOKBACK_DAYS}일)")
        _print_disclosures(filings)
    except EdgarApiError as exc:
        print(f"FAILED: {exc}")
    except KeyError as exc:
        print(f"FAILED: 크리덴셜이 .env에 설정되어 있지 않습니다 ({exc})")


def test_naver() -> None:
    from auto_stock.data.sources.naver_news_source import NaverNewsApiError, search_news

    print("=== 네이버 뉴스 (KRX 감성) ===")
    try:
        end = date.today()
        start = end - timedelta(days=_LOOKBACK_DAYS)
        articles = search_news("삼성전자", _SAMPLE_TICKER_KRX, "KRX", start, end)
        print(f"SUCCESS: 기사 {len(articles)}건 (최근 {_LOOKBACK_DAYS}일)")
        _print_articles(articles)
    except NaverNewsApiError as exc:
        print(f"FAILED: {exc}")
    except KeyError as exc:
        print(f"FAILED: 크리덴셜이 .env에 설정되어 있지 않습니다 ({exc})")


def test_gdelt() -> None:
    from auto_stock.data.sources.gdelt_source import GdeltApiError, search_articles

    print("=== GDELT (NASDAQ 감성) ===")
    try:
        end = date.today()
        start = end - timedelta(days=_LOOKBACK_DAYS)
        articles = search_articles('"Apple Inc."', _SAMPLE_TICKER_NASDAQ, "NASDAQ", start, end)
        print(f"SUCCESS: 기사 {len(articles)}건 (최근 {_LOOKBACK_DAYS}일)")
        _print_articles(articles)
    except GdeltApiError as exc:
        print(f"FAILED: {exc}")


# _TESTS = {"dart": test_dart, "edgar": test_edgar, "naver": test_naver, "gdelt": test_gdelt}

_TESTS = {"dart": test_dart, "edgar": test_edgar, "naver": test_naver, "gdelt":test_gdelt}

if __name__ == "__main__":
    if sys.stdout.encoding.lower() != "utf-8":  # Windows 콘솔 기본 cp949는 em dash(—) 등을 못 그림
        sys.stdout.reconfigure(encoding="utf-8")

    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target == "all":
        for name, fn in _TESTS.items():
            fn()
            print()
    elif target in _TESTS:
        _TESTS[target]()
    else:
        print(f"사용법: python {sys.argv[0]} [dart|edgar|naver|gdelt|all]")
        sys.exit(1)
