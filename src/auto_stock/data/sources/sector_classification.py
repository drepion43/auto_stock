"""공식 업종 분류(KRX 전용) — recommendation-synthesis-plan.md §5 1차 매칭 경로.
pykrx의 지수 API(`get_index_ticker_list`/`get_index_ticker_name`/
`get_index_portfolio_deposit_file`)로 업종명↔코드 매핑과 구성종목을 조회한다.

2026-10-05 실제 pykrx 호출로 확인: KRX_ID/KRX_PW 로그인이 필요하고(`.env`,
`pykrx_source.py`와 동일 패턴), 그걸 빼먹으면 모든 호출이 "Expecting value"로
실패한다(라이브러리 버그가 아니라 로그인 선행 누락이 원인이었음) — 그래서 모든 공개
함수가 호출 직전에 `load_dotenv()`를 부른다.

**NASDAQ은 지원하지 않는다** — pykrx는 한국거래소 전용 스크래퍼라 미국 종목의 공식
업종 분류를 제공하지 않는다(사용자 확인, 2026-10-05) — NASDAQ 섹터 질의는 전부 2차
LLM 제안+뉴스검증 경로로만 처리된다.

코스피/코스닥에 동명 업종(예: "전기전자")이 둘 다 있을 수 있어(실제 확인:
KOSPI 1013/KOSDAQ 2072 둘 다 "전기전자"), `fetch_sector_names`는 업종명 하나당
코드를 리스트로 묶어 반환한다 — `fetch_constituents`에 그 리스트를 그대로 넘기면
양쪽 시장의 구성종목을 합집합으로 받을 수 있다."""

import contextlib
import io

from dotenv import load_dotenv

# pykrx는 import되는 순간 KRX 로그인을 시도한다 — pykrx_source.py와 동일 이유로
# import 시점 노이즈만 억제한다(실제 함수 호출 시의 출력은 그대로 둔다).
with contextlib.redirect_stdout(io.StringIO()):
    from pykrx import stock

_SUB_MARKETS = ("KOSPI", "KOSDAQ")  # "KRX"는 이 둘을 합친 것 — 이 프로젝트의 기존 관례


def fetch_sector_names(market: str) -> dict[str, list[str]]:
    """업종명 -> 지수코드 목록. market이 "KRX"가 아니면 빈 dict(pykrx 대상이 아닌
    시장은 조용히 스킵 — 호출부가 2차 LLM 폴백으로 넘어가면 됨, 에러 아님)."""
    if market != "KRX":
        return {}

    load_dotenv()
    names: dict[str, list[str]] = {}
    for sub_market in _SUB_MARKETS:
        for code in stock.get_index_ticker_list(market=sub_market):
            name = stock.get_index_ticker_name(code)
            names.setdefault(name, []).append(code)
    return names


def fetch_constituents(index_codes: list[str]) -> list[str]:
    """여러 지수 코드(동명 업종의 코스피+코스닥 코드 등)의 구성종목을 중복 없이,
    처음 등장한 순서를 보존해 합집합으로 반환한다."""
    if not index_codes:
        return []

    load_dotenv()
    tickers: list[str] = []
    seen: set[str] = set()
    for code in index_codes:
        for ticker in stock.get_index_portfolio_deposit_file(code):
            if ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)
    return tickers
