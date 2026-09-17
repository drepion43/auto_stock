"""사용자가 채팅에 입력한 문자열(종목코드 또는 회사명)을 (ticker, market) 후보 목록으로
바꾼다 — DART/EDGAR의 기존 캐시 조회 함수를 재사용하며 새 외부 API 호출은 없다.

우선 KRX 6자리 숫자 / NASDAQ 대문자 심볼(1~5자 알파벳) 형태면 직접 캐시에 대조하는
shortcut을 시도하고, 그다음 한글 별칭 표(`_NASDAQ_KOREAN_ALIASES`)를 확인한다 — 둘 중
하나라도 성공하면 그것으로 끝(이름 검색 폴백을 타지 않는다). 둘 다 실패하면 DART/EDGAR
양쪽에서 `resolve_ticker_by_name`으로 이름 검색해 병합한다.

**한글 별칭이 필요한 이유**: EDGAR의 `resolve_ticker_by_name`(및 그 바탕인 SEC
`company_tickers.json`)은 영문 title만 갖고 있다 — "애플"/"테슬라" 같은 한글 질의는
그 이름 검색으로는 절대 매칭되지 않는다(실사용 중 발견, 2026-09-07). DART는 애초에
한글이 원본이라 이 문제가 없다. 별칭 표는 자주 쓰이는 종목만 다루는 근사 해법이고,
표에 없는 나스닥 종목은 여전히 영문 티커/영문 회사명으로 입력해야 한다 — 그 경우의
안내는 `prompt.py`의 시스템 프롬프트가 담당한다.

**모호(2건 이상 매치) 시 절대 임의로 고르지 않는다** — 잘못된 종목을 골라 5개 분석을
전부 그 종목에 대해 돌리는 것은 이 프로젝트가 다른 곳에서도 지키는 원칙(예: LLM 프롬프트에
실제 날짜/티커를 절대 주입하지 않는 환각 방어)과 같은 급의 리스크다. 호출부(`prompt.py`)가
후보를 나열해 사용자에게 되묻는다.
"""

from dataclasses import dataclass

from auto_stock.data.sources.dart_source import resolve_corp_name
from auto_stock.data.sources.dart_source import resolve_ticker_by_name as resolve_dart_ticker_by_name
from auto_stock.data.sources.edgar_source import resolve_company_title
from auto_stock.data.sources.edgar_source import resolve_ticker_by_name as resolve_edgar_ticker_by_name

_KRX_CODE_LENGTH = 6
_NASDAQ_SYMBOL_MAX_LENGTH = 5

# 자주 쓰이는 나스닥(EDGAR 등록) 종목의 한글 통칭 → 티커. 전체 커버리지가 아니라
# 근사 해법(모듈 docstring 참고) — 여기 없는 종목은 영문 티커/회사명으로 입력해야
# resolve_company_title로 매칭된다(2026-09-07 각 티커를 실제 EDGAR 캐시에 대조해 확인).
_NASDAQ_KOREAN_ALIASES: dict[str, str] = {
    "애플": "AAPL",
    "테슬라": "TSLA",
    "마이크로소프트": "MSFT",
    "엔비디아": "NVDA",
    "아마존": "AMZN",
    "구글": "GOOGL",
    "알파벳": "GOOGL",
    "메타": "META",
    "페이스북": "META",
    "넷플릭스": "NFLX",
    "인텔": "INTC",
    "퀄컴": "QCOM",
    "브로드컴": "AVGO",
    "어도비": "ADBE",
    "세일즈포스": "CRM",
    "오라클": "ORCL",
    "시스코": "CSCO",
    "페이팔": "PYPL",
    "스타벅스": "SBUX",
    "코스트코": "COST",
    "펩시": "PEP",
    "코카콜라": "KO",
    "월마트": "WMT",
    "디즈니": "DIS",
    "나이키": "NKE",
    "보잉": "BA",
    "골드만삭스": "GS",
    "제이피모건": "JPM",
    "비자": "V",
    "마스터카드": "MA",
    "존슨앤존슨": "JNJ",
    "화이자": "PFE",
    "모더나": "MRNA",
    "우버": "UBER",
    "에어비앤비": "ABNB",
    "스포티파이": "SPOT",
    "쇼피파이": "SHOP",
    "팔란티어": "PLTR",
    "엑슨모빌": "XOM",
    "버크셔해서웨이": "BRK-B",
}


@dataclass(frozen=True, slots=True)
class TickerMatch:
    ticker: str
    market: str
    name: str


@dataclass(frozen=True, slots=True)
class TickerResolution:
    matches: list[TickerMatch]

    @property
    def is_resolved(self) -> bool:
        return len(self.matches) == 1

    @property
    def is_ambiguous(self) -> bool:
        return len(self.matches) > 1

    @property
    def is_not_found(self) -> bool:
        return len(self.matches) == 0


def _try_direct_code_match(query: str) -> TickerMatch | None:
    if query.isdigit() and len(query) == _KRX_CODE_LENGTH:
        name = resolve_corp_name(query)
        return TickerMatch(ticker=query, market="KRX", name=name) if name is not None else None

    if query.isascii() and query.isalpha() and len(query) <= _NASDAQ_SYMBOL_MAX_LENGTH:
        upper = query.upper()
        title = resolve_company_title(upper)
        return TickerMatch(ticker=upper, market="NASDAQ", name=title) if title is not None else None

    return None


def _try_korean_alias_match(query: str) -> TickerMatch | None:
    ticker = _NASDAQ_KOREAN_ALIASES.get(query)
    if ticker is None:
        return None
    title = resolve_company_title(ticker)
    return TickerMatch(ticker=ticker, market="NASDAQ", name=title) if title is not None else None


def resolve_ticker(query: str) -> TickerResolution:
    query = query.strip()
    if not query:
        return TickerResolution(matches=[])

    direct = _try_direct_code_match(query) or _try_korean_alias_match(query)
    if direct is not None:
        return TickerResolution(matches=[direct])

    matches = [
        TickerMatch(ticker=ticker, market="KRX", name=name)
        for ticker, name in resolve_dart_ticker_by_name(query)
    ]
    matches += [
        TickerMatch(ticker=ticker, market="NASDAQ", name=name)
        for ticker, name in resolve_edgar_ticker_by_name(query)
    ]
    return TickerResolution(matches=matches)
