"""메인 대화 에이전트 도구들의 인자 모델 — `tool_registry.py`의 단일 정의 원천.

필드 순서·설명 문자열이 곧 OpenAI 함수 스키마의 `properties`/`required` 순서와
`description`이 된다(`tool_registry._property_schema`). 바꾸면 모델이 받는 바이트가
바뀌므로 `tests/chat_agent/test_tool_registry.py`의 골든 테스트가 잡는다.

이 모듈은 다른 chat_agent 모듈을 import하지 않는다(순환 import 방지의 바닥층,
docs/design/main-agent-migration-plan.md §4.0)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# 종목 단위 도구(6개)는 원자(atomic) 거래소 하나만 받는다 — OHLCVRecord가 원자값만
# 허용하므로(`data/models.py::VALID_MARKETS`) 합성 버킷을 받으면 그대로 전달했을 때
# ValueError가 난다.
AtomicMarket = Literal["KOSPI", "KOSDAQ", "NASDAQ", "NYSE"]

# 전체 스캔/섹터 추천 도구는 원자 market뿐 아니라 합성 버킷(KRX/US/SP500/DOW)도
# 받는다 — `orchestrator/market_buckets.ALL_SCAN_BUCKETS`와 반드시 동기화해야 한다.
ScanMarket = Literal["KOSPI", "KOSDAQ", "KRX", "NASDAQ", "NYSE", "US", "SP500", "DOW"]

_TICKER_DESC = (
    "종목 코드 (KOSPI/KOSDAQ: 6자리 숫자 예 005930, NASDAQ/NYSE: 티커 심볼 예 AAPL)"
)
_MARKET_DESC = "시장 구분(원자 거래소: KOSPI/KOSDAQ/NASDAQ/NYSE)"
_SCAN_MARKET_DESC = (
    "시장 구분 — 원자 거래소(KOSPI/KOSDAQ/NASDAQ/NYSE) 또는 합성 버킷"
    "(KRX=코스피+코스닥, US=나스닥+NYSE, SP500, DOW)"
)


class _StrictArgs(BaseModel):
    # 모델이 스키마에 없는 인자를 섞어 보내면 거부한다 — 스키마의 additionalProperties=false와 일치.
    model_config = ConfigDict(extra="forbid")


class TickerMarketArgs(_StrictArgs):
    ticker: str = Field(description=_TICKER_DESC)
    market: AtomicMarket = Field(description=_MARKET_DESC)


class QueryArgs(_StrictArgs):
    query: str = Field(description="사용자가 언급한 종목코드 또는 회사명")


class MarketArgs(_StrictArgs):
    market: ScanMarket = Field(description=_SCAN_MARKET_DESC)


class QueryMarketArgs(_StrictArgs):
    query: str = Field(description="섹터/업종/테마명(예: '로봇', '반도체', '2차전지')")
    market: ScanMarket = Field(description=_SCAN_MARKET_DESC)
