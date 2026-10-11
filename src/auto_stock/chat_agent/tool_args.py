"""메인 대화 에이전트 도구들의 인자 모델 — `tool_registry.py`의 단일 정의 원천.

필드 순서·설명 문자열이 곧 OpenAI 함수 스키마의 `properties`/`required` 순서와
`description`이 된다(`tool_registry._property_schema`). 바꾸면 모델이 받는 바이트가
바뀌므로 `tests/chat_agent/test_tool_registry.py`의 골든 테스트가 잡는다.

이 모듈은 다른 chat_agent 모듈을 import하지 않는다(순환 import 방지의 바닥층,
docs/design/main-agent-migration-plan.md §4.0)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Market = Literal["KRX", "NASDAQ"]

_TICKER_DESC = "종목 코드 (KRX: 6자리 숫자 예 005930, NASDAQ: 티커 심볼 예 AAPL)"
_MARKET_DESC = "시장 구분"


class _StrictArgs(BaseModel):
    # 모델이 스키마에 없는 인자를 섞어 보내면 거부한다 — 스키마의 additionalProperties=false와 일치.
    model_config = ConfigDict(extra="forbid")


class TickerMarketArgs(_StrictArgs):
    ticker: str = Field(description=_TICKER_DESC)
    market: Market = Field(description=_MARKET_DESC)


class QueryArgs(_StrictArgs):
    query: str = Field(description="사용자가 언급한 종목코드 또는 회사명")


class MarketArgs(_StrictArgs):
    market: Market = Field(description=_MARKET_DESC)


class QueryMarketArgs(_StrictArgs):
    query: str = Field(description="섹터/업종/테마명(예: '로봇', '반도체', '2차전지')")
    market: Market = Field(description=_MARKET_DESC)
