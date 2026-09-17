"""news_disclosure 도메인 모델. llm_chart_analyst/models.py와 동일하게 frozen dataclass +
slots로 불변 데이터를 표현한다.

`DisclosureReader`는 Protocol이므로 `analyst.py`가 `openai`를 import하지 않고도 이 신호원의
도메인 로직(analyze/to_reasons)을 테스트할 수 있다 — 가짜 리더만 있으면 된다.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

from auto_stock.data.models import DisclosureRecord
from auto_stock.news_disclosure.schema import DisclosureRead


@dataclass(frozen=True, slots=True)
class DisclosureSummary:
    ticker: str  # 프롬프트에 넣지 않음 — 결과 라벨링 전용
    market: str  # 동일
    as_of: date  # 동일
    items: list[DisclosureRecord]  # 최근 N건, 최신순


@dataclass(frozen=True, slots=True)
class DisclosureAnalysis:
    ticker: str
    market: str
    as_of: date
    market_impact: str  # "POSITIVE" | "NEGATIVE" | "NEUTRAL"
    confidence: str  # "LOW" | "MEDIUM" | "HIGH"
    key_event: str
    rationale: str
    caveat: str | None
    model: str  # 실제 사용한 모델 ID — 감사 추적용, 문구에는 넣지 않음


@dataclass(frozen=True, slots=True)
class LLMConfig:
    api_key: str = field(repr=False)  # never let dataclass repr/str echo the secret
    model: str
    max_tokens: int
    timeout_seconds: float
    max_retries: int
    max_calls_per_run: int


class DisclosureReader(Protocol):
    model: str  # DisclosureAnalysis.model(감사 추적용) 채우기 위해 필요

    def read_disclosures(self, system_prompt: str, user_prompt: str) -> DisclosureRead: ...
