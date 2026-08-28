"""llm_chart_analyst 도메인 모델. ml_predictor/models.py와 동일하게 frozen dataclass + slots로
불변 데이터를 표현한다.

`ChartPatternReader`는 Protocol이므로 `analyst.py`가 `openai`를 import하지 않고도 이 신호원의
도메인 로직(analyze/to_reasons)을 테스트할 수 있다 — 가짜 리더만 있으면 된다.
"""

from dataclasses import dataclass
from datetime import date
from typing import Protocol

from auto_stock.llm_chart_analyst.schema import ChartPatternRead


@dataclass(frozen=True, slots=True)
class BarSummary:
    offset: int  # t-29 … t-0 (음수 아님, 0이 최신)
    open: float  # 구간 첫 종가 = 100 기준 정규화 지수
    high: float
    low: float
    close: float
    volume_ratio: float  # 20일 평균 거래량 = 1.0 기준


@dataclass(frozen=True, slots=True)
class ChartSnapshot:
    ticker: str  # 프롬프트에 넣지 않음 — 결과 라벨링 전용
    market: str  # 동일
    date: date  # 최신 거래일, 동일
    bars: list[BarSummary]
    indicators: dict[str, float]  # ml_predictor.features.FEATURE_NAMES 11종 그대로


@dataclass(frozen=True, slots=True)
class ChartAnalysis:
    ticker: str
    market: str
    date: date
    direction: str  # "UP" | "DOWN" | "NEUTRAL"
    confidence: str  # "LOW" | "MEDIUM" | "HIGH"
    pattern_name: str
    rationale: str
    caveat: str | None
    model: str  # 실제 사용한 모델 ID — 감사 추적용, 문구에는 넣지 않음


@dataclass(frozen=True, slots=True)
class LLMConfig:
    api_key: str
    model: str
    max_tokens: int
    timeout_seconds: float
    max_retries: int
    max_calls_per_run: int


class ChartPatternReader(Protocol):
    model: str  # ChartAnalysis.model(감사 추적용) 채우기 위해 필요 — 계획 문서의 Protocol을
    # method 하나만으로는 표현할 수 없던 정보라 이 태스크에서 속성으로 보강했다.

    def read_pattern(self, system_prompt: str, user_prompt: str) -> ChartPatternRead: ...
