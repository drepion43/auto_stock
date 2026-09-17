"""`client.responses.parse(..., text_format=DisclosureRead)`가 검증하는 구조화 출력 스키마.

`market_impact`를 `sentiment`가 아닌 이 이름으로 둔 이유는 news-disclosure-plan.md 핵심 설계
결정 1 참고 — 공시는 뉴스처럼 어조가 있는 게 아니라 기업이 제출하는 사실 통지이므로 "감성"이
아니라 "통상적 영향 해석"이다. `confidence`가 서수인 이유, 목표가/수량 필드가 없는 이유는
llm_chart_analyst/schema.py의 동일 근거를 그대로 따른다(중복 설명 생략).
"""

from typing import Literal

from pydantic import BaseModel, Field


class DisclosureRead(BaseModel):
    market_impact: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = Field(
        description="이 공시들이 통상적으로 주가에 미치는 영향 방향. 뚜렷하지 않으면 NEUTRAL."
    )
    confidence: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        description="주어진 공시 목록이 이 판단을 얼마나 명확히 지지하는지. 확률/퍼센트가 아님."
    )
    key_event: str = Field(
        max_length=60, description="가장 중요한 공시 1건의 핵심 내용 요약(한국어)"
    )
    rationale: str = Field(
        max_length=200, description="어떤 공시가 근거인지 구체적으로 언급한 한국어 1~2문장."
    )
    caveat: str | None = Field(
        default=None, max_length=120, description="이 판단을 무효화할 수 있는 조건. 없으면 null."
    )
