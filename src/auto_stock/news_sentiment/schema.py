"""`client.responses.parse(..., text_format=NewsSentimentRead)`가 검증하는 구조화 출력 스키마.

`sentiment` 필드명은 `news_disclosure/schema.py::DisclosureRead.market_impact`와 의도적으로
다르다 — 공시는 논조 없는 사실 통지라 "영향 해석"이지만, 뉴스 기사는 실제로 논조·어조가
있으므로 진짜 감성분석이다(docs/design/news-sentiment-plan.md 핵심 설계 결정 1,
news-disclosure-plan.md 핵심 설계 결정 1이 예약해둔 이름 구분). `confidence`가 서수인 이유,
목표가/수량 필드가 없는 이유는 llm_chart_analyst/schema.py의 동일 근거를 그대로 따른다
(중복 설명 생략).
"""

from typing import Literal

from pydantic import BaseModel, Field


class NewsSentimentRead(BaseModel):
    sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = Field(
        description="이 뉴스 기사들의 논조가 주가에 미칠 것으로 예상되는 영향 방향. 뚜렷하지 않으면 NEUTRAL."
    )
    confidence: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        description="주어진 기사 목록이 이 판단을 얼마나 명확히 지지하는지. 확률/퍼센트가 아님."
    )
    key_headline: str = Field(
        max_length=60, description="가장 중요한 기사 1건의 제목 요약(한국어)"
    )
    rationale: str = Field(
        max_length=200, description="어떤 기사가 근거인지 구체적으로 언급한 한국어 1~2문장."
    )
    caveat: str | None = Field(
        default=None, max_length=120, description="이 판단을 무효화할 수 있는 조건. 없으면 null."
    )
