"""`client.responses.parse(..., text_format=ChartPatternRead)`가 검증하는 구조화 출력 스키마.

`confidence`가 0~1 실수가 아니라 3단계 서수인 이유: (a) LLM이 낸 수치 신뢰도는 캘리브레이션되지
않아 통계적 의미가 없다, (b) ML #2가 이미 "상승확률 XX%" 표기를 점유했으므로 두 번째 퍼센트가
같은 메시지에 나오면 사용자가 비교 가능한 척도로 오독한다. 목표가/손절가/수량 필드가 없는 이유는
그 값들이 `risk_sizing`(ATR 기반)의 소관이라 스키마 수준에서 충돌을 원천 차단하기 위함이다.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ChartPatternRead(BaseModel):
    direction: Literal["UP", "DOWN", "NEUTRAL"] = Field(
        description="향후 약 5거래일 방향성. 뚜렷한 패턴이 없으면 반드시 NEUTRAL."
    )
    confidence: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        description="주어진 수치가 이 판단을 얼마나 명확히 지지하는지. 확률/퍼센트가 아님."
    )
    pattern_name: str = Field(
        max_length=40,
        description="감지된 대표 차트/캔들 패턴 이름(한국어). 없으면 '뚜렷한 패턴 없음'.",
    )
    rationale: str = Field(
        max_length=200,
        description="어떤 수치가 근거인지 구체적으로 언급한 한국어 1~2문장.",
    )
    caveat: str | None = Field(
        default=None,
        max_length=120,
        description="이 판단을 무효화할 수 있는 조건. 없으면 null.",
    )
