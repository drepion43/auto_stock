"""chat_agent 도메인 모델. `llm_chart_analyst`/`news_disclosure`/`news_sentiment`와 동일하게
`LLMConfig`는 frozen dataclass + slots로 불변 설정을 표현한다.

`QueryBudget`는 이 프로젝트에서 유일하게 **의도적으로 mutable**한 dataclass다 — 다른 모든
dataclass(`LLMConfig` 포함)는 불변 도메인 데이터를 표현하지만, 이건 대화 한 턴 동안
소비되는 런타임 카운터라서 매 LLM 호출마다 갱신돼야 한다. frozen으로 두면 매 호출마다
`dataclasses.replace()`로 새 인스턴스를 만들어 호출부 전체에 리턴값으로 전파해야 하고,
그러면 오히려 예산 상태를 어디서 관리하는지 더 헷갈리게 된다(배치 파이프라인의
`max_calls_per_run`은 클라이언트 인스턴스 자체의 프로세스 수명 카운터라 이 문제가 없었다 —
`QueryBudget`은 턴마다 새로 만들어지는 짧은 런타임 상태라는 점이 다르다).

`ChatAgentReader`는 Protocol이므로 `loop.py`/`tools.py`가 `openai`를 import하지 않고도
대화 루프 로직을 테스트할 수 있다 — 가짜 리더만 있으면 된다.
"""

import threading
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class LLMConfig:
    api_key: str = field(repr=False)  # never let dataclass repr/str echo the secret
    model: str
    max_tokens: int
    timeout_seconds: float
    max_retries: int
    max_calls_per_run: int


@dataclass(slots=True)  # 의도적으로 not frozen — 위 모듈 docstring 참고
class QueryBudget:
    max_llm_calls: int
    llm_calls_made: int = 0
    # recommendation-synthesis-plan.md §3 — run_deep_scan이 ThreadPoolExecutor로 여러
    # 종목의 run_stock_analyst를 동시에 호출하면서 같은 budget 인스턴스를 공유한다(이전엔
    # 항상 대화 1턴 안에서 단일 스레드로만 쓰였다). 락 없는 check-then-increment는
    # 레이스 컨디션으로 과소비될 수 있어 내부 전용 락을 추가한다 — repr/비교에서는
    # 제외(Lock은 비교 불가능하고 디버그 출력에도 의미 없음).
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def try_consume_llm_call(self) -> bool:
        """예산이 남아 있으면 1건 소비하고 True, 없으면 False(호출부는 이걸 도구 결과의
        `available=False` 사유로 그대로 반영한다 — raise하지 않음)."""
        with self._lock:
            if self.llm_calls_made >= self.max_llm_calls:
                return False
            self.llm_calls_made += 1
            return True

    def try_consume_llm_calls(self, n: int) -> bool:
        """deepagents 서브에이전트(find_related_companies/stock_analyst) 호출 전 보수적
        추정치를 한 번에 선차감하기 위한 원자적(all-or-nothing) 벌크 소비. 서브에이전트
        내부의 실제 LLM 호출수는 QueryBudget이 직접 셀 수 없으므로(내부 루프가 이 객체를
        모름) 이 근사 방식을 쓴다 — 부분 소비 후 실패는 절대 없다."""
        with self._lock:
            if self.llm_calls_made + n > self.max_llm_calls:
                return False
            self.llm_calls_made += n
            return True


@dataclass(frozen=True, slots=True)
class FunctionCall:
    name: str
    arguments: str  # 파싱 안 된 raw JSON 문자열 — SDK가 검증하지 않으므로 호출부가 처리
    call_id: str


@dataclass(frozen=True, slots=True)
class ChatModelResponse:
    output_text: str
    function_calls: list[FunctionCall]
    response_id: str
    # 실제 API로 확인함(2026-09-06): reasoning={"summary": "auto"} 요청 시 SDK가 거부하지는
    # 않지만, 설정된 모델(gpt-5.6-luna)은 summary가 항상 비어 있는 reasoning 아이템만
    # 돌려준다 — 그래서 기본값을 빈 문자열로 둔다(모델이 바뀌어 실제 요약을 채워 반환하면
    # 그대로 채워진다, 코드 변경 불필요).
    reasoning_summary: str = ""


class ChatAgentReader(Protocol):
    model: str  # 감사 추적용

    def create_response(
        self,
        input: list[dict] | str,
        tools: list[dict],
        tool_choice: str,
        previous_response_id: str | None,
    ) -> ChatModelResponse: ...
