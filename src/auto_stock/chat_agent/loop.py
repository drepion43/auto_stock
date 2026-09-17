"""메인 대화 루프 — OpenAI Responses API의 네이티브 함수콜링을 수동으로 오케스트레이션
한다(LangChain/deepagents Tool Runner 없이 직접 구현, Stage B+C 승인 설계 "Tool
Runner(또는 수동 tool-use 루프)"). find_related_companies/stock_analyst 2개만 내부에서
deepagents를 쓰고, 이 루프 자체는 그것들도 다른 7개와 동일한 평평한 함수콜링 도구로만
다룬다 — 메인 루프 입장에서 agent-as-tool과 plain tool은 구분되지 않는다.

**멀티턴은 client.py가 감싼 Responses API의 stateful 특성(`previous_response_id`)으로
별도 구현 없이 해결된다** — 서버가 대화 히스토리를 들고 있으므로, 이 루프는 브랜드
뉴 대화(`previous_response_id is None`)일 때만 시스템 프롬프트를 `input`에 끼워 넣고,
그 외에는 사용자 메시지만 보낸다. 사용자의 "이 뉴스 어떻게 생각해?" 요구사항이 이
메커니즘으로 충족된다(SYSTEM_PROMPT의 "멀티턴" 지침이 모델에게 새 도구 호출 대신
히스토리 재사용을 지시).

도구 실행 자체는 `tools.py`의 각 `tool_*` 함수가 이미 raise하지 않는다는 계약을
따르지만, `_dispatch_tool_call`은 그 경계를 한 번 더 감싼다 — 모델이 만든 JSON 인자가
깨져 있거나(malformed arguments) 스키마에 없는 도구 이름을 부르는 경우까지 방어해야
하는, 이 프로젝트에서 유일하게 신뢰할 수 없는 입력(LLM 생성 JSON)을 다루는 경계이기
때문이다(프로젝트 공통 원칙: 시스템 경계에서만 검증)."""

import json
from collections.abc import Callable

from auto_stock.chat_agent.credentials import MAX_TOOL_ITERATIONS
from auto_stock.chat_agent.models import ChatAgentReader, FunctionCall
from auto_stock.chat_agent.prompt import SYSTEM_PROMPT
from auto_stock.chat_agent.tool_schemas import TOOL_SCHEMAS
from auto_stock.chat_agent.tools import TOOL_DISPATCH, ChatToolContext

_ITERATION_LIMIT_MESSAGE = (
    "도구 호출 한도에 도달해 답변을 완성하지 못했습니다. 질문을 더 구체적으로 나눠서 "
    "다시 시도해 주세요."
)
_API_FAILURE_MESSAGE = "일시적인 오류로 응답을 완성하지 못했습니다. 잠시 후 다시 시도해 주세요."


def _dispatch_tool_call(context: ChatToolContext, call: FunctionCall) -> tuple[dict, dict]:
    """API에 돌려줄 `function_call_output` dict와, CLI가 사용자에게 실시간으로 보여줄
    표시용 dict(`{"name", "arguments", "result"}`)를 함께 반환한다 — 후자는 `run_turn`의
    `on_step` 콜백으로 그대로 전달된다(사용자 요구사항: tool 호출 추적, 2026-09-06).
    `arguments`가 파싱 실패하면 표시용 dict에는 원본 문자열을 그대로 담는다."""
    try:
        arguments = json.loads(call.arguments)
    except json.JSONDecodeError as exc:
        arguments = call.arguments
        result = {"available": False, "error": f"도구 인자 파싱 실패: {exc}"}
    else:
        fn = TOOL_DISPATCH.get(call.name)
        if fn is None:
            result = {"available": False, "error": f"알 수 없는 도구: {call.name}"}
        else:
            try:
                result = fn(context, **arguments)
            except TypeError as exc:
                result = {"available": False, "error": f"도구 인자 불일치: {exc}"}

    output_item = {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": json.dumps(result, ensure_ascii=False),
    }
    return output_item, {"name": call.name, "arguments": arguments, "result": result}


def run_turn(
    reader: ChatAgentReader,
    context: ChatToolContext,
    user_message: str,
    previous_response_id: str | None,
    on_event: Callable[[dict], None] | None = None,
) -> tuple[str, str]:
    """한 턴을 실행하고 (최종 답변 텍스트, 이번 마지막 응답의 response_id)를 반환한다.
    `context.budget`은 호출부(`chat_cli.py`)가 매 턴 새 `QueryBudget`으로 갱신해
    넘긴다는 전제다 — 이 함수는 예산을 새로 만들지 않는다.

    `on_event`(선택)는 아래 4종 이벤트를 **작업이 끝난 뒤 요약이 아니라 시작되는
    시점에** 전달받는 콜백이다 — 사용자 요구사항(2026-09-06): "지금 tool 호출
    중입니다"/"reasoning 중입니다" 같은 활동 상태를 대기 중에 실시간으로 보고 싶다는
    요청. 이 모듈은 print를 하지 않는다(관심사 분리 — 실제 출력은 `chat_cli.py` 몫).

    - `{"type": "thinking"}` — `reader.create_response`를 부르기 **직전**(그 블로킹
      호출이 몇 초 걸릴 수 있으므로 호출 전에 알려야 의미가 있다). 라운드마다 1회.
    - `{"type": "reasoning", "text": str}` — 응답에 `reasoning_summary`가 있을 때만.
    - `{"type": "tool_start", "name": str, "arguments": dict | str}` — 개별 tool을
      실행(블로킹)하기 **직전**. 인자 파싱 실패 시 원본 문자열을 그대로 담는다.
    - `{"type": "tool_done", "name": str, "arguments": dict | str, "result": dict}` —
      그 tool 실행 **직후**."""

    def emit(event: dict) -> None:
        if on_event is not None:
            on_event(event)

    input_items: list[dict] | str
    if previous_response_id is None:
        input_items = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    else:
        input_items = user_message

    response_id = previous_response_id
    for _ in range(MAX_TOOL_ITERATIONS):
        emit({"type": "thinking"})
        try:
            response = reader.create_response(
                input=input_items,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                previous_response_id=response_id,
            )
        except Exception:
            # reader.create_response는 이 모듈에서 유일하게 실제 네트워크/SDK 호출이
            # 일어나는 지점이다(client.py가 모든 openai SDK 예외를 ChatAgentError로
            # 통일) — 여기서 실패해도 이 턴 전체를 "없었던 일"로 되돌린다(previous_
            # response_id를 그대로 반환), 다음 턴이 정상 상태에서 이어질 수 있도록
            # (코드 리뷰에서 발견: 이전에는 이 호출이 완전히 무방비였다).
            return _API_FAILURE_MESSAGE, previous_response_id

        if response.reasoning_summary:
            emit({"type": "reasoning", "text": response.reasoning_summary})

        if not response.function_calls:
            return response.output_text, response.response_id

        response_id = response.response_id
        dispatched = []
        for call in response.function_calls:
            try:
                preview_arguments = json.loads(call.arguments)
            except json.JSONDecodeError:
                preview_arguments = call.arguments
            emit({"type": "tool_start", "name": call.name, "arguments": preview_arguments})
            output_item, display = _dispatch_tool_call(context, call)
            emit(
                {
                    "type": "tool_done",
                    "name": display["name"],
                    "arguments": display["arguments"],
                    "result": display["result"],
                }
            )
            dispatched.append(output_item)
        input_items = dispatched

    # MAX_TOOL_ITERATIONS를 다 썼다는 것은 마지막 라운드의 function_call들이 dispatch만
    # 되고 API에 결과를 제출하지 못했다는 뜻이다(한 번 더 create_response를 부르지
    # 않고 루프가 끝남) — 그 미해결 상태의 response_id를 다음 턴 previous_response_id로
    # 넘기면 안 되므로, 이번 턴 시작 시점의 response_id로 되돌린다(코드 리뷰에서 발견).
    return _ITERATION_LIMIT_MESSAGE, previous_response_id
