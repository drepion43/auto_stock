"""1단계 메인 deepagent(`deep_loop.py`)용 미들웨어 — docs/design/main-agent-migration-plan.md §4.1.

구 루프(`loop.py`)가 손으로 보장하던 동작 계약을 LangChain 미들웨어로 옮긴다:
- `ExposureGuard`: 모델에는 레지스트리 도구만 보인다(deepagents 내장 ls/read_file/write_file/
  task 등은 메인에서 숨김). 레지스트리 도구가 하나라도 안 보이면 fail-closed로 예외(스파이크 S10).
- `ToolCallGuard`: 도구 실행을 턴 Lock으로 직렬화(C11, LangGraph는 병렬 실행 — 스파이크 S3)하고,
  그 Lock 안에서 tool_start/tool_done 이벤트를 낸다(C12, 구 루프와 같은 순서). 미등록 이름은
  구 루프 문구로 거부한다(C8 — 내장 write_file 등을 모델이 직접 불러도 실행되지 않음).
- `InvalidToolCallFeedback`: 인자 JSON이 깨진 호출을 구 루프 문구로 모델에 돌려줘 재시도 기회를
  준다(C7 ② — 이 처리가 없으면 턴이 빈 텍스트로 끝난다, 스파이크 S11).
- `SessionCallCounter`: 세션 전체 메인 모델 호출 상한(롤백과 무관하게 누적, 결정 D5).
- `TurnEvents`: 모델 호출 직전 thinking, 응답에 reasoning 요약이 있으면 reasoning(C12/C13).
- `StrictToolSchemas`: Responses API 함수 스키마에 strict=true(구 루프 스키마와 동일, 스파이크 S1).

`excluded_tools`류는 "모델 노출 차단일 뿐 보안 경계가 아니다"(deepagents 스스로 명시) — 그래서
노출 필터(`wrap_model_call`)와 실행 거부(`wrap_tool_call`)를 둘 다 둔다."""

import json
import threading
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, ToolMessage

EventCallback = Callable[[dict], None]

_ARGUMENT_ERROR_PREFIX = "도구 인자 파싱 실패"  # 구 루프와 같은 문구(계약 C7)


class ExposureError(RuntimeError):
    """레지스트리 도구가 모델에 그대로 노출되지 않는 상태 — deep 모드를 쓰면 안 된다."""


class SessionLimitExceededError(RuntimeError):
    """세션 전체 메인 모델 호출 상한 도달."""


def _tool_name(tool: Any) -> str:
    return tool["name"] if isinstance(tool, dict) else tool.name


def _emit(on_event: EventCallback | None, event: dict) -> None:
    if on_event is not None:
        on_event(event)


def _parse_result(content: Any) -> Any:
    try:
        return json.loads(content)
    except (TypeError, ValueError):
        return content


class StrictToolSchemas(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        return handler(request.override(model_settings={**request.model_settings, "strict": True}))


class ExposureGuard(AgentMiddleware):
    def __init__(self, allowed: frozenset[str]) -> None:
        super().__init__()
        self._allowed = allowed

    def wrap_model_call(self, request, handler):
        visible = [tool for tool in request.tools if _tool_name(tool) in self._allowed]
        missing = self._allowed - {_tool_name(tool) for tool in visible}
        if missing:
            raise ExposureError(f"registry tools not visible to the model: {sorted(missing)}")
        return handler(request.override(tools=visible))


class ToolCallGuard(AgentMiddleware):
    def __init__(self, allowed: frozenset[str], turn_lock: threading.Lock, on_event: EventCallback | None) -> None:
        super().__init__()
        self._allowed = allowed
        self._turn_lock = turn_lock
        self._on_event = on_event

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        name, arguments = call["name"], call["args"]
        with self._turn_lock:
            _emit(self._on_event, {"type": "tool_start", "name": name, "arguments": arguments})
            if name in self._allowed:
                message = handler(request)
            else:
                message = ToolMessage(
                    content=json.dumps({"available": False, "error": f"알 수 없는 도구: {name}"}, ensure_ascii=False),
                    tool_call_id=call["id"],
                    name=name,
                )
            result = _parse_result(getattr(message, "content", message))
            _emit(self._on_event, {"type": "tool_done", "name": name, "arguments": arguments, "result": result})
        return message


class InvalidToolCallFeedback(AgentMiddleware):
    def __init__(self, on_event: EventCallback | None) -> None:
        super().__init__()
        self._on_event = on_event

    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime):
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.invalid_tool_calls:
            return None
        feedback = []
        for call in last.invalid_tool_calls:
            name, raw_args = call.get("name") or "", call.get("args") or ""
            result = {"available": False, "error": f"{_ARGUMENT_ERROR_PREFIX}: {call.get('error') or raw_args}"}
            _emit(self._on_event, {"type": "tool_start", "name": name, "arguments": raw_args})
            _emit(self._on_event, {"type": "tool_done", "name": name, "arguments": raw_args, "result": result})
            feedback.append(
                ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=call["id"], name=name)
            )
        update: dict = {"messages": feedback}
        if not last.tool_calls:
            update["jump_to"] = "model"  # 유효 호출이 없으면 라우터가 턴을 끝내므로 모델로 되돌린다
        return update


class SessionCallCounter(AgentMiddleware):
    def __init__(self, session: Any) -> None:
        super().__init__()
        self._session = session

    def before_model(self, state, runtime):
        if self._session.model_calls >= self._session.session_call_limit:
            raise SessionLimitExceededError(f"session model-call limit {self._session.session_call_limit} reached")

    def wrap_model_call(self, request, handler):
        self._session.model_calls += 1
        return handler(request)


class TurnEvents(AgentMiddleware):
    def __init__(self, on_event: EventCallback | None) -> None:
        super().__init__()
        self._on_event = on_event

    def before_model(self, state, runtime):
        _emit(self._on_event, {"type": "thinking"})

    def after_model(self, state, runtime):
        last = state["messages"][-1]
        blocks = last.content if isinstance(last, AIMessage) and isinstance(last.content, list) else []
        summaries = [
            part.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "reasoning"
            for part in block.get("summary") or []
            if isinstance(part, dict)
        ]
        text = "\n".join(s for s in summaries if s)
        if text:
            _emit(self._on_event, {"type": "reasoning", "text": text})
