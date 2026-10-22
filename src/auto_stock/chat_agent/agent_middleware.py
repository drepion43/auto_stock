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
import re
import threading
from collections.abc import Callable
from typing import Any, Protocol

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


class TaskPolicy(Protocol):
    """`task`(서브에이전트 위임) 호출의 사전 검사와 결과 감싸기 — stock_analyst_subagent.StockAnalystTaskPolicy."""

    def check(self, arguments: dict) -> dict | None: ...  # 거부할 결과 dict, 통과면 None

    def wrap(self, raw: str) -> dict: ...  # 서브에이전트가 돌려준 원문 → 기존 도구 결과 모양 dict


def _tool_message(result: dict, call: dict) -> ToolMessage:
    return ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=call["id"], name=call["name"])


def _task_tool_message(output: Any) -> ToolMessage | None:
    """`task`의 반환은 Command(update={"messages": [ToolMessage]})다(deepagents subagents.py:710-715)."""
    if isinstance(output, ToolMessage):
        return output
    messages = (getattr(output, "update", None) or {}).get("messages") or []
    return next((m for m in messages if isinstance(m, ToolMessage)), None)


class ToolCallGuard(AgentMiddleware):
    def __init__(
        self,
        allowed: frozenset[str],
        turn_lock: threading.Lock,
        on_event: EventCallback | None,
        task_policy: TaskPolicy | None = None,
    ) -> None:
        super().__init__()
        self._allowed = allowed
        self._turn_lock = turn_lock
        self._on_event = on_event
        self._task_policy = task_policy

    def _run_task(self, request, handler, call: dict) -> Any:
        """서브에이전트 위임: 허용 종류·위임문 형식·예산을 먼저 보고(통과 못 하면 서브에이전트를 띄우지 않음),
        결과는 기존 도구 결과 모양(출처 라벨 포함)으로 감싸고, 실패는 `available=false`로 바꾼다
        (docs/design/stock-analyst-subagent-migration.md §3 — task는 예외를 그대로 올린다)."""
        refusal = self._task_policy.check(call["args"])
        if refusal is not None:
            return _tool_message(refusal, call)
        try:
            output = handler(request)
        except Exception as exc:
            return _tool_message({"available": False, "error": f"서브에이전트 실행 실패: {type(exc).__name__}"}, call)
        message = _task_tool_message(output)
        wrapped = self._task_policy.wrap(message.text if message is not None else "")
        return _tool_message(wrapped, call)

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        name, arguments = call["name"], call["args"]
        with self._turn_lock:
            _emit(self._on_event, {"type": "tool_start", "name": name, "arguments": arguments})
            if name == "task" and self._task_policy is not None and name in self._allowed:
                message = self._run_task(request, handler, call)
            elif name in self._allowed:
                message = handler(request)
            else:
                message = _tool_message({"available": False, "error": f"알 수 없는 도구: {name}"}, call)
            result = _parse_result(getattr(message, "content", message))
            _emit(self._on_event, {"type": "tool_done", "name": name, "arguments": arguments, "result": result})
        return message


_DELEGATION_TARGET = re.compile(r"^\s*대상\s*:\s*(\S+)\s+(KOSPI|KOSDAQ|NASDAQ|NYSE)\s*$")
# deepagents가 서브에이전트에도 붙이는 내장 도구(파일시스템·위임·할일) — 스파이크(2026-10-22)에서 서브에이전트
# 모델에 ls/read_file/write_file/edit_file/delete/glob/grep이 그대로 노출되는 것을 확인했다.
_BUILTIN_TOOL_NAMES = frozenset(
    {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute", "task", "write_todos"}
)


def parse_delegation_target(text: str) -> tuple[str, str] | None:
    """위임문 첫 줄 `대상: <ticker> <원자 시장>` → (ticker 대문자, market). 형식이 아니면 None."""
    first_line = next((line for line in (text or "").splitlines() if line.strip()), "")
    match = _DELEGATION_TARGET.match(first_line)
    return (match.group(1).upper(), match.group(2)) if match else None


class SubagentToolGuard(AgentMiddleware):
    """서브에이전트용 가드(stock_analyst) — 종목 고정·노출 차단·이벤트 중계.

    - `before_agent`: 첫 메시지(메인이 쓴 위임문)에서 대상(ticker, market)을 읽는다.
    - `wrap_model_call`: deepagents 내장 도구(파일시스템 등)를 모델에서 숨긴다. 구조화 출력용 도구 같은 그 밖의
      도구는 남긴다(허용 목록 방식으로 막으면 response_format이 깨질 수 있다).
    - `wrap_tool_call`: 허용 도구가 아니거나 인자의 ticker/market이 대상과 다르면 실행하지 않는다 — 기존 클로저
      바인딩과 같은 보장. tool_start/tool_done 이벤트에 `subagent` 이름을 붙여 메인의 `on_event`로 낸다(평가 캡처)."""

    def __init__(self, subagent_name: str, allowed: frozenset[str], on_event: EventCallback | None) -> None:
        super().__init__()
        self._subagent_name = subagent_name
        self._allowed = allowed
        self._on_event = on_event
        self._target: tuple[str, str] | None = None

    def before_agent(self, state, runtime):
        first = next((m for m in state["messages"] if getattr(m, "type", None) == "human"), None)
        self._target = parse_delegation_target(first.text if first is not None else "")
        return None

    def wrap_model_call(self, request, handler):
        visible = [tool for tool in request.tools if _tool_name(tool) not in _BUILTIN_TOOL_NAMES]
        return handler(request.override(tools=visible))

    def _refusal(self, name: str, arguments: dict) -> str | None:
        if name not in self._allowed:
            return f"알 수 없는 도구: {name}"
        if self._target is None:
            return "위임 대상(대상: <ticker> <market>)이 없어 분석할 수 없습니다"
        ticker, market = self._target
        if str(arguments.get("ticker", "")).upper() != ticker or arguments.get("market") != market:
            return f"위임 대상({ticker} {market})과 다른 종목은 분석할 수 없습니다"
        return None

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        name, arguments = call["name"], call["args"]
        tag = {"name": name, "arguments": arguments, "subagent": self._subagent_name}
        _emit(self._on_event, {"type": "tool_start", **tag})
        refusal = self._refusal(name, arguments if isinstance(arguments, dict) else {})
        message = handler(request) if refusal is None else _tool_message({"available": False, "error": refusal}, call)
        _emit(self._on_event, {"type": "tool_done", **tag, "result": _parse_result(getattr(message, "content", message))})
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
