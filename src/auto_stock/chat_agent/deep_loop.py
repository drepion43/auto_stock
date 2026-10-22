"""메인 대화 루프 — `create_deep_agent` 기반. 2026-10-12부터 기본값(`CHAT_AGENT_LOOP` 미설정 시).

docs/design/main-agent-migration-plan.md §4.1/§4.1b. 구 루프(`loop.py`)는 `CHAT_AGENT_LOOP=legacy`
롤백용으로 2주 실사용 기간 동안만 남는다.
이 루프는 같은 13개 도구(`tool_registry`)·같은 시스템 프롬프트·같은 폴백 문구를 쓰고, 구 루프가
손으로 지키던 계약은 `agent_middleware.py`의 미들웨어가 지킨다.

**멀티턴(결정 D4 ②)**: 체크포인터를 쓰지 않는다. `DeepChatSession`이 지난 턴들의 메시지 목록을
들고 있고, 전송은 `ChatOpenAI(use_previous_response_id=True)`라 서버가 이미 가진 앞부분은 다시
보내지 않는다(스파이크 S8). 턴이 실패하면 세션 목록에 반영하지 않는 것만으로 롤백된다(S4/S5).

**비용 상한 3겹**: 턴당 메인 호출 `MAIN_AGENT_RUN_LIMIT`(ModelCallLimitMiddleware), 세션 전체
`session_call_limit`(롤백돼도 누적), 서브에이전트·LLM 도구는 기존 턴당 `QueryBudget`(context).
서브에이전트 내부 호출은 ModelCallLimit에 합산되지 않으므로 QueryBudget 선차감을 계속 쓴다(§5).

**트레이싱 차단**: LangSmith 트레이싱이 켜져 있으면 대화 전체와 서브에이전트 기록이 외부로
전송된다 — 명시적 허용(`CHAT_AGENT_ALLOW_TRACING=1`) 없이는 deep 모드를 시작하지 않는다."""

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from deepagents import FilesystemMiddleware, create_deep_agent
from deepagents.backends import StateBackend
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import Field

from auto_stock.chat_agent.agent_middleware import (
    ExposureError,
    ExposureGuard,
    InvalidToolCallFeedback,
    SessionCallCounter,
    SessionLimitExceededError,
    StrictToolSchemas,
    ToolCallGuard,
    TurnEvents,
)
from auto_stock.chat_agent.credentials import (
    MAIN_AGENT_COMPACT_KEEP_TURNS,
    MAIN_AGENT_INPUT_TOKEN_LIMIT,
    MAIN_AGENT_RUN_LIMIT,
    MAIN_AGENT_SESSION_CALL_LIMIT,
    load_stock_analyst_mode,
)
from auto_stock.chat_agent.fallbacks import (
    API_ERROR_FALLBACK,
    SESSION_LIMIT_FALLBACK,
    TOOL_LIMIT_FALLBACK,
)
from auto_stock.chat_agent.models import LLMConfig, QueryBudget
from auto_stock.chat_agent.prompt import SUBAGENT_MODE_ADDENDUM, SYSTEM_PROMPT
from auto_stock.chat_agent.stock_analyst_subagent import StockAnalystTaskPolicy, stock_analyst_spec
from auto_stock.chat_agent.tool_registry import as_langchain_tools, main_tool_specs
from auto_stock.chat_agent.tools import ChatToolContext

__all__ = [
    "DeepChatSession",
    "ExposureError",
    "TracingNotAllowedError",
    "check_tracing_allowed",
    "make_chat_model",
    "run_turn_deep",
    "verify_exposure",
]

logger = logging.getLogger(__name__)

# langsmith.utils.tracing_is_enabled가 보는 키 전부(LANGSMITH_/LANGCHAIN_ × TRACING_V2/TRACING).
_TRACING_ENV_KEYS = ("LANGSMITH_TRACING_V2", "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")
_TRUTHY = {"1", "true", "yes", "on"}

EventCallback = Callable[[dict], None]


class TracingNotAllowedError(RuntimeError):
    """트레이싱 환경변수가 켜져 있는데 명시적 허용이 없다."""


@dataclass(slots=True)
class DeepChatSession:
    """대화 세션 1개의 가변 상태 핸들(턴마다 갱신되므로 의도적으로 frozen이 아니다)."""

    model: BaseChatModel
    messages: list[BaseMessage] = field(default_factory=list)
    model_calls: int = 0  # 체크포인트 밖 세션 카운터 — 실패·롤백된 턴의 호출도 센다
    session_call_limit: int = MAIN_AGENT_SESSION_CALL_LIMIT
    input_token_limit: int = MAIN_AGENT_INPUT_TOKEN_LIMIT
    turn_lock: threading.Lock = field(default_factory=threading.Lock)
    # subagent 모드에서 stock_analyst 서브에이전트가 쓸 모델 — None이면 기존과 같은 "openai:<agent_model>".
    # 테스트·노출 점검이 가짜 모델을 넣는 자리다.
    subagent_model: Any = None


def make_chat_model(llm_config: LLMConfig) -> ChatOpenAI:
    """구 클라이언트(client.py)와 같은 키·모델·토큰·타임아웃·재시도 값을 명시적으로 넘긴다(C16)."""
    return ChatOpenAI(
        model=llm_config.model,
        api_key=llm_config.api_key,
        use_responses_api=True,
        use_previous_response_id=True,
        reasoning={"summary": "auto"},
        max_tokens=llm_config.max_tokens,
        timeout=llm_config.timeout_seconds,
        max_retries=llm_config.max_retries,
    )


def check_tracing_allowed() -> None:
    tracing_on = any(os.environ.get(key, "").strip().lower() in _TRUTHY for key in _TRACING_ENV_KEYS)
    if tracing_on and os.environ.get("CHAT_AGENT_ALLOW_TRACING", "").strip() != "1":
        raise TracingNotAllowedError(
            "LangSmith tracing is enabled; set CHAT_AGENT_ALLOW_TRACING=1 to send conversations externally"
        )


def _allowed_tool_names(mode: str) -> frozenset[str]:
    """모델에 보여야 하는 도구 정확히 — subagent 모드는 신호 도구·stock_analyst 대신 `task`
    (docs/design/stock-analyst-subagent-migration.md, 결정 D1 안 A)."""
    names = frozenset(spec.name for spec in main_tool_specs(mode))
    return names | {"task"} if mode == "subagent" else names


def _expected_tool_names() -> frozenset[str]:
    """`verify_exposure`의 기대값 — 현재 모드 기준."""
    return _allowed_tool_names(load_stock_analyst_mode())


def _subagent_kwargs(mode: str, session: DeepChatSession, context: ChatToolContext, on_event: EventCallback | None) -> dict:
    if mode != "subagent":
        return {}
    model = session.subagent_model or f"openai:{context.agent_model}"
    return {"subagents": [stock_analyst_spec(context, model, on_event)]}


def _build_main_agent(session: DeepChatSession, context: ChatToolContext, on_event: EventCallback | None):
    mode = load_stock_analyst_mode()
    allowed = _allowed_tool_names(mode)
    task_policy = StockAnalystTaskPolicy(context.budget) if mode == "subagent" else None
    backend = StateBackend()  # 실제 파일시스템·셸 백엔드 금지(내장 도구는 어차피 노출·실행 차단)
    return create_deep_agent(
        model=session.model,
        # 실행 직렬화는 ToolCallGuard의 턴 Lock이 맡는다 — 레지스트리 래퍼에는 별도 Lock을 줘서
        # 같은 Lock을 두 번 잡는(교착) 일이 없게 한다.
        tools=as_langchain_tools(context, threading.Lock(), main_tool_specs(mode)),
        system_prompt=SYSTEM_PROMPT + (SUBAGENT_MODE_ADDENDUM if mode == "subagent" else ""),
        backend=backend,
        **_subagent_kwargs(mode, session, context, on_event),
        middleware=[
            # after_model 훅은 목록의 역순으로 실행된다. 깨진 호출 피드백은 jump_to="model"로 다른
            # after_model을 건너뛰므로 반드시 맨 앞(= 마지막 실행)에 둔다 — 그렇지 않으면
            # ModelCallLimit의 호출 수 증가(after_model)가 빠져 턴 상한이 무력화된다(코드 리뷰 HIGH).
            InvalidToolCallFeedback(on_event),
            # before_model 순서가 중요하다: 한도 검사가 thinking 이벤트보다 먼저여야 거부된 호출에
            # "생각 중" 표시가 남지 않는다(구 루프와 같은 thinking 횟수).
            ModelCallLimitMiddleware(run_limit=MAIN_AGENT_RUN_LIMIT, exit_behavior="error"),
            SessionCallCounter(session),
            TurnEvents(on_event),
            ExposureGuard(allowed),
            StrictToolSchemas(),
            ToolCallGuard(allowed, session.turn_lock, on_event, task_policy=task_policy),
            # 대용량 결과를 가상 파일로 빼는 기본 동작 끄기 — read_file을 숨겼으니 회수할 수 없다.
            FilesystemMiddleware(
                backend=backend, tool_token_limit_before_evict=None, human_message_token_limit_before_evict=None
            ),
        ],
    )


def _final_text(messages: list[BaseMessage]) -> str:
    last = messages[-1] if messages else None
    return last.text if isinstance(last, AIMessage) else ""


def _compact_if_needed(messages: list[BaseMessage], limit: int) -> list[BaseMessage]:
    """마지막 응답의 실제 입력 토큰이 상한을 넘으면 최근 (질문, 최종답변) 쌍만 평문으로 남긴다.
    응답 ID가 없는 새 메시지라 다음 턴은 서버 체인을 새로 시작한다(도구 기록·체인 누적 제거)."""
    last = messages[-1] if messages else None
    usage = getattr(last, "usage_metadata", None) or {}
    if usage.get("input_tokens", 0) <= limit:
        return messages
    pairs: list[tuple[str, str]] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            pairs.append((message.text, ""))
        elif isinstance(message, AIMessage) and message.text and pairs:
            pairs[-1] = (pairs[-1][0], message.text)
    kept = pairs[-MAIN_AGENT_COMPACT_KEEP_TURNS:]
    return [m for human, ai in kept for m in (HumanMessage(content=human), AIMessage(content=ai))]


def run_turn_deep(
    session: DeepChatSession,
    context: ChatToolContext,
    user_message: str,
    on_event: EventCallback | None = None,
) -> str:
    """한 턴을 실행하고 최종 답변 텍스트를 반환한다. 절대 raise하지 않는다 — 실패하면 고정 폴백
    문구를 반환하고 `session.messages`는 그대로 둔다(턴 롤백). 이벤트 4종의 의미·순서는 구 루프
    `run_turn` docstring과 같다(thinking/reasoning/tool_start/tool_done)."""
    turn_input = [*session.messages, HumanMessage(content=user_message)]
    try:
        agent = _build_main_agent(session, context, on_event)
        result = agent.invoke({"messages": turn_input})
    except ModelCallLimitExceededError:
        return TOOL_LIMIT_FALLBACK
    except SessionLimitExceededError:
        return SESSION_LIMIT_FALLBACK
    except Exception as exc:  # 모델/네트워크/노출 검사 실패 — 키·본문 없이 타입만 남긴다(C16)
        logger.warning("main agent turn failed: %s", type(exc).__name__)
        return API_ERROR_FALLBACK

    messages = list(result["messages"])
    text = _final_text(messages)
    if not text.strip():
        return API_ERROR_FALLBACK  # 유효한 답 없이 끝난 턴(C7 ②의 최후 방어선)도 롤백
    session.messages = _compact_if_needed(messages, session.input_token_limit)
    return text


class _ProbeModel(BaseChatModel):
    """`verify_exposure` 전용 — 네트워크 호출 없이 바인딩된 도구 이름만 기록하고 즉시 답한다."""

    bound_names: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "exposure-probe"

    def bind_tools(self, tools, **kwargs):
        self.bound_names = [t["name"] if isinstance(t, dict) else t.name for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


def verify_exposure() -> None:
    """fail-closed 사전 점검 — 실제 그래프를 조립해 모델에 바인딩되는 도구가 정확히 레지스트리
    도구인지 확인한다(내장 도구 노출·레지스트리 도구 누락 모두 거부). 실패 시 `ExposureError`.
    deep 모드 시작 시 1회 호출한다."""
    probe = _ProbeModel()
    context = ChatToolContext(
        cache=None, ml_model=None, llm_client=None, news_client=None, sentiment_client=None,
        budget=QueryBudget(max_llm_calls=0), account=None, agent_model="", scan_cache=None,
        recommendation_coordinator=None,
    )
    agent = _build_main_agent(DeepChatSession(model=probe, subagent_model=_ProbeModel()), context, on_event=None)
    agent.invoke({"messages": [HumanMessage(content="exposure probe")]})
    if set(probe.bound_names) != _expected_tool_names():
        raise ExposureError(f"model-visible tools {sorted(probe.bound_names)} != registry")
