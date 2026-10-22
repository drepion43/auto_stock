"""stock_analyst를 메인 에이전트의 `subagents=`로 등록하는 정의 — docs/design/stock-analyst-subagent-migration.md.

`CHAT_AGENT_STOCK_ANALYST_MODE=subagent`일 때 `deep_loop`가 쓴다. 기존 agent-as-tool(`stock_analyst.py`의
`run_stock_analyst`, 배치 `deep_scan`도 사용)과 **같은 지시문·같은 구조화 출력 스키마·같은 출처 라벨**을 쓴다 —
다른 점은 연결 방식뿐이다:

- 신호 도구 8개는 `ticker`/`market` 인자를 받고(`tool_registry.as_subagent_signal_tools`), 종목 고정은
  `SubagentToolGuard`가 위임문 첫 줄(`대상: <ticker> <market>`)과 비교해 지킨다.
- 예산 선차감·위임문 형식 검사·결과 감싸기(출처 라벨)·실패 격리는 메인 `ToolCallGuard`가 이 모듈의
  `StockAnalystTaskPolicy`로 한다(`task`는 사전 훅이 없고 예외를 그대로 올리므로).
- 재귀 상한은 `ModelCallLimitMiddleware`로(`task`는 서브에이전트에 recursion_limit을 넘기지 않음).

`stock_analyst.py`에 두지 않는 이유: `tool_registry`를 import해야 하는데, `tools.py`가 `stock_analyst.py`를
import하므로 순환 import가 된다(그 모듈 docstring 참고)."""

import json
import threading
from typing import Any

from langchain.agents.middleware import ModelCallLimitMiddleware

from auto_stock.chat_agent.agent_middleware import EventCallback, SubagentToolGuard, parse_delegation_target
from auto_stock.chat_agent.credentials import STOCK_ANALYST_LLM_CALL_ESTIMATE, STOCK_ANALYST_RECURSION_LIMIT
from auto_stock.chat_agent.models import QueryBudget
from auto_stock.chat_agent.stock_analyst import PROVENANCE_LABEL, SYSTEM_PROMPT, StockAnalystResult
from auto_stock.chat_agent.tool_registry import SIGNAL_TOOL_NAMES, as_subagent_signal_tools
from auto_stock.chat_agent.tools import ChatToolContext

SUBAGENT_NAME = "stock_analyst"

# 메인 LLM이 보는 `task`의 서브에이전트 설명 — 위임문 형식을 여기서 강제한다(메인 가드가 형식 위반은 거부).
SUBAGENT_DESCRIPTION = (
    "종목 1개를 맡아 규칙엔진·ML예측·차트패턴·기술지표·수급·공시·뉴스감성·포지션사이징 중 필요한 신호를 스스로 골라 "
    "조사하고 종합 판단을 돌려주는 리서치 애널리스트. description은 반드시 첫 줄 '대상: <ticker> <market>'"
    "(resolve_ticker로 확정한 값, market은 KOSPI/KOSDAQ/NASDAQ/NYSE 중 하나), 둘째 줄 '요청: <사용자가 알고 싶은 것>'"
    " 형식으로 쓴다. 한 번에 한 종목. 결과의 provenance는 항상 '서브에이전트 자율 조사 결과'다 — 답변에서 이 출처를 밝혀라."
)

_TOOL_ARGS_RULE = (
    " 각 신호 도구는 ticker·market 인자를 받는다 — 위임문 첫 줄 '대상:'의 값을 그대로 넣어라. 다른 종목을 넣으면 거부된다."
)

_FORMAT_ERROR = (
    "위임문 첫 줄은 '대상: <ticker> <market>'(KOSPI/KOSDAQ/NASDAQ/NYSE) 형식이어야 합니다 — resolve_ticker로 확정한 값을 쓰세요"
)
_BUDGET_ERROR = "이번 대화 턴의 LLM 호출 예산을 모두 사용했습니다"
_JUDGMENT_KEYS = ("summary", "outlook", "signals_used")


def stock_analyst_spec(context: ChatToolContext, model: Any, on_event: EventCallback | None) -> dict:
    """메인 `create_deep_agent(subagents=[...])`에 넣을 SubAgent 정의. 턴마다 그 턴의 context에 바인딩한다
    (메인 그래프가 턴마다 새로 만들어지므로 — deep_loop._build_main_agent)."""
    tools = as_subagent_signal_tools(context, threading.Lock())
    return {
        "name": SUBAGENT_NAME,
        "description": SUBAGENT_DESCRIPTION,
        "system_prompt": SYSTEM_PROMPT + _TOOL_ARGS_RULE,
        "model": model,
        "tools": tools,
        "response_format": StockAnalystResult,
        "middleware": [
            SubagentToolGuard(SUBAGENT_NAME, SIGNAL_TOOL_NAMES, on_event),
            ModelCallLimitMiddleware(run_limit=STOCK_ANALYST_RECURSION_LIMIT, exit_behavior="end"),
        ],
    }


class StockAnalystTaskPolicy:
    """메인 `ToolCallGuard`가 `task` 호출에 적용하는 정책(agent_middleware.TaskPolicy)."""

    def __init__(self, budget: QueryBudget) -> None:
        self._budget = budget

    def check(self, arguments: dict) -> dict | None:
        if arguments.get("subagent_type") != SUBAGENT_NAME:
            return {"available": False, "error": f"사용할 수 없는 서브에이전트입니다: {arguments.get('subagent_type')}"}
        if parse_delegation_target(arguments.get("description", "")) is None:
            return {"available": False, "error": _FORMAT_ERROR}
        # 기존 run_stock_analyst와 같은 선차감 — 서브에이전트 내부 호출은 메인 호출 상한에 합산되지 않는다.
        if not self._budget.try_consume_llm_calls(STOCK_ANALYST_LLM_CALL_ESTIMATE):
            return {"available": False, "error": _BUDGET_ERROR}
        return None

    def wrap(self, raw: str) -> dict:
        """서브에이전트 결과(구조화 출력 JSON) → 기존 agent-as-tool 결과와 같은 모양 + 코드가 붙이는 출처 라벨."""
        try:
            data = json.loads(raw)
            judgment = {key: data[key] for key in _JUDGMENT_KEYS}
        except (TypeError, ValueError, KeyError):
            return {"available": False, "error": "종목 분석 결과를 해석하지 못했습니다(구조화 출력 없음)"}
        return {"available": True, "judgment": judgment, "provenance": PROVENANCE_LABEL}
