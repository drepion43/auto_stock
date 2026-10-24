"""메인 에이전트의 `subagents=` 전체 조립 — docs/design/stock-analyst-subagent-migration.md §12.

subagent 모드(`CHAT_AGENT_STOCK_ANALYST_MODE=subagent`, 2026-10-24부터 기본)에서 메인은 종목 분석·관련기업 탐색·ETF
분석을 직접 도구로 부르지 않고 `task`로 위임한다. 이 모듈은 세 서브에이전트 정의와, 메인 `ToolCallGuard`에 넘길
`SubagentTaskRouter`를 만든다. 각 서브에이전트는 기존 agent-as-tool(`run_find_related_companies`,
`run_etf_composition_synthesis`)과 **같은 지시문·같은 구조화 출력 스키마·같은 결과 모양·같은 출처 라벨**을 쓴다.

- `find_related_companies`: 위임 정책이 공시 기준 회사명(DART/EDGAR)을 결정론적으로 해석해 위임문에 `회사명:` 줄로
  덧붙인다(기존 도구도 이름을 먼저 해석한 뒤 서브에이전트를 띄웠다). 뉴스 공동언급 확인 도구는 후보 회사명을 인자로
  받으므로 종목 고정(`lock_target`)은 끈다.
- `analyze_etf_composition`: 기존 도구는 "결정론적 수집 → 종합 전용 서브에이전트"였다. 여기서는 수집 단계를
  서브에이전트의 도구(`collect_etf_holding_signals`, 위임 대상 ETF에 고정)로 주고, 결과의 holdings·data_as_of_note는
  LLM 출력이 아니라 그 도구가 실제로 돌려준 값을 `EtfCollectionRecorder`로 받아 붙인다(환각 방지).

호출 실행은 메인 `ToolCallGuard`의 턴 Lock 안에서 직렬화되므로(위임 1건씩) 턴 단위 recorder 하나로 충분하다.
`recommendation_synthesis`와 섹터 2차 탐색(`sector_resolution.run_sector_theme_search`)은 메인 대화가 아니라
백그라운드 배치(`orchestrator/recommendation_coordinator.py`)가 부르므로 메인 `subagents=` 대상이 아니다."""

import json
import threading
from typing import Any

from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from auto_stock.chat_agent import tools as _tools
from auto_stock.chat_agent.agent_middleware import (
    EventCallback,
    SubagentTaskRouter,
    SubagentToolGuard,
    TaskAdmission,
    parse_delegation_target,
    refuse,
)
from auto_stock.chat_agent.credentials import (
    ETF_COMPOSITION_LLM_CALL_ESTIMATE,
    ETF_COMPOSITION_RECURSION_LIMIT,
    FIND_RELATED_LLM_CALL_ESTIMATE,
    FIND_RELATED_RECURSION_LIMIT,
)
from auto_stock.chat_agent.etf_composition import PROVENANCE_LABEL as ETF_PROVENANCE_LABEL
from auto_stock.chat_agent.etf_composition import SYSTEM_PROMPT as ETF_SYSTEM_PROMPT
from auto_stock.chat_agent.etf_composition import EtfCompositionResult
from auto_stock.chat_agent.models import QueryBudget
from auto_stock.chat_agent.related_companies import SYSTEM_PROMPT as RELATED_SYSTEM_PROMPT
from auto_stock.chat_agent.related_companies import (
    RelatedCompaniesResult,
    related_payload,
    verify_companies_co_mentioned_in_news,
)
from auto_stock.chat_agent.stock_analyst_subagent import (
    BUDGET_ERROR,
    DELEGATION_FORMAT_ERROR,
    StockAnalystTaskPolicy,
    stock_analyst_spec,
)
from auto_stock.chat_agent.stock_analyst_subagent import SUBAGENT_NAME as STOCK_ANALYST
from auto_stock.chat_agent.tool_args import TickerMarketArgs
from auto_stock.chat_agent.tools import COMPANY_NAME_UNRESOLVED_ERROR, ChatToolContext

RELATED_COMPANIES = "find_related_companies"
ETF_COMPOSITION = "analyze_etf_composition"
SUBAGENT_NAMES = (STOCK_ANALYST, RELATED_COMPANIES, ETF_COMPOSITION)
_CO_MENTION_TOOL = verify_companies_co_mentioned_in_news.name
_ETF_COLLECT_TOOL = "collect_etf_holding_signals"

_DELEGATION_FORMAT = (
    "description은 반드시 첫 줄 '대상: <ticker> <market>'(resolve_ticker로 확정한 값, market은 KOSPI/KOSDAQ/NASDAQ/NYSE "
    "중 하나), 둘째 줄 '요청: <사용자가 알고 싶은 것>' 형식으로 쓴다. 한 번에 한 종목."
)

RELATED_DESCRIPTION = (
    "종목 1개의 관련기업(경쟁사/공급망/계열사)을 자기 자신 제외 최대 4개 찾고, 뉴스 공동언급으로 'confirmed'(확인됨) "
    "또는 'inferred'(추정)를 매기는 리서치 에이전트. 사용자가 '관련기업'/'경쟁사'/'계열사' 등 종목 자신을 넘어선 범위를 "
    f"궁금해할 때만 쓴다. {_DELEGATION_FORMAT} 답변에서 confirmed/inferred를 반드시 구분해 표기하라."
)
_RELATED_NAME_RULE = (
    " 위임문의 '회사명:' 줄이 조사 대상 회사의 공식 이름(공시 기준)이다 — 그 이름으로 조사하고 뉴스 공동언급을 확인하라."
    # 2026-10-24 실측: 자유 형식 '요청:'을 받자 relation에 사업 설명 문단을 써 메인 문맥이 커졌다(기존 고정 지시문에선 짧았음).
    " relation에는 '경쟁사'·'공급망(고객사)'·'계열사'처럼 짧은 관계 유형만 쓰고 설명 문장은 넣지 마라 — 요청 줄이 상세 설명을"
    " 원해도 마찬가지다(종목별 분석은 메인이 따로 위임한다)."
)

ETF_DESCRIPTION = (
    "ETF 1개를 맡아 비중 상위 구성종목(최대 10개)의 공시·뉴스감성 신호를 수집해 'ETF가 왜 움직였는지'를 종합하는 "
    f"애널리스트. 국내(KOSPI/KOSDAQ)·미국(NASDAQ/NYSE) ETF 모두 지원. {_DELEGATION_FORMAT} 결과의 provenance는 "
    "'ETF 구성종목 집계 분석'이며, 반환된 holdings(분석한 구성종목·비중)만 분석했다는 점과 data_as_of_note가 있으면 그 "
    "내용(미국 ETF는 분기 공시 기준이라 데이터가 묵었을 수 있음)을 답변에서 반드시 밝혀라. holdings가 null이면 ETF가 "
    "아니거나 구성종목 데이터가 없다는 뜻이다."
)
_ETF_COLLECT_RULE = (
    f" 먼저 {_ETF_COLLECT_TOOL}를 위임문 첫 줄 '대상:'의 ticker·market 그대로 정확히 한 번 호출해 입력(holding_signals)을"
    " 받아라. 그 결과가 available=false이거나 holdings가 null이면 summary에 그 사실만 쓰고 outlook은 NEUTRAL로 하라."
)
_ETF_COLLECT_DESCRIPTION = (
    "위임 대상 ETF의 비중 상위 구성종목과 각 종목의 공시·뉴스감성 해석(holding_signals)을 수집한다. 종합은 하지 않는다."
)

_UNPARSED = "결과를 해석하지 못했습니다(구조화 출력 없음)"


_NAME_LINE_PREFIX = "회사명:"


def _with_resolved_name(description: str, name: str) -> str:
    """대상 줄 바로 뒤에 공시 기준 회사명을 넣는다. 메인(또는 사용자 입력을 옮긴 메인)이 쓴 `회사명:` 줄은 지운다 —
    서브에이전트가 보는 회사명 줄은 항상 코드가 해석한 하나뿐이어야 한다."""
    lines = description.splitlines()
    first = next(i for i, text in enumerate(lines) if text.strip())
    rest = [text for text in lines[first + 1 :] if not text.strip().startswith(_NAME_LINE_PREFIX)]
    return "\n".join([*lines[: first + 1], f"{_NAME_LINE_PREFIX} {name}", *rest])


class RelatedCompaniesTaskPolicy:
    """`task(subagent_type="find_related_companies")` — 회사명 해석 실패는 기존 도구와 같은 문구로 위임 전에 거부."""

    def __init__(self, budget: QueryBudget) -> None:
        self._budget = budget

    def admit(self, arguments: dict) -> TaskAdmission:
        target = parse_delegation_target(arguments.get("description", ""))
        if target is None:
            return refuse(DELEGATION_FORMAT_ERROR)
        try:
            name = _tools.resolve_company_name(*target)
        except Exception as exc:
            return refuse(str(exc))
        if name is None:
            return refuse(COMPANY_NAME_UNRESOLVED_ERROR)
        # 이름 해석이 끝난 뒤에 선차감 — 기존 도구도 이름 해석 실패 시 예산을 쓰지 않았다.
        if not self._budget.try_consume_llm_calls(FIND_RELATED_LLM_CALL_ESTIMATE):
            return refuse(BUDGET_ERROR)
        return TaskAdmission(arguments={**arguments, "description": _with_resolved_name(arguments["description"], name)})

    def wrap(self, arguments: dict, raw: str) -> dict:
        try:
            structured = RelatedCompaniesResult.model_validate(json.loads(raw))
        except (TypeError, ValueError, ValidationError):
            return {"available": False, "error": f"관련기업 조사 {_UNPARSED}"}
        return {"available": True, "related": related_payload(structured)}


class EtfCollectionRecorder:
    """ETF 서브에이전트의 수집 도구가 실제로 돌려준 값 — 결과의 holdings/data_as_of_note는 여기서만 가져온다."""

    def __init__(self) -> None:
        self.last: dict | None = None

    def reset(self) -> None:
        self.last = None

    def record(self, result: dict) -> None:
        self.last = result


class EtfCompositionTaskPolicy:
    """`task(subagent_type="analyze_etf_composition")` — 예산은 기존 도구와 같은 방식이다: 구성종목 1개도 감당할 수 없으면
    위임 전에 거부하고, 종합 몫은 수집 도구가 실제 holdings를 받았을 때만 차감한다(ETF가 아니면 종합 몫을 쓰지 않음)."""

    def __init__(self, budget: QueryBudget, recorder: EtfCollectionRecorder) -> None:
        self._budget = budget
        self._recorder = recorder

    def admit(self, arguments: dict) -> TaskAdmission:
        if parse_delegation_target(arguments.get("description", "")) is None:
            return refuse(DELEGATION_FORMAT_ERROR)
        if _tools.affordable_etf_holdings(self._budget) == 0:
            return refuse(BUDGET_ERROR)
        self._recorder.reset()
        return TaskAdmission(arguments=arguments)

    def wrap(self, arguments: dict, raw: str) -> dict:
        collected = self._recorder.last
        if collected is None:
            return {"available": False, "error": "ETF 구성종목 데이터를 수집하지 못했습니다(수집 도구 미호출)"}
        if not collected.get("available"):
            return {"available": False, "error": collected.get("error", "ETF 구성종목 조회 실패")}
        if collected.get("holdings") is None:
            return {"available": True, "holdings": None}
        try:
            structured = EtfCompositionResult.model_validate(json.loads(raw))
        except (TypeError, ValueError, ValidationError):
            return {"available": False, "error": f"ETF 구성종목 종합 {_UNPARSED}"}
        result = {
            "available": True,
            "summary": structured.summary,
            "outlook": structured.outlook,
            "provenance": ETF_PROVENANCE_LABEL,
        }
        return {**result, **{key: collected[key] for key in ("data_as_of_note", "holdings") if key in collected}}


def _collect_once(context: ChatToolContext, ticker: str, market: str) -> dict:
    try:
        result = _tools.collect_etf_holding_signals(context, ticker, market)
    except Exception as exc:
        return {"available": False, "error": f"ETF 구성종목 조회 실패: {exc}"}
    # 실제 holdings가 있을 때만 종합 몫 차감 — 기존 도구가 수집 뒤 종합 서브에이전트를 띄울 때 차감한 것과 같은 시점.
    # 수집이 종합 몫을 떼어 두고 구성종목 수를 정했으므로 정상이라면 실패하지 않는다.
    if result.get("available") and result.get("holdings") is not None:
        if not context.budget.try_consume_llm_calls(ETF_COMPOSITION_LLM_CALL_ESTIMATE):
            return {"available": False, "error": BUDGET_ERROR}
    return result


def _etf_collect_tool(context: ChatToolContext, recorder: EtfCollectionRecorder, lock: threading.Lock) -> StructuredTool:
    def _run(ticker: str, market: str) -> str:
        with lock:
            if recorder.last is not None:
                # 위임 1건에 수집은 한 번 — 두 번째 호출은 비용을 두 번 쓰고 holdings와 요약이 어긋나게 만든다
                # (구성종목 공시·뉴스 문구에 "다시 수집하라" 같은 지시가 섞여 들어와도 여기서 막힌다).
                refusal = {"available": False, "error": "이미 수집했습니다 — 첫 수집 결과(holding_signals)만으로 종합하라"}
                return json.dumps(refusal, ensure_ascii=False)
            result = _collect_once(context, ticker, market)
            recorder.record(result)
        return json.dumps(result, ensure_ascii=False)

    return StructuredTool.from_function(
        func=_run, name=_ETF_COLLECT_TOOL, description=_ETF_COLLECT_DESCRIPTION, args_schema=TickerMarketArgs
    )


def _related_spec(model: Any, on_event: EventCallback | None) -> dict:
    return {
        "name": RELATED_COMPANIES,
        "description": RELATED_DESCRIPTION,
        "system_prompt": RELATED_SYSTEM_PROMPT + _RELATED_NAME_RULE,
        "model": model,
        "tools": [verify_companies_co_mentioned_in_news],
        "response_format": RelatedCompaniesResult,
        "middleware": [
            SubagentToolGuard(RELATED_COMPANIES, frozenset({_CO_MENTION_TOOL}), on_event, lock_target=False),
            ModelCallLimitMiddleware(run_limit=FIND_RELATED_RECURSION_LIMIT, exit_behavior="end"),
        ],
    }


def _etf_spec(context: ChatToolContext, recorder: EtfCollectionRecorder, model: Any, on_event: EventCallback | None) -> dict:
    return {
        "name": ETF_COMPOSITION,
        "description": ETF_DESCRIPTION,
        "system_prompt": ETF_SYSTEM_PROMPT + _ETF_COLLECT_RULE,
        "model": model,
        "tools": [_etf_collect_tool(context, recorder, threading.Lock())],
        "response_format": EtfCompositionResult,
        "middleware": [
            SubagentToolGuard(ETF_COMPOSITION, frozenset({_ETF_COLLECT_TOOL}), on_event),
            ModelCallLimitMiddleware(run_limit=ETF_COMPOSITION_RECURSION_LIMIT, exit_behavior="end"),
        ],
    }


def build_main_subagents(
    context: ChatToolContext, model: Any, on_event: EventCallback | None
) -> tuple[list[dict], SubagentTaskRouter]:
    """턴마다(메인 그래프가 턴마다 새로 만들어지므로) 그 턴의 context·예산에 바인딩한 서브에이전트 정의와 위임 정책."""
    recorder = EtfCollectionRecorder()
    specs = [
        stock_analyst_spec(context, model, on_event),
        _related_spec(model, on_event),
        _etf_spec(context, recorder, model, on_event),
    ]
    router = SubagentTaskRouter(
        {
            STOCK_ANALYST: StockAnalystTaskPolicy(context.budget),
            RELATED_COMPANIES: RelatedCompaniesTaskPolicy(context.budget),
            ETF_COMPOSITION: EtfCompositionTaskPolicy(context.budget, recorder),
        }
    )
    return specs, router
