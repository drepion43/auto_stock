"""stock_analyst를 메인 에이전트의 subagents=로 — docs/design/stock-analyst-subagent-migration.md.
실제 LLM은 부르지 않는다(메인·서브에이전트 모두 ScriptedModel)."""

import json
import threading

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from auto_stock.chat_agent.agent_middleware import SubagentToolGuard, parse_delegation_target
from auto_stock.chat_agent.credentials import STOCK_ANALYST_LLM_CALL_ESTIMATE, load_stock_analyst_mode
from auto_stock.chat_agent.deep_loop import DeepChatSession, run_turn_deep, verify_exposure
from auto_stock.chat_agent.models import QueryBudget
from auto_stock.chat_agent.stock_analyst import PROVENANCE_LABEL
from auto_stock.chat_agent.stock_analyst_subagent import SUBAGENT_NAME, StockAnalystTaskPolicy
from auto_stock.chat_agent.tool_registry import SIGNAL_TOOL_NAMES, as_subagent_signal_tools, main_tool_specs
from auto_stock.chat_agent.tools import ChatToolContext

DELEGATION = "대상: 005930 KOSPI\n요청: 지금 사도 되는지"


class ScriptedModel(BaseChatModel):
    script: list = Field(default_factory=list)
    inputs: list = Field(default_factory=list)
    bound: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound.append([t["name"] if isinstance(t, dict) else t.name for t in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.inputs.append(list(messages))
        step = self.script[min(len(self.inputs) - 1, len(self.script) - 1)]
        return ChatResult(generations=[ChatGeneration(message=step)])


def tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def context() -> ChatToolContext:
    return ChatToolContext(
        cache=object(), ml_model=None, llm_client=None, news_client=None, sentiment_client=None,
        budget=QueryBudget(max_llm_calls=50), account=None, agent_model="gpt-5.6-luna",
        scan_cache=object(), recommendation_coordinator=object(),
    )


def test_parse_delegation_target_reads_first_line_only_with_atomic_market():
    assert parse_delegation_target(DELEGATION) == ("005930", "KOSPI")
    assert parse_delegation_target("  대상:  nvda   NASDAQ \n요청") == ("NVDA", "NASDAQ")
    assert parse_delegation_target("대상: 005930 KRX") is None  # 합성 버킷은 안 됨
    assert parse_delegation_target("요청: x\n대상: 005930 KOSPI") is None
    assert parse_delegation_target("") is None


def test_main_tool_specs_subagent_mode_moves_signal_tools_out_of_the_main_agent():
    names = {spec.name for spec in main_tool_specs("subagent")}
    assert not names & SIGNAL_TOOL_NAMES and "stock_analyst" not in names
    assert {"resolve_ticker", "get_price_data", "find_related_companies"} <= names
    assert {spec.name for spec in main_tool_specs("tool")} >= SIGNAL_TOOL_NAMES | {"stock_analyst"}


def test_subagent_signal_tools_take_ticker_and_market_and_isolate_failures(mocker):
    mocker.patch("auto_stock.chat_agent.tools.tool_analyze_rule_engine", side_effect=RuntimeError("boom"))
    tools = {t.name: t for t in as_subagent_signal_tools(context(), threading.Lock())}
    assert set(tools) == SIGNAL_TOOL_NAMES
    result = json.loads(tools["analyze_rule_engine"].invoke({"ticker": "005930", "market": "KOSPI"}))
    assert result["available"] is False


class _Request:
    def __init__(self, name, args):
        self.tool_call = {"name": name, "args": args, "id": "x"}


def test_subagent_guard_refuses_other_ticker_and_builtin_tools_and_tags_events():
    events = []
    guard = SubagentToolGuard(SUBAGENT_NAME, SIGNAL_TOOL_NAMES, events.append)
    guard.before_agent({"messages": [HumanMessage(content=DELEGATION)]}, None)
    ran = []

    def handler(request):
        ran.append(request.tool_call["name"])
        return ToolMessage(content='{"available": true}', tool_call_id="x", name=request.tool_call["name"])

    other = guard.wrap_tool_call(_Request("analyze_rule_engine", {"ticker": "000660", "market": "KOSPI"}), handler)
    builtin = guard.wrap_tool_call(_Request("read_file", {"path": "/x"}), handler)
    same = guard.wrap_tool_call(_Request("analyze_rule_engine", {"ticker": "005930", "market": "KOSPI"}), handler)

    assert ran == ["analyze_rule_engine"]
    assert json.loads(other.content)["available"] is False and "005930" in json.loads(other.content)["error"]
    assert json.loads(builtin.content)["available"] is False
    assert json.loads(same.content) == {"available": True}
    assert all(e["subagent"] == SUBAGENT_NAME for e in events)
    assert [e["type"] for e in events] == ["tool_start", "tool_done"] * 3


def test_task_policy_checks_type_format_and_budget_before_running():
    budget = QueryBudget(max_llm_calls=STOCK_ANALYST_LLM_CALL_ESTIMATE)
    policy = StockAnalystTaskPolicy(budget)
    assert policy.check({"subagent_type": "general-purpose", "description": DELEGATION})["available"] is False
    assert "대상" in policy.check({"subagent_type": SUBAGENT_NAME, "description": "삼성전자 분석해줘"})["error"]
    assert policy.check({"subagent_type": SUBAGENT_NAME, "description": DELEGATION}) is None
    assert budget.llm_calls_made == STOCK_ANALYST_LLM_CALL_ESTIMATE
    assert "예산" in policy.check({"subagent_type": SUBAGENT_NAME, "description": DELEGATION})["error"]


def test_task_policy_wraps_structured_result_with_code_provenance():
    policy = StockAnalystTaskPolicy(QueryBudget(max_llm_calls=50))
    wrapped = policy.wrap(json.dumps({"summary": "s", "outlook": "UP", "signals_used": ["analyze_rule_engine"]}))
    assert wrapped == {
        "available": True,
        "judgment": {"summary": "s", "outlook": "UP", "signals_used": ["analyze_rule_engine"]},
        "provenance": PROVENANCE_LABEL,
    }
    assert policy.wrap("plain text without structure")["available"] is False


@pytest.fixture
def subagent_mode(monkeypatch):
    monkeypatch.setenv("CHAT_AGENT_STOCK_ANALYST_MODE", "subagent")


def test_subagent_mode_end_to_end_delegates_locks_target_and_wraps_result(subagent_mode, mocker):
    rule = mocker.patch(
        "auto_stock.chat_agent.tools.tool_analyze_rule_engine", return_value={"available": True, "candidate": None}
    )
    main = ScriptedModel(script=[
        AIMessage(content="", tool_calls=[tool_call("task", {"description": DELEGATION, "subagent_type": SUBAGENT_NAME}, "t1")]),
        AIMessage(content="서브에이전트 자율 조사 결과: 상승 전망입니다. 참고용입니다."),
    ])
    structured = {"summary": "규칙엔진 신호 없음", "outlook": "NEUTRAL", "signals_used": ["analyze_rule_engine"]}
    sub = ScriptedModel(script=[
        AIMessage(content="", tool_calls=[tool_call("analyze_rule_engine", {"ticker": "000660", "market": "KOSPI"}, "s1")]),
        AIMessage(content="", tool_calls=[tool_call("analyze_rule_engine", {"ticker": "005930", "market": "KOSPI"}, "s2")]),
        AIMessage(content="", tool_calls=[tool_call("StockAnalystResult", structured, "s3")]),
    ])
    events = []
    session = DeepChatSession(model=main, subagent_model=sub)

    answer = run_turn_deep(session, context(), "삼성전자 사도 돼?", on_event=events.append)

    assert answer.startswith("서브에이전트 자율 조사 결과")
    assert [c.kwargs["ticker"] for c in rule.call_args_list] == ["005930"]  # 000660 was refused
    task_done = next(e for e in events if e["type"] == "tool_done" and e["name"] == "task")
    assert task_done["result"]["provenance"] == PROVENANCE_LABEL
    assert task_done["result"]["judgment"]["outlook"] == "NEUTRAL"
    sub_done = [e for e in events if e["type"] == "tool_done" and e["name"] == "analyze_rule_engine"]
    assert sub_done and all(e["subagent"] == SUBAGENT_NAME for e in sub_done)
    assert "analyze_rule_engine" not in main.bound[0] and "task" in main.bound[0]


def test_subagent_mode_refuses_unknown_subagent_without_running_it(subagent_mode):
    main = ScriptedModel(script=[
        AIMessage(content="", tool_calls=[tool_call("task", {"description": DELEGATION, "subagent_type": "general-purpose"}, "t1")]),
        AIMessage(content="다른 방법으로 답합니다."),
    ])
    sub = ScriptedModel(script=[AIMessage(content="should not run")])
    events = []
    run_turn_deep(DeepChatSession(model=main, subagent_model=sub), context(), "x", on_event=events.append)
    assert sub.inputs == []
    task_done = next(e for e in events if e["type"] == "tool_done" and e["name"] == "task")
    assert task_done["result"]["available"] is False


def test_verify_exposure_passes_in_both_modes(monkeypatch):
    verify_exposure()
    monkeypatch.setenv("CHAT_AGENT_STOCK_ANALYST_MODE", "subagent")
    verify_exposure()


def test_unknown_mode_fails_loudly(monkeypatch):
    monkeypatch.setenv("CHAT_AGENT_STOCK_ANALYST_MODE", "maybe")
    with pytest.raises(ValueError):
        load_stock_analyst_mode()
