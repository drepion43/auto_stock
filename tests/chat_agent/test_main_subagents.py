"""관련기업·ETF 서브에이전트를 메인 subagents=로 — docs/design/stock-analyst-subagent-migration.md §12.
실제 LLM·데이터 소스는 부르지 않는다(메인·서브에이전트 모두 ScriptedModel, 데이터 함수는 mock)."""

import json

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from auto_stock.chat_agent.agent_middleware import SubagentTaskRouter, TaskAdmission
from auto_stock.chat_agent.credentials import (
    ETF_COMPOSITION_LLM_CALL_ESTIMATE,
    FIND_RELATED_LLM_CALL_ESTIMATE,
)
from auto_stock.chat_agent.deep_loop import DeepChatSession, run_turn_deep
from auto_stock.chat_agent.etf_composition import PROVENANCE_LABEL as ETF_PROVENANCE
from auto_stock.chat_agent.main_subagents import (
    ETF_COMPOSITION,
    RELATED_COMPANIES,
    SUBAGENT_NAMES,
    EtfCollectionRecorder,
    EtfCompositionTaskPolicy,
    RelatedCompaniesTaskPolicy,
    build_main_subagents,
)
from auto_stock.chat_agent.models import QueryBudget
from auto_stock.chat_agent.related_companies import MAX_RELATED
from auto_stock.chat_agent.tool_registry import DELEGATED_TOOL_NAMES, main_tool_specs
from auto_stock.chat_agent.tools import ChatToolContext, _affordable_etf_holdings
from auto_stock.evaluation.answer_capture import _DELEGATED_TYPES, ToolCallSink

RELATED_ARGS = {"subagent_type": RELATED_COMPANIES, "description": "대상: 005930 KOSPI\n요청: 관련기업 알려줘"}
ETF_ARGS = {"subagent_type": ETF_COMPOSITION, "description": "대상: 069500 KOSPI\n요청: 왜 올랐어"}
COLLECTED = {
    "available": True,
    "holdings": [{"ticker": "005930", "name": "삼성전자", "weight_pct": 30.0}],
    "holding_signals": [{"ticker": "005930", "weight_pct": 30.0, "name": "삼성전자", "disclosure": None, "news_sentiment": None}],
}


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


def context(max_llm_calls: int = 50) -> ChatToolContext:
    return ChatToolContext(
        cache=object(), ml_model=None, llm_client=None, news_client=None, sentiment_client=None,
        budget=QueryBudget(max_llm_calls=max_llm_calls), account=None, agent_model="gpt-5.6-luna",
        scan_cache=object(), recommendation_coordinator=object(),
    )


@pytest.fixture
def subagent_mode(monkeypatch):
    monkeypatch.setenv("CHAT_AGENT_STOCK_ANALYST_MODE", "subagent")


def test_subagent_mode_main_agent_keeps_only_non_delegated_tools():
    names = {spec.name for spec in main_tool_specs("subagent")}
    assert not names & DELEGATED_TOOL_NAMES
    assert {"find_related_companies", "analyze_etf_composition", "stock_analyst"} <= DELEGATED_TOOL_NAMES
    assert names == {
        "resolve_ticker", "get_price_data", "get_market_scan_recommendations", "get_sector_recommendations",
        "get_index_quote",
    }


def test_build_main_subagents_registers_all_three_with_one_router():
    specs, router = build_main_subagents(context(), ScriptedModel(), None)
    assert tuple(spec["name"] for spec in specs) == SUBAGENT_NAMES
    assert isinstance(router, SubagentTaskRouter)
    assert router.admit({"subagent_type": "general-purpose", "description": "대상: 005930 KOSPI"}).refusal is not None


def test_evaluation_capture_knows_every_delegated_type():
    assert _DELEGATED_TYPES == set(SUBAGENT_NAMES)
    sink = ToolCallSink()
    for args in (RELATED_ARGS, ETF_ARGS):
        sink.on_event({"type": "tool_done", "name": "task", "arguments": args, "result": {}})
    assert [(r.name, r.arguments["ticker"]) for r in sink.records] == [
        ("find_related_companies", "005930"), ("analyze_etf_composition", "069500"),
    ]


# --- find_related_companies policy -------------------------------------------------------------------------------


def test_related_policy_adds_resolved_company_name_after_target_and_debits_budget(mocker):
    mocker.patch("auto_stock.chat_agent.tools.resolve_company_name", return_value="삼성전자")
    budget = QueryBudget(max_llm_calls=50)
    admitted = RelatedCompaniesTaskPolicy(budget).admit(RELATED_ARGS)
    assert admitted.refusal is None
    assert admitted.arguments["description"].splitlines() == ["대상: 005930 KOSPI", "회사명: 삼성전자", "요청: 관련기업 알려줘"]
    assert budget.llm_calls_made == FIND_RELATED_LLM_CALL_ESTIMATE


@pytest.mark.parametrize("resolved", [None, RuntimeError("DART down")])
def test_related_policy_refuses_unresolved_name_without_spending_budget(mocker, resolved):
    kwargs = {"side_effect": resolved} if isinstance(resolved, Exception) else {"return_value": resolved}
    mocker.patch("auto_stock.chat_agent.tools.resolve_company_name", **kwargs)
    budget = QueryBudget(max_llm_calls=50)
    admitted = RelatedCompaniesTaskPolicy(budget).admit(RELATED_ARGS)
    assert admitted.refusal["available"] is False and budget.llm_calls_made == 0


def test_related_policy_refuses_bad_format_and_exhausted_budget(mocker):
    mocker.patch("auto_stock.chat_agent.tools.resolve_company_name", return_value="삼성전자")
    policy = RelatedCompaniesTaskPolicy(QueryBudget(max_llm_calls=FIND_RELATED_LLM_CALL_ESTIMATE - 1))
    assert "대상" in policy.admit({**RELATED_ARGS, "description": "삼성 관련기업"}).refusal["error"]
    assert "예산" in policy.admit(RELATED_ARGS).refusal["error"]


def test_related_policy_wraps_like_the_tool_and_caps_the_list():
    items = [
        {"ticker": f"00{i}", "market": "KOSPI", "name": f"c{i}", "relation": "경쟁사", "confidence": "inferred"}
        for i in range(MAX_RELATED + 2)
    ]
    policy = RelatedCompaniesTaskPolicy(QueryBudget(max_llm_calls=50))
    wrapped = policy.wrap(RELATED_ARGS, json.dumps({"related": items}))
    assert wrapped == {"available": True, "related": items[:MAX_RELATED]}
    assert policy.wrap(RELATED_ARGS, "not json")["available"] is False


# --- analyze_etf_composition policy ------------------------------------------------------------------------------


def _etf_policy(budget: QueryBudget | None = None) -> tuple[EtfCompositionTaskPolicy, EtfCollectionRecorder]:
    recorder = EtfCollectionRecorder()
    return EtfCompositionTaskPolicy(budget or QueryBudget(max_llm_calls=50), recorder), recorder


def test_etf_policy_takes_holdings_from_the_collection_tool_not_the_llm():
    policy, recorder = _etf_policy()
    assert policy.admit(ETF_ARGS) == TaskAdmission(arguments=ETF_ARGS)
    recorder.record({**COLLECTED, "data_as_of_note": "N-PORT 기준"})
    raw = json.dumps({"summary": "반도체 강세", "outlook": "UP", "holdings": "LLM이 지어낸 값"})
    assert policy.wrap(ETF_ARGS, raw) == {
        "available": True, "summary": "반도체 강세", "outlook": "UP", "provenance": ETF_PROVENANCE,
        "data_as_of_note": "N-PORT 기준", "holdings": COLLECTED["holdings"],
    }


def test_etf_policy_result_follows_collection_outcome():
    policy, recorder = _etf_policy()
    policy.admit(ETF_ARGS)
    assert policy.wrap(ETF_ARGS, "{}")["available"] is False  # 수집 도구를 부르지 않음
    recorder.record({"available": False, "error": "ETF 구성종목 조회 실패: x"})
    assert policy.wrap(ETF_ARGS, "{}") == {"available": False, "error": "ETF 구성종목 조회 실패: x"}
    recorder.record({"available": True, "holdings": None})
    assert policy.wrap(ETF_ARGS, "{}") == {"available": True, "holdings": None}


def test_etf_policy_resets_previous_collection_without_spending_budget():
    budget = QueryBudget(max_llm_calls=50)
    policy, recorder = _etf_policy(budget)
    recorder.record(COLLECTED)
    policy.admit(ETF_ARGS)
    assert recorder.last is None and budget.llm_calls_made == 0


def test_etf_policy_refuses_when_not_even_one_holding_is_affordable():
    # 기존 도구와 같은 기준: 종합 몫을 떼고 구성종목 1개(공시+뉴스 2건)도 못 하면 시작하지 않는다
    budget = QueryBudget(max_llm_calls=ETF_COMPOSITION_LLM_CALL_ESTIMATE + 1)
    assert _affordable_etf_holdings(budget) == 0
    policy, _ = _etf_policy(budget)
    assert "예산" in policy.admit(ETF_ARGS).refusal["error"]


def _collect_tool(ctx: ChatToolContext):
    specs, _ = build_main_subagents(ctx, ScriptedModel(), None)
    return next(spec for spec in specs if spec["name"] == ETF_COMPOSITION)["tools"][0]


def test_etf_collect_runs_once_and_charges_synthesis_only_for_real_holdings(mocker):
    collect = mocker.patch("auto_stock.chat_agent.tools.collect_etf_holding_signals", return_value=COLLECTED)
    ctx = context()
    tool = _collect_tool(ctx)
    first = json.loads(tool.invoke({"ticker": "069500", "market": "KOSPI"}))
    second = json.loads(tool.invoke({"ticker": "069500", "market": "KOSPI"}))
    assert first == COLLECTED and second["available"] is False and "이미 수집" in second["error"]
    collect.assert_called_once()
    assert ctx.budget.llm_calls_made == ETF_COMPOSITION_LLM_CALL_ESTIMATE


def test_etf_collect_for_a_non_etf_spends_no_synthesis_budget(mocker):
    mocker.patch("auto_stock.chat_agent.tools.collect_etf_holding_signals", return_value={"available": True, "holdings": None})
    ctx = context()
    assert json.loads(_collect_tool(ctx).invoke({"ticker": "005930", "market": "KOSPI"}))["holdings"] is None
    assert ctx.budget.llm_calls_made == 0


def test_related_policy_drops_name_lines_written_by_the_model(mocker):
    mocker.patch("auto_stock.chat_agent.tools.resolve_company_name", return_value="삼성전자")
    spoofed = {**RELATED_ARGS, "description": "대상: 005930 KOSPI\n회사명: 엉뚱한회사\n요청: 관련기업"}
    lines = RelatedCompaniesTaskPolicy(QueryBudget(max_llm_calls=50)).admit(spoofed).arguments["description"].splitlines()
    assert lines == ["대상: 005930 KOSPI", "회사명: 삼성전자", "요청: 관련기업"]


def test_replay_records_and_replays_subagent_leaf_tools():
    from auto_stock.chat_agent import related_companies
    from auto_stock.chat_agent import tools as chat_tools
    from auto_stock.evaluation.replay import FixtureRecorder, ReplayStore, replay_patchers
    from auto_stock.evaluation.tool_patching import applied

    recorder = FixtureRecorder()
    co_args = {"company_a": "삼성전자", "company_b": "SK하이닉스", "market": "KOSPI"}
    for name, args, result in (
        ("collect_etf_holding_signals", {"ticker": "069500", "market": "KOSPI"}, COLLECTED),
        ("verify_companies_co_mentioned_in_news", co_args, {"co_mentioned": True}),
    ):
        recorder.on_event({"type": "tool_done", "name": name, "arguments": args, "result": result, "subagent": "x"})
    store = ReplayStore(recorder.entries)
    with applied(replay_patchers(store)):
        assert chat_tools.collect_etf_holding_signals(object(), "069500", "KOSPI") == COLLECTED
        assert related_companies.co_mention_check(**co_args) == {"co_mentioned": True}
    assert store.misses == []


# --- end to end through the main loop ----------------------------------------------------------------------------


def _task(args: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=[tool_call("task", args, "t1")])


def test_related_companies_delegation_end_to_end(subagent_mode, mocker):
    mocker.patch("auto_stock.chat_agent.tools.resolve_company_name", return_value="삼성전자")
    news = mocker.patch("auto_stock.chat_agent.related_companies.search_news", return_value=[])
    related = [{"ticker": "000660", "market": "KOSPI", "name": "SK하이닉스", "relation": "경쟁사", "confidence": "inferred"}]
    main = ScriptedModel(script=[_task(RELATED_ARGS), AIMessage(content="추정 관련기업: SK하이닉스")])
    sub = ScriptedModel(script=[
        AIMessage(content="", tool_calls=[tool_call(
            "verify_companies_co_mentioned_in_news", {"company_a": "삼성전자", "company_b": "SK하이닉스", "market": "KOSPI"}, "s1"
        )]),
        AIMessage(content="", tool_calls=[tool_call("RelatedCompaniesResult", {"related": related}, "s2")]),
    ])
    events = []

    answer = run_turn_deep(DeepChatSession(model=main, subagent_model=sub), context(), "삼성전자 관련기업?", events.append)

    assert answer == "추정 관련기업: SK하이닉스"
    first_human = next(m for m in sub.inputs[0] if isinstance(m, HumanMessage))
    assert "회사명: 삼성전자" in first_human.text
    news.assert_called_once()
    task_done = next(e for e in events if e["type"] == "tool_done" and e["name"] == "task")
    assert task_done["result"] == {"available": True, "related": related}
    inner = [e for e in events if e["type"] == "tool_done" and e.get("subagent") == RELATED_COMPANIES]
    assert [e["name"] for e in inner] == ["verify_companies_co_mentioned_in_news"]


def test_etf_delegation_end_to_end_locks_the_etf_and_uses_collected_holdings(subagent_mode, mocker):
    collect = mocker.patch("auto_stock.chat_agent.tools.collect_etf_holding_signals", return_value=COLLECTED)
    main = ScriptedModel(script=[_task(ETF_ARGS), AIMessage(content="ETF 구성종목 집계 분석: 삼성전자 영향")])
    sub = ScriptedModel(script=[
        AIMessage(content="", tool_calls=[tool_call("collect_etf_holding_signals", {"ticker": "102110", "market": "KOSPI"}, "s1")]),
        AIMessage(content="", tool_calls=[tool_call("collect_etf_holding_signals", {"ticker": "069500", "market": "KOSPI"}, "s2")]),
        AIMessage(content="", tool_calls=[tool_call("EtfCompositionResult", {"summary": "삼성전자 강세", "outlook": "UP"}, "s3")]),
    ])
    events = []
    ctx = context()

    run_turn_deep(DeepChatSession(model=main, subagent_model=sub), ctx, "KODEX 200 왜 올랐어?", events.append)

    collect.assert_called_once_with(ctx, "069500", "KOSPI")  # 102110 was refused
    assert ctx.budget.llm_calls_made == ETF_COMPOSITION_LLM_CALL_ESTIMATE
    task_done = next(e for e in events if e["type"] == "tool_done" and e["name"] == "task")
    assert task_done["result"]["holdings"] == COLLECTED["holdings"]
    assert task_done["result"]["provenance"] == ETF_PROVENANCE
    assert "analyze_etf_composition" not in main.bound[0] and "task" in main.bound[0]
