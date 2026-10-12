"""tool_registry.py — docs/design/main-agent-migration-plan.md §4.0(0단계).

도구는 `TOOL_SPECS` 한 곳에만 정의하고 OpenAI 스키마·디스패치·stock_analyst 바인딩·
LangChain 도구를 전부 거기서 파생한다. **모델이 받는 바이트는 바뀌면 안 된다** — 골든
픽스처(`fixtures/`)는 레지스트리 도입 직전 코드에서 그대로 덤프한 것이다."""

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from auto_stock.chat_agent import tool_registry
from auto_stock.chat_agent.models import QueryBudget
from auto_stock.chat_agent.prompt import SYSTEM_PROMPT
from auto_stock.chat_agent.tool_args import TickerMarketArgs
from auto_stock.chat_agent.tool_registry import (
    TOOL_SPECS,
    ToolSpec,
    as_langchain_tools,
    bind_stock_analyst_tools,
    call_isolated,
    dispatch,
    openai_schemas,
)
from auto_stock.chat_agent.tools import ChatToolContext

FIXTURES = Path(__file__).parent / "fixtures"
DEEPAGENTS_BUILTIN_TOOL_NAMES = {
    "ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute", "task", "write_todos",
}


def _context() -> ChatToolContext:
    return ChatToolContext(
        cache=object(), ml_models={}, llm_client=None, news_client=None, sentiment_client=None,
        budget=QueryBudget(max_llm_calls=8), account=None, agent_model="gpt-5.6-luna",
        scan_cache=object(), recommendation_coordinator=object(),
    )


# --- 골든: 모델이 받는 바이트 무변경 ---


def test_openai_schemas_equal_pre_registry_golden():
    golden = json.loads((FIXTURES / "tool_schemas_golden.json").read_text(encoding="utf-8"))

    assert openai_schemas() == golden


def test_openai_schemas_preserve_key_order_of_golden():
    """JSON 동등을 넘어 직렬화 결과(키 순서 포함)까지 같아야 요청 바이트가 같다."""
    golden_text = (FIXTURES / "tool_schemas_golden.json").read_text(encoding="utf-8")

    assert json.dumps(openai_schemas(), ensure_ascii=False, indent=2) + "\n" == golden_text


def test_tool_schemas_module_reexports_registry_schemas():
    from auto_stock.chat_agent.tool_schemas import TOOL_SCHEMAS

    assert TOOL_SCHEMAS == openai_schemas()


def test_stock_analyst_bound_tools_equal_pre_registry_golden():
    golden = json.loads((FIXTURES / "stock_analyst_tools_golden.json").read_text(encoding="utf-8"))

    bound = bind_stock_analyst_tools(_context(), "005930", "KRX")

    assert [{"name": t.name, "description": t.description, "args": t.args} for t in bound] == golden


def test_system_prompt_is_byte_identical_to_golden():
    assert SYSTEM_PROMPT.encode("utf-8") == (FIXTURES / "system_prompt_golden.txt").read_bytes()


def test_every_registry_tool_is_mentioned_in_system_prompt():
    assert [s.name for s in TOOL_SPECS if s.name not in SYSTEM_PROMPT] == []


# --- 레지스트리 무결성 ---


def test_registry_names_are_unique():
    names = [s.name for s in TOOL_SPECS]

    assert len(names) == len(set(names))


def test_registry_descriptions_are_nonempty():
    assert all(len(s.description) > 10 for s in TOOL_SPECS)


def test_exactly_the_six_signal_tools_are_bound_to_stock_analyst():
    assert [s.name for s in TOOL_SPECS if s.subagent_description is not None] == [
        "analyze_rule_engine",
        "analyze_ml_prediction",
        "analyze_chart_pattern",
        "analyze_disclosures",
        "analyze_news_sentiment",
        "analyze_position_sizing",
    ]


def test_registry_names_do_not_collide_with_deepagents_builtins():
    """겹치면 내장 도구 비활성화 장치에 의해 우리 도구가 조용히 사라진다."""
    assert {s.name for s in TOOL_SPECS} & DEEPAGENTS_BUILTIN_TOOL_NAMES == set()


def test_every_spec_points_to_an_existing_tools_function():
    from auto_stock.chat_agent import tools

    assert [s.fn_name for s in TOOL_SPECS if not callable(getattr(tools, s.fn_name, None))] == []


# --- 도구 추가 비용: ToolSpec 1개 추가로 모든 파생물에 나타나야 한다 ---


def test_adding_one_spec_propagates_to_schema_dispatch_and_langchain_tools(monkeypatch):
    extra = ToolSpec(
        name="analyze_dummy",
        description="테스트용 가짜 도구 — 레지스트리 전파 확인",
        args_model=TickerMarketArgs,
        fn_name="tool_get_price_data",
    )
    monkeypatch.setattr(tool_registry, "TOOL_SPECS", (*TOOL_SPECS, extra))

    assert "analyze_dummy" in {s["name"] for s in tool_registry.openai_schemas()}
    assert "analyze_dummy" in tool_registry.dispatch()
    assert "analyze_dummy" in {t.name for t in tool_registry.as_langchain_tools(_context(), threading.Lock())}


# --- 디스패치: 구 루프 동작 보존 ---


def test_dispatch_covers_every_spec():
    assert set(dispatch()) == {s.name for s in TOOL_SPECS}


def test_dispatch_resolves_function_at_call_time_so_patches_apply(mocker):
    mock_price = mocker.patch(
        "auto_stock.chat_agent.tools.tool_get_price_data", return_value={"available": True, "latest": None}
    )
    context = _context()

    result = dispatch()["get_price_data"](context, ticker="005930", market="KRX")

    assert result == {"available": True, "latest": None}
    mock_price.assert_called_once_with(context, ticker="005930", market="KRX")


def test_dispatch_keeps_typeerror_for_legacy_loop_to_handle():
    """구 루프(loop._dispatch_tool_call)는 TypeError를 '도구 인자 불일치'로 격리한다 —
    디스패치가 그 예외를 삼키면 기존 계약(C9)이 바뀐다."""
    with pytest.raises(TypeError):
        dispatch()["get_price_data"](_context(), wrong="x")


# --- call_isolated: 절대 raise 안 함 ---


def test_call_isolated_returns_tool_result(mocker):
    mocker.patch("auto_stock.chat_agent.tools.tool_get_price_data", return_value={"available": True, "latest": None})
    spec = next(s for s in TOOL_SPECS if s.name == "get_price_data")

    assert call_isolated(spec, _context(), {"ticker": "005930", "market": "KOSPI"}) == {"available": True, "latest": None}


def test_call_isolated_isolates_unexpected_tool_exception(mocker):
    mocker.patch("auto_stock.chat_agent.tools.tool_get_price_data", side_effect=RuntimeError("boom"))
    spec = next(s for s in TOOL_SPECS if s.name == "get_price_data")

    result = call_isolated(spec, _context(), {"ticker": "005930", "market": "KOSPI"})

    assert result["available"] is False
    assert "boom" in result["error"]


def test_call_isolated_rejects_invalid_market_without_calling_tool(mocker):
    mock_price = mocker.patch("auto_stock.chat_agent.tools.tool_get_price_data")
    spec = next(s for s in TOOL_SPECS if s.name == "get_price_data")

    result = call_isolated(spec, _context(), {"ticker": "005930", "market": "TOKYO"})

    assert result["available"] is False
    assert result["error"].startswith("도구 인자 파싱 실패")
    mock_price.assert_not_called()


# --- LangChain 도구 (1단계 메인 에이전트용) ---


def test_langchain_tools_return_json_string_of_tool_result(mocker):
    mocker.patch(
        "auto_stock.chat_agent.tools.tool_get_price_data", return_value={"available": True, "latest": {"close": 61000}}
    )
    tool = next(t for t in as_langchain_tools(_context(), threading.Lock()) if t.name == "get_price_data")

    out = tool.invoke({"ticker": "005930", "market": "KOSPI"})

    assert out == json.dumps({"available": True, "latest": {"close": 61000}}, ensure_ascii=False)


def test_langchain_tools_shape_validation_error_like_legacy_loop(mocker):
    mock_price = mocker.patch("auto_stock.chat_agent.tools.tool_get_price_data")
    tool = next(t for t in as_langchain_tools(_context(), threading.Lock()) if t.name == "get_price_data")

    out = json.loads(tool.invoke({"ticker": "005930", "market": "TOKYO"}))

    assert out["available"] is False
    assert out["error"].startswith("도구 인자 파싱 실패")
    mock_price.assert_not_called()


def test_langchain_tools_serialize_execution_under_turn_lock(mocker):
    """LangGraph ToolNode는 한 응답의 여러 tool_call을 병렬 실행한다(스파이크 S3) —
    기존 순차 실행 계약(C11)은 턴 Lock으로 지킨다."""
    lock = threading.Lock()
    seen = []
    mocker.patch(
        "auto_stock.chat_agent.tools.tool_get_price_data",
        side_effect=lambda *a, **k: seen.append(lock.locked()) or {"available": True},
    )
    tool = next(t for t in as_langchain_tools(_context(), lock) if t.name == "get_price_data")

    tool.invoke({"ticker": "005930", "market": "KOSPI"})

    assert seen == [True]
    assert not lock.locked()


# --- import 순서: 순환 import 회귀 방지(새 인터프리터) ---


@pytest.mark.parametrize(
    "module",
    [
        "auto_stock.chat_agent.loop",
        "auto_stock.orchestrator.recommendation_coordinator",
        "auto_stock.chat_agent.tool_schemas",
        "auto_stock.chat_agent.tool_registry",
    ],
)
def test_module_imports_cleanly_in_fresh_interpreter(module):
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"], check=False, capture_output=True, text=True, timeout=120
    )

    assert proc.returncode == 0, proc.stderr[-2000:]


def test_stock_analyst_bound_tools_never_run_concurrently(mocker):
    """deepagents(LangGraph)는 서브에이전트가 한 응답에 낸 여러 tool_call도 병렬 실행한다 —
    바인딩 도구들은 같은 OHLCVCache DuckDB 연결을 공유하므로 동시에 돌면 'tuple index out of
    range' 같은 무작위 오류가 났다(2026-10-12 재현, NVDA 분석 실패의 원인)."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    active = {"n": 0, "max": 0}
    guard = threading.Lock()

    def slow(context, ticker, market):
        with guard:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        time.sleep(0.03)
        with guard:
            active["n"] -= 1
        return {"available": True}

    for spec in TOOL_SPECS:
        if spec.subagent_description is not None:
            mocker.patch(f"auto_stock.chat_agent.tools.{spec.fn_name}", side_effect=slow)
    bound = bind_stock_analyst_tools(_context(), "NVDA", "NASDAQ")

    with ThreadPoolExecutor(max_workers=len(bound)) as executor:
        list(executor.map(lambda t: t.invoke({}), bound))

    assert active["max"] == 1
