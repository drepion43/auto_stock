"""deep_loop.py — 1단계: 메인 대화 루프를 create_deep_agent로 교체(기능 플래그 뒤).
docs/design/main-agent-migration-plan.md §4.1 테스트 계획. 실제 LLM은 부르지 않는다 —
`ScriptedModel`이 정해진 AIMessage를 순서대로 돌려주고, 받은 입력·바인딩된 도구를 기록한다."""

import json
import threading
import time

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from auto_stock.chat_agent import deep_loop
from auto_stock.chat_agent.credentials import MAIN_AGENT_RUN_LIMIT
from auto_stock.chat_agent.deep_loop import (
    DeepChatSession,
    ExposureError,
    TracingNotAllowedError,
    check_tracing_allowed,
    run_turn_deep,
    verify_exposure,
)
from auto_stock.chat_agent.fallbacks import (
    API_ERROR_FALLBACK,
    SESSION_LIMIT_FALLBACK,
    TOOL_LIMIT_FALLBACK,
)
from auto_stock.chat_agent.models import QueryBudget
from auto_stock.chat_agent.prompt import SYSTEM_PROMPT
from auto_stock.chat_agent.tool_registry import TOOL_SPECS
from auto_stock.chat_agent.tools import ChatToolContext

REGISTRY_NAMES = [s.name for s in TOOL_SPECS]


class ScriptedModel(BaseChatModel):
    script: list = Field(default_factory=list)
    inputs: list = Field(default_factory=list)
    bound: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound.append(([t["name"] if isinstance(t, dict) else t.name for t in tools], kwargs))
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.inputs.append(list(messages))
        step = self.script[min(len(self.inputs) - 1, len(self.script) - 1)]
        if isinstance(step, Exception):
            raise step
        msg = step(messages) if callable(step) else step
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _tc(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _price_call(call_id: str = "c1", ticker: str = "005930") -> dict:
    return _tc("get_price_data", {"ticker": ticker, "market": "KOSPI"}, call_id)


def _context(**overrides) -> ChatToolContext:
    values = dict(
        cache=object(), ml_model=None, llm_client=None, news_client=None, sentiment_client=None,
        budget=QueryBudget(max_llm_calls=50), account=None, agent_model="gpt-5.6-luna",
        scan_cache=object(), recommendation_coordinator=object(),
    )
    values.update(overrides)
    return ChatToolContext(**values)


def _session(script: list, **kwargs) -> DeepChatSession:
    return DeepChatSession(model=ScriptedModel(script=script), **kwargs)


@pytest.fixture
def price_tool(mocker):
    return mocker.patch(
        "auto_stock.chat_agent.tools.tool_get_price_data",
        return_value={"available": True, "latest": {"close": 263000.0}},
    )


# --- 1~3: 기본 응답, 시스템 프롬프트, 멀티턴 ---


def test_immediate_answer_returns_text_and_emits_single_thinking():
    session = _session([AIMessage(content="안녕하세요")])
    events = []

    text = run_turn_deep(session, _context(), "안녕", on_event=events.append)

    assert text == "안녕하세요"
    assert events == [{"type": "thinking"}]


def test_model_receives_exactly_one_system_message_equal_to_system_prompt_each_turn():
    session = _session([AIMessage(content="첫 답"), AIMessage(content="둘째 답")])

    run_turn_deep(session, _context(), "첫 질문")
    run_turn_deep(session, _context(), "둘째 질문")

    for model_input in session.model.inputs:
        systems = [m for m in model_input if isinstance(m, SystemMessage)]
        assert len(systems) == 1
        assert systems[0].text == SYSTEM_PROMPT


def test_second_turn_input_contains_first_turn_and_sessions_do_not_mix():
    session = _session([AIMessage(content="철수님 반가워요"), AIMessage(content="철수입니다")])
    other = _session([AIMessage(content="다른 세션")])

    run_turn_deep(session, _context(), "내 이름은 철수야")
    run_turn_deep(other, _context(), "누구세요")
    run_turn_deep(session, _context(), "내 이름이 뭐지?")

    second_turn_texts = [m.text for m in session.model.inputs[1] if isinstance(m, (HumanMessage, AIMessage))]
    assert second_turn_texts == ["내 이름은 철수야", "철수님 반가워요", "내 이름이 뭐지?"]
    assert all("누구세요" not in m.text for m in session.model.inputs[1])


# --- 4: 도구 왕복과 이벤트 ---


def test_tool_result_round_trips_as_legacy_json_and_emits_ordered_events(price_tool):
    session = _session([AIMessage(content="", tool_calls=[_price_call()]), AIMessage(content="263,000원입니다")])
    events = []

    text = run_turn_deep(session, _context(), "삼성전자 얼마야", on_event=events.append)

    assert text == "263,000원입니다"
    tool_messages = [m for m in session.model.inputs[1] if isinstance(m, ToolMessage)]
    assert tool_messages[0].content == json.dumps({"available": True, "latest": {"close": 263000.0}}, ensure_ascii=False)
    assert events == [
        {"type": "thinking"},
        {"type": "tool_start", "name": "get_price_data", "arguments": {"ticker": "005930", "market": "KOSPI"}},
        {
            "type": "tool_done",
            "name": "get_price_data",
            "arguments": {"ticker": "005930", "market": "KOSPI"},
            "result": {"available": True, "latest": {"close": 263000.0}},
        },
        {"type": "thinking"},
    ]


def test_tool_receives_the_turn_context(price_tool):
    context = _context()
    session = _session([AIMessage(content="", tool_calls=[_price_call()]), AIMessage(content="끝")])

    run_turn_deep(session, context, "q")

    assert price_tool.call_args.args[0] is context


# --- 5: 오류 형식(C7/C8/C9) ---


def test_unknown_and_builtin_tool_calls_are_refused_with_legacy_message(price_tool):
    session = _session([
        AIMessage(content="", tool_calls=[_tc("nope_tool", {}, "c1"), _tc("write_file", {"file_path": "/x", "content": "y"}, "c2")]),
        AIMessage(content="계속"),
    ])

    text = run_turn_deep(session, _context(), "q")

    assert text == "계속"
    contents = {m.tool_call_id: json.loads(m.content) for m in session.model.inputs[1] if isinstance(m, ToolMessage)}
    assert contents["c1"] == {"available": False, "error": "알 수 없는 도구: nope_tool"}
    assert contents["c2"] == {"available": False, "error": "알 수 없는 도구: write_file"}


def test_schema_violating_arguments_get_legacy_parse_error_and_tool_is_not_called(price_tool):
    session = _session([
        AIMessage(content="", tool_calls=[_tc("get_price_data", {"ticker": "005930", "market": "TOKYO"}, "c1")]),
        AIMessage(content="다시 시도할게요"),
    ])

    run_turn_deep(session, _context(), "q")

    content = json.loads(next(m for m in session.model.inputs[1] if isinstance(m, ToolMessage)).content)
    assert content["available"] is False
    assert content["error"].startswith("도구 인자 파싱 실패")
    price_tool.assert_not_called()


def test_invalid_tool_call_json_is_fed_back_to_model_for_retry():
    bad = {"name": "get_price_data", "args": "{broken", "id": "c9", "error": "parse", "type": "invalid_tool_call"}
    session = _session([AIMessage(content="", invalid_tool_calls=[bad]), AIMessage(content="재시도 후 답변")])

    text = run_turn_deep(session, _context(), "q")

    assert text == "재시도 후 답변"
    fed_back = next(m for m in session.model.inputs[1] if isinstance(m, ToolMessage))
    assert fed_back.tool_call_id == "c9"
    assert json.loads(fed_back.content)["error"].startswith("도구 인자 파싱 실패")


def test_empty_final_text_returns_api_error_fallback_and_rolls_back():
    session = _session([AIMessage(content="")])

    text = run_turn_deep(session, _context(), "q")

    assert text == API_ERROR_FALLBACK
    assert session.messages == []


def test_tool_exception_is_isolated_and_loop_survives(mocker):
    mocker.patch("auto_stock.chat_agent.tools.tool_get_price_data", side_effect=RuntimeError("db locked"))
    session = _session([AIMessage(content="", tool_calls=[_price_call()]), AIMessage(content="가격 조회 실패를 안내")])

    text = run_turn_deep(session, _context(), "q")

    assert text == "가격 조회 실패를 안내"
    content = json.loads(next(m for m in session.model.inputs[1] if isinstance(m, ToolMessage)).content)
    assert content["available"] is False and "db locked" in content["error"]


# --- 6~7: 한도·실패와 턴 롤백(C5/C6) ---


def test_run_limit_stops_after_exact_model_calls_returns_fallback_and_rolls_back(price_tool):
    looping = lambda msgs: AIMessage(content="", tool_calls=[_price_call(f"c{len(msgs)}")])
    session = _session([looping])
    events = []

    text = run_turn_deep(session, _context(), "q", on_event=events.append)

    assert text == TOOL_LIMIT_FALLBACK
    assert len(session.model.inputs) == MAIN_AGENT_RUN_LIMIT
    assert events.count({"type": "thinking"}) == MAIN_AGENT_RUN_LIMIT
    assert session.messages == []


def test_five_tool_calls_in_one_response_use_only_two_model_calls(price_tool):
    calls = [_price_call(f"c{i}", f"00000{i}") for i in range(5)]
    session = _session([AIMessage(content="", tool_calls=calls), AIMessage(content="다섯 종목 정리")])

    text = run_turn_deep(session, _context(), "q")

    assert text == "다섯 종목 정리"
    assert len(session.model.inputs) == 2
    assert price_tool.call_count == 5


def test_failed_turn_then_success_then_next_turn_keep_consistent_history():
    session = _session([RuntimeError("upstream 500"), AIMessage(content="둘째 답"), AIMessage(content="셋째 답")])

    first = run_turn_deep(session, _context(), "첫 질문")
    second = run_turn_deep(session, _context(), "둘째 질문")
    third = run_turn_deep(session, _context(), "셋째 질문")

    assert (first, second, third) == (API_ERROR_FALLBACK, "둘째 답", "셋째 답")
    third_input = [m.text for m in session.model.inputs[2] if isinstance(m, (HumanMessage, AIMessage))]
    assert third_input == ["둘째 질문", "둘째 답", "셋째 질문"]


# --- 8~9: 이벤트 순서와 직렬화 ---


def test_three_tool_calls_run_serially_with_paired_start_done_events(mocker):
    active = {"n": 0, "max": 0}
    lock = threading.Lock()

    def slow_price(context, ticker, market):
        with lock:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        time.sleep(0.03)
        with lock:
            active["n"] -= 1
        return {"available": True, "t": ticker}

    mocker.patch("auto_stock.chat_agent.tools.tool_get_price_data", side_effect=slow_price)
    calls = [_price_call(f"c{i}", f"00000{i}") for i in range(3)]
    session = _session([AIMessage(content="", tool_calls=calls), AIMessage(content="끝")])
    events = []

    run_turn_deep(session, _context(), "q", on_event=events.append)

    assert active["max"] == 1
    kinds = [e["type"] for e in events]
    assert kinds == ["thinking", "tool_start", "tool_done", "tool_start", "tool_done", "tool_start", "tool_done", "thinking"]
    starts = [e["arguments"]["ticker"] for e in events if e["type"] == "tool_start"]
    dones = [e["arguments"]["ticker"] for e in events if e["type"] == "tool_done"]
    assert starts == dones


# --- 10~11: 노출 도구(fail-closed)와 내장 기능 ---


def test_model_sees_exactly_the_registry_tools_with_strict_schemas():
    session = _session([AIMessage(content="ok")])

    run_turn_deep(session, _context(), "q")

    names, kwargs = session.model.bound[-1]
    assert sorted(names) == sorted(REGISTRY_NAMES)
    assert kwargs.get("strict") is True


def test_verify_exposure_passes_for_current_registry():
    verify_exposure()


def test_verify_exposure_fails_closed_when_a_registry_tool_is_not_visible(monkeypatch):
    monkeypatch.setattr(deep_loop, "_expected_tool_names", lambda: {*REGISTRY_NAMES, "ghost_tool"})

    with pytest.raises(ExposureError):
        verify_exposure()


# --- 12: 세션 호출 상한(롤백과 무관하게 누적) ---


def test_session_counter_accumulates_across_rolled_back_turns_and_caps(price_tool):
    looping = lambda msgs: AIMessage(content="", tool_calls=[_price_call(f"c{len(msgs)}")])
    session = _session([looping], session_call_limit=MAIN_AGENT_RUN_LIMIT * 2)

    assert run_turn_deep(session, _context(), "q1") == TOOL_LIMIT_FALLBACK
    assert run_turn_deep(session, _context(), "q2") == TOOL_LIMIT_FALLBACK
    assert session.model_calls == MAIN_AGENT_RUN_LIMIT * 2
    assert run_turn_deep(session, _context(), "q3") == SESSION_LIMIT_FALLBACK


# --- 입력 토큰 가드: 서버 체인이 길어지면 압축 후 새 체인 ---


def test_history_is_compacted_when_last_input_tokens_exceed_limit(price_tool):
    big = AIMessage(content="답1", usage_metadata={"input_tokens": 999_999, "output_tokens": 1, "total_tokens": 1_000_000})
    session = _session(
        [AIMessage(content="", tool_calls=[_price_call()]), big, AIMessage(content="답2")],
        input_token_limit=100_000,
    )

    run_turn_deep(session, _context(), "질문1")
    run_turn_deep(session, _context(), "질문2")

    second_turn = session.model.inputs[2]
    assert not any(isinstance(m, ToolMessage) for m in second_turn)
    assert [m.text for m in second_turn if isinstance(m, (HumanMessage, AIMessage))] == ["질문1", "답1", "질문2"]


# --- 트레이싱: 대화 전체가 외부로 나가지 않게 ---


def test_tracing_env_without_explicit_allow_is_refused(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("CHAT_AGENT_ALLOW_TRACING", raising=False)

    with pytest.raises(TracingNotAllowedError):
        check_tracing_allowed()


def test_tracing_env_with_explicit_allow_passes(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("CHAT_AGENT_ALLOW_TRACING", "1")

    check_tracing_allowed()


def test_no_tracing_env_passes(monkeypatch):
    for key in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
        monkeypatch.delenv(key, raising=False)

    check_tracing_allowed()


# --- 코드 리뷰 반영(2026-10-11) ---


def test_repeated_invalid_tool_calls_still_hit_the_per_turn_limit():
    """after_model 훅은 역순으로 실행된다 — 깨진 호출 피드백이 jump_to로 모델에 바로 되돌아가면
    ModelCallLimit의 카운트 증가(after_model)를 건너뛰어 턴 상한이 무력화됐었다(리뷰 HIGH)."""
    bad = lambda msgs: AIMessage(
        content="",
        invalid_tool_calls=[{"name": "get_price_data", "args": "{x", "id": f"c{len(msgs)}", "error": "parse", "type": "invalid_tool_call"}],
    )
    session = _session([bad], session_call_limit=50)

    text = run_turn_deep(session, _context(), "q")

    assert text == TOOL_LIMIT_FALLBACK
    assert len(session.model.inputs) == MAIN_AGENT_RUN_LIMIT


def test_tracing_v2_variant_is_also_refused(monkeypatch):
    for key in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "true")
    monkeypatch.delenv("CHAT_AGENT_ALLOW_TRACING", raising=False)

    with pytest.raises(TracingNotAllowedError):
        check_tracing_allowed()
