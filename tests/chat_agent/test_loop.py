"""loop.py — run_turn()은 OpenAI Responses API의 tool-calling 왕복을 오케스트레이션한다.
FakeReader(ChatAgentReader Protocol 구현)로 실제 openai SDK 없이 결정론적으로 검증한다.
previous_response_id 유무로 시스템 프롬프트를 딱 한 번만(대화 최초 호출에만) 넣는
멀티턴 설계(client.py의 stateful Responses API 특성 재사용)가 이 파일의 핵심 검증
대상이다."""

import json

from auto_stock.chat_agent.loop import run_turn
from auto_stock.chat_agent.models import ChatModelResponse, FunctionCall, QueryBudget
from auto_stock.chat_agent.tools import ChatToolContext
from auto_stock.risk_sizing.models import AccountState


class FakeReader:
    model = "gpt-5.6-luna"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create_response(self, input, tools, tool_choice, previous_response_id):
        self.calls.append(
            {
                "input": input,
                "tools": tools,
                "tool_choice": tool_choice,
                "previous_response_id": previous_response_id,
            }
        )
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _context(**overrides) -> ChatToolContext:
    values = dict(
        cache=object(),
        ml_models={},
        llm_client=None,
        news_client=None,
        sentiment_client=None,
        budget=QueryBudget(max_llm_calls=8),
        account=AccountState(equity=1.0, held_tickers=frozenset(), total_exposure_pct=0.0),
        agent_model="gpt-5.6-luna",
        scan_cache=object(),
        scan_coordinator=object(),
    )
    values.update(overrides)
    return ChatToolContext(**values)


def test_run_turn_returns_text_immediately_when_no_tool_calls():
    reader = FakeReader(
        [ChatModelResponse(output_text="안녕하세요", function_calls=[], response_id="resp_1")]
    )

    text, response_id = run_turn(reader, _context(), "삼성전자 어때?", previous_response_id=None)

    assert text == "안녕하세요"
    assert response_id == "resp_1"
    assert len(reader.calls) == 1


def test_run_turn_injects_system_prompt_only_on_brand_new_conversation():
    reader = FakeReader([ChatModelResponse(output_text="ok", function_calls=[], response_id="resp_1")])

    run_turn(reader, _context(), "삼성전자 어때?", previous_response_id=None)

    first_input = reader.calls[0]["input"]
    assert isinstance(first_input, list)
    assert first_input[0]["role"] == "system"
    assert first_input[1] == {"role": "user", "content": "삼성전자 어때?"}


def test_run_turn_does_not_repeat_system_prompt_on_followup_turn():
    reader = FakeReader([ChatModelResponse(output_text="ok", function_calls=[], response_id="resp_2")])

    run_turn(reader, _context(), "그건 왜 그래?", previous_response_id="resp_1")

    assert reader.calls[0]["input"] == "그건 왜 그래?"
    assert reader.calls[0]["previous_response_id"] == "resp_1"


def test_run_turn_dispatches_tool_calls_and_feeds_output_back(mocker):
    mocker.patch(
        "auto_stock.chat_agent.loop.TOOL_DISPATCH",
        {"resolve_ticker": lambda context, query: {"available": True, "matches": []}},
    )
    reader = FakeReader(
        [
            ChatModelResponse(
                output_text="",
                function_calls=[
                    FunctionCall(
                        name="resolve_ticker",
                        arguments=json.dumps({"query": "삼성전자"}),
                        call_id="call_1",
                    )
                ],
                response_id="resp_1",
            ),
            ChatModelResponse(output_text="찾지 못했습니다", function_calls=[], response_id="resp_2"),
        ]
    )

    text, response_id = run_turn(reader, _context(), "삼성전자 어때?", previous_response_id=None)

    assert text == "찾지 못했습니다"
    assert response_id == "resp_2"
    second_input = reader.calls[1]["input"]
    assert second_input[0]["type"] == "function_call_output"
    assert second_input[0]["call_id"] == "call_1"
    output_payload = json.loads(second_input[0]["output"])
    assert output_payload == {"available": True, "matches": []}


def test_run_turn_isolates_unknown_tool_name():
    reader = FakeReader(
        [
            ChatModelResponse(
                output_text="",
                function_calls=[FunctionCall(name="nonexistent_tool", arguments="{}", call_id="call_1")],
                response_id="resp_1",
            ),
            ChatModelResponse(output_text="죄송합니다", function_calls=[], response_id="resp_2"),
        ]
    )

    text, response_id = run_turn(reader, _context(), "무엇이든", previous_response_id=None)

    assert text == "죄송합니다"
    output_payload = json.loads(reader.calls[1]["input"][0]["output"])
    assert output_payload["available"] is False


def test_run_turn_isolates_malformed_tool_arguments():
    reader = FakeReader(
        [
            ChatModelResponse(
                output_text="",
                function_calls=[FunctionCall(name="resolve_ticker", arguments="not json", call_id="call_1")],
                response_id="resp_1",
            ),
            ChatModelResponse(output_text="다시 말씀해 주세요", function_calls=[], response_id="resp_2"),
        ]
    )

    text, response_id = run_turn(reader, _context(), "무엇이든", previous_response_id=None)

    assert text == "다시 말씀해 주세요"
    output_payload = json.loads(reader.calls[1]["input"][0]["output"])
    assert output_payload["available"] is False


def test_run_turn_stops_after_max_iterations_with_fallback_message(mocker):
    """코드 리뷰 발견사항(Stage E) — 마지막 라운드의 function_call은 dispatch만 되고
    API에 결과를 제출한 적이 없으므로(한 번 더 create_response를 호출하지 않고 루프가
    끝남), 그 응답의 response_id를 다음 턴 previous_response_id로 그대로 넘기면 미해결
    함수호출이 매달린 상태를 계속 이어가게 된다. 안전하게 원래 previous_response_id로
    되돌려 이번 턴 전체를 "없었던 일"로 취급한다."""
    mocker.patch("auto_stock.chat_agent.loop.MAX_TOOL_ITERATIONS", 2)
    mocker.patch(
        "auto_stock.chat_agent.loop.TOOL_DISPATCH",
        {"resolve_ticker": lambda context, query: {"available": True, "matches": []}},
    )
    call = FunctionCall(name="resolve_ticker", arguments=json.dumps({"query": "x"}), call_id="call_1")
    reader = FakeReader(
        [
            ChatModelResponse(output_text="", function_calls=[call], response_id="resp_1"),
            ChatModelResponse(output_text="", function_calls=[call], response_id="resp_2"),
        ]
    )

    text, response_id = run_turn(reader, _context(), "무엇이든", previous_response_id="resp_0")

    assert response_id == "resp_0"
    assert text
    assert len(reader.calls) == 2


def test_run_turn_emits_thinking_before_each_create_response_call(mocker):
    """사용자 요구사항(2026-09-06) — "현재 tool 호출 중입니다"/"reasoning 중입니다" 같은
    활동 상태를 결과가 나온 "뒤"가 아니라 대기가 "시작될 때" 실시간으로 보여달라는
    요청. `thinking` 이벤트는 각 라운드에서 `reader.create_response`를 부르기 직전에
    emit된다 — 그 호출이 실제로 몇 초 걸리는 블로킹 호출이기 때문에 이 시점에 알려줘야
    의미가 있다."""
    mocker.patch(
        "auto_stock.chat_agent.loop.TOOL_DISPATCH",
        {"resolve_ticker": lambda context, query: {"available": True, "matches": []}},
    )
    reader = FakeReader(
        [
            ChatModelResponse(
                output_text="",
                function_calls=[
                    FunctionCall(name="resolve_ticker", arguments=json.dumps({"query": "삼성전자"}), call_id="call_1")
                ],
                response_id="resp_1",
            ),
            ChatModelResponse(output_text="찾지 못했습니다", function_calls=[], response_id="resp_2"),
        ]
    )
    events = []

    run_turn(reader, _context(), "삼성전자 어때?", previous_response_id=None, on_event=events.append)

    thinking_events = [e for e in events if e["type"] == "thinking"]
    assert len(thinking_events) == 2  # 라운드마다 한 번


def test_run_turn_emits_reasoning_only_when_summary_is_non_empty():
    reader = FakeReader(
        [
            ChatModelResponse(
                output_text="ok", function_calls=[], response_id="resp_1", reasoning_summary="충분히 명확하다"
            )
        ]
    )
    events = []

    run_turn(reader, _context(), "삼성전자 어때?", previous_response_id=None, on_event=events.append)

    reasoning_events = [e for e in events if e["type"] == "reasoning"]
    assert reasoning_events == [{"type": "reasoning", "text": "충분히 명확하다"}]


def test_run_turn_does_not_emit_reasoning_when_summary_is_empty():
    reader = FakeReader([ChatModelResponse(output_text="ok", function_calls=[], response_id="resp_1")])
    events = []

    run_turn(reader, _context(), "삼성전자 어때?", previous_response_id=None, on_event=events.append)

    assert not any(e["type"] == "reasoning" for e in events)


def test_run_turn_emits_tool_start_before_tool_done_with_matching_name_and_arguments(mocker):
    """tool_start는 실제 도구 실행(블로킹) 이전에, tool_done은 그 직후에 emit되어야
    "지금 이 tool을 호출 중이다"라는 실시간 상태를 표현할 수 있다."""
    mocker.patch(
        "auto_stock.chat_agent.loop.TOOL_DISPATCH",
        {"resolve_ticker": lambda context, query: {"available": True, "matches": []}},
    )
    reader = FakeReader(
        [
            ChatModelResponse(
                output_text="",
                function_calls=[
                    FunctionCall(name="resolve_ticker", arguments=json.dumps({"query": "삼성전자"}), call_id="call_1")
                ],
                response_id="resp_1",
            ),
            ChatModelResponse(output_text="찾지 못했습니다", function_calls=[], response_id="resp_2"),
        ]
    )
    events = []

    run_turn(reader, _context(), "삼성전자 어때?", previous_response_id=None, on_event=events.append)

    tool_events = [e for e in events if e["type"] in ("tool_start", "tool_done")]
    assert [e["type"] for e in tool_events] == ["tool_start", "tool_done"]
    assert tool_events[0] == {"type": "tool_start", "name": "resolve_ticker", "arguments": {"query": "삼성전자"}}
    assert tool_events[1] == {
        "type": "tool_done",
        "name": "resolve_ticker",
        "arguments": {"query": "삼성전자"},
        "result": {"available": True, "matches": []},
    }


def test_run_turn_tool_start_uses_raw_string_when_arguments_malformed():
    reader = FakeReader(
        [
            ChatModelResponse(
                output_text="",
                function_calls=[FunctionCall(name="resolve_ticker", arguments="not json", call_id="call_1")],
                response_id="resp_1",
            ),
            ChatModelResponse(output_text="다시 말씀해 주세요", function_calls=[], response_id="resp_2"),
        ]
    )
    events = []

    run_turn(reader, _context(), "무엇이든", previous_response_id=None, on_event=events.append)

    start_event = next(e for e in events if e["type"] == "tool_start")
    assert start_event["arguments"] == "not json"


def test_run_turn_works_without_on_event_callback():
    """on_event는 선택 파라미터다 — 기존 호출부(테스트 포함)를 깨지 않아야 한다."""
    reader = FakeReader([ChatModelResponse(output_text="ok", function_calls=[], response_id="resp_1")])

    text, response_id = run_turn(reader, _context(), "안녕", previous_response_id=None)

    assert text == "ok"


def test_run_turn_returns_fallback_message_and_previous_response_id_on_api_failure():
    """코드 리뷰 발견사항(Stage E) — reader.create_response가 완전히 무방비였다: 일시적
    API 오류(타임아웃/레이트리밋/연결실패 등, client.py가 ChatAgentError로 통일하는
    모든 경우)가 그대로 run_turn 밖으로 새어나가 chat_cli.py의 전체 대화 루프를
    죽였다."""
    reader = FakeReader([Exception("일시적 LLM 연결 실패")])

    text, response_id = run_turn(reader, _context(), "삼성전자 어때?", previous_response_id="resp_prev")

    assert response_id == "resp_prev"
    assert text
    assert len(reader.calls) == 1
