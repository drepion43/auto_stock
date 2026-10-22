from datetime import date

from auto_stock.evaluation.answer_capture import (
    CapturedTurn,
    ToolCallRecord,
    ToolCallSink,
    build_context,
    main_visible,
    to_jsonable,
)


def _rec(name, result, arguments=None):
    return ToolCallRecord(name=name, arguments=arguments or {}, result=result)


def test_sink_collects_tool_done_events_in_order():
    sink = ToolCallSink()
    sink.on_event({"type": "thinking"})
    sink.on_event({"type": "tool_done", "name": "resolve_ticker", "arguments": {"query": "삼성"}, "result": {"a": 1}})
    sink.on_event({"type": "tool_done", "name": "get_price_data", "arguments": {}, "result": {"b": 2}})
    assert [r.name for r in sink.records] == ["resolve_ticker", "get_price_data"]
    assert sink.records[0].arguments == {"query": "삼성"}


def test_sink_records_stock_analyst_delegation_under_the_tool_mode_name():
    # subagent 모드의 task 위임은 tool 모드 stock_analyst와 같은 이름·인자로 기록돼야 M1·M2 비교가 공정하다
    sink = ToolCallSink()
    delegation = {"subagent_type": "stock_analyst", "description": "대상: 005930 KOSPI\n요청: 사도 돼?"}
    sink.on_event({"type": "tool_done", "name": "task", "arguments": delegation, "result": {"available": True}})
    sink.on_event({"type": "tool_done", "name": "task", "arguments": {"subagent_type": "x"}, "result": {}})
    assert [r.name for r in sink.records] == ["stock_analyst", "task"]
    assert sink.records[0].arguments == {"ticker": "005930", "market": "KOSPI", "request": "요청: 사도 돼?"}


def test_sink_keeps_subagent_tag_and_it_survives_a_round_trip():
    sink = ToolCallSink()
    event = {"type": "tool_done", "name": "analyze_rule_engine", "arguments": {}, "result": {}, "subagent": "stock_analyst"}
    sink.on_event(event)
    sink.on_event({"type": "tool_done", "name": "get_price_data", "arguments": {}, "result": {}})
    assert [r.subagent for r in sink.records] == ["stock_analyst", None]
    assert [r.name for r in main_visible(sink.records)] == ["get_price_data"]
    turn = CapturedTurn(
        case_id="Q1#t0#r1", query_id="Q1", turn_index=0, repeat=1, question="q", answer="a", tool_calls=sink.records,
        history_tool_calls=(), fallback=False, injected_fault=None, replay_fixture=None, injection_marker=None,
    )
    assert CapturedTurn.from_dict(turn.to_dict()).tool_calls == sink.records
    legacy = turn.to_dict()
    for record in legacy["tool_calls"]:
        record.pop("subagent")  # 태그 도입 전에 저장된 answers 파일
    assert [r.subagent for r in CapturedTurn.from_dict(legacy).tool_calls] == [None, None]


def test_to_jsonable_converts_unserializable_values():
    assert to_jsonable({"d": date(2026, 10, 20), "s": {1}}) == {"d": "2026-10-20", "s": "{1}"}


def test_build_context_keeps_everything_under_limit():
    calls = (_rec("get_price_data", {"latest": {"close": 100}}),)
    bundle = build_context(calls, "종가 100원", limit=1000)
    assert bundle.truncated is False
    assert "get_price_data" in bundle.text and "100" in bundle.text


def test_build_context_prefers_list_items_mentioned_in_answer():
    recs = [{"ticker": f"00{i:04d}", "name": f"종목{i}", "summary": "x" * 200} for i in range(30)]
    calls = (_rec("get_market_scan_recommendations", {"recommendations": recs}),)
    bundle = build_context(calls, "추천: 종목27 (000027)", limit=800)
    assert bundle.truncated is True
    assert "000027" in bundle.text
    assert len(bundle.text) <= 800


def test_build_context_includes_called_tool_descriptions_once():
    calls = (_rec("get_price_data", {"latest": {"close": 1}}), _rec("get_price_data", {"latest": {"close": 2}}))
    text = build_context(calls, "", limit=10_000).text
    assert text.count("[도구 설명: get_price_data]") == 1
    assert "실시간 시세가 아니라 최근 종가 기준" in text


def test_truncation_keeps_tool_descriptions():
    recs = [{"ticker": f"{i:06d}", "summary": "x" * 300} for i in range(40)]
    calls = (_rec("get_market_scan_recommendations", {"scanned_at": None, "recommendations": recs}),)
    bundle = build_context(calls, "없음", limit=3000)
    assert bundle.truncated is True
    assert "[도구 설명: get_market_scan_recommendations]" in bundle.text


def test_build_context_reports_llm_generated_tools():
    calls = (_rec("resolve_ticker", {}), _rec("stock_analyst", {"judgment": {}}))
    assert build_context(calls, "", limit=1000).llm_generated_tools == ("stock_analyst",)


def test_captured_turn_round_trips_through_dict():
    turn = CapturedTurn(
        case_id="Q1#0", query_id="Q1", turn_index=0, repeat=1, question="q", answer="a",
        tool_calls=(_rec("get_price_data", {"x": 1}, {"ticker": "005930"}),), history_tool_calls=(),
        fallback=False, injected_fault=None, replay_fixture=None, injection_marker=None,
        usage={"main_calls": 2},
    )
    assert CapturedTurn.from_dict(turn.to_dict()) == turn
