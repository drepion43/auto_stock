from datetime import date

from auto_stock.evaluation.answer_capture import (
    CapturedTurn,
    ToolCallRecord,
    ToolCallSink,
    build_context,
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
