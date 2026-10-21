import json

from auto_stock.chat_agent import tools as chat_tools
from auto_stock.chat_agent.tool_registry import TOOL_SPECS, call_isolated
from auto_stock.evaluation.replay import (
    REPLAY_MISS_REASON,
    FixtureRecorder,
    ReplayStore,
    call_key,
    replay_patchers,
)
from auto_stock.evaluation.tool_patching import applied

_PRICE_SPEC = next(spec for spec in TOOL_SPECS if spec.name == "get_price_data")


def _done(name, arguments, result):
    return {"type": "tool_done", "name": name, "arguments": arguments, "result": result}


def test_call_key_ignores_argument_order():
    assert call_key("get_price_data", {"ticker": "005930", "market": "KOSPI"}) == call_key(
        "get_price_data", {"market": "KOSPI", "ticker": "005930"}
    )


def test_recorder_keeps_only_registry_tool_done_events():
    recorder = FixtureRecorder()
    recorder.on_event({"type": "tool_start", "name": "get_price_data", "arguments": {}})
    recorder.on_event(_done("write_file", {"path": "x"}, {"available": False}))
    recorder.on_event(_done("get_price_data", {"ticker": "005930", "market": "KOSPI"}, {"available": True}))
    assert list(recorder.entries) == [call_key("get_price_data", {"ticker": "005930", "market": "KOSPI"})]


def test_record_then_replay_round_trip(tmp_path):
    recorder = FixtureRecorder()
    result = {"available": True, "latest": {"date": "2026-10-20", "close": 71200.0}}
    recorder.on_event(_done("get_price_data", {"ticker": "005930", "market": "KOSPI"}, result))
    path = tmp_path / "fixture.json"
    recorder.save(path, meta={"git_commit": "abc"})

    store = ReplayStore.load(path)
    with applied(replay_patchers(store)):
        replayed = call_isolated(_PRICE_SPEC, None, {"ticker": "005930", "market": "KOSPI"})
    assert replayed == result
    assert store.misses == []
    assert json.loads(path.read_text(encoding="utf-8"))["meta"]["git_commit"] == "abc"


def test_replay_miss_returns_unavailable_and_is_counted():
    store = ReplayStore({})
    with applied(replay_patchers(store)):
        replayed = chat_tools.tool_get_price_data(None, ticker="000660", market="KOSPI")
    assert replayed["available"] is False
    assert replayed["reason"] == REPLAY_MISS_REASON
    assert store.misses == [call_key("get_price_data", {"ticker": "000660", "market": "KOSPI"})]


def test_repeated_calls_replay_in_recorded_order_then_repeat_last():
    key = call_key("get_market_scan_recommendations", {"market": "KRX"})
    store = ReplayStore({key: [{"refreshing": True}, {"refreshing": False}]})
    assert store.lookup("get_market_scan_recommendations", {"market": "KRX"}) == {"refreshing": True}
    assert store.lookup("get_market_scan_recommendations", {"market": "KRX"}) == {"refreshing": False}
    assert store.lookup("get_market_scan_recommendations", {"market": "KRX"}) == {"refreshing": False}


def test_replay_patches_every_registry_tool():
    store = ReplayStore({})
    with applied(replay_patchers(store)):
        for spec in TOOL_SPECS:
            assert getattr(chat_tools, spec.fn_name).__name__ == "_replayed"
