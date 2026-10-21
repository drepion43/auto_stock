import pytest

from auto_stock.chat_agent import tools as chat_tools
from auto_stock.evaluation.faults import (
    INJECTED_ERROR,
    FaultSpec,
    InjectedToolError,
    fault_patchers,
    injection_patchers,
    injection_payload,
    make_faulty,
    new_marker,
    shift_iso_dates,
)
from auto_stock.evaluation.tool_patching import applied


def _price_tool(context, ticker, market):
    return {"available": True, "latest": {"date": "2026-10-20", "close": 100.0}}


def test_unavailable_fault_returns_unavailable_dict():
    faulty = make_faulty(FaultSpec("get_price_data", "unavailable"), _price_tool)
    assert faulty(None, ticker="005930", market="KOSPI") == {"available": False, "error": INJECTED_ERROR}


def test_exception_fault_raises():
    faulty = make_faulty(FaultSpec("get_price_data", "exception"), _price_tool)
    with pytest.raises(InjectedToolError):
        faulty(None, ticker="005930", market="KOSPI")


def test_empty_fault_uses_tool_specific_empty_field():
    faulty = make_faulty(FaultSpec("get_price_data", "empty"), _price_tool)
    assert faulty(None, ticker="005930", market="KOSPI") == {"available": True, "latest": None}
    news = make_faulty(FaultSpec("analyze_news_sentiment", "empty"), _price_tool)
    assert news(None, ticker="NVDA", market="NASDAQ") == {"available": True, "analysis": None}


def test_stale_fault_shifts_dates_without_mutating_original():
    original = {"available": True, "latest": {"date": "2026-10-20", "close": 100.0}}
    faulty = make_faulty(FaultSpec("get_price_data", "stale", stale_days=10), lambda c, **kw: original)
    shifted = faulty(None, ticker="005930", market="KOSPI")
    assert shifted["latest"]["date"] == "2026-10-10"
    assert original["latest"]["date"] == "2026-10-20"


def test_shift_iso_dates_handles_datetimes_and_nested_lists():
    value = {"scanned_at": "2026-10-20T09:30:00", "items": [{"as_of": "2026-01-05"}, "text 2026-10-20"]}
    shifted = shift_iso_dates(value, 5)
    assert shifted["scanned_at"] == "2026-10-15T09:30:00"
    assert shifted["items"][0]["as_of"] == "2025-12-31"
    assert shifted["items"][1] == "text 2026-10-20"  # only whole-value dates are shifted


def test_fault_patchers_replace_registry_function_and_wrap_current_one():
    with applied(fault_patchers(FaultSpec("get_price_data", "unavailable"))):
        assert chat_tools.tool_get_price_data(None, ticker="005930", market="KOSPI")["available"] is False
    assert chat_tools.tool_get_price_data is not None


def test_unknown_tool_in_fault_spec_is_rejected():
    with pytest.raises(ValueError):
        fault_patchers(FaultSpec("place_order", "unavailable"))


def test_injection_payload_embeds_unique_marker():
    first, second = new_marker(), new_marker()
    assert first != second
    assert first in injection_payload(first)


def test_injection_patchers_feed_marker_into_disclosure_and_news_tools():
    marker = new_marker()
    with applied(injection_patchers(marker)):
        disclosure = chat_tools.tool_analyze_disclosures(None, ticker="005930", market="KOSPI")
        news = chat_tools.tool_analyze_news_sentiment(None, ticker="005930", market="KOSPI")
    assert marker in disclosure["analysis"]["key_event"]
    assert marker in news["analysis"]["key_headline"]
