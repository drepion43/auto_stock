"""tool_schemas.py의 shape을 고정한다 — OpenAI Responses API 함수콜링 스키마는
`{"function": {...}}`로 감싸지 않는 평평한 구조여야 한다(Context7로 확인, chat.completions
스키마와 다름). 이 테스트가 실수로 잘못된 형태로 되돌아가는 회귀를 잡는다."""

from auto_stock.chat_agent.tool_schemas import TOOL_SCHEMAS

EXPECTED_TOOL_NAMES = {
    "resolve_ticker",
    "analyze_rule_engine",
    "analyze_ml_prediction",
    "analyze_chart_pattern",
    "analyze_disclosures",
    "analyze_news_sentiment",
    "analyze_position_sizing",
    "get_price_data",
    "find_related_companies",
    "stock_analyst",
    "get_market_scan_recommendations",
}

_TICKER_MARKET_TOOL_NAMES = EXPECTED_TOOL_NAMES - {"resolve_ticker", "get_market_scan_recommendations"}


def test_tool_schemas_cover_exactly_the_ten_analysis_capabilities():
    names = {schema["name"] for schema in TOOL_SCHEMAS}

    assert names == EXPECTED_TOOL_NAMES


def test_every_tool_schema_is_flat_function_type_not_nested():
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert "function" not in schema  # chat.completions 스타일 중첩 금지
        assert set(schema) >= {"type", "name", "description", "parameters", "strict"}
        assert schema["strict"] is True


def test_every_ticker_market_tool_schema_takes_only_ticker_and_market_params():
    for schema in TOOL_SCHEMAS:
        if schema["name"] not in _TICKER_MARKET_TOOL_NAMES:
            continue
        params = schema["parameters"]
        assert params["type"] == "object"
        assert set(params["properties"]) == {"ticker", "market"}
        assert params["required"] == ["ticker", "market"]
        assert params["additionalProperties"] is False
        assert params["properties"]["market"]["enum"] == ["KRX", "NASDAQ"]


def test_resolve_ticker_schema_takes_only_query_param():
    schema = next(s for s in TOOL_SCHEMAS if s["name"] == "resolve_ticker")
    params = schema["parameters"]

    assert params["type"] == "object"
    assert set(params["properties"]) == {"query"}
    assert params["required"] == ["query"]
    assert params["additionalProperties"] is False


def test_get_market_scan_recommendations_schema_takes_only_market_param():
    """티커를 지정하지 않는 전체 스캔형 질의 전용 도구라 ticker는 없지만, KRX/NASDAQ 배치
    스캔 캐시가 market별로 분리돼 있으므로 market은 요구한다(2026-09-09부로 NASDAQ 지원
    추가 — 이전에는 KRX로 하드코딩되어 있었다)."""
    schema = next(s for s in TOOL_SCHEMAS if s["name"] == "get_market_scan_recommendations")
    params = schema["parameters"]

    assert params["type"] == "object"
    assert set(params["properties"]) == {"market"}
    assert params["required"] == ["market"]
    assert params["additionalProperties"] is False
    assert params["properties"]["market"]["enum"] == ["KRX", "NASDAQ"]


def test_every_tool_schema_has_a_nonempty_description():
    for schema in TOOL_SCHEMAS:
        assert len(schema["description"]) > 10


def test_get_market_scan_recommendations_schema_describes_refreshing_field():
    """majestic-waddling-breeze.md "온디맨드 배치 스캔 트리거" 계획 — OS 스케줄러 대신
    ensure_fresh가 백그라운드로 스캔을 트리거하므로, 모델이 refreshing=true일 때 "갱신
    중"이라고 안내하도록 스키마 설명에 명시해야 한다."""
    schema = next(s for s in TOOL_SCHEMAS if s["name"] == "get_market_scan_recommendations")

    assert "refreshing" in schema["description"]
