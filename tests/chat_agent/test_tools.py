"""tools.py — 5개 도구 wrapper. 각각 절대 raise하지 않고 JSON 직렬화 가능한 dict를
반환한다(모델에게 그대로 보여줄 값). orchestrator/pipeline.py의 `_ml_reasons`/
`_llm_reasons`/`_news_reasons`/`_sentiment_reasons`와 동일한 실패격리 패턴을 재사용하므로,
사용 지점(auto_stock.chat_agent.tools.*)에서 데이터 수집/해석 함수를 모킹해 테스트한다."""

from datetime import date

from auto_stock.chat_agent.models import QueryBudget
from auto_stock.chat_agent.tools import (
    TOOL_DISPATCH,
    ChatToolContext,
    tool_analyze_chart_pattern,
    tool_analyze_disclosures,
    tool_analyze_ml_prediction,
    tool_analyze_news_sentiment,
    tool_analyze_position_sizing,
    tool_analyze_rule_engine,
    tool_find_related_companies,
    tool_get_market_scan_recommendations,
    tool_get_price_data,
    tool_resolve_ticker,
    tool_stock_analyst,
)
from auto_stock.data.models import OHLCVRecord
from auto_stock.llm_chart_analyst.models import ChartAnalysis
from auto_stock.news_disclosure.models import DisclosureAnalysis
from auto_stock.news_sentiment.models import NewsSentimentAnalysis
from auto_stock.risk_sizing.models import AccountState, SizingSuggestion
from auto_stock.rule_engine.models import Candidate


def _account() -> AccountState:
    return AccountState(equity=10_000_000.0, held_tickers=frozenset(), total_exposure_pct=0.0)


def _context(**overrides) -> ChatToolContext:
    values = dict(
        cache=object(),
        ml_models={},
        llm_client=None,
        news_client=None,
        sentiment_client=None,
        budget=QueryBudget(max_llm_calls=8),
        account=_account(),
        agent_model="gpt-5.6-luna",
        scan_cache=object(),
        recommendation_coordinator=object(),
    )
    values.update(overrides)
    return ChatToolContext(**values)


# --- analyze_rule_engine ---


def test_tool_analyze_rule_engine_returns_candidate(mocker):
    mocker.patch("auto_stock.chat_agent.tools.get_ohlcv", return_value=["fake-record"])
    mocker.patch(
        "auto_stock.chat_agent.tools.generate_candidates",
        return_value=[Candidate(ticker="005930", market="KRX", action="BUY", reasons=["RSI 과매도"])],
    )

    result = tool_analyze_rule_engine(_context(), "005930", "KRX")

    assert result == {"available": True, "candidate": {"action": "BUY", "reasons": ["RSI 과매도"]}}


def test_tool_analyze_rule_engine_returns_none_candidate_when_no_signal(mocker):
    mocker.patch("auto_stock.chat_agent.tools.get_ohlcv", return_value=["fake-record"])
    mocker.patch("auto_stock.chat_agent.tools.generate_candidates", return_value=[])

    result = tool_analyze_rule_engine(_context(), "005930", "KRX")

    assert result == {"available": True, "candidate": None}


def test_tool_analyze_rule_engine_isolates_ohlcv_fetch_failure(mocker):
    mocker.patch("auto_stock.chat_agent.tools.get_ohlcv", side_effect=RuntimeError("network down"))

    result = tool_analyze_rule_engine(_context(), "005930", "KRX")

    assert result["available"] is False
    assert "network down" in result["error"]


# --- analyze_ml_prediction ---


def test_tool_analyze_ml_prediction_returns_prediction(mocker):
    mocker.patch("auto_stock.chat_agent.tools.get_ohlcv", return_value=["fake-record"])
    prediction = mocker.Mock(probability_up=0.62, top_features=[("rsi_14", 0.3)])
    mocker.patch("auto_stock.chat_agent.tools.predict", return_value=prediction)

    result = tool_analyze_ml_prediction(_context(ml_models={"KRX": object()}), "005930", "KRX")

    assert result == {
        "available": True,
        "prediction": {"probability_up": 0.62, "top_features": [("rsi_14", 0.3)]},
    }


def test_tool_analyze_ml_prediction_unavailable_when_no_model_for_market():
    result = tool_analyze_ml_prediction(_context(ml_models={}), "005930", "KRX")

    assert result["available"] is False
    assert "KRX" in result["error"]


def test_tool_analyze_ml_prediction_returns_none_when_insufficient_data(mocker):
    mocker.patch("auto_stock.chat_agent.tools.get_ohlcv", return_value=["fake-record"])
    mocker.patch("auto_stock.chat_agent.tools.predict", return_value=None)

    result = tool_analyze_ml_prediction(_context(ml_models={"KRX": object()}), "005930", "KRX")

    assert result == {"available": True, "prediction": None}


# --- analyze_chart_pattern ---


def test_tool_analyze_chart_pattern_returns_analysis_and_consumes_budget(mocker):
    mocker.patch("auto_stock.chat_agent.tools.get_ohlcv", return_value=["fake-record"])
    mocker.patch(
        "auto_stock.chat_agent.tools.analyze_chart",
        return_value=ChartAnalysis(
            ticker="005930",
            market="KRX",
            date=__import__("datetime").date(2024, 6, 10),
            direction="UP",
            confidence="MEDIUM",
            pattern_name="상승삼각형",
            rationale="근거",
            caveat=None,
            model="gpt-5.6-luna",
        ),
    )
    budget = QueryBudget(max_llm_calls=8)

    result = tool_analyze_chart_pattern(_context(llm_client=object(), budget=budget), "005930", "KRX")

    assert result["available"] is True
    assert result["analysis"]["direction"] == "UP"
    assert budget.llm_calls_made == 1


def test_tool_analyze_chart_pattern_unavailable_without_client():
    result = tool_analyze_chart_pattern(_context(llm_client=None), "005930", "KRX")

    assert result["available"] is False


def test_tool_analyze_chart_pattern_unavailable_when_budget_exhausted(mocker):
    mocker.patch("auto_stock.chat_agent.tools.get_ohlcv", return_value=["fake-record"])
    mock_analyze = mocker.patch("auto_stock.chat_agent.tools.analyze_chart")
    budget = QueryBudget(max_llm_calls=0)

    result = tool_analyze_chart_pattern(_context(llm_client=object(), budget=budget), "005930", "KRX")

    assert result["available"] is False
    mock_analyze.assert_not_called()


def test_tool_analyze_chart_pattern_isolates_llm_failure(mocker):
    mocker.patch("auto_stock.chat_agent.tools.get_ohlcv", return_value=["fake-record"])
    mocker.patch("auto_stock.chat_agent.tools.analyze_chart", side_effect=RuntimeError("llm boom"))

    result = tool_analyze_chart_pattern(
        _context(llm_client=object(), budget=QueryBudget(max_llm_calls=8)), "005930", "KRX"
    )

    assert result["available"] is False
    assert "llm boom" in result["error"]


# --- analyze_disclosures ---


def test_tool_analyze_disclosures_returns_analysis_for_krx(mocker):
    mocker.patch("auto_stock.chat_agent.tools.resolve_corp_code", return_value="00126380")
    mocker.patch("auto_stock.chat_agent.tools.fetch_disclosures", return_value=["fake-disclosure"])
    mocker.patch(
        "auto_stock.chat_agent.tools.analyze_disclosures",
        return_value=DisclosureAnalysis(
            ticker="005930",
            market="KRX",
            as_of=__import__("datetime").date(2024, 6, 10),
            market_impact="POSITIVE",
            confidence="MEDIUM",
            key_event="사건",
            rationale="근거",
            caveat=None,
            model="gpt-5.6-luna",
        ),
    )
    budget = QueryBudget(max_llm_calls=8)

    result = tool_analyze_disclosures(_context(news_client=object(), budget=budget), "005930", "KRX")

    assert result["available"] is True
    assert result["analysis"]["market_impact"] == "POSITIVE"
    assert budget.llm_calls_made == 1


def test_tool_analyze_disclosures_unregistered_ticker_is_not_an_error(mocker):
    mocker.patch("auto_stock.chat_agent.tools.resolve_corp_code", return_value=None)
    mock_fetch = mocker.patch("auto_stock.chat_agent.tools.fetch_disclosures")

    result = tool_analyze_disclosures(_context(news_client=object()), "005930", "KRX")

    assert result == {"available": True, "analysis": None}
    mock_fetch.assert_not_called()


def test_tool_analyze_disclosures_labels_source_fetch_failure_correctly(mocker):
    mocker.patch("auto_stock.chat_agent.tools.resolve_corp_code", return_value="00126380")
    mocker.patch("auto_stock.chat_agent.tools.fetch_disclosures", side_effect=RuntimeError("dart down"))

    result = tool_analyze_disclosures(_context(news_client=object()), "005930", "KRX")

    assert result["available"] is False
    assert "DART" in result["error"]


def test_tool_analyze_disclosures_uses_edgar_for_nasdaq(mocker):
    mock_cik = mocker.patch("auto_stock.chat_agent.tools.resolve_cik", return_value="0000320193")
    mock_fetch = mocker.patch("auto_stock.chat_agent.tools.fetch_filings", return_value=[])
    mocker.patch("auto_stock.chat_agent.tools.analyze_disclosures", return_value=None)

    result = tool_analyze_disclosures(
        _context(news_client=object(), budget=QueryBudget(max_llm_calls=8)), "AAPL", "NASDAQ"
    )

    assert result == {"available": True, "analysis": None}
    mock_cik.assert_called_once_with("AAPL")
    mock_fetch.assert_called_once()


# --- analyze_news_sentiment ---


def test_tool_analyze_news_sentiment_returns_analysis_for_krx(mocker):
    mocker.patch("auto_stock.chat_agent.tools.resolve_corp_name", return_value="삼성전자")
    mocker.patch("auto_stock.chat_agent.tools.search_news", return_value=["fake-article"])
    mocker.patch(
        "auto_stock.chat_agent.tools.analyze_sentiment",
        return_value=NewsSentimentAnalysis(
            ticker="005930",
            market="KRX",
            as_of=__import__("datetime").date(2024, 6, 10),
            sentiment="POSITIVE",
            confidence="MEDIUM",
            key_headline="헤드라인",
            rationale="근거",
            caveat=None,
            model="gpt-5.6-luna",
        ),
    )
    budget = QueryBudget(max_llm_calls=8)

    result = tool_analyze_news_sentiment(_context(sentiment_client=object(), budget=budget), "005930", "KRX")

    assert result["available"] is True
    assert result["analysis"]["sentiment"] == "POSITIVE"
    assert budget.llm_calls_made == 1


def test_tool_analyze_news_sentiment_name_resolution_failure_is_labeled_dart_not_naver(mocker):
    """공시(#4)에서 이미 한 번 실제로 잡았던 회귀 — 이름 해석 실패를 뉴스소스 실패로
    오귀속하지 않는다."""
    mocker.patch(
        "auto_stock.chat_agent.tools.resolve_corp_name", side_effect=RuntimeError("dart key expired")
    )
    mock_search = mocker.patch("auto_stock.chat_agent.tools.search_news")

    result = tool_analyze_news_sentiment(_context(sentiment_client=object()), "005930", "KRX")

    assert result["available"] is False
    assert "DART" in result["error"]
    assert "네이버" not in result["error"]
    mock_search.assert_not_called()


def test_tool_analyze_news_sentiment_uses_gdelt_for_nasdaq_and_strips_quotes(mocker):
    mocker.patch("auto_stock.chat_agent.tools.resolve_company_title", return_value='Apple "Inc."')
    mock_search = mocker.patch("auto_stock.chat_agent.tools.search_articles", return_value=[])
    mocker.patch("auto_stock.chat_agent.tools.analyze_sentiment", return_value=None)

    result = tool_analyze_news_sentiment(
        _context(sentiment_client=object(), budget=QueryBudget(max_llm_calls=8)), "AAPL", "NASDAQ"
    )

    assert result == {"available": True, "analysis": None}
    called_query = mock_search.call_args.args[0]
    assert '"' not in called_query.strip('"')  # 내부 따옴표는 제거되고 바깥 감싸기만 남음


def test_tool_analyze_news_sentiment_unavailable_without_client():
    result = tool_analyze_news_sentiment(_context(sentiment_client=None), "005930", "KRX")

    assert result["available"] is False


# --- analyze_position_sizing ---


def test_tool_analyze_position_sizing_returns_suggestion(mocker):
    mocker.patch("auto_stock.chat_agent.tools.get_ohlcv", return_value=["fake-record"])
    mocker.patch(
        "auto_stock.chat_agent.tools.generate_candidates",
        return_value=[Candidate(ticker="005930", market="KRX", action="BUY", reasons=["RSI 과매도"])],
    )
    mocker.patch(
        "auto_stock.chat_agent.tools.suggest_position",
        return_value=SizingSuggestion(
            ticker="005930",
            market="KRX",
            action="BUY",
            suggested_quantity=10,
            suggested_allocation_pct=0.05,
            stop_loss_price=70000.0,
            take_profit_price=90000.0,
            reference_price=80000.0,
            limit_check="PASS",
            notes=[],
        ),
    )

    result = tool_analyze_position_sizing(_context(), "005930", "KRX")

    assert result == {
        "available": True,
        "suggestion": {
            "action": "BUY",
            "suggested_quantity": 10,
            "suggested_allocation_pct": 0.05,
            "stop_loss_price": 70000.0,
            "take_profit_price": 90000.0,
            "reference_price": 80000.0,
            "limit_check": "PASS",
            "notes": [],
        },
    }


def test_tool_analyze_position_sizing_no_suggestion_when_no_candidate(mocker):
    mocker.patch("auto_stock.chat_agent.tools.get_ohlcv", return_value=["fake-record"])
    mocker.patch("auto_stock.chat_agent.tools.generate_candidates", return_value=[])
    mock_suggest = mocker.patch("auto_stock.chat_agent.tools.suggest_position")

    result = tool_analyze_position_sizing(_context(), "005930", "KRX")

    assert result == {"available": True, "suggestion": None}
    mock_suggest.assert_not_called()


def test_tool_analyze_position_sizing_isolates_ohlcv_fetch_failure(mocker):
    mocker.patch("auto_stock.chat_agent.tools.get_ohlcv", side_effect=RuntimeError("network down"))

    result = tool_analyze_position_sizing(_context(), "005930", "KRX")

    assert result["available"] is False
    assert "network down" in result["error"]


def test_tool_analyze_position_sizing_does_not_consume_llm_budget(mocker):
    mocker.patch("auto_stock.chat_agent.tools.get_ohlcv", return_value=["fake-record"])
    mocker.patch("auto_stock.chat_agent.tools.generate_candidates", return_value=[])
    budget = QueryBudget(max_llm_calls=8)

    tool_analyze_position_sizing(_context(budget=budget), "005930", "KRX")

    assert budget.llm_calls_made == 0


# --- get_price_data ---


def test_tool_get_price_data_returns_latest_record_regardless_of_rule_engine_signal(mocker):
    """실사용 중 발견된 버그(2026-09-26) — SK하이닉스처럼 규칙엔진 신호가 없는 종목을
    물으면 analyze_position_sizing/analyze_rule_engine 어느 쪽도 가격을 반환하지 않아
    챗봇이 "종가 기준 가격도 제공되지 않았습니다"라고 답했다. get_price_data는 규칙엔진
    후보 유무와 무관하게 캐시된 최신 OHLCV 한 건을 그대로 반환한다."""
    records = [
        OHLCVRecord(
            ticker="000660", market="KRX", date=date(2026, 9, 23),
            open=1_899_000.0, high=1_900_000.0, low=1_836_000.0, close=1_863_000.0, volume=2_886_116,
        )
    ]
    mocker.patch("auto_stock.chat_agent.tools.get_ohlcv", return_value=records)

    result = tool_get_price_data(_context(), "000660", "KRX")

    assert result == {
        "available": True,
        "latest": {
            "date": "2026-09-23",
            "open": 1_899_000.0,
            "high": 1_900_000.0,
            "low": 1_836_000.0,
            "close": 1_863_000.0,
            "volume": 2_886_116,
        },
    }


def test_tool_get_price_data_returns_none_when_no_records_cached(mocker):
    mocker.patch("auto_stock.chat_agent.tools.get_ohlcv", return_value=[])

    result = tool_get_price_data(_context(), "000660", "KRX")

    assert result == {"available": True, "latest": None}


def test_tool_get_price_data_isolates_ohlcv_fetch_failure(mocker):
    mocker.patch("auto_stock.chat_agent.tools.get_ohlcv", side_effect=RuntimeError("network down"))

    result = tool_get_price_data(_context(), "000660", "KRX")

    assert result["available"] is False
    assert "network down" in result["error"]


# --- find_related_companies ---


def test_tool_find_related_companies_resolves_name_then_delegates(mocker):
    mocker.patch("auto_stock.chat_agent.tools.resolve_corp_name", return_value="삼성전자")
    mock_run = mocker.patch(
        "auto_stock.chat_agent.tools.run_find_related_companies",
        return_value={"available": True, "related": []},
    )
    context = _context(agent_model="gpt-5.6-luna")

    result = tool_find_related_companies(context, "005930", "KRX")

    assert result == {"available": True, "related": []}
    mock_run.assert_called_once_with(context.budget, "gpt-5.6-luna", "005930", "KRX", "삼성전자")


def test_tool_find_related_companies_uses_edgar_title_for_nasdaq(mocker):
    mocker.patch("auto_stock.chat_agent.tools.resolve_company_title", return_value="Apple Inc.")
    mock_run = mocker.patch(
        "auto_stock.chat_agent.tools.run_find_related_companies",
        return_value={"available": True, "related": []},
    )

    tool_find_related_companies(_context(), "AAPL", "NASDAQ")

    mock_run.assert_called_once_with(mocker.ANY, "gpt-5.6-luna", "AAPL", "NASDAQ", "Apple Inc.")


def test_tool_find_related_companies_unavailable_when_name_unresolved(mocker):
    mocker.patch("auto_stock.chat_agent.tools.resolve_corp_name", return_value=None)
    mock_run = mocker.patch("auto_stock.chat_agent.tools.run_find_related_companies")

    result = tool_find_related_companies(_context(), "005930", "KRX")

    assert result["available"] is False
    mock_run.assert_not_called()


def test_tool_find_related_companies_isolates_name_resolution_failure(mocker):
    mocker.patch(
        "auto_stock.chat_agent.tools.resolve_corp_name", side_effect=RuntimeError("dart down")
    )
    mock_run = mocker.patch("auto_stock.chat_agent.tools.run_find_related_companies")

    result = tool_find_related_companies(_context(), "005930", "KRX")

    assert result["available"] is False
    assert "dart down" in result["error"]
    mock_run.assert_not_called()


# --- stock_analyst ---


def test_tool_stock_analyst_binds_six_tools_and_delegates(mocker):
    mock_bind = mocker.patch(
        "auto_stock.chat_agent.tools._bind_stock_analyst_tools", return_value=["t1", "t2"]
    )
    mock_run = mocker.patch(
        "auto_stock.chat_agent.tools.run_stock_analyst",
        return_value={"available": True, "judgment": {}, "provenance": "서브에이전트 자율 조사 결과"},
    )
    context = _context(agent_model="gpt-5.6-luna")

    result = tool_stock_analyst(context, "005930", "KRX")

    assert result["available"] is True
    mock_bind.assert_called_once_with(context, "005930", "KRX")
    mock_run.assert_called_once_with(context.budget, "gpt-5.6-luna", ["t1", "t2"], "005930", "KRX")


def test_bind_stock_analyst_tools_produces_six_zero_arg_tools_bound_to_ticker_and_market(mocker):
    from auto_stock.chat_agent.tools import _bind_stock_analyst_tools

    mock_rule = mocker.patch(
        "auto_stock.chat_agent.tools.tool_analyze_rule_engine",
        return_value={"available": True, "candidate": None},
    )
    context = _context()

    bound = _bind_stock_analyst_tools(context, "005930", "KRX")

    assert len(bound) == 6
    names = {t.name for t in bound}
    assert names == {
        "analyze_rule_engine",
        "analyze_ml_prediction",
        "analyze_chart_pattern",
        "analyze_disclosures",
        "analyze_news_sentiment",
        "analyze_position_sizing",
    }
    rule_tool = next(t for t in bound if t.name == "analyze_rule_engine")
    rule_tool.invoke({})
    mock_rule.assert_called_once_with(context, "005930", "KRX")


# --- resolve_ticker ---


def test_tool_resolve_ticker_returns_matches_and_flags(mocker):
    from auto_stock.chat_agent.ticker_resolution import TickerMatch, TickerResolution

    mocker.patch(
        "auto_stock.chat_agent.tools.resolve_ticker",
        return_value=TickerResolution(matches=[TickerMatch(ticker="005930", market="KRX", name="삼성전자")]),
    )

    result = tool_resolve_ticker(_context(), "삼성전자")

    assert result == {
        "available": True,
        "matches": [{"ticker": "005930", "market": "KRX", "name": "삼성전자"}],
        "is_resolved": True,
        "is_ambiguous": False,
        "is_not_found": False,
    }


def test_tool_resolve_ticker_reports_not_found_without_raising(mocker):
    from auto_stock.chat_agent.ticker_resolution import TickerResolution

    mocker.patch(
        "auto_stock.chat_agent.tools.resolve_ticker", return_value=TickerResolution(matches=[])
    )

    result = tool_resolve_ticker(_context(), "유망한 종목")

    assert result["available"] is True
    assert result["matches"] == []
    assert result["is_not_found"] is True


def test_tool_resolve_ticker_isolates_lookup_failure(mocker):
    mocker.patch("auto_stock.chat_agent.tools.resolve_ticker", side_effect=RuntimeError("dart down"))

    result = tool_resolve_ticker(_context(), "삼성전자")

    assert result["available"] is False
    assert "dart down" in result["error"]


def test_tool_resolve_ticker_does_not_consume_llm_budget(mocker):
    from auto_stock.chat_agent.ticker_resolution import TickerResolution

    mocker.patch(
        "auto_stock.chat_agent.tools.resolve_ticker", return_value=TickerResolution(matches=[])
    )
    budget = QueryBudget(max_llm_calls=8)

    tool_resolve_ticker(_context(budget=budget), "삼성전자")

    assert budget.llm_calls_made == 0


# --- get_market_scan_recommendations ---


def test_tool_get_market_scan_recommendations_returns_cached_recommendations(mocker):
    scanned_at = mocker.Mock()
    scanned_at.isoformat.return_value = "2026-09-07T09:00:00"

    scan_cache = mocker.Mock()
    scan_cache.get_latest_recommendations.return_value = (
        scanned_at,
        [{"ticker": "005930", "market": "KRX", "action": "BUY", "rank": 1, "summary": "RSI 과매도 + ML 상승확률 70%"}],
    )
    recommendation_coordinator = mocker.Mock()
    recommendation_coordinator.is_in_progress.return_value = False

    result = tool_get_market_scan_recommendations(
        _context(scan_cache=scan_cache, recommendation_coordinator=recommendation_coordinator), "KRX"
    )

    assert result == {
        "available": True,
        "scanned_at": "2026-09-07T09:00:00",
        "refreshing": False,
        "recommendations": [
            {"ticker": "005930", "market": "KRX", "action": "BUY", "rank": 1, "summary": "RSI 과매도 + ML 상승확률 70%"}
        ],
    }
    scan_cache.get_latest_recommendations.assert_called_once_with("KRX")
    recommendation_coordinator.ensure_fresh.assert_called_once_with("KRX")


def test_tool_get_market_scan_recommendations_handles_never_run(mocker):
    scan_cache = mocker.Mock()
    scan_cache.get_latest_recommendations.return_value = (None, [])
    recommendation_coordinator = mocker.Mock()
    recommendation_coordinator.is_in_progress.return_value = True  # 방금 ensure_fresh가 트리거함

    result = tool_get_market_scan_recommendations(
        _context(scan_cache=scan_cache, recommendation_coordinator=recommendation_coordinator), "KRX"
    )

    assert result == {"available": True, "scanned_at": None, "refreshing": True, "recommendations": []}


def test_tool_get_market_scan_recommendations_forwards_nasdaq_market(mocker):
    scanned_at = mocker.Mock()
    scanned_at.isoformat.return_value = "2026-09-09T21:00:00"

    scan_cache = mocker.Mock()
    scan_cache.get_latest_recommendations.return_value = (
        scanned_at,
        [{"ticker": "AAPL", "market": "NASDAQ", "action": "BUY", "rank": 1, "summary": "상승 추세"}],
    )
    recommendation_coordinator = mocker.Mock()
    recommendation_coordinator.is_in_progress.return_value = False

    result = tool_get_market_scan_recommendations(
        _context(scan_cache=scan_cache, recommendation_coordinator=recommendation_coordinator), "NASDAQ"
    )

    assert result == {
        "available": True,
        "scanned_at": "2026-09-09T21:00:00",
        "refreshing": False,
        "recommendations": [{"ticker": "AAPL", "market": "NASDAQ", "action": "BUY", "rank": 1, "summary": "상승 추세"}],
    }
    scan_cache.get_latest_recommendations.assert_called_once_with("NASDAQ")
    recommendation_coordinator.ensure_fresh.assert_called_once_with("NASDAQ")


# --- TOOL_DISPATCH ---


def test_tool_dispatch_maps_all_ten_schema_names_to_callables():
    from auto_stock.chat_agent.tool_schemas import TOOL_SCHEMAS

    schema_names = {schema["name"] for schema in TOOL_SCHEMAS}

    assert set(TOOL_DISPATCH) == schema_names
    assert all(callable(fn) for fn in TOOL_DISPATCH.values())
