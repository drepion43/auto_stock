"""recommendation-synthesis-plan.md §1/§7 확정 — RecommendationCoordinator는
ScanCoordinator와 동일한 "stale 체크 + in-progress 추적 + 백그라운드 스레드" 패턴을
별도 클래스로 구현한다(scan_market의 계약을 안 건드린다는 §2 원칙과 일관). sleep 기반
타이밍은 쓰지 않고 threading.Event로 결정적으로 제어한다(test_scan_coordinator.py와
동일 원칙)."""

from datetime import date, timedelta
from threading import Event

import pytest

from auto_stock.chat_agent.credentials import STOCK_ANALYST_LLM_CALL_ESTIMATE
from auto_stock.data.cache import OHLCVCache
from auto_stock.data.scan_cache import MarketScanCache
from auto_stock.ml_predictor.models import MLPrediction
from auto_stock.orchestrator.deep_scan import TickerJudgment
from auto_stock.orchestrator.recommendation_coordinator import RecommendationCoordinator
from auto_stock.orchestrator.shortlist import ShortlistEntry
from auto_stock.recommendation_synthesis.credentials import (
    RECOMMENDATION_SYNTHESIS_LLM_CALL_ESTIMATE,
    RECOMMENDATION_SYNTHESIS_RECURSION_LIMIT,
)
from auto_stock.risk_sizing.models import AccountState
from auto_stock.rule_engine.models import Candidate


@pytest.fixture
def cache(tmp_path):
    return OHLCVCache(tmp_path / "cache.duckdb")


@pytest.fixture
def scan_cache(tmp_path):
    return MarketScanCache(tmp_path / "scan.duckdb")


@pytest.fixture
def account():
    return AccountState(equity=10_000_000.0, held_tickers=frozenset(), total_exposure_pct=0.0)


def _coordinator(cache, scan_cache, account, **overrides):
    defaults = dict(
        ml_models={"KRX": None, "NASDAQ": None}, llm_client=None, news_client=None,
        sentiment_client=None, account=account, agent_model="gpt-5.6-luna",
        stale_after=timedelta(hours=24), universe_size=200,
    )
    defaults.update(overrides)
    return RecommendationCoordinator(cache=cache, scan_cache=scan_cache, **defaults)


def _entry(ticker, market="KRX", rule_candidate=None, probability_up=None):
    ml_prediction = (
        MLPrediction(ticker=ticker, market=market, date=date(2026, 1, 2), probability_up=probability_up, top_features=[])
        if probability_up is not None else None
    )
    return ShortlistEntry(ticker=ticker, market=market, rule_candidate=rule_candidate, ml_prediction=ml_prediction)


def _judgment(ticker, market="KRX"):
    return TickerJudgment(
        ticker=ticker, market=market, rule_candidate=None, ml_prediction=None,
        stock_analyst_result={"available": True, "judgment": {"summary": "ok"}},
    )


def _recommendation(ticker, rank=1):
    return {"ticker": ticker, "market": "KRX", "action": "BUY", "rank": rank, "summary": "근거"}


def test_ensure_fresh_noop_when_not_stale(mocker, cache, scan_cache, account):
    scan_cache.put_recommendations("KRX", [])
    coordinator = _coordinator(cache, scan_cache, account, stale_after=timedelta(hours=999))
    mock_build_shortlist = mocker.patch("auto_stock.orchestrator.recommendation_coordinator.build_shortlist")

    thread = coordinator.ensure_fresh("KRX")

    assert thread is None
    mock_build_shortlist.assert_not_called()


def test_ensure_fresh_runs_full_pipeline_and_persists_result(mocker, cache, scan_cache, account):
    mocker.patch("auto_stock.orchestrator.recommendation_coordinator._fresh_universe", return_value=["005930"])
    mocker.patch(
        "auto_stock.orchestrator.recommendation_coordinator.build_shortlist",
        return_value=([_entry("005930", probability_up=0.7)], []),
    )
    mocker.patch(
        "auto_stock.orchestrator.recommendation_coordinator.run_deep_scan",
        return_value=[_judgment("005930")],
    )
    mocker.patch(
        "auto_stock.orchestrator.recommendation_coordinator.run_recommendation_synthesis",
        return_value={"available": True, "recommendations": [_recommendation("005930")], "provenance": "x"},
    )
    coordinator = _coordinator(cache, scan_cache, account)

    thread = coordinator.ensure_fresh("KRX")
    assert thread is not None
    thread.join(timeout=5)

    scanned_at, recommendations = scan_cache.get_latest_recommendations("KRX")
    assert scanned_at is not None
    assert [r["ticker"] for r in recommendations] == ["005930"]


def test_shortlist_is_capped_and_ranked_before_deep_scan(mocker, cache, scan_cache, account):
    """숏리스트 캡(기본 20)보다 많으면 상위만 ②단계로 넘겨야 한다 — 규칙엔진 후보
    존재 + |probability_up - 0.5| 합산 점수 기준(recommendation-synthesis-plan.md §2)."""
    low_score = _entry("000660", probability_up=0.51)  # 점수 거의 0
    high_score = _entry("005930", rule_candidate=Candidate(ticker="005930", market="KRX", action="BUY", reasons=["RSI"]), probability_up=0.9)
    mocker.patch("auto_stock.orchestrator.recommendation_coordinator._fresh_universe", return_value=["000660", "005930"])
    mocker.patch(
        "auto_stock.orchestrator.recommendation_coordinator.build_shortlist",
        return_value=([low_score, high_score], []),
    )
    mock_deep_scan = mocker.patch(
        "auto_stock.orchestrator.recommendation_coordinator.run_deep_scan", return_value=[]
    )
    mocker.patch(
        "auto_stock.orchestrator.recommendation_coordinator.run_recommendation_synthesis",
        return_value={"available": True, "recommendations": [], "provenance": "x"},
    )
    coordinator = _coordinator(cache, scan_cache, account, shortlist_cap=1)

    thread = coordinator.ensure_fresh("KRX")
    thread.join(timeout=5)

    passed_shortlist = mock_deep_scan.call_args.kwargs["shortlist"]
    assert [e.ticker for e in passed_shortlist] == ["005930"]  # 더 높은 점수만 남음


def test_budget_sized_for_worst_case_and_shared_across_stages(mocker, cache, scan_cache, account):
    mocker.patch("auto_stock.orchestrator.recommendation_coordinator._fresh_universe", return_value=["005930"])
    mocker.patch(
        "auto_stock.orchestrator.recommendation_coordinator.build_shortlist",
        return_value=([_entry("005930", probability_up=0.7)], []),
    )
    mock_deep_scan = mocker.patch(
        "auto_stock.orchestrator.recommendation_coordinator.run_deep_scan", return_value=[]
    )
    mock_synthesis = mocker.patch(
        "auto_stock.orchestrator.recommendation_coordinator.run_recommendation_synthesis",
        return_value={"available": True, "recommendations": [], "provenance": "x"},
    )
    coordinator = _coordinator(cache, scan_cache, account)

    thread = coordinator.ensure_fresh("KRX")
    thread.join(timeout=5)

    deep_scan_budget = mock_deep_scan.call_args.kwargs["budget"]
    synthesis_budget = mock_synthesis.call_args.kwargs["budget"]
    assert deep_scan_budget is synthesis_budget  # §7 — 이중 집계 방지를 위한 공유
    expected = 1 * STOCK_ANALYST_LLM_CALL_ESTIMATE + RECOMMENDATION_SYNTHESIS_LLM_CALL_ESTIMATE + (
        RECOMMENDATION_SYNTHESIS_RECURSION_LIMIT * STOCK_ANALYST_LLM_CALL_ESTIMATE
    )
    assert deep_scan_budget.max_llm_calls == expected


def test_does_not_spawn_second_thread_while_in_progress(mocker, cache, scan_cache, account):
    started, release = Event(), Event()

    def fake_run_deep_scan(*args, **kwargs):
        started.set()
        release.wait(timeout=5)
        return []

    mocker.patch("auto_stock.orchestrator.recommendation_coordinator._fresh_universe", return_value=[])
    mocker.patch(
        "auto_stock.orchestrator.recommendation_coordinator.build_shortlist", return_value=([], [])
    )
    mocker.patch(
        "auto_stock.orchestrator.recommendation_coordinator.run_deep_scan", side_effect=fake_run_deep_scan
    )
    mocker.patch(
        "auto_stock.orchestrator.recommendation_coordinator.run_recommendation_synthesis",
        return_value={"available": True, "recommendations": [], "provenance": "x"},
    )
    coordinator = _coordinator(cache, scan_cache, account)

    t1 = coordinator.ensure_fresh("KRX")
    assert started.wait(timeout=5)
    assert coordinator.is_in_progress("KRX") is True

    t2 = coordinator.ensure_fresh("KRX")
    assert t2 is None

    release.set()
    t1.join(timeout=5)
    assert coordinator.is_in_progress("KRX") is False


def test_background_failure_is_swallowed_and_leaves_market_retriable(mocker, cache, scan_cache, account, capsys):
    mocker.patch("auto_stock.orchestrator.recommendation_coordinator._fresh_universe", return_value=[])
    mocker.patch(
        "auto_stock.orchestrator.recommendation_coordinator.build_shortlist",
        side_effect=RuntimeError("네트워크 오류"),
    )
    coordinator = _coordinator(cache, scan_cache, account)

    thread = coordinator.ensure_fresh("KRX")
    thread.join(timeout=5)

    assert coordinator.is_in_progress("KRX") is False
    assert "KRX" in capsys.readouterr().err

    scanned_at, _ = scan_cache.get_latest_recommendations("KRX")
    assert scanned_at is None  # 실패해서 캐시 안 적재 -> 여전히 stale -> 재시도 가능

    mocker.patch(
        "auto_stock.orchestrator.recommendation_coordinator.build_shortlist", return_value=([], [])
    )
    mocker.patch("auto_stock.orchestrator.recommendation_coordinator.run_deep_scan", return_value=[])
    mocker.patch(
        "auto_stock.orchestrator.recommendation_coordinator.run_recommendation_synthesis",
        return_value={"available": True, "recommendations": [], "provenance": "x"},
    )
    retry_thread = coordinator.ensure_fresh("KRX")
    assert retry_thread is not None
    retry_thread.join(timeout=5)


def test_run_uses_thread_local_cursor_connections(mocker, cache, scan_cache, account):
    mocker.patch("auto_stock.orchestrator.recommendation_coordinator._fresh_universe", return_value=[])
    mocker.patch(
        "auto_stock.orchestrator.recommendation_coordinator.build_shortlist", return_value=([], [])
    )
    mocker.patch("auto_stock.orchestrator.recommendation_coordinator.run_deep_scan", return_value=[])
    mocker.patch(
        "auto_stock.orchestrator.recommendation_coordinator.run_recommendation_synthesis",
        return_value={"available": True, "recommendations": [], "provenance": "x"},
    )
    cache_cursor_spy = mocker.spy(cache, "cursor")
    scan_cache_cursor_spy = mocker.spy(scan_cache, "cursor")
    coordinator = _coordinator(cache, scan_cache, account)

    thread = coordinator.ensure_fresh("KRX")
    thread.join(timeout=5)

    cache_cursor_spy.assert_called_once()
    scan_cache_cursor_spy.assert_called_once()
