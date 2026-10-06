"""deep_scan.py TDD — run_deep_scan은 숏리스트 종목마다 기존 run_stock_analyst(agent
자율 판단)를 ThreadPoolExecutor로 동시 호출한다(recommendation-synthesis-plan.md §3).
`run_stock_analyst`는 모킹해 래퍼 로직(병렬 호출 배선, 결과 순서 보존, budget 공유,
실패격리)만 검증한다 — 실제 agent 판단 품질은 stock_analyst 자신의 테스트 몫이다."""

from datetime import date

from auto_stock.chat_agent.models import QueryBudget
from auto_stock.ml_predictor.models import MLPrediction
from auto_stock.orchestrator.deep_scan import TickerJudgment, run_deep_scan
from auto_stock.orchestrator.shortlist import ShortlistEntry
from auto_stock.rule_engine.models import Candidate


def _entry(ticker, market="KRX"):
    return ShortlistEntry(
        ticker=ticker, market=market,
        rule_candidate=Candidate(ticker=ticker, market=market, action="BUY", reasons=["RSI 과매도"]),
        ml_prediction=MLPrediction(
            ticker=ticker, market=market, date=date(2026, 1, 2),
            probability_up=0.7, top_features=[],
        ),
    )


def _bind_tools(ticker, market):
    return [f"tool-for-{ticker}-{market}"]


def test_calls_run_stock_analyst_with_bound_tools_per_ticker(mocker):
    mock_run = mocker.patch(
        "auto_stock.orchestrator.deep_scan.run_stock_analyst",
        return_value={"available": True, "judgment": {"summary": "ok"}},
    )
    budget = QueryBudget(max_llm_calls=100)

    run_deep_scan(
        shortlist=[_entry("005930")], budget=budget, agent_model="gpt-5.6-luna",
        bind_tools=_bind_tools,
    )

    mock_run.assert_called_once_with(budget, "gpt-5.6-luna", ["tool-for-005930-KRX"], "005930", "KRX")


def test_result_preserves_input_order(mocker):
    def fake_run_stock_analyst(budget, model, tools, ticker, market):
        return {"available": True, "judgment": {"summary": f"judged {ticker}"}}

    mocker.patch(
        "auto_stock.orchestrator.deep_scan.run_stock_analyst", side_effect=fake_run_stock_analyst
    )
    shortlist = [_entry("005930"), _entry("000660"), _entry("035720")]

    judgments = run_deep_scan(
        shortlist=shortlist, budget=QueryBudget(max_llm_calls=100), agent_model="gpt-5.6-luna",
        bind_tools=_bind_tools,
    )

    assert [j.ticker for j in judgments] == ["005930", "000660", "035720"]
    assert [j.stock_analyst_result["judgment"]["summary"] for j in judgments] == [
        "judged 005930", "judged 000660", "judged 035720",
    ]


def test_carries_over_rule_candidate_and_ml_prediction_from_shortlist_entry(mocker):
    mocker.patch(
        "auto_stock.orchestrator.deep_scan.run_stock_analyst",
        return_value={"available": True, "judgment": {"summary": "ok"}},
    )
    entry = _entry("005930")

    judgments = run_deep_scan(
        shortlist=[entry], budget=QueryBudget(max_llm_calls=100), agent_model="gpt-5.6-luna",
        bind_tools=_bind_tools,
    )

    assert judgments == [
        TickerJudgment(
            ticker="005930", market="KRX",
            rule_candidate=entry.rule_candidate, ml_prediction=entry.ml_prediction,
            stock_analyst_result={"available": True, "judgment": {"summary": "ok"}},
        )
    ]


def test_one_ticker_failure_does_not_block_the_rest(mocker):
    """stock_analyst.py 자신은 절대 raise하지 않지만, bind_tools 등 이 함수 호출부의
    다른 실수로 예외가 나더라도 다른 종목까지 막으면 안 된다 — scan_market/build_shortlist와
    동일한 실패격리 원칙."""

    def flaky_bind_tools(ticker, market):
        if ticker == "000660":
            raise RuntimeError("바인딩 실패")
        return [f"tool-for-{ticker}"]

    mocker.patch(
        "auto_stock.orchestrator.deep_scan.run_stock_analyst",
        return_value={"available": True, "judgment": {"summary": "ok"}},
    )

    judgments = run_deep_scan(
        shortlist=[_entry("005930"), _entry("000660")],
        budget=QueryBudget(max_llm_calls=100), agent_model="gpt-5.6-luna",
        bind_tools=flaky_bind_tools,
    )

    assert judgments[0].stock_analyst_result["available"] is True
    assert judgments[1].ticker == "000660"
    assert judgments[1].stock_analyst_result["available"] is False
    assert "바인딩 실패" in judgments[1].stock_analyst_result["error"]


def test_shares_a_single_budget_instance_across_all_calls(mocker):
    """②단계 전체가 같은 budget을 공유해야 ③단계(재조사)까지 이어지는 예산 집계가
    정확하다(recommendation-synthesis-plan.md §7) — 종목마다 새 QueryBudget을 만들면
    안 된다."""
    seen_budgets = []

    def fake_run_stock_analyst(budget, model, tools, ticker, market):
        seen_budgets.append(budget)
        return {"available": True, "judgment": {"summary": "ok"}}

    mocker.patch(
        "auto_stock.orchestrator.deep_scan.run_stock_analyst", side_effect=fake_run_stock_analyst
    )
    budget = QueryBudget(max_llm_calls=100)

    run_deep_scan(
        shortlist=[_entry("005930"), _entry("000660")], budget=budget, agent_model="gpt-5.6-luna",
        bind_tools=_bind_tools,
    )

    assert all(b is budget for b in seen_budgets)


def test_empty_shortlist_returns_empty_list(mocker):
    mock_run = mocker.patch("auto_stock.orchestrator.deep_scan.run_stock_analyst")

    judgments = run_deep_scan(
        shortlist=[], budget=QueryBudget(max_llm_calls=100), agent_model="gpt-5.6-luna",
        bind_tools=_bind_tools,
    )

    assert judgments == []
    mock_run.assert_not_called()
