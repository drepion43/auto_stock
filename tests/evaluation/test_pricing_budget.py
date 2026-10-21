import pytest

from auto_stock.evaluation.budget import (
    BudgetExceeded,
    BudgetLimits,
    CostLedger,
    JudgeBudgetExceeded,
)
from auto_stock.evaluation.pricing import cost_usd, price_for


def test_known_model_prices():
    assert price_for("gpt-5.6-terra", env={}) == (2.00, 12.00)
    assert price_for("gpt-5.6-luna", env={}) == (0.20, 1.20)


def test_env_override_wins():
    env = {"EVAL_PRICE_GPT_5_6_TERRA_IN": "3.0", "EVAL_PRICE_GPT_5_6_TERRA_OUT": "15"}
    assert price_for("gpt-5.6-terra", env=env) == (3.0, 15.0)


def test_unknown_model_without_override_fails_closed():
    with pytest.raises(KeyError):
        price_for("gpt-unknown", env={})


def test_cost_usd_per_million_tokens():
    assert cost_usd("gpt-5.6-terra", 1_000_000, 100_000, env={}) == pytest.approx(2.0 + 1.2)


def test_limits_from_env_and_defaults():
    assert BudgetLimits.from_env({}) == BudgetLimits(total_usd=8.0, judge_usd=6.0, judge_calls=400)
    env = {"EVAL_MAX_COST_USD": "2", "EVAL_JUDGE_MAX_COST_USD": "1", "EVAL_JUDGE_MAX_CALLS": "10"}
    assert BudgetLimits.from_env(env) == BudgetLimits(total_usd=2.0, judge_usd=1.0, judge_calls=10)


def test_judge_reservation_blocks_before_exceeding_judge_cap():
    ledger = CostLedger(BudgetLimits(total_usd=10.0, judge_usd=1.0, judge_calls=100))
    ledger.check_judge(0.6)
    ledger.add_judge(0.6)
    with pytest.raises(JudgeBudgetExceeded):
        ledger.check_judge(0.5)
    assert ledger.judge_usd == pytest.approx(0.6)


def test_judge_call_cap():
    ledger = CostLedger(BudgetLimits(total_usd=10.0, judge_usd=10.0, judge_calls=1))
    ledger.add_judge(0.01)
    with pytest.raises(JudgeBudgetExceeded):
        ledger.check_judge(0.01)


def test_total_cap_counts_generation_cost():
    ledger = CostLedger(BudgetLimits(total_usd=1.0, judge_usd=1.0, judge_calls=100))
    ledger.add_generation(0.9)
    with pytest.raises(JudgeBudgetExceeded):
        ledger.check_judge(0.2)
    with pytest.raises(BudgetExceeded):
        ledger.check_generation(0.2)


def test_snapshot_reports_spend():
    ledger = CostLedger(BudgetLimits())
    ledger.add_generation(0.5)
    ledger.add_judge(0.25)
    snap = ledger.snapshot()
    assert snap["total_usd"] == pytest.approx(0.75)
    assert snap["judge_calls"] == 1
    assert snap["limits"]["total_usd"] == 8.0
