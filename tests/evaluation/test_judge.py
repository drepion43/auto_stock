import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from auto_stock.evaluation.budget import BudgetLimits, CostLedger, JudgeBudgetExceeded
from auto_stock.evaluation.judge import BudgetedOpenAIJudge, JudgeOutputError
from auto_stock.evaluation.pricing import cost_usd


class _Verdict(BaseModel):
    verdict: str


def _judge(client, limits=None):
    ledger = CostLedger(limits or BudgetLimits())
    return BudgetedOpenAIJudge(ledger, model_name="gpt-5.6-luna", client=client, max_output_tokens=500), ledger


def test_generate_with_schema_returns_parsed_and_charges_actual_usage(fake_client):
    judge, ledger = _judge(fake_client)
    result = judge.generate("프롬프트", schema=_Verdict)
    assert isinstance(result, _Verdict)
    assert ledger.judge_calls == 1
    assert ledger.judge_usd == pytest.approx(cost_usd("gpt-5.6-luna", 1000, 100))
    assert fake_client.responses.calls[0]["text_format"] is _Verdict


def test_generate_without_schema_returns_text(fake_client):
    judge, _ledger = _judge(fake_client)
    assert judge.generate("프롬프트") == "ok"


def test_budget_checked_before_calling_api(fake_client):
    judge, _ledger = _judge(fake_client, BudgetLimits(total_usd=1.0, judge_usd=0.000001, judge_calls=10))
    with pytest.raises(JudgeBudgetExceeded):
        judge.generate("프롬프트", schema=_Verdict)
    assert fake_client.responses.calls == []


def test_missing_parsed_output_is_an_error(fake_responses_cls):
    judge, _ledger = _judge(SimpleNamespace(responses=fake_responses_cls(fail_parse=True)))
    with pytest.raises(JudgeOutputError):
        judge.generate("프롬프트", schema=_Verdict)


def test_failed_api_call_is_still_charged_at_max_estimate():
    class _Raising:
        def parse(self, **kwargs):
            raise RuntimeError("output truncated")

    judge, ledger = _judge(SimpleNamespace(responses=_Raising()))
    with pytest.raises(RuntimeError):
        judge.generate("프롬프트", schema=_Verdict)
    assert ledger.judge_calls == 1
    assert ledger.judge_usd == pytest.approx(judge.estimate_max_cost("프롬프트"))


def test_missing_usage_is_charged_at_max_estimate():
    class _NoUsage:
        def create(self, **kwargs):
            return SimpleNamespace(output_text="ok", usage=None)

    judge, ledger = _judge(SimpleNamespace(responses=_NoUsage()))
    judge.generate("프롬프트")
    assert ledger.judge_usd == pytest.approx(judge.estimate_max_cost("프롬프트"))


def test_async_generate_delegates_to_sync(fake_client):
    judge, ledger = _judge(fake_client)
    assert isinstance(asyncio.run(judge.a_generate("p", schema=_Verdict)), _Verdict)
    assert ledger.judge_calls == 1


def test_unknown_model_fails_at_construction(fake_client):
    with pytest.raises(KeyError):
        BudgetedOpenAIJudge(CostLedger(BudgetLimits()), model_name="gpt-unknown", client=fake_client)


def test_model_name_reported(fake_client):
    judge, _ledger = _judge(fake_client)
    assert judge.get_model_name() == "gpt-5.6-luna"


def test_default_judge_model_is_luna():
    from auto_stock.evaluation.judge import DEFAULT_JUDGE_MODEL

    assert DEFAULT_JUDGE_MODEL == "gpt-5.6-luna"
