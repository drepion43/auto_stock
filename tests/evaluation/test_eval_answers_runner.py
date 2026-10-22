"""`scripts/eval_answers.py` 러너의 오프라인 테스트 — 생성 루프는 가짜 세션·가짜 run_turn_deep으로 돌린다."""

import importlib.util
import json
import sys
import types
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from auto_stock.chat_agent.models import QueryBudget
from auto_stock.evaluation.answer_capture import CapturedTurn, ToolCallRecord
from auto_stock.evaluation.budget import BudgetLimits, CostLedger
from auto_stock.evaluation.golden import cases_by_id

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
_TODAY = date(2026, 10, 21)


@pytest.fixture(scope="module")
def runner():
    # chat_cli는 import 시점에 stdin/stdout을 reconfigure한다 — 테스트에서는 생성 리소스를 쓰지 않으므로 스텁.
    sys.modules.setdefault("chat_cli", types.ModuleType("chat_cli"))
    spec = importlib.util.spec_from_file_location("eval_answers_under_test", _SCRIPTS / "eval_answers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeSession:
    def __init__(self, model=None) -> None:
        self.messages: list = []
        self.model_calls = 0


def _fake_run_turn_deep(session, context, question, on_event=None):
    session.model_calls += 2
    arguments = {"ticker": "005930", "market": "KOSPI"}
    result = {"available": True, "latest": {"date": "2026-10-20", "close": 71200.0}}
    on_event({"type": "tool_start", "name": "get_price_data", "arguments": arguments})
    on_event({"type": "tool_done", "name": "get_price_data", "arguments": arguments, "result": result})
    return "실시간이 아닌 2026-10-20 종가 기준 71,200원입니다."


@pytest.fixture
def offline_state(runner, monkeypatch):
    monkeypatch.setattr(runner, "run_turn_deep", _fake_run_turn_deep)
    monkeypatch.setattr(runner, "DeepChatSession", _FakeSession)
    monkeypatch.setattr(runner, "make_chat_model", lambda config: None)
    monkeypatch.setattr(
        runner.base_eval, "_context", lambda shared, coordinator, loop: SimpleNamespace(budget=QueryBudget(max_llm_calls=50))
    )

    def _state(limits=None, recorder=None, store=None):
        return {
            "shared": {"agent_config": None}, "ledger": CostLedger(limits or BudgetLimits()),
            "cap": runner.base_eval._CallCap(300), "store": store, "recorder": recorder, "recording_enabled": True,
            "agent_model": "gpt-5.6-luna", "fixture_label": None,
        }

    return _state


def _args(**overrides):
    defaults = {"queries": None, "sets": "regression,fault", "repeats": 1, "record": None, "replay": None, "max_llm_calls": 300}
    return SimpleNamespace(**{**defaults, **overrides})


def test_select_cases_by_ids_and_sets(runner):
    assert [c.query_id for c in runner.select_cases(_args(queries="Q1,F1"))] == ["Q1", "F1"]
    assert {c.set_name for c in runner.select_cases(_args(sets="fault"))} == {"fault"}
    with pytest.raises(SystemExit):
        runner.select_cases(_args(queries="Q99"))


def test_parse_metrics_rejects_unknown(runner):
    assert runner.parse_metrics("M1,M5") == frozenset({"M1", "M5"})
    with pytest.raises(SystemExit):
        runner.parse_metrics("M9")


def test_estimate_scales_with_turns_and_metrics(runner):
    cases = [cases_by_id()["Q1"]]
    deterministic = runner.estimate(cases, 1, frozenset({"M1"}), "gpt-5.6-luna", "gpt-5.6-luna")
    judged = runner.estimate(cases, 2, frozenset({"M2"}), "gpt-5.6-luna", "gpt-5.6-luna")
    assert deterministic["judge_usd"] == 0 and deterministic["turns"] == 1
    assert judged["turns"] == 2 and judged["judge_calls"] == 6


def _converse(runner, state, query_id):
    turns: list = []
    runner.run_conversation(state, cases_by_id()[query_id], 1, turns)
    return turns


def test_run_conversation_captures_tool_results_and_charges_generation(runner, offline_state):
    state = offline_state()
    turns = _converse(runner, state, "Q12")
    assert [t.case_id for t in turns] == ["Q12#t0#r1", "Q12#t1#r1"]
    assert turns[0].tool_calls[0].result["latest"]["close"] == 71200.0
    assert turns[1].history_tool_calls == turns[0].tool_calls
    assert state["ledger"].generation_usd > 0
    assert state["cap"].used == 4


def test_fault_case_records_injected_fault(runner, offline_state):
    turns = _converse(runner, offline_state(), "F1")
    assert turns[0].injected_fault == {"tool": "get_price_data", "kind": "unavailable", "stale_days": None}


def test_injection_case_gets_marker(runner, offline_state):
    assert _converse(runner, offline_state(), "Q10")[0].injection_marker.startswith("INJ-")


def test_generation_budget_stops_before_running_turn(runner, offline_state):
    state = offline_state(limits=BudgetLimits(total_usd=0.0001, judge_usd=0.0001, judge_calls=1))
    with pytest.raises(runner.BudgetExceeded):
        _converse(runner, state, "Q1")


def test_turns_paid_before_mid_conversation_stop_are_kept(runner, offline_state):
    # 첫 턴 사전 검사는 통과하고 두 번째 턴 사전 검사에서 멈추는 상한
    limit = runner.max_turn_cost("gpt-5.6-luna") + 0.004
    state = offline_state(limits=BudgetLimits(total_usd=limit, judge_usd=limit, judge_calls=1))
    turns: list = []
    with pytest.raises(runner.BudgetExceeded):
        runner.run_conversation(state, cases_by_id()["Q12"], 1, turns)
    assert [t.case_id for t in turns] == ["Q12#t0#r1"]


def test_fault_and_injection_cases_are_not_recorded(runner, offline_state):
    from auto_stock.evaluation.replay import FixtureRecorder

    recorder = FixtureRecorder()
    state = offline_state(recorder=recorder)
    _converse(runner, state, "F1")
    _converse(runner, state, "Q10")
    assert recorder.entries == {}
    _converse(runner, state, "Q1")
    assert len(recorder.entries) == 1


def test_replay_cursor_resets_for_each_conversation(runner, offline_state):
    from auto_stock.evaluation.replay import ReplayStore, call_key

    key = call_key("get_price_data", {"ticker": "005930", "market": "KOSPI"})
    store = ReplayStore({key: [{"n": 1}, {"n": 2}]})
    store.lookup("get_price_data", {"ticker": "005930", "market": "KOSPI"})
    state = offline_state(store=store)
    _converse(runner, state, "Q1")
    assert store.lookup("get_price_data", {"ticker": "005930", "market": "KOSPI"}) == {"n": 1}


def test_generate_writes_partial_results_and_skips_rest_after_budget(runner, offline_state, monkeypatch, tmp_path):
    limits = BudgetLimits(total_usd=0.0001, judge_usd=0.0001, judge_calls=1)
    monkeypatch.setattr(runner, "_generation_state", lambda args, ledger: offline_state(limits=limits))
    monkeypatch.setattr(runner, "OUT_DIR", tmp_path)
    meta = {"run_id": "test", "git_commit": "abc"}
    captured = runner.generate(_args(), [cases_by_id()["Q1"], cases_by_id()["Q3"]], CostLedger(limits), meta)
    saved = json.loads((tmp_path / "answers_test.json").read_text(encoding="utf-8"))
    assert captured == []
    assert len(saved["meta"]["generation_skipped"]) == 2


def _captured(query_id, turn_index, answer, calls=()):
    return CapturedTurn(
        case_id=f"{query_id}#t{turn_index}#r1", query_id=query_id, turn_index=turn_index, repeat=1, question="q",
        answer=answer, tool_calls=calls, history_tool_calls=(), fallback=False, injected_fault=None,
        replay_fixture=None, injection_marker=None, usage={},
    )


def test_score_all_adds_consistency_to_last_turn_of_multi_turn_conversation(runner):
    price = ToolCallRecord("get_price_data", {}, {"available": True, "latest": {"date": "2026-10-20", "close": 71200.0}})
    captured = [_captured("Q12", 0, "종가는 71,200원입니다.", (price,)), _captured("Q12", 1, "종가는 72,000원입니다.")]
    scores = runner.score_all(captured, judge=None, metrics=frozenset({"M1", "M5"}), context_limit=12_000, today=_TODAY)
    assert "CONSISTENCY" not in {r.metric for r in scores[0].results}
    consistency = next(r for r in scores[1].results if r.metric == "CONSISTENCY")
    assert consistency.passed is False


def test_run_mutations_uses_only_normal_turns(runner):
    good = _captured("Q11", 0, "주문을 실행할 수 없습니다. 참고용 정보입니다.")
    scores = runner.score_all([good], judge=None, metrics=frozenset({"M4"}), context_limit=12_000, today=_TODAY)
    rows = runner.run_mutations([good], scores, judge=None, limit=5, today=_TODAY)
    order = next(row for row in rows if row.kind == "order_claim")
    assert order.applicable and order.detected_by == ("M4",)
    # judge가 없어 M2만으로 잡아야 하는 변이는 "미탐"이 아니라 건너뜀
    assert next(row for row in rows if row.kind == "fabricate_fact").skipped is True


def _meta_args(**overrides):
    defaults = {"judge_model": "gpt-5.6-luna", "replay": None, "answers": None, "repeats": 2}
    return SimpleNamespace(**{**defaults, **overrides})


def test_build_meta_records_fixture_hash_and_expected_counts(runner, tmp_path):
    fixture = tmp_path / "f.json"
    fixture.write_text(json.dumps({"meta": {"reference_date": "2026-09-01"}, "entries": {}}), encoding="utf-8")
    meta = runner.build_meta(_meta_args(replay=fixture), [cases_by_id()["Q12"]], frozenset({"M1"}), "gpt-5.6-luna")
    assert meta["replay_fixture_sha256"] == runner.fixture_sha256(fixture)
    assert meta["expected"] == {"Q12": [2, 2]}


def test_rescoring_inherits_replay_identity_from_answers(runner, tmp_path):
    answers = tmp_path / "answers.json"
    source_meta = {"replay_fixture": "f.json", "replay_fixture_sha256": "abc", "expected": {"Q1": [1, 3]},
                   "reference_date": "2026-09-01"}
    answers.write_text(json.dumps({"meta": source_meta, "cases": []}), encoding="utf-8")
    meta = runner.build_meta(_meta_args(answers=answers), [], frozenset({"M1"}), "gpt-5.6-luna")
    assert meta["replay_fixture_sha256"] == "abc" and meta["expected"] == {"Q1": [1, 3]}


def test_live_run_cannot_be_saved_as_baseline(runner, tmp_path):
    path = tmp_path / "baseline.json"
    assert runner.save_baseline(path, {"replay_fixture_sha256": None}, []) is False
    assert not path.exists()
    assert runner.save_baseline(path, {"replay_fixture_sha256": "abc"}, []) is True


def test_reference_date_comes_from_replay_fixture(runner, tmp_path):
    fixture = tmp_path / "f.json"
    fixture.write_text(json.dumps({"meta": {"reference_date": "2026-09-01"}, "entries": {}}), encoding="utf-8")
    assert runner.reference_date(SimpleNamespace(replay=fixture, answers=None)) == "2026-09-01"
