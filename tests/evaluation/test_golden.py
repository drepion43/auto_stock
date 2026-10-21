from auto_stock.chat_agent.tool_registry import TOOL_SPECS
from auto_stock.evaluation.golden import (
    ALL_CASES,
    FAULT_CASES,
    LLM_GENERATED_TOOLS,
    REGRESSION_CASES,
    cases_by_id,
    regression_queries,
)

_TOOL_NAMES = {spec.name for spec in TOOL_SPECS}


def test_regression_set_keeps_q1_to_q12():
    assert [case.query_id for case in REGRESSION_CASES] == [f"Q{i}" for i in range(1, 13)]


def test_every_expected_tool_exists_in_registry():
    for case in ALL_CASES:
        for turn in case.turns:
            for call in turn.expected_calls:
                assert call.name in _TOOL_NAMES, (case.query_id, call.name)


def test_llm_generated_tools_exist_in_registry():
    assert LLM_GENERATED_TOOLS <= _TOOL_NAMES


def test_expected_args_use_only_string_values_with_known_keys():
    for case in ALL_CASES:
        for turn in case.turns:
            for call in turn.expected_calls:
                assert all(isinstance(v, str) for v in call.args.values())
                assert set(call.args) <= {"ticker", "market", "query", "index"}


def test_safety_cases_are_injection_and_order_requests():
    safety_ids = {case.query_id for case in ALL_CASES if case.safety}
    assert {"Q10", "Q11"} <= safety_ids
    assert cases_by_id()["Q10"].injection is True


def test_fault_cases_target_an_expected_tool():
    assert FAULT_CASES
    for case in FAULT_CASES:
        assert case.fault is not None
        expected_names = {call.name for turn in case.turns for call in turn.expected_calls}
        assert case.fault.tool in expected_names


def test_case_ids_are_unique():
    ids = [case.query_id for case in ALL_CASES]
    assert len(ids) == len(set(ids))


def test_regression_queries_match_turn_questions():
    queries = regression_queries()
    assert queries["Q1"] == ["삼성전자 지금 얼마야?"]
    assert len(queries["Q9"]) == 10
