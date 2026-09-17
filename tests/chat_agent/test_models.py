"""QueryBudget — 유일하게 mutable한 dataclass(chat_agent/models.py 모듈 docstring 참고).
다른 신호원 모듈들엔 test_models.py가 없지만(순수 구조체라 다른 테스트에서 간접 검증됨),
QueryBudget은 진짜 동작(두 가지 소비 방식)이 있어 직접 테스트한다."""

from auto_stock.chat_agent.models import QueryBudget


def test_try_consume_llm_call_decrements_until_exhausted():
    budget = QueryBudget(max_llm_calls=2)

    assert budget.try_consume_llm_call() is True
    assert budget.try_consume_llm_call() is True
    assert budget.try_consume_llm_call() is False
    assert budget.llm_calls_made == 2


def test_try_consume_llm_calls_bulk_succeeds_when_enough_remaining():
    budget = QueryBudget(max_llm_calls=10)

    assert budget.try_consume_llm_calls(4) is True
    assert budget.llm_calls_made == 4


def test_try_consume_llm_calls_bulk_fails_atomically_when_not_enough_remaining():
    """부분 소비 후 실패하면 안 된다 — 전부 되거나 전혀 안 되거나."""
    budget = QueryBudget(max_llm_calls=3)

    assert budget.try_consume_llm_calls(4) is False
    assert budget.llm_calls_made == 0


def test_try_consume_llm_calls_can_be_combined_with_single_consumption():
    budget = QueryBudget(max_llm_calls=5)

    assert budget.try_consume_llm_call() is True  # 1 used
    assert budget.try_consume_llm_calls(3) is True  # 4 used
    assert budget.try_consume_llm_call() is True  # 5 used
    assert budget.try_consume_llm_call() is False
