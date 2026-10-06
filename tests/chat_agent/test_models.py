"""QueryBudget — 유일하게 mutable한 dataclass(chat_agent/models.py 모듈 docstring 참고).
다른 신호원 모듈들엔 test_models.py가 없지만(순수 구조체라 다른 테스트에서 간접 검증됨),
QueryBudget은 진짜 동작(두 가지 소비 방식)이 있어 직접 테스트한다."""

from threading import Thread

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


def test_try_consume_llm_call_is_thread_safe_under_concurrent_access():
    """recommendation-synthesis-plan.md §3의 run_deep_scan이 ThreadPoolExecutor로
    여러 종목의 run_stock_analyst를 동시에 호출하면서 같은 budget 인스턴스를 공유한다
    — QueryBudget이 스레드 간 공유되는 건 이번이 처음이라, check-then-increment가
    락 없이 레이스 컨디션으로 과소비될 수 있었다. sleep 없이 join()으로만 동기화되는
    결정적 테스트(tests/data/test_cache.py의 동시성 테스트와 동일 패턴)."""
    budget = QueryBudget(max_llm_calls=20)
    results: list[bool] = []

    def worker():
        ok = budget.try_consume_llm_call()
        results.append(ok)

    threads = [Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert budget.llm_calls_made == 20
    assert sum(results) == 20
