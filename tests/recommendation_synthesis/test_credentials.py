from auto_stock.recommendation_synthesis.credentials import (
    RECOMMENDATION_SYNTHESIS_LLM_CALL_ESTIMATE,
    RECOMMENDATION_SYNTHESIS_RECURSION_LIMIT,
)


def test_llm_call_estimate_covers_real_recursion_limit_worst_case():
    """chat_agent/credentials.py와 동일한 불변식 — 과소 사전차감 금지."""
    assert RECOMMENDATION_SYNTHESIS_LLM_CALL_ESTIMATE >= RECOMMENDATION_SYNTHESIS_RECURSION_LIMIT


def test_recursion_limit_is_positive_and_bounded():
    assert 0 < RECOMMENDATION_SYNTHESIS_RECURSION_LIMIT <= 20
