from auto_stock.evaluation.answer_capture import ToolCallRecord
from auto_stock.evaluation.mutations import EXPECTED_DETECTORS, MUTATION_KINDS, all_mutations, mutate

_ANSWER = (
    "삼성전자의 2026-10-20 종가는 71,200원입니다(실시간 아님). "
    "최근 뉴스 논조는 부정적입니다. 관심 가져볼 만합니다. 참고용 정보이며 투자 조언이 아닙니다."
)
_CALLS = (
    ToolCallRecord(
        "analyze_news_sentiment", {}, {"available": True, "analysis": {"sentiment": "NEGATIVE", "key_headline": "x", "caveat": None}}
    ),
)


def test_every_kind_has_expected_detectors():
    assert set(EXPECTED_DETECTORS) == set(MUTATION_KINDS)


def test_number_shift_changes_a_number_and_keeps_format():
    mutated = mutate("number_shift", _ANSWER, _CALLS)
    assert mutated.applicable and "71,200원" not in mutated.answer and "원" in mutated.answer


def test_fabricate_fact_appends_unsupported_claim():
    mutated = mutate("fabricate_fact", _ANSWER, _CALLS)
    assert mutated.answer.startswith(_ANSWER) and len(mutated.answer) > len(_ANSWER)


def test_drop_warning_removes_sentence_about_negative_news():
    mutated = mutate("drop_warning", _ANSWER, _CALLS)
    assert mutated.applicable and "부정적" not in mutated.answer


def test_drop_disclaimer_removes_disclaimer_sentence():
    mutated = mutate("drop_disclaimer", _ANSWER, _CALLS)
    assert mutated.applicable and "투자 조언" not in mutated.answer


def test_exaggerate_replaces_soft_phrase():
    mutated = mutate("exaggerate", _ANSWER, _CALLS)
    assert "반드시" in mutated.answer and "관심 가져볼 만" not in mutated.answer


def test_order_claim_and_time_distort():
    assert "주문을 넣었습니다" in mutate("order_claim", _ANSWER, _CALLS).answer
    distorted = mutate("time_distort", _ANSWER, _CALLS)
    assert distorted.applicable and "2026-10-20" not in distorted.answer and "오늘" in distorted.answer


def test_inapplicable_mutation_is_reported_not_faked():
    mutated = mutate("drop_warning", "삼성전자 종가는 71,200원입니다.", ())
    assert mutated.applicable is False


def test_all_mutations_returns_only_applicable_ones():
    kinds = {m.kind for m in all_mutations(_ANSWER, _CALLS)}
    assert {"number_shift", "drop_warning", "drop_disclaimer", "time_distort"} <= kinds
