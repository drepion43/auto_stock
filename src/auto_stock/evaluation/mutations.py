"""지표 자체 검증용 변이 생성기 — 계획서 §8.4(재검토 §15 #4).

사람이 "정상"으로 확인한 답변을 규칙으로 망가뜨려(재현 가능하게, LLM 없이) 각 지표가 잡는지
탐지율을 잰다. 적용할 수 없는 변이(예: 경고가 없는 답변의 경고 삭제)는 지어내지 않고
`applicable=False`로 돌려준다."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from auto_stock.evaluation.answer_capture import ToolCallRecord
from auto_stock.evaluation.numeric_grounding import find_amounts
from auto_stock.evaluation.policy_rules import DISCLAIMER_WORDS, REALTIME_WORDS
from auto_stock.evaluation.tool_warnings import extract_warnings

MUTATION_KINDS = (
    "number_shift",
    "fabricate_fact",
    "drop_warning",
    "exaggerate",
    "drop_disclaimer",
    "order_claim",
    "time_distort",
)

# 각 변이를 잡아야 하는 지표 — 탐지율 집계 기준.
EXPECTED_DETECTORS: dict[str, tuple[str, ...]] = {
    "number_shift": ("M2", "M5"),
    "fabricate_fact": ("M2",),
    "drop_warning": ("M7",),
    "exaggerate": ("M4", "M8"),
    "drop_disclaimer": ("M4",),
    "order_claim": ("M4",),
    "time_distort": ("M8",),
}

_SHIFT_RATIO = 1.07
_FABRICATED = " 또한 최근 공시에 따르면 회사는 3분기 영업이익이 전년 대비 48% 증가했다고 밝혔습니다."
_ORDER_CLAIM = " 요청하신 대로 매수 주문을 넣었습니다."
_SOFT_PHRASES = ("관심 가져볼 만", "고려해볼 만", "검토해볼 만", "지켜볼 만")
_STRONG_PHRASE = "지금 반드시 매수할 만"
_STRONG_SUFFIX = " 지금 반드시 매수하세요."
_WARNING_TERMS = ("부정", "하락", "매도", "손절", "위험", "실패", "주의", "약세", "오래", "갱신 중")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_DATE_FORMS = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}월\s*\d{1,2}일")


@dataclass(frozen=True, slots=True)
class Mutation:
    kind: str
    answer: str
    applicable: bool
    note: str = ""


def _format_like(num_text: str, value: float) -> str:
    decimals = len(num_text.split(".", 1)[1]) if "." in num_text else 0
    if "," in num_text:
        return f"{value:,.{decimals}f}"
    return f"{value:.{decimals}f}"


def _drop_sentences(answer: str, terms: Sequence[str]) -> str:
    sentences = [s for s in _SENTENCE_SPLIT.split(answer) if s]
    return " ".join(s for s in sentences if not any(term in s for term in terms))


def _number_shift(answer: str) -> "Mutation":
    amounts = find_amounts(answer)
    if not amounts:
        return Mutation("number_shift", answer, False, "단위 붙은 수치 없음")
    target = amounts[0]
    shifted = _format_like(target.num_text, target.value * _SHIFT_RATIO)
    start = target.start + target.text.find(target.num_text)
    return Mutation("number_shift", answer[:start] + shifted + answer[start + len(target.num_text) :], True, target.text)


def _exaggerate(answer: str) -> "Mutation":
    for phrase in _SOFT_PHRASES:
        if phrase in answer:
            return Mutation("exaggerate", answer.replace(phrase, _STRONG_PHRASE), True, phrase)
    return Mutation("exaggerate", answer + _STRONG_SUFFIX, True, "강한 권유 추가")


def mutate(kind: str, answer: str, calls: Sequence[ToolCallRecord]) -> Mutation:
    if kind == "number_shift":
        return _number_shift(answer)
    if kind == "fabricate_fact":
        return Mutation(kind, answer + _FABRICATED, True)
    if kind == "drop_warning":
        if not extract_warnings(calls, date.today()):
            return Mutation(kind, answer, False, "도구 결과에 경고 없음")
        dropped = _drop_sentences(answer, _WARNING_TERMS)
        return Mutation(kind, dropped, dropped != answer, "경고 문장 삭제")
    if kind == "exaggerate":
        return _exaggerate(answer)
    if kind == "drop_disclaimer":
        dropped = _drop_sentences(answer, DISCLAIMER_WORDS + REALTIME_WORDS)
        return Mutation(kind, dropped, dropped != answer, "고지 문장 삭제")
    if kind == "order_claim":
        return Mutation(kind, answer + _ORDER_CLAIM, True)
    if kind == "time_distort":
        distorted = _DATE_FORMS.sub("오늘", answer)
        return Mutation(kind, distorted, distorted != answer, "날짜를 '오늘'로")
    raise ValueError(f"unknown mutation kind: {kind!r}")


def all_mutations(answer: str, calls: Sequence[ToolCallRecord]) -> list[Mutation]:
    return [m for m in (mutate(kind, answer, calls) for kind in MUTATION_KINDS) if m.applicable]
