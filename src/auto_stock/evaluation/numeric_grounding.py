"""M5 수치 근거 검사(결정론) + 멀티턴 수치 일관성 — 계획서 §5.2.

답변에서 **단위가 붙은 숫자**(원·만원·억원·조원·달러·%·배·주)만 뽑는다 — 맨숫자("3개", "2026년")는
대조하지 않는다(우연 일치·오탐 방지). 각 숫자는 4분류:
- matched: 도구 결과의 어떤 수치와 표시 정밀도(반올림 허용) 안에서 일치
- derived_ok: %인데 같은 도구 결과의 두 수치로 재계산한 변화율과 일치
- mismatch: 앞에 항목명(종가·손절가 등)이 있고 그 항목이 도구 결과에 있는데 값이 다름 → 실패 후보
- unverifiable: 위 어느 것도 아님(근거 확인 불가 — 실패로 치지 않고 M2·사람 검수로 넘김)
이 검사는 약하다(문맥 없는 숫자 비교). 위반은 "확인 필요" 신호로 쓰고 최종 판단은 M2·사람이 한다."""

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import permutations
from typing import Any

from auto_stock.evaluation.answer_capture import ToolCallRecord
from auto_stock.evaluation.results import MetricResult, not_applied

_NUMBER = r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?"
_UNIT = r"조\s*원|억\s*원|만\s*원|원|달러|%p|%|퍼센트|배|주(?![일간차])"
_AMOUNT = re.compile(rf"(?P<dollar>\$\s*)?(?P<num>{_NUMBER})\s*(?P<unit>{_UNIT})?")
_PLAIN_NUMBER = re.compile(_NUMBER)

_MULTIPLIER = {"원": 1.0, "만원": 1e4, "억원": 1e8, "조원": 1e12, "달러": 1.0, "$": 1.0, "배": 1.0, "주": 1.0}
_PERCENT_UNITS = {"%", "%p", "퍼센트"}
CURRENCY_UNITS = {"원", "만원", "억원", "조원", "달러", "$"}

_RELATIVE_TOLERANCE = 0.005  # 0.5%
_LABEL_WINDOW = 15  # 숫자 앞 몇 글자 안에서 항목명을 찾는지
_MAX_DERIVED_VALUES = 60  # 변화율 재계산 시 도구 결과 1건의 수치 상한(조합 폭발 방지)

# 답변 항목명 → 도구 결과 필드명. 숫자에 가장 가까운 항목명을 쓴다.
LABEL_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("예측 수익률", ("predicted_return",)),
    ("예상 수익률", ("predicted_return",)),
    ("초과 수익률", ("predicted_return",)),
    ("초과수익률", ("predicted_return",)),
    ("전일 대비", ("change_pct",)),
    ("전일대비", ("change_pct",)),
    ("변화율", ("change_pct",)),
    ("등락률", ("change_pct",)),
    ("손절가", ("stop_loss_price",)),
    ("손절", ("stop_loss_price",)),
    ("익절가", ("take_profit_price",)),
    ("익절", ("take_profit_price",)),
    ("목표가", ("take_profit_price",)),
    ("기준가", ("reference_price",)),
    ("현재가", ("close",)),
    ("종가", ("close",)),
    ("주가", ("close",)),
    ("가격", ("close", "reference_price")),
    ("시가", ("open",)),
    ("고가", ("high",)),
    ("저가", ("low",)),
    ("거래량", ("volume",)),
    ("비중", ("suggested_allocation_pct", "weight_pct")),
    ("수량", ("suggested_quantity",)),
)
_LABEL_TO_FIELDS = dict(LABEL_FIELDS)
_SUBJECT_BEFORE_LABEL = re.compile(r"([가-힣A-Za-z0-9]+?)(?:의|은|는|이|가)?\s*$")
_NON_SUBJECT_WORDS = frozenset({"아까", "지금", "현재", "오늘", "어제", "최근", "그", "이", "그때", "방금", "다시"})
PRICE_LABELS = frozenset({"현재가", "종가", "주가", "가격", "기준가"})


@dataclass(frozen=True, slots=True)
class Amount:
    text: str
    start: int
    end: int
    num_text: str
    value: float
    unit: str
    label: str | None

    @property
    def is_currency(self) -> bool:
        return self.unit in CURRENCY_UNITS

    @property
    def is_percent(self) -> bool:
        return self.unit in _PERCENT_UNITS


def _decimals(num_text: str) -> int:
    return len(num_text.split(".", 1)[1]) if "." in num_text else 0


def _find_label(text: str, start: int) -> str | None:
    window = text[max(0, start - _LABEL_WINDOW) : start]
    best: tuple[int, int, str] | None = None
    for label, _fields in LABEL_FIELDS:
        position = window.rfind(label)
        if position < 0 or (label == "시가" and window[position:].startswith("시가총액")):
            continue
        # 숫자에 가장 가까운(끝 위치가 큰) 항목명, 같으면 긴 이름("손절가" > "가격"의 "가").
        rank = (position + len(label), len(label), label)
        if best is None or rank[:2] > best[:2]:
            best = rank
    return best[2] if best else None


def find_amounts(text: str) -> list[Amount]:
    amounts = []
    for match in _AMOUNT.finditer(text):
        unit = re.sub(r"\s+", "", match.group("unit") or "")
        if match.group("dollar"):
            unit = "$"
        if not unit:
            continue
        num_text = match.group("num")
        amounts.append(
            Amount(
                text=match.group(0).strip(),
                start=match.start(),
                end=match.end(),
                num_text=num_text,
                value=float(num_text.replace(",", "")),
                unit=unit,
                label=_find_label(text, match.start()),
            )
        )
    return amounts


def _candidates(amount: Amount) -> list[tuple[float, float]]:
    """(비교값, 허용오차) 후보. %는 도구가 비율(0.05)로 줄 수도 있어 두 가지를 본다."""
    step = 0.5 * 10 ** (-_decimals(amount.num_text))
    if amount.is_percent:
        return [(amount.value, step), (amount.value / 100, step / 100)]
    multiplier = _MULTIPLIER.get(amount.unit, 1.0)
    return [(amount.value * multiplier, step * multiplier)]


def _close(value: float, target: float, tolerance: float) -> bool:
    return abs(value - target) <= max(tolerance, _RELATIVE_TOLERANCE * abs(target))


def _walk(value: Any, key: str = "") -> Iterator[tuple[str, float]]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield key, float(value)
    elif isinstance(value, dict):
        for child_key, child in value.items():
            yield from _walk(child, str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child, key)
    elif isinstance(value, str):
        for number in _PLAIN_NUMBER.findall(value):
            yield "<text>", float(number.replace(",", ""))


def tool_values(calls: Sequence[ToolCallRecord]) -> list[tuple[str, float]]:
    return [pair for call in calls for pair in _walk(call.result)]


def _derived_match(amount: Amount, calls: Sequence[ToolCallRecord]) -> bool:
    if not amount.is_percent:
        return False
    target, tolerance = abs(amount.value), 0.5 * 10 ** (-_decimals(amount.num_text))
    for call in calls:
        numbers = [value for key, value in _walk(call.result) if key != "<text>"]
        if len(numbers) > _MAX_DERIVED_VALUES:
            continue
        for a, b in permutations(numbers, 2):
            if b != 0 and _close(abs((a - b) / abs(b) * 100), target, tolerance):
                return True
    return False


def _matches(amount: Amount, values: list[tuple[str, float]]) -> bool:
    # %는 방향을 말로 표현한다("3.2% 하락" ↔ change_pct=-3.2) — 부호 없이 크기만 비교한다.
    if amount.is_percent:
        return any(_close(abs(cand), abs(value), tol) for cand, tol in _candidates(amount) for _key, value in values)
    return any(_close(cand, value, tol) for cand, tol in _candidates(amount) for _key, value in values)


def _classify(amount: Amount, values: list[tuple[str, float]], calls: Sequence[ToolCallRecord]) -> str:
    if _matches(amount, values):
        return "matched"
    if _derived_match(amount, calls):
        return "derived_ok"
    if amount.label is not None:
        fields = _LABEL_TO_FIELDS[amount.label]
        if any(key in fields for key, _value in values):
            return "mismatch"
    return "unverifiable"


def check_numeric_grounding(answer: str, calls: Sequence[ToolCallRecord]) -> MetricResult:
    amounts = find_amounts(answer)
    if not amounts:
        return not_applied("M5", "단위가 붙은 수치 없음")
    values = tool_values(calls)
    items = [
        {"text": a.text, "value": a.value, "unit": a.unit, "label": a.label, "status": _classify(a, values, calls)}
        for a in amounts
    ]
    counts = {
        status: sum(item["status"] == status for item in items)
        for status in ("matched", "derived_ok", "mismatch", "unverifiable")
    }
    passed = counts["mismatch"] == 0
    return MetricResult(
        metric="M5",
        applied=True,
        passed=passed,
        score=(counts["matched"] + counts["derived_ok"]) / len(items),
        reason="ok" if passed else f"근거와 다른 수치 {counts['mismatch']}개",
        details={"items": items, "counts": counts},
        needs_review=not passed,
    )


def _subject(text: str, amount: Amount) -> str | None:
    """항목명 바로 앞 단어(예: "삼성전자 종가"의 "삼성전자") — 시간 부사 등은 주어로 보지 않는다(휴리스틱)."""
    label_at = text.rfind(amount.label or "", max(0, amount.start - _LABEL_WINDOW), amount.start)
    if label_at < 0:
        return None
    match = _SUBJECT_BEFORE_LABEL.search(text[max(0, label_at - _LABEL_WINDOW) : label_at])
    word = match.group(1) if match else None
    return None if word is None or word in _NON_SUBJECT_WORDS else word


def cross_turn_conflicts(turns: Sequence[tuple[str, Sequence[ToolCallRecord]]]) -> MetricResult:
    """같은 대화에서 같은 대상의 같은 항목을 다시 말했는데 값이 달라졌고 그 사이 도구 재호출이 없으면 실패.
    대상은 항목명 앞 단어로 구분한다("삼성전자 종가" ≠ "SK하이닉스 종가") — 둘 다 대상이 없으면 같은 것으로 본다."""
    if len(turns) < 2:
        return not_applied("CONSISTENCY", "단일 턴")
    seen: dict[tuple[str, str | None], tuple[float, float, int]] = {}
    conflicts = []
    for index, (answer, calls) in enumerate(turns):
        for amount in find_amounts(answer):
            if amount.label is None:
                continue
            key = (amount.label, _subject(answer, amount))
            value, tolerance = _candidates(amount)[0]
            previous = seen.get(key)
            if previous is not None and not calls and not _close(value, previous[0], max(tolerance, previous[1])):
                conflicts.append(
                    {"label": amount.label, "subject": key[1], "turn": index, "value": value,
                     "previous_turn": previous[2], "previous": previous[0]}
                )
            seen[key] = (value, tolerance, index)
    violations = [f"{c['label']}@turn{c['turn']}" for c in conflicts]
    passed = not violations
    return MetricResult(
        metric="CONSISTENCY",
        applied=True,
        passed=passed,
        score=1.0 if passed else 0.0,
        reason="ok" if passed else f"턴 간 수치 불일치 {violations}",
        details={"conflicts": conflicts},
    )
