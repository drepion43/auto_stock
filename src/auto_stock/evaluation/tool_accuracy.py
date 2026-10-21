"""M1 도구 정확도(결정론) — 계획서 §5.1.

계획서는 DeepEval `ToolCorrectnessMetric`을 쓰기로 했지만, 4.3.1은 `available_tools` 없이(LLM 미사용
모드) 만들어도 생성 시점에 OpenAI 키를 요구한다(2026-10-21 확인). 비용 0·키 불필요여야 하는 M1을
그 위에 둘 이유가 없어 직접 구현한다 — 기대 도구 호출 여부 + 티커·시장 인자 일치 + 불필요 호출 보고."""

from collections.abc import Sequence

from auto_stock.evaluation.answer_capture import ToolCallRecord
from auto_stock.evaluation.golden import ExpectedCall
from auto_stock.evaluation.results import MetricResult, not_applied


def _args_match(expected: ExpectedCall, call: ToolCallRecord) -> bool:
    return all(str(call.arguments.get(key, "")).upper() == value.upper() for key, value in expected.args.items())


def score_tool_accuracy(expected: Sequence[ExpectedCall], calls: Sequence[ToolCallRecord]) -> MetricResult:
    if not expected:
        return not_applied("M1", "기대 도구가 정의되지 않은 턴")
    missing: list[str] = []
    arg_mismatch: list[dict] = []
    for exp in expected:
        same_name = [call for call in calls if call.name == exp.name]
        if not same_name:
            missing.append(exp.name)
        elif exp.args and not any(_args_match(exp, call) for call in same_name):
            arg_mismatch.append(
                {"name": exp.name, "expected": dict(exp.args), "actual": [dict(call.arguments) for call in same_name]}
            )
    expected_names = {exp.name for exp in expected}
    extra = [call.name for call in calls if call.name not in expected_names]
    matched = len(expected) - len(missing) - len(arg_mismatch)
    passed = not missing and not arg_mismatch
    return MetricResult(
        metric="M1",
        applied=True,
        passed=passed,
        score=matched / len(expected),
        reason="ok" if passed else f"missing={missing} arg_mismatch={[m['name'] for m in arg_mismatch]}",
        details={"missing": missing, "arg_mismatch": arg_mismatch, "extra_calls": extra},
    )
