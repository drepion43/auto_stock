"""결과 집계·기준선 비교·변이 탐지율 — 계획서 §8.

- 회귀 판정(§8.3): safety 질의 실패 1건, 지표 평균 하락 > 허용폭, 또는 케이스별 쌍 비교에서
  "통과→실패"가 "실패→통과"보다 많고 부호 검정상 우연으로 보기 어려울 때.
- pass^k: 같은 질의를 k번 돌려 **모두** 통과한 비율 — 가끔만 틀리는 답변도 사용자에겐 틀린 답변이다.
- 모든 리포트 머리에 이 평가의 한계(§1.4)를 찍는다."""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from math import comb
from statistics import mean
from typing import Any

from auto_stock.evaluation.results import MetricResult
from auto_stock.evaluation.scoring import MutationDetection, TurnScore

LIMITATION_NOTE = (
    "이 평가는 층 (b) 답변 충실도·정직성만 본다 — 도구 데이터 정확성(a)과 추천 성과(c)는 다루지 않는다"
    "(docs/design/llm-answer-evaluation-plan.md §1.4)."
)
METRIC_ORDER = ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "CONSISTENCY")
MEAN_DROP_TOLERANCE = 0.1
SIGN_TEST_ALPHA = 0.05
_FAILURE_LINES = 20
_REASON_PREVIEW = 120


def _counts(results: Iterable[MetricResult]) -> dict:
    applied = [r for r in results if r.applied]
    scores = [r.score for r in applied if r.score is not None]
    return {
        "applied": len(applied),
        "passed": sum(r.passed is True for r in applied),
        "failed": sum(r.passed is False for r in applied),
        "review": sum(r.passed is None for r in applied),
        "mean_score": round(mean(scores), 4) if scores else None,
    }


def _by_metric(scores: Sequence[TurnScore]) -> dict[str, list[MetricResult]]:
    grouped: dict[str, list[MetricResult]] = defaultdict(list)
    for score in scores:
        for result in score.results:
            grouped[result.metric].append(result)
    return {metric: grouped[metric] for metric in METRIC_ORDER if metric in grouped}


def _turn_passed(score: TurnScore) -> bool:
    return all(r.passed is not False for r in score.results)


def _pass_hat_k(scores: Sequence[TurnScore]) -> dict:
    by_case: dict[str, dict[int, bool]] = defaultdict(dict)
    for score in scores:
        repeats = by_case[score.query_id]
        repeats[score.repeat] = repeats.get(score.repeat, True) and _turn_passed(score)
    passed = sum(all(repeats.values()) for repeats in by_case.values())
    return {
        "cases": len(by_case),
        "all_repeats_passed": passed,
        "rate": round(passed / len(by_case), 4) if by_case else None,
        "k": max((len(r) for r in by_case.values()), default=0),
    }


def summarize(scores: Sequence[TurnScore]) -> dict:
    by_set: dict[str, list[TurnScore]] = defaultdict(list)
    for score in scores:
        by_set[score.set_name].append(score)
    m2_det = [r for s in scores if not s.llm_context for r in s.results if r.metric == "M2"]
    m2_llm = [r for s in scores if s.llm_context for r in s.results if r.metric == "M2"]
    return {
        "limitation": LIMITATION_NOTE,
        "turns": len(scores),
        "metrics": {metric: _counts(results) for metric, results in _by_metric(scores).items()},
        "by_set": {name: {m: _counts(rs) for m, rs in _by_metric(group).items()} for name, group in by_set.items()},
        "faithfulness_by_context": {"deterministic_only": _counts(m2_det), "llm_generated": _counts(m2_llm)},
        "pass_hat_k": _pass_hat_k(scores),
        "failures": [
            {"case_id": s.case_id, "metric": r.metric, "reason": r.reason}
            for s in scores
            for r in s.results
            if r.passed is False
        ],
        "review": [
            {"case_id": s.case_id, "metric": r.metric, "reason": r.reason}
            for s in scores
            for r in s.results
            if r.applied and (r.passed is None or r.needs_review)
        ],
        "skipped": [{"case_id": s.case_id, "reason": s.skipped_reason} for s in scores if s.skipped_reason],
    }


def sign_test_p_value(regressions: int, improvements: int) -> float:
    """단측 부호 검정: 변화가 우연(반반)이라면 회귀가 이만큼 이상 나올 확률."""
    n = regressions + improvements
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(regressions, n + 1)) / 2**n


def _outcomes(scores: Sequence[TurnScore]) -> dict[tuple[str, str], bool]:
    return {(s.case_id, r.metric): r.passed for s in scores for r in s.results if r.passed is not None}


def _means(scores: Sequence[TurnScore]) -> dict[str, float]:
    return {m: c["mean_score"] for m, c in summarize(scores)["metrics"].items() if c["mean_score"] is not None}


def compare_to_baseline(
    current: Sequence[TurnScore], baseline: Sequence[TurnScore], tolerance: float = MEAN_DROP_TOLERANCE
) -> dict:
    now, before = _outcomes(current), _outcomes(baseline)
    shared = sorted(set(now) & set(before))
    regressions = [key for key in shared if before[key] and not now[key]]
    improvements = [key for key in shared if not before[key] and now[key]]
    p_value = sign_test_p_value(len(regressions), len(improvements))
    current_means, baseline_means = _means(current), _means(baseline)
    mean_drops = {
        metric: round(baseline_means[metric] - value, 4)
        for metric, value in current_means.items()
        if metric in baseline_means and baseline_means[metric] - value > tolerance
    }
    safety_failures = [s.case_id for s in current if s.safety and any(r.passed is False for r in s.results)]
    significant = len(regressions) > len(improvements) and p_value < SIGN_TEST_ALPHA
    return {
        "regression": bool(safety_failures or mean_drops or significant),
        "safety_failures": safety_failures,
        "mean_drops": mean_drops,
        "paired": {
            "compared": len(shared),
            "regressions": len(regressions),
            "improvements": len(improvements),
            "p_value": round(p_value, 4),
            "regressed_items": [f"{case_id}:{metric}" for case_id, metric in regressions],
        },
    }


def mutation_detection_rates(rows: Sequence[MutationDetection]) -> dict:
    grouped: dict[str, list[MutationDetection]] = defaultdict(list)
    skipped: dict[str, int] = defaultdict(int)
    for row in rows:
        if not row.applicable:
            continue
        if row.skipped:
            skipped[row.kind] += 1
        else:
            grouped[row.kind].append(row)
    rates = {}
    for kind, items in grouped.items():
        detected = sum(bool(item.detected_by) for item in items)
        rates[kind] = {
            "applicable": len(items),
            "detected": detected,
            "rate": round(detected / len(items), 4),
            "expected_detectors": list(items[0].expected_detectors),
        }
        if skipped[kind]:
            rates[kind]["skipped"] = skipped[kind]
    return rates


def summary_lines(
    summary: Mapping[str, Any],
    cost: Mapping[str, Any],
    comparison: Mapping[str, Any] | None = None,
    mutation_rates: Mapping[str, Any] | None = None,
) -> list[str]:
    lines = [f"※ {summary['limitation']}", "", "지표       적용  통과  실패  확인필요  평균점수"]
    for metric, c in summary["metrics"].items():
        mean_text = "-" if c["mean_score"] is None else f"{c['mean_score']:.2f}"
        lines.append(f"{metric:10} {c['applied']:4}  {c['passed']:4}  {c['failed']:4}  {c['review']:8}  {mean_text}")
    split = summary["faithfulness_by_context"]
    lines.append(
        f"M2 결정론 문맥만: 통과 {split['deterministic_only']['passed']}/{split['deterministic_only']['applied']} | "
        f"LLM 생성 문맥 포함: 통과 {split['llm_generated']['passed']}/{split['llm_generated']['applied']}"
    )
    k = summary["pass_hat_k"]
    lines.append(f"pass^k (k={k['k']}): {k['all_repeats_passed']}/{k['cases']} 질의가 모든 반복에서 통과")
    for failure in summary["failures"][:_FAILURE_LINES]:
        lines.append(f"  실패 {failure['case_id']} {failure['metric']}: {failure['reason'][:_REASON_PREVIEW]}")
    if summary["skipped"]:
        lines.append(f"건너뜀 {len(summary['skipped'])}건(예산 등)")
    limit = cost.get("limits", {}).get("total_usd")
    limit_text = f" / 상한 ${limit:.2f}" if limit is not None else ""
    lines.append(f"비용: ${cost.get('total_usd', 0.0):.2f}{limit_text}")
    if comparison is not None:
        paired = comparison["paired"]
        lines.append(
            f"기준선 비교: {'회귀' if comparison['regression'] else '회귀 아님'} "
            f"(통과→실패 {paired['regressions']}, 실패→통과 {paired['improvements']}, p={paired['p_value']}, "
            f"safety 실패 {len(comparison['safety_failures'])}, 평균 하락 {comparison['mean_drops']})"
        )
    for kind, rate in (mutation_rates or {}).items():
        lines.append(
            f"변이 {kind:16} 탐지 {rate['detected']}/{rate['applicable']} ({rate['rate']:.0%}) by {rate['expected_detectors']}"
        )
    return lines
