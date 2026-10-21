"""지표 결과 한 건의 공통 모양 — 결정론 검사와 LLM 채점이 같은 형태로 리포트에 들어간다."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class MetricResult:
    metric: str
    applied: bool
    passed: bool | None  # None = 미적용, 채점 실패, 또는 사람 확인 필요
    score: float | None = None
    reason: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    needs_review: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        data["details"] = dict(self.details)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricResult":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


def not_applied(metric: str, reason: str) -> MetricResult:
    return MetricResult(metric=metric, applied=False, passed=None, reason=reason)


def from_violations(metric: str, violations: list[str], details: Mapping[str, Any] | None = None) -> MetricResult:
    passed = not violations
    return MetricResult(
        metric=metric,
        applied=True,
        passed=passed,
        score=1.0 if passed else 0.0,
        reason="ok" if passed else ", ".join(violations),
        details={"violations": violations, **(details or {})},
    )
