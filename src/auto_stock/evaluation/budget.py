"""평가 1회 비용 상한을 코드로 강제한다 — 계획서 §7.1, §7.3.

호출 **전에** "누적 + 이번 호출 최대 예상 비용"을 검사해 상한을 넘을 상황이면 예외를 던진다.
러너는 예외를 잡아 남은 케이스를 skipped로 남기고 부분 결과로 끝낸다."""

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass


class BudgetExceeded(RuntimeError):
    """평가 1회 전체 상한(생성 + 채점) 도달."""


class JudgeBudgetExceeded(BudgetExceeded):
    """채점 비용·호출 수 상한 도달."""


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    total_usd: float = 8.0
    judge_usd: float = 6.0
    judge_calls: int = 400

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "BudgetLimits":
        env = os.environ if env is None else env
        defaults = cls()
        return cls(
            total_usd=float(env.get("EVAL_MAX_COST_USD", defaults.total_usd)),
            judge_usd=float(env.get("EVAL_JUDGE_MAX_COST_USD", defaults.judge_usd)),
            judge_calls=int(env.get("EVAL_JUDGE_MAX_CALLS", defaults.judge_calls)),
        )


class CostLedger:
    """생성·채점 비용 누적. 채점 모델 래퍼와 러너가 같은 인스턴스를 공유한다."""

    def __init__(self, limits: BudgetLimits) -> None:
        self.limits = limits
        self.generation_usd = 0.0
        self.judge_usd = 0.0
        self.judge_calls = 0

    @property
    def total_usd(self) -> float:
        return self.generation_usd + self.judge_usd

    def check_generation(self, max_cost: float) -> None:
        if self.total_usd + max_cost > self.limits.total_usd:
            raise BudgetExceeded(f"total budget ${self.limits.total_usd:.2f} would be exceeded")

    def check_judge(self, max_cost: float) -> None:
        if self.judge_calls + 1 > self.limits.judge_calls:
            raise JudgeBudgetExceeded(f"judge call cap {self.limits.judge_calls} reached")
        if self.judge_usd + max_cost > self.limits.judge_usd:
            raise JudgeBudgetExceeded(f"judge budget ${self.limits.judge_usd:.2f} would be exceeded")
        if self.total_usd + max_cost > self.limits.total_usd:
            raise JudgeBudgetExceeded(f"total budget ${self.limits.total_usd:.2f} would be exceeded")

    def add_generation(self, cost: float) -> None:
        self.generation_usd += cost

    def add_judge(self, cost: float) -> None:
        self.judge_usd += cost
        self.judge_calls += 1

    def snapshot(self) -> dict:
        return {
            "generation_usd": round(self.generation_usd, 6),
            "judge_usd": round(self.judge_usd, 6),
            "total_usd": round(self.total_usd, 6),
            "judge_calls": self.judge_calls,
            "limits": asdict(self.limits),
        }
