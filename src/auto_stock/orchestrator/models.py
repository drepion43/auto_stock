from dataclasses import dataclass

from auto_stock.explainer.models import Explanation


@dataclass(frozen=True, slots=True)
class PipelineResult:
    sent: list[Explanation]
    errors: list[tuple[str, str]]
