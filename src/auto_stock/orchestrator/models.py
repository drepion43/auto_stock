from dataclasses import dataclass

from auto_stock.explainer.models import Explanation

DEFAULT_LOOKBACK_DAYS = 120  # SMA60 warmup + ATR(14), padded for weekends/holidays


@dataclass(frozen=True, slots=True)
class PipelineResult:
    sent: list[Explanation]
    errors: list[tuple[str, str]]
