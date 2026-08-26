from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Candidate:
    ticker: str
    market: str
    action: str  # "BUY" | "SELL"
    reasons: list[str]
