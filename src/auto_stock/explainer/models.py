from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Explanation:
    ticker: str
    market: str
    action: str  # "BUY" | "SELL"
    summary: str
