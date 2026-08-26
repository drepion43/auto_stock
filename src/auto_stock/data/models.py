from dataclasses import dataclass
from datetime import date

VALID_MARKETS = frozenset({"KRX", "NASDAQ"})


@dataclass(frozen=True, slots=True)
class OHLCVRecord:
    ticker: str
    market: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __post_init__(self) -> None:
        if self.market not in VALID_MARKETS:
            raise ValueError(f"unknown market: {self.market!r} (expected one of {sorted(VALID_MARKETS)})")
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) cannot be below low ({self.low})")
