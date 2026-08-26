from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AccountState:
    equity: float
    held_tickers: frozenset[str]
    total_exposure_pct: float

    def __post_init__(self) -> None:
        if self.equity < 0:
            raise ValueError(f"equity cannot be negative: {self.equity}")
        if not 0 <= self.total_exposure_pct <= 1:
            raise ValueError(f"total_exposure_pct must be within [0, 1]: {self.total_exposure_pct}")


@dataclass(frozen=True, slots=True)
class SizingSuggestion:
    ticker: str
    market: str
    action: str  # "BUY" | "SELL"
    suggested_quantity: int | None
    suggested_allocation_pct: float | None
    stop_loss_price: float | None
    take_profit_price: float | None
    limit_check: str  # "PASS" | "EXCEEDS_MAX_POSITIONS" | "EXCEEDS_EXPOSURE_CAP" | "NOT_APPLICABLE"
    notes: list[str]
