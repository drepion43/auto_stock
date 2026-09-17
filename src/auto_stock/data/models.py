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


@dataclass(frozen=True, slots=True)
class DisclosureRecord:
    corp_code: str
    ticker: str
    report_name: str  # DART report_nm
    filed_date: date
    remark: str  # DART rm (정정/첨부 등 플래그, 없으면 "")


@dataclass(frozen=True, slots=True)
class NewsArticle:
    ticker: str
    market: str
    title: str
    url: str
    published_at: date
    source: str  # 네이버: 언론사 도메인(originallink에서 추출), GDELT: domain 필드
