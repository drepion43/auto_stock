"""llm_chart_analyst 테스트 전용 fixture.

`OPENAI_API_KEY`를 더미 값으로 monkeypatch하는 autouse 픽스처는 실수로도 실제 API 호출이
발생하지 않도록 하는 안전장치다(모든 테스트가 이 conftest를 통해 이 픽스처를 자동 적용받는다).
"""

import random
from datetime import date, timedelta

import pytest

from auto_stock.data.models import OHLCVRecord
from auto_stock.llm_chart_analyst.models import ChartAnalysis
from auto_stock.llm_chart_analyst.schema import ChartPatternRead


@pytest.fixture(autouse=True)
def _dummy_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key-not-real")


def make_records(
    prices: list[float],
    ticker: str = "005930",
    market: str = "KRX",
    start: date = date(2024, 1, 1),
    volumes: list[int] | None = None,
) -> list[OHLCVRecord]:
    n = len(prices)
    if volumes is None:
        volumes = [1_000 + i for i in range(n)]
    records = []
    for i, price in enumerate(prices):
        records.append(
            OHLCVRecord(
                ticker=ticker,
                market=market,
                date=start + timedelta(days=i),
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=volumes[i],
            )
        )
    return records


def random_walk_prices(n: int, seed: int = 42, start_price: float = 100.0) -> list[float]:
    rng = random.Random(seed)
    prices = [start_price]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + rng.uniform(-0.02, 0.025)))
    return prices


@pytest.fixture
def warmed_up_records() -> list[OHLCVRecord]:
    """SMA60 워밍업을 충분히 넘기는 결정론적 합성 OHLCV(80봉)."""
    return make_records(random_walk_prices(80))


class FakeChartPatternReader:
    """`ChartPatternReader` Protocol 구현체 — 진짜 SDK 없이 analyst.py를 테스트하기 위함."""

    def __init__(
        self,
        response: ChartPatternRead | None = None,
        error: Exception | None = None,
        model: str = "gpt-5.6-luna",
    ) -> None:
        self.model = model
        self._response = response or ChartPatternRead(
            direction="UP",
            confidence="MEDIUM",
            pattern_name="상승 삼각수렴",
            rationale="RSI가 반등하고 종가가 SMA20을 상회합니다.",
            caveat=None,
        )
        self._error = error
        self.calls: list[tuple[str, str]] = []

    def read_pattern(self, system_prompt: str, user_prompt: str) -> ChartPatternRead:
        self.calls.append((system_prompt, user_prompt))
        if self._error is not None:
            raise self._error
        return self._response


@pytest.fixture
def fake_reader() -> FakeChartPatternReader:
    return FakeChartPatternReader()


def make_chart_analysis(**overrides) -> ChartAnalysis:
    values = dict(
        ticker="005930",
        market="KRX",
        date=date(2024, 6, 1),
        direction="UP",
        confidence="MEDIUM",
        pattern_name="상승 삼각수렴",
        rationale="RSI가 반등하고 종가가 SMA20을 상회합니다.",
        caveat=None,
        model="gpt-5.6-luna",
    )
    values.update(overrides)
    return ChartAnalysis(**values)
