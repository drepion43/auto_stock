"""ml_predictor 테스트 전용 결정론적 합성 데이터 fixture.

실제 시장 데이터로 단위테스트를 하면 느리고 비결정적이므로, 재현 가능한 합성
가격열/피처셋을 사용한다 (수치 자체의 의미보다 배선(wiring)/속성(property) 검증 목적).
"""

import random
from datetime import date, timedelta

import pytest

from auto_stock.data.models import OHLCVRecord
from auto_stock.ml_predictor.features import FEATURE_NAMES
from auto_stock.ml_predictor.models import FeatureVector, LabeledSample, TrainingDataset


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
def long_prices() -> list[float]:
    return random_walk_prices(160)


@pytest.fixture
def long_records(long_prices) -> list[OHLCVRecord]:
    return make_records(long_prices)


@pytest.fixture
def synthetic_training_dataset() -> TrainingDataset:
    """LogReg가 배선을 검증할 수 있을 정도로 학습 가능한 관계를 갖는 합성 데이터셋.

    실제 피처 계산 경로(pandas-ta)를 타지 않고 FeatureVector를 직접 구성한다 —
    training.py 테스트는 "학습 파이프라인이 올바르게 배선됐는가"만 검증하면 충분하다.
    """
    rng = random.Random(0)
    samples: list[LabeledSample] = []
    start = date(2020, 1, 1)
    for i in range(200):
        values = {name: rng.uniform(-1, 1) for name in FEATURE_NAMES}
        label = 1 if values["return_1d"] + values["return_5d"] > 0 else 0
        feature = FeatureVector(ticker="TEST", market="KRX", date=start + timedelta(days=i), values=values)
        samples.append(LabeledSample(feature=feature, label=label))
    return TrainingDataset(samples=samples, feature_names=FEATURE_NAMES)
