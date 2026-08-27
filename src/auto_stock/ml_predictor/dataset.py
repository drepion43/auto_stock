"""피처+레이블 결합, 다종목 pooled 데이터셋 구성, 시간순 분할(embargo 포함).

`train_test_split`/`KFold`/`shuffle=True`는 이 모듈에서 절대 사용하지 않는다 — 랜덤
분할은 미래 정보가 학습 세트로 새는 것을 감출 수 있어 금융 시계열에서 금지된다.
다종목 데이터셋은 종목별이 아니라 전역 날짜 기준으로 분할한다: 종목별 분할은
한 종목의 미래가 다른 종목의 과거와 뒤섞여도 감지되지 않는 시간적 정보 누출을 낳는다.
"""

from auto_stock.data.models import OHLCVRecord
from auto_stock.ml_predictor.features import FEATURE_NAMES, build_feature_vectors
from auto_stock.ml_predictor.labeling import LABEL_HORIZON_DAYS, LABEL_THRESHOLD, forward_returns, to_label
from auto_stock.ml_predictor.models import DatasetSplit, LabeledSample, TrainingDataset


def build_samples(records: list[OHLCVRecord]) -> list[LabeledSample]:
    """워밍업 구간(피처 None)과 레이블 없는 마지막 horizon개 행을 모두 제거한 표본."""
    if not records:
        return []

    vectors = build_feature_vectors(records)
    closes = [r.close for r in records]
    fwd_returns = forward_returns(closes, LABEL_HORIZON_DAYS)

    samples: list[LabeledSample] = []
    for vector, fwd_return in zip(vectors, fwd_returns):
        if vector is None or fwd_return is None:
            continue
        label = to_label(fwd_return, LABEL_THRESHOLD)
        samples.append(LabeledSample(feature=vector, label=label))
    return samples


def build_training_dataset(records_by_ticker: list[list[OHLCVRecord]]) -> TrainingDataset:
    """여러 종목의 레코드를 하나의 pooled 데이터셋으로 결합, 전역 날짜순 정렬한다."""
    all_samples: list[LabeledSample] = []
    for records in records_by_ticker:
        all_samples.extend(build_samples(records))

    all_samples.sort(key=lambda s: (s.feature.date, s.feature.ticker))
    return TrainingDataset(samples=all_samples, feature_names=FEATURE_NAMES)


def chronological_split(
    dataset: TrainingDataset, test_ratio: float = 0.2, embargo_days: int | None = None
) -> DatasetSplit:
    """전역 날짜 기준 시간순 분할 + embargo 갭. 기본 embargo=LABEL_HORIZON_DAYS."""
    if embargo_days is None:
        embargo_days = LABEL_HORIZON_DAYS

    unique_dates = sorted({s.feature.date for s in dataset.samples})
    n_dates = len(unique_dates)
    split_idx = int(n_dates * (1 - test_ratio))
    split_idx = max(1, min(split_idx, n_dates - 1))

    embargo_start_idx = max(0, split_idx - embargo_days)
    train_dates = set(unique_dates[:embargo_start_idx])
    test_dates = set(unique_dates[split_idx:])

    train_samples = sorted(
        (s for s in dataset.samples if s.feature.date in train_dates),
        key=lambda s: (s.feature.date, s.feature.ticker),
    )
    test_samples = sorted(
        (s for s in dataset.samples if s.feature.date in test_dates),
        key=lambda s: (s.feature.date, s.feature.ticker),
    )

    return DatasetSplit(
        train=TrainingDataset(samples=train_samples, feature_names=dataset.feature_names),
        test=TrainingDataset(samples=test_samples, feature_names=dataset.feature_names),
    )


def to_xy(dataset: TrainingDataset) -> tuple[list[list[float]], list[int]]:
    X = [[s.feature.values[name] for name in dataset.feature_names] for s in dataset.samples]
    y = [s.label for s in dataset.samples]
    return X, y
