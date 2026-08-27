"""ml_predictor 도메인 모델. rule_engine/models.py와 동일하게 frozen dataclass + slots로
불변 데이터를 표현한다."""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class FeatureVector:
    ticker: str
    market: str
    date: date
    values: dict[str, float]  # keyed by ml_predictor.features.FEATURE_NAMES


@dataclass(frozen=True, slots=True)
class LabeledSample:
    feature: FeatureVector
    label: int  # 0 | 1


@dataclass(frozen=True, slots=True)
class TrainingDataset:
    samples: list[LabeledSample]
    feature_names: list[str]


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    train: TrainingDataset
    test: TrainingDataset


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    market: str
    algorithm: str
    horizon_days: int
    feature_names: list[str]
    universe_size: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_samples: int
    test_samples: int
    test_base_rate: float
    test_roc_auc: float
    test_accuracy: float
    trained_at: datetime


@dataclass(frozen=True, slots=True)
class ModelBundle:
    estimator: Any  # fitted sklearn Pipeline/estimator — untyped by design (3rd-party object)
    metadata: ModelMetadata


@dataclass(frozen=True, slots=True)
class MLPrediction:
    ticker: str
    market: str
    date: date
    probability_up: float
    top_features: list[tuple[str, float]]  # (feature_name, contribution), sorted by |contribution| desc
