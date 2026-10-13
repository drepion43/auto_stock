"""ml_predictor 도메인 모델. rule_engine/models.py와 동일하게 frozen dataclass + slots로
불변 데이터를 표현한다.

Mamba(SSM) 교체(docs/design/ml-predictor-mamba-plan.md)로 "표본 1개 = 피처벡터 1개"의
flat pooled 구조가 "표본 1개 = 종목 하나의 (가변 길이) 시퀀스" 구조로 바뀌었다 —
`TrainingDataset`/`DatasetSplit`(flat) 대신 `TickerSequence`/`SequenceDataset`/
`SequenceDatasetSplit`을 쓴다. 레이블도 N거래일 후 상승/하락 이진 분류에서 다음날
수익률 regression으로 바뀌어 `LabeledSample.label`이 `int`에서 `float`가 됐다."""

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
    label: float  # 다음날 수익률(close[t+1]/close[t] - 1)


@dataclass(frozen=True, slots=True)
class TickerSequence:
    """종목 1개의 시간순 (피처, 레이블) 시퀀스 — Mamba 배치의 원자 단위(배치 축 = 종목)."""

    ticker: str
    market: str
    samples: list[LabeledSample]


@dataclass(frozen=True, slots=True)
class SequenceDataset:
    sequences: list[TickerSequence]
    feature_names: list[str]


@dataclass(frozen=True, slots=True)
class SequenceDatasetSplit:
    train: SequenceDataset
    test: SequenceDataset


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    market: str  # 통합 단일 모델이라 항상 "UNIFIED" (사용자 결정 — market/종목별 모델 분리 안 함)
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
    test_mse: float
    test_rmse: float
    test_mae: float
    test_r2: float
    model_config: dict[str, int]  # MambaRegressor 재구성에 필요한 하이퍼파라미터(in_dim/hidden_dim/n_layers)
    trained_at: datetime


@dataclass(frozen=True, slots=True)
class ModelBundle:
    estimator: Any  # 학습된 MambaRegressor(nn.Module) — untyped by design (3rd-party 패턴 재사용)
    metadata: ModelMetadata


@dataclass(frozen=True, slots=True)
class MLPrediction:
    ticker: str
    market: str
    date: date
    predicted_return: float  # 다음 거래일 예측 수익률(음수 가능), sigmoid로 유사-확률화하지 않음
