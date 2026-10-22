"""선형 기준선(ridge) — Mamba를 쓰려면 같은 피처·같은 타깃으로 이것을 검증 IC에서 이겨야 한다.

2026-10-22 진단: 같은 17개 피처로 ridge가 국내 test IC +0.025~+0.046을 낸 반면 Mamba는 ≈0이었다. 즉 신호는
데이터에 있는데 Mamba 학습이 그것을 꺼내지 못했다. 그래서 학습 리포트는 항상 이 기준선과 나란히 보여 주고,
Mamba가 이기지 못하면 배포하지 말라고 표시한다(scripts/train_ml_model.py).

전처리는 Mamba와 같다: train 구간 통계로 표준화, ±`clip_sigma`로 자르기, 레이블 ±`label_clip`. 시퀀스 상태가
없는 시점별 선형 회귀라 "그 시점 피처만으로" 예측한다."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from auto_stock.ml_predictor.models import SequenceDataset

DEFAULT_ALPHA = 1.0
DEFAULT_LABEL_CLIP = 0.2
DEFAULT_CLIP_SIGMA = 5.0


@dataclass(frozen=True)
class RidgeModel:
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray  # 마지막 원소는 절편
    clip_sigma: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = np.clip((X - self.mean) / self.std, -self.clip_sigma, self.clip_sigma)
        return np.hstack([Z, np.ones((len(Z), 1))]) @ self.weights


def _matrix(dataset: SequenceDataset, feature_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = [[s.feature.values[n] for n in feature_names] for seq in dataset.sequences for s in seq.samples]
    labels = [s.label for seq in dataset.sequences for s in seq.samples]
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=float)


def fit_ridge(
    dataset: SequenceDataset,
    feature_names: Sequence[str],
    alpha: float = DEFAULT_ALPHA,
    label_clip: float | None = DEFAULT_LABEL_CLIP,
    clip_sigma: float = DEFAULT_CLIP_SIGMA,
) -> RidgeModel:
    X, y = _matrix(dataset, feature_names)
    if label_clip is not None:
        y = np.clip(y, -label_clip, label_clip)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    Z = np.clip((X - mean) / std, -clip_sigma, clip_sigma)
    Z1 = np.hstack([Z, np.ones((len(Z), 1))])
    penalty = alpha * np.eye(Z1.shape[1])
    penalty[-1, -1] = 0.0  # 절편은 벌점 없음
    weights = np.linalg.solve(Z1.T @ Z1 + penalty, Z1.T @ y)
    return RidgeModel(mean=mean, std=std, weights=weights, clip_sigma=clip_sigma)


def ridge_predictions(
    model: RidgeModel, dataset: SequenceDataset, feature_names: Sequence[str], window_dates: set[date] | None = None
) -> tuple[list[float], list[float], list[date], list[str]]:
    """(예측, 실제 레이블, 날짜, 티커) — `window_dates` 안 시점만. evaluation.daily_rank_ic 입력 형태와 맞춘다."""
    points = [
        (seq.ticker, s)
        for seq in dataset.sequences
        for s in seq.samples
        if window_dates is None or s.feature.date in window_dates
    ]
    if not points:
        return [], [], [], []
    X = np.asarray([[s.feature.values[n] for n in feature_names] for _t, s in points], dtype=float)
    predictions = model.predict(X).tolist()
    return predictions, [s.label for _t, s in points], [s.feature.date for _t, s in points], [t for t, _s in points]
