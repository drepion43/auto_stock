"""학습된 모델의 구간 평가 — validation/test 구간의 **모든 (종목, 날짜)** 시점을 채점한다.

이전 평가(`training.evaluate`, 2026-10-18까지)는 종목마다 마지막 시점 1개만 봤다. 그런데 test
구간은 모든 종목이 같은 날짜에서 끝나므로 "592종목 평가"가 사실상 **하루치 시장 움직임 1번**이었다
(국내 baseline 적중률 13% 같은 극단값이 그 증상). 여기서는 구간 안의 모든 시점을 채점한다.

모델에는 각 종목의 **전체 히스토리**를 넣고(상태가 과거를 모두 본 뒤) 구간 안 시점만 채점한다 —
추론(`predictor.predict`)이 3년치 히스토리를 넣고 마지막 시점을 쓰는 것과 같은 조건이다. 구간
밖 시점의 레이블은 채점에 쓰지 않으므로 누출은 없다(모델 입력은 그 시점까지의 피처뿐 — 인과적).

비교 기준 2개: `zero`(항상 0 — 예측력이 없으면 MSE로는 이게 사실상 최선)와 `persistence`
(내일 수익률 = 오늘 수익률). 예측 분포(평균/표준편차/방향성 비율)는 모델이 0 근처만 내는
"평균 수렴" 상태인지 판별하는 데 쓴다."""

import math
import statistics
from dataclasses import dataclass
from datetime import date

import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from auto_stock.data.market_routing import KRX_MARKETS
from auto_stock.ml_predictor.dataset import collate_sequences
from auto_stock.ml_predictor.models import SequenceDataset

PERSISTENCE_FEATURE_NAME = "return_1d"
EVAL_BATCH_SIZE = 32


@dataclass(frozen=True, slots=True)
class WindowEvaluation:
    n_samples: int  # 채점한 (종목, 날짜) 시점 수
    n_dates: int  # 그중 서로 다른 날짜 수 — 지표가 며칠치 시장을 반영하는지
    model: dict[str, float]
    zero: dict[str, float]
    persistence: dict[str, float]
    prediction_mean: float
    prediction_std: float
    directional_share: float  # |예측| >= directional_threshold 비율 (ML 신호가 의견을 내는 비율)
    positive_share: float  # 예측 > 0 비율


def _nan_metrics() -> dict[str, float]:
    return {key: float("nan") for key in ("mse", "rmse", "mae", "r2", "hit_rate")}


def _hit_rate(actuals: list[float], predictions: list[float]) -> float:
    """방향 적중률 — 예측과 실제 수익률의 부호가 같은 비율. 실제 수익률이 정확히 0인 표본은
    방향이 없으므로 분모에서 뺀다(예측 0은 어느 방향도 맞히지 못한 것으로 친다)."""
    pairs = [(a, p) for a, p in zip(actuals, predictions) if a != 0]
    if not pairs:
        return float("nan")
    return sum(1 for a, p in pairs if (a > 0) == (p > 0) and p != 0) / len(pairs)


def regression_metrics(actuals: list[float], predictions: list[float]) -> dict[str, float]:
    if len(actuals) < 2:
        return _nan_metrics()
    mse = mean_squared_error(actuals, predictions)
    return {
        "mse": float(mse),
        "rmse": float(mse**0.5),
        "mae": float(mean_absolute_error(actuals, predictions)),
        "r2": float(r2_score(actuals, predictions)),
        "hit_rate": _hit_rate(actuals, predictions),
    }


def split_by_region(dataset: SequenceDataset) -> dict[str, SequenceDataset]:
    """시장별 성능 비교용 — 국내(KOSPI/KOSDAQ)와 미국(NASDAQ/NYSE)으로 종목 시퀀스를 나눈다."""
    domestic = [seq for seq in dataset.sequences if seq.market in KRX_MARKETS]
    us = [seq for seq in dataset.sequences if seq.market not in KRX_MARKETS]
    return {
        "domestic": SequenceDataset(sequences=domestic, feature_names=dataset.feature_names),
        "us": SequenceDataset(sequences=us, feature_names=dataset.feature_names),
    }


def _window_points(
    model: nn.Module,
    dataset: SequenceDataset,
    feature_names: list[str],
    window_dates: set[date] | None,
    batch_size: int,
) -> tuple[list[float], list[float], list[float], set[date]]:
    """(모델 예측, 실제 레이블, persistence 예측, 채점한 날짜 집합) — 구간 안 시점만."""
    model.eval()
    device = next(model.parameters()).device
    predictions: list[float] = []
    actuals: list[float] = []
    persistence: list[float] = []
    dates: set[date] = set()
    sequences = dataset.sequences
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            X, _, _ = collate_sequences(batch, feature_names)
            output = model(torch.tensor(X, dtype=torch.float32, device=device)).cpu()
            for row, seq in enumerate(batch):
                for step, sample in enumerate(seq.samples):
                    if window_dates is not None and sample.feature.date not in window_dates:
                        continue
                    predictions.append(float(output[row, step]))
                    actuals.append(sample.label)
                    persistence.append(sample.feature.values.get(PERSISTENCE_FEATURE_NAME, math.nan))
                    dates.add(sample.feature.date)
    return predictions, actuals, persistence, dates


def evaluate_window(
    model: nn.Module,
    dataset: SequenceDataset,
    feature_names: list[str],
    window_dates: set[date] | None = None,
    *,
    directional_threshold: float = 0.0,
    batch_size: int = EVAL_BATCH_SIZE,
) -> WindowEvaluation:
    """`window_dates`(None이면 전 구간) 안의 모든 시점을 채점한다. `dataset`에는 종목의 전체
    히스토리를 넘겨야 구간 시작 전 과거가 모델 상태에 반영된다(구간만 잘라 넘기면 그 앞 히스토리
    없이 평가된다 — validation loss 계산과 같은 조건)."""
    predictions, actuals, persistence, dates = _window_points(model, dataset, feature_names, window_dates, batch_size)
    persistence_available = bool(persistence) and not any(math.isnan(p) for p in persistence)
    return WindowEvaluation(
        n_samples=len(predictions),
        n_dates=len(dates),
        model=regression_metrics(actuals, predictions),
        zero=regression_metrics(actuals, [0.0] * len(actuals)),
        persistence=regression_metrics(actuals, persistence) if persistence_available else _nan_metrics(),
        prediction_mean=statistics.fmean(predictions) if predictions else math.nan,
        prediction_std=statistics.pstdev(predictions) if predictions else math.nan,
        directional_share=(
            sum(1 for p in predictions if abs(p) >= directional_threshold) / len(predictions) if predictions else math.nan
        ),
        positive_share=sum(1 for p in predictions if p > 0) / len(predictions) if predictions else math.nan,
    )
