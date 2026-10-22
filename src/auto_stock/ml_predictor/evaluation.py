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
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from auto_stock.data.market_routing import KRX_MARKETS
from auto_stock.ml_predictor.dataset import collate_sequences
from auto_stock.ml_predictor.models import SequenceDataset

PERSISTENCE_FEATURE_NAME = "return_1d"
EVAL_BATCH_SIZE = 32
MIN_IC_TICKERS = 10  # 하루에 이보다 적은 종목만 있으면 순위 상관이 불안정해 그날은 뺀다
TOP_BOTTOM_QUANTILE = 0.2  # 상위/하위 20% 종목으로 스프레드를 잰다


@dataclass(frozen=True, slots=True)
class IcSummary:
    """날짜별 종목 간 순위 상관(Spearman rank IC) 요약 — "그날 어느 종목이 상대적으로 더 오를지"를
    맞히는 능력. 종목 추천에 실제로 필요한 건 절대 수익률이 아니라 이 순위다."""

    n_days: int  # IC를 계산한 날 수(종목 수 미달·예측 상수인 날 제외)
    mean_ic: float
    ic_t_stat: float  # mean / std * sqrt(n_days) — 대략 2를 넘으면 우연이 아닐 가능성이 크다
    positive_day_share: float  # IC > 0 인 날 비율
    mean_top_bottom_spread: float  # 예측 상위 20% 평균 실제수익률 − 하위 20% (일평균, 비용 전)
    # 매일 상위·하위 20%를 다시 짠다고 볼 때 — 두 다리(상위·하위)에서 전날 대비 바뀐 종목 비율의 평균(0~1).
    mean_turnover: float = math.nan
    # 비용 차감 스프레드: 스프레드 − 왕복비용 × (상위 교체비율 + 하위 교체비율). 첫날은 전부 새로 산 것으로 친다.
    # 단기 반전처럼 매일 종목이 바뀌는 신호는 비용 전 스프레드가 양수여도 이 값이 음수일 수 있다.
    mean_net_spread: float = math.nan


def _nan_ic() -> IcSummary:
    return IcSummary(n_days=0, mean_ic=math.nan, ic_t_stat=math.nan, positive_day_share=math.nan,
                     mean_top_bottom_spread=math.nan)


def _replaced_share(current: set[str], previous: set[str] | None) -> float:
    if previous is None:
        return 1.0
    return len(current - previous) / len(current) if current else 0.0


def daily_rank_ic(
    predictions_by_day: list[list[float]],
    actuals_by_day: list[list[float]],
    min_tickers: int = MIN_IC_TICKERS,
    quantile: float = TOP_BOTTOM_QUANTILE,
    *,
    tickers_by_day: list[list[str]] | None = None,
    round_trip_cost: float = 0.0,
) -> IcSummary:
    """날짜마다 (예측, 실제) 순위 상관과 상위−하위 스프레드를 구해 평균낸다. 종목이 `min_tickers`
    미만이거나 예측/실제가 그날 전부 같은 값이면(순위가 없음) 그날은 뺀다. `tickers_by_day`가 있으면
    상위·하위 종목 교체 비율(회전율)과 `round_trip_cost`(왕복 거래비용, 비율)를 뺀 순 스프레드도 낸다."""
    ics: list[float] = []
    spreads: list[float] = []
    turnovers: list[float] = []
    net_spreads: list[float] = []
    previous_legs: tuple[set[str], set[str]] | None = None
    for day_index, (predictions, actuals) in enumerate(zip(predictions_by_day, actuals_by_day)):
        p, a = np.asarray(predictions, dtype=float), np.asarray(actuals, dtype=float)
        if len(p) < min_tickers or np.ptp(p) == 0 or np.ptp(a) == 0:
            continue
        ics.append(float(np.corrcoef(rankdata(p), rankdata(a))[0, 1]))
        k = max(1, int(len(p) * quantile))
        order = np.argsort(p)
        spread = float(a[order[-k:]].mean() - a[order[:k]].mean())
        spreads.append(spread)
        if tickers_by_day is not None:
            names = tickers_by_day[day_index]
            legs = ({names[i] for i in order[-k:]}, {names[i] for i in order[:k]})
            long_share = _replaced_share(legs[0], previous_legs[0] if previous_legs else None)
            short_share = _replaced_share(legs[1], previous_legs[1] if previous_legs else None)
            turnovers.append((long_share + short_share) / 2)
            net_spreads.append(spread - round_trip_cost * (long_share + short_share))
            previous_legs = legs
    if not ics:
        return _nan_ic()
    std = statistics.stdev(ics) if len(ics) >= 2 else 0.0
    return IcSummary(
        n_days=len(ics),
        mean_ic=statistics.fmean(ics),
        ic_t_stat=statistics.fmean(ics) / std * math.sqrt(len(ics)) if std > 0 else math.nan,
        positive_day_share=sum(1 for ic in ics if ic > 0) / len(ics),
        mean_top_bottom_spread=statistics.fmean(spreads),
        mean_turnover=statistics.fmean(turnovers) if turnovers else math.nan,
        mean_net_spread=statistics.fmean(net_spreads) if net_spreads else math.nan,
    )


def _group_by_day(
    values: list[float], actuals: list[float], dates: list[date], tickers: list[str] | None = None
) -> tuple[list[list[float]], list[list[float]], list[list[str]]]:
    by_day: dict[date, tuple[list[float], list[float], list[str]]] = defaultdict(lambda: ([], [], []))
    names = tickers if tickers is not None else [""] * len(values)
    for value, actual, day, name in zip(values, actuals, dates, names):
        by_day[day][0].append(value)
        by_day[day][1].append(actual)
        by_day[day][2].append(name)
    days = sorted(by_day)
    return [by_day[d][0] for d in days], [by_day[d][1] for d in days], [by_day[d][2] for d in days]


def summarize_predictions(
    predictions: list[float],
    actuals: list[float],
    dates: list[date],
    tickers: list[str],
    *,
    round_trip_cost: float = 0.0,
    min_tickers: int = MIN_IC_TICKERS,
) -> IcSummary:
    """임의 모델(예: baseline.ridge_predictions)의 시점별 예측을 같은 IC·비용 지표로 요약한다."""
    by_value, by_actual, by_ticker = _group_by_day(predictions, actuals, dates, tickers)
    return daily_rank_ic(by_value, by_actual, min_tickers, tickers_by_day=by_ticker, round_trip_cost=round_trip_cost)


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
    model_ic: IcSummary  # 날짜별 종목 간 순위 상관(모델 예측)
    persistence_ic: IcSummary  # 같은 지표를 오늘 수익률(return_1d) 순위로 — 단기 반전/모멘텀 기준선


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
) -> tuple[list[float], list[float], list[float], list[date], list[str]]:
    """(모델 예측, 실제 레이블, persistence 예측, 날짜, 티커) — 구간 안 시점만, 같은 순서."""
    model.eval()
    device = next(model.parameters()).device
    predictions: list[float] = []
    actuals: list[float] = []
    persistence: list[float] = []
    dates: list[date] = []
    tickers: list[str] = []
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
                    dates.append(sample.feature.date)
                    tickers.append(seq.ticker)
    return predictions, actuals, persistence, dates, tickers


def evaluate_window(
    model: nn.Module,
    dataset: SequenceDataset,
    feature_names: list[str],
    window_dates: set[date] | None = None,
    *,
    directional_threshold: float = 0.0,
    batch_size: int = EVAL_BATCH_SIZE,
    min_ic_tickers: int = MIN_IC_TICKERS,
    round_trip_cost: float = 0.0,
) -> WindowEvaluation:
    """`window_dates`(None이면 전 구간) 안의 모든 시점을 채점한다. `dataset`에는 종목의 전체
    히스토리를 넘겨야 구간 시작 전 과거가 모델 상태에 반영된다(구간만 잘라 넘기면 그 앞 히스토리
    없이 평가된다 — validation loss 계산과 같은 조건). `round_trip_cost`는 순 스프레드 계산용 왕복비용."""
    predictions, actuals, persistence, dates, tickers = _window_points(
        model, dataset, feature_names, window_dates, batch_size
    )
    persistence_available = bool(persistence) and not any(math.isnan(p) for p in persistence)
    model_ic = summarize_predictions(
        predictions, actuals, dates, tickers, round_trip_cost=round_trip_cost, min_tickers=min_ic_tickers
    )
    persistence_ic = (
        summarize_predictions(
            persistence, actuals, dates, tickers, round_trip_cost=round_trip_cost, min_tickers=min_ic_tickers
        )
        if persistence_available else _nan_ic()
    )
    return WindowEvaluation(
        n_samples=len(predictions),
        n_dates=len(set(dates)),
        model=regression_metrics(actuals, predictions),
        zero=regression_metrics(actuals, [0.0] * len(actuals)),
        persistence=regression_metrics(actuals, persistence) if persistence_available else _nan_metrics(),
        prediction_mean=statistics.fmean(predictions) if predictions else math.nan,
        prediction_std=statistics.pstdev(predictions) if predictions else math.nan,
        directional_share=(
            sum(1 for p in predictions if abs(p) >= directional_threshold) / len(predictions) if predictions else math.nan
        ),
        positive_share=sum(1 for p in predictions if p > 0) / len(predictions) if predictions else math.nan,
        model_ic=model_ic,
        persistence_ic=persistence_ic,
    )
