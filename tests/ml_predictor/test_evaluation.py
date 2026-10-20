import math
from datetime import date, timedelta

import pytest
import torch
import torch.nn as nn

from auto_stock.ml_predictor.evaluation import daily_rank_ic, evaluate_window, regression_metrics, split_by_region
from auto_stock.ml_predictor.models import FeatureVector, LabeledSample, SequenceDataset, TickerSequence

FEATURES = ["return_1d"]
D0 = date(2025, 1, 1)


def _day(i: int) -> date:
    return D0 + timedelta(days=i)


def _sequence(ticker: str, market: str, returns: list[float], labels: list[float], start: int = 0) -> TickerSequence:
    samples = [
        LabeledSample(
            feature=FeatureVector(ticker=ticker, market=market, date=_day(start + i), values={"return_1d": r}),
            label=label,
        )
        for i, (r, label) in enumerate(zip(returns, labels))
    ]
    return TickerSequence(ticker=ticker, market=market, samples=samples)


class ConstantModel(nn.Module):
    """Predicts the same return at every step."""

    def __init__(self, value: float):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))  # gives the module a device
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full(x.shape[:2], self.value) + self.anchor * 0


class CumulativeModel(nn.Module):
    """Prediction at step t = sum of return_1d up to t — only correct if it saw the full history."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(x[..., 0], dim=1) + self.anchor * 0


@pytest.fixture
def two_tickers() -> SequenceDataset:
    return SequenceDataset(
        sequences=[
            _sequence("AAA", "KOSPI", [0.01, -0.02, 0.03, 0.01], [0.02, -0.01, 0.01, -0.03]),
            _sequence("BBB", "NASDAQ", [0.02, 0.01, -0.01], [0.01, 0.02, -0.02], start=1),
        ],
        feature_names=FEATURES,
    )


def test_scores_every_step_not_just_the_last_one(two_tickers):
    result = evaluate_window(ConstantModel(0.01), two_tickers, FEATURES)

    assert result.n_samples == 7
    assert result.n_dates == 4


def test_window_dates_restrict_which_steps_are_scored(two_tickers):
    result = evaluate_window(ConstantModel(0.01), two_tickers, FEATURES, window_dates={_day(2), _day(3)})

    assert result.n_samples == 4  # AAA day2, day3 + BBB day2, day3
    assert result.n_dates == 2


def test_prediction_distribution_and_hit_rate_of_constant_model(two_tickers):
    result = evaluate_window(ConstantModel(0.01), two_tickers, FEATURES, directional_threshold=0.003)

    assert result.prediction_mean == pytest.approx(0.01)
    assert result.prediction_std == pytest.approx(0.0)
    assert result.directional_share == pytest.approx(1.0)
    assert result.positive_share == pytest.approx(1.0)
    # Always predicting "up" is right exactly on the up days: 4 of 7 labels are positive.
    assert result.model["hit_rate"] == pytest.approx(4 / 7)


def test_zero_baseline_has_mse_equal_to_mean_squared_label(two_tickers):
    labels = [0.02, -0.01, 0.01, -0.03, 0.01, 0.02, -0.02]

    result = evaluate_window(ConstantModel(0.01), two_tickers, FEATURES)

    assert result.zero["mse"] == pytest.approx(sum(v * v for v in labels) / len(labels))
    assert result.zero["hit_rate"] == 0.0  # a zero prediction never calls a direction


def test_persistence_baseline_predicts_todays_return(two_tickers):
    # Window day0+day1: AAA (+0.01 -> +0.02), AAA (-0.02 -> -0.01), BBB (+0.02 -> +0.01) — all signs agree.
    result = evaluate_window(ConstantModel(0.0), two_tickers, FEATURES, window_dates={_day(0), _day(1)})

    assert result.n_samples == 3
    assert result.persistence["hit_rate"] == pytest.approx(1.0)
    expected_mse = ((0.01 - 0.02) ** 2 + (-0.02 + 0.01) ** 2 + (0.02 - 0.01) ** 2) / 3
    assert result.persistence["mse"] == pytest.approx(expected_mse)


def test_window_steps_are_predicted_with_full_history_as_context():
    dataset = SequenceDataset(
        sequences=[_sequence("AAA", "KOSPI", [0.01, 0.02, 0.03, 0.04], [0.0, 0.0, 0.0, 0.0])],
        feature_names=FEATURES,
    )

    result = evaluate_window(CumulativeModel(), dataset, FEATURES, window_dates={_day(3)})

    # With the whole history as context the last-step prediction is 0.01+0.02+0.03+0.04.
    assert result.prediction_mean == pytest.approx(0.10)


def test_empty_window_returns_nan_metrics(two_tickers):
    result = evaluate_window(ConstantModel(0.01), two_tickers, FEATURES, window_dates={_day(99)})

    assert result.n_samples == 0
    assert math.isnan(result.model["mse"])
    assert math.isnan(result.prediction_mean)


def test_regression_metrics_hit_rate_counts_sign_agreement():
    metrics = regression_metrics([0.02, -0.03, 0.01, -0.01], [0.01, -0.01, 0.02, 0.01])

    assert metrics["hit_rate"] == pytest.approx(0.75)
    assert set(metrics) == {"mse", "rmse", "mae", "r2", "hit_rate"}


class ReturnEchoModel(nn.Module):
    """Predicts today's return_1d — ranks tickers exactly like the persistence baseline."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., 0] + self.anchor * 0


def _cross_section(n_tickers: int, n_days: int, label_sign: float) -> SequenceDataset:
    """Ticker i has return_1d = i/100 every day; next-day label = label_sign * that (perfectly (anti-)ranked)."""
    return SequenceDataset(
        sequences=[
            _sequence(f"T{i}", "NASDAQ", [i / 100] * n_days, [label_sign * i / 100] * n_days)
            for i in range(1, n_tickers + 1)
        ],
        feature_names=FEATURES,
    )


def test_daily_rank_ic_is_one_for_a_perfect_ranking():
    summary = daily_rank_ic([[0.1, 0.2, 0.3, 0.4, 0.5]] * 3, [[1.0, 2.0, 3.0, 4.0, 5.0]] * 3, min_tickers=5)

    assert summary.n_days == 3
    assert summary.mean_ic == pytest.approx(1.0)
    assert summary.positive_day_share == pytest.approx(1.0)
    # top 20% (1 ticker, label 5) minus bottom 20% (1 ticker, label 1)
    assert summary.mean_top_bottom_spread == pytest.approx(4.0)


def test_daily_rank_ic_skips_days_with_too_few_tickers_or_constant_predictions():
    summary = daily_rank_ic([[0.1, 0.2], [0.0, 0.0, 0.0, 0.0, 0.0]], [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0]], min_tickers=5)

    assert summary.n_days == 0
    assert math.isnan(summary.mean_ic)


def test_evaluate_window_reports_cross_sectional_ic_for_model_and_persistence():
    dataset = _cross_section(n_tickers=10, n_days=4, label_sign=-1.0)  # tomorrow reverses today's ranking

    result = evaluate_window(ReturnEchoModel(), dataset, FEATURES, min_ic_tickers=10)

    assert result.model_ic.n_days == 4
    assert result.model_ic.mean_ic == pytest.approx(-1.0)
    assert result.persistence_ic.mean_ic == pytest.approx(-1.0)
    assert result.model_ic.mean_top_bottom_spread < 0


def test_constant_model_has_no_ranking_ic(two_tickers):
    result = evaluate_window(ConstantModel(0.01), _cross_section(10, 3, 1.0), FEATURES, min_ic_tickers=10)

    assert result.model_ic.n_days == 0
    assert result.persistence_ic.mean_ic == pytest.approx(1.0)


def test_split_by_region_groups_domestic_and_us_tickers(two_tickers):
    regions = split_by_region(two_tickers)

    assert [seq.ticker for seq in regions["domestic"].sequences] == ["AAA"]
    assert [seq.ticker for seq in regions["us"].sequences] == ["BBB"]
