"""2026-10-22 진단 후속: 시장 평균을 뺀 라벨(2번), 선형 기준선(3번), 거래비용 반영 스프레드(4번)."""

from datetime import date, timedelta

import numpy as np
import pytest

from auto_stock.ml_predictor.baseline import fit_ridge, ridge_predictions
from auto_stock.ml_predictor.dataset import cross_sectional_demean, filter_regions, region_of
from auto_stock.ml_predictor.evaluation import daily_rank_ic
from auto_stock.ml_predictor.models import FeatureVector, LabeledSample, SequenceDataset, TickerSequence

D0 = date(2025, 1, 1)


def _seq(ticker, market, labels, xs=None):
    xs = xs or [0.0] * len(labels)
    samples = [
        LabeledSample(FeatureVector(ticker=ticker, market=market, date=D0 + timedelta(days=i), values={"x": x}), label=y)
        for i, (x, y) in enumerate(zip(xs, labels))
    ]
    return TickerSequence(ticker=ticker, market=market, samples=samples)


def test_region_of():
    assert region_of("KOSDAQ") == "KR" and region_of("NYSE") == "US"


def test_cross_sectional_demean_subtracts_same_day_mean_within_region_only():
    dataset = SequenceDataset(
        sequences=[_seq("A", "KOSPI", [0.03, 0.0]), _seq("B", "KOSDAQ", [0.01, 0.02]), _seq("C", "NASDAQ", [0.05, -0.01])],
        feature_names=["x"],
    )
    demeaned = cross_sectional_demean(dataset)
    labels = {seq.ticker: [s.label for s in seq.samples] for seq in demeaned.sequences}
    assert labels["A"] == pytest.approx([0.01, -0.01])
    assert labels["B"] == pytest.approx([-0.01, 0.01])
    assert labels["C"] == pytest.approx([0.0, 0.0])  # alone in its region that day
    assert dataset.sequences[0].samples[0].label == 0.03  # original untouched


def test_filter_regions_keeps_requested_regions():
    dataset = SequenceDataset(sequences=[_seq("A", "KOSPI", [0.0]), _seq("C", "NYSE", [0.0])], feature_names=["x"])
    assert [s.ticker for s in filter_regions(dataset, {"KR"}).sequences] == ["A"]


def test_ridge_recovers_a_linear_signal_and_predicts_every_window_point():
    rng = np.random.default_rng(0)
    sequences = []
    for t in range(30):
        xs = rng.normal(size=40).tolist()
        sequences.append(_seq(f"T{t}", "KOSPI", [0.01 * x + rng.normal(scale=0.001) for x in xs], xs))
    dataset = SequenceDataset(sequences=sequences, feature_names=["x"])
    model = fit_ridge(dataset, ["x"], alpha=1.0)
    window = {D0 + timedelta(days=i) for i in range(20, 40)}
    predictions, actuals, dates, tickers = ridge_predictions(model, dataset, ["x"], window)
    assert len(predictions) == 30 * 20 and set(dates) == window
    assert np.corrcoef(predictions, actuals)[0, 1] > 0.9


def test_daily_rank_ic_net_spread_charges_cost_for_replaced_names():
    # 10 tickers, perfect ranking each day; top/bottom 20% = 2 names each
    preds = [[float(i) for i in range(10)]] * 2
    actuals = [[0.01 * i for i in range(10)]] * 2
    same_names = [[f"T{i}" for i in range(10)]] * 2
    summary = daily_rank_ic(preds, actuals, tickers_by_day=same_names, round_trip_cost=0.01)
    gross = 0.01 * 8.5 - 0.01 * 0.5
    # day 1: both legs fully opened (turnover 1+1) → cost 0.02; day 2: nothing replaced → cost 0
    assert summary.mean_top_bottom_spread == pytest.approx(gross)
    assert summary.mean_net_spread == pytest.approx(gross - 0.01)
    assert summary.mean_turnover == pytest.approx(0.5)


def test_daily_rank_ic_without_tickers_keeps_old_behaviour():
    summary = daily_rank_ic([[1.0, 2.0, 3.0] * 4], [[0.1, 0.2, 0.3] * 4])
    assert summary.mean_ic > 0.9
    assert np.isnan(summary.mean_net_spread)


def test_cross_sectional_rank_gauss_orders_within_region_and_day():
    from auto_stock.ml_predictor.dataset import cross_sectional_rank_gauss
    from auto_stock.ml_predictor.models import RANK_TARGET_SCALE

    dataset = SequenceDataset(
        sequences=[
            _seq("A", "KOSPI", [0.30, 0.0]),  # huge outlier day 0 — rank caps its influence
            _seq("B", "KOSDAQ", [0.01, 0.02]),
            _seq("D", "KOSPI", [-0.02, 0.02]),  # tie with B on day 1
            _seq("C", "NASDAQ", [0.05, -0.01]),
        ],
        feature_names=["x"],
    )
    ranked = cross_sectional_rank_gauss(dataset)
    labels = {seq.ticker: [s.label for s in seq.samples] for seq in ranked.sequences}

    assert labels["A"][0] > labels["B"][0] > labels["D"][0]
    assert labels["A"][0] == pytest.approx(-labels["D"][0])  # symmetric around 0, outlier is just "top"
    assert abs(labels["A"][0]) < 3 * RANK_TARGET_SCALE
    assert labels["B"][1] == labels["D"][1] > labels["A"][1]  # ties share the average rank
    assert labels["C"] == pytest.approx([0.0, 0.0])  # alone in its region that day
    assert dataset.sequences[0].samples[0].label == 0.30  # original untouched


def _ic(mean_ic, t_stat):
    from auto_stock.ml_predictor.evaluation import IcSummary

    return IcSummary(n_days=100, mean_ic=mean_ic, ic_t_stat=t_stat, positive_day_share=0.5, mean_top_bottom_spread=0.0)


@pytest.mark.parametrize(
    ("model", "baseline", "passed", "reason_part"),
    [
        ((-0.003, -0.3), (-0.017, -1.2), False, "신호 없음"),  # 2026-10-24 US: was wrongly "사용 가능"
        ((0.0075, 0.74), (0.0376, 3.36), False, "우연과 구분 안 됨"),
        ((0.03, 2.5), (0.04, 3.0), False, "선형 기준선"),
        ((0.05, 3.1), (0.04, 3.0), True, "사용 가능"),
        ((float("nan"), float("nan")), (0.04, 3.0), False, "신호 없음"),
    ],
)
def test_deployment_verdict_requires_positive_significant_ic_above_baseline(model, baseline, passed, reason_part):
    from auto_stock.ml_predictor.evaluation import deployment_verdict

    ok, reason = deployment_verdict(_ic(*model), _ic(*baseline))

    assert ok is passed
    assert reason_part in reason
