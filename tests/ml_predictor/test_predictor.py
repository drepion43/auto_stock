from datetime import date, datetime

import pytest
import torch

from auto_stock.ml_predictor.features import FEATURE_NAMES
from auto_stock.ml_predictor.models import FeatureVector, LabeledSample, ModelBundle, ModelMetadata, TickerSequence
from auto_stock.ml_predictor.predictor import DISCLAIMER, predict, to_reasons


def _metadata(feature_names: list[str] = FEATURE_NAMES) -> ModelMetadata:
    return ModelMetadata(
        market="UNIFIED",
        algorithm="mamba",
        horizon_days=1,
        feature_names=feature_names,
        universe_size=1,
        train_start=date(2020, 1, 1),
        train_end=date(2020, 6, 1),
        test_start=date(2020, 6, 8),
        test_end=date(2020, 8, 1),
        train_samples=10,
        test_samples=5,
        test_mse=0.01,
        test_rmse=0.1,
        test_mae=0.08,
        test_r2=0.0,
        model_config={"in_dim": len(feature_names), "hidden_dim": 4, "n_layers": 1},
        trained_at=datetime(2026, 8, 25, 12, 0, 0),
    )


def _sequence(**value_overrides) -> TickerSequence:
    values = {name: 0.1 for name in FEATURE_NAMES}
    values.update(value_overrides)
    feature = FeatureVector(ticker="005930", market="KOSPI", date=date(2026, 8, 25), values=values)
    return TickerSequence(ticker="005930", market="KOSPI", samples=[LabeledSample(feature=feature, label=0.0)])


def _bundle() -> ModelBundle:
    return ModelBundle(estimator={}, metadata=_metadata())


class _StubModel:
    """`build_model`이 반환하는 nn.Module 대역 — forward(x) -> (B, L) 고정 출력."""

    def __init__(self, last_value: float):
        self._last_value = last_value

    def __call__(self, x):
        batch, length, _ = x.shape
        return torch.full((batch, length), self._last_value)


def test_predicted_return_is_passed_through_from_model_output(mocker):
    mocker.patch("auto_stock.ml_predictor.predictor.build_ticker_sequence", return_value=_sequence())
    mocker.patch("auto_stock.ml_predictor.predictor.build_model", return_value=_StubModel(0.012))

    prediction = predict(_bundle(), records=[])

    assert prediction is not None
    assert prediction.predicted_return == pytest.approx(0.012)
    assert prediction.ticker == "005930"
    assert prediction.market == "KOSPI"


def test_high_predicted_return_produces_agreeing_reason_for_buy_candidate(mocker):
    mocker.patch("auto_stock.ml_predictor.predictor.build_ticker_sequence", return_value=_sequence())
    mocker.patch("auto_stock.ml_predictor.predictor.build_model", return_value=_StubModel(0.01))

    prediction = predict(_bundle(), records=[])
    reasons = to_reasons(prediction, action="BUY")

    assert any("동의" in r for r in reasons)


def test_low_predicted_return_produces_conflict_warning_for_buy_candidate(mocker):
    mocker.patch("auto_stock.ml_predictor.predictor.build_ticker_sequence", return_value=_sequence())
    mocker.patch("auto_stock.ml_predictor.predictor.build_model", return_value=_StubModel(-0.01))

    prediction = predict(_bundle(), records=[])
    reasons = to_reasons(prediction, action="BUY")

    assert prediction.predicted_return == pytest.approx(-0.01)
    assert any("상충" in r for r in reasons)


def test_neutral_predicted_return_produces_neutral_reason(mocker):
    mocker.patch("auto_stock.ml_predictor.predictor.build_ticker_sequence", return_value=_sequence())
    mocker.patch("auto_stock.ml_predictor.predictor.build_model", return_value=_StubModel(0.0001))

    prediction = predict(_bundle(), records=[])
    reasons = to_reasons(prediction, action="BUY")

    assert any("중립" in r for r in reasons)


def test_insufficient_records_returns_none_prediction(mocker):
    mocker.patch("auto_stock.ml_predictor.predictor.build_ticker_sequence", return_value=None)
    mock_build_model = mocker.patch("auto_stock.ml_predictor.predictor.build_model")

    prediction = predict(_bundle(), records=[])

    assert prediction is None
    mock_build_model.assert_not_called()


def test_none_prediction_produces_empty_reason_list():
    assert to_reasons(None, action="BUY") == []


def test_reasons_always_include_unvalidated_signal_disclaimer(mocker):
    mocker.patch("auto_stock.ml_predictor.predictor.build_ticker_sequence", return_value=_sequence())
    mocker.patch("auto_stock.ml_predictor.predictor.build_model", return_value=_StubModel(0.0))

    prediction = predict(_bundle(), records=[])
    reasons = to_reasons(prediction, action="BUY")

    assert DISCLAIMER in reasons


def test_top_features_field_no_longer_exists_on_prediction(mocker):
    """Mamba 교체로 계수 기반 설명가능성(top_features)은 제공하지 않는다 — 사용자 승인된
    트레이드오프."""
    mocker.patch("auto_stock.ml_predictor.predictor.build_ticker_sequence", return_value=_sequence())
    mocker.patch("auto_stock.ml_predictor.predictor.build_model", return_value=_StubModel(0.0))

    prediction = predict(_bundle(), records=[])

    assert not hasattr(prediction, "top_features")
