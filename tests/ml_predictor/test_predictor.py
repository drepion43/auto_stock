from datetime import date, datetime

import pytest

from auto_stock.ml_predictor.features import FEATURE_NAMES
from auto_stock.ml_predictor.models import FeatureVector, ModelBundle, ModelMetadata
from auto_stock.ml_predictor.predictor import DISCLAIMER, predict, to_reasons


def _metadata(feature_names: list[str] = FEATURE_NAMES) -> ModelMetadata:
    return ModelMetadata(
        market="KRX",
        algorithm="logreg",
        horizon_days=5,
        feature_names=feature_names,
        universe_size=1,
        train_start=date(2020, 1, 1),
        train_end=date(2020, 6, 1),
        test_start=date(2020, 6, 8),
        test_end=date(2020, 8, 1),
        train_samples=10,
        test_samples=5,
        test_base_rate=0.5,
        test_roc_auc=0.6,
        test_accuracy=0.55,
        trained_at=datetime(2026, 8, 25, 12, 0, 0),
    )


def _feature_vector(**value_overrides) -> FeatureVector:
    values = {name: 0.1 for name in FEATURE_NAMES}
    values.update(value_overrides)
    return FeatureVector(ticker="005930", market="KRX", date=date(2026, 8, 25), values=values)


def _bundle_with_estimator(estimator) -> ModelBundle:
    return ModelBundle(estimator=estimator, metadata=_metadata())


def test_high_probability_produces_agreeing_reason_for_buy_candidate(mocker):
    mocker.patch("auto_stock.ml_predictor.predictor.latest_feature_vector", return_value=_feature_vector())
    estimator = mocker.Mock()
    estimator.predict_proba.return_value = [[0.2, 0.8]]
    estimator.named_steps = {}
    bundle = _bundle_with_estimator(estimator)

    prediction = predict(bundle, records=[])
    reasons = to_reasons(prediction, action="BUY")

    assert prediction is not None
    assert prediction.probability_up == pytest.approx(0.8)
    assert any("동의" in r for r in reasons)


def test_low_probability_produces_conflict_warning_for_buy_candidate(mocker):
    mocker.patch("auto_stock.ml_predictor.predictor.latest_feature_vector", return_value=_feature_vector())
    estimator = mocker.Mock()
    estimator.predict_proba.return_value = [[0.9, 0.1]]
    estimator.named_steps = {}
    bundle = _bundle_with_estimator(estimator)

    prediction = predict(bundle, records=[])
    reasons = to_reasons(prediction, action="BUY")

    assert prediction.probability_up == pytest.approx(0.1)
    assert any("상충" in r for r in reasons)


def test_insufficient_records_returns_none_prediction(mocker):
    mocker.patch("auto_stock.ml_predictor.predictor.latest_feature_vector", return_value=None)
    estimator = mocker.Mock()
    bundle = _bundle_with_estimator(estimator)

    prediction = predict(bundle, records=[])

    assert prediction is None
    estimator.predict_proba.assert_not_called()


def test_none_prediction_produces_empty_reason_list():
    assert to_reasons(None, action="BUY") == []


def test_reasons_include_top_contributing_features(mocker):
    n = len(FEATURE_NAMES)
    vector = _feature_vector(return_1d=0.5, return_5d=-0.5)
    mocker.patch("auto_stock.ml_predictor.predictor.latest_feature_vector", return_value=vector)

    scaler = mocker.Mock()
    standardized_row = [0.5, -0.5] + [0.0] * (n - 2)
    scaler.transform.return_value = [standardized_row]
    clf = mocker.Mock()
    clf.coef_ = [[3.0, -2.0] + [0.0] * (n - 2)]  # |3.0*0.5|=1.5 > |-2.0*-0.5|=1.0

    estimator = mocker.Mock()
    estimator.predict_proba.return_value = [[0.4, 0.6]]
    estimator.named_steps = {"scaler": scaler, "clf": clf}
    bundle = _bundle_with_estimator(estimator)

    prediction = predict(bundle, records=[])
    reasons = to_reasons(prediction, action="BUY")

    assert prediction.top_features[0][0] == "return_1d"
    assert any("return_1d" in r for r in reasons)


def test_reasons_include_unvalidated_signal_disclaimer(mocker):
    mocker.patch("auto_stock.ml_predictor.predictor.latest_feature_vector", return_value=_feature_vector())
    estimator = mocker.Mock()
    estimator.predict_proba.return_value = [[0.5, 0.5]]
    estimator.named_steps = {}
    bundle = _bundle_with_estimator(estimator)

    prediction = predict(bundle, records=[])
    reasons = to_reasons(prediction, action="BUY")

    assert DISCLAIMER in reasons
