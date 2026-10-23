from datetime import date, datetime

import pytest
import torch

from auto_stock.ml_predictor.dataset import build_ticker_sequence
from auto_stock.ml_predictor.features import FEATURE_NAMES
from auto_stock.ml_predictor.models import TARGET_DEMEANED, TARGET_RAW, FeatureVector, ModelBundle, ModelMetadata
from auto_stock.ml_predictor.predictor import DISCLAIMER, describe_prediction, predict, to_reasons

from .conftest import make_records, random_walk_prices


def _metadata(feature_names: list[str] = FEATURE_NAMES, target: str = TARGET_RAW) -> ModelMetadata:
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
        target=target,
    )


def _features(**value_overrides) -> list[FeatureVector]:
    values = {name: 0.1 for name in FEATURE_NAMES}
    values.update(value_overrides)
    return [FeatureVector(ticker="005930", market="KOSPI", date=date(2026, 8, 25), values=values)]


def _bundle(target: str = TARGET_RAW) -> ModelBundle:
    return ModelBundle(estimator={}, metadata=_metadata(target=target))


class _StubModel:
    """`build_model`이 반환하는 nn.Module 대역 — forward(x) -> (B, L) 고정 출력."""

    def __init__(self, last_value: float):
        self._last_value = last_value

    def __call__(self, x):
        batch, length, _ = x.shape
        return torch.full((batch, length), self._last_value)


def test_predicted_return_is_passed_through_from_model_output(mocker):
    mocker.patch("auto_stock.ml_predictor.predictor.build_inference_features", return_value=_features())
    mocker.patch("auto_stock.ml_predictor.predictor.build_model", return_value=_StubModel(0.012))

    prediction = predict(_bundle(), records=[])

    assert prediction is not None
    assert prediction.predicted_return == pytest.approx(0.012)
    assert prediction.ticker == "005930"
    assert prediction.market == "KOSPI"


def test_high_predicted_return_produces_agreeing_reason_for_buy_candidate(mocker):
    mocker.patch("auto_stock.ml_predictor.predictor.build_inference_features", return_value=_features())
    mocker.patch("auto_stock.ml_predictor.predictor.build_model", return_value=_StubModel(0.01))

    prediction = predict(_bundle(), records=[])
    reasons = to_reasons(prediction, action="BUY")

    assert any("동의" in r for r in reasons)


def test_low_predicted_return_produces_conflict_warning_for_buy_candidate(mocker):
    mocker.patch("auto_stock.ml_predictor.predictor.build_inference_features", return_value=_features())
    mocker.patch("auto_stock.ml_predictor.predictor.build_model", return_value=_StubModel(-0.01))

    prediction = predict(_bundle(), records=[])
    reasons = to_reasons(prediction, action="BUY")

    assert prediction.predicted_return == pytest.approx(-0.01)
    assert any("상충" in r for r in reasons)


def test_neutral_predicted_return_produces_neutral_reason(mocker):
    mocker.patch("auto_stock.ml_predictor.predictor.build_inference_features", return_value=_features())
    mocker.patch("auto_stock.ml_predictor.predictor.build_model", return_value=_StubModel(0.0001))

    prediction = predict(_bundle(), records=[])
    reasons = to_reasons(prediction, action="BUY")

    assert any("중립" in r for r in reasons)


def test_insufficient_records_returns_none_prediction(mocker):
    mocker.patch("auto_stock.ml_predictor.predictor.build_inference_features", return_value=[])
    mock_build_model = mocker.patch("auto_stock.ml_predictor.predictor.build_model")

    prediction = predict(_bundle(), records=[])

    assert prediction is None
    mock_build_model.assert_not_called()


def test_none_prediction_produces_empty_reason_list():
    assert to_reasons(None, action="BUY") == []


def test_reasons_always_include_unvalidated_signal_disclaimer(mocker):
    mocker.patch("auto_stock.ml_predictor.predictor.build_inference_features", return_value=_features())
    mocker.patch("auto_stock.ml_predictor.predictor.build_model", return_value=_StubModel(0.0))

    prediction = predict(_bundle(), records=[])
    reasons = to_reasons(prediction, action="BUY")

    assert DISCLAIMER in reasons


def test_top_features_field_no_longer_exists_on_prediction(mocker):
    """Mamba 교체로 계수 기반 설명가능성(top_features)은 제공하지 않는다 — 사용자 승인된
    트레이드오프."""
    mocker.patch("auto_stock.ml_predictor.predictor.build_inference_features", return_value=_features())
    mocker.patch("auto_stock.ml_predictor.predictor.build_model", return_value=_StubModel(0.0))

    prediction = predict(_bundle(), records=[])

    assert not hasattr(prediction, "top_features")


def test_prediction_is_based_on_the_latest_record_not_the_day_before(mocker):
    # 학습 시퀀스는 레이블 없는 마지막 날을 버린다 — 추론까지 그러면 오늘 종가가 빠진 하루 전 기준 예측이 된다.
    records = make_records(random_walk_prices(90))
    seen_lengths: list[int] = []

    class _Recording(_StubModel):
        def __call__(self, x):
            seen_lengths.append(x.shape[1])
            return super().__call__(x)

    mocker.patch("auto_stock.ml_predictor.predictor.build_model", return_value=_Recording(0.0))

    prediction = predict(_bundle(), records)

    assert prediction is not None
    assert prediction.date == records[-1].date
    labeled = build_ticker_sequence(records)
    assert seen_lengths == [len(labeled.samples) + 1]  # 학습 시퀀스 + 레이블 없는 마지막 날 1행


def test_prediction_carries_the_training_target_from_metadata(mocker):
    mocker.patch("auto_stock.ml_predictor.predictor.build_inference_features", return_value=_features())
    mocker.patch("auto_stock.ml_predictor.predictor.build_model", return_value=_StubModel(0.01))

    assert predict(_bundle(TARGET_DEMEANED), records=[]).target == TARGET_DEMEANED
    assert predict(_bundle(), records=[]).target == TARGET_RAW


def test_demeaned_prediction_is_described_as_excess_return_not_absolute(mocker):
    mocker.patch("auto_stock.ml_predictor.predictor.build_inference_features", return_value=_features())
    mocker.patch("auto_stock.ml_predictor.predictor.build_model", return_value=_StubModel(0.01))
    prediction = predict(_bundle(TARGET_DEMEANED), records=[])

    payload = describe_prediction(prediction)
    reasons = to_reasons(prediction, action="BUY")

    assert payload["return_basis"] == "excess_vs_region_mean"
    assert payload["as_of"] == "2026-08-25"
    assert "평균" in payload["interpretation"] and "절대" in payload["interpretation"]
    assert "시장 평균 대비 예상 초과 수익률 +1.00%" in reasons[0]


def test_raw_prediction_is_described_as_absolute_return(mocker):
    mocker.patch("auto_stock.ml_predictor.predictor.build_inference_features", return_value=_features())
    mocker.patch("auto_stock.ml_predictor.predictor.build_model", return_value=_StubModel(0.01))
    prediction = predict(_bundle(), records=[])

    assert describe_prediction(prediction)["return_basis"] == "absolute"
    assert to_reasons(prediction, action="BUY")[0].startswith("ML 모델(예상 수익률 +1.00%)")
