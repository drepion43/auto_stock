import json
from datetime import date, datetime

import pytest

from auto_stock.ml_predictor.artifact import DEFAULT_MODEL_DIR, ModelArtifactError, artifact_path, load_model, save_model
from auto_stock.ml_predictor.models import ModelBundle, ModelMetadata
from auto_stock.ml_predictor.training import build_estimator


def _metadata(**overrides) -> ModelMetadata:
    defaults = dict(
        market="KRX",
        algorithm="logreg",
        horizon_days=5,
        feature_names=["return_1d", "return_5d"],
        universe_size=2,
        train_start=date(2020, 1, 1),
        train_end=date(2020, 6, 1),
        test_start=date(2020, 6, 8),
        test_end=date(2020, 8, 1),
        train_samples=100,
        test_samples=20,
        test_base_rate=0.5,
        test_roc_auc=0.61,
        test_accuracy=0.58,
        trained_at=datetime(2026, 8, 25, 12, 0, 0),
    )
    defaults.update(overrides)
    return ModelMetadata(**defaults)


def _fitted_bundle(market="KRX") -> ModelBundle:
    estimator = build_estimator("logreg")
    estimator.fit([[0.1, 0.2], [0.3, -0.1], [0.2, 0.1], [-0.1, 0.4]], [1, 0, 1, 0])
    return ModelBundle(estimator=estimator, metadata=_metadata(market=market))


def test_default_model_dir_is_models_ml_predictor():
    assert str(DEFAULT_MODEL_DIR).replace("\\", "/") == "models/ml_predictor"


def test_artifact_path_format(tmp_path):
    path = artifact_path("KRX", model_dir=tmp_path)
    assert path.name == "KRX_h5_logreg.joblib"
    assert path.parent == tmp_path


def test_save_model_writes_joblib_and_metadata_sidecar(tmp_path):
    bundle = _fitted_bundle()

    path = save_model(bundle, model_dir=tmp_path)

    assert path.exists()
    metadata_path = path.with_suffix(".metadata.json")
    assert metadata_path.exists()
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert raw["market"] == "KRX"
    assert raw["algorithm"] == "logreg"
    assert raw["test_roc_auc"] == pytest.approx(0.61)


def test_load_model_round_trips_estimator_and_metadata(tmp_path):
    bundle = _fitted_bundle()
    save_model(bundle, model_dir=tmp_path)

    loaded = load_model("KRX", model_dir=tmp_path)

    assert loaded.metadata.market == "KRX"
    assert loaded.metadata.algorithm == "logreg"
    assert loaded.metadata.train_start == date(2020, 1, 1)
    assert loaded.metadata.trained_at == datetime(2026, 8, 25, 12, 0, 0)
    proba = loaded.estimator.predict_proba([[0.1, 0.2]])
    assert proba.shape == (1, 2)


def test_load_model_raises_helpful_error_when_artifact_missing(tmp_path):
    with pytest.raises(ModelArtifactError, match="scripts/train_ml_model.py"):
        load_model("KRX", model_dir=tmp_path)
