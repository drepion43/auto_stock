import json
from datetime import date, datetime

import pytest
import torch

from auto_stock.ml_predictor.artifact import DEFAULT_MODEL_DIR, ModelArtifactError, artifact_path, load_model, save_model
from auto_stock.ml_predictor.models import ModelBundle, ModelMetadata
from auto_stock.ml_predictor.training import MambaRegressor


def _metadata(**overrides) -> ModelMetadata:
    defaults = dict(
        market="UNIFIED",
        algorithm="mamba",
        horizon_days=1,
        feature_names=["return_1d", "return_5d"],
        universe_size=2,
        train_start=date(2020, 1, 1),
        train_end=date(2020, 6, 1),
        test_start=date(2020, 6, 8),
        test_end=date(2020, 8, 1),
        train_samples=100,
        test_samples=20,
        test_mse=0.01,
        test_rmse=0.1,
        test_mae=0.08,
        test_r2=0.02,
        model_config={"in_dim": 2, "hidden_dim": 4, "n_layers": 1},
        trained_at=datetime(2026, 8, 25, 12, 0, 0),
    )
    defaults.update(overrides)
    return ModelMetadata(**defaults)


def _fitted_bundle() -> ModelBundle:
    model = MambaRegressor(in_dim=2, hidden_dim=4, n_layers=1)
    return ModelBundle(estimator=model.state_dict(), metadata=_metadata())


def test_default_model_dir_is_models_ml_predictor():
    assert str(DEFAULT_MODEL_DIR).replace("\\", "/") == "models/ml_predictor"


def test_artifact_path_format(tmp_path):
    path = artifact_path(model_dir=tmp_path)
    assert path.name == "unified_h1_mamba.pt"
    assert path.parent == tmp_path


def test_save_model_writes_pt_and_metadata_sidecar(tmp_path):
    bundle = _fitted_bundle()

    path = save_model(bundle, model_dir=tmp_path)

    assert path.exists()
    metadata_path = path.with_suffix(".metadata.json")
    assert metadata_path.exists()
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert raw["market"] == "UNIFIED"
    assert raw["algorithm"] == "mamba"
    assert raw["test_mse"] == pytest.approx(0.01)


def test_load_model_round_trips_estimator_and_metadata(tmp_path):
    bundle = _fitted_bundle()
    save_model(bundle, model_dir=tmp_path)

    loaded = load_model(model_dir=tmp_path)

    assert loaded.metadata.market == "UNIFIED"
    assert loaded.metadata.algorithm == "mamba"
    assert loaded.metadata.train_start == date(2020, 1, 1)
    assert loaded.metadata.trained_at == datetime(2026, 8, 25, 12, 0, 0)
    assert loaded.metadata.model_config == {"in_dim": 2, "hidden_dim": 4, "n_layers": 1}

    model = MambaRegressor(**loaded.metadata.model_config)
    model.load_state_dict(loaded.estimator)
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 2))
    assert out.shape == (1, 3)


def test_load_model_raises_helpful_error_when_artifact_missing(tmp_path):
    with pytest.raises(ModelArtifactError, match="scripts/train_ml_model.py"):
        load_model(model_dir=tmp_path)


def test_training_target_round_trips_through_metadata(tmp_path):
    bundle = ModelBundle(estimator=MambaRegressor(in_dim=2, hidden_dim=4, n_layers=1).state_dict(), metadata=_metadata(target="demeaned"))
    save_model(bundle, model_dir=tmp_path)

    assert load_model(model_dir=tmp_path).metadata.target == "demeaned"


def test_metadata_without_target_loads_as_raw(tmp_path):
    # target 필드 도입 전 아티팩트는 모두 원래 다음날 수익률(raw)로 학습됐다.
    path = save_model(_fitted_bundle(), model_dir=tmp_path)
    metadata_path = path.with_suffix(".metadata.json")
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    del raw["target"]
    metadata_path.write_text(json.dumps(raw), encoding="utf-8")

    assert load_model(model_dir=tmp_path).metadata.target == "raw"
