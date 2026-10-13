import pytest
import torch

from auto_stock.ml_predictor.dataset import chronological_split
from auto_stock.ml_predictor.models import ModelBundle, ModelMetadata
from auto_stock.ml_predictor.training import (
    RANDOM_STATE,
    MambaRegressor,
    evaluate,
    persistence_baseline_metrics,
    train,
)


def test_random_state_constant_is_zero():
    assert RANDOM_STATE == 0


def test_mamba_regressor_forward_shape():
    model = MambaRegressor(in_dim=16, hidden_dim=8, n_layers=1)
    x = torch.randn(3, 20, 16)

    y = model(x)

    assert y.shape == (3, 20)


def test_mamba_regressor_backward_pass_updates_weights():
    model = MambaRegressor(in_dim=4, hidden_dim=4, n_layers=1)
    x = torch.randn(2, 10, 4)
    before = model.output_proj.weight.clone()

    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer.step()

    assert not torch.allclose(before, model.output_proj.weight)


@pytest.mark.slow
def test_train_fits_mamba_and_returns_bundle_with_metadata(synthetic_sequence_dataset):
    split = chronological_split(synthetic_sequence_dataset, test_ratio=0.2)

    bundle = train(split, universe_size=2, epochs=2, hidden_dim=4, n_layers=1)

    assert isinstance(bundle, ModelBundle)
    assert isinstance(bundle.metadata, ModelMetadata)
    assert bundle.metadata.market == "UNIFIED"
    assert bundle.metadata.algorithm == "mamba"
    assert bundle.metadata.horizon_days == 1
    assert bundle.metadata.model_config == {"in_dim": len(split.train.feature_names), "hidden_dim": 4, "n_layers": 1}
    # Wiring check: the state_dict can rebuild a working model.
    model = MambaRegressor(in_dim=len(split.train.feature_names), hidden_dim=4, n_layers=1)
    model.load_state_dict(bundle.estimator)
    with torch.no_grad():
        out = model(torch.zeros(1, 5, len(split.train.feature_names)))
    assert out.shape == (1, 5)


@pytest.mark.slow
def test_evaluate_reports_mse_rmse_mae_r2(synthetic_sequence_dataset):
    split = chronological_split(synthetic_sequence_dataset, test_ratio=0.2)
    model = MambaRegressor(in_dim=len(split.train.feature_names), hidden_dim=4, n_layers=1)

    metrics = evaluate(model, split.test, split.train.feature_names)

    assert set(metrics.keys()) == {"mse", "rmse", "mae", "r2"}


def test_persistence_baseline_uses_return_1d_feature(synthetic_sequence_dataset):
    # 전체 데이터셋(두 종목 다 포함, split 없이)에 대해 계산해 "표본이 2개 미만"
    # 가드에 걸리지 않도록 한다 — chronological_split 적용 후의 작은 test 세트는
    # 종목이 1개뿐이라 nan을 반환할 수 있다(그 자체가 의도된 방어 동작).
    metrics = persistence_baseline_metrics(synthetic_sequence_dataset, synthetic_sequence_dataset.feature_names)

    assert set(metrics.keys()) == {"mse", "rmse", "mae", "r2"}
    assert metrics["mse"] >= 0.0


def test_persistence_baseline_returns_nan_when_fewer_than_two_sequences():
    import auto_stock.ml_predictor.training as training_module
    from auto_stock.ml_predictor.models import SequenceDataset

    empty = SequenceDataset(sequences=[], feature_names=["return_1d"])
    metrics = persistence_baseline_metrics(empty, empty.feature_names)
    assert metrics["mse"] != metrics["mse"]  # nan != nan


def test_training_module_never_uses_random_shuffle_split():
    import auto_stock.ml_predictor.training as training_module

    assert "train_test_split" not in dir(training_module)
    assert "KFold" not in dir(training_module)
