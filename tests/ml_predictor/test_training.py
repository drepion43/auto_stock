from datetime import date

import pytest
import torch

from auto_stock.ml_predictor.dataset import chronological_split, train_val_test_split
from auto_stock.ml_predictor.features import FEATURE_NAMES
from auto_stock.ml_predictor.models import (
    FeatureVector,
    LabeledSample,
    ModelBundle,
    ModelMetadata,
    SequenceDataset,
    TickerSequence,
)
from auto_stock.ml_predictor.training import (
    split_by_region,
    RANDOM_STATE,
    MambaRegressor,
    evaluate,
    persistence_baseline_metrics,
    resolve_device,
    train,
)

CUDA_AVAILABLE = torch.cuda.is_available()


def test_resolve_device_auto_falls_back_to_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device(None) == torch.device("cpu")
    assert resolve_device("auto") == torch.device("cpu")


def test_resolve_device_auto_picks_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device("auto") == torch.device("cuda")


def test_resolve_device_explicit_cuda_without_gpu_raises(monkeypatch):
    # An explicit GPU request must fail loudly instead of silently training on CPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA"):
        resolve_device("cuda")


def test_resolve_device_explicit_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


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
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"))],
)
def test_train_returns_cpu_state_dict_on_any_device(synthetic_sequence_dataset, device):
    # The artifact is loaded on CPU-only inference machines, so it must never hold CUDA tensors.
    split = chronological_split(synthetic_sequence_dataset, test_ratio=0.2)

    bundle = train(split, universe_size=2, epochs=1, hidden_dim=4, n_layers=1, device=device)

    assert all(t.device.type == "cpu" for t in bundle.estimator.values())


@pytest.mark.slow
def test_train_logs_epoch_loss_and_learning_rate(synthetic_sequence_dataset, caplog):
    split = chronological_split(synthetic_sequence_dataset, test_ratio=0.2)

    with caplog.at_level("INFO", logger="auto_stock.ml_predictor.training"):
        train(split, universe_size=2, epochs=3, hidden_dim=4, n_layers=1, learning_rate=0.005, device="cpu", log_every=1)

    epoch_lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("epoch")]
    assert len(epoch_lines) == 3
    assert epoch_lines[0].startswith("epoch   1/3")
    assert "loss=" in epoch_lines[0]
    assert "lr=5.000e-03" in epoch_lines[0]


@pytest.mark.slow
def test_train_log_every_controls_frequency_and_always_logs_last_epoch(synthetic_sequence_dataset, caplog):
    split = chronological_split(synthetic_sequence_dataset, test_ratio=0.2)

    with caplog.at_level("INFO", logger="auto_stock.ml_predictor.training"):
        train(split, universe_size=2, epochs=5, hidden_dim=4, n_layers=1, device="cpu", log_every=2)

    epochs_logged = [r.getMessage().split()[1] for r in caplog.records if r.getMessage().startswith("epoch")]
    assert epochs_logged == ["2/5", "4/5", "5/5"]


@pytest.mark.slow
def test_train_is_silent_when_log_every_is_zero(synthetic_sequence_dataset, caplog):
    split = chronological_split(synthetic_sequence_dataset, test_ratio=0.2)

    with caplog.at_level("INFO", logger="auto_stock.ml_predictor.training"):
        train(split, universe_size=2, epochs=2, hidden_dim=4, n_layers=1, device="cpu", log_every=0)

    assert not [r for r in caplog.records if r.getMessage().startswith("epoch")]


@pytest.mark.slow
def test_evaluate_reports_mse_rmse_mae_r2(synthetic_sequence_dataset):
    split = chronological_split(synthetic_sequence_dataset, test_ratio=0.2)
    model = MambaRegressor(in_dim=len(split.train.feature_names), hidden_dim=4, n_layers=1)

    metrics = evaluate(model, split.test, split.train.feature_names)

    assert set(metrics.keys()) == {"mse", "rmse", "mae", "r2", "hit_rate"}
    # The small test window holds a single ticker (-> nan by the <2 guard); use both tickers for the range check.
    full_metrics = evaluate(model, synthetic_sequence_dataset, split.train.feature_names)
    assert 0.0 <= full_metrics["hit_rate"] <= 1.0


@pytest.mark.slow
def test_train_records_train_and_val_loss_history_when_validation_given(synthetic_sequence_dataset):
    split = train_val_test_split(synthetic_sequence_dataset, val_ratio=0.1, test_ratio=0.2)

    bundle = train(split, universe_size=2, epochs=3, hidden_dim=4, n_layers=1, learning_rate=0.005, device="cpu")

    assert [h.epoch for h in bundle.history] == [1, 2, 3]
    assert all(h.train_loss >= 0 and h.val_loss is not None and h.val_loss >= 0 for h in bundle.history)
    assert all(h.learning_rate == 0.005 for h in bundle.history)


@pytest.mark.slow
def test_train_history_has_no_val_loss_without_validation(synthetic_sequence_dataset):
    split = chronological_split(synthetic_sequence_dataset, test_ratio=0.2)

    bundle = train(split, universe_size=2, epochs=2, hidden_dim=4, n_layers=1, device="cpu")

    assert len(bundle.history) == 2
    assert all(h.val_loss is None for h in bundle.history)


@pytest.mark.slow
def test_train_log_line_includes_val_loss_when_validation_given(synthetic_sequence_dataset, caplog):
    split = train_val_test_split(synthetic_sequence_dataset, val_ratio=0.1, test_ratio=0.2)

    with caplog.at_level("INFO", logger="auto_stock.ml_predictor.training"):
        train(split, universe_size=2, epochs=1, hidden_dim=4, n_layers=1, device="cpu", log_every=1)

    assert any("val_loss=" in r.getMessage() for r in caplog.records)


def _sequence(ticker: str, market: str, return_1d: float, label: float) -> TickerSequence:
    values = {name: 0.0 for name in FEATURE_NAMES} | {"return_1d": return_1d}
    feature = FeatureVector(ticker=ticker, market=market, date=date(2024, 1, 1), values=values)
    return TickerSequence(ticker=ticker, market=market, samples=[LabeledSample(feature=feature, label=label)])


def test_split_by_region_groups_domestic_and_us_tickers():
    dataset = SequenceDataset(
        sequences=[
            _sequence("005930", "KOSPI", 0.01, 0.02),
            _sequence("035720", "KOSDAQ", 0.01, 0.02),
            _sequence("AAPL", "NASDAQ", 0.01, 0.02),
            _sequence("JPM", "NYSE", 0.01, 0.02),
        ],
        feature_names=FEATURE_NAMES,
    )

    regions = split_by_region(dataset)

    assert {seq.ticker for seq in regions["domestic"].sequences} == {"005930", "035720"}
    assert {seq.ticker for seq in regions["us"].sequences} == {"AAPL", "JPM"}


def test_persistence_baseline_hit_rate_counts_sign_agreement():
    # return_1d is the baseline's prediction: 3 of 4 signs agree with the label.
    dataset = SequenceDataset(
        sequences=[
            _sequence("A", "KOSPI", 0.01, 0.02),
            _sequence("B", "KOSPI", -0.01, -0.03),
            _sequence("C", "KOSPI", 0.02, 0.01),
            _sequence("D", "KOSPI", 0.01, -0.01),
        ],
        feature_names=FEATURE_NAMES,
    )

    metrics = persistence_baseline_metrics(dataset, FEATURE_NAMES)

    assert metrics["hit_rate"] == pytest.approx(0.75)


def test_persistence_baseline_uses_return_1d_feature(synthetic_sequence_dataset):
    # 전체 데이터셋(두 종목 다 포함, split 없이)에 대해 계산해 "표본이 2개 미만"
    # 가드에 걸리지 않도록 한다 — chronological_split 적용 후의 작은 test 세트는
    # 종목이 1개뿐이라 nan을 반환할 수 있다(그 자체가 의도된 방어 동작).
    metrics = persistence_baseline_metrics(synthetic_sequence_dataset, synthetic_sequence_dataset.feature_names)

    assert set(metrics.keys()) == {"mse", "rmse", "mae", "r2", "hit_rate"}
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
