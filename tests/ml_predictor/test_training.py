from datetime import date, timedelta

import pytest
import torch

import auto_stock.ml_predictor.training as training_module
from auto_stock.ml_predictor.dataset import chronological_split, train_val_test_split
from auto_stock.ml_predictor.models import (
    FeatureVector,
    LabeledSample,
    ModelBundle,
    ModelMetadata,
    SequenceDataset,
    TickerSequence,
)
from auto_stock.ml_predictor.training import (
    RANDOM_STATE,
    MambaRegressor,
    epoch_order,
    fit_feature_normalization,
    prepare_training_batch,
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
def test_train_metadata_scores_every_step_of_the_test_window(synthetic_sequence_dataset):
    # The synthetic test window holds a single ticker: last-step-only scoring had 1 sample -> nan.
    # Scoring every step of the window yields real numbers.
    split = chronological_split(synthetic_sequence_dataset, test_ratio=0.2)

    bundle = train(split, universe_size=2, epochs=1, hidden_dim=4, n_layers=1, device="cpu")

    assert bundle.metadata.test_mse >= 0.0
    assert bundle.metadata.test_mae >= 0.0


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


def test_training_module_never_uses_random_shuffle_split():
    assert "train_test_split" not in dir(training_module)
    assert "KFold" not in dir(training_module)


# --- input normalization / label hygiene / training stability (2026-10-18) ---


def _seq(ticker: str, market: str, rows: list[tuple[float, float]], labels: list[float]) -> TickerSequence:
    start = date(2025, 1, 1)
    samples = [
        LabeledSample(
            feature=FeatureVector(ticker=ticker, market=market, date=start + timedelta(days=i), values={"a": a, "b": b}),
            label=label,
        )
        for i, ((a, b), label) in enumerate(zip(rows, labels))
    ]
    return TickerSequence(ticker=ticker, market=market, samples=samples)


def test_fit_feature_normalization_uses_train_statistics_and_guards_constant_features():
    dataset = SequenceDataset(
        sequences=[_seq("A", "NASDAQ", [(1.0, 5.0), (3.0, 5.0)], [0.0, 0.0]), _seq("B", "NASDAQ", [(5.0, 5.0)], [0.0])],
        feature_names=["a", "b"],
    )

    mean, std = fit_feature_normalization(dataset, ["a", "b"])

    assert mean == pytest.approx([3.0, 5.0])
    assert std[0] == pytest.approx((8 / 3) ** 0.5)
    assert std[1] == 1.0  # constant feature: no division by zero


def test_regressor_standardizes_and_clips_inputs_inside_forward():
    torch.manual_seed(0)
    normalized = MambaRegressor(in_dim=2, hidden_dim=4, n_layers=1)
    normalized.set_normalization(mean=[1.0, 10.0], std=[2.0, 0.5], clip_sigma=5.0)
    plain = MambaRegressor(in_dim=2, hidden_dim=4, n_layers=1)
    plain.load_state_dict({k: v for k, v in normalized.state_dict().items() if not k.startswith("feature_")}, strict=False)
    x = torch.tensor([[[3.0, 100.0], [1.0, 10.0]]])

    expected_input = torch.clamp((x - torch.tensor([1.0, 10.0])) / torch.tensor([2.0, 0.5]), -5.0, 5.0)

    assert torch.allclose(normalized(x), plain(expected_input), atol=1e-6)


def test_legacy_state_dict_without_normalization_buffers_still_loads():
    legacy = {k: v for k, v in MambaRegressor(in_dim=3, hidden_dim=4, n_layers=1).state_dict().items()
              if not k.startswith("feature_")}
    model = MambaRegressor(in_dim=3, hidden_dim=4, n_layers=1)

    model.load_compatible_state_dict(legacy)

    assert torch.equal(model.feature_mean, torch.zeros(3))
    assert torch.equal(model.feature_std, torch.ones(3))
    assert torch.isinf(model.feature_clip_sigma)


def test_prepare_training_batch_clips_labels_and_masks_impossible_krx_moves():
    batch = [
        _seq("001465", "KOSPI", [(0.0, 0.0), (0.0, 0.0)], [-0.914, 0.01]),  # -91% is beyond KRX's ±30% limit
        _seq("MRNA", "NASDAQ", [(0.0, 0.0), (0.0, 0.0)], [1.77, -0.05]),  # real, just extreme
    ]

    _, y, mask = prepare_training_batch(batch, ["a", "b"], torch.device("cpu"), label_clip=0.2)

    assert mask.tolist() == [[0.0, 1.0], [1.0, 1.0]]
    assert y[1].tolist() == pytest.approx([0.2, -0.05])
    assert y[0, 1].item() == pytest.approx(0.01)


def test_epoch_order_is_a_deterministic_permutation_that_changes_per_epoch():
    first = epoch_order(20, epoch=1)

    assert sorted(first) == list(range(20))
    assert epoch_order(20, epoch=1) == first
    assert epoch_order(20, epoch=2) != first


@pytest.mark.slow
def test_train_fits_normalization_into_the_saved_state_dict(synthetic_sequence_dataset):
    split = chronological_split(synthetic_sequence_dataset, test_ratio=0.2)

    bundle = train(split, universe_size=2, epochs=1, hidden_dim=4, n_layers=1, device="cpu")

    assert not torch.equal(bundle.estimator["feature_std"], torch.ones_like(bundle.estimator["feature_std"]))
    assert float(bundle.estimator["feature_clip_sigma"]) == pytest.approx(training_module.DEFAULT_FEATURE_CLIP_SIGMA)


@pytest.mark.slow
def test_train_clips_gradient_norm(synthetic_sequence_dataset, monkeypatch):
    calls = []
    real_clip = training_module.clip_grad_norm_
    monkeypatch.setattr(training_module, "clip_grad_norm_", lambda params, max_norm: calls.append(max_norm) or real_clip(params, max_norm))
    split = chronological_split(synthetic_sequence_dataset, test_ratio=0.2)

    train(split, universe_size=2, epochs=1, hidden_dim=4, n_layers=1, device="cpu", grad_clip=0.5)

    assert calls and all(max_norm == 0.5 for max_norm in calls)
