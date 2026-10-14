"""MambaRegressor(PyTorch) 팩토리 + 학습/평가 — MambaStock(`main.py::Net`/`PredictWithData`)의
구조를 가변 길이 다종목 배치로 확장한 버전.

학습 루프는 sklearn `estimator.fit(X, y)` 1줄이 아니라 PyTorch epoch 루프다. 손실은
패딩된 타임스텝을 제외하도록 마스킹한다(`dataset.py::collate_sequences`가 만든
`mask`) — 종목마다 길이가 다른데 짧은 종목의 패딩 구간이 손실에 들어가면 그 종목이
사실상 "수익률 0을 예측하라"는 가짜 레이블로 학습되는 것과 같다."""

import logging
import time
from datetime import datetime

import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from auto_stock.data.market_routing import KRX_MARKETS
from auto_stock.ml_predictor.dataset import collate_sequences
from auto_stock.ml_predictor.labeling import LABEL_HORIZON_DAYS
from auto_stock.ml_predictor.mamba_block import Mamba, MambaConfig
from auto_stock.ml_predictor.models import (
    EpochRecord,
    ModelBundle,
    ModelMetadata,
    SequenceDataset,
    SequenceDatasetSplit,
)

logger = logging.getLogger(__name__)

RANDOM_STATE = 0

DEFAULT_HIDDEN_DIM = 16
DEFAULT_N_LAYERS = 2
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_BATCH_SIZE = 32

# "다음날 수익률 = 전일 수익률 그대로 유지(persistence)" 베이스라인 — regression용
# 하한선 지표. MambaStock 원본엔 벤치마크 알고리즘(rf/dummy류)이 없어 이 지표를
# 그 대체로 둔다(§12 실행 순서 "regression 베이스라인으로 교체").
PERSISTENCE_FEATURE_NAME = "return_1d"


class MambaRegressor(nn.Module):
    """`nn.Linear(in_dim, hidden) -> Mamba -> nn.Linear(hidden, 1)` — MambaStock
    `main.py::Net`과 동일한 구조. `Tanh()`는 두지 않는다(원본은 일별 % 변화량이
    [-1,1] 근방이라 뒀지만, 이 프로젝트의 레이블은 그 가정이 없어 범위를 인위적으로
    제한하지 않는다). 입력 (B, L, in_dim) -> 출력 (B, L) 시점별 예측 수익률."""

    def __init__(self, in_dim: int, hidden_dim: int = DEFAULT_HIDDEN_DIM, n_layers: int = DEFAULT_N_LAYERS):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        config = MambaConfig(d_model=hidden_dim, n_layers=n_layers)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.mamba = Mamba(config)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, in_dim) -> (B, L)
        h = self.input_proj(x)
        h = self.mamba(h)
        return self.output_proj(h).squeeze(-1)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def resolve_device(device: str | None) -> torch.device:
    """`None`/`"auto"`는 CUDA가 있으면 GPU, 없으면 CPU. 명시적으로 `"cuda"`를 요청했는데
    GPU가 없으면 조용히 CPU로 떨어지지 않고 실패한다 — CPU 전용 torch 빌드가 깔린 줄
    모르고 수 시간짜리 학습을 CPU로 돌리는 상황을 막기 위함."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            f"device={device!r}를 요청했지만 CUDA를 사용할 수 없습니다 "
            f"(torch {torch.__version__}). CUDA 빌드 torch가 설치됐는지 확인하세요."
        )
    return resolved


def _masked_mse_loss(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """패딩된 타임스텝(mask=False)을 제외한 MSE. 배치 전체에 유효 타임스텝이 하나도
    없으면(이론상 발생하지 않지만 방어적으로) 0을 반환한다."""
    squared_error = (predicted - target) ** 2 * mask
    valid_count = mask.sum()
    if valid_count == 0:
        return torch.tensor(0.0, requires_grad=True, device=predicted.device)
    return squared_error.sum() / valid_count


def _log_epoch(record: EpochRecord, epochs: int, epoch_seconds: float) -> None:
    val_part = "" if record.val_loss is None else f"  val_loss={record.val_loss:.6e}"
    logger.info(
        "epoch %3d/%d  loss=%.6e%s  lr=%.3e  time=%.2fs  elapsed=%.1fs",
        record.epoch, epochs, record.train_loss, val_part, record.learning_rate, epoch_seconds, record.elapsed_seconds,
    )


def _batches(sequences: list, batch_size: int) -> list[list]:
    return [sequences[i : i + batch_size] for i in range(0, len(sequences), batch_size)]


def _to_tensors(
    batch: list, feature_names: list[str], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X, y, mask = collate_sequences(batch, feature_names)
    return (
        torch.tensor(X, dtype=torch.float32, device=device),
        torch.tensor(y, dtype=torch.float32, device=device),
        torch.tensor(mask, dtype=torch.float32, device=device),
    )


def _train_one_epoch(
    model: MambaRegressor,
    optimizer: torch.optim.Optimizer,
    sequences: list,
    feature_names: list[str],
    batch_size: int,
    device: torch.device,
) -> float:
    """1 epoch 학습 후 패딩 제외 타임스텝 가중평균 train loss를 반환한다. loss는 GPU에서
    누적하고 epoch 끝에 한 번만 꺼낸다(배치마다 GPU 동기화하지 않도록)."""
    model.train()
    weighted_loss_sum = torch.zeros((), device=device)
    valid_steps = torch.zeros((), device=device)
    for batch in _batches(sequences, batch_size):
        x_tensor, y_tensor, mask_tensor = _to_tensors(batch, feature_names, device)
        loss = _masked_mse_loss(model(x_tensor), y_tensor, mask_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_valid = mask_tensor.sum()
        weighted_loss_sum += loss.detach() * batch_valid
        valid_steps += batch_valid
    return float(weighted_loss_sum / valid_steps) if float(valid_steps) > 0 else float("nan")


def _masked_loss_on(
    model: MambaRegressor, dataset: SequenceDataset, feature_names: list[str], batch_size: int, device: torch.device
) -> float:
    """train loss와 같은 정의(패딩 제외 전체 타임스텝 masked MSE)로 계산한 loss — 학습
    곡선에서 train/val을 같은 척도로 비교하기 위함. 가중치는 바꾸지 않는다."""
    model.eval()
    weighted_loss_sum = torch.zeros((), device=device)
    valid_steps = torch.zeros((), device=device)
    with torch.no_grad():
        for batch in _batches(dataset.sequences, batch_size):
            x_tensor, y_tensor, mask_tensor = _to_tensors(batch, feature_names, device)
            batch_valid = mask_tensor.sum()
            weighted_loss_sum += _masked_mse_loss(model(x_tensor), y_tensor, mask_tensor) * batch_valid
            valid_steps += batch_valid
    return float(weighted_loss_sum / valid_steps) if float(valid_steps) > 0 else float("nan")


def train(
    split: SequenceDatasetSplit,
    universe_size: int,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    n_layers: int = DEFAULT_N_LAYERS,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    log_every: int = 0,
) -> ModelBundle:
    """통합(unified) 단일 모델 학습 — market/종목별로 모델을 나누지 않는다(사용자 결정).
    `split.train.sequences`를 종목 단위로 미니배치로 묶어 PyTorch epoch 루프를 돈다.
    `device`는 `resolve_device` 규칙을 따른다(기본: GPU 있으면 GPU). `log_every > 0`이면
    그 간격(과 마지막 epoch)마다 train/val loss·학습률·소요시간을 INFO 로그로 남긴다.
    `split.validation`이 있으면 매 epoch val loss를 계산해 `ModelBundle.history`에 기록한다.
    test 지표는 학습 중에 보지 않는다(그걸 보고 epoch를 고르면 test 결과가 낙관적으로
    왜곡됨) — epoch 선택은 validation으로 한다."""
    _set_seed(RANDOM_STATE)
    torch_device = resolve_device(device)

    feature_names = split.train.feature_names
    in_dim = len(feature_names)
    model = MambaRegressor(in_dim=in_dim, hidden_dim=hidden_dim, n_layers=n_layers).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    validation = split.validation if split.validation is not None and split.validation.sequences else None

    history: list[EpochRecord] = []
    train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = _train_one_epoch(model, optimizer, split.train.sequences, feature_names, batch_size, torch_device)
        val_loss = (
            None if validation is None
            else _masked_loss_on(model, validation, feature_names, batch_size, torch_device)
        )
        now = time.perf_counter()
        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=optimizer.param_groups[0]["lr"],
            elapsed_seconds=now - train_start,
        )
        history.append(record)
        if log_every > 0 and (epoch % log_every == 0 or epoch == epochs):
            _log_epoch(record, epochs, now - epoch_start)

    metrics = evaluate(model, split.test, feature_names)

    train_dates = [s.feature.date for seq in split.train.sequences for s in seq.samples]
    test_dates = [s.feature.date for seq in split.test.sequences for s in seq.samples]

    metadata = ModelMetadata(
        market="UNIFIED",
        algorithm="mamba",
        horizon_days=LABEL_HORIZON_DAYS,
        feature_names=feature_names,
        universe_size=universe_size,
        train_start=min(train_dates),
        train_end=max(train_dates),
        test_start=min(test_dates),
        test_end=max(test_dates),
        train_samples=sum(len(seq.samples) for seq in split.train.sequences),
        test_samples=sum(len(seq.samples) for seq in split.test.sequences),
        test_mse=metrics["mse"],
        test_rmse=metrics["rmse"],
        test_mae=metrics["mae"],
        test_r2=metrics["r2"],
        model_config={"in_dim": in_dim, "hidden_dim": hidden_dim, "n_layers": n_layers},
        trained_at=datetime.now(),
    )
    # 추론은 CPU 전용 환경에서도 돌아야 하므로 GPU 학습이어도 CPU 텐서로 저장한다.
    cpu_state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    return ModelBundle(estimator=cpu_state_dict, metadata=metadata, history=tuple(history))


def _nan_metrics() -> dict[str, float]:
    return {key: float("nan") for key in ("mse", "rmse", "mae", "r2", "hit_rate")}


def _hit_rate(actuals: list[float], predictions: list[float]) -> float:
    """방향 적중률 — 예측 수익률과 실제 수익률의 부호가 같은 비율. 실제 수익률이 정확히
    0인 표본은 방향이 없으므로 분모에서 뺀다(예측 0은 어느 방향도 맞히지 못한 것으로 침)."""
    pairs = [(a, p) for a, p in zip(actuals, predictions) if a != 0]
    if not pairs:
        return float("nan")
    return sum(1 for a, p in pairs if (a > 0) == (p > 0) and p != 0) / len(pairs)


def _regression_metrics(actuals: list[float], predictions: list[float]) -> dict[str, float]:
    if len(actuals) < 2:
        return _nan_metrics()
    mse = mean_squared_error(actuals, predictions)
    return {
        "mse": float(mse),
        "rmse": float(mse**0.5),
        "mae": float(mean_absolute_error(actuals, predictions)),
        "r2": float(r2_score(actuals, predictions)),
        "hit_rate": _hit_rate(actuals, predictions),
    }


def split_by_region(dataset: SequenceDataset) -> dict[str, SequenceDataset]:
    """시장별 성능 비교용 — 국내(KOSPI/KOSDAQ)와 미국(NASDAQ/NYSE)으로 종목 시퀀스를 나눈다."""
    domestic = [seq for seq in dataset.sequences if seq.market in KRX_MARKETS]
    us = [seq for seq in dataset.sequences if seq.market not in KRX_MARKETS]
    return {
        "domestic": SequenceDataset(sequences=domestic, feature_names=dataset.feature_names),
        "us": SequenceDataset(sequences=us, feature_names=dataset.feature_names),
    }


def evaluate(model: MambaRegressor, dataset: SequenceDataset, feature_names: list[str]) -> dict[str, float]:
    """시퀀스별로 마지막 시점의 예측만 평가에 쓴다 — `predict()`(추론 경로)가 매 호출마다
    "그 시점까지의 전체 과거"를 받아 마지막 예측만 꺼내는 것과 동일한 평가 조건을 만들기
    위함(패딩된 중간 시점을 평가에 섞지 않음)."""
    if not dataset.sequences:
        return _nan_metrics()

    model.eval()
    model_device = next(model.parameters()).device
    predictions: list[float] = []
    actuals: list[float] = []
    with torch.no_grad():
        for seq in dataset.sequences:
            rows = [[s.feature.values[name] for name in feature_names] for s in seq.samples]
            x = torch.tensor([rows], dtype=torch.float32, device=model_device)
            predicted = model(x)[0]
            predictions.append(float(predicted[-1]))
            actuals.append(seq.samples[-1].label)

    return _regression_metrics(actuals, predictions)


def persistence_baseline_metrics(dataset: SequenceDataset, feature_names: list[str]) -> dict[str, float]:
    """"다음날 수익률 = 어제의 1일 수익률(return_1d) 그대로" 베이스라인. 모델이 이 단순
    규칙보다 나은지 가늠하는 하한선 — MambaStock 원본엔 없는 비교 지표지만, 기존
    REPORT_ALGORITHMS(rf/dummy)가 하던 "하한선 리포트" 역할을 regression에서 대신한다."""
    if PERSISTENCE_FEATURE_NAME not in feature_names:
        return _nan_metrics()

    predictions: list[float] = []
    actuals: list[float] = []
    for seq in dataset.sequences:
        sample = seq.samples[-1]
        predictions.append(sample.feature.values[PERSISTENCE_FEATURE_NAME])
        actuals.append(sample.label)

    return _regression_metrics(actuals, predictions)
