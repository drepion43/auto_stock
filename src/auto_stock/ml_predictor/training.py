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
from torch.nn.utils import clip_grad_norm_

from auto_stock.data.market_routing import KRX_MARKETS
from auto_stock.ml_predictor.dataset import collate_sequences
from auto_stock.ml_predictor.evaluation import evaluate_window
from auto_stock.ml_predictor.labeling import LABEL_HORIZON_DAYS
from auto_stock.ml_predictor.mamba_block import Mamba, MambaConfig
from auto_stock.ml_predictor.models import (
    EpochRecord,
    ModelBundle,
    ModelMetadata,
    TARGET_RAW,
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

# 2026-10-18 데이터 진단(docs/_review_parts/03_ml_predictor.md §11): 피처 스케일이 제각각
# (수익률 std≈0.03 vs sales_yield std≈2.3·최대 41.7, pbr 최대 50)이고 레이블 꼬리가 두꺼워
# (|r|>20%가 표본의 0.1%인데 MSE의 13.6%) 학습이 epoch ~50에서 반복적으로 발산했다.
DEFAULT_FEATURE_CLIP_SIGMA = 5.0  # 표준화 후 ±5σ로 자른다
DEFAULT_LABEL_CLIP = 0.2  # 학습 손실에서만 레이블을 ±20%로 자른다(평가는 원래 레이블)
DEFAULT_GRAD_CLIP = 1.0  # gradient norm 상한
KRX_PRICE_LIMIT = 0.3001  # 국내 일일 가격제한폭 ±30% — 넘는 레이블은 수정주가 미반영 등 데이터 오류
_NORMALIZATION_BUFFERS = ("feature_mean", "feature_std", "feature_clip_sigma")


class MambaRegressor(nn.Module):
    """`normalize -> nn.Linear(in_dim, hidden) -> Mamba -> nn.Linear(hidden, 1)` — MambaStock
    `main.py::Net`과 같은 구조에 입력 표준화를 앞에 붙였다. `Tanh()`는 두지 않는다(원본은 일별 %
    변화량이 [-1,1] 근방이라 뒀지만, 이 프로젝트의 레이블은 그 가정이 없다). 입력 (B, L, in_dim)
    -> 출력 (B, L) 시점별 예측 수익률.

    표준화 통계(`feature_mean`/`feature_std`/`feature_clip_sigma`)는 학습 때 train 구간으로 맞춘
    **버퍼**라 state_dict에 함께 저장된다 — 학습·평가·추론이 별도 처리 없이 같은 변환을 쓴다.
    기본값(평균 0, 표준편차 1, 자르기 없음)은 항등 변환이다."""

    def __init__(self, in_dim: int, hidden_dim: int = DEFAULT_HIDDEN_DIM, n_layers: int = DEFAULT_N_LAYERS):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.register_buffer("feature_mean", torch.zeros(in_dim))
        self.register_buffer("feature_std", torch.ones(in_dim))
        self.register_buffer("feature_clip_sigma", torch.tensor(float("inf")))
        config = MambaConfig(d_model=hidden_dim, n_layers=n_layers)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.mamba = Mamba(config)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def set_normalization(self, mean: list[float], std: list[float], clip_sigma: float) -> None:
        with torch.no_grad():
            self.feature_mean.copy_(torch.tensor(mean, dtype=torch.float32))
            self.feature_std.copy_(torch.tensor(std, dtype=torch.float32))
            self.feature_clip_sigma.fill_(clip_sigma)

    def load_compatible_state_dict(self, state_dict: dict) -> None:
        """표준화 버퍼가 없는 예전(2026-10-18 이전) state_dict도 받는다 — 빠진 버퍼는 항등 변환
        기본값으로 채우고, 그 외 키는 엄격하게 검사한다."""
        defaults = {name: buffer for name, buffer in self.state_dict().items() if name in _NORMALIZATION_BUFFERS}
        self.load_state_dict({**defaults, **state_dict})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, in_dim) -> (B, L)
        x = (x - self.feature_mean) / self.feature_std
        x = torch.clamp(x, min=-self.feature_clip_sigma, max=self.feature_clip_sigma)
        h = self.input_proj(x)
        h = self.mamba(h)
        return self.output_proj(h).squeeze(-1)


def fit_feature_normalization(dataset: SequenceDataset, feature_names: list[str]) -> tuple[list[float], list[float]]:
    """train 구간 전체 시점의 피처별 평균·표준편차(모집단). 상수 피처는 표준편차 1로 둔다(0 나눗셈 방지)."""
    values = torch.tensor(
        [[s.feature.values[name] for name in feature_names] for seq in dataset.sequences for s in seq.samples],
        dtype=torch.float64,
    )
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-12, torch.ones_like(std), std)
    return mean.tolist(), std.tolist()


def epoch_order(n_sequences: int, epoch: int) -> list[int]:
    """epoch마다 다른, 재현 가능한 종목 순서. 섞는 단위는 종목 시퀀스 전체라 시간 순서는 그대로다
    (날짜를 섞는 게 아니므로 미래 정보 누출과 무관) — 같은 배치가 매 epoch 같은 시점에 반복되는 것만 막는다."""
    generator = torch.Generator().manual_seed(RANDOM_STATE * 100_003 + epoch)
    return torch.randperm(n_sequences, generator=generator).tolist()


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
    if record.val_ic is not None:
        val_part += f"  val_ic={record.val_ic:+.4f}"
    logger.info(
        "epoch %3d/%d  loss=%.6e%s  lr=%.3e  time=%.2fs  elapsed=%.1fs",
        record.epoch, epochs, record.train_loss, val_part, record.learning_rate, epoch_seconds, record.elapsed_seconds,
    )


def _batches(sequences: list, batch_size: int) -> list[list]:
    return [sequences[i : i + batch_size] for i in range(0, len(sequences), batch_size)]


def prepare_training_batch(
    batch: list, feature_names: list[str], device: torch.device, label_clip: float | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """학습 손실용 (X, y, mask). 국내 종목에서 가격제한폭(±30%)을 넘는 레이블은 데이터 오류로 보고
    mask에서 빼고(예: 001465 2024-04-16 −91%), 나머지 레이블은 ±`label_clip`으로 자른다 — 소수의
    폭등·폭락이 MSE를 지배하지 않게. 평가(`evaluation.py`)는 이 처리 없이 원래 레이블을 쓴다."""
    X, y, mask = collate_sequences(batch, feature_names)
    for row, seq in enumerate(batch):
        if seq.market not in KRX_MARKETS:
            continue
        for step, sample in enumerate(seq.samples):
            if abs(sample.label) > KRX_PRICE_LIMIT:
                mask[row][step] = False
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    if label_clip is not None:
        y_tensor = torch.clamp(y_tensor, -label_clip, label_clip)
    return (
        torch.tensor(X, dtype=torch.float32, device=device),
        y_tensor,
        torch.tensor(mask, dtype=torch.float32, device=device),
    )


def _train_one_epoch(
    model: MambaRegressor,
    optimizer: torch.optim.Optimizer,
    sequences: list,
    feature_names: list[str],
    batch_size: int,
    device: torch.device,
    epoch: int,
    label_clip: float | None,
    grad_clip: float | None,
) -> float:
    """1 epoch 학습 후 패딩 제외 타임스텝 가중평균 train loss를 반환한다. 종목 순서는 epoch마다
    섞고(`epoch_order`), gradient norm은 `grad_clip`으로 자른다. loss는 GPU에서 누적하고 epoch 끝에
    한 번만 꺼낸다(배치마다 GPU 동기화하지 않도록)."""
    model.train()
    shuffled = [sequences[i] for i in epoch_order(len(sequences), epoch)]
    weighted_loss_sum = torch.zeros((), device=device)
    valid_steps = torch.zeros((), device=device)
    for batch in _batches(shuffled, batch_size):
        x_tensor, y_tensor, mask_tensor = prepare_training_batch(batch, feature_names, device, label_clip)
        loss = _masked_mse_loss(model(x_tensor), y_tensor, mask_tensor)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        batch_valid = mask_tensor.sum()
        weighted_loss_sum += loss.detach() * batch_valid
        valid_steps += batch_valid
    return float(weighted_loss_sum / valid_steps) if float(valid_steps) > 0 else float("nan")


def _masked_loss_on(
    model: MambaRegressor,
    dataset: SequenceDataset,
    feature_names: list[str],
    batch_size: int,
    device: torch.device,
    label_clip: float | None,
) -> float:
    """train loss와 같은 정의(같은 레이블 자르기·마스킹)로 계산한 loss — 학습 곡선에서
    train/val을 같은 척도로 비교하기 위함. 가중치는 바꾸지 않는다."""
    model.eval()
    weighted_loss_sum = torch.zeros((), device=device)
    valid_steps = torch.zeros((), device=device)
    with torch.no_grad():
        for batch in _batches(dataset.sequences, batch_size):
            x_tensor, y_tensor, mask_tensor = prepare_training_batch(batch, feature_names, device, label_clip)
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
    label_clip: float | None = DEFAULT_LABEL_CLIP,
    grad_clip: float | None = DEFAULT_GRAD_CLIP,
    feature_clip_sigma: float = DEFAULT_FEATURE_CLIP_SIGMA,
    select_by_val_ic: bool = True,
    target: str = TARGET_RAW,  # split 레이블이 어떤 타깃인지 — 변환은 호출부 몫, 여기선 메타데이터에 기록만
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
    model = MambaRegressor(in_dim=in_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    # Normalization statistics come from the train window only (no peeking at validation/test).
    feature_mean, feature_std = fit_feature_normalization(split.train, feature_names)
    model.set_normalization(feature_mean, feature_std, feature_clip_sigma)
    model = model.to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    validation = split.validation if split.validation is not None and split.validation.sequences else None

    history: list[EpochRecord] = []
    best: tuple[float, dict] | None = None  # (val IC, 그때의 CPU state_dict)
    train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = _train_one_epoch(
            model, optimizer, split.train.sequences, feature_names, batch_size, torch_device,
            epoch, label_clip, grad_clip,
        )
        val_loss = (
            None if validation is None
            else _masked_loss_on(model, validation, feature_names, batch_size, torch_device, label_clip)
        )
        val_ic = None if validation is None else evaluate_window(model, validation, feature_names).model_ic.mean_ic
        if select_by_val_ic and val_ic is not None and val_ic == val_ic and (best is None or val_ic > best[0]):
            best = (val_ic, {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()})
        now = time.perf_counter()
        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=optimizer.param_groups[0]["lr"],
            elapsed_seconds=now - train_start,
            val_ic=val_ic,
        )
        history.append(record)
        if log_every > 0 and (epoch % log_every == 0 or epoch == epochs):
            _log_epoch(record, epochs, now - epoch_start)

    if best is not None:
        # 마지막 epoch이 아니라 validation 순위 IC가 가장 높았던 epoch의 가중치를 쓴다(2026-10-22 진단:
        # 이전에는 val loss 최저가 epoch 58인데 epoch 100 가중치를 저장했다). test는 여전히 학습 중에 보지 않는다.
        model.load_state_dict(best[1])

    # Every step of the test window (not just each ticker's last day — see evaluation.py).
    metrics = evaluate_window(model, split.test, feature_names).model

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
        target=target,
    )
    # 추론은 CPU 전용 환경에서도 돌아야 하므로 GPU 학습이어도 CPU 텐서로 저장한다.
    cpu_state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    return ModelBundle(estimator=cpu_state_dict, metadata=metadata, history=tuple(history))
