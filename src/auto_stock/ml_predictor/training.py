"""MambaRegressor(PyTorch) 팩토리 + 학습/평가 — MambaStock(`main.py::Net`/`PredictWithData`)의
구조를 가변 길이 다종목 배치로 확장한 버전.

학습 루프는 sklearn `estimator.fit(X, y)` 1줄이 아니라 PyTorch epoch 루프다. 손실은
패딩된 타임스텝을 제외하도록 마스킹한다(`dataset.py::collate_sequences`가 만든
`mask`) — 종목마다 길이가 다른데 짧은 종목의 패딩 구간이 손실에 들어가면 그 종목이
사실상 "수익률 0을 예측하라"는 가짜 레이블로 학습되는 것과 같다."""

from datetime import datetime

import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from auto_stock.ml_predictor.dataset import collate_sequences
from auto_stock.ml_predictor.labeling import LABEL_HORIZON_DAYS
from auto_stock.ml_predictor.mamba_block import Mamba, MambaConfig
from auto_stock.ml_predictor.models import ModelBundle, ModelMetadata, SequenceDataset, SequenceDatasetSplit

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


def _masked_mse_loss(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """패딩된 타임스텝(mask=False)을 제외한 MSE. 배치 전체에 유효 타임스텝이 하나도
    없으면(이론상 발생하지 않지만 방어적으로) 0을 반환한다."""
    squared_error = (predicted - target) ** 2 * mask
    valid_count = mask.sum()
    if valid_count == 0:
        return torch.tensor(0.0, requires_grad=True)
    return squared_error.sum() / valid_count


def _batches(sequences: list, batch_size: int) -> list[list]:
    return [sequences[i : i + batch_size] for i in range(0, len(sequences), batch_size)]


def train(
    split: SequenceDatasetSplit,
    universe_size: int,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    n_layers: int = DEFAULT_N_LAYERS,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> ModelBundle:
    """통합(unified) 단일 모델 학습 — market/종목별로 모델을 나누지 않는다(사용자 결정).
    `split.train.sequences`를 종목 단위로 미니배치로 묶어 PyTorch epoch 루프를 돈다."""
    _set_seed(RANDOM_STATE)

    feature_names = split.train.feature_names
    in_dim = len(feature_names)
    model = MambaRegressor(in_dim=in_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_sequences = split.train.sequences
    for _epoch in range(epochs):
        model.train()
        for batch in _batches(train_sequences, batch_size):
            X, y, mask = collate_sequences(batch, feature_names)
            x_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            mask_tensor = torch.tensor(mask, dtype=torch.float32)

            predicted = model(x_tensor)
            loss = _masked_mse_loss(predicted, y_tensor, mask_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
    return ModelBundle(estimator=model.state_dict(), metadata=metadata)


def evaluate(model: MambaRegressor, dataset: SequenceDataset, feature_names: list[str]) -> dict[str, float]:
    """시퀀스별로 마지막 시점의 예측만 평가에 쓴다 — `predict()`(추론 경로)가 매 호출마다
    "그 시점까지의 전체 과거"를 받아 마지막 예측만 꺼내는 것과 동일한 평가 조건을 만들기
    위함(패딩된 중간 시점을 평가에 섞지 않음)."""
    if not dataset.sequences:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    model.eval()
    predictions: list[float] = []
    actuals: list[float] = []
    with torch.no_grad():
        for seq in dataset.sequences:
            rows = [[s.feature.values[name] for name in feature_names] for s in seq.samples]
            x = torch.tensor([rows], dtype=torch.float32)
            predicted = model(x)[0]
            predictions.append(float(predicted[-1]))
            actuals.append(seq.samples[-1].label)

    if len(actuals) < 2:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    mse = mean_squared_error(actuals, predictions)
    return {
        "mse": float(mse),
        "rmse": float(mse**0.5),
        "mae": float(mean_absolute_error(actuals, predictions)),
        "r2": float(r2_score(actuals, predictions)),
    }


def persistence_baseline_metrics(dataset: SequenceDataset, feature_names: list[str]) -> dict[str, float]:
    """"다음날 수익률 = 어제의 1일 수익률(return_1d) 그대로" 베이스라인. 모델이 이 단순
    규칙보다 나은지 가늠하는 하한선 — MambaStock 원본엔 없는 비교 지표지만, 기존
    REPORT_ALGORITHMS(rf/dummy)가 하던 "하한선 리포트" 역할을 regression에서 대신한다."""
    if PERSISTENCE_FEATURE_NAME not in feature_names:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    predictions: list[float] = []
    actuals: list[float] = []
    for seq in dataset.sequences:
        sample = seq.samples[-1]
        predictions.append(sample.feature.values[PERSISTENCE_FEATURE_NAME])
        actuals.append(sample.label)

    if len(actuals) < 2:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    mse = mean_squared_error(actuals, predictions)
    return {
        "mse": float(mse),
        "rmse": float(mse**0.5),
        "mae": float(mean_absolute_error(actuals, predictions)),
        "r2": float(r2_score(actuals, predictions)),
    }
