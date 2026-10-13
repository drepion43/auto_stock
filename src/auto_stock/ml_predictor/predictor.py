"""추론 경로: 종목의 과거 시퀀스 -> 다음 거래일 예측 수익률, 규칙엔진 후보에 대한 보조 문구.

`predict`는 데이터 부족 시 예외 대신 `None`을 반환한다 — rule_engine의 "후보 없음"
관례(engine.py가 상충/부재 시 빈 리스트를 반환하는 것)와 동일한 보수적 원칙이다.
`to_reasons`가 만드는 문구는 PRD §10 "보조 시그널로만 사용" 원칙에 따라 항상
백테스트 미검증 고지를 포함한다.

Mamba(블랙박스 SSM) 교체로 계수 기반 피처 기여도 분해(`_top_contributing_features`,
선형모델 전용)는 더 이상 제공하지 않는다 — 사용자가 승인한 트레이드오프
(docs/design/ml-predictor-mamba-plan.md "사용자 확정 결정 1")."""

import torch

from auto_stock.data.models import OHLCVRecord
from auto_stock.ml_predictor.dataset import build_ticker_sequence
from auto_stock.ml_predictor.models import MLPrediction, ModelBundle
from auto_stock.ml_predictor.training import MambaRegressor

ML_AGREE_RETURN_THRESHOLD = 0.003  # 다음날 예측 수익률 >= +0.3% 를 "상승 예측"으로 간주
ML_CONFLICT_RETURN_THRESHOLD = -0.003  # <= -0.3% 를 "하락 예측"으로 간주
DISCLAIMER = "(ML 신호는 백테스트 검증 전 참고용 보조 지표입니다)"

# Mamba는 종목 전체 히스토리를 하나의 시퀀스로 받는다(MambaStock 원본 방식, 사용자
# 결정) — 기존 규칙엔진/오케스트레이터의 DEFAULT_LOOKBACK_DAYS(120일, SMA60 워밍업 전용)
# 로는 시퀀스가 너무 짧아 학습 때와 다른 분포로 추론하게 된다. 추론 호출부가 ML
# 전용으로 이 lookback을 따로 써야 한다.
ML_LOOKBACK_DAYS = 365 * 3


def build_model(bundle: ModelBundle) -> MambaRegressor:
    """`bundle.metadata.model_config`로 `MambaRegressor`를 재구성하고 저장된
    가중치(`bundle.estimator`, state_dict)를 로드한다."""
    config = bundle.metadata.model_config
    model = MambaRegressor(
        in_dim=config["in_dim"], hidden_dim=config["hidden_dim"], n_layers=config["n_layers"]
    )
    model.load_state_dict(bundle.estimator)
    model.eval()
    return model


def predict(
    bundle: ModelBundle,
    records: list[OHLCVRecord],
    fundamentals_by_date: dict | None = None,
) -> MLPrediction | None:
    sequence = build_ticker_sequence(records, fundamentals_by_date)
    if sequence is None:
        return None

    feature_names = bundle.metadata.feature_names
    rows = [[s.feature.values[name] for name in feature_names] for s in sequence.samples]

    model = build_model(bundle)
    with torch.no_grad():
        x = torch.tensor([rows], dtype=torch.float32)  # (1, L, D)
        predicted_returns = model(x)[0]  # (L,)

    last_sample = sequence.samples[-1]
    predicted_return = float(predicted_returns[-1])

    return MLPrediction(
        ticker=last_sample.feature.ticker,
        market=last_sample.feature.market,
        date=last_sample.feature.date,
        predicted_return=predicted_return,
    )


def _stance(prediction: MLPrediction, action: str) -> str:
    pct = prediction.predicted_return * 100
    predicts_up = prediction.predicted_return >= ML_AGREE_RETURN_THRESHOLD
    predicts_down = prediction.predicted_return <= ML_CONFLICT_RETURN_THRESHOLD

    agrees = (action == "BUY" and predicts_up) or (action == "SELL" and predicts_down)
    conflicts = (action == "BUY" and predicts_down) or (action == "SELL" and predicts_up)

    if agrees:
        return f"ML 모델(예상 수익률 {pct:+.2f}%)도 {action} 신호에 동의합니다"
    if conflicts:
        return f"ML 모델(예상 수익률 {pct:+.2f}%)은 {action} 신호와 상충됩니다 — 주의"
    return f"ML 모델 예상 수익률 {pct:+.2f}% (중립)"


def to_reasons(prediction: MLPrediction | None, action: str) -> list[str]:
    if prediction is None:
        return []

    reasons = [_stance(prediction, action), DISCLAIMER]
    return reasons
