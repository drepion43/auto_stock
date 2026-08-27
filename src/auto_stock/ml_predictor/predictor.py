"""추론 경로: 최신 시점 피처 -> 확률 + top-3 기여 피처, 규칙엔진 후보에 대한 보조 문구.

`predict`는 데이터 부족 시 예외 대신 `None`을 반환한다 — rule_engine의 "후보 없음"
관례(engine.py가 상충/부재 시 빈 리스트를 반환하는 것)와 동일한 보수적 원칙이다.
`to_reasons`가 만드는 문구는 PRD §10 "보조 시그널로만 사용" 원칙에 따라 항상
백테스트 미검증 고지를 포함한다."""

from typing import Any

from auto_stock.data.models import OHLCVRecord
from auto_stock.ml_predictor.features import latest_feature_vector
from auto_stock.ml_predictor.models import FeatureVector, MLPrediction, ModelBundle

TOP_FEATURE_COUNT = 3
ML_AGREE_THRESHOLD = 0.55
ML_CONFLICT_THRESHOLD = 0.45
DISCLAIMER = "(ML 신호는 백테스트 검증 전 참고용 보조 지표입니다)"


def _top_contributing_features(estimator: Any, feature_names: list[str], vector: FeatureVector) -> list[tuple[str, float]]:
    """선형모델(StandardScaler+LogisticRegression Pipeline)이면 `coef_ * 표준화값`으로
    기여도를 분해한다. 비선형 추정기(named_steps 없음)는 해석 가능한 분해가 없으므로 빈 리스트."""
    named_steps = getattr(estimator, "named_steps", None)
    if not named_steps:
        return []

    scaler = named_steps.get("scaler")
    clf = named_steps.get("clf")
    if scaler is None or clf is None or not hasattr(clf, "coef_"):
        return []

    raw_row = [[vector.values[name] for name in feature_names]]
    standardized_row = scaler.transform(raw_row)[0]
    coefficients = clf.coef_[0]

    contributions = [
        (name, float(coef) * float(value)) for name, coef, value in zip(feature_names, coefficients, standardized_row)
    ]
    contributions.sort(key=lambda item: abs(item[1]), reverse=True)
    return contributions[:TOP_FEATURE_COUNT]


def predict(bundle: ModelBundle, records: list[OHLCVRecord]) -> MLPrediction | None:
    vector = latest_feature_vector(records)
    if vector is None:
        return None

    feature_names = bundle.metadata.feature_names
    row = [[vector.values[name] for name in feature_names]]
    probability_up = float(bundle.estimator.predict_proba(row)[0][1])
    top_features = _top_contributing_features(bundle.estimator, feature_names, vector)

    return MLPrediction(
        ticker=vector.ticker,
        market=vector.market,
        date=vector.date,
        probability_up=probability_up,
        top_features=top_features,
    )


def _stance(prediction: MLPrediction, action: str) -> str:
    pct = prediction.probability_up * 100
    predicts_up = prediction.probability_up >= ML_AGREE_THRESHOLD
    predicts_down = prediction.probability_up <= ML_CONFLICT_THRESHOLD

    agrees = (action == "BUY" and predicts_up) or (action == "SELL" and predicts_down)
    conflicts = (action == "BUY" and predicts_down) or (action == "SELL" and predicts_up)

    if agrees:
        return f"ML 모델(상승확률 {pct:.1f}%)도 {action} 신호에 동의합니다"
    if conflicts:
        return f"ML 모델(상승확률 {pct:.1f}%)은 {action} 신호와 상충됩니다 — 주의"
    return f"ML 모델 상승확률 {pct:.1f}% (중립)"


def to_reasons(prediction: MLPrediction | None, action: str) -> list[str]:
    if prediction is None:
        return []

    reasons = [_stance(prediction, action)]
    if prediction.top_features:
        feature_list = ", ".join(name for name, _ in prediction.top_features)
        reasons.append(f"주요 기여 피처: {feature_list}")
    reasons.append(DISCLAIMER)
    return reasons
