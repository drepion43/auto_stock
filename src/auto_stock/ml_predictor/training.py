"""추정기 팩토리 + 학습/평가/walk-forward 검증.

`StandardScaler`는 반드시 `Pipeline` 안에 넣는다 — fold/split 밖에서 전체 데이터로
스케일러를 fit하면 테스트 구간의 분포 정보가 학습에 새어 들어간다(lookahead 방어
5중 축 중 하나). RandomForest/DummyClassifier는 학습 스크립트의 평가 리포트 전용
벤치마크이며 프로덕션에는 배선하지 않는다 (docs/design/ml-predictor-plan.md 핵심 설계 결정 1).
"""

from datetime import datetime
from typing import Any, Literal

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from auto_stock.ml_predictor.dataset import to_xy
from auto_stock.ml_predictor.labeling import LABEL_HORIZON_DAYS
from auto_stock.ml_predictor.models import DatasetSplit, ModelBundle, ModelMetadata, TrainingDataset

RANDOM_STATE = 0

Algorithm = Literal["logreg", "rf", "dummy"]


def build_estimator(algorithm: Algorithm) -> Any:
    if algorithm == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    # L2 is already scikit-learn's default; passing penalty="l2" explicitly
                    # triggers a FutureWarning on 1.8+ and will raise once 1.10 removes it.
                    LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE),
                ),
            ]
        )
    if algorithm == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE)
    if algorithm == "dummy":
        return DummyClassifier(strategy="prior")
    raise ValueError(f"unknown algorithm: {algorithm!r} (expected one of 'logreg', 'rf', 'dummy')")


def evaluate(estimator: Any, dataset: TrainingDataset) -> dict[str, float]:
    X, y = to_xy(dataset)
    if not X:
        return {"roc_auc": float("nan"), "accuracy": float("nan"), "base_rate": float("nan")}

    base_rate = sum(y) / len(y)
    predictions = estimator.predict(X)
    accuracy = accuracy_score(y, predictions)
    try:
        probabilities = estimator.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y, probabilities)
    except ValueError:
        # a single class present in y (degenerate holdout) makes AUC undefined
        roc_auc = float("nan")

    return {"roc_auc": float(roc_auc), "accuracy": float(accuracy), "base_rate": float(base_rate)}


def train(split: DatasetSplit, market: str, universe_size: int, algorithm: Algorithm = "logreg") -> ModelBundle:
    estimator = build_estimator(algorithm)
    X_train, y_train = to_xy(split.train)
    estimator.fit(X_train, y_train)

    metrics = evaluate(estimator, split.test)
    train_dates = [s.feature.date for s in split.train.samples]
    test_dates = [s.feature.date for s in split.test.samples]

    metadata = ModelMetadata(
        market=market,
        algorithm=algorithm,
        horizon_days=LABEL_HORIZON_DAYS,
        feature_names=split.train.feature_names,
        universe_size=universe_size,
        train_start=min(train_dates),
        train_end=max(train_dates),
        test_start=min(test_dates),
        test_end=max(test_dates),
        train_samples=len(split.train.samples),
        test_samples=len(split.test.samples),
        test_base_rate=metrics["base_rate"],
        test_roc_auc=metrics["roc_auc"],
        test_accuracy=metrics["accuracy"],
        trained_at=datetime.now(),
    )
    return ModelBundle(estimator=estimator, metadata=metadata)


def walk_forward_scores(dataset: TrainingDataset, n_splits: int = 5, algorithm: Algorithm = "logreg") -> list[float]:
    """`TimeSeriesSplit`로 fold마다 과거만 학습/미래만 평가하는 walk-forward AUC 리스트."""
    samples = sorted(dataset.samples, key=lambda s: (s.feature.date, s.feature.ticker))
    ordered_dataset = TrainingDataset(samples=samples, feature_names=dataset.feature_names)
    X, y = to_xy(ordered_dataset)
    if len(X) <= n_splits:
        return []

    splitter = TimeSeriesSplit(n_splits=n_splits)
    scores: list[float] = []
    for train_idx, test_idx in splitter.split(X):
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]

        if len(set(y_test)) < 2:
            continue  # AUC undefined when the fold's holdout is single-class

        estimator = build_estimator(algorithm)
        estimator.fit(X_train, y_train)
        probabilities = estimator.predict_proba(X_test)[:, 1]
        scores.append(float(roc_auc_score(y_test, probabilities)))

    return scores
