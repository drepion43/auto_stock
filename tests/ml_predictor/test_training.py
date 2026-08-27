import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from auto_stock.ml_predictor.dataset import chronological_split
from auto_stock.ml_predictor.models import ModelBundle, ModelMetadata
from auto_stock.ml_predictor.training import RANDOM_STATE, build_estimator, evaluate, train, walk_forward_scores


def test_random_state_constant_is_zero():
    assert RANDOM_STATE == 0


def test_build_estimator_logreg_is_a_scaler_plus_logreg_pipeline():
    estimator = build_estimator("logreg")

    assert isinstance(estimator, Pipeline)
    assert "scaler" in estimator.named_steps
    assert "clf" in estimator.named_steps
    clf = estimator.named_steps["clf"]
    # penalty is intentionally left at its default (L2) rather than passed explicitly —
    # scikit-learn 1.8+ deprecates the "l2" string value and removes it in 1.10.
    assert isinstance(clf, LogisticRegression)
    assert clf.solver == "lbfgs"
    assert clf.max_iter == 1000
    assert clf.random_state == RANDOM_STATE


def test_build_estimator_rf_returns_random_forest_with_expected_hyperparams():
    estimator = build_estimator("rf")

    assert isinstance(estimator, RandomForestClassifier)
    assert estimator.n_estimators == 200
    assert estimator.max_depth == 6
    assert estimator.random_state == RANDOM_STATE


def test_build_estimator_dummy_returns_prior_strategy_classifier():
    estimator = build_estimator("dummy")

    assert isinstance(estimator, DummyClassifier)
    assert estimator.strategy == "prior"


def test_build_estimator_rejects_unknown_algorithm():
    with pytest.raises(ValueError):
        build_estimator("not-a-real-algorithm")


@pytest.mark.slow
def test_train_fits_logreg_and_returns_bundle_with_metadata(synthetic_training_dataset):
    split = chronological_split(synthetic_training_dataset, test_ratio=0.2)

    bundle = train(split, market="KRX", universe_size=1, algorithm="logreg")

    assert isinstance(bundle, ModelBundle)
    assert isinstance(bundle.metadata, ModelMetadata)
    assert bundle.metadata.market == "KRX"
    assert bundle.metadata.algorithm == "logreg"
    assert bundle.metadata.horizon_days == 5
    assert bundle.metadata.train_samples == len(split.train.samples)
    assert bundle.metadata.test_samples == len(split.test.samples)
    assert 0.0 <= bundle.metadata.test_base_rate <= 1.0
    # Wiring check: the fitted pipeline can score a real feature row.
    proba = bundle.estimator.predict_proba([[0.0] * len(split.train.feature_names)])
    assert proba.shape == (1, 2)


@pytest.mark.slow
def test_evaluate_reports_roc_auc_accuracy_and_base_rate(synthetic_training_dataset):
    split = chronological_split(synthetic_training_dataset, test_ratio=0.2)
    estimator = build_estimator("logreg")
    from auto_stock.ml_predictor.dataset import to_xy

    X_train, y_train = to_xy(split.train)
    estimator.fit(X_train, y_train)

    metrics = evaluate(estimator, split.test)

    assert set(metrics.keys()) == {"roc_auc", "accuracy", "base_rate"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["base_rate"] <= 1.0


@pytest.mark.slow
def test_walk_forward_scores_uses_time_series_split_and_returns_auc_per_fold(synthetic_training_dataset):
    scores = walk_forward_scores(synthetic_training_dataset, n_splits=5, algorithm="logreg")

    assert isinstance(scores, list)
    assert len(scores) <= 5
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_walk_forward_scores_never_uses_random_shuffle_split():
    import auto_stock.ml_predictor.training as training_module

    assert "train_test_split" not in dir(training_module)
    assert "KFold" not in dir(training_module)
