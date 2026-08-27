"""모델 아티팩트 저장/로드 — joblib(.joblib) + 메타데이터 JSON 사이드카.

`.joblib`은 `data/*.duckdb`와 같은 이유로 커밋하지 않는다(.gitignore). 대신 메타데이터
JSON은 커밋해 "어떤 모델이 어떤 성능으로 배포됐는지"를 git 히스토리에 남긴다
(docs/design/ml-predictor-plan.md 핵심 설계 결정 4).
"""

import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

import joblib

from auto_stock.ml_predictor.labeling import LABEL_HORIZON_DAYS
from auto_stock.ml_predictor.models import ModelBundle, ModelMetadata

DEFAULT_MODEL_DIR = Path("models/ml_predictor")


class ModelArtifactError(Exception):
    pass


def artifact_path(market: str, model_dir: Path | str = DEFAULT_MODEL_DIR) -> Path:
    # only logreg is ever persisted as a production artifact (see plan §핵심 설계 결정 1),
    # so the algorithm/horizon in the filename are fixed rather than parameterized.
    return Path(model_dir) / f"{market}_h{LABEL_HORIZON_DAYS}_logreg.joblib"


def _metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(".metadata.json")


def save_model(bundle: ModelBundle, model_dir: Path | str = DEFAULT_MODEL_DIR) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    path = artifact_path(bundle.metadata.market, model_dir)
    joblib.dump(bundle.estimator, path)

    metadata_dict = asdict(bundle.metadata)
    for key, value in metadata_dict.items():
        if isinstance(value, (datetime, date)):
            metadata_dict[key] = value.isoformat()
    _metadata_path(path).write_text(json.dumps(metadata_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    return path


def load_model(market: str, model_dir: Path | str = DEFAULT_MODEL_DIR) -> ModelBundle:
    path = artifact_path(market, model_dir)
    if not path.exists():
        raise ModelArtifactError(
            f"모델 아티팩트를 찾을 수 없습니다: {path}. scripts/train_ml_model.py를 먼저 실행하세요."
        )

    metadata_path = _metadata_path(path)
    if not metadata_path.exists():
        raise ModelArtifactError(
            f"모델 메타데이터를 찾을 수 없습니다: {metadata_path}. scripts/train_ml_model.py를 먼저 실행하세요."
        )

    # joblib.load executes pickle-based deserialization; safe here because the artifact
    # is produced exclusively by scripts/train_ml_model.py from this same local repo/
    # filesystem (models/ml_predictor/), never from a network fetch or user upload.
    estimator = joblib.load(path)
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = ModelMetadata(
        market=raw["market"],
        algorithm=raw["algorithm"],
        horizon_days=raw["horizon_days"],
        feature_names=raw["feature_names"],
        universe_size=raw["universe_size"],
        train_start=date.fromisoformat(raw["train_start"]),
        train_end=date.fromisoformat(raw["train_end"]),
        test_start=date.fromisoformat(raw["test_start"]),
        test_end=date.fromisoformat(raw["test_end"]),
        train_samples=raw["train_samples"],
        test_samples=raw["test_samples"],
        test_base_rate=raw["test_base_rate"],
        test_roc_auc=raw["test_roc_auc"],
        test_accuracy=raw["test_accuracy"],
        trained_at=datetime.fromisoformat(raw["trained_at"]),
    )
    return ModelBundle(estimator=estimator, metadata=metadata)
