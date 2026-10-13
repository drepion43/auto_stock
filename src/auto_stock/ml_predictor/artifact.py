"""모델 아티팩트 저장/로드 — torch(.pt, state_dict) + 메타데이터 JSON 사이드카.

`.pt`은 `data/*.duckdb`와 같은 이유로 커밋하지 않는다(.gitignore). 대신 메타데이터
JSON은 커밋해 "어떤 모델이 어떤 성능으로 배포됐는지"를 git 히스토리에 남긴다. 통합
단일 모델이라(사용자 결정 — market/종목별 모델 분리 안 함) 아티팩트도 하나뿐이다
— market별 파일 경로 분기가 사라졌다."""

import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

import torch

from auto_stock.ml_predictor.labeling import LABEL_HORIZON_DAYS
from auto_stock.ml_predictor.models import ModelBundle, ModelMetadata

DEFAULT_MODEL_DIR = Path("models/ml_predictor")


class ModelArtifactError(Exception):
    pass


def artifact_path(model_dir: Path | str = DEFAULT_MODEL_DIR) -> Path:
    # production에 저장되는 알고리즘은 mamba 하나뿐이고, 통합 단일 모델이라 market별
    # 경로 분기도 없다 — 파일명을 파라미터화하지 않고 고정한다.
    return Path(model_dir) / f"unified_h{LABEL_HORIZON_DAYS}_mamba.pt"


def _metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(".metadata.json")


def save_model(bundle: ModelBundle, model_dir: Path | str = DEFAULT_MODEL_DIR) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    path = artifact_path(model_dir)
    torch.save(bundle.estimator, path)

    metadata_dict = asdict(bundle.metadata)
    for key, value in metadata_dict.items():
        if isinstance(value, (datetime, date)):
            metadata_dict[key] = value.isoformat()
    _metadata_path(path).write_text(json.dumps(metadata_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    return path


def load_model(model_dir: Path | str = DEFAULT_MODEL_DIR) -> ModelBundle:
    path = artifact_path(model_dir)
    if not path.exists():
        raise ModelArtifactError(
            f"모델 아티팩트를 찾을 수 없습니다: {path}. scripts/train_ml_model.py를 먼저 실행하세요."
        )

    metadata_path = _metadata_path(path)
    if not metadata_path.exists():
        raise ModelArtifactError(
            f"모델 메타데이터를 찾을 수 없습니다: {metadata_path}. scripts/train_ml_model.py를 먼저 실행하세요."
        )

    # torch.load는 pickle 기반 역직렬화를 실행하는데, 이 아티팩트가 오직
    # scripts/train_ml_model.py에 의해 같은 로컬 저장소/파일시스템에서만 생성되고
    # 네트워크 다운로드나 사용자 업로드를 거치지 않기 때문에 여기서는 안전하다
    # (artifact.py의 기존 joblib.load 보안 판단과 동일한 전제).
    state_dict = torch.load(path, weights_only=True)
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
        test_mse=raw["test_mse"],
        test_rmse=raw["test_rmse"],
        test_mae=raw["test_mae"],
        test_r2=raw["test_r2"],
        model_config=raw["model_config"],
        trained_at=datetime.fromisoformat(raw["trained_at"]),
    )
    return ModelBundle(estimator=state_dict, metadata=metadata)
