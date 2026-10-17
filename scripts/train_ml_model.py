"""ML 예측 모델(#2) 학습 스크립트 — MambaStock 기반 PyTorch Mamba SSM, 통합(unified) 단일 모델.

**2026-10-13부로 production 모델이 scikit-learn LogisticRegression에서 Mamba(SSM)로
교체됐다**(docs/design/ml-predictor-mamba-plan.md) — 계수 기반 설명가능성(Explainability)은
포기하고(사용자 승인된 트레이드오프), 예측 수익률(regression) 자체를 신호로 쓴다.
market/종목별로 모델을 나누지 않고 **KRX(코스피+코스닥)+NASDAQ+NYSE 전체 유니버스를
하나의 통합 모델로 pooled 학습**한다(사용자 결정) — 과거 "두 시장을 하나로 합치면
계수 해석이 무의미해진다"는 이유는 더 이상 적용되지 않는다(Mamba는 애초에 계수
해석을 제공하지 않음).

PER/PBR/배당수익률/회전율/PSR 등 국내전용 펀더멘털 피처는 KOSPI/KOSDAQ 종목에만
채워지고 NASDAQ/NYSE는 중립값으로 깔린다(`features.py` 참고).

Run from the project root so `data/` resolves correctly:
    .venv/Scripts/python scripts/train_ml_model.py   (Windows)
    .venv/bin/python scripts/train_ml_model.py        (macOS/Linux)

스모크 테스트(작은 유니버스/짧은 lookback/적은 epoch)로 빠르게 돌려보려면:
    ML_TRAIN_UNIVERSE_SIZE=5 ML_TRAIN_LOOKBACK_YEARS=1 ML_TRAIN_EPOCHS=5 .venv/Scripts/python scripts/train_ml_model.py

학습 device는 `ML_TRAIN_DEVICE`(auto|cuda|cpu, 기본 auto = GPU 있으면 GPU). `cuda`를 명시했는데
GPU를 못 쓰면 데이터 수집 전에 바로 실패한다. 저장되는 아티팩트는 device와 무관하게 CPU 텐서다.

학습 로그는 `ML_TRAIN_LOG_EVERY` epoch마다(기본 1, 0이면 끔) train/val loss·학습률·소요시간을 출력한다.
파일로 남기려면: ... python -u scripts/train_ml_model.py 2>&1 | tee train.log

데이터는 시간순 train/validation/test(기본 ~70/10/20, `ML_TRAIN_VAL_RATIO`)로 나뉜다. 학습이 끝나면
validation/test(국내·미국별) 성능을 persistence 베이스라인과 비교 출력하고, 모델 옆에
학습 곡선 `unified_h1_mamba.history.png`와 `unified_h1_mamba.history.csv`를 저장한다.

**데이터 수집을 미리 끝내두고 학습만 나중에 하려면**: `scripts/collect_ml_dataset.py`를 먼저
실행해 `data/ml_training_dataset.pkl`(기본 경로)을 만들어두면, 이 스크립트는 그 파일이 있으면
OHLCV/펀더멘털을 다시 수집하지 않고 바로 읽어서 학습한다. 경로를 바꾸려면 수집 스크립트와
이 스크립트 모두에 같은 `ML_TRAIN_DATASET_PATH`를 지정하면 된다. `ML_TRAIN_FORCE_COLLECT=1`을
주면 저장된 파일이 있어도 무시하고 다시 수집한다(데이터를 갱신하고 싶을 때).
"""

import logging

from tqdm.contrib.logging import logging_redirect_tqdm
import os
from datetime import date, timedelta
from pathlib import Path

from auto_stock.data.cache import OHLCVCache
from auto_stock.ml_predictor.artifact import save_model
from auto_stock.ml_predictor.collection import (
    DEFAULT_DATASET_PATH,
    collect_universe_records,
    load_dataset,
    save_dataset,
    universe_for,
)
from auto_stock.ml_predictor.dataset import build_sequence_dataset, train_val_test_split
from auto_stock.ml_predictor.models import SequenceDataset, SequenceDatasetSplit
from auto_stock.ml_predictor.predictor import build_model
from auto_stock.ml_predictor.reporting import best_val_epoch, plot_training_history, save_history_csv
from auto_stock.ml_predictor.training import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_N_LAYERS,
    DEFAULT_WEIGHT_DECAY,
    MambaRegressor,
    evaluate,
    persistence_baseline_metrics,
    resolve_device,
    split_by_region,
    train,
)

UNIVERSE_MARKETS = ["KRX", "NASDAQ", "NYSE"]  # bucket_markets로 원자 거래소 전체 합집합
UNIVERSE_SIZE = int(os.environ.get("ML_TRAIN_UNIVERSE_SIZE", 200))
LOOKBACK_YEARS = int(os.environ.get("ML_TRAIN_LOOKBACK_YEARS", 5))
EPOCHS = int(os.environ.get("ML_TRAIN_EPOCHS", DEFAULT_EPOCHS))
BATCH_SIZE = int(os.environ.get("ML_TRAIN_BATCH_SIZE", DEFAULT_BATCH_SIZE))
LEARNING_RATE = float(os.environ.get("ML_TRAIN_LR", DEFAULT_LEARNING_RATE))
HIDDEN_DIM = int(os.environ.get("ML_TRAIN_HIDDEN_DIM", DEFAULT_HIDDEN_DIM))
N_LAYERS = int(os.environ.get("ML_TRAIN_N_LAYERS", DEFAULT_N_LAYERS))
DEVICE = os.environ.get("ML_TRAIN_DEVICE", "auto")
LOG_EVERY = int(os.environ.get("ML_TRAIN_LOG_EVERY", 1))
VAL_RATIO = float(os.environ.get("ML_TRAIN_VAL_RATIO", 0.1))  # 전체 날짜 대비, test 바로 앞 구간
TEST_RATIO = 0.2  # chronological_split 기본값과 동일 — test 구간을 바꾸면 이전 결과와 비교 불가
DATASET_PATH = Path(os.environ.get("ML_TRAIN_DATASET_PATH", DEFAULT_DATASET_PATH))
FORCE_COLLECT = os.environ.get("ML_TRAIN_FORCE_COLLECT", "").strip() == "1"


def _collect_dataset() -> SequenceDataset:
    cache = OHLCVCache("data/ohlcv.duckdb")
    end = date.today()
    start = end - timedelta(days=365 * LOOKBACK_YEARS)

    all_records: list[list] = []
    all_fundamentals: list = []
    # Progress/skip log lines go through tqdm.write so they don't break the live progress bar.
    with logging_redirect_tqdm():
        for market in UNIVERSE_MARKETS:
            tickers = universe_for(market)[:UNIVERSE_SIZE]
            print(f"[{market}] 유니버스: {len(tickers)}종목, 기간: {start} ~ {end}")
            records, fundamentals = collect_universe_records(cache, tickers, start, end, market)
            all_records.extend(records)
            all_fundamentals.extend(fundamentals)

    return build_sequence_dataset(all_records, all_fundamentals)


def main() -> None:
    # Fail fast on a bad device request before the (slow) data collection starts.
    device = resolve_device(DEVICE)
    print(f"학습 device: {device}")
    # Root stays at WARNING: pykrx calls logging.info(args, kwargs) with a bad format string, which at
    # INFO level floods the training log with "--- Logging error ---" dumps (seen 2026-10-14).
    # Only our own loggers (epoch lines from ml_predictor.training) are raised to INFO.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("auto_stock").setLevel(logging.INFO)

    if DATASET_PATH.exists() and not FORCE_COLLECT:
        print(f"수집된 데이터셋 로드: {DATASET_PATH}")
        dataset = load_dataset(DATASET_PATH)
    else:
        dataset = _collect_dataset()
        saved_path = save_dataset(dataset, DATASET_PATH)
        print(f"데이터셋 저장: {saved_path} (다음 실행부터 재수집 없이 재사용)")

    universe_size = len(dataset.sequences)
    split = train_val_test_split(dataset, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO)

    if not split.train.sequences or not split.validation or not split.validation.sequences or not split.test.sequences:
        print("학습/검증/테스트 표본이 부족합니다 - lookback을 늘리거나 유니버스를 확장하세요.")
        return

    _print_window("학습", split.train)
    _print_window("검증", split.validation)
    _print_window("테스트", split.test)

    print(
        f"[하이퍼파라미터] epochs={EPOCHS} batch_size={BATCH_SIZE} lr={LEARNING_RATE} "
        f"weight_decay={DEFAULT_WEIGHT_DECAY} hidden_dim={HIDDEN_DIM} n_layers={N_LAYERS} "
        f"features={len(split.train.feature_names)}"
    )
    bundle = train(
        split,
        universe_size=universe_size,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        device=str(device),
        log_every=LOG_EVERY,
    )

    _print_evaluation_report(build_model(bundle), split)

    path = save_model(bundle)
    print(f"저장 완료: {path} (+ {path.with_suffix('.metadata.json')})")
    csv_path = save_history_csv(bundle.history, path.with_name(f"{path.stem}.history.csv"))
    png_path = plot_training_history(bundle.history, path.with_name(f"{path.stem}.history.png"))
    print(f"학습 곡선: {png_path} (+ {csv_path})")
    best = best_val_epoch(bundle.history)
    if best is not None and best < EPOCHS:
        print(
            f"참고: val loss 최저는 epoch {best}이고 저장된 모델은 마지막 epoch {EPOCHS}의 가중치입니다 "
            f"(과적합 가능성 — ML_TRAIN_EPOCHS={best}로 재학습을 고려)."
        )


def _print_window(name: str, dataset: SequenceDataset) -> None:
    dates = [s.feature.date for seq in dataset.sequences for s in seq.samples]
    print(f"{name} 구간: {min(dates)} ~ {max(dates)} ({len(dataset.sequences)}종목, {len(dates)}표본)")


def _format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"MSE={metrics['mse']:.6f} MAE={metrics['mae']:.6f} R2={metrics['r2']:+.4f} "
        f"방향적중률={metrics['hit_rate']:.1%}"
    )


def _print_evaluation_report(model: MambaRegressor, split: SequenceDatasetSplit) -> None:
    """학습 후 검증 리포트 — validation/test 전체, 그리고 test를 국내/미국으로 나눠
    Mamba와 persistence 베이스라인(내일 수익률 = 오늘 수익률)을 나란히 비교한다.
    지표는 종목별 마지막 시점 예측 기준(추론 경로와 같은 조건)."""
    feature_names = split.train.feature_names
    print("\n===== 검증 리포트 =====")
    sections = [("validation 전체", split.validation), ("test 전체", split.test)]
    sections += [(f"test {region}", ds) for region, ds in split_by_region(split.test).items()]
    for name, dataset in sections:
        if dataset is None or not dataset.sequences:
            print(f"[{name}] 종목 없음")
            continue
        print(f"[{name}] ({len(dataset.sequences)}종목)")
        print(f"  mamba    : {_format_metrics(evaluate(model, dataset, feature_names))}")
        print(f"  baseline : {_format_metrics(persistence_baseline_metrics(dataset, feature_names))}")
    print("=======================\n")


if __name__ == "__main__":
    main()
