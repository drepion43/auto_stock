"""ML 예측 모델(#2) 학습 스크립트 — logreg/rf/dummy 3종 평가 리포트 후 logreg만 저장.

rf/dummy는 평가 리포트 전용 벤치마크이며 프로덕션에는 배선하지 않는다
(docs/design/ml-predictor-plan.md 핵심 설계 결정 1). 승격 규칙: RandomForest가
walk-forward AUC 중앙값 기준 LogReg 대비 +0.02 이상 우세할 때만 재검토한다.

Run from the project root so `data/` resolves correctly:
    .venv/Scripts/python scripts/train_ml_model.py   (Windows)
    .venv/bin/python scripts/train_ml_model.py        (macOS/Linux)

스모크 테스트(작은 유니버스/짧은 lookback)로 빠르게 돌려보려면:
    ML_TRAIN_UNIVERSE_SIZE=5 ML_TRAIN_LOOKBACK_YEARS=1 .venv/Scripts/python scripts/train_ml_model.py
"""

import os
from datetime import date, timedelta

from auto_stock.data.cache import OHLCVCache
from auto_stock.data.service import get_ohlcv, get_universe
from auto_stock.ml_predictor.artifact import save_model
from auto_stock.ml_predictor.dataset import build_training_dataset, chronological_split, to_xy
from auto_stock.ml_predictor.training import build_estimator, evaluate, train, walk_forward_scores

MARKET = "KRX"
UNIVERSE_SIZE = int(os.environ.get("ML_TRAIN_UNIVERSE_SIZE", 200))
LOOKBACK_YEARS = int(os.environ.get("ML_TRAIN_LOOKBACK_YEARS", 5))
REPORT_ALGORITHMS = ("logreg", "rf", "dummy")
PRODUCTION_ALGORITHM = "logreg"


def _collect_universe_records(cache: OHLCVCache, tickers: list[str], start: date, end: date) -> list[list]:
    all_records = []
    for ticker in tickers:
        try:
            records = get_ohlcv(cache, ticker, start, end, MARKET)
            if records:
                all_records.append(records)
        except Exception as exc:  # per-ticker isolation, same principle as the orchestrator
            print(f"  스킵 {ticker}: {exc}")
    return all_records


def main() -> None:
    cache = OHLCVCache("data/ohlcv.duckdb")

    tickers = get_universe(MARKET)[:UNIVERSE_SIZE]
    end = date.today()
    start = end - timedelta(days=365 * LOOKBACK_YEARS)

    print(f"유니버스: {len(tickers)}종목, 기간: {start} ~ {end}")
    all_records = _collect_universe_records(cache, tickers, start, end)

    dataset = build_training_dataset(all_records)
    split = chronological_split(dataset)

    if not split.train.samples or not split.test.samples:
        print("학습/테스트 표본이 부족합니다 - lookback을 늘리거나 유니버스를 확장하세요.")
        return

    train_start = split.train.samples[0].feature.date
    train_end = split.train.samples[-1].feature.date
    test_start = split.test.samples[0].feature.date
    test_end = split.test.samples[-1].feature.date
    print(f"학습 구간: {train_start} ~ {train_end} ({len(split.train.samples)}건)")
    print(f"테스트 구간: {test_start} ~ {test_end} ({len(split.test.samples)}건)")

    X_train, y_train = to_xy(split.train)
    base_rate = sum(y_train) / len(y_train) if y_train else float("nan")
    print(f"학습 세트 base rate(상승 비율): {base_rate:.3f}")

    for algorithm in REPORT_ALGORITHMS:
        estimator = build_estimator(algorithm)
        estimator.fit(X_train, y_train)
        metrics = evaluate(estimator, split.test)
        wf_scores = walk_forward_scores(split.train, algorithm=algorithm)
        wf_mean = sum(wf_scores) / len(wf_scores) if wf_scores else float("nan")
        print(
            f"[{algorithm}] test AUC={metrics['roc_auc']:.3f} accuracy={metrics['accuracy']:.3f} "
            f"base_rate={metrics['base_rate']:.3f} | walk-forward AUC(5-fold, mean)={wf_mean:.3f} "
            f"(folds={len(wf_scores)})"
        )

    bundle = train(split, market=MARKET, universe_size=len(all_records), algorithm=PRODUCTION_ALGORITHM)
    path = save_model(bundle)
    print(f"저장 완료: {path} (+ {path.with_suffix('.metadata.json')})")


if __name__ == "__main__":
    main()
