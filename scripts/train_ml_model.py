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
"""

import os
from datetime import date, timedelta

from tqdm import tqdm

from auto_stock.data.cache import OHLCVCache
from auto_stock.data.service import get_ohlcv, get_universe
from auto_stock.data.sources.pykrx_source import resolve_market
from auto_stock.ml_predictor.artifact import save_model
from auto_stock.ml_predictor.dataset import build_sequence_dataset, chronological_split
from auto_stock.ml_predictor.fundamentals import build_fundamentals_by_date
from auto_stock.ml_predictor.training import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_N_LAYERS,
    DEFAULT_WEIGHT_DECAY,
    persistence_baseline_metrics,
    train,
)
from auto_stock.orchestrator.market_buckets import bucket_markets

UNIVERSE_MARKETS = ["KRX", "NASDAQ", "NYSE"]  # bucket_markets로 원자 거래소 전체 합집합
UNIVERSE_SIZE = int(os.environ.get("ML_TRAIN_UNIVERSE_SIZE", 200))
LOOKBACK_YEARS = int(os.environ.get("ML_TRAIN_LOOKBACK_YEARS", 5))
EPOCHS = int(os.environ.get("ML_TRAIN_EPOCHS", DEFAULT_EPOCHS))
BATCH_SIZE = int(os.environ.get("ML_TRAIN_BATCH_SIZE", DEFAULT_BATCH_SIZE))
LEARNING_RATE = float(os.environ.get("ML_TRAIN_LR", DEFAULT_LEARNING_RATE))
HIDDEN_DIM = int(os.environ.get("ML_TRAIN_HIDDEN_DIM", DEFAULT_HIDDEN_DIM))
N_LAYERS = int(os.environ.get("ML_TRAIN_N_LAYERS", DEFAULT_N_LAYERS))


def _universe_for(market: str) -> list[str]:
    """`market`이 합성 "KRX"면 `bucket_markets`로 KOSPI+KOSDAQ을 각각 조회해 중복
    없이 합친다(처음 등장한 순서 보존). 원자 market이면 그대로 `get_universe`."""
    atomic_markets = bucket_markets(market)
    if len(atomic_markets) == 1 and atomic_markets[0] == market:
        return get_universe(market)
    tickers: list[str] = []
    seen: set[str] = set()
    for atomic_market in atomic_markets:
        for ticker in get_universe(atomic_market):
            if ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)
    return tickers


def _atomic_market_for(ticker: str, market: str) -> str:
    """`market`이 합성 "KRX"면 종목의 실제 소속 거래소(KOSPI/KOSDAQ)를 역조회한다
    (역조회 실패는 KOSPI로 보수적 폴백). 그 외는 이미 원자값이라 그대로 반환."""
    if market != "KRX":
        return market
    return resolve_market(ticker) or "KOSPI"


def _collect_universe_records(
    cache: OHLCVCache, tickers: list[str], start: date, end: date, market: str
) -> tuple[list[list], list]:
    """OHLCV 레코드와, 국내 종목에 대해서만 채워진 펀더멘털(dict | None)을
    1:1 대응 리스트로 함께 수집한다 — 미국 종목은 fundamentals가 None(중립값 폴백)."""
    all_records = []
    all_fundamentals = []
    skipped = 0
    progress = tqdm(tickers, desc=f"[{market}] OHLCV 수집", unit="종목")
    for ticker in progress:
        try:
            atomic_market = _atomic_market_for(ticker, market)
            records = get_ohlcv(cache, ticker, start, end, atomic_market)
            if records:
                all_records.append(records)
                all_fundamentals.append(build_fundamentals_by_date(ticker, atomic_market, start, end))
        except Exception as exc:  # per-ticker isolation, same principle as the orchestrator
            skipped += 1
            tqdm.write(f"  스킵 {ticker}: {exc}")
        progress.set_postfix(로드=len(all_records), 스킵=skipped)
    return all_records, all_fundamentals


def main() -> None:
    cache = OHLCVCache("data/ohlcv.duckdb")
    end = date.today()
    start = end - timedelta(days=365 * LOOKBACK_YEARS)

    all_records: list[list] = []
    all_fundamentals: list = []
    for market in UNIVERSE_MARKETS:
        tickers = _universe_for(market)[:UNIVERSE_SIZE]
        print(f"[{market}] 유니버스: {len(tickers)}종목, 기간: {start} ~ {end}")
        records, fundamentals = _collect_universe_records(cache, tickers, start, end, market)
        all_records.extend(records)
        all_fundamentals.extend(fundamentals)

    dataset = build_sequence_dataset(all_records, all_fundamentals)
    split = chronological_split(dataset)

    if not split.train.sequences or not split.test.sequences:
        print("학습/테스트 표본이 부족합니다 - lookback을 늘리거나 유니버스를 확장하세요.")
        return

    train_dates = [s.feature.date for seq in split.train.sequences for s in seq.samples]
    test_dates = [s.feature.date for seq in split.test.sequences for s in seq.samples]
    print(f"학습 구간: {min(train_dates)} ~ {max(train_dates)} ({len(split.train.sequences)}종목)")
    print(f"테스트 구간: {min(test_dates)} ~ {max(test_dates)} ({len(split.test.sequences)}종목)")

    baseline = persistence_baseline_metrics(split.test, split.train.feature_names)
    print(
        f"[persistence baseline] MSE={baseline['mse']:.6f} RMSE={baseline['rmse']:.6f} "
        f"MAE={baseline['mae']:.6f} R2={baseline['r2']:.4f}"
    )

    bundle = train(
        split,
        universe_size=len(all_records),
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
    )
    print(
        f"[mamba] test MSE={bundle.metadata.test_mse:.6f} RMSE={bundle.metadata.test_rmse:.6f} "
        f"MAE={bundle.metadata.test_mae:.6f} R2={bundle.metadata.test_r2:.4f}"
    )

    path = save_model(bundle)
    print(f"저장 완료: {path} (+ {path.with_suffix('.metadata.json')})")


if __name__ == "__main__":
    main()
