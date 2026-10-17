"""ML 예측 모델(#2) 학습용 데이터 수집 전용 스크립트 — OHLCV+펀더멘털만 모아 디스크에 저장하고
종료한다(학습은 하지 않음). `scripts/train_ml_model.py`가 수집과 학습을 한 번에 하므로, 수집을
먼저 끝내두고 학습은 나중에(또는 여러 번 하이퍼파라미터를 바꿔가며) 하고 싶을 때 쓴다.

Run from the project root so `data/` resolves correctly:
    .venv/Scripts/python scripts/collect_ml_dataset.py   (Windows)
    .venv/bin/python scripts/collect_ml_dataset.py        (macOS/Linux)

선택 환경변수(`scripts/train_ml_model.py`와 이름을 공유한다 — 같은 유니버스/기간으로 수집해야
나중에 그 스크립트가 그대로 로드할 수 있다):
    ML_TRAIN_UNIVERSE_SIZE(기본 200), ML_TRAIN_LOOKBACK_YEARS(5), ML_TRAIN_DATASET_PATH
    (기본 data/ml_training_dataset.pkl)

진행 상황은 화면(tqdm 진행바 + 로그 줄)과 `logs/ml_training/collect_<YYYYmmdd_HHMMSS>.log`에 함께
남는다 — 10종목마다 `[KRX] 40/200 로드=40 스킵=0 펀더멘털없음=0 경과=33분 종목당=44.2초 남은≈1시간58분`
한 줄, 스킵 종목은 WARNING으로 사유와 함께. 로그 파일 경로는 시작할 때 출력된다.

저장된 파일이 있으면 `scripts/train_ml_model.py`는 재수집 없이 바로 그 파일을 읽어 학습한다
(`ML_TRAIN_FORCE_COLLECT=1`로 그 스크립트를 돌리면 무시하고 재수집함)."""

import logging
import os
from datetime import date, timedelta
from pathlib import Path

from tqdm.contrib.logging import logging_redirect_tqdm

from auto_stock.data.cache import OHLCVCache
from auto_stock.ml_predictor.collection import (
    DEFAULT_DATASET_PATH,
    collect_universe_records,
    save_dataset,
    universe_for,
)
from auto_stock.ml_predictor.dataset import build_sequence_dataset
from auto_stock.ml_predictor.run_logging import configure_run_logging

UNIVERSE_MARKETS = ["KRX", "NASDAQ", "NYSE"]  # bucket_markets로 원자 거래소 전체 합집합
UNIVERSE_SIZE = int(os.environ.get("ML_TRAIN_UNIVERSE_SIZE", 200))
LOOKBACK_YEARS = int(os.environ.get("ML_TRAIN_LOOKBACK_YEARS", 5))
DATASET_PATH = Path(os.environ.get("ML_TRAIN_DATASET_PATH", DEFAULT_DATASET_PATH))

logger = logging.getLogger("auto_stock.scripts.collect_ml_dataset")


def main() -> None:
    log_path = configure_run_logging("collect")
    logger.info("로그 파일: %s", log_path)

    cache = OHLCVCache("data/ohlcv.duckdb")
    end = date.today()
    start = end - timedelta(days=365 * LOOKBACK_YEARS)

    all_records: list[list] = []
    all_fundamentals: list = []
    # Log lines go through tqdm.write on screen so they don't break the live progress bar
    # (the file handler is untouched and gets every line).
    with logging_redirect_tqdm():
        for market in UNIVERSE_MARKETS:
            tickers = universe_for(market)[:UNIVERSE_SIZE]
            logger.info("[%s] 유니버스: %d종목, 기간: %s ~ %s", market, len(tickers), start, end)
            records, fundamentals = collect_universe_records(cache, tickers, start, end, market)
            all_records.extend(records)
            all_fundamentals.extend(fundamentals)

    dataset = build_sequence_dataset(all_records, all_fundamentals)
    if not dataset.sequences:
        logger.warning("수집된 표본이 없습니다 - lookback을 늘리거나 유니버스를 확장하세요.")
        return

    path = save_dataset(dataset, DATASET_PATH)
    logger.info("저장 완료: %s (%d종목)", path, len(dataset.sequences))
    logger.info("학습하려면: ML_TRAIN_DATASET_PATH=%s .venv/Scripts/python scripts/train_ml_model.py", path)


if __name__ == "__main__":
    main()
