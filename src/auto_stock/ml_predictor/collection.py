"""학습용 데이터 수집(OHLCV+펀더멘털 → `SequenceDataset`) — `scripts/collect_ml_dataset.py`와
`scripts/train_ml_model.py`가 공유한다.

수집(네트워크 I/O 위주, 느림)과 학습(GPU/CPU 연산 위주)을 별도 실행으로 쪼개고 싶을 때,
수집 결과(`SequenceDataset`)를 디스크에 캐싱해두고 재학습 때마다 다시 수집하지 않도록
하기 위한 모듈이다. `build_sequence_dataset`이 만드는 `SequenceDataset`은 frozen dataclass
+ 표준 타입(str/float/date/dict)만 담으므로 `pickle`로 안전하게 왕복한다."""

import logging
import pickle
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path

from tqdm import tqdm

from auto_stock.data.cache import OHLCVCache
from auto_stock.data.market_routing import KRX_MARKETS
from auto_stock.data.service import get_ohlcv
from auto_stock.data.sources.pykrx_source import resolve_market
from auto_stock.ml_predictor.fundamentals import build_fundamentals_by_date
from auto_stock.ml_predictor.models import SequenceDataset
from auto_stock.ml_predictor.universe import select_universe

logger = logging.getLogger(__name__)

# 원자 시장마다 따로 고른다 — 합성 "KRX"를 한 리스트로 잘라 KOSDAQ이 빠졌던 문제(universe.py 참고).
TRAINING_MARKETS = ("KOSPI", "KOSDAQ", "NASDAQ", "NYSE")
DEFAULT_DATASET_PATH = Path("data/ml_training_dataset.pkl")
PROGRESS_LOG_EVERY = 10  # tickers between progress log lines (the tqdm bar stays per-ticker on screen)


def format_duration(seconds: float) -> str:
    """사람이 읽는 남은/경과 시간 — `45초`, `2분`, `1시간58분`."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}초"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}분"
    return f"{minutes // 60}시간{minutes % 60}분"


def collect_training_records(
    cache: OHLCVCache,
    start: date,
    end: date,
    size: int,
    select: Callable[[str, date, int], list[str]] = select_universe,
) -> tuple[list[list], list, dict[str, list[str]]]:
    """원자 시장마다 학습 구간 시작일(`start`) 기준으로 `size`개를 골라(universe.py) OHLCV·펀더멘털을
    모은다 — 학습 스크립트와 수집 스크립트가 공유한다. 셋째 반환값은 시장별 선정 종목(기록용)."""
    all_records: list[list] = []
    all_fundamentals: list = []
    universes: dict[str, list[str]] = {}
    for market in TRAINING_MARKETS:
        tickers = select(market, start, size)
        universes[market] = tickers
        logger.info("[%s] 유니버스 %d종목(%s 시점 기준), 수집 기간 %s ~ %s", market, len(tickers), start, start, end)
        records, fundamentals = collect_universe_records(cache, tickers, start, end, market)
        all_records.extend(records)
        all_fundamentals.extend(fundamentals)
    return all_records, all_fundamentals, universes


def atomic_market_for(ticker: str, market: str) -> str:
    """`market`이 합성 "KRX"면 종목의 실제 소속 거래소(KOSPI/KOSDAQ)를 역조회한다
    (역조회 실패는 KOSPI로 보수적 폴백). 그 외는 이미 원자값이라 그대로 반환."""
    if market != "KRX":
        return market
    return resolve_market(ticker) or "KOSPI"


def _progress_line(
    market: str, done: int, total: int, loaded: int, skipped: int, missing_fundamentals: int, elapsed: float
) -> str:
    per_ticker = elapsed / done
    return (
        f"[{market}] {done}/{total} 로드={loaded} 스킵={skipped} 펀더멘털없음={missing_fundamentals} "
        f"경과={format_duration(elapsed)} 종목당={per_ticker:.1f}초 남은≈{format_duration(per_ticker * (total - done))}"
    )


def collect_universe_records(
    cache: OHLCVCache,
    tickers: list[str],
    start: date,
    end: date,
    market: str,
    log_every: int = PROGRESS_LOG_EVERY,
) -> tuple[list[list], list]:
    """OHLCV 레코드와, 국내 종목에 대해서만 채워진 펀더멘털(dict | None)을
    1:1 대응 리스트로 함께 수집한다 — 미국 종목은 fundamentals가 None(중립값 폴백).

    진행 상황은 화면용 tqdm 진행바와 별도로 `log_every`종목마다(+마지막) INFO 로그 한 줄로 남기고,
    스킵 사유는 WARNING으로 남긴다 — 파일 로그(`run_logging.configure_run_logging`)에서 수집이 어디서
    느려지거나 멈췄는지 추적하기 위함. `펀더멘털없음`은 국내 종목인데 펀더멘털 조회가 빈 결과였던 수다
    (`fundamentals.py`는 이 경우 예외 없이 중립값으로 채우므로, 세지 않으면 조용히 묻힌다)."""
    all_records = []
    all_fundamentals = []
    skipped = 0
    missing_fundamentals = 0
    started = time.perf_counter()
    progress = tqdm(tickers, desc=f"[{market}] OHLCV 수집", unit="종목")
    for done, ticker in enumerate(progress, start=1):
        try:
            atomic_market = atomic_market_for(ticker, market)
            records = get_ohlcv(cache, ticker, start, end, atomic_market)
            if records:
                fundamentals = build_fundamentals_by_date(ticker, atomic_market, start, end)
                all_records.append(records)
                all_fundamentals.append(fundamentals)
                if atomic_market in KRX_MARKETS and not fundamentals:
                    missing_fundamentals += 1
        except Exception as exc:  # per-ticker isolation, same principle as the orchestrator
            skipped += 1
            logger.warning("[%s] 스킵 %s: %s", market, ticker, exc)
        progress.set_postfix(로드=len(all_records), 스킵=skipped)
        if done % log_every == 0 or done == len(tickers):
            elapsed = time.perf_counter() - started
            logger.info(
                _progress_line(market, done, len(tickers), len(all_records), skipped, missing_fundamentals, elapsed)
            )
    return all_records, all_fundamentals


def save_dataset(dataset: SequenceDataset, path: Path | str = DEFAULT_DATASET_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(dataset, f)
    return path


def load_dataset(path: Path | str = DEFAULT_DATASET_PATH) -> SequenceDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"수집된 데이터셋을 찾을 수 없습니다: {path}. scripts/collect_ml_dataset.py를 먼저 실행하세요."
        )
    # pickle은 신뢰 경계 밖 입력(네트워크 다운로드, 사용자 업로드)에는 안전하지 않지만,
    # 이 파일은 scripts/collect_ml_dataset.py가 같은 로컬 저장소에만 쓰는 캐시다
    # (ml_predictor/artifact.py의 torch.load weights_only 미사용 판단과 동일한 전제).
    with path.open("rb") as f:
        return pickle.load(f)
