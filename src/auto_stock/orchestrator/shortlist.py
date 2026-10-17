"""숏리스트(①단계) — 규칙엔진 ∪ ML 방향성 신호(recommendation-synthesis.md §2).

규칙엔진 후보만 숏리스트로 쓰면 AND 게이트가 되어 ML만 잡는 종목을 영구히 놓친다 —
그래서 둘 중 하나라도 방향성 있는 의견을 내면 숏리스트에 넣는다(OR). 둘 다 LLM을 쓰지
않는 공짜 계산(규칙엔진은 순수 계산, ML은 이미 학습된 모델의 `.predict_proba()`
호출)이라 전체 유니버스에 돌려도 비용 문제가 없다 — 비싼 LLM 신호(차트/공시/뉴스감성)는
여기서 나온 숏리스트에만 쓴다(②단계).

`scan_market`(무료 경로)의 계약은 건드리지 않는다 — 텔레그램 발송 등 다른 소비자가
있을 수 있어 하위 호환을 지킨다. 대신 이 모듈을 새로 추가해 전체 유니버스 추천
파이프라인에서만 쓴다."""

from dataclasses import dataclass
from datetime import date, timedelta

from auto_stock.data.cache import OHLCVCache
from auto_stock.data.service import get_ohlcv
from auto_stock.ml_predictor.fundamentals import build_fundamentals_by_date
from auto_stock.ml_predictor.models import MLPrediction, ModelBundle
from auto_stock.ml_predictor.predictor import (
    ML_AGREE_RETURN_THRESHOLD,
    ML_CONFLICT_RETURN_THRESHOLD,
    ML_LOOKBACK_DAYS,
    predict,
)
from auto_stock.orchestrator.models import DEFAULT_LOOKBACK_DAYS
from auto_stock.rule_engine.engine import generate_candidates
from auto_stock.rule_engine.models import Candidate


@dataclass(frozen=True, slots=True)
class ShortlistEntry:
    ticker: str
    market: str
    rule_candidate: Candidate | None
    ml_prediction: MLPrediction | None


def _ml_has_directional_opinion(prediction: MLPrediction | None) -> bool:
    if prediction is None:
        return False
    return (
        prediction.predicted_return >= ML_AGREE_RETURN_THRESHOLD
        or prediction.predicted_return <= ML_CONFLICT_RETURN_THRESHOLD
    )


def build_shortlist(
    cache: OHLCVCache,
    tickers: list[str],
    market: str,
    ml_model: ModelBundle | None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> tuple[list[ShortlistEntry], list[tuple[str, str]]]:
    end = date.today()
    start = end - timedelta(days=lookback_days)

    ml_start = end - timedelta(days=ML_LOOKBACK_DAYS)

    shortlist: list[ShortlistEntry] = []
    errors: list[tuple[str, str]] = []
    for ticker in tickers:
        try:
            records = get_ohlcv(cache, ticker, start, end, market)
            candidates = generate_candidates(records)
            rule_candidate = candidates[0] if candidates else None

            if ml_model is not None:
                # Mamba는 종목 전체 히스토리를 시퀀스로 받는다(MambaStock 원본 방식) —
                # 규칙엔진용 짧은 lookback(records)과 별도로 더 긴 기간을 다시 조회한다.
                ml_records = get_ohlcv(cache, ticker, ml_start, end, market)
                fundamentals = build_fundamentals_by_date(ticker, market, ml_start, end)
                ml_prediction = predict(ml_model, ml_records, fundamentals)
            else:
                ml_prediction = None

            if rule_candidate is not None or _ml_has_directional_opinion(ml_prediction):
                shortlist.append(
                    ShortlistEntry(
                        ticker=ticker, market=market,
                        rule_candidate=rule_candidate, ml_prediction=ml_prediction,
                    )
                )
        except Exception as exc:  # per-ticker isolation, same as scan_market
            errors.append((ticker, str(exc)))

    return shortlist, errors
