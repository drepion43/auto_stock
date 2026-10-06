"""숏리스트(①단계) — 규칙엔진 ∪ ML 방향성 신호(recommendation-synthesis-plan.md §2).

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
from auto_stock.ml_predictor.models import MLPrediction, ModelBundle
from auto_stock.ml_predictor.predictor import ML_AGREE_THRESHOLD, ML_CONFLICT_THRESHOLD, predict
from auto_stock.orchestrator.pipeline import DEFAULT_LOOKBACK_DAYS
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
    return prediction.probability_up >= ML_AGREE_THRESHOLD or prediction.probability_up <= ML_CONFLICT_THRESHOLD


def build_shortlist(
    cache: OHLCVCache,
    tickers: list[str],
    market: str,
    ml_model: ModelBundle | None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> tuple[list[ShortlistEntry], list[tuple[str, str]]]:
    end = date.today()
    start = end - timedelta(days=lookback_days)

    shortlist: list[ShortlistEntry] = []
    errors: list[tuple[str, str]] = []
    for ticker in tickers:
        try:
            records = get_ohlcv(cache, ticker, start, end, market)
            candidates = generate_candidates(records)
            rule_candidate = candidates[0] if candidates else None

            ml_prediction = predict(ml_model, records) if ml_model is not None else None

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
