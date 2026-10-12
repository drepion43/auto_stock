"""원자 market(KOSPI/KOSDAQ/NASDAQ/NYSE) → 공시/뉴스 소스 라벨 매핑.

`orchestrator/pipeline.py`와 `chat_agent/tools.py`가 과거 각자 `_NEWS_SOURCE_LABELS`/
`_SENTIMENT_SOURCE_LABELS`를 중복 정의했었다 — market을 KOSPI/KOSDAQ으로 세분화하는
작업에서 두 곳을 매번 동시에 고쳐야 하는 번거로움을 없애기 위해 이 모듈로 통합했다.

KOSPI/KOSDAQ은 둘 다 DART/네이버뉴스를 쓰고(국내 공통), NASDAQ/NYSE는 둘 다
EDGAR/GDELT를 쓴다(미국 공통) — 거래소가 늘어나도 "그 거래소가 국내냐 미국이냐"만
알면 되므로, `.get(market)`이 `None`이면 지원하지 않는 시장으로 조용히 스킵한다
(호출부가 2차 폴백 없이 그냥 넘어가면 됨, 에러 아님 — `향후 시장 확장 대비`)."""

NEWS_SOURCE_LABELS = {
    "KOSPI": "DART",
    "KOSDAQ": "DART",
    "NASDAQ": "EDGAR",
    "NYSE": "EDGAR",
}

SENTIMENT_SOURCE_LABELS = {
    "KOSPI": "네이버뉴스",
    "KOSDAQ": "네이버뉴스",
    "NASDAQ": "GDELT",
    "NYSE": "GDELT",
}

# "국내(KRX 계열)"/"미국" 양쪽으로 분기하는 코드가 반복되므로(공시 ID 해석, 이름 해석
# 등 — DART/EDGAR 자체는 코드를 공유하지 않지만 "어느 쪽을 부를지" 판단은 공유 가능)
# 멤버십 집합도 함께 둔다.
KRX_MARKETS = frozenset({"KOSPI", "KOSDAQ"})
US_MARKETS = frozenset({"NASDAQ", "NYSE"})

# ML 모델(ml_predictor)은 재학습하지 않고 기존 합성 "KRX" 모델을 그대로 쓴다(사용자
# 결정, market 세분화 작업) — KOSPI/KOSDAQ 둘 다 "KRX" 모델 아티팩트를 조회하고,
# NYSE는 애초에 모델이 없던 것과 동일하게 취급한다(값이 None이면 해당 도구가
# unavailable로 저하됨, tool_analyze_ml_prediction과 동일 계약).
ML_MODEL_KEYS = {"KOSPI": "KRX", "KOSDAQ": "KRX", "NASDAQ": "NASDAQ", "NYSE": None}
