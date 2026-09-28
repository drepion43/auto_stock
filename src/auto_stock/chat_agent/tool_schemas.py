"""메인 대화 에이전트에게 노출하는 11개 도구의 OpenAI Responses API 함수콜링 스키마.
(resolve_ticker 1개 + plain function 8개(get_price_data 포함) + deepagents
서브에이전트를 감싼 agent-as-tool 2개 — find_related_companies/stock_analyst.)

`chat.completions`의 `{"function": {...}}` 중첩 형태가 아니라 Responses API 고유의
평평한 형태(`type`/`name`/`description`/`parameters`/`strict`가 최상위 키)를 쓴다
(Context7 `/openai/openai-python`로 확인, `tests/chat_agent/test_tool_schemas.py`가 이
shape을 고정한다).

resolve_ticker만 `query` 단일 파라미터를 받고, get_market_scan_recommendations만
`market` 단일 파라미터를 받는다(특정 종목을 지정하지 않는 전체 스캔형 질의 전용이라
ticker는 없지만, 배치 스캔 캐시가 market별로 분리되어 있어 어느 시장을 볼지는 여전히
필요하다 — 2026-09-09부로 NASDAQ도 지원하면서 KRX 하드코딩을 걷어내고 추가함), 나머지
9개는 `ticker`+`market`을 받는다 — 사용자가 대화 중 언급한 종목명/코드 문자열을
모델이 직접 ticker/market으로 추측하게 하지 않고(환각 방지), 반드시 이 도구를 거쳐
확정된 값만 쓰게 한다(사용자 승인 설계 — ticker_resolution.py를 별도 전처리 단계가
아니라 다른 도구들과 동일한 tool-calling 루프 안의 도구로 노출).
"""

_MARKET_PROPERTY = {"type": "string", "enum": ["KRX", "NASDAQ"], "description": "시장 구분"}


def _ticker_market_schema(name: str, description: str, ticker_description: str) -> dict:
    return {
        "type": "function",
        "name": name,
        "strict": True,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": ticker_description},
                "market": _MARKET_PROPERTY,
            },
            "required": ["ticker", "market"],
            "additionalProperties": False,
        },
    }


_TICKER_DESC = "종목 코드 (KRX: 6자리 숫자 예 005930, NASDAQ: 티커 심볼 예 AAPL)"


def _no_params_schema(name: str, description: str) -> dict:
    return {
        "type": "function",
        "name": name,
        "strict": True,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }


def _market_only_schema(name: str, description: str) -> dict:
    return {
        "type": "function",
        "name": name,
        "strict": True,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {"market": _MARKET_PROPERTY},
            "required": ["market"],
            "additionalProperties": False,
        },
    }

_RESOLVE_TICKER_SCHEMA = {
    "type": "function",
    "name": "resolve_ticker",
    "strict": True,
    "description": (
        "사용자가 언급한 종목코드 또는 회사명 문자열을 실제 (ticker, market) 후보로 "
        "해석한다. **다른 어떤 도구를 호출하기 전에도 반드시 이 도구로 먼저 종목을 "
        "확정하라** — ticker/market을 스스로 추측해서는 안 된다. matches가 1개면 "
        "해석 성공(그 ticker/market을 이후 다른 도구 호출에 그대로 쓰라). 0개면 "
        "종목을 찾지 못한 것이므로 다른 도구를 호출하지 말고 사용자에게 정확한 "
        "종목코드나 회사명을 다시 물어라. 2개 이상이면 모호한 것이므로 절대 임의로 "
        "하나를 고르지 말고 후보 목록(ticker/market/name)을 사용자에게 보여주며 "
        "어떤 종목인지 되물어라."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "사용자가 언급한 종목코드 또는 회사명"},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

TOOL_SCHEMAS = [
    _RESOLVE_TICKER_SCHEMA,
    _ticker_market_schema(
        "analyze_rule_engine",
        "RSI 과매수/과매도, SMA20/60 골든·데드크로스 등 규칙 기반 기술적 매매 후보를 조회한다. "
        "LLM이나 뉴스 없이 순수 계산 규칙만으로 판단한 1차 신호다. 후보가 없으면 candidate가 "
        "null이며, 이는 정상적인 결과(신호 없음)이지 실패가 아니다.",
        _TICKER_DESC,
    ),
    _ticker_market_schema(
        "analyze_ml_prediction",
        "학습된 ML 분류 모델의 향후 상승확률 예측과 주요 기여 피처를 조회한다(백테스트 검증 전 "
        "보조 지표). 해당 시장에 학습된 모델 아티팩트가 없거나 데이터가 부족하면 prediction이 "
        "null이다.",
        _TICKER_DESC,
    ),
    _ticker_market_schema(
        "analyze_chart_pattern",
        "최근 차트(정규화 OHLCV+기술지표)를 LLM으로 해석해 패턴명·방향성(UP/DOWN/NEUTRAL)·"
        "신뢰도·근거를 조회한다. 뚜렷한 패턴이 없으면 analysis가 null이 아니라 direction=NEUTRAL"
        "로 온다 — null은 데이터 자체가 부족할 때만.",
        _TICKER_DESC,
    ),
    _ticker_market_schema(
        "analyze_disclosures",
        "최근 공식 공시(KRX: DART, NASDAQ: SEC EDGAR)를 LLM으로 해석해 시장 영향(POSITIVE/"
        "NEGATIVE/NEUTRAL)·핵심 이벤트·근거를 조회한다. 최근 90일 내 공시가 없으면 analysis가 "
        "null이다.",
        _TICKER_DESC,
    ),
    _ticker_market_schema(
        "analyze_news_sentiment",
        "최근 뉴스 기사(KRX: 네이버뉴스, NASDAQ: GDELT)를 LLM으로 해석해 논조(POSITIVE/NEGATIVE/"
        "NEUTRAL)·핵심 헤드라인·근거를 조회한다. 최근 14일 내 관련 기사가 없으면 analysis가 "
        "null이다.",
        _TICKER_DESC,
    ),
    _ticker_market_schema(
        "analyze_position_sizing",
        "규칙엔진 매매 후보가 있을 때 ATR 기반 손절가/익절가·참고용 매수 수량·배분 비중과 "
        "함께 기준가(reference_price, 사이징 계산에 쓰인 최신 종가)도 반환한다. reference_price는 "
        "**실시간 시세가 아니라 최근 종가 기준**이다 — 답변에서 이 점을 함께 밝혀라. **참고용 "
        "제안이며 실제 주문 실행이 아니고 투자 조언도 아니다** — 반드시 이 사실을 답변에 "
        "포함하라. 매매 후보가 없으면 suggestion이 null이다.",
        _TICKER_DESC,
    ),
    _ticker_market_schema(
        "get_price_data",
        "규칙엔진 신호 유무와 무관하게 캐시된 최신 일봉(OHLCV) 한 건을 그대로 조회한다 — "
        "단순히 '지금 얼마야', '현재가 알려줘' 같은 가격 자체를 묻는 질의에 쓴다. "
        "analyze_rule_engine/analyze_position_sizing은 매매 후보가 없으면 가격을 함께 "
        "반환하지 않으므로(candidate/suggestion이 null), 가격만 필요할 때는 이 도구를 "
        "대신 호출하라. **실시간 시세가 아니라 최근 종가 기준**이다 — 답변에서 이 점을 "
        "함께 밝혀라. 캐시에 데이터가 전혀 없으면 latest가 null이다.",
        _TICKER_DESC,
    ),
    _ticker_market_schema(
        "find_related_companies",
        "해당 종목의 관련기업(경쟁사/공급망/계열사)을 자기 자신 포함 최대 5개까지 LLM "
        "서브에이전트가 스스로 조사해 찾는다. 각 관련기업은 뉴스 공동언급으로 검증돼 "
        "'confirmed'(확인됨) 또는 'inferred'(추정, LLM 지식 기반)로 confidence가 표시된다 "
        "— 반드시 답변에서 이 둘을 구분해 표기하라. 사용자가 '관련기업'/'경쟁사'/'계열사' "
        "등 종목 자신을 넘어선 범위를 궁금해할 때만 호출하라.",
        _TICKER_DESC,
    ),
    _ticker_market_schema(
        "stock_analyst",
        "해당 종목 하나에 대해 LLM 서브에이전트가 규칙엔진/ML예측/차트패턴/공시/뉴스감성/"
        "포지션사이징 중 필요하다고 스스로 판단한 신호만 조회해 종합 판단을 반환한다. "
        "결과의 provenance는 항상 '서브에이전트 자율 조사 결과'다 — 반드시 답변에서 이 "
        "출처를 밝혀라. find_related_companies가 찾은 관련기업 각각에도 이 도구를 호출해 "
        "딥다이브하라.",
        _TICKER_DESC,
    ),
    _market_only_schema(
        "get_market_scan_recommendations",
        "특정 종목을 지정하지 않고 추천을 묻는 질의('추천해줄만한 주식 있어?', '나스닥 "
        "추천해줄만한거 있어?' 등)에 쓴다. market으로 지정한 시장(KRX 또는 NASDAQ)의 배치 "
        "스캔 캐시를 조회한다 — 실시간이 아니라 scanned_at 시각 기준 스냅샷이며, **답변에서 "
        "이 시각을 반드시 함께 밝혀라**. scanned_at이 null이면 스캔이 아직 한 번도 실행된 "
        "적 없다는 뜻이다(candidates도 항상 빈 배열) — 이는 실패가 아니라 '아직 스캔 안 됨' "
        "상태이므로 그대로 안내하라. scanned_at이 값을 가지고 있는데 candidates만 빈 "
        "배열이면 스캔은 실행됐지만 이번 회차에 발견된 후보가 없다는 뜻이다 — '아직 스캔 "
        "안 됨'과 혼동하지 말고 'OO시 기준 스캔 결과 후보 없음'처럼 구분해서 안내하라. "
        "refreshing이 true이면 백그라운드로 최신 데이터를 갱신하는 중이라는 뜻이다 — 이때 "
        "scanned_at/candidates는 이전 스냅샷(최초 조회라면 빈 값)이니 '지금 갱신 중입니다, "
        "잠시 후 다시 물어봐 주세요'를 답변에 포함하라. "
        "사용자가 시장을 명시하지 않았으면 KRX를 기본으로 조회하라.",
    ),
]
