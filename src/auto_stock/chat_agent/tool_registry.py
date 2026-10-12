"""메인 대화 에이전트 도구의 단일 레지스트리 — docs/design/main-agent-migration-plan.md §4.0.

도구 1개 추가 = `TOOL_SPECS`에 `ToolSpec` 1개(+ `tools.py`의 `tool_*` 구현 함수)만 추가하면
아래가 전부 자동으로 파생된다:
- `openai_schemas()` / `TOOL_SCHEMAS`: 구 루프(loop.py)가 Responses API에 넘기는 함수 스키마
- `dispatch()` / `TOOL_DISPATCH`: 구 루프의 이름 → 함수 매핑
- `bind_stock_analyst_tools`: `subagent_description`이 있는 도구만 종목에 고정해 stock_analyst에 전달
- `as_langchain_tools`: 1단계 메인 deepagent용 LangChain 도구
(`prompt.py`의 안내 문구는 아직 자동 생성하지 않는다 — 결정 D3. 대신 테스트가 "모든 도구 이름이
프롬프트에 언급되는지"를 확인한다.)

**모델이 받는 바이트는 레지스트리 도입 전과 같다** — 설명 문자열은 기존 `tool_schemas.py`/
`tools.py`에서 그대로 옮겼고, `tests/chat_agent/fixtures/`의 골든 파일이 이를 고정한다.

import 방향(순환 방지): `tool_args` ← `tools` ← `tool_registry` ← `tool_schemas`. `tools.py`는
이 모듈을 top-level에서 import하지 않는다(`TOOL_DISPATCH`/`_bind_stock_analyst_tools`는 지연 위임).
`ToolSpec.fn_name`을 함수 객체가 아니라 이름으로 두고 호출 시점에 `getattr`로 찾는 이유는
`mocker.patch("auto_stock.chat_agent.tools.tool_*")`가 그대로 먹게 하기 위해서다."""

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, get_args, get_origin

from langchain_core.tools import StructuredTool
from langchain_core.tools import tool as langchain_tool
from pydantic import BaseModel, ValidationError

from auto_stock.chat_agent import tools as _tools
from auto_stock.chat_agent.tool_args import (
    MarketArgs,
    QueryArgs,
    QueryMarketArgs,
    TickerMarketArgs,
)

_ARGUMENT_ERROR_PREFIX = "도구 인자 파싱 실패"  # 구 루프(loop._dispatch_tool_call)와 같은 문구(계약 C7)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str  # 메인 에이전트가 보는 설명
    args_model: type[BaseModel]
    fn_name: str  # tools.py의 tool_* 함수 이름 — 호출 시점에 조회(patch 호환)
    subagent_description: str | None = None  # 있으면 stock_analyst 바인딩 대상(짧은 설명)


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="resolve_ticker",
        description=(
            "사용자가 언급한 종목코드 또는 회사명 문자열을 실제 (ticker, market) 후보로 "
            "해석한다. **다른 어떤 도구를 호출하기 전에도 반드시 이 도구로 먼저 종목을 "
            "확정하라** — ticker/market을 스스로 추측해서는 안 된다. matches가 1개면 "
            "해석 성공(그 ticker/market을 이후 다른 도구 호출에 그대로 쓰라). 0개면 "
            "종목을 찾지 못한 것이므로 다른 도구를 호출하지 말고 사용자에게 정확한 "
            "종목코드나 회사명을 다시 물어라. 2개 이상이면 모호한 것이므로 절대 임의로 "
            "하나를 고르지 말고 후보 목록(ticker/market/name)을 사용자에게 보여주며 "
            "어떤 종목인지 되물어라."
        ),
        args_model=QueryArgs,
        fn_name="tool_resolve_ticker",
    ),
    ToolSpec(
        name="analyze_rule_engine",
        description=(
            "RSI 과매수/과매도, SMA20/60 골든·데드크로스 등 규칙 기반 기술적 매매 후보를 조회한다. "
            "LLM이나 뉴스 없이 순수 계산 규칙만으로 판단한 1차 신호다. 후보가 없으면 candidate가 "
            "null이며, 이는 정상적인 결과(신호 없음)이지 실패가 아니다."
        ),
        args_model=TickerMarketArgs,
        fn_name="tool_analyze_rule_engine",
        subagent_description="규칙 기반 기술적 매매 후보(RSI/SMA 크로스 등)를 조회한다.",
    ),
    ToolSpec(
        name="analyze_ml_prediction",
        description=(
            "학습된 ML 분류 모델의 향후 상승확률 예측과 주요 기여 피처를 조회한다(백테스트 검증 전 "
            "보조 지표). 해당 시장에 학습된 모델 아티팩트가 없거나 데이터가 부족하면 prediction이 "
            "null이다."
        ),
        args_model=TickerMarketArgs,
        fn_name="tool_analyze_ml_prediction",
        subagent_description="ML 분류 모델의 상승확률 예측을 조회한다.",
    ),
    ToolSpec(
        name="analyze_chart_pattern",
        description=(
            "최근 차트(정규화 OHLCV+기술지표)를 LLM으로 해석해 패턴명·방향성(UP/DOWN/NEUTRAL)·"
            "신뢰도·근거를 조회한다. 뚜렷한 패턴이 없으면 analysis가 null이 아니라 direction=NEUTRAL"
            "로 온다 — null은 데이터 자체가 부족할 때만."
        ),
        args_model=TickerMarketArgs,
        fn_name="tool_analyze_chart_pattern",
        subagent_description="LLM 차트 패턴 해석(방향성/신뢰도/근거)을 조회한다.",
    ),
    ToolSpec(
        name="analyze_technical_context",
        description=(
            "ADX(+DI/-DI, 추세 강도·방향), 볼린저밴드(%B·밴드폭), 스토캐스틱(%K/%D), OBV 방향, "
            "Aroon(up/down/oscillator), TTM 스퀴즈(변동성 수축 여부), RSI 다이버전스 등 7개 기술지표를 "
            "판단 없이 숫자 그대로 조회한다 — 매수/매도 판단은 하지 않으니 수치를 직접 해석해 설명하라. "
            "LLM을 쓰지 않는 계산이며 최근 종가 기준(실시간 아님)이다. 가격 이력이 부족하면 context가 "
            "null이다(실패 아님)."
        ),
        args_model=TickerMarketArgs,
        fn_name="tool_analyze_technical_context",
        subagent_description="ADX·볼린저밴드·스토캐스틱·OBV·Aroon·스퀴즈·RSI 다이버전스 기술지표 수치를 조회한다.",
    ),
    ToolSpec(
        name="analyze_disclosures",
        description=(
            "최근 공식 공시(KOSPI/KOSDAQ: DART, NASDAQ/NYSE: SEC EDGAR)를 LLM으로 해석해 시장 영향(POSITIVE/"
            "NEGATIVE/NEUTRAL)·핵심 이벤트·근거를 조회한다. 최근 90일 내 공시가 없으면 analysis가 "
            "null이다."
        ),
        args_model=TickerMarketArgs,
        fn_name="tool_analyze_disclosures",
        subagent_description="최근 공식 공시(DART/EDGAR)에 대한 LLM 해석을 조회한다.",
    ),
    ToolSpec(
        name="analyze_news_sentiment",
        description=(
            "최근 뉴스 기사(KOSPI/KOSDAQ: 네이버뉴스, NASDAQ/NYSE: GDELT)를 LLM으로 해석해 논조(POSITIVE/NEGATIVE/"
            "NEUTRAL)·핵심 헤드라인·근거를 조회한다. 최근 14일 내 관련 기사가 없으면 analysis가 "
            "null이다."
        ),
        args_model=TickerMarketArgs,
        fn_name="tool_analyze_news_sentiment",
        subagent_description="최근 뉴스 논조에 대한 LLM 해석을 조회한다.",
    ),
    ToolSpec(
        name="analyze_position_sizing",
        description=(
            "규칙엔진 매매 후보가 있을 때 ATR 기반 손절가/익절가·참고용 매수 수량·배분 비중과 "
            "함께 기준가(reference_price, 사이징 계산에 쓰인 최신 종가)도 반환한다. reference_price는 "
            "**실시간 시세가 아니라 최근 종가 기준**이다 — 답변에서 이 점을 함께 밝혀라. **참고용 "
            "제안이며 실제 주문 실행이 아니고 투자 조언도 아니다** — 반드시 이 사실을 답변에 "
            "포함하라. 매매 후보가 없으면 suggestion이 null이다."
        ),
        args_model=TickerMarketArgs,
        fn_name="tool_analyze_position_sizing",
        subagent_description="ATR 기반 손절/익절가·참고용 매수 수량을 조회한다.",
    ),
    ToolSpec(
        name="get_price_data",
        description=(
            "규칙엔진 신호 유무와 무관하게 캐시된 최신 일봉(OHLCV) 한 건을 그대로 조회한다 — "
            "단순히 '지금 얼마야', '현재가 알려줘' 같은 가격 자체를 묻는 질의에 쓴다. "
            "analyze_rule_engine/analyze_position_sizing은 매매 후보가 없으면 가격을 함께 "
            "반환하지 않으므로(candidate/suggestion이 null), 가격만 필요할 때는 이 도구를 "
            "대신 호출하라. **실시간 시세가 아니라 최근 종가 기준**이다 — 답변에서 이 점을 "
            "함께 밝혀라. 캐시에 데이터가 전혀 없으면 latest가 null이다."
        ),
        args_model=TickerMarketArgs,
        fn_name="tool_get_price_data",
    ),
    ToolSpec(
        name="find_related_companies",
        description=(
            "해당 종목의 관련기업(경쟁사/공급망/계열사)을 자기 자신 포함 최대 5개까지 LLM "
            "서브에이전트가 스스로 조사해 찾는다. 각 관련기업은 뉴스 공동언급으로 검증돼 "
            "'confirmed'(확인됨) 또는 'inferred'(추정, LLM 지식 기반)로 confidence가 표시된다 "
            "— 반드시 답변에서 이 둘을 구분해 표기하라. 사용자가 '관련기업'/'경쟁사'/'계열사' "
            "등 종목 자신을 넘어선 범위를 궁금해할 때만 호출하라."
        ),
        args_model=TickerMarketArgs,
        fn_name="tool_find_related_companies",
    ),
    ToolSpec(
        name="stock_analyst",
        description=(
            "해당 종목 하나에 대해 LLM 서브에이전트가 규칙엔진/ML예측/차트패턴/공시/뉴스감성/"
            "포지션사이징 중 필요하다고 스스로 판단한 신호만 조회해 종합 판단을 반환한다. "
            "결과의 provenance는 항상 '서브에이전트 자율 조사 결과'다 — 반드시 답변에서 이 "
            "출처를 밝혀라. find_related_companies가 찾은 관련기업 각각에도 이 도구를 호출해 "
            "딥다이브하라."
        ),
        args_model=TickerMarketArgs,
        fn_name="tool_stock_analyst",
    ),
    ToolSpec(
        name="get_market_scan_recommendations",
        description=(
            "특정 종목을 지정하지 않고 추천을 묻는 질의('추천해줄만한 주식 있어?', '나스닥 "
            "추천해줄만한거 있어?' 등)에 쓴다. market으로 지정한 시장 — 원자 거래소(KOSPI/"
            "KOSDAQ/NASDAQ/NYSE) 또는 합성 버킷(KRX=코스피+코스닥, US=나스닥+NYSE, SP500, "
            "DOW) — 에서 규칙엔진·ML예측·차트분석·공시·뉴스감성을 종합해 상위 종목을 순위(rank)와 근거"
            "(summary)와 함께 미리 선별해둔 배치 결과를 조회한다(recommendation-synthesis-"
            "plan.md) — 실시간이 아니라 scanned_at 시각 기준 스냅샷이며, **답변에서 이 시각을 "
            "반드시 함께 밝혀라**. recommendations의 각 summary는 이미 종목별로 어떤 신호를 "
            "확인했는지(예: '규칙엔진·ML만 확인됨, 추가 신호 미조사') 구체적으로 담고 있으니 "
            "그 내용을 그대로 전달하라 — 뭉뚱그리지 말 것. scanned_at이 null이면 아직 한 번도 "
            "실행된 적 없다는 뜻이다(recommendations도 항상 빈 배열) — 이는 실패가 아니라 "
            "'아직 분석 안 됨' 상태이므로 그대로 안내하라. scanned_at이 값을 가지고 있는데 "
            "recommendations만 빈 배열이면 분석은 실행됐지만 이번 회차에 추천할 종목이 없다는 "
            "뜻이다 — 'OO시 기준 추천 결과 없음'처럼 구분해서 안내하라. refreshing이 true이면 "
            "백그라운드로 최신 데이터를 갱신하는 중이라는 뜻이다 — 이때 scanned_at/"
            "recommendations는 이전 스냅샷(최초 조회라면 빈 값)이니 '지금 갱신 중입니다, 잠시 "
            "후 다시 물어봐 주세요'를 답변에 포함하라. 사용자가 시장을 명시하지 않았으면 KRX(국내 "
            "코스피+코스닥 합성)를 기본으로 조회하라."
        ),
        args_model=MarketArgs,
        fn_name="tool_get_market_scan_recommendations",
    ),
    ToolSpec(
        name="get_sector_recommendations",
        description=(
            "섹터/업종/테마를 지정해 추천을 묻는 질의('로봇섹터 추천해줄만한 거 있어?', '반도체 "
            "관련 추천해줄 거 있어?' 등)에 쓴다 — get_market_scan_recommendations와 달리 query로 "
            "섹터/테마명을 받는다. query가 공식 업종명과 명확히 일치하면 결정론적으로 그 업종 "
            "구성종목을, 아니면 LLM이 제안+뉴스검증한 종목을 대상으로 규칙엔진·ML예측·차트분석·"
            "공시·뉴스감성을 종합해 상위 종목을 순위(rank)·근거(summary)와 함께 반환한다. 반환 "
            "형태(scanned_at/recommendations/refreshing)는 get_market_scan_recommendations와 "
            "동일하게 해석하라 — 실시간이 아니라 스냅샷이며 scanned_at을 반드시 함께 밝혀라. "
            "NASDAQ/NYSE 종목의 섹터 질의는 공식 업종 매칭이 불가능해 항상 LLM 제안+뉴스검증 경로로만 "
            "처리된다."
        ),
        args_model=QueryMarketArgs,
        fn_name="tool_get_sector_recommendations",
    ),
    ToolSpec(
        name="analyze_etf_composition",
        description=(
            "ETF를 지정한 질의('KODEX 200 왜 올랐어?', 'SPY 어때?' 등)에 쓴다. ETF 자체에는 "
            "공시/뉴스가 거의 없으므로, 비중 상위 구성종목(최대 10개)의 공시·뉴스감성을 모아 "
            "'ETF가 왜 움직였는지'를 종합한다. **국내(KOSPI/KOSDAQ)·미국(NASDAQ/NYSE) ETF 모두 "
            "지원**한다. 미국 ETF는 SEC EDGAR Form N-PORT 분기 공시 기준이라 국내(거의 일별)보다 "
            "데이터가 최대 2개월 묵었을 수 있다 — 이때 결과에 data_as_of_note 필드가 함께 오니 "
            "답변에서 반드시 그 내용을 밝혀라. holdings가 null이면 ETF가 아니거나 구성종목 데이터가 "
            "없다는 뜻이다. 결과의 provenance는 'ETF 구성종목 집계 분석'이며, 반환된 holdings(분석한 "
            "구성종목·비중)만 분석했다는 점과 함께 답변에서 반드시 밝혀라."
        ),
        args_model=TickerMarketArgs,
        fn_name="tool_analyze_etf_composition",
    ),
)


def _property_schema(field: Any) -> dict:
    """Pydantic 필드 1개 → 기존 수작업 스키마와 같은 키 순서의 property dict."""
    if get_origin(field.annotation) is Literal:
        return {"type": "string", "enum": list(get_args(field.annotation)), "description": field.description}
    return {"type": "string", "description": field.description}


def _openai_schema(spec: ToolSpec) -> dict:
    fields = spec.args_model.model_fields
    return {
        "type": "function",
        "name": spec.name,
        "strict": True,
        "description": spec.description,
        "parameters": {
            "type": "object",
            "properties": {name: _property_schema(field) for name, field in fields.items()},
            "required": list(fields),
            "additionalProperties": False,
        },
    }


def openai_schemas() -> list[dict]:
    """구 루프가 Responses API에 넘기는 평평한 함수 스키마 목록(TOOL_SPECS 순서)."""
    return [_openai_schema(spec) for spec in TOOL_SPECS]


def _late_bound(fn_name: str) -> Callable[..., dict]:
    def _call(context: "_tools.ChatToolContext", **kwargs: Any) -> dict:
        return getattr(_tools, fn_name)(context, **kwargs)

    return _call


def dispatch() -> dict[str, Callable[..., dict]]:
    """구 루프용 이름 → 함수 매핑. 예외를 삼키지 않는다 — TypeError 격리는 구 루프의 몫(계약 C9)."""
    return {spec.name: _late_bound(spec.fn_name) for spec in TOOL_SPECS}


def _argument_error(exc: Exception) -> dict:
    return {"available": False, "error": f"{_ARGUMENT_ERROR_PREFIX}: {exc}"}


def call_isolated(spec: ToolSpec, context: "_tools.ChatToolContext", raw_args: dict) -> dict:
    """인자 검증 + 실행을 감싸 **절대 raise하지 않는다**. 1단계 메인 deepagent 경로에서 쓴다 —
    LangGraph ToolNode는 도구 예외를 영어 오류 문자열로 바꿔버리므로, 모든 실패를 여기서
    기존 `{"available": False, "error": ...}` 형태로 만든다(계약 C9 강화)."""
    try:
        args = spec.args_model.model_validate(raw_args)
    except ValidationError as exc:
        return _argument_error(exc)
    try:
        return getattr(_tools, spec.fn_name)(context, **args.model_dump())
    except Exception as exc:
        return {"available": False, "error": f"도구 실행 실패: {exc}"}


def _validation_error_json(exc: ValidationError) -> str:
    return json.dumps(_argument_error(exc), ensure_ascii=False)


def _langchain_tool(spec: ToolSpec, context: "_tools.ChatToolContext", turn_lock: threading.Lock) -> StructuredTool:
    def _run(**kwargs: Any) -> str:
        # LangGraph는 한 응답의 여러 tool_call을 병렬 실행한다(스파이크 S3) — 기존 순차 실행
        # 계약(C11)과 공유 캐시(DuckDB 등) 안전을 위해 턴 단위 Lock으로 직렬화한다.
        with turn_lock:
            result = call_isolated(spec, context, kwargs)
        return json.dumps(result, ensure_ascii=False)  # 구 루프와 같은 직렬화(계약 C10)

    return StructuredTool.from_function(
        func=_run,
        name=spec.name,
        description=spec.description,
        args_schema=spec.args_model,
        handle_validation_error=_validation_error_json,
    )


def as_langchain_tools(context: "_tools.ChatToolContext", turn_lock: threading.Lock) -> list[StructuredTool]:
    """1단계 메인 deepagent용 도구 — 턴마다 그 턴의 context(QueryBudget 포함)에 바인딩한다."""
    return [_langchain_tool(spec, context, turn_lock) for spec in TOOL_SPECS]


def bind_stock_analyst_tools(context: "_tools.ChatToolContext", ticker: str, market: str) -> list:
    """`stock_analyst` 서브에이전트용 — `subagent_description`이 있는 도구를 종목/시장에
    클로저로 고정한, 인자 없는 langchain `@tool`로 감싼다. 서브에이전트 인스턴스 1개는 종목
    1개만 다루므로 모델이 판단할 것은 "어떤 도구를 부를지"뿐이다(기존 `tools._bind_stock_analyst_tools`
    동작 그대로 — 골든 테스트로 고정)."""

    # deepagents(LangGraph)는 서브에이전트가 한 응답에 낸 tool_call들도 병렬 실행한다. 바인딩 도구들은
    # 같은 OHLCVCache DuckDB 연결을 공유하므로 동시에 돌면 'tuple index out of range' 같은 무작위
    # 오류가 났다(2026-10-12 재현 — NVDA 분석의 "산출 오류"·"분석 오류" 원인). 서브에이전트 인스턴스
    # 1개(= 이 바인딩 1회) 안에서만 직렬화한다 — 배치의 서로 다른 종목은 각자 cursor를 쓰므로 병렬 유지.
    binding_lock = threading.Lock()

    def _make_bound_tool(fn_name: str) -> Callable[[], dict]:
        def _bound_tool() -> dict:
            with binding_lock:
                return getattr(_tools, fn_name)(context, ticker, market)

        return _bound_tool

    return [
        langchain_tool(spec.name, description=spec.subagent_description)(_make_bound_tool(spec.fn_name))
        for spec in TOOL_SPECS
        if spec.subagent_description is not None
    ]


TOOL_SCHEMAS = openai_schemas()
TOOL_DISPATCH = dispatch()
