# LLM 차트분석 에이전트 (#3) 구현 계획

> `docs/IMPLEMENTATION_PLAN.md`의 "신호 소스 확장" §2(LLM #3)를 실행 가능한 수준으로 구체화한 사전 계획 문서. `docs/design/ml-predictor-plan.md`와 같은 위치·형식이며, 구현 완료 후에는 실제 결과를 기록하는 `docs/design/llm-chart-analyst.md`를 별도로 작성한다 (Phase 4).

## 배경

MVP-0(`60523cd`)과 ML 예측 모듈 #2(`0d87b95`, `c75ee52`, `679646c`)가 구현·커밋 완료됐고 테스트 126개 전량 통과(커버리지 98%) 상태다. 다음 신호 소스로 **차트분석/예측 에이전트 — LLM (#3)**을 진행한다.

PRD §4는 "차트 데이터를 LLM에 입력해 패턴 해석 및 예측 시그널 생성", §5는 "정성적 판단 보강", §5.1은 설명 예시에 "LLM 차트 분석상 단기 반등 패턴 감지"를 이미 명시했다. §10은 "LLM 기반 매매 판단은 환각 가능성이 있어 보조 시그널로만 사용"을 못박았다.

**목표**: OHLCV + 기술적 지표의 수치 요약을 LLM에 입력해 차트 패턴을 구조화된 형태로 해석하고, 규칙엔진 후보에 대한 **보조 근거**로 파이프라인에 배선한다. ML #2와 동시에 켤 수 있어야 한다.

> **프로바이더 결정 (사용자 요청 반영)**: 이 계획은 최초에 Anthropic Claude를 전제로 planner 에이전트가 작성했으나, 사용자가 **OpenAI GPT**(`gpt-5.6-luna`)를 쓰기로 확정해 아래 내용을 전면 갱신했다. 아키텍처(모듈 구조, 환각/동조 방어, `extra_reasons` 통합, 실패 격리, 테스트 전략)는 프로바이더에 무관하므로 원안 그대로 유지했고, SDK 호출·에러 클래스·크리덴셜·모델명 등 구현 세부만 OpenAI 기준으로 바꿨다. LangGraph/`deepagents`는 프로바이더에 종속되지 않으므로(LangChain의 provider-agnostic 모델 추상화로 Claude·GPT 둘 다 사용 가능) 이 선택과 설계 결정 7(SDK 직접 호출, 프레임워크 미도입)은 서로 무관하다.

## 핵심 설계 결정

### 1. 입력은 이미지가 아니라 수치/텍스트 요약

캔들스틱 차트 이미지를 렌더링해 vision 입력으로 보내는 방식은 (a) 차트 렌더링 라이브러리(`mplfinance` 등) 의존성이 새로 필요하고, (b) 이미지 토큰 비용·레이턴시가 텍스트 요약 대비 수 배이며, (c) 이 프로젝트가 이미 계산해 둔 지표값(`ml_predictor.features`의 11종)을 버리고 모델이 픽셀에서 다시 읽어내게 만드는 낭비다.

ML #2에서 딥러닝 대신 해석 가능한 로지스틱회귀를 택한 것과 동일한 원칙 — **가장 단순하고 저렴한 형태로 신호의 가치를 먼저 확인한다.** 이미지 입력은 이번 스코프 밖으로 명시하고, 텍스트 요약본이 실전 표본 검증에서 명백히 부족하다고 판명될 때만 재검토한다.

### 2. 프롬프트에 종목 식별 정보를 넣지 않는다 (환각 방어)

**티커, 종목명, 시장(KRX/NASDAQ), 실제 날짜를 프롬프트에 넣지 않는다.** 가격은 구간 첫 종가를 100으로 하는 정규화 지수로, 날짜는 `t-29 … t-0` 오프셋으로 표기한다.

- 모델의 사전지식(예: "삼성전자는 우량주", "2024년 그 시점엔 랠리였다")에 의한 편향·환각을 **구조적으로** 차단한다 (PRD §10 대응).
- 부수 효과로 KRX(원)/NASDAQ(달러) 구분이 프롬프트 수준에서 불필요해진다 — 시장별 분기 로직이 아예 생기지 않는다. ML #2가 `MARKET="KRX"` 하드코딩에 묶여 나스닥을 못 쓰는 한계(`CODEBASE_REVIEW.md` "현재 한계")를 이 신호원은 처음부터 회피한다.
- `ChartAnalysis`의 `ticker`/`market`/`date`는 **LLM 응답이 아니라 입력 `records`에서** 호출자가 채운다.

### 3. 규칙엔진의 action도 프롬프트에 넣지 않는다 (동조 방어)

"이 종목에 BUY 신호가 떴는데 어떻게 보나?"라고 물으면 모델은 동조(sycophancy)한다 — 상충 신호를 잡아낼 능력을 스스로 없애는 셈이다.

`analyze(client, records)`는 `action`을 받지 않고 독립적으로 방향성을 판단하고, `to_reasons(analysis, action)`이 **코드에서** 동의/상충/중립을 비교한다. 이는 `predictor.py`의 `predict(bundle, records)` / `to_reasons(prediction, action)` 분리와 정확히 같은 구조다.

### 4. `extra_reasons` 경유 보조 신호 (ML #2와 동일)

자체 `Candidate`를 만들지 않고, **규칙엔진 후보가 이미 존재하는 티커에 대해서만** `extra_reasons`로 보조 문장을 추가한다. 알림 건수를 늘리지도, 후보를 필터링하지도 않는다. **`explainer/generator.py`는 한 줄도 수정하지 않는다.**

부수 효과로 **호출 빈도가 자연히 제한된다** — 전체 유니버스 스캔이 아니라 후보 몇 개 수준. 이것이 아래 §비용 통제와 모델 선택의 전제다.

### 5. 구조화된 출력 (OpenAI Responses API `responses.parse` + Pydantic)

OpenAI Python SDK의 `client.responses.parse(model=..., input=[{"role": "system", ...}, {"role": "user", ...}], text_format=ChartPatternRead)` → `response.output_parsed`가 검증된 Pydantic 인스턴스로 반환된다(2026-08-27 기준 공식 문서로 확인 — 과거의 `chat.completions.parse`가 아니라 Responses API로 이관됨). raw JSON 문자열 파싱 + 수동 검증보다 파싱 실패 경로가 근본적으로 줄어든다.

> Phase 0에서 설치된 SDK 버전에 `responses.parse`/`output_parsed`가 실제로 존재하는지, 그리고 정확한 예외 클래스 계층(`openai.APIStatusError` 등 공통 베이스 존재 여부)을 먼저 확인한다(`python -c "import openai; print(openai.__version__)"` + 실제 속성 확인). 없거나 다르면 `responses.create` + 수동 `ChartPatternRead.model_validate()` 폴백으로 전환하고 이 계획을 갱신한다. **클라이언트 계층 한 곳만 바뀌므로 나머지 설계는 그대로 유효하다.**

### 6. LLM 실패는 추천 발송을 절대 막지 않는다

API 에러·레이트리밋·타임아웃·스키마 검증 실패 무엇이든 `PipelineResult.errors`에 기록만 하고 `sent`에는 정상 진행한다. ML #2의 설계 결정 5와 동일.

### 7. SDK를 직접 호출한다 (LangChain `deepagents` 미도입)

PRD §4는 `deepagents` 기반 서브에이전트 구성을 제안하지만, 현 코드베이스는 `data`/`rule_engine`/`risk_sizing`/`explainer`/`notifier`/`orchestrator`/`ml_predictor` 어느 모듈도 이를 쓰지 않는다. 이 신호원 하나만 에이전트 프레임워크를 도입하면 일관성이 깨지고, 실제로 필요한 것은 "구조화된 응답을 반환하는 단일 호출" 하나뿐이라 프레임워크의 이득이 없다(YAGNI). SDK 직접 호출을 유지한다. (참고: LangChain/`deepagents`는 특정 LLM 프로바이더에 종속되지 않는다 — `langchain-openai`로 GPT를 붙일 수도 있었지만, 이 결정은 어떤 프로바이더를 쓰느냐와 무관하게 유효하다.)

### 8. 프로바이더: OpenAI GPT (`gpt-5.6-luna`)

Claude가 아니라 OpenAI를 쓰기로 한 것은 이 프로젝트의 다른 어떤 설계 원칙과도 상충하지 않는다 — `client.py`가 SDK 호출을 캡슐화하는 유일한 파일이라는 구조(설계 결정 5 아래) 덕분에 프로바이더 교체가 그 파일 하나로 국한된다. `gpt-5.6-luna`(최저가·경량 등급)를 사용자가 확정했다. 2026-08-27 기준 공식 문서로 재확인한 가격은 §모델 선택 표를 참고.

---

## 신규 모듈 구조

`ml_predictor/` 패턴(models.py + 순수계산 + 진입점)을 그대로 따른다.

```
src/auto_stock/llm_chart_analyst/
├── __init__.py
├── models.py       # BarSummary, ChartSnapshot, ChartAnalysis, LLMConfig,
│                   # ChartPatternReader(Protocol) — dataclass는 전부 frozen + slots
├── schema.py       # ChartPatternRead — responses.parse의 text_format용 Pydantic 모델
├── snapshot.py     # build_snapshot(records, recent_bars=30) -> ChartSnapshot | None
├── prompt.py       # SYSTEM_PROMPT, build_user_prompt(snapshot) -> str
├── credentials.py  # load_llm_config() -> LLMConfig  (load_dotenv 패턴)
├── client.py       # OpenAIChartClient, LLMChartAnalystError — openai를 import하는 유일한 파일
└── analyst.py      # analyze(client, records) -> ChartAnalysis | None
                    # to_reasons(analysis, action) -> list[str]
```

`tests/llm_chart_analyst/`에 각 모듈당 대응 테스트 파일 + `conftest.py`.

### `models.py`

```python
@dataclass(frozen=True, slots=True)
class BarSummary:
    offset: int          # t-29 … t-0 (음수 아님, 0이 최신)
    open: float          # 구간 첫 종가 = 100 기준 정규화 지수
    high: float
    low: float
    close: float
    volume_ratio: float  # 20일 평균 거래량 = 1.0 기준

@dataclass(frozen=True, slots=True)
class ChartSnapshot:
    ticker: str                    # 프롬프트에 넣지 않음 — 결과 라벨링 전용
    market: str                    # 동일
    date: date                     # 최신 거래일, 동일
    bars: list[BarSummary]
    indicators: dict[str, float]   # ml_predictor.features.FEATURE_NAMES 11종 그대로

@dataclass(frozen=True, slots=True)
class ChartAnalysis:
    ticker: str
    market: str
    date: date
    direction: str        # "UP" | "DOWN" | "NEUTRAL"
    confidence: str       # "LOW" | "MEDIUM" | "HIGH"
    pattern_name: str
    rationale: str
    caveat: str | None
    model: str            # 실제 사용한 모델 ID — 감사 추적용, 문구에는 넣지 않음

@dataclass(frozen=True, slots=True)
class LLMConfig:
    api_key: str
    model: str
    max_tokens: int
    timeout_seconds: float
    max_retries: int
    max_calls_per_run: int

class ChartPatternReader(Protocol):
    def read_pattern(self, system_prompt: str, user_prompt: str) -> ChartPatternRead: ...
```

`ChartPatternReader` Protocol 덕분에 `analyst.py`는 `openai`를 전혀 import하지 않는다 — 테스트에서 가짜 리더를 넣기만 하면 되고, SDK가 설치돼 있지 않아도 도메인 로직 테스트가 돈다.

### `schema.py` — 출력 스키마

```python
class ChartPatternRead(BaseModel):
    direction: Literal["UP", "DOWN", "NEUTRAL"] = Field(
        description="향후 약 5거래일 방향성. 뚜렷한 패턴이 없으면 반드시 NEUTRAL."
    )
    confidence: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        description="주어진 수치가 이 판단을 얼마나 명확히 지지하는지. 확률/퍼센트가 아님."
    )
    pattern_name: str = Field(
        max_length=40,
        description="감지된 대표 차트/캔들 패턴 이름(한국어). 없으면 '뚜렷한 패턴 없음'.",
    )
    rationale: str = Field(
        max_length=200,
        description="어떤 수치가 근거인지 구체적으로 언급한 한국어 1~2문장.",
    )
    caveat: str | None = Field(
        default=None, max_length=120,
        description="이 판단을 무효화할 수 있는 조건. 없으면 null.",
    )
```

**설계 근거**

- **`confidence`를 0~1 실수가 아니라 3단계 서수로 받는 이유**: (a) LLM이 생성한 수치 신뢰도는 캘리브레이션되지 않아 "72%"가 아무 통계적 의미가 없다. (b) ML #2가 이미 "**상승확률 XX%**"라는 숫자 표기를 점유했다 — 두 번째 퍼센트가 같은 메시지에 나오면 사용자가 두 값을 비교 가능한 척도로 오독한다. 서수 라벨은 형식 자체가 달라 혼동이 구조적으로 불가능하다.
- **`direction`이 3값 enum인 이유**: `predictor.py::_stance`가 확립한 동의/상충/중립 3분기와 1:1 대응시켜 문구 생성 로직을 대칭으로 유지한다.
- **목표가/손절가/수량 필드가 없는 이유**: ATR 기반 손절·익절과 포지션 사이징은 PRD §6에서 확정된 `risk_sizing`의 소관이다. LLM이 숫자를 내놓으면 리스크 정책과 충돌하는 값이 같은 메시지에 실린다. 스키마에서 아예 제거하고 시스템 프롬프트에서도 금지한다.
- **`max_length` 제약의 실용적 이유**: 텔레그램 메시지 상한(4096자)에서 규칙엔진 근거 + 사이징 + ML 블록 + LLM 블록이 모두 들어가야 한다.

### `snapshot.py`

```python
RECENT_BARS = 30
SNAPSHOT_BASE_INDEX = 100.0

def build_snapshot(records: list[OHLCVRecord], recent_bars: int = RECENT_BARS) -> ChartSnapshot | None:
    """워밍업(최장 SMA60) 미충족이면 None — latest_feature_vector와 동일한 관례."""
```

**재사용**: 지표 계산을 새로 짜지 않는다. `ml_predictor.features.latest_feature_vector(records)`를 그대로 호출해 `FeatureVector.values`(11종)를 `indicators`에 넣는다. 이 함수는 이미 `rule_engine.indicators`의 `sma/rsi/macd/atr`를 재사용하고 후행 윈도우만 쓰는 causal 구현임이 테스트로 검증돼 있다(`test_feature_at_index_is_unchanged_when_future_bars_are_appended`). **새 지표 수식은 한 줄도 추가하지 않는다.**

`bars`는 최근 `recent_bars`개의 OHLCV를 구간 첫 종가 기준 지수(=100)로, 거래량은 20일 평균 대비 배율로 변환한다. 절대 가격 앵커링을 없애면서 캔들 형상(몸통/꼬리 비율, 갭)은 보존된다.

`RECENT_BARS = 30`인 이유: 30행 × 약 10토큰 ≈ 300토큰으로 비용이 작고, 일반적인 캔들·단기 차트 패턴(더블바텀, 삼각수렴, 갭, 장악형 등)이 관찰되기에 충분한 구간이다. `DEFAULT_LOOKBACK_DAYS = 120`(캘린더 ≈ 80거래일) 안에서 SMA60 워밍업과 함께 여유롭게 확보된다 — **오케스트레이터의 lookback 변경 불필요.**

### `prompt.py`

```python
SYSTEM_PROMPT = """당신은 주가 차트의 기술적 패턴을 해석하는 분석가다.

입력으로 최근 거래일의 정규화된 OHLCV 표와 기술적 지표 값만 주어진다.
종목명·종목코드·시장·실제 날짜·뉴스·재무·거시 정보는 의도적으로 제공하지 않는다.

규칙:
1. 주어진 수치만으로 판단한다. 제공되지 않은 정보(기업명, 업종, 뉴스, 실적)를
   추측하거나 지어내지 않는다.
2. 뚜렷한 패턴이 없으면 반드시 direction="NEUTRAL"을 선택한다.
   억지로 패턴을 만들어내지 않는다. NEUTRAL은 실패가 아니라 정상적인 답이다.
3. confidence는 수치가 판단을 얼마나 명확히 지지하는지를 LOW/MEDIUM/HIGH로만
   표현한다. 확률이나 퍼센트를 쓰지 않는다.
4. 목표가·손절가·매매 수량은 제시하지 않는다. 별도의 리스크 관리 모듈이 결정한다.
5. rationale은 한국어 1~2문장으로, 어떤 수치가 근거인지 구체적으로 언급한다.
6. 이 분석은 매매 결정 자체가 아니라, 다른 신호와 함께 사람에게 제시되는
   보조 참고 자료다. 단정적 확언을 피한다.
"""

def render_bar_table(snapshot: ChartSnapshot) -> str: ...
def render_indicators(snapshot: ChartSnapshot) -> str: ...
def build_user_prompt(snapshot: ChartSnapshot) -> str: ...
```

`build_user_prompt` 출력 형태:

```
아래는 어떤 상장 종목의 최근 30거래일 차트 요약이다.

[정규화 가격] 구간 첫 종가 = 100 기준
오프셋  시가     고가     저가     종가    거래량비
t-29   99.4   100.8    98.9   100.0     0.87
...
t-0   104.2   105.1   103.0   104.8     1.63

[기술적 지표] 최신 거래일 기준
RSI(14): 28.4
MACD 히스토그램/종가: -0.0031
종가/SMA20 - 1: -4.2%
종가/SMA60 - 1: -7.8%
SMA20/SMA60 - 1: -3.7%
ATR(14)/종가: 2.6%
20일 거래량비: 1.63
20일 채널 내 위치(0=저점, 1=고점): 0.18
수익률 1일/5일/20일: +0.8% / -3.1% / -6.4%

이 차트에서 관찰되는 패턴과 향후 약 5거래일 방향성을 판단하라.
```

**"약 5거래일"은 하드코딩하지 않고 `ml_predictor.labeling.LABEL_HORIZON_DAYS`를 import해 삽입한다.** ML #2와 LLM #3이 같은 예측 지평을 말하게 만들어야 두 보조 신호의 동의/상충 비교가 의미를 갖는다. ML 지평이 바뀌면 프롬프트도 자동으로 따라간다.

### `credentials.py`

```python
DEFAULT_MODEL = "gpt-5.6-luna"         # 사용자 확정
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2                # SDK 내장 백오프 재시도
DEFAULT_MAX_CALLS_PER_RUN = 20         # 비용 안전밸브

def load_llm_config() -> LLMConfig:
    load_dotenv()
    return LLMConfig(
        api_key=os.environ["OPENAI_API_KEY"],   # 없으면 KeyError — 진입점에서 즉시 실패
        model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        max_tokens=DEFAULT_MAX_TOKENS,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
        max_calls_per_run=int(os.environ.get("LLM_MAX_CALLS_PER_RUN", DEFAULT_MAX_CALLS_PER_RUN)),
    )
```

`notifier/credentials.py`가 확립한 "각 진입점이 필요한 시점에 `load_dotenv()`를 직접 호출, 없으면 `KeyError`" 패턴과 동일. 모델 ID는 `PRODUCTION_ALGORITHM = "logreg"`처럼 **상수 한 곳**에 두고 `OPENAI_MODEL` 환경변수로 코드 수정 없이 교체 가능하게 한다(`ML_TRAIN_UNIVERSE_SIZE` 선례).

### `client.py`

```python
class LLMChartAnalystError(Exception):
    pass

class OpenAIChartClient:
    """openai SDK를 import하는 유일한 파일. 모든 SDK 예외를 LLMChartAnalystError로 통일한다
    (notifier/telegram_bot.py가 TelegramNotificationError로 통일한 것과 동일한 관례)."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._calls_made = 0
        self._client = openai.OpenAI(
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    def read_pattern(self, system_prompt: str, user_prompt: str) -> ChartPatternRead: ...
```

---

## 오케스트레이터 통합

### 변경될 `pipeline.py`

```python
from auto_stock.llm_chart_analyst.analyst import analyze as analyze_chart
from auto_stock.llm_chart_analyst.analyst import to_reasons as to_chart_reasons
from auto_stock.llm_chart_analyst.models import ChartPatternReader
from auto_stock.ml_predictor.predictor import predict, to_reasons   # ← 이름 그대로 유지 (아래 주의)


def _llm_reasons(
    llm_client: ChartPatternReader | None, records: list[OHLCVRecord], action: str
) -> tuple[list[str], str | None]:
    """절대 raise하지 않는다 — LLM 실패가 추천 발송을 막아서는 안 된다(설계 결정 6)."""
    if llm_client is None:
        return [], None
    try:
        analysis = analyze_chart(llm_client, records)
        return to_chart_reasons(analysis, action), None
    except Exception as exc:
        return [], str(exc)


def run_recommendation_pipeline(
    cache, tickers, market, account, credentials,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ml_model: ModelBundle | None = None,
    llm_client: ChartPatternReader | None = None,
) -> PipelineResult:
    ...
    for candidate in generate_candidates(records):
        sizing = suggest_position(candidate, records, account)

        if ml_model is None and llm_client is None:
            # 회귀 잠금: 보조 신호가 하나도 없으면 2-위치인자 호출을 그대로 보존한다
            # (tests/orchestrator/test_pipeline.py:163의 assert_called_once_with(ANY, ANY))
            explanation = generate_explanation(candidate, sizing)
        else:
            extra_reasons: list[str] = []
            ml_r, ml_error = _ml_reasons(ml_model, records, candidate.action)
            extra_reasons.extend(ml_r)
            llm_r, llm_error = _llm_reasons(llm_client, records, candidate.action)
            extra_reasons.extend(llm_r)

            explanation = generate_explanation(candidate, sizing, extra_reasons=extra_reasons)

            if ml_error is not None:
                errors.append((ticker, f"ML 예측 실패: {ml_error}"))
            if llm_error is not None:
                errors.append((ticker, f"LLM 차트분석 실패: {llm_error}"))

        send_notification(explanation, credentials)
        sent.append(explanation)
```

**핵심 포인트 4가지**

1. **분기 조건이 `ml_model is None and llm_client is None`이어야 한다.** 기존 회귀 테스트가 `generate_explanation`을 2-위치인자로만 호출할 것과 2-인자 lambda로 모킹할 것을 요구한다(`test_pipeline.py:102`, `:149`, `:163`). 소스별 `if`로 쪼개면 이 세 테스트가 깨진다.
2. **`_ml_reasons`는 무변경.** 이미 `ml_model is None`이면 `([], None)`을 반환하므로 병합 분기에 그대로 들어와도 안전하다. LLM만 켠 경우 ML 코드 경로는 `predict`를 호출하지 않는다.
3. **두 헬퍼가 완전히 독립적**이라 한쪽 실패가 다른 쪽 문구를 삼키지 않는다. LLM이 죽어도 ML 근거는 그대로 실린다(신규 테스트로 잠금).
4. **import 이름 주의**: 기존 테스트가 `mocker.patch("auto_stock.orchestrator.pipeline.to_reasons")`(`test_pipeline.py:176`)와 `...pipeline.predict`(`:152`, `:174`, `:202`)를 정확한 이름으로 패치한다. ML 쪽 `to_reasons`/`predict`는 **이름을 바꾸지 않고**, LLM 쪽만 `analyze_chart`/`to_chart_reasons`로 alias한다. 대칭성이 살짝 깨지지만 기존 테스트 수정 0건이라는 이득이 크다.
   > 대안(양쪽 다 `to_ml_reasons`/`to_chart_reasons`로 대칭 개명)을 택하려면 `test_pipeline.py:176`의 패치 타깃 한 줄만 고치면 된다. **사용자 확인 필요 항목**으로 남긴다.

### 병합 순서: ML → LLM

결정론적·재현 가능한 수치 신호를 먼저, 확률적 정성 신호를 뒤에 놓는다. `#4` 뉴스는 그 뒤에 붙는다.

> PRD §5.1 예시 문장은 "…LLM 차트 분석상 단기 반등 패턴 감지. 예측 모델 신뢰도 XX%."로 LLM을 먼저 두지만, 이는 예시일 뿐이며 §5.1 자체가 "구체적인 요약 템플릿은 TBD"라고 명시한다. 순서는 저비용 변경이므로 Phase 3 육안 확인 후 조정 가능.

### 상충 시 표현: 중재하지 않는다

ML이 "상승확률 38%(BUY와 상충)", LLM이 "반등 패턴 감지(BUY에 동의)"를 동시에 말할 수 있다. **파이프라인은 이를 중재하지 않고 두 문장을 그대로 병렬 노출한다.**

- 각 소스가 이미 규칙엔진 신호 대비 자신의 동의/상충/중립을 명시하므로, 두 소스가 엇갈리면 사용자에게 그대로 보인다 — 추가 로직 없이 이미 가시적이다.
- 어떤 형태의 중재(가중 평균, 다수결, "엇갈림 경고")도 PRD §5가 TBD로 남긴 **앙상블 가중치 결정**에 해당한다. 백테스트 없이 가중치를 정하면 근거 없는 숫자가 된다(ML #2 계획의 동일 논거).
- `orchestrator.md`의 "오케스트레이터에는 새로운 계산 로직이 없다 — 배선만" 원칙도 지킨다.
- 신호원이 3개(#4 추가)가 되어 사람이 눈으로 비교하기 어려워지면 그때 재검토한다.

### 고지 문구 중복은 수용한다

두 신호를 다 켜면 "(ML 신호는 백테스트 검증 전 참고용 보조 지표입니다)"와 "(LLM 차트해석은 백테스트 미검증 정성 신호입니다)"가 함께 실린다. 다소 장황하지만, 공용 고지로 합치려면 `generator.py`를 수정하거나 두 모듈을 상호 의존시켜야 한다 — 둘 다 원칙 위반이다. **각 신호원은 자기 고지를 스스로 책임진다.**

### `analyst.py` 문구 생성

```python
LLM_DISCLAIMER = "(LLM 차트해석은 백테스트 미검증 정성 신호입니다)"
CONFIDENCE_LABELS = {"LOW": "낮음", "MEDIUM": "보통", "HIGH": "높음"}
```

`predictor.py::_stance`와 동형:

| direction | action | 문구 |
|---|---|---|
| UP | BUY | `LLM 차트분석: {pattern_name} 감지 — BUY 신호에 동의 (신뢰도 {label})` |
| DOWN | SELL | `LLM 차트분석: {pattern_name} 감지 — SELL 신호에 동의 (신뢰도 {label})` |
| DOWN | BUY | `LLM 차트분석: {pattern_name} 감지 — BUY 신호와 상충됩니다 — 주의 (신뢰도 {label})` |
| UP | SELL | `LLM 차트분석: {pattern_name} 감지 — SELL 신호와 상충됩니다 — 주의 (신뢰도 {label})` |
| NEUTRAL | any | `LLM 차트분석: {pattern_name} (방향성 중립, 신뢰도 {label})` |

이어서 `rationale`, `caveat`(있을 때만 `"단서: {caveat}"`), 마지막에 항상 `LLM_DISCLAIMER`. `analysis is None`이면 `[]`.

---

## 모델 선택 (확정: `gpt-5.6-luna`)

2026-08-27 공식 문서로 재확인한 현재 OpenAI 모델 라인업(GPT-5.6 계열, `chat.completions`가 아니라 Responses API 기준):

| 모델 | 입력 $/1M | 출력 $/1M | 용도 | 이 신호원 적합성 |
|---|---|---|---|---|
| **gpt-5.6-sol** | $4.00 | $20.00 | 복잡한 추론/코딩 | 과잉 스펙 — 이 과제는 3분류+2문장 구조화 출력이라 최상위 추론 능력이 필수는 아니다 |
| **gpt-5.6-terra** | $2.00 | $12.00 | 성능-비용 균형 | "억지 패턴 생성 억제"(NEUTRAL 선택 성향)가 luna보다 나을 가능성 — sol/luna 사이 대안 |
| **gpt-5.6-luna** (확정) | $0.20 | $1.20 | 저비용 대규모 워크로드 | 사용자가 "gpt-5-mini 또는 유사 경량형"으로 요청 — 가장 근접한 등급. 호출이 규칙엔진 후보 수(설계 결정 4)로 제한되므로 상위 등급 대비 절대 비용 차이 자체는 작지만, 사용자 의도(경량·저비용)에 맞춰 확정 |

**`DEFAULT_MODEL = "gpt-5.6-luna"`로 확정.** `credentials.py::DEFAULT_MODEL` 상수 한 곳 + `OPENAI_MODEL` 환경변수로 코드 수정 없이 언제든 `terra`/`sol`로 교체 가능하게 설계했다 — Phase 5 표본 검증에서 "환각/억지 패턴" 실패 모드가 두드러지면 재검토.

정확한 모델 ID 문자열·가격은 구현 시점에 재확인한다(OpenAI 가격 정책은 자주 갱신됨).

---

## 에러 처리 · 실패 격리 (4중 방어)

**1층 — 데이터 부족: 예외를 만들지 않는다**
`build_snapshot`이 워밍업 미충족 시 `None` → `analyze`가 `None` 반환 → `to_reasons(None, action)`이 `[]`. **API 호출 자체가 발생하지 않는다.**

**2층 — SDK 예외 통일 (`client.py`)**

| 잡는 예외 | 변환 메시지 |
|---|---|
| `openai.APITimeoutError` | `LLM 응답 시간 초과({timeout}s)` |
| `openai.RateLimitError` | `LLM 레이트리밋 초과` |
| `openai.APIConnectionError` | `LLM 연결 실패` |
| `openai.AuthenticationError` | `LLM 인증 실패 — OPENAI_API_KEY 확인 필요` |
| `openai.APIStatusError`(공통 베이스, `BadRequestError`/`InternalServerError`/`NotFoundError`/`PermissionDeniedError`/`UnprocessableEntityError` 등 포함) | `LLM API 오류(status={exc.status_code})` |
| `pydantic.ValidationError` | `LLM 응답 스키마 검증 실패` |
| `output_parsed is None` | `LLM 응답 파싱 결과가 비어 있음` |

**포착 순서가 중요하다**: `APITimeoutError`/`RateLimitError`/`AuthenticationError`가 `APIStatusError`의 하위 클래스일 가능성이 높으므로(정확한 계층은 Phase 0에서 재확인) 반드시 좁은 것부터 잡는다.

**메시지에 API 키·전체 응답 본문·프롬프트 원문을 절대 포함하지 않는다.**

**3층 — 파이프라인 격리 (`_llm_reasons`)**
`except Exception`으로 넓게 잡아 `([], str(exc))` 반환. 절대 raise하지 않는다.

**4층 — 크리덴셜 부재는 진입점에서 즉시 실패**
`load_llm_config()`의 `os.environ["OPENAI_API_KEY"]`가 `KeyError`를 던진다.

### 재시도 · 타임아웃 · 비용 통제

- **재시도는 직접 구현하지 않는다.** `openai.OpenAI(max_retries=2)`가 429/5xx에 지수 백오프 재시도를 이미 제공한다.
- **타임아웃 30초.**
- **호출 예산 상한**: `OpenAIChartClient`가 `_calls_made` 카운터를 갖고 `max_calls_per_run`(기본 20) 초과 시 `LLMChartAnalystError`를 던진다. `_llm_reasons`가 잡아 `errors`에만 기록하므로 나머지 추천은 정상 발송된다.
  - **왜 클라이언트에 두는가**: PRD §7.3은 전 종목 스캔을 목표로 하므로, 실수로 대형 유니버스에 LLM을 켜면 실제 금전 사고가 된다.
- **폴백은 "그냥 스킵"** — 실패 시 대체 LLM/대체 모델로 재시도하지 않는다.

---

## 테스트 전략

**절대 원칙: 단위테스트는 실제 OpenAI API를 호출하지 않는다.** `conftest.py`에 autouse 픽스처를 두어 `OPENAI_API_KEY`를 더미 값으로 monkeypatch하고 `openai.OpenAI`를 Mock으로 대체한다.

- `test_snapshot.py`: 워밍업 미충족→None, 스케일 불변성(KRX/NASDAQ 혼용 안전성), 인과성, `indicators` 키 일치
- `test_prompt.py`: 프롬프트에 ticker/market/실제날짜 미등장(환각 방어 잠금), `action` 파라미터 없음(동조 방어 잠금), `LABEL_HORIZON_DAYS` 연동
- `test_client.py`: SDK 완전 모킹, 예외 매핑 6종, 예산 상한, API 키 미유출
- `test_analyst.py`: `ticker`/`market`/`date`를 응답이 아니라 `records`에서 채움, stance 6케이스, disclaimer 항상 마지막
- `tests/orchestrator/test_pipeline.py` 신규 7케이스(기존 7개 무수정 통과 필수): 무보조신호 회귀, LLM단독, ML+LLM 병합 순서, LLM실패 격리, 교차 실패 격리(LLM실패해도 ML문구 유지, 역방향), 이중실패 시 errors 2건

전체 커버리지 80% 이상 유지(현재 98%).

---

## 의존성 · 설정 변경

| 파일 | 변경 |
|---|---|
| `pyproject.toml` | `dependencies`에 `"openai"`, `"pydantic>=2"` 추가 |
| `.env.example` | `OPENAI_API_KEY=` 추가. `OPENAI_MODEL=`, `LLM_MAX_CALLS_PER_RUN=`은 선택 항목 |
| `.gitignore` | Phase 0에서 `.env`가 이미 무시되는지 `git check-ignore .env`로 확인만 |
| `docs/PRD.md` | §5 LLM 계층에 확정 내용 반영 |
| `docs/IMPLEMENTATION_PLAN.md` | "신호 소스 확장" §2 갱신 |
| `docs/CODEBASE_REVIEW.md` | 모듈 표 + "아직 안 된 것"에서 #3 제거 |

**변경하지 않는 것**: `explainer/generator.py`, `rule_engine/*`, `risk_sizing/*`, `notifier/*`, `data/*`, `ml_predictor/*`(읽기 재사용만), `orchestrator/models.py`, 기존 `scripts/run_recommendations*.py`.

## 재사용할 기존 코드

- `ml_predictor.features.latest_feature_vector`/`FEATURE_NAMES` — 지표 계산 전체 재사용
- `ml_predictor.labeling.LABEL_HORIZON_DAYS`/`forward_returns` — 예측 지평 동기화, 표본 검증
- `explainer.generator.generate_explanation(..., extra_reasons=)` — 확장 포인트, 무수정
- `orchestrator.pipeline._ml_reasons` — `_llm_reasons`의 형태·계약 복제
- `ml_predictor.predictor._stance` — 동의/상충/중립 문구 패턴 복제
- `notifier.credentials.load_telegram_credentials` — `load_dotenv()` 패턴
- `notifier.telegram_bot.TelegramNotificationError` — 예외 통일 관례
- `scripts/verify_pykrx.py`, `run_recommendations_with_ml.py` — verify/실행 스크립트 골격

---

## 구현 순서 (Phase)

각 Phase는 독립적으로 머지 가능하다.

**Phase 0 — 의존성·SDK 게이트 (코드 작성 전, 비용 0)**: `openai` 설치 → 기존 126개 테스트 재실행(회귀 확인) → `responses.parse`/`output_parsed` 존재 확인 + 예외 클래스 계층 재확인(없으면 폴백 경로로 계획 갱신) → `.env.example`에 `OPENAI_API_KEY=` 추가 → API 키 발급/설정(사용자 작업 — 아직 미완료, 이 Phase의 완료 조건).

**Phase 1 — 순수 계층 (API 호출 0)**: `models.py` → `schema.py` → `snapshot.py` → `prompt.py` + 테스트. 독립 머지 가능.

**Phase 2 — 클라이언트·분석 계층 (전부 모킹)**: `credentials.py` → `client.py` → `analyst.py` + 테스트. 독립 머지 가능.

**Phase 3 — 배선 + 실제 API 최초 호출**: `pipeline.py` 확장 → `test_pipeline.py` 신규 7케이스 → `scripts/verify_llm_chart_analyst.py`(여기서 처음 실제 API 호출) → `scripts/run_recommendations_with_signals.py`(ML+LLM 동시) → 텔레그램 메시지 육안 확인.

**Phase 4 — 문서화 + 리뷰**: `docs/design/llm-chart-analyst.md` 작성 → PRD/IMPLEMENTATION_PLAN/CODEBASE_REVIEW 갱신 → `code-reviewer` + `security-reviewer`(외부 API·크리덴셜 취급이므로 필수) 서브에이전트 리뷰.

**Phase 5 (선택) — 표본 신뢰도 검증**: `scripts/sample_llm_chart_analyst.py` — 5~10종목×5개 과거 시점=25~50회 호출로 예측과 실제 수익률을 나란히 출력(자동 판정 없음, 육안 확인용).

---

## 이번 스코프가 아닌 것

| 이연 항목 | 왜 이번이 아닌가 |
|---|---|
| 차트 이미지(vision) 입력 | 렌더링 의존성 + 비용·레이턴시 수 배. 텍스트 요약이 명백히 부족할 때만 재검토 |
| 대량 백테스트 | 과거 시점마다 API 호출 필요 — ML #2와의 근본적 차이. 대신 Phase 5 표본 검증 |
| 응답 캐싱 | 현 진입점은 1회 실행 후 종료라 히트 불가. `scheduler.py` 배선 시 재검토 |
| 프롬프트 캐싱 | 시스템 프롬프트가 최소 캐시 단위 미만으로 추정, 호출 수도 적어 이득 미미 |
| LLM 단독 후보 생성/필터링 | PRD §10. 아래 승격 규칙 충족 전까지 불가 |
| 4계층 앙상블 가중치 확정, ML/LLM 상충 자동 중재 | PRD §5 TBD — 백테스트 없이 정하면 근거 없는 숫자 |
| 뉴스/공시를 프롬프트에 주입 | #4의 스코프 — 지금 섞으면 신호원별 기여 분리 불가 |
| 멀티 프로바이더 지원(Claude/Gemini 등을 선택형으로 추가) | 사용자와 논의 후 보류 확정. `ChartPatternReader` Protocol이 이미 `client.py`(SDK 호출)와 `analyst.py`(도메인 로직)를 분리해뒀으므로, 나중에 필요해지면 `OpenAIChartClient`와 나란히 `AnthropicChartClient`/`GeminiChartClient`를 추가 + 환경변수 기반 팩토리 함수 하나만 얹으면 된다 — 아키텍처 재작업이 아니라 확장이다. 지금 하지 않는 이유는 프로바이더별 의존성·에러 매핑·구조화 출력 방식·테스트가 N배로 늘어나는데, 아직 어떤 프로바이더가 실제로 필요한지 검증되지 않았기 때문(YAGNI) |
| 비동기/병렬 호출, 시장별 프롬프트 분기, LangChain deepagents 도입, 프롬프트 A/B 실험 | 각각 불필요/시기상조(YAGNI) |

**승격 규칙**: ML #2와 동일 — (1) 3개 이상 비중첩 기간에서 일관된 우위, (2) 거래비용 반영 전략 백테스트로 개선 입증, (3) 데이터 정합성 검증 해소, (4) 앙상블 방식 확정 — 네 가지 모두 충족해야 후보 생성 권한으로 승격 검토.

---

## 검증 (Verification)

- Phase 0: `pytest -q`(기존 126개 회귀 없음), `messages.parse` 존재 확인, `git check-ignore .env`
- Phase 1: `pytest tests/llm_chart_analyst/test_snapshot.py tests/llm_chart_analyst/test_prompt.py -v`
- Phase 2: `pytest tests/llm_chart_analyst/test_client.py tests/llm_chart_analyst/test_analyst.py -v`
- Phase 3: `pytest tests/orchestrator/test_pipeline.py -v`(기존 7 + 신규 7)
- 전체: `pytest --cov=src --cov-report=term-missing`(80%+)
- 수동(API 1회): `.venv/Scripts/python scripts/verify_llm_chart_analyst.py`
- 수동(dry-run, 비용 0): `LLM_VERIFY_DRY_RUN=1`로 프롬프트만 출력
- 수동 E2E: `run_recommendations_with_signals.py`로 텔레그램에 ML+LLM 두 블록 육안 확인
- 마지막: `code-reviewer` + `security-reviewer` 서브에이전트

## API 키

**새로 필요: `OPENAI_API_KEY`.** 이 프로젝트에서 처음 쓰는 유료 외부 API 신호원이다(FDR/pykrx/pandas-ta/scikit-learn 무료, 텔레그램 무료). `.env.example`에 추가하고 `notifier/credentials.py` 패턴을 그대로 따른다. 코드에 절대 하드코딩하지 않는다. **아직 발급 전 — Phase 3(실제 API 호출)의 선행 조건이며, Phase 0~2(의존성/순수 계층/모킹 기반 클라이언트·분석 계층)는 키 없이 진행 가능하다.**

## 결정 현황

| 항목 | 상태 |
|---|---|
| 모델 선택 | **확정 — `gpt-5.6-luna`** |
| `OPENAI_API_KEY` 발급 | **미완료 — 사용자가 발급 예정. Phase 0~2는 선행 가능, Phase 3부터 필요** |
| `to_reasons` 네이밍 | 미확정 — 권고안(LLM만 `to_chart_reasons`로 alias, 기존 테스트 무수정)대로 진행. 이견 있으면 알려달라 |
| `DEFAULT_MAX_CALLS_PER_RUN = 20` | 미확정 — 권고값대로 진행. 이견 있으면 알려달라 |
| Phase 5 표본 검증 | **보류 확정 — 나중에 별도 요청 시 진행** |
