# 뉴스/공시 분석 에이전트 (#4) 구현 계획

> `docs/IMPLEMENTATION_PLAN.md`의 "신호 소스 확장" §3(뉴스/공시 #4)을 실행 가능한 수준으로 구체화한 사전 계획 문서. `docs/design/ml-predictor-plan.md`/`docs/design/llm-chart-analyst-plan.md`와 같은 위치·형식이며, 구현 완료 후 실제 결과를 기록하는 `docs/design/news-disclosure.md`를 별도로 작성한다(Phase 4).

## 배경

MVP-0, ML #2, LLM차트분석 #3(배선 완료, Phase 4 리뷰 대기 중)까지 진행됐다. PRD §4가 정의한 4개 신호 계층 중 마지막인 **뉴스/공시 분석 에이전트 (#4)**를 다음으로 진행한다.

PRD §7.2는 데이터 소스를 이미 조사 완료로 확정해 뒀다: **DART 오픈API(전자공시) 1순위 확정**, 빅카인즈(뉴스)는 유력 후보지만 rate limit·유료 티어가 미확인(TBD), GDELT는 한국어 커버리지가 약해 보조 후보. §5는 "감성분석·핵심 이벤트 추출 후 시그널로 반영"을, §10은 "오분류 가능성 있어 단독 매매 트리거가 아닌 보조 시그널로만 사용"을 못박았다.

**사용자 결정 사항 (이번 계획 착수 시 확인)**:
- **범위**: 1차는 **DART(공시)만** — 빅카인즈는 rate limit/유료 티어 확인 후 2단계로 별도 착수. GDELT는 이번 스코프 밖(PRD상 보조 후보일 뿐이며 채택 확정 아님).
- **감성·이벤트 분석 방식**: **LLM 재사용(OpenAI, #3와 동일 패턴)** — 별도 경량 분류기(로컬 NLP 모델)는 채택하지 않음.
- **API 키**: DART 인증키 아직 미발급. `OPENAI_API_KEY`(#3용)와 별개로 신규 발급 필요.

**목표**: 종목별 최근 공시 목록을 DART API로 조회하고, LLM으로 각 공시의 시장 영향(호재/악재/중립)을 해석해 규칙엔진 후보에 대한 **보조 근거**로 파이프라인에 배선한다. ML #2·LLM차트 #3과 동시에 켤 수 있어야 한다.

## 핵심 설계 결정

### 1. "감성분석"이 아니라 "공시 영향 해석"이다

PRD §5의 원래 표현("감성분석·핵심 이벤트 추출")은 뉴스 기사를 염두에 둔 것이다. 공시는 기업이 직접 제출하는 공식 문서라 뉴스처럼 어조·논조가 없다 — "유상증자 결정", "감사보고서 제출", "임원 지분 변동" 같은 **사실 통지**다. 따라서 이 신호원이 실제로 하는 일은 감성분석이 아니라 **"이 유형의 공시가 통상 주가에 호재/악재/중립 중 무엇으로 해석되는지 + 왜 그런지"**를 LLM에 판단시키는 것이다. 스키마의 필드명도 `sentiment`가 아니라 `market_impact`로 둔다 — 나중에 빅카인즈(뉴스, 진짜 감성분석이 필요)를 추가할 때 이름 충돌·의미 혼동을 피하기 위함이다.

### 2. 입력은 공시 원문 전체가 아니라 보고서명 + 메타데이터 목록

DART의 개별 공시 원문(`document.xml`)은 종종 수십 페이지 분량의 정형 문서(재무제표, 사업보고서 등)라 전체를 LLM에 넣는 것은 (a) 토큰 비용이 과도하고, (b) 이 신호원이 필요로 하는 것은 원문 정밀 분석이 아니라 "최근 어떤 공시들이 있었는지"에 대한 개괄적 해석이다. LLM차트분석 #3이 원본 캔들 이미지 대신 수치 요약을 쓴 것과 같은 원칙이다.

`list.json`(공시검색) 응답의 `report_nm`(보고서명), `rcept_dt`(접수일자), `rm`(비고 — 정정/첨부 등 플래그)만 최근 N건 모아 LLM에 전달한다. 원문 조회(`document.xml`)는 이번 스코프에 포함하지 않는다.

### 3. 종목코드 → DART 고유번호(`corp_code`) 매핑이 새로 필요하다

DART API는 KRX 종목코드(`005930`)가 아니라 자체 `corp_code`(8자리)로 종목을 식별한다. DART가 제공하는 `corpCode.xml`(전체 상장·비상장법인 매핑, zip 압축) 하나를 다운로드해 로컬에 캐싱하고, 종목코드로 역조회한다. 이 매핑은 자주 바뀌지 않으므로(신규상장 시에만 갱신) 파일 mtime 기준 일정 기간(예: 7일) 지나면만 재다운로드한다 — pykrx의 매일 재로그인과 달리 캐싱이 실제로 유효한 케이스다.

이 매핑 유틸은 **데이터 수집 계층(#0)의 확장**으로 분류한다 — `pykrx_source.py`가 "FDR이 약한 부분을 보조"하는 것과 같은 위치의 공용 인프라이지, LLM 해석 로직의 일부가 아니다.

### 4. `extra_reasons` 경유 보조 신호 (ML #2·LLM #3과 동일)

자체 `Candidate`를 만들지 않고, **규칙엔진 후보가 이미 존재하는 티커에 대해서만** 최근 공시를 조회해 `extra_reasons`로 보조 문장을 추가한다. `explainer/generator.py`는 한 줄도 수정하지 않는다. 병합 순서는 ML → LLM차트 → **뉴스/공시(신규, 항상 마지막)** — `llm-chart-analyst-plan.md`가 이미 "#4 뉴스는 그 뒤에 붙는다"고 예정해 둔 순서 그대로다.

### 5. 공시가 없는 경우가 정상이다 (규칙엔진의 "신호 없음"과 동형)

최근 조회 기간 내 공시가 없는 종목이 대다수다. `fetch_disclosures`가 빈 리스트를 반환하면 `analyze()`는 LLM을 호출하지 않고 `None`을 반환한다 — API 호출 자체가 발생하지 않아 불필요한 비용이 없다. LLM차트분석 #3의 "워밍업 미충족 시 `None`, API 호출 0" 방어와 동일한 층위다.

### 6. DART 호출과 OpenAI 호출은 독립적으로 실패 격리한다

DART 조회 실패(네트워크, rate limit, 잘못된 corp_code)와 LLM 해석 실패(API 에러, 스키마 검증 실패)는 서로 다른 실패 지점이다. `_news_reasons` 헬퍼 내부에서 두 단계를 각각 try/except로 감싸 **어느 쪽이 실패해도 절대 raise하지 않고** `([], str(exc))`를 반환한다 — ML #2 설계 결정 5, LLM #3 설계 결정 6과 동일한 계약.

### 7. SDK/HTTP를 직접 호출한다 (프레임워크 미도입)

DART는 REST+JSON이라 이미 프로젝트에 있는 `requests`로 충분하다(신규 의존성 없음). OpenAI 클라이언트는 #3의 `client.py`와 동일한 패턴(`responses.parse` + Pydantic)을 쓰되, **코드는 공유하지 않고 이 모듈 안에 독립적으로 둔다** — `_ml_reasons`/`_llm_reasons`가 구조는 같지만 코드는 공유하지 않는 이 프로젝트의 기존 관례(`llm-chart-analyst-plan.md` 참고)를 그대로 따른다.

---

## 신규 모듈 구조

### 데이터 수집 계층 확장 (`src/auto_stock/data/`)

```
src/auto_stock/data/
├── models.py               # DisclosureRecord(corp_code, ticker, report_name, filed_date, remark) 추가
└── sources/
    └── dart_source.py      # resolve_corp_code(ticker) -> str | None  (corpCode.xml 캐싱)
                             # fetch_disclosures(corp_code, start, end) -> list[DisclosureRecord]
```

### 해석 계층 (`src/auto_stock/news_disclosure/`)

`llm_chart_analyst/` 패턴을 그대로 따른다.

```
src/auto_stock/news_disclosure/
├── __init__.py
├── models.py       # DisclosureSummary, DisclosureAnalysis, LLMConfig, DisclosureReader(Protocol)
├── schema.py       # DisclosureRead — responses.parse의 text_format용 Pydantic 모델
├── prompt.py       # SYSTEM_PROMPT, build_user_prompt(summary) -> str
├── credentials.py  # load_llm_config() -> LLMConfig  (OPENAI_API_KEY 재사용, 모델은 별도 env var로 재정의 가능)
├── client.py       # OpenAIDisclosureClient, NewsDisclosureError — openai를 import하는 유일한 파일
└── analyst.py       # analyze(reader, disclosures, ticker, market, as_of) -> DisclosureAnalysis | None
                     # to_reasons(analysis, action) -> list[str]
```

`tests/data/test_dart_source.py`, `tests/news_disclosure/`(각 모듈당 대응 테스트 + `conftest.py`).

### `data/models.py` 추가분

```python
@dataclass(frozen=True, slots=True)
class DisclosureRecord:
    corp_code: str
    ticker: str
    report_name: str      # DART report_nm
    filed_date: date      # DART rcept_dt
    remark: str           # DART rm (정정/첨부 등 플래그, 없으면 "")
```

### `data/sources/dart_source.py`

```python
CORP_CODE_CACHE_PATH = "data/dart_corp_codes.json"   # .gitignore 대상 (data/*.duckdb와 동일 취급)
CORP_CODE_REFRESH_DAYS = 7

def resolve_corp_code(ticker: str) -> str | None:
    """corpCode.xml을 캐시에서 읽거나(오래됐으면 재다운로드) ticker로 역조회."""

def fetch_disclosures(corp_code: str, start: date, end: date) -> list[DisclosureRecord]:
    """DART list.json 호출. 결과 없으면 빈 리스트(예외 아님)."""
```

**재시도**: DART 공식 문서 기준 요청 한도는 넉넉한 편(무료 API, 유료 결제 항목 아님)이라 OpenAI처럼 별도 호출 예산 상한을 강제하지는 않되, 네트워크 오류는 `requests`의 기본 예외를 그대로 전파해 상위(`_news_reasons`)에서 잡는다.

### `news_disclosure/models.py`

```python
@dataclass(frozen=True, slots=True)
class DisclosureSummary:
    ticker: str
    market: str
    as_of: date
    items: list[DisclosureRecord]   # 최근 N건, 최신순

@dataclass(frozen=True, slots=True)
class DisclosureAnalysis:
    ticker: str
    market: str
    as_of: date
    market_impact: str     # "POSITIVE" | "NEGATIVE" | "NEUTRAL"
    confidence: str        # "LOW" | "MEDIUM" | "HIGH"
    key_event: str
    rationale: str
    caveat: str | None
    model: str              # 감사 추적용

class DisclosureReader(Protocol):
    def read_disclosures(self, system_prompt: str, user_prompt: str) -> DisclosureRead: ...
```

`market_impact`를 `sentiment`가 아닌 이 이름으로 둔 이유는 핵심 설계 결정 1 참고. `direction`(UP/DOWN/NEUTRAL, #3)이 아니라 `POSITIVE/NEGATIVE/NEUTRAL`로 어휘를 다르게 둔 것도 의도적이다 — #3은 가격 움직임 방향, #4는 공시 자체의 성격이라는 차이를 텔레그램 메시지에서도 구분되게 한다.

### `news_disclosure/schema.py`

```python
class DisclosureRead(BaseModel):
    market_impact: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = Field(
        description="이 공시들이 통상적으로 주가에 미치는 영향 방향. 뚜렷하지 않으면 NEUTRAL."
    )
    confidence: Literal["LOW", "MEDIUM", "HIGH"]
    key_event: str = Field(max_length=60, description="가장 중요한 공시 1건의 핵심 내용 요약(한국어)")
    rationale: str = Field(max_length=200)
    caveat: str | None = Field(default=None, max_length=120)
```

`confidence`를 서수로 두는 이유, 목표가/수량 필드가 없는 이유는 `llm-chart-analyst-plan.md`의 동일 근거를 그대로 따른다(중복 설명 생략).

### `news_disclosure/prompt.py`

```python
SYSTEM_PROMPT = """당신은 기업 공시가 주가에 미치는 영향을 해석하는 애널리스트다.

입력으로 특정 종목의 최근 공시 목록(보고서명·접수일자·비고)만 주어진다.
종목명·종목코드·시장·실제 현재가·뉴스·재무제표 원문은 제공되지 않는다.

규칙:
1. 보고서명과 통상적인 공시 유형 지식만으로 판단한다. 제공되지 않은 정보를 추측하지 않는다.
2. 뚜렷하게 호재/악재로 분류되는 유형이 아니면 반드시 market_impact="NEUTRAL"을 선택한다.
3. confidence는 LOW/MEDIUM/HIGH 서수로만 표현한다. 확률·퍼센트를 쓰지 않는다.
4. 목표가·매매 수량은 제시하지 않는다.
5. 이 분석은 매매 결정 자체가 아니라 보조 참고 자료다. 단정적 확언을 피한다.
"""
```

`build_user_prompt(summary)`는 종목명/티커/실제 날짜를 프롬프트에 넣지 않는다 — #3의 환각 방어(핵심 설계 결정 2) 원칙을 그대로 적용한다. 규칙엔진 `action`도 넣지 않는다(동조 방어, #3 핵심 설계 결정 3과 동일) — `analyze()`는 `action`을 받지 않고, `to_reasons(analysis, action)`이 코드에서 동의/상충/중립을 비교한다.

### `news_disclosure/credentials.py`

```python
DEFAULT_MODEL = "gpt-5.6-luna"   # #3과 동일 등급 — 구조화 3분류+짧은 요약 과제
DISCLOSURE_LOOKBACK_DAYS = 90    # 최근 3개월 공시만 조회
MAX_DISCLOSURES_PER_QUERY = 10   # 프롬프트에 넣을 최대 건수

def load_llm_config() -> LLMConfig:
    load_dotenv()
    return LLMConfig(
        api_key=os.environ["OPENAI_API_KEY"],   # #3과 동일 키 재사용, 신규 발급 불필요
        model=os.environ.get("NEWS_OPENAI_MODEL", DEFAULT_MODEL),
        ...
    )
```

`OPENAI_API_KEY`는 #3과 **공유**한다(같은 OpenAI 계정, 같은 크리덴셜) — 신규로 발급받아야 하는 것은 **DART 인증키뿐**이다. 모델 등급만 `NEWS_OPENAI_MODEL`로 독립 재정의 가능하게 해 나중에 #3과 다른 등급이 필요해져도 코드 변경이 없게 한다.

### `data/sources/dart_source.py`의 크리덴셜

```python
# data/sources/dart_source.py 내부
def _load_dart_api_key() -> str:
    load_dotenv()
    return os.environ["DART_API_KEY"]   # 없으면 KeyError
```

`notifier/credentials.py`가 확립한 "필요 시점에 직접 `load_dotenv()`" 패턴을 그대로 따른다.

---

## 오케스트레이터 통합

### 변경될 `pipeline.py`

```python
from auto_stock.data.sources.dart_source import fetch_disclosures, resolve_corp_code
from auto_stock.news_disclosure.analyst import analyze as analyze_disclosures
from auto_stock.news_disclosure.analyst import to_reasons as to_disclosure_reasons
from auto_stock.news_disclosure.models import DisclosureReader


def _news_reasons(
    news_client: DisclosureReader | None, ticker: str, market: str, action: str
) -> tuple[list[str], str | None]:
    """절대 raise하지 않는다 — DART 조회든 LLM 해석이든 실패가 추천 발송을 막아서는 안 된다."""
    if news_client is None:
        return [], None
    try:
        corp_code = resolve_corp_code(ticker)
        if corp_code is None:
            return [], None  # DART 미등록 종목(예: 나스닥) — 에러 아님, 조용히 스킵
        end = date.today()
        start = end - timedelta(days=DISCLOSURE_LOOKBACK_DAYS)
        disclosures = fetch_disclosures(corp_code, start, end)
        analysis = analyze_disclosures(news_client, disclosures, ticker, market, end)
        return to_disclosure_reasons(analysis, action), None
    except Exception as exc:
        return [], str(exc)


def run_recommendation_pipeline(
    ..., ml_model=None, llm_client=None, news_client: DisclosureReader | None = None,
) -> PipelineResult:
    ...
    if ml_model is None and llm_client is None and news_client is None:
        explanation = generate_explanation(candidate, sizing)   # 회귀 잠금, 무수정
    else:
        extra_reasons = []
        ml_r, ml_error = _ml_reasons(ml_model, records, candidate.action)
        extra_reasons.extend(ml_r)
        llm_r, llm_error = _llm_reasons(llm_client, records, candidate.action)
        extra_reasons.extend(llm_r)
        news_r, news_error = _news_reasons(news_client, candidate.ticker, candidate.market, candidate.action)
        extra_reasons.extend(news_r)
        ...
```

**핵심 포인트**:
1. 분기 조건을 `ml_model is None and llm_client is None and news_client is None`으로 **세 개로 확장**한다. `llm-chart-analyst-plan.md`가 못박은 "회귀 테스트가 2-위치인자 호출을 요구한다"는 제약이 그대로 적용된다 — 셋 다 `None`일 때만 기존 호출을 보존.
2. **나스닥 종목은 자동으로 스킵된다.** DART는 한국 기업 전용이라 `resolve_corp_code`가 나스닥 티커에 대해 `None`을 반환하는 것이 정상 흐름이다 — 에러로 취급하지 않고 조용히 빈 근거를 반환한다. `scan_nasdaq_top100_buy_only.py` 같은 나스닥 스캔에 `news_client`를 켜도 안전하다.
3. **DART 조회 + LLM 호출이 한 헬퍼 안에 순차로 들어간다** (ML/LLM차트와 달리 새 외부 조회가 추가되는 유일한 신호원). 두 단계 중 어디서 실패해도 동일한 `except Exception`이 잡는다 — 단, 필요하면 Phase 3 구현 중 "DART 실패"와 "LLM 실패"를 에러 메시지에서 구분 표기(`f"DART 조회 실패: {exc}"` vs `f"LLM 공시해석 실패: {exc}"`)할지는 구현 시 결정.
4. **병합 순서는 ML → LLM차트 → 뉴스/공시**(항상 마지막) — `llm-chart-analyst-plan.md`가 이미 예정해 둔 순서.

### 상충·중재 정책

ML/LLM차트와 동일 — **중재하지 않고 병렬 노출**한다. 세 번째 신호가 추가되면서 상충 표시가 더 많아질 수 있지만, `llm-chart-analyst-plan.md`가 이미 "신호원이 3개가 되어 비교가 어려워지면 재검토"라고 예정해 둔 바로 그 시점이다. 이번 계획에서는 아직 재검토하지 않고 기존 방식(각자 자기 동의/상충 문구를 갖고 나열)을 유지한다.

---

## 에러 처리 · 실패 격리

**1층 — 공시 없음/DART 미등록: 예외 없이 조용히 스킵**
`fetch_disclosures`가 빈 리스트 → `analyze`가 `None` 반환 → API 호출 0. `resolve_corp_code`가 `None`(나스닥 등) → `_news_reasons`가 조기 반환, DART/LLM 둘 다 호출 안 함.

**2층 — DART 응답 오류**
DART API는 성공 시에도 `status` 필드로 오류를 표현하는 경우가 있다(HTTP 200 + 바디에 에러 코드) — 공식 문서 기준 정확한 오류 코드 체계를 Phase 0에서 재확인하고, `status != "000"`(정상)이면 예외로 승격시켜 상위에서 잡는다.

**3층 — OpenAI SDK 예외 통일 (`client.py`)**
`llm_chart_analyst/client.py`와 동일한 매핑 표(APITimeoutError → APIConnectionError → RateLimitError/AuthenticationError → APIStatusError → ValidationError → output_parsed None)를 그대로 복제한다. 메시지에 API 키·DART 키·응답 원문을 절대 포함하지 않는다.

**4층 — 파이프라인 격리 (`_news_reasons`)**
`except Exception`으로 넓게 잡아 `([], str(exc))`. 절대 raise하지 않는다.

**5층 — 크리덴셜 부재는 진입점에서 즉시 실패**
`DART_API_KEY`/`OPENAI_API_KEY` 둘 중 하나라도 없으면 해당 `load_*()` 함수가 `KeyError`.

---

## 테스트 전략

**절대 원칙: 단위테스트는 실제 DART/OpenAI API를 호출하지 않는다.**

- `tests/data/test_dart_source.py`: `requests`를 모킹 — corp_code 캐시 히트/미스/만료(7일), 매핑 실패 시 `None`, 공시 없음(빈 리스트), DART 오류 status 코드 → 예외 승격
- `tests/news_disclosure/test_prompt.py`: 프롬프트에 ticker/market/실제날짜/action 미등장(환각·동조 방어 잠금, #3과 동일한 잠금 테스트 구조)
- `tests/news_disclosure/test_client.py`: SDK 완전 모킹, 예외 매핑, API 키 미유출
- `tests/news_disclosure/test_analyst.py`: 빈 공시 목록 → `None`, market_impact/action 조합별 stance 문구, disclaimer 항상 마지막
- `tests/orchestrator/test_pipeline.py` 신규 케이스(기존 전부 무수정 통과 필수): 무보조신호 회귀 확장(3-None), 뉴스단독, ML+LLM+뉴스 3개 병합 순서, 나스닥 티커는 DART 호출 자체가 없음(스킵 검증), DART 실패해도 ML/LLM 근거 유지, corp_code 없음은 에러로 기록되지 않음

전체 커버리지 80% 이상 유지(#3 완료 시점 기준 99%).

---

## 의존성 · 설정 변경

| 파일 | 변경 |
|---|---|
| `pyproject.toml` | 신규 의존성 없음 — `requests`(DART), `openai`/`pydantic`(#3에서 이미 추가됨) 재사용 |
| `.env.example` | `DART_API_KEY=` 추가. `NEWS_OPENAI_MODEL=`은 선택 항목 |
| `.gitignore` | `data/dart_corp_codes.json` 캐시 파일이 무시되는지 확인(패턴 추가 필요 시) |
| `docs/PRD.md` | §5 뉴스/공시 계층, §7.2에 확정 내용 반영 |
| `docs/IMPLEMENTATION_PLAN.md` | "신호 소스 확장" §3 갱신 |
| `docs/CODEBASE_REVIEW.md` | 모듈 표 추가 |

**변경하지 않는 것**: `explainer/generator.py`, `rule_engine/*`, `risk_sizing/*`, `notifier/*`, `ml_predictor/*`, `llm_chart_analyst/*`(읽기 재사용 없음 — 완전 독립 모듈), `orchestrator/models.py`.

## 재사용할 기존 코드

- `data/sources/fdr_source.py`/`pykrx_source.py` — "데이터 수집 계층 하위에 소스별 파일 하나씩" 구조 패턴
- `llm_chart_analyst/client.py`·`schema.py`·`prompt.py` — 구조적 패턴 복제(코드 공유 아님, 핵심 설계 결정 7)
- `llm_chart_analyst/analyst.py::to_reasons`·`ml_predictor/predictor.py::_stance` — 동의/상충/중립 문구 생성 패턴 복제
- `orchestrator/pipeline.py::_ml_reasons`/`_llm_reasons` — `_news_reasons`의 형태·계약 복제
- `notifier/credentials.py` — `load_dotenv()` 진입점 패턴

---

## 구현 순서 (Phase)

**Phase 0 — 의존성·API 게이트 (비용 0)**: DART 공식 문서로 `list.json`/`corpCode.xml` 응답 스키마·오류 코드 체계 및 `corpCode.xml` 압축 해제 방식(zip) 재확인 → `.env.example`에 `DART_API_KEY=` 추가 → DART 인증키 발급(사용자 작업 — 이 Phase의 완료 조건, 무료·즉시 발급).

**Phase 1 — 데이터 수집 계층 확장 (API 호출 0, 모킹)**: `data/models.py::DisclosureRecord` → `data/sources/dart_source.py`(corp_code 캐싱 + 조회, `requests` 모킹) + 테스트. 독립 머지 가능.

**Phase 2 — 해석 계층 (전부 모킹)**: `news_disclosure/models.py` → `schema.py` → `prompt.py` → `credentials.py` → `client.py` → `analyst.py` + 테스트. 독립 머지 가능. `OPENAI_API_KEY`는 #3에서 이미 있으므로 실호출 준비는 이 단계에서 끝남.

**Phase 3 — 배선 + 실제 API 최초 호출**: `pipeline.py` 확장 → `test_pipeline.py` 신규 케이스 → `scripts/verify_news_disclosure.py`(DART 실호출 + LLM 실호출) → `scripts/run_recommendations_with_signals.py`에 `news_client` 추가 → 텔레그램 메시지 육안 확인.

**Phase 4 — 문서화 + 리뷰**: `docs/design/news-disclosure.md` 작성 → PRD/IMPLEMENTATION_PLAN/CODEBASE_REVIEW 갱신 → `code-reviewer` + `security-reviewer`(DART/OpenAI 두 크리덴셜 취급) 리뷰.

**Phase 5 (선택, 보류) — 빅카인즈(뉴스) 확장**: 이번 계획 범위 밖. rate limit·유료 티어 확인 후 별도 `-plan.md` 갱신으로 착수.

---

## 이번 스코프가 아닌 것

| 이연 항목 | 왜 이번이 아닌가 |
|---|---|
| 빅카인즈(뉴스) 연동 | 사용자 결정 — DART 우선, rate limit 확인 후 별도 착수 |
| GDELT | PRD상 보조 후보일 뿐 채택 미확정 |
| 공시 원문(`document.xml`) 정밀 분석 | 보고서명 목록만으로 시작(핵심 설계 결정 2), 부족함이 판명되면 재검토 |
| 뉴스-종목 매칭 파이프라인 | 빅카인즈 착수 시 필요, DART는 corp_code로 이미 명확히 매칭됨 |
| 대량 백테스트 | 과거 공시 아카이브 확보 자체가 PRD §11 TBD. DART는 과거 공시도 조회 가능하나(날짜 파라미터), 대량 백테스트는 API 호출량이 커져 별도 검증 필요 |
| ML/LLM차트/뉴스 3개 신호 앙상블 가중치 확정 | PRD §5 TBD — 백테스트 없이 정하면 근거 없는 숫자 (ML #2·LLM #3 계획과 동일 논거) |
| `market_impact`/`direction`/ML 확률 3자 자동 중재 | 병렬 노출 유지, 재검토 시점은 신호원 3개가 된 이후(§상충·중재 정책 참고) |

**승격 규칙**: ML #2·LLM #3과 동일 — (1) 3개 이상 비중첩 기간에서 일관된 우위, (2) 거래비용 반영 전략 백테스트로 개선 입증, (3) 데이터 정합성 검증 해소, (4) 앙상블 방식 확정.

---

## 검증 (Verification)

- Phase 0: DART API 응답 스키마 재확인, `corpCode.xml` 압축 해제 방식 확인, `DART_API_KEY` 발급 확인
- Phase 1: `pytest tests/data/test_dart_source.py -v`
- Phase 2: `pytest tests/news_disclosure/ -v`
- Phase 3: `pytest tests/orchestrator/test_pipeline.py -v`(기존 + 신규)
- 전체: `pytest --cov=src --cov-report=term-missing`(80%+)
- 수동(API 1회, DART만 비용 0/OpenAI만 소액): `scripts/verify_news_disclosure.py`
- 수동 E2E: `run_recommendations_with_signals.py`(ML+LLM+뉴스 3개 블록 텔레그램 육안 확인)
- 마지막: `code-reviewer` + `security-reviewer`

## API 키

**새로 필요: `DART_API_KEY`만.** `OPENAI_API_KEY`는 #3에서 이미 확보됨(공유 재사용). DART는 무료 발급(개인 인증키, 즉시 발급 가능한 것으로 알려져 있으나 정확한 절차는 Phase 0에서 재확인). `.env.example`에 추가하고 코드에 절대 하드코딩하지 않는다.

## 결정 현황

| 항목 | 상태 |
|---|---|
| 분석 방식 | **확정 — LLM 재사용(OpenAI, #3 패턴)** |
| 1차 범위 | **확정 — DART(공시)만, 빅카인즈는 별도 착수** |
| `DART_API_KEY` 발급 | **미완료 — 사용자가 발급 예정. Phase 0의 완료 조건** |
| `market_impact` 필드명(vs `sentiment`) | 권고안대로 진행(핵심 설계 결정 1). 이견 있으면 알려달라 |
| DART 실패/LLM 실패 에러 메시지 구분 표기 여부 | 미확정 — Phase 3 구현 중 결정 |
| corp_code 캐시 갱신 주기(7일) | 권고값대로 진행. 이견 있으면 알려달라 |
