# 뉴스 감성분석 확장 (#4-뉴스) — 국내(네이버)/미국(GDELT) 구현 계획

> `docs/design/news-disclosure-plan.md`(공시 전용)가 명시적으로 미뤄뒀던 "진짜 뉴스" 절반을 다루는 계획. 그 문서 핵심 설계 결정 1은 "이 신호원은 공시 영향 해석이지 감성분석이 아니다"라고 못박으면서 "나중에 빅카인즈(뉴스, 진짜 감성분석이 필요)를 추가할 때 이름 충돌·의미 혼동을 피하기 위해 `market_impact`(공시)와 `sentiment`(뉴스)를 구분해둔다"고 예약해뒀다 — 이 계획이 그 예약을 실행한다.

## 배경

지금까지 구현한 #4는 DART(한국)/EDGAR(나스닥, 계획만) 공시만 다룬다. 사용자가 "DART/EDGAR는 공시뿐이고 진짜 뉴스 기사는 안 가져오고 있다"는 걸 확인 후, 국내+미국 양쪽 뉴스 기사 감성분석을 계획해 달라고 요청했다. PRD가 원래 예정했던 빅카인즈는 **2025년부터 Open API가 유료로 전환**됐다(이 세션에서 확인 — PRD §11의 TBD "빅카인즈 API의 정확한 rate limit·유료 티어 여부"가 이제 확정: 유료). 사용자와 논의 후 결정:

- **국내**: 네이버 뉴스 검색 API(무료, 이용약관상 제약 있음 — 아래 참고)
- **미국**: GDELT(무료, 글로벌 커버리지)

## 데이터 소스 확정

### 네이버 뉴스 검색 API

이 세션에서 조사 확인(커뮤니티 문서 교차 확인 — **공식 문서 재확인은 Phase 0 필수**, 아래 표시):
- 엔드포인트: `GET https://openapi.naver.com/v1/search/news.json` — 헤더에 `X-Naver-Client-Id`/`X-Naver-Client-Secret`(developers.naver.com에서 앱 등록 후 무료 발급).
- 파라미터: `query`(검색어), `display`(건당 최대 **100**), `start`(페이지 시작 위치, 최대 **1000**), `sort`(`sim`/`date`).
- 응답 `items[]` 필드: `title`(검색어에 `<b>` 태그로 하이라이트됨 — 스트립 필요), `originallink`, `link`, `description`(스니펫, 마찬가지로 `<b>` 태그 포함), `pubDate`(RFC 822 형식).
- **원문 전체가 아니라 제목+스니펫만 제공** — DART/EDGAR가 원문 대신 메타데이터만 쓴 원칙(계획 문서 핵심 설계 결정 2)과 자연스럽게 이어진다.
- 일일 호출 한도 **25,000건**(커뮤니티 소스 다수 교차 확인, 공식 수치는 Phase 0에서 developers.naver.com 문서로 재확인).
- **이용약관 리스크**: 이 API는 검색 결과 스니펫 제공이 목적이며, 상업적/자동화 대량 수집에 대한 이용약관 제약 가능성이 있다(PRD §10 "뉴스/공시 크롤링 저작권·약관 리스크"가 이미 경고해둔 바로 그 항목). **Phase 0에서 developers.naver.com 이용약관을 직접 재확인하는 것이 이 소스를 실제로 쓰기 위한 선행 조건**이다 — 사용자가 이미 이 리스크를 인지하고 선택했지만, 실제 약관 문구 확인 전에는 자동 스케줄 실행에 넣지 않는다.

### GDELT (DOC 2.0 API)

- 엔드포인트: `GET https://api.gdeltproject.org/api/v2/doc/doc?query=...&mode=artlist&format=json&maxrecords=...`
- 쿼리 문법(블로그/문서로 확인): 따옴표로 감싸면 정확한 구문 검색(`"Apple Inc"`), `sourcecountry:US`/`sourcelang:English` 등 필터 연산자 지원.
- **이 세션 중 라이브 조회를 1회 시도했으나 연결 오류(`ECONNRESET`)로 실패** — 응답 JSON의 정확한 필드명(통상 `title`/`url`/`seendate`/`domain`/`sourcecountry` 등으로 알려져 있으나 이번엔 직접 검증 못 함)과 정확한 rate limit 수치는 **Phase 0에서 반드시 재확인**해야 한다. DART/EDGAR와 달리 이번 문서는 "확인 완료"가 아니라 "확인 필요"로 남겨둔다 — 추측을 계획에 그대로 박아넣지 않기 위함.
- API 키 불필요.

## 핵심 설계 결정

### 1. 필드명은 `sentiment`다 — `market_impact`(공시)와 명확히 구분

`news-disclosure-plan.md`가 예약해둔 대로, 뉴스 기사는 실제로 논조·어조가 있으므로 진짜 감성분석이 맞다. 새 스키마 `NewsSentimentRead`는 `sentiment: Literal["POSITIVE","NEGATIVE","NEUTRAL"]`을 쓴다 — `DisclosureRead.market_impact`와 절대 같은 이름을 쓰지 않는다(원래 그 이름 충돌을 피하려고 `market_impact`를 골랐던 이유 그 자체).

### 2. 회사명 해석은 새로 만들지 않고 기존 매핑을 재사용한다

두 뉴스 API 모두 티커가 아니라 **회사명 문자열**로 검색해야 한다(DART가 `corp_code`, EDGAR가 `CIK`를 쓰는 것과 다른 종류의 문제). 이미 갖고 있는 매핑에 이름이 들어있다:
- DART의 `corpCode.xml`은 `corp_name` 필드를 이미 포함하지만, 현재 `dart_source.py::_download_corp_code_map()`이 이를 파싱만 하고 버리고 있다(`root.findall("list")` 루프에서 `stock_code`/`corp_code`만 추출). 이 함수를 확장해 `corp_name`도 캐시에 함께 저장하고, 신규 함수 `resolve_corp_name(ticker) -> str | None`을 추가한다.
- SEC의 `company_tickers.json`(계획만 된 `news-disclosure-nasdaq-plan.md`의 `resolve_cik`가 쓸 파일)도 `title`(회사명) 필드를 포함한다. 동일하게 `resolve_company_title(ticker) -> str | None`을 `edgar_source.py`에 추가한다.

**캐시 스키마 변경**: `data/dart_corp_codes.json`이 현재 `{stock_code: corp_code}` 평면 구조인데, 이름까지 저장하려면 `{stock_code: {"corp_code": ..., "corp_name": ...}}`로 바뀐다. 이건 캐시 파일이라 마이그레이션이 필요 없다 — 새 코드가 옛 스키마를 만나면 그냥 캐시 미스로 취급해 재다운로드하게 만들면 된다(`resolve_corp_name`이 옛 평면 구조에서 `.get("corp_name")`을 시도했을 때 `AttributeError`가 나지 않도록 방어 코드 필요 — Phase 1에서 처리).

**`naver_news_source.py`/`gdelt_source.py`는 DART/EDGAR를 전혀 모른다** — `search_news(company_name, start, end)`/`search_articles(query, start, end)`처럼 이미 해석된 문자열만 받는다. 티커→이름 해석은 오케스트레이터(`_sentiment_reasons`)가 시장에 맞는 함수(`resolve_corp_name` vs `resolve_company_title`)를 호출해서 넘겨준다 — 데이터 수집 모듈 간 결합을 만들지 않기 위함.

### 3. 관련성 오탐 방어

회사명이 흔한 단어와 겹칠 위험(예: 짧은 영문 사명)이 있다 — 정확 구문 검색(따옴표)으로 최소한의 방어만 하고, 그 이상의 정교한 오탐 필터링(NER, 종목 코드 병기 검증 등)은 이번 스코프에 넣지 않는다. 실사용 중 오탐이 실제로 문제가 되면 그때 재검토(YAGNI).

### 4. 오케스트레이터 — 4번째 보조 신호, market 분기 포함

기존 3개 헬퍼(`_ml_reasons`/`_llm_reasons`/`_news_reasons`)와 동일 계약(`(reasons, error)`, 절대 raise 안 함)의 `_sentiment_reasons`를 추가한다. `news-disclosure-nasdaq-plan.md`가 `_news_reasons`에 도입한 market 분기와 같은 패턴을 여기도 적용한다:

```python
def _sentiment_reasons(
    sentiment_client: NewsSentimentReader | None, ticker: str, market: str, action: str
) -> tuple[list[str], str | None]:
    if sentiment_client is None:
        return [], None

    try:
        if market == "KRX":
            name = resolve_corp_name(ticker)
        elif market == "NASDAQ":
            name = resolve_company_title(ticker)
        else:
            return [], None
        if name is None:
            return [], None

        end = date.today()
        start = end - timedelta(days=NEWS_LOOKBACK_DAYS)
        articles = search_news(name, start, end) if market == "KRX" else search_articles(name, start, end)
    except Exception as exc:
        source_label = "네이버뉴스" if market == "KRX" else "GDELT"
        return [], f"{source_label} 조회 실패: {exc}"

    try:
        analysis = analyze_sentiment(sentiment_client, articles, ticker, market, end)
        return to_sentiment_reasons(analysis, action), None
    except Exception as exc:
        return [], f"LLM 뉴스감성 해석 실패: {exc}"
```

`run_recommendation_pipeline`에 `sentiment_client: NewsSentimentReader | None = None` 5번째(ML/LLM차트/공시 다음) 파라미터로 추가하고, None-체크 분기를 4항으로 확장한다.

### 5. 신호원이 4개가 됐다 — 병합순서·상충 재검토는 이번 스코프 밖

`llm-chart-analyst-plan.md`와 `news-disclosure-plan.md`가 여러 차례 "신호원이 3개가 되면 병렬 노출을 재검토한다"고 예고해뒀는데, 이 계획으로 4번째(뉴스감성)가 추가되면 그 트리거를 넘어선다. **이번 계획에서는 재검토하지 않는다** — 기존과 동일하게 각 신호가 자기 동의/상충 문구를 갖고 `extra_reasons`에 순서대로(ML → LLM차트 → 공시 → 뉴스감성) 나열되는 방식을 유지한다. 실제 앙상블/가중치 설계는 PRD §5가 이미 TBD로 못박아 둔 별도 과제다.

### 6. 크리덴셜

- 네이버: `NAVER_CLIENT_ID`/`NAVER_CLIENT_SECRET`(무료 발급, 앱 등록 필요) — 없으면 `KeyError`.
- GDELT: 크리덴셜 불필요.
- LLM 해석(OpenAI)은 #3/#4-공시와 동일 `OPENAI_API_KEY` 재사용 — 신규 발급 불필요, 모델 등급만 `NEWS_SENTIMENT_OPENAI_MODEL`로 독립 재정의 가능(기존 `NEWS_OPENAI_MODEL`과 다른 변수명 — 공시 해석과 뉴스 감성이 다른 등급을 쓰게 될 가능성을 열어둠).

### 7. 요청 실패 처리는 DART Phase 4 교훈을 처음부터 반영

`naver_news_source.py`/`gdelt_source.py` 둘 다 `requests.exceptions.RequestException`을 잡아 각각 `NaverNewsApiError`/`GdeltApiError`로 감싼다(원본 예외 문자열 노출 금지, `news-disclosure-nasdaq-plan.md` 핵심 설계 결정 6과 동일 원칙) — 네이버는 헤더에 크리덴셜이 있어(URL 쿼리스트링이 아니지만) 여전히 예외 객체에 요청 헤더 전체가 담길 수 있는 SDK/라이브러리도 있으니 보수적으로 동일하게 방어한다.

## 신규/변경 모듈 구조

```
src/auto_stock/data/models.py                 # 변경 — NewsArticle 추가
src/auto_stock/data/sources/naver_news_source.py   # 신규
├── search_news(company_name, start, end) -> list[NewsArticle]
└── NaverNewsApiError
src/auto_stock/data/sources/gdelt_source.py    # 신규
├── search_articles(query, start, end) -> list[NewsArticle]
└── GdeltApiError
src/auto_stock/data/sources/dart_source.py     # 변경 — resolve_corp_name(ticker) 추가, 캐시에 corp_name 포함
src/auto_stock/data/sources/edgar_source.py    # 변경(nasdaq-plan.md 구현 후) — resolve_company_title(ticker) 추가

src/auto_stock/news_sentiment/                 # 신규 패키지 — news_disclosure/와 동일 구조, 코드 비공유
├── models.py       # NewsSummary, NewsSentimentAnalysis, LLMConfig, NewsSentimentReader(Protocol)
├── schema.py        # NewsSentimentRead — sentiment/confidence/key_headline/rationale/caveat
├── prompt.py         # SYSTEM_PROMPT, build_user_prompt(summary) — 기사 발행일도 D-N 상대 오프셋
├── credentials.py   # load_llm_config(), NEWS_LOOKBACK_DAYS, MAX_ARTICLES_PER_QUERY
├── client.py         # OpenAINewsSentimentClient, NewsSentimentError
└── analyst.py        # analyze(client, articles, ticker, market, as_of), to_reasons(analysis, action)

src/auto_stock/orchestrator/pipeline.py        # 변경 — _sentiment_reasons 추가, sentiment_client 파라미터
```

`NewsArticle`:
```python
@dataclass(frozen=True, slots=True)
class NewsArticle:
    ticker: str
    market: str
    title: str
    url: str
    published_at: date
    source: str  # 네이버: 언론사 도메인, GDELT: domain 필드
```

## 에러 처리 · 실패 격리

기존 3개 신호원과 동일한 4층 구조(계획 없음/오탐-스킵, 소스 API 오류 승격, LLM SDK 예외 통일, 파이프라인 격리 — `news-disclosure-plan.md` 에러 처리 절 참고)를 그대로 따른다. 회사명 해석 실패(`resolve_corp_name`/`resolve_company_title`이 `None`)는 에러가 아니라 조용한 스킵이다.

## 테스트 전략

**절대 원칙: 실제 네이버/GDELT/OpenAI API를 호출하지 않는다.**

- `tests/data/test_naver_news_source.py`, `tests/data/test_gdelt_source.py`: `requests` 모킹.
- `tests/data/test_dart_source.py` 확장: `resolve_corp_name` 신규 케이스(캐시 히트/미스, 옛 스키마 캐시 만났을 때 방어).
- `tests/news_sentiment/`: `news_disclosure/`와 동형의 테스트 스위트(conftest.py, test_credentials/test_prompt/test_client/test_analyst).
- `tests/orchestrator/test_pipeline.py`: `sentiment_client`만 단독, 4개 신호 병합 순서, market 분기(KRX→네이버, NASDAQ→GDELT), 각 실패가 나머지 3개 근거를 삼키지 않음.

## 의존성 · 설정 변경

| 파일 | 변경 |
|---|---|
| `pyproject.toml` | 신규 의존성 없음 — `requests`/`openai`/`pydantic` 재사용 |
| `.env.example` | `NAVER_CLIENT_ID=`/`NAVER_CLIENT_SECRET=` 추가. `NEWS_SENTIMENT_OPENAI_MODEL=`(선택) |
| `data/dart_corp_codes.json` | 캐시 스키마 변경(§핵심 설계 결정 2) — 마이그레이션 불필요, 자연 재다운로드 |

## 구현 순서 (Phase)

**Phase 0 — API 게이트**: 네이버 이용약관 공식 재확인(자동화 수집 허용 범위), GDELT 응답 스키마/rate limit 라이브 재확인(이번 세션 조회 실패분 재시도). 네이버 `Client ID`/`Secret` 발급(무료, 즉시).

**Phase 1 — 데이터 수집 계층**: `naver_news_source.py`/`gdelt_source.py` 신규, `dart_source.py::resolve_corp_name` 추가(+캐시 스키마 방어 테스트), `edgar_source.py::resolve_company_title`은 `news-disclosure-nasdaq-plan.md` 구현 이후 착수.

**Phase 2 — 해석 계층**: `news_sentiment/` 패키지, `news_disclosure/`와 구조적으로 동일 복제.

**Phase 3 — 오케스트레이터 배선**: `_sentiment_reasons`, `sentiment_client` 파라미터, `scripts/verify_news_sentiment.py`(dry-run/실호출).

**Phase 4 — 문서화 + 리뷰**: `docs/design/news-sentiment.md` 작성, PRD/IMPLEMENTATION_PLAN/CODEBASE_REVIEW 갱신, `code-reviewer`+`security-reviewer`.

## 이번 스코프가 아닌 것

| 이연 항목 | 왜 이번이 아닌가 |
|---|---|
| 빅카인즈(유료) | 사용자 결정 — 이번엔 네이버(국내)로 대체 |
| 정교한 관련성 필터링(NER 등) | 실사용 근거 없이 미리 만드는 건 YAGNI 위반 |
| 4개 신호원 앙상블/가중치·상충 중재 정책 재검토 | PRD §5 TBD, 백테스트 없이 정하면 근거 없는 숫자(ML #2·LLM #3·공시 #4와 동일 논거) |
| GDELT/네이버 외 소스(예: Alpha Vantage) 추가 | 사용자가 선택한 범위는 이 두 소스까지 |

## 검증 (Verification)

- Phase 0: 네이버 이용약관·GDELT 스키마 재확인 결과 기록
- Phase 1: `pytest tests/data/test_naver_news_source.py tests/data/test_gdelt_source.py tests/data/test_dart_source.py -v`
- Phase 2: `pytest tests/news_sentiment/ -v`
- Phase 3: `pytest tests/orchestrator/test_pipeline.py -v`(기존+신규)
- 전체: `pytest --cov=src --cov-report=term-missing`(현재 99% 기준선 유지)
- 마지막: `code-reviewer` + `security-reviewer`

## API 키 / 크리덴셜

**새로 필요**: `NAVER_CLIENT_ID`/`NAVER_CLIENT_SECRET`(무료, 즉시 발급). `OPENAI_API_KEY`는 기존 재사용. GDELT는 크리덴셜 불필요.

## 결정 현황

| 항목 | 상태 |
|---|---|
| 국내 뉴스 소스 | **확정 — 네이버 뉴스 검색 API**(사용자 선택) |
| 미국 뉴스 소스 | **확정 — GDELT**(사용자 선택) |
| 필드명 `sentiment`(vs `market_impact`) | **확정** — `news-disclosure-plan.md`가 이미 예약해둔 이름 |
| 회사명 해석 재사용(DART/EDGAR 캐시 확장) | 권고안대로 진행. 이견 있으면 알려달라 |
| 네이버 이용약관 공식 재확인 | **미완료 — Phase 0 필수 선행 조건**, 자동화 수집 허용 범위 확인 전 실행 금지 |
| GDELT 응답 스키마·rate limit 실측 | **미완료 — Phase 0 재확인 필요**(이번 세션 라이브 조회 실패) |
| 4개 신호원 병합순서 재검토 여부 | 권고안대로 이번엔 보류(병렬 노출 유지). 이견 있으면 알려달라 |
