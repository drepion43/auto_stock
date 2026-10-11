# 뉴스 감성분석 (#4-뉴스) — 구현 결과

> `docs/design/news-sentiment-plan.md`(사전 계획)의 실제 구현 결과 기록. Phase 1~4 전체 완료.

## 목적·입출력

종목의 최근 뉴스 기사 목록(제목·게재일·출처) — KRX는 네이버 뉴스 검색, NASDAQ은 GDELT — 을 OpenAI GPT(`gpt-5.6-luna`)에 입력해 논조(긍정/부정/중립)를 구조화된 형태로 해석하고, 규칙엔진 후보에 대한 **보조 근거**로만 추가한다. ML예측(#2)·LLM차트분석(#3)·뉴스/공시분석(#4)과 동일하게 자체 후보를 만들지도, 알림을 필터링하지도 않는다. 공시(#4)와 달리 이 신호원은 **진짜 감성분석**이다 — 뉴스 기사는 논조·어조가 있으므로.

## 모듈 구조 (계획 그대로 구현)

### 데이터 수집 계층 확장 (`src/auto_stock/data/`)

| 파일 | 역할 |
|---|---|
| `models.py` | `NewsArticle(ticker, market, title, url, published_at, source)` — 네이버/GDELT 공용 |
| `sources/naver_news_source.py` | `search_news(company_name, ticker, market, start, end)` — 네이버 뉴스 검색 API(`news.json`), HTML 태그·엔티티 정제, RFC 822 `pubDate` 파싱, `NaverNewsApiError` |
| `sources/gdelt_source.py` | `search_articles(query, ticker, market, start, end)` — GDELT DOC 2.0 API(`mode=artlist`), `seendate` 파싱, `GdeltApiError`. API 키 불필요 |
| `sources/dart_source.py`(확장) | `resolve_corp_name(ticker)` — 기존 `corpCode.xml` 캐시에 이미 있던 `corp_name` 필드를 재사용(이전엔 파싱만 하고 버렸음). `resolve_corp_code`와 캐시(다운로드/저장)를 공유 |
| `sources/edgar_source.py`(확장) | `resolve_company_title(ticker)` — 기존 `company_tickers.json` 캐시의 `title` 필드를 재사용. `resolve_cik`와 캐시 공유 |

**캐시 스키마 확장**: `data/dart_corp_codes.json`/`data/edgar_cik_map.json`이 평면(`{ticker: id}`)에서 중첩(`{ticker: {"id_key":..., "name_key":...}}`) 구조로 바뀌었다. 마이그레이션 불필요 — 구 스키마(문자열) 캐시를 만나면 id 조회는 그대로 동작하고 이름 조회만 `None`을 반환하다가, `refresh_days`(7일) 경과 후 자연히 새 스키마로 갱신된다.

### 해석 계층 (`src/auto_stock/news_sentiment/`)

`news_disclosure/`와 동일한 구조(Protocol, SDK 예외 통일, `analyze()`/`to_reasons()` 분리)를 따르되 코드는 공유하지 않는다(계획 문서 핵심 설계 결정, `news-disclosure-plan.md` 핵심 설계 결정 7 연장).

| 파일 | 역할 |
|---|---|
| `models.py` | `NewsSummary`/`NewsSentimentAnalysis`/`LLMConfig`(`api_key`는 `field(repr=False)`)/`NewsSentimentReader`(Protocol) |
| `schema.py` | `NewsSentimentRead` — `sentiment`(POSITIVE/NEGATIVE/NEUTRAL, `news_disclosure`의 `market_impact`와 의도적으로 다른 이름)/`confidence`/`key_headline`/`rationale`/`caveat` |
| `prompt.py` | `SYSTEM_PROMPT`, `build_user_prompt(summary)` — 기사 게재일도 `as_of` 기준 D-N 상대 오프셋, URL은 프롬프트에 넣지 않음 |
| `credentials.py` | `load_llm_config()` — `OPENAI_API_KEY` 재사용, `NEWS_SENTIMENT_OPENAI_MODEL`로 모델만 독립 재정의. `NEWS_LOOKBACK_DAYS=14`(공시의 90일보다 훨씬 짧음 — 뉴스는 빠르게 stale해짐), `MAX_ARTICLES_PER_QUERY=10` |
| `client.py` | `OpenAINewsSentimentClient` — `openai`를 import하는 유일한 파일, `news_disclosure/client.py`와 동일한 SDK 예외 매핑 |
| `analyst.py` | `analyze(client, articles, ticker, market, as_of)`, `to_reasons(analysis, action)` — `NEWS_SENTIMENT_DISCLAIMER`, `SENTIMENT_LABELS`("긍정적"/"부정적"/"중립" — 공시의 "호재"/"악재"와 어휘 구분) |

## 오케스트레이터 통합

`orchestrator/pipeline.py`에 `sentiment_client: NewsSentimentReader | None = None` 5번째 파라미터 추가. 4항 None 분기(`ml_model`/`llm_client`/`news_client`/`sentiment_client` 전부 `None`일 때만 2-위치인자 `generate_explanation` 호출 보존). `extra_reasons` 병합 순서는 ML → LLM차트 → 공시 → 뉴스감성(항상 마지막).

`_sentiment_reasons` 헬퍼는 `_news_reasons`(공시)와 같은 market 분기 구조를 따르되(KRX→네이버, NASDAQ→GDELT), 티커→회사명 해석부터 필요하다는 점이 다르다(뉴스 API는 회사명 문자열로 검색). **3단계 실패 격리**: ① 이름 해석(DART/EDGAR 호출, `_NEWS_SOURCE_LABELS` 재사용) ② 뉴스 조회(네이버/GDELT 호출, `_SENTIMENT_SOURCE_LABELS`) ③ LLM 해석 — 세 단계를 별도 `try/except`로 감싸 각 단계의 실패가 실제로 실패한 시스템 이름으로 정확히 라벨링되게 한다(코드 리뷰에서 처음엔 ①②가 하나로 묶여 있어 DART/EDGAR 실패가 "네이버뉴스/GDELT 조회 실패"로 잘못 표시되던 버그를 발견·수정).

`scripts/verify_news_sentiment.py`(dry-run/실호출 2모드, `NEWS_SENTIMENT_VERIFY_MARKET`으로 KRX/NASDAQ 선택)와 `scripts/run_recommendations_with_signals.py`(`sentiment_client` 배선 추가)도 함께 구현됐다.

## `code-reviewer` + `security-reviewer` 리뷰 결과 반영

CRITICAL/HIGH 없음. MEDIUM 4건 + LOW 2건(코드 리뷰), 별도 LOW 2건(보안 리뷰, 둘 다 비블로킹) 발견. MEDIUM 4건 전부 TDD로 수정, LOW 1건(GDELT 따옴표 이스케이프)도 함께 수정. 리뷰 반영 전 360개 → 반영 후 **367개**로 회귀 확인(이전에 실패하던 3개 테스트도 함께 해결되어 순증가).

**MEDIUM**
1. **`_sentiment_reasons`가 DART/EDGAR 이름 해석 실패를 네이버/GDELT 실패로 잘못 라벨링** — 이름 해석(`resolve_corp_name`/`resolve_company_title`)과 뉴스 조회(`search_news`/`search_articles`)가 하나의 `try/except`로 묶여 있어, 예를 들어 `DART_API_KEY` 만료로 이름 해석이 실패해도 에러 메시지가 "네이버뉴스 조회 실패"로 찍혔다(실제로는 네이버를 호출조차 안 했는데도). 두 단계를 별도 `try/except`로 분리하고, 이름 해석 실패는 `_news_reasons`가 이미 쓰는 `_NEWS_SOURCE_LABELS`(DART/EDGAR)를 재사용해 라벨링하도록 수정.
2. **`gdelt_source.py::_parse_seendate`가 `TypeError`를 못 잡음** — `seendate`가 JSON `null`이면 `datetime.strptime()`이 `ValueError`가 아니라 `TypeError`를 던지는데 기존 `except (KeyError, ValueError)`가 이를 놓쳐 정제되지 않은 예외가 새어나갔다. `naver_news_source.py`의 동일 클래스 실패(`pubDate`가 null)는 이미 `TypeError`까지 잡고 있어 비대칭이었음 — `TypeError` 추가.
3. **`llm_chart_analyst`/`news_disclosure`/`news_sentiment` 세 conftest 모두 `load_dotenv`를 모킹하지 않아 로컬 `.env`에 `OPENAI_API_KEY=`(빈 값)가 있으면 "키 없으면 KeyError" 테스트가 거짓으로 실패** — `monkeypatch.delenv` 직후 `load_llm_config()`가 실제 `load_dotenv()`를 호출해 그 빈 값을 다시 주입하는 게 원인. `dart_source`/`naver_news_source` 테스트가 이미 쓰는 패턴(`load_dotenv` 자체를 모킹)을 세 conftest에 모두 적용 — `news_sentiment`뿐 아니라 이미 리뷰를 통과했던 `llm_chart_analyst`/`news_disclosure`에도 동일 결함이 있었음이 이번에 드러나 함께 수정.
4. **`naver_news_source.py::_clean_html`이 엔티티로 이스케이프된 가짜 태그를 되살림** — 순서가 "태그 제거 → 엔티티 언이스케이프"였는데, `&lt;script&gt;`처럼 엔티티로 이스케이프된 문자열은 태그 제거 단계를 문자 그대로 통과한 뒤 언이스케이프되면서 진짜 `<script>` 태그로 되살아났다. 순서를 "엔티티 언이스케이프 → 태그 제거"로 뒤집어 되살아난 태그도 제거 단계에서 함께 걸러지도록 수정. 실사용 영향은 낮음(`title`은 LLM 프롬프트에만 흘러가고 Telegram에는 LLM 자신의 `key_headline`이 전달됨, `parse_mode` 없이 평문 발송) — 방어심층 성격의 수정.

**LOW (이번에 반영)**
5. **GDELT 정확구문 검색에서 회사명에 포함된 `"`가 질의 문법을 깰 수 있음** — `f'"{name}"'` 구성 시 `name`에서 `"`를 제거하도록 방어 추가.

**보류(LOW, 이번 라운드 미수정, 보안 리뷰)**
- `_sentiment_reasons`의 두 `except` 블록이 `type(exc).__name__` 대신 `f"{exc}"`로 직접 문자열화 — 이 지점에 도달하는 모든 예외가 이미 하위 계층(각 소스 모듈)에서 정제된다는 불변조건에 의존. `_news_reasons`(공시)의 기존 패턴과 동일해 새로 도입된 문제가 아니며, 별도 로컬 관례를 만들지 않기로 결정.
- `naver_news_source.py`/`gdelt_source.py` 둘 다 `response.json()`이 `try/except requests.exceptions.RequestException` 바깥에 있어, malformed JSON 응답 시 정제되지 않은 `JSONDecodeError`가 새어나갈 수 있음 — 크리덴셜 유출 위험은 없음(파싱 위치 정보만 노출), DART/EDGAR도 동일한 방어 격차가 있어 일관된 기존 관례로 판단.

**리뷰가 확인만 하고 이상 없다고 판단한 항목**: 캐시 스키마 하위호환(구/신 양쪽에서 id 조회 정상 동작, 이름 조회는 구 스키마에서 크래시 없이 `None`), 캐시 다운로드 공유(같은 티커에 대해 id/이름 조회를 순서대로 호출해도 API 이중 호출 없음), `NewsArticle.ticker`/`.market`이 호출자 스탬프 방식이며 다운스트림에서 실제로 읽히지 않음(EDGAR의 `DisclosureRecord.ticker`와 동일한 "미사용이지만 잠재적" 패턴, 의도된 것), 네이버 날짜 파싱(`parsedate_to_datetime` + 3종 예외 방어), `extra_reasons` 병합 순서가 실제 코드에서 문서와 일치, market 분기·미해석 스킵·2단계 실패격리 테스트 커버리지가 `_news_reasons`와 대칭, Naver 크리덴셜이 헤더 방식이라 DART식 URL 쿼리스트링 유출 벡터가 애초에 해당 없음(게다가 코드가 이 구분에 의존하지 않고 원본 예외를 아예 문자열화하지 않아 이중으로 안전), `LLMConfig.api_key`의 `repr=False`, `client.py`의 SDK 예외 매핑이 `news_disclosure/client.py`와 완전히 동일, 회사명이 `requests`의 `params=` 딕셔너리를 통해서만 전달되어(문자열 연결 없음) URL 인젝션 여지 없음.

## 테스트 현황

**367개 통과** (Phase 3 완료 시점 360개 + 리뷰 반영 7개 신규 − 이전부터 있던 3개의 무관한 로컬 환경 실패가 함께 해결되어 순증가). `news_sentiment/`·`naver_news_source.py`·`gdelt_source.py`·`orchestrator/pipeline.py` 전부 100%에 가까운 커버리지(`pipeline.py` 98%, 남은 2줄은 "지원하지 않는 시장" 조용한 스킵 분기 — 현재 KRX/NASDAQ 외 시장이 없어 도달 불가능한 코드, EDGAR 리뷰 때와 동일한 판단으로 보류). `requests`/`openai` 전부 모킹, 실제 API 호출 없음.

## 남은 작업 (사용자 액션 필요)

- **네이버 이용약관 재확인** — 이 세션에서 `developers.naver.com` 접근이 차단되어 재확인하지 못함. 실제 프로덕션 자동 실행(스케줄러)에 넣기 전 반드시 사용자가 직접 확인 필요(`news-sentiment-plan.md` Phase 0 미완료 항목).
- **GDELT `seendate` 형식 라이브 검증** — 이 세션에서 `api.gdeltproject.org`로의 아웃바운드 연결이 전부 타임아웃되어(WebFetch·curl 모두) 확인 못 함. `%Y%m%dT%H%M%SZ` 가정은 공식 문서·커뮤니티 자료 교차 확인 값이나, 실제 사용 전 `scripts/verify_news_sentiment.py`(NASDAQ 모드)로 1회 검증 권장.
- `NAVER_CLIENT_ID`/`NAVER_CLIENT_SECRET` 발급 후: `scripts/verify_news_sentiment.py` dry-run(비용 0) → 실호출(소액) 순서로 KRX 검증
- 위 GDELT 검증과 함께: NASDAQ 모드도 동일하게 검증
- `scripts/run_recommendations_with_signals.py`로 ML+LLM차트+공시+뉴스감성 4개 블록이 텔레그램에 함께 뜨는지 육안 확인

## 갱신 (2026-10-12) — GDELT 요청 제한·검색어

- **검색어**: NASDAQ 경로의 GDELT 질의는 이제 `gdelt_source.company_phrase(name)`로 만든다 — 따옴표 제거(위 리뷰 5번 방어 유지)에 더해 SEC 법인 접미사를 떼어 뉴스 표기와 맞춘다(14일 실측 "NVIDIA CORP" 27건 vs "NVIDIA" 50건 이상).
- **요청 제한**: GDELT는 HTTP 429("5초에 1회")를 돌려준다. 프로세스 전체 요청 간격 조절(5.5초) + 429 시 15초 지연 후 1회 재시도를 넣었지만, 실측상 이 IP에는 더 엄격해 여전히 실패할 수 있다.
- **영향**: 미국 종목 분석에서 "뉴스 감성 확인하지 못함"이 잦고, 뉴스가 필요한 질의는 느려질 수 있다. 현황·선택지(캐시, 대체 소스 등)·결정 대기 항목은 `us-news-source-reliability.md`.
