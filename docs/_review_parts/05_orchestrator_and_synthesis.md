# `orchestrator/` + `recommendation_synthesis/` 상세 리뷰

> 이 문서는 "배선(wiring) 계층" — `src/auto_stock/orchestrator/`와 그 위에서 세 번째 `deepagents` 서브에이전트를 구성하는 `src/auto_stock/recommendation_synthesis/` — 를 함수/클래스 단위로 상세히 정리한다. 상위 요약은 `docs/CODEBASE_REVIEW.md`, 최초 MVP-0 설계 근거는 `docs/design/orchestrator.md`를 참고하되, 이 문서는 그 이후 추가된 "전체 유니버스 추천 파이프라인"(숏리스트→딥다이브→종합)과, 이후 market 세분화 작업(아래 핵심 설계 원칙)까지 포함해 현재 코드 상태를 기준으로 다시 정리한 것이다.

## 핵심 설계 원칙 — "원자 market"과 "합성 scan_key"의 분리

`data/models.py::VALID_MARKETS`는 원래 `{"KRX", "NASDAQ"}` 2개뿐이었으나, 사용자가 "코스피만", "코스닥만", "뉴욕거래소만", "S&P500 중에서" 같은 더 세분화된 질의에 응답할 수 있어야 한다는 요구로 **원자(atomic) 거래소 4개** `{"KOSPI", "KOSDAQ", "NASDAQ", "NYSE"}`로 확장됐다. `OHLCVRecord.market`과 `OHLCVCache`의 캐시 PK는 이 4개 원자값만 허용한다 — "KRX"(코스피+코스닥 합성)와 "US"(나스닥+NYSE 합성)는 더 이상 레코드 레벨에서 유효한 값이 아니다.

하지만 사용자는 "코스피+코스닥 통합", "미국 전체", "S&P500 중에서" 같은 **여러 원자 market을 합친 단위**로도 추천을 묻는다. 이걸 레코드 레벨에서 표현하면 캐시 조회가 애매해진다(코스피 종목을 "KRX"로 캐싱했다가 "KOSPI"로 조회하면 캐시미스가 나는 문제). 그래서 이 합성 단위는 **백그라운드 스캔 버킷(scan_key) 레벨에서만** 존재하도록 분리했다 — 신설된 `orchestrator/market_buckets.py`가 그 매핑을 정의한다(아래 섹션). 사용자 요구사항에 따라 최종적으로 지원하는 백그라운드 스캔 버킷은 8개다: `KOSPI`, `KOSDAQ`, `KRX`(코스피+코스닥), `NASDAQ`, `NYSE`, `US`(나스닥+NYSE), `SP500`(S&P500 구성종목), `DOW`(다우존스 구성종목).

종목 하나를 다루는 6개 신호 tool(`analyze_rule_engine` 등, `chat_agent/tools.py`)과 ①숏리스트(`build_shortlist`)/무료스캔(`scan_market`)은 여전히 **원자 market 하나**만 받는다 — 종목 하나는 항상 원자 거래소 하나에만 속하므로 이 계약은 바뀌지 않았다. 합성 버킷을 다뤄야 하는 지점은 오직 `RecommendationCoordinator`(아래에서 상세히 설명)뿐이다.

## 전체 그림 — 이 레포에는 서로 다른 두 종류의 파이프라인이 있다

1. **MVP-0 알림 파이프라인** (`pipeline.py::run_recommendation_pipeline`) — 호출자가 명시적으로 지정한 소수의 워치리스트 티커를 돌며, 신호가 있으면 **텔레그램으로 실제 알림을 보낸다**. `scripts/run_recommendations*.py` 계열이 이 경로를 쓴다.
2. **전체 유니버스 배치 추천 파이프라인** — 챗봇이 "지금 추천할 만한 종목 있어?" 질의에 즉답하기 위해 백그라운드에서 전체 시장(또는 섹터)을 훑어 결과를 캐시에 적재하는 경로다. 세 단계로 나뉜다(각 단계는 `docs/design/`에 없는 `recommendation-synthesis.md`라는 계획 문서를 참조하는 주석이 코드 곳곳에 있는데, **그 파일 자체는 현재 디스크에 존재하지 않는다** — 삭제됐거나 다른 이름으로 옮겨진 것으로 보이며, 설계 결정 근거는 각 코드 docstring에 남은 설명으로만 추적 가능하다):
   - **① 숏리스트** (`shortlist.py::build_shortlist`) — 규칙엔진 ∪ ML 중 하나라도 방향성 의견을 내면 통과시키는 저비용 1차 필터.
   - **② 심층분석(딥다이브)** (`deep_scan.py::run_deep_scan`) — 숏리스트에 오른 종목마다 기존 `stock_analyst` deepagents 서브에이전트를 재사용해 자율적으로 신호를 골라 심층 판단.
   - **③ 종합(synthesis)** (`recommendation_synthesis/synthesizer.py::run_recommendation_synthesis`) — 숏리스트 전체의 심층판단을 한 번에 비교해 상위 N개만 순위와 근거를 매기는 세 번째 deepagents 서브에이전트.

   이 세 단계를 전체 유니버스 경로(시장 전체)와 섹터/테마 경로(사용자가 지정한 섹터만) 양쪽에서 공유하도록 조립하는 것이 `RecommendationCoordinator`(`recommendation_coordinator.py`)다. `ScanCoordinator`(`scan_coordinator.py`)는 이와 별개로 존재하는, **더 단순하고 더 오래된** 무료 스캔 경로(①②③ 없음, LLM 전혀 안 씀)의 조율기다.

두 파이프라인이 공유하지 않는 이유가 코드에 명시돼 있다: `scan_market`(무료 경로)의 기존 계약(하위 호환)을 건드리지 않기 위해, 전체 유니버스 추천 경로는 `build_shortlist`/`run_deep_scan`/`run_recommendation_synthesis`라는 **별도 모듈**로 신설됐다(`shortlist.py:9-11`, `recommendation_coordinator.py:5-6`).

```
                                    ┌─ (무료, LLM 없음) ──────────────────────────┐
                                    │  ScanCoordinator._run                        │
                                    │   get_universe → scan_market                 │
                                    │   → MarketScanCache.put_scan                 │
                                    └───────────────────────────────────────────────┘

                                    ┌─ (유료, LLM 사용) RecommendationCoordinator ──────────────────────────┐
전체 유니버스 경로(bucket):         │  _run(bucket)   — bucket은 원자 market 또는 8개 스캔 버킷 중 하나      │
  bucket_universe_by_atomic_market(bucket, fetch_universe)                                                     │
    → 원자 market마다 _fresh_universe로 유니버스 획득(합성 버킷이면 2개 이상으로 쪼개짐)                        │
    → 원자 market마다 각각 build_shortlist 호출(①, 규칙엔진 ∪ ML) 후 결과를 하나로 합침                        │
    → 점수순 정렬 후 상위 shortlist_cap(20)개로 절단                                                           │
    → _run_deep_scan_and_synthesize                                                                            │
        → run_deep_scan (②, stock_analyst agent-as-tool, ThreadPoolExecutor)                                   │
        → run_recommendation_synthesis (③, 세 번째 deepagents 서브에이전트)                                    │
    → MarketScanCache.put_recommendations(scan_key=bucket, ...)                                                 │
                                    │                                                                           │
섹터/테마 경로:                     │  _run_sector(sector_query, market)  — market은 원자/합성 상관없이 그대로 전달│
  resolve_sector_tickers (①숏리스트 생략 — stock_analyst가 스스로 규칙엔진/ML 호출)                            │
    → MarketScanCache.put_sector_tickers                                                                        │
    → _run_deep_scan_and_synthesize (②③ 공유)                                                                   │
    → MarketScanCache.put_recommendations(scan_key="SECTOR:{query}:{market}", ...)                              │
                                    └───────────────────────────────────────────────────────────────────────────┘
```

이 전체 경로는 `scripts/chat_cli.py`가 시작 시 **8개 스캔 버킷 전체**(`orchestrator/market_buckets.py::ALL_SCAN_BUCKETS`)에 대해 `ensure_fresh`를 선제 호출하고, 챗봇 tool(`tool_get_market_scan_recommendations` 등, `chat_agent/tools.py`)이 질의 시점마다 같은 메서드를 호출해 "이미 신선하면 아무것도 안 함, stale이면 백그라운드 스레드로 갱신 트리거" 하는 **온디맨드 자가치유** 패턴으로 돌아간다. 둘 다 idempotent — 중복 트리거가 안 생기도록 "stale 체크 + in-progress 추적"을 하나의 락 안에서 원자적으로 수행한다.

---

## `orchestrator/models.py`

### `PipelineResult`

```python
@dataclass(frozen=True, slots=True)
class PipelineResult:
    sent: list[Explanation]
    errors: list[tuple[str, str]]
```

`run_recommendation_pipeline`의 반환형. `sent`는 실제로 텔레그램 발송에 성공한 `Explanation` 목록, `errors`는 `(ticker, 에러메시지)` 튜플 목록이다. `frozen=True, slots=True`로 불변·경량 값 객체로 고정 — 호출 후 결과를 변형할 일이 없다는 설계 의도.

---

## `orchestrator/pipeline.py` — MVP-0 알림 파이프라인

`run_recommendation_pipeline`은 이 레포에서 가장 먼저 만들어진 오케스트레이터이자, 유일하게 **실제 텔레그램 알림을 전송하는** 함수다. 상세 설계 근거는 `docs/design/orchestrator.md`가 원본이지만, 코드는 그 문서 작성 이후 ML(#2)/LLM차트(#3)/공시(#4)/뉴스감성(#4-뉴스) 4개 보조신호가 추가 배선되며 크게 확장됐다.

### 상수

- `DEFAULT_LOOKBACK_DAYS = 120` (`pipeline.py:56`) — SMA60 워밍업 + ATR(14) 계산에 필요한 최소 60거래일을 확보하기 위해, 주말·공휴일을 감안해 캘린더 기준 120일로 설정. `market_scan.py`/`shortlist.py`도 이 상수를 그대로 import해서 재사용한다.

### `_ml_reasons(ml_model, records, action) -> tuple[list[str], str | None]`

- **역할**: ML 예측(#2) 보조신호를 문구로 변환. `ml_model`이 `None`이면 `([], None)` 즉시 반환(스킵).
- **동작**: `predict(ml_model, records)` → `to_reasons(prediction, action)`.
- **에러 처리**: `except Exception` 전체를 잡아 `([], str(exc))`로 변환 — **절대 raise하지 않는다**(docstring 명시, 설계 결정 5). ML 실패가 알림 발송 자체를 막으면 안 된다는 원칙.
- 다른 세 헬퍼와 달리 에러 메시지에 소스 라벨 접두사를 **스스로 붙이지 않는다** — 호출부(`run_recommendation_pipeline`)가 `"ML 예측 실패: {ml_error}"`로 접두사를 붙인다(`pipeline.py:238`).

### `_llm_reasons(llm_client, records, action) -> tuple[list[str], str | None]`

`_ml_reasons`와 완전히 동일한 패턴(설계 결정 6) — `llm_client`가 `None`이면 스킵, 아니면 `analyze_chart` → `to_chart_reasons`, 실패는 삼켜서 반환, 접두사는 호출부가 붙임(`"LLM 차트분석 실패: ..."`).

### `_news_reasons(news_client, ticker, market, action) -> tuple[list[str], str | None]`

- **역할**: 공시(#4) 보조신호. KOSPI/KOSDAQ은 DART, NASDAQ/NYSE는 EDGAR — **해석 계층(`analyze_disclosures`/`to_disclosure_reasons`)은 시장과 무관하게 완전히 동일한 코드를 재사용**하고, 시장별 분기는 이 함수 내부에만 있다. market→소스 라벨 매핑은 신설된 `data/market_routing.py::NEWS_SOURCE_LABELS`(`{"KOSPI": "DART", "KOSDAQ": "DART", "NASDAQ": "EDGAR", "NYSE": "EDGAR"}`)로 옮겨졌다 — 과거에는 `pipeline.py`와 `chat_agent/tools.py`가 각자 동일한 딕셔너리(`_NEWS_SOURCE_LABELS = {"KRX": "DART", "NASDAQ": "EDGAR"}`)를 중복 정의했는데, market 세분화 작업에서 두 곳을 매번 똑같이 고쳐야 하는 번거로움을 없애려고 공용 모듈로 통합했다. "국내냐 미국이냐" 판정도 `data/market_routing.py::KRX_MARKETS`(`frozenset({"KOSPI", "KOSDAQ"})`) 멤버십으로 통일됐다.
- **동작 순서**:
  1. `source_label = NEWS_SOURCE_LABELS.get(market)` — 지원하지 않는 시장이면 조용히 `([], None)` 스킵(향후 시장 확장 대비, 에러 아님).
  2. 티커 → 공시ID 해석: `market in KRX_MARKETS`면 `resolve_corp_code`, 아니면(NASDAQ/NYSE) `resolve_cik`. `None`이면(미등록 종목) 조용히 스킵.
  3. `fetch_disclosures`/`fetch_filings`로 최근 `DISCLOSURE_LOOKBACK_DAYS`(90일) 공시 조회.
  4. `analyze_disclosures(news_client, disclosures, ticker, market, end)` → `to_disclosure_reasons`.
- **에러 처리**: **공시 조회 단계와 LLM 해석 단계를 별도의 `try/except`로 분리**한다 — 이것이 다른 두 헬퍼(`_ml_reasons`/`_llm_reasons`)와 구조적으로 다른 지점이다. 이 함수는 두 개의 서로 다른 실패 지점(외부 데이터 소스 vs. LLM)이 있어서 **자기 자신이 접두사를 결정**한다: `"DART 조회 실패: ..."`/`"EDGAR 조회 실패: ..."`/`"LLM 공시해석 실패: ..."`.

### `_sentiment_reasons(sentiment_client, ticker, market, action) -> tuple[list[str], str | None]`

- **역할**: 뉴스 감성분석(#4-뉴스) 보조신호. `_news_reasons`와 동일한 시장 분기 골격(KOSPI/KOSDAQ→네이버뉴스, NASDAQ/NYSE→GDELT, `data/market_routing.py::SENTIMENT_SOURCE_LABELS`)이지만, 뉴스 API는 티커가 아니라 **회사명으로 검색해야 하므로** 이름 해석 단계가 하나 더 끼어든다.
- **동작 순서**:
  1. 시장 미지원이면 스킵.
  2. 이름 해석: `market in KRX_MARKETS`면 `resolve_corp_name`(DART 캐시 재사용), 아니면(NASDAQ/NYSE) `resolve_company_title`(EDGAR 캐시 재사용) — **공시용 캐시 인프라를 그대로 재사용**, 새 인프라를 만들지 않음. `None`이면(해석 실패) 조용히 스킵.
  3. 뉴스 조회: KOSPI/KOSDAQ은 `search_news(name, ...)`, NASDAQ/NYSE는 `search_articles(f'"{safe_name}"', ...)`(정확 구문 검색을 위해 회사명을 따옴표로 감싼다 — `safe_name = name.replace('"', "")`로 회사명 자체에 든 따옴표를 먼저 제거해 GDELT 질의 문법이 깨지는 것을 방지, 코드리뷰 LOW 수정).

     > **갱신 (2026-10-12)**: GDELT 검색어는 이제 `gdelt_source.company_phrase(name)`로 만든다 — 따옴표 제거에 더해 SEC 법인 접미사(Corp, Inc, Co, Ltd 등)를 떼어 뉴스 표기와 맞춘다("NVIDIA CORP" 27건 → "NVIDIA" 50건 이상, 14일 실측). `safe_name`은 없어졌다. 상세: `docs/design/us-news-source-reliability.md`.
  4. `analyze_sentiment` → `to_sentiment_reasons`.
- **에러 처리**: **이름 해석 / 뉴스 조회 / LLM 해석 세 단계가 각각 분리된 `try/except`**. 이름 해석 실패는 `NEWS_SOURCE_LABELS`(DART/EDGAR)를 재사용한 라벨로("DART 조회 실패: ..."), 뉴스 조회 실패는 `SENTIMENT_SOURCE_LABELS`(네이버뉴스/GDELT)로, LLM 해석 실패는 `"LLM 뉴스감성 해석 실패: ..."`로 각각 구분해서 반환한다. **왜 분리했는지**: 하나로 묶으면 DART/EDGAR 호출 실패가 "네이버뉴스/GDELT 조회 실패"로 잘못 라벨링되는 버그가 실제로 있었고(코드리뷰 MEDIUM으로 발견·수정), 그 재발을 막기 위해 구조적으로 분리됐다.
- **NYSE 추가로 인한 변화**: 과거엔 `market == "KRX"`/`else`(2-way)였던 분기가 `market in KRX_MARKETS`/`else`(사실상 4-way 중 국내/미국 2분류)로 바뀌었을 뿐, DART/EDGAR/네이버/GDELT 호출 코드 자체는 수정되지 않았다 — NYSE는 그냥 "미국 쪽" 분기를 그대로 탄다.

### `run_recommendation_pipeline(cache, tickers, market, account, credentials, lookback_days=120, ml_model=None, llm_client=None, news_client=None, sentiment_client=None) -> PipelineResult`

- **역할**: 이 모듈의 메인 진입점. 각 티커에 대해 `get_ohlcv → generate_candidates → (각 후보마다) suggest_position → generate_explanation → send_notification` 순서로 실행.
- **핵심 분기**(`pipeline.py:211-235`): `ml_model`/`llm_client`/`news_client`/`sentiment_client` **네 개가 모두 `None`**이면(기본값 그대로 호출된 경우) `generate_explanation(candidate, sizing)`을 **2-위치인자**로 호출한다 — 보조신호 도입 이전과 **바이트 단위로 동일한 동작**을 보장. 이 분기는 `if`를 네 개 쪼개지 않고 **하나의 `and`-체인 조건**으로 통일했는데(`pipeline.py:211-216`), 그 이유는 기존 회귀 테스트가 2-위치인자 호출을 `assert_called_once_with(ANY, ANY)`로 못박고 있어서, 분기를 쪼개면 그 테스트가 깨지기 때문이다.
- 하나라도 켜져 있으면 네 헬퍼를 **순서대로**(ML → LLM차트 → 공시 → 뉴스감성, 항상 이 순서) 호출해 `extra_reasons` 리스트에 이어붙이고, `generate_explanation(candidate, sizing, extra_reasons=extra_reasons)`로 호출한다.
- 네 헬퍼 중 어느 것이 실패해도 **다른 세 개의 결과는 그대로 살아남는다** — 각 헬퍼가 독립적으로 실패를 삼키기 때문. 실패는 `errors` 리스트에 `(ticker, 에러메시지)`로 쌓이되, **알림 발송(`send_notification`) 자체는 멈추지 않는다**.
- **종목 단위 격리**: 전체 for-loop을 감싸는 바깥쪽 `try/except Exception`(`pipeline.py:207`, `251`)이 있어서, `get_ohlcv` 실패나 `send_notification`(텔레그램 전송) 실패 등 **어떤 단계에서든** 예외가 나면 해당 티커만 `errors`에 기록되고 **다음 티커로 계속 진행**한다. 전체 상장종목을 겨냥한 시스템에서 한 종목의 일시적 오류로 전체 배치가 죽으면 안 된다는 원칙(`docs/design/orchestrator.md` "종목 단위 에러 격리" 절).

---

## `orchestrator/market_scan.py` — 무료 전체 스캔 (①②③ 없음)

### `scan_market(cache, tickers, market, account, lookback_days=120) -> tuple[list[Explanation], list[tuple[str, str]]]`

- **역할**: "추천해줄만한 주식 있어?" 같은 전체 스캔형 질의에 챗봇이 즉답할 수 있도록, `MarketScanCache`에 적재할 결과를 만드는 **원샷 스캔**. `scripts/run_market_scan.py`가 직접 호출하고, `ScanCoordinator._run`도 내부적으로 이 함수를 호출한다.
- **`run_recommendation_pipeline`과의 차이점**(핵심):
  1. **텔레그램을 발송하지 않는다** — 수백 종목을 스캔하며 매 후보마다 알림을 보내면 스팸이 된다는 이유로 분리된 별도 함수.
  2. **보조신호(ML/LLM차트/공시/뉴스감성)를 전혀 쓰지 않는다** — 규칙엔진 + 사이징만 쓰는 완전 무료 경로(LLM 호출 0회).
- **동작**: 티커마다 `get_ohlcv → generate_candidates → (각 후보) suggest_position → generate_explanation`(2-위치인자, extra_reasons 없음) → 결과를 `explanations` 리스트에 누적. 티커 단위 `try/except`로 실패 격리(`run_recommendation_pipeline`과 동일 원칙).
- **반환값**: `(explanations, errors)` 튜플. `PipelineResult`를 쓰지 않고 평범한 튜플을 쓰는 이유는 명시돼 있지 않지만, 이 함수가 텔레그램 발송 결과(`sent`)가 아니라 "스캔된 모든 설명"을 반환하기 때문으로 추정된다(`PipelineResult.sent`는 "발송 성공분"이라는 의미가 실려 있어 의미가 다르다).
- **market 세분화 이후 영향**: 이 함수 자체는 로직이 바뀌지 않았다 — 여전히 "market 하나 + tickers 리스트"를 받는 범용 함수다. 다만 `scripts/run_market_scan.py`의 `MARKETS` 상수가 `["KRX", "NASDAQ"]`에서 `["KOSPI", "KOSDAQ", "NASDAQ", "NYSE"]`(원자 4개)로 바뀌어, 호출부가 이제 원자 market만 넘긴다. 이 레거시 무료경로는 합성 버킷(KRX/US/SP500/DOW) 개념 자체를 모른다.

---

## `orchestrator/shortlist.py` — ① 숏리스트 단계

### `ShortlistEntry` (frozen dataclass)

```python
ticker: str
market: str
rule_candidate: Candidate | None
ml_prediction: MLPrediction | None
```

규칙엔진 후보와 ML 예측을 **원본 그대로** 담아 다음 단계(②딥다이브, ③종합)에 전달하는 중간 산출물.

### `_ml_has_directional_opinion(prediction) -> bool`

ML이 "방향성 있는 의견"을 냈는지 판정. `prediction`이 `None`이면 `False`. 아니면 `probability_up >= ML_AGREE_THRESHOLD`(상승 쪽 확신) 또는 `probability_up <= ML_CONFLICT_THRESHOLD`(하락 쪽 확신) — 즉 확률이 0.5 근처(애매함)면 `False`가 된다. 두 임계값은 `ml_predictor.predictor` 모듈이 이미 정의한 상수를 재사용한다.

### `build_shortlist(cache, tickers, market, ml_model, lookback_days=120) -> tuple[list[ShortlistEntry], list[tuple[str, str]]]`

- **역할**: 전체 유니버스 중 "분석할 가치가 있는" 종목만 1차로 걸러낸다.
- **핵심 설계 결정 — OR 게이트**: 규칙엔진 후보 **또는** ML 방향성 의견 중 **하나라도** 있으면 숏리스트에 넣는다. docstring이 명시적으로 설명한다 — 규칙엔진 후보만 요구하는 AND 게이트로 만들면, ML만 신호를 잡는 종목을 영구히 놓치게 된다.
- **동작**: 티커마다 `get_ohlcv → generate_candidates`(첫 번째 후보만 사용, `candidates[0] if candidates else None`) + `predict(ml_model, records)`(ml_model이 있을 때만) → OR 조건 통과 시 `ShortlistEntry`로 적재.
- **비용 특성**: 규칙엔진은 순수 계산, ML은 이미 학습된 모델의 `predict_proba()` 호출 — 둘 다 LLM을 쓰지 않는 "공짜" 계산이라, **전체 유니버스(수백 종목)에 돌려도 비용 문제가 없다**. 비싼 LLM 신호(차트/공시/뉴스감성)는 이 단계에서 추려진 숏리스트에만 쓴다(②단계).
- **실패 격리**: `scan_market`과 동일한 티커 단위 `try/except`.
- **왜 `scan_market`의 계약을 건드리지 않았는가**: `scan_market`은 다른 소비자(텔레그램 알림 등)가 있을 수 있어 하위 호환을 지켜야 한다는 이유로, 이 모듈이 새로 추가됐다(`shortlist.py:9-11`).
- **market 세분화 이후 영향**: 이 함수도 로직이 바뀌지 않았다 — `market: str` 하나만 받는 범용 계약 그대로다. 합성 버킷(KRX/US/SP500/DOW)을 다루려면 호출부가 원자 market별로 이 함수를 여러 번 호출해야 한다 — 그 분기는 이 함수가 아니라 `RecommendationCoordinator._run`이 담당한다(아래 섹션).

---

## `orchestrator/deep_scan.py` — ② 심층분석(딥다이브) 단계

### `TickerJudgment` (frozen dataclass)

```python
ticker: str
market: str
rule_candidate: Candidate | None
ml_prediction: MLPrediction | None
stock_analyst_result: dict
```

`ShortlistEntry`에 `stock_analyst_result`(딥다이브 결과)를 더한, ③종합 단계로 넘어갈 최종 입력 단위.

### `DEFAULT_MAX_WORKERS = 8`

병렬 처리 워커 수. 설계 문서상 확정값으로 명시.

### `run_deep_scan(shortlist, budget, agent_model, bind_tools, max_workers=8) -> list[TickerJudgment]`

- **역할**: 숏리스트에 오른 종목마다 기존 `chat_agent.stock_analyst.run_stock_analyst`(이미 구현된 deepagents 서브에이전트)를 **재사용**해 자율 판단을 받는다. **신규 agent를 만들지 않는다** — 사용자가 2026-10-02에 "고정 fan-out(모든 종목에 3신호 균일 호출)보다 agent 자율성을 우선"하기로 확정했다(docstring).
- **트레이드오프**: 종목마다 `stock_analyst`가 실제로 호출한 신호 종류(`signals_used`)가 다를 수 있다 — 어떤 종목은 규칙엔진·ML만 확인됐고, 어떤 종목은 뉴스·공시·차트까지 확인됐을 수 있다. 이 비대칭은 ③종합 단계의 프롬프트가 명시적으로 흡수하도록 설계됐다(아래 synthesizer 섹션 참고).
- **병렬화**: `ThreadPoolExecutor(max_workers=8).map(_judge, shortlist)`를 쓴다. **`map()`을 선택한 이유**: 표준 라이브러리가 "입력 순서를 그대로 보존해서 결과를 반환한다"는 계약을 보장하므로, 동시 실행을 하면서도 "결과 순서가 입력 종목 순서와 일치해야 한다"는 요구(③단계의 비교·순위 매기기에 필요)를 추가 로직 없이 만족시킨다. 이 레포에 asyncio 전례가 없고 threading만 쓰는 기존 관행(`ScanCoordinator` 등)과도 일치한다.
- **내부 헬퍼 `_judge(entry) -> TickerJudgment`**: `bind_tools(entry.ticker, entry.market)` → `run_stock_analyst(budget, agent_model, tools, entry.ticker, entry.market)`. 이 호출을 **한 번 더 `try/except`로 감싼다**(`deep_scan.py:50-54`) — `run_stock_analyst` 자신은 이미 종목별 실패격리를 보장하지만(절대 raise 안 함), 호출부가 넘기는 `bind_tools` 콜백은 호출부(여기서는 `RecommendationCoordinator`)의 실수로 예외를 던질 수 있기 때문에 한 겹 더 방어한다. 실패 시 `{"available": False, "error": str(exc)}`를 `stock_analyst_result`에 담아 처리를 계속한다.
- `shortlist`가 빈 리스트면 즉시 `[]` 반환(ThreadPoolExecutor도 안 만듦).

---

## `recommendation_synthesis/` — ③ 종합 단계 (세 번째 deepagents 서브에이전트)

### 설계 배경 — 왜 plain function이 아니라 agent인가

`synthesizer.py` 상단 docstring(`synthesizer.py:1-17`)에 설계 전환 과정이 그대로 남아 있다. **처음에는 "입력/출력이 고정돼 있어 agent가 필요 없다"고 판단해 plain function으로 설계했으나**, "두 종목의 우열이 신호만으로 애매하면 `stock_analyst`를 다시 호출해 추가 증거를 확보할 수 있어야 한다"는 **재조사 자율성 요구**가 생기면서 agent 정당화 기준(이 레포에서는 "자율 추가조사가 필요한 경우에만 deepagents를 쓴다"는 기준, `CLAUDE.md`/`tech-stack` 메모리 참고)을 충족하게 되어 결정이 번복됐다. 이 번복 자체가 이 레포의 아키텍처 원칙("필요한 지점에만 프레임워크를 쓴다")이 실제로 어떻게 적용되는지 보여주는 사례다.

`find_related_companies`/`stock_analyst(ticker)`가 각각 "종목 1개"를 다루는 서브에이전트인 것과 달리, `recommendation_synthesis`는 **에이전트 인스턴스 1개가 숏리스트 전체(여러 종목)를 한 번에 다룬다**. 그래서 `reinvestigate_ticker` tool은 `tools.py._bind_stock_analyst_tools`처럼 종목 1개에 미리 묶인 인자 없는 tool이 아니라, **ticker/market을 그때그때 받는 파라미터화된 tool**이다.

### `recommendation_synthesis/credentials.py`

```python
RECOMMENDATION_SYNTHESIS_RECURSION_LIMIT = 6
RECOMMENDATION_SYNTHESIS_LLM_CALL_ESTIMATE = RECOMMENDATION_SYNTHESIS_RECURSION_LIMIT
```

`chat_agent/credentials.py`의 `FIND_RELATED_*`/`STOCK_ANALYST_*` 상수 쌍과 **동일한 패턴**: LangGraph가 실제로 강제하는 재귀 상한(`RECURSION_LIMIT`)과 `QueryBudget` 사전차감용 추정치(`LLM_CALL_ESTIMATE`)를 **항상 같은 값**으로 고정한다. 왜 같아야 하는가 — 추정치가 재귀상한보다 작으면, agent가 실제로 재귀상한까지 `reinvestigate_ticker`를 호출했을 때 `QueryBudget`이 그 실제 지출을 **과소 계상**하는 보안 버그가 된다(Stage E 코드리뷰에서 발견된 패턴과 동일, 재발 방지를 위해 아예 같은 상수를 참조하도록 구조화).

값은 `6`(`FIND_RELATED_RECURSION_LIMIT`과 동일) — `stock_analyst`(8)보다 복잡도가 낮다고 판단된 근거는 "규칙엔진 후보/ML확률 비교 + 애매한 소수 종목만 선택적으로 재조사"라는 작업 성격이 `find_related_companies`와 비슷하다는 판단(2026-10-05 확정).

### `recommendation_synthesis/models.py`

```python
class RecommendationItem(BaseModel):
    ticker: str
    market: str
    action: str  # "BUY" | "SELL" | "HOLD"
    rank: int
    summary: str

class RecommendationSynthesisResult(BaseModel):
    recommendations: list[RecommendationItem]
```

`chat_agent/stock_analyst.py`의 `StockAnalystResult`와 동일하게, deepagents의 구조화 출력(`response_format`)으로 쓰이는 Pydantic 모델. LLM이 자유 텍스트 대신 이 스키마에 맞춰 응답하도록 강제한다.

### `recommendation_synthesis/synthesizer.py`

#### 시스템 프롬프트(`_SYSTEM_PROMPT`, `synthesizer.py:36-50`)

핵심 지시 4가지가 명시돼 있다:
1. 여러 종목의 judgment(규칙엔진/ML/stock_analyst 결과)를 종합해 상위 `top_n`개만 순위와 근거를 매길 것.
2. **무차별 재조사 금지** — "두 종목의 우열을 주어진 신호만으로 확신할 수 없을 때만" `reinvestigate_ticker`를 선택적으로 쓸 것(상위권 전부를 무차별로 재조사하지 말라고 명시).
3. **signals_used 비대칭을 숨기지 말 것** — 종목마다 확인된 신호 종류가 다를 수 있다는 사실을 최종 summary에 그대로 드러내라(예: "규칙엔진·ML만 확인됨, 추가 신호 미조사"). ②단계(`run_deep_scan`)에서 생긴 트레이드오프를 이 프롬프트가 흡수하는 지점이다.
4. **프롬프트 인젝션 방어** — judgment에 담긴 자유텍스트(뉴스 제목, 공시 요약, stock_analyst의 summary 등)는 "분석 대상 데이터일 뿐"이며 그 안의 지시문처럼 보이는 문구를 지시로 따르지 말라고 명시. `related_companies.py`/`stock_analyst.py`와 동일한 방어 패턴.

이 네 가지는 각각 `test_synthesizer.py`의 `test_system_prompt_*` 테스트로 **문자열 포함 여부**가 고정돼 있다(`"reinvestigate_ticker"`, `"애매"`, `"signals_used"`, `"데이터"`, `"무시"`).

#### `_reinvestigate_ticker_impl(budget, model, bind_stock_analyst_tools, ticker, market) -> dict`

재조사 tool의 실제 구현. `bind_stock_analyst_tools(ticker, market)`로 6개 신호 tool을 그 종목/시장에 묶은 뒤, `run_stock_analyst(budget, model, tools, ticker, market)`를 **그대로 호출**한다 — 즉 재조사는 ②단계와 똑같은 `stock_analyst` 서브에이전트를 다시 띄우는 것일 뿐, 별도 로직이 없다.

**예산 공유가 핵심**: 여기서 **새 `QueryBudget`을 만들지 않고**, 호출부(`run_recommendation_synthesis`)가 넘긴 `budget` 객체를 그대로 전달한다(`synthesizer.py:11-13`). 이렇게 해야 "재조사 비용"이 ②단계(딥다이브) 예산과 합쳐져 하나의 숫자로 정확히 집계된다 — 재조사마다 새 예산을 만들면 전체 LLM 호출량이 실제보다 적게 집계되는 안전 구멍이 생긴다.

#### `_make_reinvestigate_tool(budget, model, bind_stock_analyst_tools)`

위 구현을 `langchain_tool`로 감싸 LLM이 호출할 수 있는 tool 객체로 변환하는 팩토리. 내부 함수 `reinvestigate_ticker(ticker, market)`의 docstring 자체가 LLM에게 보이는 tool 설명이다("이미 받은 judgment로 다른 종목과의 우열을 확신할 수 없는 애매한 경우에만 선택적으로 써라").

#### `_judgment_to_payload(judgment) -> dict`

`TickerJudgment`(dataclass)를 LLM 프롬프트에 넣을 수 있는 평범한 `dict`로 변환. `rule_candidate`/`ml_prediction`이 `None`이면 그대로 `None`으로 직렬화(필드 생략이 아니라 명시적 `None` — LLM에게 "이 신호는 확인 안 됐다"는 사실을 숨기지 않기 위한 선택으로 읍):`stock_analyst_result`는 가공 없이 그대로 전달.

#### `run_recommendation_synthesis(budget, model, bind_stock_analyst_tools, judgments, top_n) -> dict`

이 모듈의 메인 진입점. 흐름:

1. **예산 사전차감**: `budget.try_consume_llm_calls(RECOMMENDATION_SYNTHESIS_LLM_CALL_ESTIMATE)`가 `False`면(예산 소진) agent를 만들지도 않고 즉시 `{"available": False, "error": "이번 배치의 LLM 호출 예산을 모두 사용했습니다"}` 반환.
2. **agent 생성**: `create_deep_agent(model=f"openai:{model}", tools=[reinvestigate_tool 1개], system_prompt=_SYSTEM_PROMPT, response_format=RecommendationSynthesisResult)`.
3. **입력 구성**: 모든 `judgments`를 `_judgment_to_payload`로 변환한 리스트를 사용자 메시지에 통째로 넣고(`f"다음 {len(payload)}개 종목의 분석 결과를 비교해 상위 {top_n}개만 순위를 매겨라: {payload}"`), `agent.invoke(..., config={"recursion_limit": RECOMMENDATION_SYNTHESIS_RECURSION_LIMIT})`로 실행.
4. **결과 추출**: `result["structured_response"]`에서 `RecommendationSynthesisResult.recommendations`를 꺼내 평범한 dict 리스트로 변환.
5. **provenance 라벨 부착**: 성공 시 `{"available": True, "recommendations": [...], "provenance": "서브에이전트 자율 조사 결과"}`를 반환하는데, 이 `provenance` 문자열은 **LLM 출력이 아니라 함수가 결정론적으로 하드코딩해서 붙인다**(`_PROVENANCE_LABEL` 상수). `stock_analyst.py`/`related_companies.py`와 동일한 "환각 방지 컨벤션" — LLM이 스스로 "나는 자율조사했다"고 주장하는 문구를 신뢰하지 않고, 코드가 그 사실을 보증한다.

**에러 처리 — "절대 raise 안 함" 계약**: 2~4단계 전체가 하나의 `try/except Exception`으로 감싸져 있다(`synthesizer.py:107-145`). 특히 주석이 명시하는 비자명한 포인트: `result["structured_response"]`라는 **키가 존재해도**, 그 값이 기대한 pydantic 인스턴스가 아니라 `None`이거나 다른 타입이면 `.recommendations` 접근에서 `AttributeError`가 날 수 있다. 이 후처리(4단계)가 try 블록 **밖**에 있으면 그 예외가 그대로 새어나가 "절대 raise 안 함" 계약이 깨진다 — `stock_analyst.py` Stage E 리뷰에서 발견된 동일한 버그 패턴이라 처음부터 try 안에 포함시켰다. `test_never_raises_when_structured_response_is_present_but_malformed`(테스트) 가 이 케이스를 `structured_response: None`으로 재현해 고정하고 있다.

**테스트 커버리지** (`tests/recommendation_synthesis/test_synthesizer.py`): `create_deep_agent`/`.invoke`를 모킹해 (1) 성공 시 provenance 라벨·recommendations 매핑·예산 소비량·tools 개수·recursion_limit 전달, (2) 예산 소진 시 agent 생성 자체를 안 하는지(`mock_create.assert_not_called()`), (3) `invoke`가 예외를 던져도 삼키는지, (4) `structured_response` 키가 없을 때, (5) 키는 있지만 값이 깨졌을 때, (6) 시스템 프롬프트에 4가지 핵심 지시 문자열이 포함돼 있는지, (7) `_reinvestigate_ticker_impl`이 budget을 그대로 공유하고 `bind_tools` 결과를 올바르게 넘기는지를 검증한다. "실제로 애매한 경우를 잘 포착해 재조사하는지"처럼 LLM의 실제 판단력은 pytest로 주장하지 않는다(이 레포 전체의 일관된 테스트 철학 — Stage C 수동 체크리스트 전례와 동일).

---

## `orchestrator/scan_coordinator.py` — 무료 스캔 조율기

### 모듈 docstring 핵심

OS 스케줄러(cron 등) 없이, **챗봇 프로세스 자신이** 배치 스캔을 트리거하는 조율기. `scripts/chat_cli.py`가 시작 시 시장별로 한 번씩 `ensure_fresh`를 호출하고, `chat_agent.tools.tool_get_market_scan_recommendations`도 질의 시점마다 자가치유 삼아 같은 메서드를 호출한다. 두 호출부 모두 idempotent — 이미 신선하거나 진행 중이면 아무것도 안 함.

이 클래스는 **챗봇 전용 로직(LLM, QueryBudget)이 전혀 없는 순수 오케스트레이션**이라 `chat_agent/`가 아니라 `orchestrator/`에 둔다.

### `_now() -> datetime`

`datetime.now(timezone.utc).replace(tzinfo=None)` — UTC naive datetime을 반환하는 내부 헬퍼. DuckDB에 저장되는 타임스탬프와 비교 가능한 형태로 통일하기 위한 것으로 보인다(timezone-aware와 naive를 섞으면 비교 시 예외가 나는 Python datetime의 특성 때문).

### `_fresh_universe(scan_cache, market, universe_size, stale_after) -> list[str]`

- **역할**: 유니버스(종목 코드 리스트) 자체를 캐싱한다. 나스닥 상장목록 조회가 실측 약 4분 걸려서, 매 트리거마다 다시 받아오면 낭비라는 이유.
- **동작**: 캐시가 없거나(`fetched_at is None`) stale이면(`_now() - fetched_at >= stale_after`) `get_universe(market, date.today())`로 새로 받아 `put_universe`에 저장. 캐시가 신선하면 그대로 쓴다. 마지막에 `tickers[:universe_size]`로 캡 적용 — **캡은 저장 시점이 아니라 읽는 시점에 적용**한다(`universe_size` 설정이 바뀌어도 캐시 자체를 무효화하지 않기 위해, `scan_cache.py:129-130`의 `put_universe` docstring에 명시).
- **모듈 레벨 함수인 이유**: `ScanCoordinator._run`뿐 아니라 `scripts/run_market_scan.py`(스레드/코디네이터 없는 수동 실행)도 재사용할 수 있게 하기 위해 클래스 메서드가 아니라 독립 함수로 뺐다. `RecommendationCoordinator`도 이 함수를 그대로 import해서 재사용한다(아래 섹션).

### `ScanCoordinator`

**생성자 파라미터**: `cache`(OHLCVCache), `scan_cache`(MarketScanCache), `account`(AccountState), `stale_after=12시간`, `universe_size=200`, `lookback_days=120`.

**인스턴스 상태**: `self._lock`(threading.Lock), `self._in_progress: set[str]` — market별 "지금 스캔 중인지" 추적.

#### `ensure_fresh(market) -> threading.Thread | None`

- 이미 in-flight거나 신선하면 **아무것도 안 하고 `None`** 반환.
- 아니면 백그라운드 스레드를 스폰해 반환한다. 프로덕션 호출부는 반환값을 버리는 fire-and-forget 방식으로 쓰지만, **테스트는 이 반환값을 `.join()`으로 결정적으로 기다릴 수 있다** — 반환값을 굳이 넘기는 이유.
- **원자성**: staleness 확인(`get_latest`로 `scanned_at` 조회) + in-progress 체크·set 추가를 **하나의 락 안에서** 처리해, 두 요청이 동시에 들어와도 이중 트리거가 안 생기게 한다.

#### `is_in_progress(market) -> bool`

단순 조회. 락으로 보호된 `self._in_progress` set 멤버십 확인.

#### `_run(market) -> None` (스레드 타깃)

1. `self._cache.cursor()`/`self._scan_cache.cursor()`로 **스레드 전용 DuckDB 커넥션**을 뜬다 — DuckDB 공식 스레딩 패턴(스레드마다 전용 커넥션)을 지키기 위해 메인 인스턴스의 커넥션을 공유하지 않는다.
2. `_fresh_universe`로 유니버스 획득 → `scan_market`(무료 경로) 실행 → `scan_cache.put_scan(market, explanations)`로 결과 적재.
3. 종목별 에러(`errors`)가 있으면 `stderr`에 경고만 출력(발송 차단 없음, `scan_market` 자체가 이미 종목 단위로 격리했으므로).
4. **바깥 `try/except Exception`**(`scan_coordinator.py:99-102`) — 백그라운드 스레드이므로 **절대 예외를 전파하지 않는다**(그 자리에서 죽으면 스레드가 조용히 사라지고 아무도 모른다). 실패 시 캐시는 stale인 채로 남고, 다음 `ensure_fresh` 호출이 자연스럽게 재시도한다.
5. `finally` 블록에서 `self._in_progress.discard(market)` — 성공/실패 어느 쪽이든 반드시 in-progress 플래그를 해제해, 다음 트리거가 영구히 막히지 않게 한다.

**스레드 안전성 책임 분리**: `OHLCVCache`/`MarketScanCache`가 각자 내부 `_write_lock` + `cursor()`로 DB 레벨 동시성을 보장하므로, `ScanCoordinator`는 **market별 in-progress 상태만** 자체 락으로 지킨다 — DB 동시성과 애플리케이션 레벨 중복 트리거 방지를 서로 다른 레이어가 책임지는 구조.

---

## `orchestrator/market_buckets.py` — 스캔 버킷(scan_key) → 원자 market 매핑 (신규)

market 세분화 작업에서 신설된 모듈. `data/models.py::VALID_MARKETS`가 원자 거래소 4개(KOSPI/KOSDAQ/NASDAQ/NYSE)만 허용하는 반면, 사용자는 그보다 넓거나 필터링된 단위로 추천을 묻는다 — 이 모듈은 그 "스캔 버킷"이 어떤 원자 market들의 조합인지만 정의한다. `RecommendationCoordinator`(바로 아래 섹션)만 이 모듈을 쓴다 — 종목 단위 6개 신호 tool이나 ①숏리스트/무료스캔은 원자 market만 다루므로 이 모듈을 몰라도 된다.

### `MARKET_BUCKETS: dict[str, list[str]]`

```python
MARKET_BUCKETS = {
    "KRX": ["KOSPI", "KOSDAQ"],
    "US": ["NASDAQ", "NYSE"],
}
```

단순 합집합 버킷 — 유니버스를 그냥 원자 market들의 유니버스를 합친 것으로 취급한다.

### `INDEX_FILTER_BUCKETS: dict[str, tuple[list[str], Callable[[], set[str]]]]`

```python
INDEX_FILTER_BUCKETS = {
    "SP500": (["NASDAQ", "NYSE"], fetch_sp500_constituents),
    "DOW": (["NASDAQ", "NYSE"], fetch_dow_constituents),
}
```

지수 구성종목으로 필터링하는 버킷. 값은 `(원자 market 목록, 구성종목 조회 함수)` 튜플 — 원자 유니버스를 합친 뒤 `fetch_sp500_constituents()`/`fetch_dow_constituents()`(`data/sources/index_constituents.py`)가 반환하는 티커 집합으로 다시 교집합 필터링한다.

### `ALL_SCAN_BUCKETS`

```python
ALL_SCAN_BUCKETS = ["KOSPI", "KOSDAQ", "KRX", "NASDAQ", "NYSE", "US", "SP500", "DOW"]
```

백그라운드 스캔 대상 전체(8개) — `scripts/chat_cli.py`가 시작 시 이 리스트 전체를 순회하며 `ensure_fresh`를 선제 호출한다.

### `bucket_markets(bucket: str) -> list[str]`

스캔 버킷(합성 또는 원자)이 어떤 원자 market들로 구성되는지 반환한다. `MARKET_BUCKETS`나 `INDEX_FILTER_BUCKETS`에 있으면 그 목록을, 둘 다 아니면(원자 market 자신, 즉 KOSPI/KOSDAQ/NASDAQ/NYSE) `[bucket]`을 반환한다 — "버킷이 곧 그 market 하나"라는 뜻. 이 폴백 덕분에 원자 market 단독 스캔은 별도 매핑 엔트리가 필요 없다.

### `is_index_filter_bucket(bucket: str) -> bool` / `fetch_index_constituents(bucket: str) -> set[str]`

`is_index_filter_bucket`은 `bucket in INDEX_FILTER_BUCKETS`를 그대로 반환. `fetch_index_constituents`는 지수 필터 버킷이 아니면 `ValueError`를 던지고(호출 전 `is_index_filter_bucket`으로 확인해야 하는 전제조건), 맞으면 등록된 구성종목 조회 함수를 호출해 반환한다.

### `bucket_universe_by_atomic_market(bucket, fetch_universe) -> dict[str, list[str]]` — 이 모듈의 핵심 함수

```python
def bucket_universe_by_atomic_market(bucket, fetch_universe):
    atomic_markets = bucket_markets(bucket)
    universe_by_market = {market: fetch_universe(market) for market in atomic_markets}
    if not is_index_filter_bucket(bucket):
        return universe_by_market
    constituents = fetch_index_constituents(bucket)
    return {market: [t for t in tickers if t in constituents] for market, tickers in universe_by_market.items()}
```

**왜 이 함수가 필요한가**: 합성 버킷(예: "KRX")의 티커를 하나의 리스트로 뭉쳐서 `build_shortlist`/`scan_market`처럼 "market 하나 + tickers 리스트"를 받는 함수에 그대로 넘기면, 그 함수들이 모든 티커에 동일한 `market` 문자열("KRX")을 붙여 `OHLCVRecord`를 생성하려다 `ValueError`가 난다("KRX"는 이제 원자값이 아니므로). 이 함수는 유니버스를 **원자 market별로 쪼갠 채**(`{"KOSPI": [...], "KOSDAQ": [...]}`) 반환해서, 호출부가 원자 market마다 별도로 `build_shortlist`를 호출하고 결과를 합치는 식으로 쓸 수 있게 한다.

`fetch_universe(atomic_market) -> list[str]`은 호출부가 캐싱 로직(`_fresh_universe`)을 쥔 콜백으로 주입한다 — 이 함수 자신은 캐싱을 전혀 모른다. 원자 market 자신(예: "KOSPI")을 넘기면 `bucket_markets`가 `["KOSPI"]`를 반환하므로 결과는 `{"KOSPI": [...]}` 하나짜리 dict가 되어 사실상 기존(세분화 이전)과 동일하게 동작한다 — 즉 이 함수를 거쳐도 원자 market 단독 스캔 경로에는 실질적인 동작 변화가 없다.

지수 필터 버킷(SP500/DOW)은 원자 유니버스를 구한 뒤 구성종목 집합으로 필터링하되, 필터링 후에도 dict의 키(원자 market 구분)는 그대로 유지된다 — 예를 들어 SP500 버킷을 스캔하면 `{"NASDAQ": [S&P500에 속하는 나스닥 종목들], "NYSE": [S&P500에 속하는 NYSE 종목들]}`이 반환된다.

---

## `orchestrator/recommendation_coordinator.py` — 유료 추천 조율기

### 모듈 docstring 핵심

`ScanCoordinator`와 동일한 "stale 체크 + in-progress 추적 + 백그라운드 스레드" 패턴을 쓰지만, `scan_market`의 계약을 안 건드린다는 원칙에 따라 **별도 클래스**로 분리됐다(기존 `ScanCoordinator`를 확장하지 않음, 2026-10-05 사용자 확정).

이 클래스는 `_bind_stock_analyst_tools`(`chat_agent/tools.py`)를 재사용하기 위해 `ChatToolContext`를 매번 구성해야 하는데, 6개 신호 tool 중 `context.scan_cache`/`context.recommendation_coordinator`를 읽는 함수는 **하나도 없다**(그 두 필드는 `tool_get_market_scan_recommendations`만 쓴다) — 그래서 이 조율기는 그 두 필드에 그냥 `None`을 넘긴다. 종목마다 `bind_tools`가 **새 `ChatToolContext`(+ `cache.cursor()`)를 매번 새로 만드는 이유**는, `run_deep_scan`이 `ThreadPoolExecutor`로 종목을 동시 처리하기 때문 — DuckDB 공식 스레딩 패턴(스레드마다 전용 커넥션)을 지키려면 종목별 호출 시점(해당 스레드 안)에 커서를 떠야 한다.

섹터/테마 경로는 `scan_key`를 `"SECTOR:{query}:{market}"` 합성키로 써서 기존 `recommendation_cache`/`recommendation_cache_meta`(`market` 컬럼을 그대로 재사용)에 적재한다 — 새 테이블을 만들지 않는다.

### 상수

- `DEFAULT_STALE_AFTER = 24시간` — `ScanCoordinator`의 12시간보다 길다. 일봉 데이터라 12시간은 과도하다는 판단(2026-10-05 확정).
- `DEFAULT_SHORTLIST_CAP = 20` — 숏리스트를 점수순 정렬 후 상위 20개로 절단(비싼 ②③단계에 들어가는 종목 수를 제한).
- `DEFAULT_TOP_N = 5` — ③단계가 최종 반환하는 추천 종목 수.
- `DEFAULT_MAX_WORKERS = 8` — `run_deep_scan`에 그대로 전달.

### `_sector_scan_key(sector_query, market) -> str`

`f"SECTOR:{sector_query}:{market}"` 합성키 생성.

### `_shortlist_score(entry: ShortlistEntry) -> float`

숏리스트를 정렬하기 위한 점수 함수. `rule_score`(규칙엔진 후보가 있으면 1.0, 없으면 0.0) + `ml_score`(ML 확률이 0.5에서 얼마나 먼지의 절댓값, 없으면 0.0)를 **단순 합산**. 둘 다 신호가 강할수록 점수가 높아지는 설계.

### `RecommendationCoordinator.__init__`

의존성이 많다 — `cache`, `scan_cache`, `ml_models: dict[str, ModelBundle | None]`(시장별 모델), `llm_client`, `news_client`, `sentiment_client`(세 보조신호 클라이언트, ②단계에서 `stock_analyst`에 넘겨짐), `account`, `agent_model`(사용할 LLM 모델명), `reader`(섹터 1차 분류용 — `chat_agent`의 기존 reader를 재사용), 그리고 각종 상한값. `self._lock`/`self._in_progress: set[str]`는 `ScanCoordinator`와 동일한 역할이지만, 여기서는 **market 이름뿐 아니라 섹터 합성키도 같은 set에 들어간다**(공유 범용 메서드로 처리하기 때문).

### `_ensure_fresh_for(scan_key, run) -> threading.Thread | None`

전체 유니버스/섹터 두 경로가 공유하는 범용 stale체크+in-progress 추적 — `ScanCoordinator.ensure_fresh`와 동일한 원자적 락 패턴을 `scan_key`라는 일반화된 키로 재사용 가능하게 만든 버전. `run`은 호출할 실행 함수를 콜백으로 받는다.

### `ensure_fresh(bucket)` / `ensure_fresh_sector(sector_query, market)`

`ensure_fresh`의 매개변수명이 market 세분화 작업에서 `market`→`bucket`으로 바뀌었다(동작은 동일, "원자 market 또는 8개 합성 버킷 중 하나를 받을 수 있다"는 사실을 명확히 하려는 목적) — `orchestrator/market_buckets.py::ALL_SCAN_BUCKETS` 8개 중 어느 것이든 넘길 수 있다. `_ensure_fresh_for`에 자신의 `scan_key`(=bucket 그 자체)와 실행 람다(`self._run(scan_key)`)를 넘기는 얇은 래퍼라는 구조는 변하지 않았다. 섹터 경로(`ensure_fresh_sector`)는 매개변수명이 그대로 `market`이고, 원자/합성 구분 없이 받은 값을 그대로 `_sector_scan_key`/`resolve_sector_tickers`에 전달한다 — 섹터 경로는 ①숏리스트를 생략해서 애초에 `build_shortlist`를 호출하지 않으므로(아래 `_run_sector` 참고) `OHLCVRecord` market 검증 문제 자체가 발생하지 않기 때문이다.

### `is_in_progress(bucket)` / `is_sector_in_progress(sector_query, market)`

단순 조회 메서드, `ScanCoordinator.is_in_progress`와 동일한 패턴. `is_in_progress`도 매개변수명이 `bucket`으로 바뀌었다(동작 동일).

### `_budget_for(shortlist_size) -> QueryBudget`

**최악의 경우 상한**을 계산해 하나의 `QueryBudget`을 만든다:

```
worst_case = shortlist_size × STOCK_ANALYST_LLM_CALL_ESTIMATE
           + RECOMMENDATION_SYNTHESIS_LLM_CALL_ESTIMATE
           + RECOMMENDATION_SYNTHESIS_RECURSION_LIMIT × STOCK_ANALYST_LLM_CALL_ESTIMATE
```

세 항의 의미: ① 숏리스트 전체가 각각 `stock_analyst` 재귀상한까지 다 쓰는 경우(②단계), ② 종합 agent 자신의 기본 호출(③단계 본체), ③ 종합 agent가 재귀상한만큼 `reinvestigate_ticker`를 호출해 그만큼 `stock_analyst`를 재귀적으로 다시 띄우는 경우(③단계의 재조사 비용). 이 세 가지를 전부 더한 것이 "이론상 최대 소비량"이고, 이 값으로 **하나의 budget 객체**를 만들어 ②③ 전체 과정에서 공유한다 — 재조사 비용이 전체 예산에 정확히 반영되는 이유가 바로 이 budget 객체 공유다(앞서 synthesizer 섹션에서 설명한 `_reinvestigate_ticker_impl`의 budget 전달과 맞물린 설계).

### `_make_bind_tools(budget) -> Callable[[str, str], list]`

클로저로 `budget`을 포획한 `bind_tools(ticker, market)` 함수를 반환. 호출될 때마다 새 `ChatToolContext`를 구성(`cache.cursor()`로 스레드 전용 커넥션 사용, `scan_cache`/`recommendation_coordinator`는 `None`)하고 `_bind_stock_analyst_tools(context, ticker, market)`를 호출해 그 종목에 묶인 6개 신호 tool 리스트를 반환한다. 이 함수가 `run_deep_scan`과 `run_recommendation_synthesis`(재조사용) **양쪽에** 전달되는 `bind_tools` 콜백이다.

### `_run_deep_scan_and_synthesize(shortlist, scan_key, scan_cache) -> None`

②+③을 공통 실행하는 내부 헬퍼 — 전체 유니버스/섹터 두 경로가 공유한다. `scan_cache`는 호출부가 이미 떠둔 스레드 전용 커서를 그대로 받는다(중복 `cursor()` 호출 방지).

흐름: `_budget_for(len(shortlist))`로 예산 생성 → `_make_bind_tools(budget)` → `run_deep_scan(shortlist, budget, agent_model, bind_tools, max_workers)`(②) → `run_recommendation_synthesis(budget, agent_model, bind_tools, judgments, top_n)`(③, 같은 `budget`과 `bind_tools`를 전달). 결과가 `synthesis["available"]`이면 `scan_cache.put_recommendations(scan_key, synthesis["recommendations"])`로 캐시에 적재, 아니면 `stderr`에 경고만 출력(알림/캐시 적재 모두 조용히 스킵, 예외 전파 없음).

### `_run(bucket) -> None` (전체 유니버스 경로, 스레드 타깃) — market 세분화로 재작성됨

세분화 이전에는 `_run(market)`이 `_fresh_universe` 한 번 → `build_shortlist` 한 번만 호출하는 단순한 흐름이었다. 지금은 `bucket`이 원자 market일 수도, 합성 버킷(KRX/US/SP500/DOW)일 수도 있어서 다음과 같이 바뀌었다:

1. 스레드 전용 커서 획득(`cache.cursor()`/`scan_cache.cursor()`).
2. 내부 클로저 `fetch_universe(atomic_market)`을 정의 — `_fresh_universe(scan_cache, atomic_market, universe_size, stale_after)`를 그대로 호출하는 콜백.
3. `bucket_universe_by_atomic_market(bucket, fetch_universe)`(`orchestrator/market_buckets.py`)를 호출해 `{원자market: tickers}` dict를 얻는다. `bucket`이 원자 market이면 이 dict는 `{bucket: tickers}` 하나뿐이고, 합성 버킷이면 2개(KRX→KOSPI+KOSDAQ, US→NASDAQ+NYSE) 또는 SP500/DOW 필터링을 거친 2개(NASDAQ+NYSE 중 지수 구성종목만)가 된다.
4. 이 dict를 순회하며 **원자 market마다 각각** `build_shortlist(cache, tickers, atomic_market, self._ml_models.get(atomic_market), lookback_days)`(①)를 호출하고, 반환된 `ShortlistEntry`들과 에러들을 하나의 리스트로 누적한다. **`self._ml_models.get(atomic_market)`가 핵심** — `ml_models` dict의 키는 원자 4개(KOSPI/KOSDAQ/NASDAQ/NYSE)뿐이므로(ML은 원자시장별로만 학습됨, `scripts/train_ml_model.py` 참고), 합성 버킷을 스캔할 때도 각 원자 market 자신의 ML 모델을 올바르게 조회해 쓴다.
5. 누적된 전체 숏리스트를 `_shortlist_score` 기준 내림차순 정렬 → `shortlist[:self._shortlist_cap]`(상위 20개로 절단) — 이 단계는 세분화 이전과 동일, 여러 원자 market에서 모인 종목들이 점수 하나의 기준으로 함께 경쟁한다.
6. `_run_deep_scan_and_synthesize(shortlist, bucket, scan_cache)`(②③) — 이 단계는 세분화 이전과 동일한 코드다. 숏리스트에 여러 market의 종목이 섞여 있어도 문제없다 — 각 `ShortlistEntry`가 자기 `market` 필드를 들고 있어서 ②단계(`run_deep_scan`)가 종목별로 올바른 market을 사용한다. 결과는 `scan_key=bucket`(합성 버킷 이름 그대로, 예: "KRX")으로 캐시에 적재된다.
7. 숏리스트 빌드 중 생긴 종목별 에러는 `stderr` 경고로만 남김(합쳐진 `build_errors` 전체 건수).
8. 바깥 `try/except Exception` — `ScanCoordinator`와 동일 철학("백그라운드 스레드는 절대 예외를 전파하지 않는다").
9. `finally`에서 `self._in_progress.discard(bucket)`.

### `_run_sector(sector_query, market, scan_key) -> None` (섹터/테마 경로, 스레드 타깃)

1. `_budget_for(0)`으로 "종합 전용 상한" 예산 생성(해석 단계는 별도 예산이라는 의미 — 즉 숏리스트 크기 0 가정이므로 `shortlist_size × STOCK_ANALYST_LLM_CALL_ESTIMATE` 항이 0이 되어, 종합 단계 전용 예산만 남는다).
2. `resolve_sector_tickers(self._reader, budget_for_resolution, agent_model, sector_query, market)` — 섹터/테마 질의("로봇 관련주")를 실제 종목 리스트로 해석. 실패하면(`resolution["available"]`이 거짓) 경고만 남기고 조용히 리턴(②③ 단계 자체를 시작하지 않음).
3. 해석된 티커 목록을 `scan_cache.put_sector_tickers`에 캐싱(종목별 confidence 라벨 — `t.get("confidence", "confirmed")` — 포함).
4. **①숏리스트 단계를 생략한다** — `ShortlistEntry`를 `rule_candidate=None, ml_prediction=None`으로 바로 구성해서 넘긴다. 이유가 코드에 명시돼 있다: 규칙엔진/ML은 `stock_analyst` 자신이 필요하면 **스스로 호출**하므로, 섹터 경로에서는 사전 필터링이 불필요하다(섹터 질의는 보통 소수 종목이라 비용 절감 필요성도 낮다는 암묵적 가정으로 보인다).
5. `_run_deep_scan_and_synthesize`(②③) 공통 실행.
6. 바깥 `try/except`, `finally` 블록은 `_run`과 동일한 패턴.

---

## 전체 데이터 흐름 재정리 (실행 진입점부터)

1. `scripts/chat_cli.py`가 기동 시 `RecommendationCoordinator(cache=..., scan_cache=..., ml_models=..., llm_client=..., news_client=..., sentiment_client=..., account=..., agent_model=..., reader=...)`를 만든다. `ml_models`는 `_ATOMIC_MARKETS = ["KOSPI", "KOSDAQ", "NASDAQ", "NYSE"]`(원자 4개만, ML은 합성버킷 학습 안 함) 각각에 대해 `load_model`을 시도해 조립한다. 그 다음 `orchestrator.market_buckets.ALL_SCAN_BUCKETS`(8개: KOSPI/KOSDAQ/KRX/NASDAQ/NYSE/US/SP500/DOW) 전체를 순회하며 `recommendation_coordinator.ensure_fresh(bucket)`를 호출한다(`chat_cli.py:174-175`) — market 세분화 이전에는 `_MARKETS = ["KRX", "NASDAQ"]` 2개뿐이었다.
2. 각 버킷의 캐시가 24시간 이상 stale이면 백그라운드 스레드가 뜬다 — `_run(bucket)` → `bucket_universe_by_atomic_market`으로 원자 market별 유니버스 획득 → 원자 market마다 ①`build_shortlist` 호출 후 결과 병합 → 정렬·절단 → ②`run_deep_scan` → ③`run_recommendation_synthesis` → `MarketScanCache.put_recommendations(scan_key=bucket, ...)`. 8개 버킷은 서로 완전히 독립적으로 트리거·캐싱된다(각자 다른 `scan_key`) — "KRX" 버킷을 갱신해도 "KOSPI"/"KOSDAQ" 버킷의 캐시는 별도로 그대로 남는다(세 캐시 모두 내부적으로는 같은 원자 유니버스 캐시(`_fresh_universe`)를 공유해 재사용하지만, 숏리스트~추천 결과 자체는 버킷별로 독립 저장된다).
3. 사용자가 대화 중 "추천해줘" 류의 질의를 하면 `chat_agent/tools.py`의 해당 tool이 `MarketScanCache.get_latest_recommendations(scan_key)`로 **이미 캐시에 적재된 결과**를 즉시 반환한다 — `market` 인자로 8개 버킷 중 어느 것을 넘기든 그대로 `scan_key`로 쓰인다(이 조율기 체인이 생성한 결과를 소비하는 쪽은 이 리뷰의 범위 밖, `chat_agent/tools.py` 문서 참고).
4. 섹터 질의("로봇 관련주 추천해줘", "반도체관련 나스닥 종목 추천해줘")는 같은 체인을 `_run_sector`로 타되, ①을 생략하고 `resolve_sector_tickers`로 대체한다 — 섹터 경로의 `market` 인자는 원자/합성 구분 없이 그대로 전달되므로(①숏리스트가 없어 `OHLCVRecord` 생성 자체가 발생하지 않음), "로봇관련 KRX 종목"처럼 합성 버킷을 섹터 질의에 써도 코드상 문제는 없다 — 다만 1차(공식 업종 매칭)는 `sector_classification.py`가 KOSPI/KOSDAQ/KRX만 지원하므로, NASDAQ/NYSE/US 쪽 섹터 질의는 항상 2차(LLM 제안+뉴스검증) 경로로 간다.

이 구조에서 신규 참여자가 가장 먼저 이해해야 할 것은 두 가지다:

- **"무료 경로"(`ScanCoordinator`/`scan_market`, LLM 0회)와 "유료 경로"(`RecommendationCoordinator`/`build_shortlist`+`run_deep_scan`+`run_recommendation_synthesis`, LLM 다수 호출)가 완전히 독립된 두 개의 캐시 네임스페이스(`market_scan` 테이블 vs. `recommendation_cache` 테이블)와 두 개의 조율기 클래스로 나뉘어 공존한다는 점**이다. 둘을 혼동하면 "왜 같은 질의에 두 가지 다른 결과가 나오는가"를 오해하기 쉽다. 무료 경로(`ScanCoordinator`/`scripts/run_market_scan.py`)는 원자 market 4개만 알고, 합성 버킷(KRX/US/SP500/DOW) 개념 자체가 없다.
- **"원자 market"(레코드/캐시 PK, 종목 하나가 실제로 속한 거래소)과 "합성 scan_key"(백그라운드 스캔 버킷 전용, 여러 원자 market을 아우르는 질의 단위)가 레이어가 다르다는 점**이다. 이 구분은 `RecommendationCoordinator._run`과 `orchestrator/market_buckets.py` 안에서만 존재하고, 그 아래(①숏리스트, ②딥다이브, ③종합, 종목단위 신호 tool)로 내려가면 전부 원자 market만 다룬다 — 합성 버킷은 `_run`이 원자 market별로 쪼개는 순간 사라진다.
