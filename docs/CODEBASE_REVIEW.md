# auto_stock 코드베이스 리뷰 (현재까지 구현 현황)

> 최초 작성 2026-08-27, 갱신 2026-08-28. 이 문서는 각 모듈이 "무엇을 담당하고 어떤 역할을 하는지"를 정리한 것이며, 설계 근거·의사결정 배경은 `docs/design/*.md`에 더 자세히 있다(이 문서는 그것들의 상위 요약). 코드가 구현/수정될 때마다 이 문서도 함께 갱신한다.

## 진행 중: LLM 차트분석 에이전트 (#3)

`docs/design/llm-chart-analyst-plan.md`(OpenAI `gpt-5.6-luna` 기준) 승인 완료, **Phase 0(의존성 게이트) 완료**:
- `pyproject.toml`에 `openai`(3.5.0 설치됨), `pydantic>=2` 추가
- 기존 126개 테스트 회귀 없음 확인
- 설치된 SDK로 `client.responses.parse`/`response.output_parsed` 존재 및 예외 계층(`APITimeoutError ⊂ APIConnectionError`, `RateLimitError`/`AuthenticationError` 등 ⊂ `APIStatusError` ⊂ `APIError` ⊂ `OpenAIError`) 직접 확인 — 계획서 설계와 일치
- `.env.example`에 `OPENAI_API_KEY`(+ 선택 `OPENAI_MODEL`, `LLM_MAX_CALLS_PER_RUN`) 추가

**Phase 1(순수 계층) + Phase 2(클라이언트·분석 계층, 전부 모킹) 완료**:
- `src/auto_stock/llm_chart_analyst/` 8개 파일(`models.py`/`schema.py`/`snapshot.py`/`prompt.py`/`credentials.py`/`client.py`/`analyst.py` + `__init__.py`) 신규 구현 — 파일별 역할은 아래 모듈 표 참고
- `tests/llm_chart_analyst/` 신규 6개 테스트 파일(`conftest.py` + `test_snapshot.py`/`test_prompt.py`/`test_client.py`/`test_credentials.py`/`test_analyst.py`) — 55개 테스트 전부 통과(code-reviewer 리뷰 후 회귀 테스트 1개 추가), 신규 파일 커버리지 100%
- TDD RED→GREEN을 파일 단위로 진행: 각 모듈 테스트를 먼저 작성해 `ModuleNotFoundError`로 실패 확인 후 최소 구현, 매 단계 후 전체 스위트(126개 기존 포함) 회귀 없음 확인
- 계획 대비 변경점 1가지: `ChartAnalysis.model`(감사 추적용 필드)을 채우기 위해 `ChartPatternReader` Protocol에 `model: str` 속성을 추가했다 — 계획 문서의 Protocol 원안은 `read_pattern` 메서드 하나만 정의했지만, `analyze()`가 응답이 아니라 클라이언트 자신에게서 실제 사용 모델 ID를 가져와야 하는데 메서드 시그니처만으로는 이를 표현할 수 없었다. `OpenAIChartClient.__init__`에서 `self.model = config.model`로 채우고, 테스트용 `FakeChartPatternReader`도 동일하게 `model` 속성을 갖는다. 그 외에는 계획 문서 그대로 구현했다.
- 실제 SDK 예외 계층을 재확인해 `client.py`의 예외 매핑 순서(`APITimeoutError` → `APIConnectionError` → `RateLimitError`/`AuthenticationError` → `APIStatusError` → `pydantic.ValidationError` → `output_parsed is None`)를 정확히 구현하고, `openai.OpenAI`를 전부 모킹한 6종류 예외 매핑 + 호출 예산 상한 + API 키 미유출 테스트로 고정
- `OPENAI_API_KEY`는 아직 미발급 — Phase 3(실제 API 호출, 오케스트레이터 배선)부터 필요. Phase 3(`pipeline.py` 배선)은 이번 태스크 범위 밖.

**`code-reviewer` 리뷰 후 수정 2건** (커밋 전 반영, 181개 테스트로 회귀 확인):
- **MEDIUM**: `LLMConfig.max_tokens`가 실제로는 `responses.parse` 호출에 전달되지 않아 출력 토큰 상한이 무제한이었다 — `client.py`에 `max_output_tokens=self._config.max_tokens` 인자를 추가해 실제로 강제되게 고쳤다.
- **LOW-MEDIUM**: `snapshot.py`의 정규화 기준값(`window[0].close`, 즉 최신 시점에서 `recent_bars`일 전 종가)이 0이면 `ZeroDivisionError`가 날 수 있었다(SMA60/RSI/ATR는 모두 최신 시점 위주 계산이라 이 값이 0이어도 `latest_feature_vector` 자체는 정상 반환되는 경우가 있어 놓치기 쉬운 케이스). `base_close == 0`이면 `None`을 반환하도록 가드 추가 + 회귀 테스트(`test_build_snapshot_returns_none_when_base_close_is_zero`) 추가.

**Phase 3(오케스트레이터 배선) 완료** (커밋 전, 188개 테스트로 회귀 확인 — 기존 181개 + 신규 7개):
- `src/auto_stock/orchestrator/pipeline.py`에 `llm_client: ChartPatternReader | None = None` 파라미터와 `_llm_reasons` 헬퍼(`_ml_reasons`와 동일 계약 — 절대 raise하지 않고 `([], str(exc))` 반환) 추가. import는 계획대로 `analyze_chart`/`to_chart_reasons`로 alias하고 ML 쪽 `predict`/`to_reasons`는 이름을 그대로 유지해 기존 `mocker.patch("...pipeline.predict")`/`...to_reasons` 패치 타깃을 깨지 않았다.
- 분기 조건은 계획대로 `if ml_model is None and llm_client is None:`(2-위치인자 `generate_explanation` 호출 보존) / `else:`(ML 근거 → LLM 근거 순서로 병합해 `extra_reasons`에 전달) 하나로 통일 — 소스별 `if`로 쪼개지 않아 기존 회귀 테스트(2-인자 호출을 단정하는 3개) 무수정 통과.
- ML/LLM 실패는 각각 독립적으로 `errors`에 `"ML 예측 실패: ..."`/`"LLM 차트분석 실패: ..."`로 기록되며, 한쪽이 실패해도 다른 쪽 근거 문구는 그대로 살아남는다(신규 테스트로 양방향 모두 잠금) — 실패해도 알림 발송은 계속됨.
- `tests/orchestrator/test_pipeline.py`에 신규 7케이스 추가(기존 7개는 문자 그대로 무수정): 무보조신호 회귀 확장판(`analyze_chart` 미호출 포함), LLM단독, ML+LLM 병합 순서(ML 먼저 → LLM 나중), LLM실패 격리, LLM실패해도 ML 근거 유지, ML실패해도 LLM 근거 유지, 이중실패 시 `errors` 2건 기록 + 발송 유지.
- `scripts/verify_llm_chart_analyst.py` 신규 작성 — `LLM_VERIFY_DRY_RUN=1`이면 실제 종목(005930) OHLCV로 `build_snapshot` → 시스템/유저 프롬프트만 출력(API 호출 0, 비용 0), 아니면 `load_llm_config()` + `OpenAIChartClient(max_calls_per_run을 1로 강제)`로 실제 1회 호출 후 구조화 결과 출력. **이번 태스크에서는 dry-run 모드만 직접 실행해 확인**(정상 동작 확인됨 — SUCCESS 출력 + 프롬프트에 ticker/market/실제 날짜/action 미노출 육안 확인). 실호출 경로는 코드만 작성했고 실행하지 않았다(`OPENAI_API_KEY` 미발급).
- `scripts/run_recommendations_with_signals.py` 신규 작성(`run_recommendations_with_ml.py`의 ML+LLM 동시 실행 변형) — ML 아티팩트 또는 `OPENAI_API_KEY` 중 하나라도 없으면 안내 후 종료. **이 스크립트는 작성만 하고 실행하지 않았다**(실제 유료 API 호출 경로이므로 키 발급 전 실행 금지 지침 준수).
- 계획과 다른 점 없음 — `docs/design/llm-chart-analyst-plan.md` "오케스트레이터 통합" 절의 코드 스케치와 4가지 핵심 포인트를 그대로 구현.
- `OPENAI_API_KEY` 발급 전이라 **배선은 완료됐지만 실제 API 호출로 검증된 적은 아직 없다** — 발급 후 `scripts/verify_llm_chart_analyst.py`(실호출 모드)와 `scripts/run_recommendations_with_signals.py`로 사용자가 직접 1회 검증 필요.

## 전체 그림

```
데이터 수집(data) ─┬─▶ 규칙엔진(rule_engine) ─┬─▶ 리스크 사이징(risk_sizing) ─┐
                    │                          │                              │
                    └─▶ ML 예측(ml_predictor) ─┘(보조 신호)                   ▼
                                                                   설명 생성(explainer)
                                                                          │
                                                                          ▼
                                                                   알림봇(notifier)
                                                                          │
                        오케스트레이터(orchestrator)가 위 전체를 배선 ◀──┘
```

주문 실행(#8, MVP-1)은 아직 없다 — 지금은 "추천까지"만 하는 MVP-0 + ML 보조신호(#2) 단계다.

---

## `src/auto_stock/data/` — 데이터 수집 계층 (#0)

시세 데이터를 확보해 다른 모든 모듈에 공급하는 공용 인프라. 에이전트가 아니라 순수 인프라 모듈.

| 파일 | 역할 |
|---|---|
| `models.py` | `OHLCVRecord(ticker, market, date, open, high, low, close, volume)` — 시스템 전체에서 쓰는 유일한 시세 데이터 형태. `market`이 `KRX`/`NASDAQ`이 아니거나 `high < low`면 생성 시점에 예외를 던져 잘못된 데이터가 시스템에 들어오는 걸 막는다. |
| `sources/fdr_source.py` | `fetch_ohlcv(ticker, start, end, market)` — `FinanceDataReader`로 KRX/NASDAQ 시세를 동일한 방식으로 조회. |
| `sources/pykrx_source.py` | `get_ticker_list`(KRX 전체 상장종목), `get_market_cap`(개별 종목 시가총액) — `pykrx`로 KRX 전용 데이터를 보완. **`load_dotenv()`를 각 함수 안에서 호출** — pykrx가 `KRX_ID`/`KRX_PW` 로그인을 요구하는데, import 시점의 최초 로그인은 항상 실패하지만(모듈 로드 시 `.env`가 아직 안 읽힘) 함수 호출 시점에 재로그인을 시도해 성공한다(이번 세션에서 고친 버그). |
| `cache.py` | `OHLCVCache` — DuckDB(`data/ohlcv.duckdb`) 기반 로컬 캐시. `put`/`get`/`covers`(요청 구간을 캐시가 이미 포함하는지 최소/최대 날짜로 근사 판단). |
| `service.py` | `get_ohlcv`(캐시 우선 조회, 미스 시 `fdr_source` 호출 후 캐시 적재), `get_universe`(KRX는 `pykrx_source`, NASDAQ은 FDR 상장목록), `refresh_recent`(폴링용 최근 N일 갱신). |
| `scheduler.py` | `is_market_open`(시장별 정규장 시간 판단, 타임존 인지), `poll_market`/`register_polling_jobs` — APScheduler로 장중에만 주기적으로 시세를 갱신하는 인프라. **아직 실제로 실행 파이프라인에 연결되지는 않았다** — 정의만 돼 있고 어떤 스크립트도 호출하지 않음. |

## `src/auto_stock/rule_engine/` — 규칙엔진 (#1)

기술적 지표 기반 1차 매수/매도 후보 필터링. 외부 API/LLM 의존 없어 가장 먼저 만든 신호원.

| 파일 | 역할 |
|---|---|
| `models.py` | `Candidate(ticker, market, action, reasons)` — 규칙엔진(및 이후 신호원 확장 시)의 출력 형태. |
| `indicators.py` | `sma`/`ema`/`rsi`/`atr`/`macd` — 직접 구현하지 않고 검증된 `pandas-ta` 라이브러리를 얇게 래핑. 입력/출력은 순수 `list[float \| None]`(인덱스 정렬)이라 호출자는 pandas-ta를 몰라도 됨. |
| `engine.py` | `generate_candidates(records)` — RSI 과매수/과매도 + SMA20/60 골든·데드크로스를 결합해 BUY/SELL 후보 생성. BUY·SELL 근거가 동시에 있으면(상충) 후보를 만들지 않는 보수적 규칙. |

## `src/auto_stock/risk_sizing/` — 리스크·포지션 사이징 (#5, 참고용)

PRD §6 리스크 정책을 코드화하되, **1차 목표 범위는 참고용 제안까지만** — 실제 주문 차단(집행)은 아직 없음.

| 파일 | 역할 |
|---|---|
| `models.py` | `AccountState(equity, held_tickers, total_exposure_pct)`(입력값 검증 포함), `SizingSuggestion(ticker, market, action, suggested_quantity, suggested_allocation_pct, stop_loss_price, take_profit_price, limit_check, notes)`. |
| `sizing.py` | `suggest_position(candidate, records, account)` — ATR 기반으로 손절(`-1.5×ATR`)/익절(`+3×ATR`)가 및 포지션 크기(변동성 높을수록 배분 축소, 종목당 최대 5%)를 계산하고, 최대 동시보유(10종목)/총 익스포저(50%) 한도 초과 여부를 `limit_check`로 표시만 한다(차단 실행 없음). SELL 후보는 "신규 매수 사이징 대상 아님"으로 처리. |

## `src/auto_stock/explainer/` — 추천 설명 생성기 (#6)

여러 신호 계층의 근거를 사람이 읽을 수 있는 문장으로 종합. **LLM을 쓰지 않는 결정론적 템플릿**(MVP-0 시점엔 구조화된 데이터만 있어 템플릿으로 충분).

| 파일 | 역할 |
|---|---|
| `models.py` | `Explanation(ticker, market, action, summary)`. |
| `generator.py` | `generate_explanation(candidate, sizing, extra_reasons=None)` — 규칙엔진 근거 + 사이징 제안(수량/손절익절) + `extra_reasons`(ML #2/LLM 차트 #3/뉴스 #4가 채워 넣을 확장 포인트)를 문장으로 조립. **ML 예측 모듈 추가 시 이 파일은 한 줄도 수정되지 않았다** — 애초에 이 확장을 염두에 두고 설계된 덕분. |

## `src/auto_stock/notifier/` — 알림봇 (#7, 알림 전용)

승인 UI 요소는 없고 텔레그램으로 편도 알림만 보낸다. 승인/거부 콜백 처리는 MVP-1(#8)에서 추가 예정.

| 파일 | 역할 |
|---|---|
| `models.py` | `TelegramCredentials(bot_token, chat_id)`. |
| `credentials.py` | `load_telegram_credentials()` — `.env`에서 `load_dotenv()` 후 `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`를 읽음(없으면 `KeyError`). 이 리포에서 "진입점이 필요 시점에 직접 `load_dotenv()` 호출"하는 패턴의 원조. |
| `telegram_bot.py` | `send_notification(explanation, credentials)` — 텔레그램 Bot API로 메시지 전송, 네트워크 오류/HTTP 실패/API 오류를 모두 `TelegramNotificationError`로 통일해 던짐. |

## `src/auto_stock/orchestrator/` — 오케스트레이터

위 모듈들을 실제로 연결하는 배선(wiring) 코드. **새로운 계산 로직은 없다** — 이미 각자 테스트된 함수들을 순서대로 호출.

| 파일 | 역할 |
|---|---|
| `models.py` | `PipelineResult(sent: list[Explanation], errors: list[tuple[str, str]])`. |
| `pipeline.py` | `run_recommendation_pipeline(cache, tickers, market, account, credentials, lookback_days=120, ml_model=None, llm_client=None)` — 티커마다 `get_ohlcv → generate_candidates → suggest_position → (선택)ML예측/(선택)LLM차트분석 → generate_explanation → send_notification` 순으로 호출. 종목 단위로 예외를 격리(`errors`에 기록하고 다음 종목 계속). `ml_model`과 `llm_client`가 **둘 다** `None`(기본값)이면 두 보조신호 코드 경로를 전혀 타지 않아 ML/LLM 추가 전과 100% 동일하게 동작(2-위치인자 `generate_explanation` 호출 보존) — 회귀 테스트로 고정돼 있음. 둘 중 하나라도 켜지면 `extra_reasons`에 ML 근거 → LLM 근거 순서로 병합해 전달한다. ML/LLM 예측이 각각 실패해도 알림 발송은 막지 않고 `errors`에 독립적으로 기록(`_ml_reasons`/`_llm_reasons` 헬퍼가 절대 예외를 밖으로 던지지 않으며, 한쪽 실패가 다른 쪽 근거를 삼키지 않음). |

## `src/auto_stock/ml_predictor/` — ML 예측 모듈 (#2, 보조 신호)

규칙엔진 후보에 대해 "N거래일 후 상승 확률"을 계산해 **보조 근거**로만 추가. 자체 후보를 만들지도, 알림을 필터링하지도 않는다.

| 파일 | 역할 |
|---|---|
| `models.py` | `FeatureVector`, `LabeledSample`, `TrainingDataset`, `DatasetSplit`, `ModelMetadata`(학습 시점의 기간/샘플수/성능지표 등 전부 기록), `ModelBundle`(학습된 estimator + 메타데이터), `MLPrediction`(확률 + 기여 피처). |
| `features.py` | `build_feature_vectors(records)` — OHLCV로부터 11종 스케일 프리 피처(수익률 1/5/20일, RSI, MACD 히스토그램 정규화, SMA20/60 이격도, ATR%, 거래량비, 채널위치) 산출. 전부 **후행(trailing) 윈도우만 사용**(lookahead 방어). `latest_feature_vector`는 추론 전용이며 내부적으로 학습용 함수를 그대로 호출해 학습/추론 간 피처 계산 로직이 갈라지지 않게 함. |
| `labeling.py` | `forward_returns`/`to_label` — `close[t+5]` 기준 이진 레이블(상승/하락). 레이블 없는 마지막 5개 행은 자동 제외. |
| `dataset.py` | `build_training_dataset`(다종목 pooled, 전역 날짜순 정렬), `chronological_split`(시간순 분할 + embargo 갭, **랜덤 분할 절대 금지**), `to_xy`(sklearn 입력 형태 변환). |
| `training.py` | `build_estimator("logreg"/"rf"/"dummy")` — logreg는 `StandardScaler`+`LogisticRegression`을 `Pipeline`으로 묶어 fold 밖 통계 누출 차단. `train`(학습+평가+메타데이터 생성), `evaluate`(AUC/accuracy/base_rate), `walk_forward_scores`(`TimeSeriesSplit` 5-fold). rf/dummy는 평가 리포트 전용 벤치마크, 프로덕션엔 logreg만 사용. |
| `artifact.py` | `save_model`/`load_model` — `.joblib`(모델) + `.metadata.json`(성능 기록) 사이드카. `.joblib`은 `.gitignore`로 커밋 제외, 메타데이터만 커밋 대상. |
| `predictor.py` | `predict(bundle, records)` — 최신 피처로 상승확률 계산, 선형모델이면 `coef_ × 표준화값`으로 top-3 기여 피처 산출. `to_reasons(prediction, action)` — 규칙엔진 신호와의 동의/상충/중립 문구 + "백테스트 검증 전 참고용" 고지를 항상 포함(PRD §10 준수). |

**현재 한계**: 학습/추론 모두 `MARKET="KRX"`로 하드코딩돼 있어 **나스닥은 학습·추천 어디에도 포함되지 않는다**(사용자와 확인된 사항 — 확장하려면 시장별로 별도 모델을 만들어야 하고, `chronological_split`이 단일 거래캘린더를 전제하므로 KRX+NASDAQ을 하나로 풀링하려면 재검증이 필요). 실제 200종목·5년치 학습도 아직 실행 전(사용자 판단으로 보류 중) — 지금까지는 3~8종목짜리 스모크 테스트만 돌려봤고 그 산출물은 커밋하지 않음.

## `src/auto_stock/llm_chart_analyst/` — LLM 차트분석 에이전트 (#3, 보조 신호, Phase 1~2 완료)

OHLCV + 기술적 지표의 수치 요약을 OpenAI GPT(`gpt-5.6-luna`)에 입력해 차트 패턴을 구조화된 형태로 해석하고, 규칙엔진 후보에 대한 **보조 근거**로만 추가한다(ML #2와 동일하게 자체 후보를 만들지도 알림을 필터링하지도 않음). 프롬프트에 티커·종목명·시장·실제 날짜·규칙엔진 action을 절대 넣지 않는 것이 이 모듈의 핵심 설계 결정(환각·동조 방어, PRD §10). Phase 3(오케스트레이터 배선, 실제 API 최초 호출)는 아직 미착수.

| 파일 | 역할 |
|---|---|
| `models.py` | `BarSummary`(오프셋별 정규화 OHLCV+거래량비), `ChartSnapshot`(ticker/market/date + `bars` + `indicators` 11종), `ChartAnalysis`(LLM 판독 결과 + 감사용 `model` 필드), `LLMConfig`, `ChartPatternReader`(Protocol — `read_pattern` 메서드 + `model` 속성). 이 Protocol 덕분에 `analyst.py`가 `openai`를 전혀 import하지 않고도 테스트 가능. |
| `schema.py` | `ChartPatternRead` — `client.responses.parse(..., text_format=ChartPatternRead)`가 검증하는 Pydantic 출력 스키마. `direction`(UP/DOWN/NEUTRAL)/`confidence`(LOW/MEDIUM/HIGH, 서수 — ML의 퍼센트 표기와 혼동 방지)/`pattern_name`/`rationale`/`caveat`. 목표가·손절가·수량 필드는 의도적으로 없음(리스크사이징 소관과 충돌 방지). |
| `snapshot.py` | `build_snapshot(records, recent_bars=30)` — 워밍업(SMA60) 미충족이면 `None`. 가격은 구간 첫 종가=100 기준 정규화, 거래량은 20일 평균 대비 배율. `indicators`는 **새 지표 수식을 짜지 않고** `ml_predictor.features.latest_feature_vector`를 그대로 재사용해 채운다. |
| `prompt.py` | `SYSTEM_PROMPT`(NEUTRAL 우선·목표가 금지·확률 대신 서수 confidence 등 규칙 6개 명시), `render_bar_table`/`render_indicators`/`build_user_prompt(snapshot)`. **`build_user_prompt`는 ticker/market/실제 날짜/action을 받지 않는다** — 함수 시그니처 자체가 환각·동조 방어. 예측 지평 문구("약 N거래일")는 `ml_predictor.labeling.LABEL_HORIZON_DAYS`를 재사용해 ML #2와 지평을 동기화. |
| `credentials.py` | `load_llm_config()` — `notifier/credentials.py`와 동일하게 `load_dotenv()` + `os.environ["OPENAI_API_KEY"]`(없으면 `KeyError`). `DEFAULT_MODEL = "gpt-5.6-luna"`, `OPENAI_MODEL`/`LLM_MAX_CALLS_PER_RUN` 환경변수로 재정의 가능. |
| `client.py` | `OpenAIChartClient`, `LLMChartAnalystError` — **`openai`를 import하는 유일한 파일**. `read_pattern`이 `client.responses.parse(model=..., input=[...], text_format=ChartPatternRead, max_output_tokens=...)` 호출 후 `response.output_parsed` 반환. 좁은 예외부터(`APITimeoutError`→`APIConnectionError`→`RateLimitError`/`AuthenticationError`→`APIStatusError`→`pydantic.ValidationError`→`output_parsed is None`) 전부 `LLMChartAnalystError`로 통일(`TelegramNotificationError`와 동일 관례). `_calls_made` 카운터로 `max_calls_per_run` 초과 시 SDK를 건드리지 않고 즉시 raise(비용 안전밸브). 에러 메시지에 API 키를 절대 포함하지 않음. |
| `analyst.py` | `analyze(client, records) -> ChartAnalysis \| None` — `build_snapshot`이 `None`이면 리더를 호출하지 않고 `None` 반환. **`action` 파라미터가 없다**(동조 방어). `ticker`/`market`/`date`는 LLM 응답이 아니라 `records`에서 채움. `to_reasons(analysis, action)` — `ml_predictor.predictor._stance`와 동형인 5가지 동의/상충/중립 문구 + `caveat`(있을 때만) + 항상 마지막에 오는 `LLM_DISCLAIMER`. |

**배선 완료, 실제 API 호출은 `OPENAI_API_KEY` 발급 후 사용자가 직접 검증 필요**: `orchestrator/pipeline.py`에 `llm_client` 파라미터로 연결 완료(Phase 3, 아래 orchestrator 표 참고). `OPENAI_API_KEY`는 아직 미발급이라 실제 API 호출은 한 번도 없었다 — 모든 테스트가 `openai.OpenAI`를 모킹하고, `scripts/verify_llm_chart_analyst.py`도 이번 태스크에서는 dry-run 모드로만 실행했다.

---

## `scripts/` — 실행 진입점

| 파일 | 역할 |
|---|---|
| `run_recommendations.py` | MVP-0 end-to-end 실행(ML 없이). 예시 워치리스트(005930, 000660) 대상. |
| `run_recommendations_with_ml.py` | 위와 동일 + `load_model("KRX")`로 ML 보조신호 활성화. 아티팩트 없으면 안내 후 종료. |
| `run_recommendations_with_signals.py` | 위와 동일 + ML(`load_model`) **및** LLM(`load_llm_config`+`OpenAIChartClient`) 보조신호를 동시 활성화(Phase 3, 이번 태스크에 추가). 둘 중 하나라도 아티팩트/키가 없으면 안내 후 종료. **실제 OpenAI API를 호출한다(유료)** — 이번 태스크에서는 작성만 하고 실행하지 않았다. |
| `scan_nasdaq_top100_buy_only.py` | 나스닥 상위 100종목 BUY 신호만 스캔하는 변형 스크립트. |
| `train_ml_model.py` | KRX 상위 200종목·5년치로 logreg/rf/dummy 평가 리포트 출력 후 logreg만 저장. `ML_TRAIN_UNIVERSE_SIZE`/`ML_TRAIN_LOOKBACK_YEARS` 환경변수로 스모크 테스트 규모 조정 가능. |
| `verify_telegram.py` | 텔레그램 연동만 단독 확인(실제 메시지 1건 전송). |
| `verify_pykrx.py` | pykrx 로그인·유니버스 조회·시가총액 조회 단독 확인(이번 세션에 추가). |
| `verify_fdr_nasdaq.py` | FinanceDataReader의 나스닥 유니버스·OHLCV 조회 단독 확인(이번 세션에 추가). |
| `verify_llm_chart_analyst.py` | LLM 차트분석(#3) 배선 단독 확인(Phase 3, 이번 태스크에 추가). `LLM_VERIFY_DRY_RUN=1`이면 API 호출 없이 프롬프트만 출력(비용 0), 기본값은 실제 1회 호출. |

## `tests/` — 테스트 구조

총 188개, 전부 통과(커버리지 99%). `rule_engine`이 확립한 패턴을 전체가 따른다 — **순수 계산 함수는 수학적으로 자명한 극단 케이스로**, **의사결정/배선 로직은 계산 함수를 모킹해서** 독립적으로 검증. `ml_predictor` 쪽은 추가로 lookahead 방어(인과성 속성, embargo 갭, 전역 날짜 분할)를 자동 검증하는 테스트가 있고, 학습 로직은 실제 시장 데이터 대신 결정론적 합성 데이터셋(`conftest.py`)으로 검증해 단위테스트를 빠르고 재현 가능하게 유지한다. `llm_chart_analyst`(55개 신규)는 `conftest.py`의 autouse 픽스처로 `OPENAI_API_KEY`를 더미 값으로 monkeypatch해 실수로도 실제 API 호출이 불가능하게 하고, `client.py`는 `openai.OpenAI`를 완전히 모킹(`FakeChartPatternReader`는 `analyst.py` 테스트용으로 SDK 자체를 우회)한다.

디렉터리: `tests/data/`, `tests/rule_engine/`, `tests/risk_sizing/`, `tests/explainer/`, `tests/notifier/`, `tests/orchestrator/`, `tests/ml_predictor/`, `tests/llm_chart_analyst/` — `src/auto_stock/` 패키지 구조와 1:1 대응.

---

## 아직 안 된 것 (다음 후보)

- 주문 실행 에이전트(#8, MVP-1) — 브로커 모의투자 연동, 승인→주문 흐름
- LLM 차트분석(#3) — Phase 0~3(의존성/순수 계층/클라이언트·분석 계층/오케스트레이터 배선, 전부 모킹 기반) 완료. 실제 API 최초 호출은 `OPENAI_API_KEY` 발급 후 사용자가 직접 수행 필요(위 "진행 중" 참고). Phase 4(문서화 + `code-reviewer`/`security-reviewer` 리뷰)부터 남음.
- 뉴스/공시 분석(#4) 신호원 — 아직 계획 전
- ML 모듈의 나스닥 모델 확장(현재 `MARKET="KRX"` 하드코딩) — 실제 200종목 KRX 학습은 완료됨(`models/ml_predictor/KRX_h5_logreg.metadata.json`, AUC 0.528)
- `scheduler.py`(장중 폴링)를 실제 실행 경로에 연결
- 리스크 에이전트의 일일/월간 손실한도 자동중단(집행 로직, MVP-1과 함께)
