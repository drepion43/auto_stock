# `scripts/` — 실행 진입점 상세 리뷰

> **2026-10-14 갱신**: ML 관련 서술(§2/§3/§6, 환경변수 표, 실행 가이드)을 현재 코드 기준(Mamba 통합 단일 모델
> `models/ml_predictor/unified_h1_mamba.pt`, 인자 없는 `load_model()`, GPU 학습, `scripts/train_full.sh`)으로 고쳤다.

이 문서는 `scripts/` 디렉터리의 모든 실행 스크립트를 다룬다. 이 스크립트들은 전부 `src/auto_stock/` 패키지의 함수를 "배선"만 하는 얇은 진입점이며, 새로운 비즈니스 로직을 포함하지 않는다. 각 스크립트는 프로젝트 루트에서 실행해야 한다 — 상대경로로 `.env`와 `data/` 디렉터리를 참조하기 때문이다.

---

## 1~3. `run_recommendations.py` / `run_recommendations_with_ml.py` / `run_recommendations_with_signals.py` — **삭제됨 (2026-10-15)**

> **삭제 노트**: 이 세 스크립트는 전부 `orchestrator/pipeline.py::run_recommendation_pipeline`(텔레그램 알림을 실제로 발송하는 MVP-0 파이프라인)을 호출하는 수동 실행 진입점이었다 — 각각 ①보조신호 없음(규칙엔진+사이징만), ②ML 보조신호 추가, ③ML+LLM차트+공시+뉴스감성 네 신호 전부 추가 버전이었다. 메인 챗봇(deepagents 루프)이 쓰는 어떤 경로도 `run_recommendation_pipeline`을 호출하지 않아, 그 함수와 유일한 소비자였던 `notifier/`(텔레그램 발송 모듈) 전체를 코드에서 제거하면서 이 세 스크립트도 함께 삭제했다. 수동 알림 경로가 다시 필요해지면(MVP-1 재개 등) `docs/design/orchestrator.md`/`docs/design/telegram-notifier.md`의 설계 근거를 참고해 재구현하면 된다.

---

## 4. `run_market_scan.py`

**목적**: 국내·미국 4개 원자 거래소 전체 유니버스(기본 상위 200종목씩)를 스캔해 `MarketScanCache`(`data/market_scan.duckdb`)에 캐시 적재하는 배치 스크립트. 챗봇의 "추천해줄만한 주식 있어?" 같은 전체 스캔형 질의에 즉답하기 위한 사전 준비 작업이다(`run_market_scan.py:1-6`).

**실행 방법**:
```
.venv/Scripts/python scripts/run_market_scan.py   (Windows)
.venv/bin/python scripts/run_market_scan.py        (macOS/Linux)

MARKET_SCAN_UNIVERSE_SIZE=20 .venv/Scripts/python scripts/run_market_scan.py   (소규모 테스트)
```
필수 환경변수 없음(텔레그램도, OpenAI도 쓰지 않음 — 규칙엔진+사이징만 사용, `run_market_scan.py:14`).
선택 환경변수: `MARKET_SCAN_UNIVERSE_SIZE`(기본 200), `MARKET_SCAN_STALE_HOURS`(기본값은 `DEFAULT_STALE_AFTER`를 시간 단위로 환산), `ACCOUNT_EQUITY`.

**동작 흐름**: `MARKETS = ["KOSPI", "KOSDAQ", "NASDAQ", "NYSE"]`를 순회하며 각 시장마다 `_fresh_universe(scan_cache, market, SCAN_UNIVERSE_SIZE, STALE_AFTER)`로 유니버스를 가져오고, `scan_market(cache, tickers, market, account)`로 규칙엔진+사이징만 돌려 후보를 찾은 뒤 `scan_cache.put_scan(market, explanations)`로 캐시에 저장한다(`run_market_scan.py:46,59-70`).

> **market 세분화(2026-10)**: `MARKETS`가 `["KRX", "NASDAQ"]`(2개)에서 `["KOSPI", "KOSDAQ", "NASDAQ", "NYSE"]`(원자 거래소 4개)로 확장됐다 — `data/models.py::VALID_MARKETS`가 원자값만 허용하도록 바뀌면서, 이 레거시 무료경로(`ScanCoordinator`/`scan_market`, LLM 미사용)는 "KRX"/"US" 같은 합성 버킷을 아예 모른다. 합성 버킷(코스피+코스닥 통합, 나스닥+NYSE 통합, S&P500, 다우존스) 스캔은 이 스크립트가 아니라 챗봇이 쓰는 유료 경로(`RecommendationCoordinator`, 아래 "`chat_cli.py`"/`chat_agent/` 리뷰 문서 참고)에서만 지원된다 — `orchestrator/market_buckets.py::ALL_SCAN_BUCKETS`에 전체 8개 버킷 목록이 있다.

**API 호출**: 없음(완전 무료) — 텔레그램 전송도 하지 않는다(`run_market_scan.py:34`).

**중요한 운영 주의사항**: docstring에 두 가지 강한 경고가 있다.
1. **OS 스케줄러로 이 스크립트를 등록하지 말 것** — 정상 운영에서는 `chat_cli.py`가 시작 시점/질의 시점에 `ScanCoordinator.ensure_fresh`로 스캔을 직접 트리거하므로, 이 스크립트는 챗봇 없이 캐시만 미리 데우고 싶을 때의 수동 fallback일 뿐이다(`run_market_scan.py:22-25`).
2. **`chat_cli.py`가 실행 중인 동안 같이 실행하지 말 것** — 둘 다 같은 DuckDB 파일(`data/ohlcv.duckdb`, `data/market_scan.duckdb`)에 read-write 커넥션을 열려 하므로, 두 번째로 여는 쪽이 `duckdb.IOException`을 던진다(DuckDB는 단일 프로세스만 파일을 열 수 있음)(`run_market_scan.py:26-30`).

---

## 5. `scan_nasdaq_top100_buy_only.py` — **삭제됨 (2026-10-15)**

> **삭제 노트**: NASDAQ 상위 100종목을 스캔하되 BUY 신호만 텔레그램으로 전송하던 변형 스크립트. `send_notification`(텔레그램 발송)을 직접 호출하는 유일한 소비자 중 하나였고, 메인 챗봇이 쓰지 않는 알림 경로라 `notifier/` 삭제와 함께 제거했다.

---

## 6. `train_ml_model.py` / `collect_ml_dataset.py` / `train_full.sh`

**목적**: ML 예측 모델(#2)을 **KRX(코스피+코스닥)+NASDAQ+NYSE 전체를 합친 통합 Mamba 모델 1개**로 학습해 저장한다. 모델 상세는 `03_ml_predictor.md`.

> **갱신 (2026-10-15)**: 데이터 수집(OHLCV+펀더멘털 → `SequenceDataset`)과 학습을 분리할 수 있게 됐다. 공유 로직은 신규 모듈 `ml_predictor/collection.py`(`universe_for`/`atomic_market_for`/`collect_universe_records`/`save_dataset`/`load_dataset`)로 추출됐다 — 과거 `train_ml_model.py`에 있던 `_universe_for`/`_atomic_market_for`/`_collect_universe_records` 사설 헬퍼는 이 모듈로 옮겨졌다(이름에서 밑줄 제거, 공개 함수로 전환).

**실행 방법**:
```
.venv/Scripts/python scripts/train_ml_model.py   (Windows, 수집+학습 한 번에 — 기존과 동일)
.venv/bin/python scripts/train_ml_model.py        (macOS/Linux)

.venv/Scripts/python scripts/collect_ml_dataset.py   (수집만, 학습 안 함 — 신규)
.venv/Scripts/python scripts/train_ml_model.py        (수집된 파일이 있으면 재수집 없이 바로 학습)

ML_TRAIN_DEVICE=cuda ML_TRAIN_UNIVERSE_SIZE=5 ML_TRAIN_LOOKBACK_YEARS=1 ML_TRAIN_EPOCHS=5 \
  .venv/Scripts/python scripts/train_ml_model.py   (스모크 테스트)

bash scripts/train_full.sh                        (전체 학습 + 로그 파일, 아래)
```
필수 환경변수 없음(국내 펀더멘털의 매출액은 `DART_API_KEY`가 있으면 채워지고 없으면 그 필드만 빠짐).
선택 환경변수: `ML_TRAIN_UNIVERSE_SIZE`(시장별, 기본 200), `ML_TRAIN_LOOKBACK_YEARS`(5), `ML_TRAIN_EPOCHS`(100), `ML_TRAIN_BATCH_SIZE`(32), `ML_TRAIN_LR`(0.01), `ML_TRAIN_HIDDEN_DIM`(16), `ML_TRAIN_N_LAYERS`(2), `ML_TRAIN_DEVICE`(auto|cuda|cpu, 기본 auto), `ML_TRAIN_LOG_EVERY`(1, 0이면 epoch 로그 끔), `ML_TRAIN_VAL_RATIO`(0.1), **신규** `ML_TRAIN_DATASET_PATH`(기본 `data/ml_training_dataset.pkl`, 두 스크립트가 공유), **신규** `ML_TRAIN_FORCE_COLLECT`(`train_ml_model.py`에서 `1`이면 저장된 파일이 있어도 무시하고 재수집).

**`collect_ml_dataset.py` 동작 흐름**: ① `UNIVERSE_MARKETS = ["KRX", "NASDAQ", "NYSE"]`마다 `universe_for(market)[:UNIVERSE_SIZE]`로 종목을 고르고(KRX는 KOSPI+KOSDAQ 합집합, 목록 앞에서부터 자름 — 시가총액 순인지는 미검증) `collect_universe_records`가 OHLCV와 국내 펀더멘털을 수집(종목 단위 예외 격리, tqdm 진행바) → ② `build_sequence_dataset`으로 조립 → ③ `save_dataset`으로 `pickle`에 저장(`SequenceDataset`은 frozen dataclass + 표준 타입만 담아 pickle로 안전하게 왕복, 신뢰 경계 — 이 파일은 로컬에서만 생성·소비됨). 학습은 전혀 하지 않는다.

**`train_ml_model.py` 동작 흐름**: ① `resolve_device`로 device 확인(`cuda`를 명시했는데 GPU가 없으면 **수집/로드 전에** 실패) → ② `ML_TRAIN_DATASET_PATH`에 파일이 있고 `ML_TRAIN_FORCE_COLLECT`가 아니면 `load_dataset`으로 즉시 로드, 없으면 `collect_ml_dataset.py`와 동일한 수집 로직을 돌리고 끝에 `save_dataset`으로 저장(다음 실행부터 재사용) → ③ `train_val_test_split`(시간순 ~70/10/20, embargo) → ④ 구간별 기간·종목 수, 하이퍼파라미터 출력 → ⑤ `train`(epoch마다 train/val loss·lr 로그) → ⑥ `_print_evaluation_report`(validation/test 전체, test 국내/미국별로 Mamba vs persistence 베이스라인 MSE/MAE/R²/방향적중률) → ⑦ `save_model` + `unified_h1_mamba.history.{csv,png}`(학습 곡선) 저장 → val 최저 epoch가 마지막보다 앞이면 재학습 안내(저장 가중치는 마지막 epoch).

**API 호출**: 없음(완전 무료) — 시세·펀더멘털 수집(FDR/pykrx/DART)과 로컬 학습뿐. 수집이 전체 시간의 대부분을 차지한다(OHLCV는 `data/ohlcv.duckdb`에 캐시, pykrx 펀더멘털은 캐시 없음) — 그래서 분리해두면 하이퍼파라미터를 바꿔가며 재학습할 때마다 이 시간을 반복할 필요가 없다.

**종료/실패 조건**: 개별 종목 수집 실패는 스킵하고 계속. train/validation/test 중 하나라도 비면 "표본이 부족합니다" 출력 후 저장 없이 종료(기존 모델 파일은 그대로). `collect_ml_dataset.py`는 수집 결과가 통째로 비면 저장하지 않고 안내만 출력.

**`data/ml_training_dataset.pkl`은 git에 커밋하지 않는다**(`.gitignore`의 `data/*.pkl`) — `data/*.duckdb`와 같은 로컬 캐시 취급.

**`train_full.sh`**: 전체 학습 래퍼. 프로젝트 루트로 이동 → `.venv` 파이썬 탐색(Windows/Unix) → 기본 `ML_TRAIN_DEVICE=cuda`, `ML_TRAIN_LOG_EVERY=5`, `PYTHONUTF8=1`(파이프 출력 한글 깨짐 방지) → `logs/ml_training/train_full_<YYYYmmdd_HHMM>.log`로 `tee`. `set -euo pipefail`이라 학습이 실패하면 스크립트도 비0으로 끝난다. `ML_TRAIN_*`는 앞에 붙여 덮어쓸 수 있다(`ML_TRAIN_EPOCHS=40 bash scripts/train_full.sh`). `logs/`는 gitignore. 이 스크립트는 `train_ml_model.py`를 그대로 호출하므로, 미리 `collect_ml_dataset.py`로 수집해뒀으면 재수집 없이 바로 학습부터 시작한다.

---

## 7. `verify_telegram.py` — **삭제됨 (2026-10-15)**

> **삭제 노트**: 텔레그램 봇 연동만 단독으로 확인하던 수동 검증 스크립트(`send_notification` 호출). `notifier/` 모듈 전체가 메인 챗봇 미사용으로 삭제되면서 함께 제거했다.

---

## 8. `verify_pykrx.py`

**목적**: `pykrx` 라이브러리가 `.env`의 `KRX_ID`/`KRX_PW`로 로그인해 실데이터를 가져올 수 있는지 확인. `pykrx_source.py`가 `load_dotenv()`를 함수 내부에서 호출하도록 고친 수정사항을 검증하기 위한 스크립트다(`verify_pykrx.py:1-16`).

**실행 방법**: `.venv/Scripts/python scripts/verify_pykrx.py` (Windows) / `.venv/bin/python scripts/verify_pykrx.py` (macOS/Linux).
필수 환경변수: `KRX_ID`, `KRX_PW`(단, **`.env.example`에는 이 두 키가 포함되어 있지 않다** — 아래 "환경변수 전체 목록" 섹션 참고).

**동작 흐름**: `get_ticker_list(today, market="ALL")`로 전체 유니버스 조회 후 `get_market_cap("005930", today)`로 삼성전자 시가총액 조회(`verify_pykrx.py:25-37`).

**API 호출**: KRX 로그인 기반 조회(무료, 로그인 필요).

**종료/실패 조건**: 유니버스가 빈 리스트면 `FAILED`(로그인 실패 또는 휴장일 가능성 안내), 시가총액이 `None`이면 `FAILED`(휴장일 가능성), 둘 다 성공하면 각각 `SUCCESS` 출력. 전체를 `try/except Exception`으로 감싸 예외 메시지를 그대로 출력(`verify_pykrx.py:25-39`). docstring은 pykrx 라이브러리 자신이 로그인 ID(비밀번호는 아님)를 stdout에 출력하는 것은 이 스크립트의 책임이 아니라고 명시한다(`verify_pykrx.py:13-15`).

---

## 9. `verify_fdr_nasdaq.py`

**목적**: `FinanceDataReader`가 NASDAQ 유니버스 목록과 OHLCV를 가져올 수 있는지 확인. NASDAQ은 pykrx 대응재가 없어 전부 FDR을 통해 조회되므로, 이 스크립트가 `verify_pykrx.py`의 NASDAQ 측 대응품이다(`verify_fdr_nasdaq.py:1-10`).

**실행 방법**: `.venv/Scripts/python scripts/verify_fdr_nasdaq.py` (Windows) / `.venv/bin/python scripts/verify_fdr_nasdaq.py` (macOS/Linux).
필수 환경변수 없음 — API 키나 로그인 불필요(`verify_fdr_nasdaq.py:10`).

**동작 흐름**: `get_universe("NASDAQ")`로 유니버스 조회 후 `fetch_ohlcv("AAPL", start, end, "NASDAQ")`로 최근 30일 OHLCV 조회(`verify_fdr_nasdaq.py:21-29`).

**API 호출**: FDR 조회(무료, 키 불필요).

**종료/실패 조건**: 유니버스가 빈 리스트거나 OHLCV가 빈 리스트면 각각 `FAILED` 출력, 둘 다 성공하면 `SUCCESS`(최근 종가 포함). 전체를 `try/except Exception`으로 감쌈(`verify_fdr_nasdaq.py:20-36`).

---

## 10. `verify_llm_chart_analyst.py`

**목적**: LLM 차트분석 에이전트(#3)의 배선을 end-to-end로 검증. 두 가지 모드(dry-run/실호출)를 지원하는 첫 번째 스크립트다(`verify_llm_chart_analyst.py:1-22`).

**실행 방법**:
```
.venv/Scripts/python scripts/verify_llm_chart_analyst.py                          (Windows, 실제 API 1회 호출)
LLM_VERIFY_DRY_RUN=1 .venv/Scripts/python scripts/verify_llm_chart_analyst.py      (dry-run, 비용 0)
.venv/bin/python scripts/verify_llm_chart_analyst.py                               (macOS/Linux, 실제 API 1회 호출)
```
필수 환경변수(dry-run 아닐 때만): `OPENAI_API_KEY`.
선택 환경변수: `LLM_VERIFY_DRY_RUN`(값이 `"1"`일 때만 dry-run).

**동작 흐름**:
- **dry-run** (`_run_dry_run`, `verify_llm_chart_analyst.py:48-61`): 삼성전자(005930) OHLCV를 가져와 `build_snapshot`으로 `ChartSnapshot`을 만들고, `SYSTEM_PROMPT`와 `build_user_prompt(snapshot)` 결과를 그대로 출력한다. OpenAI 클라이언트 자체를 만들지 않으므로 네트워크 호출이 전혀 없다.
- **실호출** (`_run_real_call`, `verify_llm_chart_analyst.py:64-96`): `load_llm_config()`로 설정을 불러온 뒤 `dataclasses.replace(config, max_calls_per_run=1)`로 호출 예산을 강제로 1로 깎는다(사용자의 `LLM_MAX_CALLS_PER_RUN` 설정과 무관하게, 실수로 여러 번 호출되는 사고를 방지). `OpenAIChartClient(config)`를 만들어 `analyze(client, records)` 1회 실행 후 `direction`/`confidence`/`pattern_name`/`rationale`/`caveat`를 출력.

**API 호출**: dry-run은 완전 무료. 실호출 모드는 OpenAI API 1회 호출(유료, 단 `max_calls_per_run=1`로 강제되어 추가 호출 위험 없음).

**종료/실패 조건**: `build_snapshot`이 `None`을 반환하면(SMA60 워밍업 기간 미충족) dry-run/실호출 모두 `FAILED` 출력 후 `sys.exit(1)`(`verify_llm_chart_analyst.py:51-53, 86-88`). 실호출에서 `OPENAI_API_KEY` 누락이면 `KeyError`를 잡아 `FAILED` 출력 후 종료(`verify_llm_chart_analyst.py:66-69`). `LLMChartAnalystError` 발생 시에도 동일하게 `FAILED` 출력 후 종료(`verify_llm_chart_analyst.py:82-84`).

> **market 세분화(2026-10)**: `SAMPLE_MARKET = "KRX"`가 `SAMPLE_MARKET = "KOSPI"`로 바뀌었다(`verify_llm_chart_analyst.py:38`) — "KRX"는 더 이상 `get_ohlcv`가 받는 원자 market 값이 아니다. 샘플 종목(005930, 삼성전자)은 그대로 코스피이므로 다른 로직 변경은 없다.

---

## 11. `verify_news_disclosure.py`

**목적**: 뉴스/공시 분석 에이전트(#4)의 배선을 국내(DART)와 미국(EDGAR) 양쪽 모두에 대해 검증. 시장 선택과 dry-run 여부라는 **두 개의 독립적인 축**을 환경변수로 조합할 수 있다(`verify_news_disclosure.py:1-27`).

**실행 방법**:
```
.venv/Scripts/python scripts/verify_news_disclosure.py                                   (Windows, KOSPI/DART 실호출, 기본값)
NEWS_VERIFY_MARKET=KOSDAQ .venv/Scripts/python scripts/verify_news_disclosure.py          (KOSDAQ/DART 실호출)
NEWS_VERIFY_MARKET=NASDAQ .venv/Scripts/python scripts/verify_news_disclosure.py          (NASDAQ/EDGAR 실호출)
NEWS_VERIFY_MARKET=NYSE .venv/Scripts/python scripts/verify_news_disclosure.py            (NYSE/EDGAR 실호출)
NEWS_VERIFY_DRY_RUN=1 .venv/Scripts/python scripts/verify_news_disclosure.py              (dry-run, DART/EDGAR만 호출)
.venv/bin/python scripts/verify_news_disclosure.py                                        (macOS/Linux, KOSPI/DART 실호출)
```
필수 환경변수:
- KOSPI/KOSDAQ 모드(기본값 KOSPI)에서 실호출: `DART_API_KEY` + `OPENAI_API_KEY`
- NASDAQ/NYSE 모드에서 실호출: `SEC_EDGAR_USER_AGENT` + `OPENAI_API_KEY`
- dry-run 모드는 각 시장의 공시 소스 키만 필요(OpenAI 불필요)

선택 환경변수: `NEWS_VERIFY_MARKET`(`KOSPI` 기본값, 또는 `KOSDAQ`/`NASDAQ`/`NYSE`), `NEWS_VERIFY_DRY_RUN`(`"1"`이면 dry-run).

**동작 흐름**: `_resolve_sample_market()`으로 대상 시장 결정(`verify_news_disclosure.py:55-60`) → `_fetch_sample_disclosures(market)`가 `market in KRX_MARKETS`(신규 공용 모듈 `data/market_routing.py`의 멤버십 체크)로 분기해 `resolve_corp_code`+`fetch_disclosures`(KOSPI/KOSDAQ·DART) 또는 `resolve_cik`+`fetch_filings`(NASDAQ/NYSE·EDGAR)를 호출(`verify_news_disclosure.py:63-75`) → dry-run이면 `DisclosureSummary`를 만들어 `build_user_prompt`로 렌더링한 결과만 출력(`verify_news_disclosure.py:78-96`) → 실호출이면 `load_llm_config()` → `max_calls_per_run=1` 강제 → `OpenAIDisclosureClient` → `analyze()` 실행 후 `market_impact`/`confidence`/`key_event`/`rationale`/`caveat` 출력(`verify_news_disclosure.py:99-130`).

**API 호출**: dry-run은 DART/EDGAR 조회만 발생(둘 다 무료). 실호출은 추가로 OpenAI API 1회(유료, 예산 강제로 1회 제한).

**종료/실패 조건**: 샘플 종목에 corp_code/cik를 찾지 못하면 `FAILED` 후 종료(`verify_news_disclosure.py:66-70`). **샘플 종목에 최근 공시가 없으면(`analyze()`가 `None` 반환) 이것은 실패가 아니라 정상 흐름으로 안내된다** — "공시 없음 — 정상 흐름" 메시지만 출력하고 종료코드 0으로 끝난다(`verify_news_disclosure.py:83-85, 120-122`). 크리덴셜 누락은 `KeyError`를 최상위에서 잡아 어떤 환경변수가 빠졌는지 안내(`verify_news_disclosure.py:142-145`, `_MISSING_CREDENTIAL_ENV_VAR` 매핑 사용).

> **market 세분화(2026-10)**: 이 스크립트는 네 시장 체계로 꽤 많이 바뀌었다.
> - `_SAMPLE_TICKERS`가 `{"KRX": "005930", "NASDAQ": "AAPL"}`(2개)에서 `{"KOSPI": "005930", "KOSDAQ": "247540", "NASDAQ": "AAPL", "NYSE": "JPM"}`(4개 시장 각각 샘플 종목)로 확장됐다(`verify_news_disclosure.py:48`).
> - `_MISSING_CREDENTIAL_ENV_VAR`도 4개 키로 확장됐다 — KOSPI/KOSDAQ 둘 다 `DART_API_KEY`, NASDAQ/NYSE 둘 다 `SEC_EDGAR_USER_AGENT`를 가리킨다(`verify_news_disclosure.py:49-52`).
> - `NEWS_VERIFY_MARKET`의 기본값이 `"KRX"`에서 `"KOSPI"`로 바뀌었다(`verify_news_disclosure.py:56`) — "KRX"는 더 이상 받을 수 있는 값이 아니다.
> - 분기 로직(`if market == "KRX": DART else: EDGAR`)이 신규 공용 모듈 `data/market_routing.py`의 `KRX_MARKETS`(멤버십 체크)/`NEWS_SOURCE_LABELS`(DART/EDGAR 라벨)를 쓰도록 바뀌었다(`verify_news_disclosure.py:34,65,68,79`) — `pipeline.py`/`tools.py`가 과거 각자 중복 정의했던 라우팅 딕셔너리를 이 스크립트도 함께 재사용한다.
> - 모듈 docstring도 "DART(KRX)와 EDGAR(NASDAQ)"에서 "DART(코스피/코스닥)와 EDGAR(나스닥/NYSE)"로 갱신됐다(`verify_news_disclosure.py:1-2,12-15`).

---

## 12. `verify_news_sentiment.py`

**목적**: 뉴스 감성분석 에이전트(#4-뉴스)의 배선을 국내(네이버)와 미국(GDELT) 양쪽 모두에 대해 검증. `verify_news_disclosure.py`와 완전히 동일한 구조(시장 축 + dry-run 축)를 따른다(`verify_news_sentiment.py:1-30`).

**실행 방법**:
```
.venv/Scripts/python scripts/verify_news_sentiment.py                                           (Windows, KOSPI/네이버 실호출, 기본값)
NEWS_SENTIMENT_VERIFY_MARKET=KOSDAQ .venv/Scripts/python scripts/verify_news_sentiment.py        (KOSDAQ/네이버 실호출)
NEWS_SENTIMENT_VERIFY_MARKET=NASDAQ .venv/Scripts/python scripts/verify_news_sentiment.py        (NASDAQ/GDELT 실호출)
NEWS_SENTIMENT_VERIFY_MARKET=NYSE .venv/Scripts/python scripts/verify_news_sentiment.py          (NYSE/GDELT 실호출)
NEWS_SENTIMENT_VERIFY_DRY_RUN=1 .venv/Scripts/python scripts/verify_news_sentiment.py            (dry-run, 뉴스 소스만)
.venv/bin/python scripts/verify_news_sentiment.py                                                (macOS/Linux, KOSPI/네이버 실호출)
```
필수 환경변수:
- KOSPI/KOSDAQ 모드(기본값 KOSPI)에서 실호출: `NAVER_CLIENT_ID` + `NAVER_CLIENT_SECRET` + `OPENAI_API_KEY`
- NASDAQ/NYSE 모드에서 실호출: `OPENAI_API_KEY`만(GDELT는 크리덴셜 자체가 없음)
- dry-run 모드는 OpenAI 불필요

선택 환경변수: `NEWS_SENTIMENT_VERIFY_MARKET`(`KOSPI` 기본값, 또는 `KOSDAQ`/`NASDAQ`/`NYSE`), `NEWS_SENTIMENT_VERIFY_DRY_RUN`(`"1"`이면 dry-run).

**동작 흐름**: `_fetch_sample_articles(market)`가 `market in KRX_MARKETS`(`data/market_routing.py` 멤버십 체크)로 분기해 KOSPI/KOSDAQ이면 `resolve_corp_name` + `search_news`(네이버), NASDAQ/NYSE면 `resolve_company_title` + `search_articles`(GDELT, 회사명을 쌍따옴표로 감싸 질의)를 호출(`verify_news_sentiment.py:37,62-78`). 이하 흐름은 공시 버전과 동일 — dry-run은 프롬프트만 출력, 실호출은 `max_calls_per_run=1`로 강제한 `OpenAINewsSentimentClient`로 `analyze()` 실행 후 `sentiment`/`confidence`/`key_headline`/`rationale`/`caveat` 출력(`verify_news_sentiment.py:96-126`).

**API 호출**: dry-run은 네이버/GDELT 조회만(둘 다 무료). 실호출은 추가로 OpenAI API 1회(유료, 1회 제한).

**종료/실패 조건**: 회사명을 DART/EDGAR 캐시에서 못 찾으면 `FAILED` 후 종료(`verify_news_sentiment.py:66-69`). 최근 기사가 없으면 실패가 아니라 정상 흐름으로 안내(`verify_news_sentiment.py:85-87, 116-118`). 크리덴셜 누락은 `KeyError`로 안내(`verify_news_sentiment.py:138-140`).

**주의사항(docstring에 명시)**: GDELT의 `seendate` 필드 형식은 네트워크 제약으로 라이브 검증이 안 된 상태다 — NASDAQ/NYSE 모드를 처음 실행하기 전에 `gdelt_source.py`의 `_SEENDATE_FORMAT` 가정이 맞는지 먼저 짚어보라는 경고가 있다(`verify_news_sentiment.py:21-23`).

> **갱신 (2026-10-12)**: `seendate` 형식은 라이브 호출로 확인됐다. 검색어는 이제 `company_phrase(name)`(법인 접미사 제거)로 만든다. GDELT 요청 제한(HTTP 429) 때문에 NASDAQ/NYSE 모드는 실패하거나 수십 초 걸릴 수 있다 — `docs/design/us-news-source-reliability.md`.

> **market 세분화(2026-10)**: `_SAMPLE_TICKERS`가 `{"KRX": "005930", "NASDAQ": "AAPL"}`(2개)에서 `{"KOSPI": "005930", "KOSDAQ": "247540", "NASDAQ": "AAPL", "NYSE": "JPM"}`(4개 시장)로 확장됐다(`verify_news_sentiment.py:51`). `NEWS_SENTIMENT_VERIFY_MARKET` 기본값이 `"KRX"`에서 `"KOSPI"`로 바뀌었다(`verify_news_sentiment.py:55`). 분기 로직이 `data/market_routing.py`의 `KRX_MARKETS`/`SENTIMENT_SOURCE_LABELS`를 쓰도록 바뀌었다(`verify_news_sentiment.py:37,64,82`). 모듈 docstring도 "네이버(KRX)와 GDELT(NASDAQ)"에서 "네이버(코스피/코스닥)와 GDELT(나스닥/NYSE)"로 갱신됐다(`verify_news_sentiment.py:1-2,12-15`).

---

## 13. `verify_sources_individually.py`

**목적**: 위 두 `verify_news_*` 스크립트가 "시장 분기 + LLM 해석"까지 포함한 end-to-end 검증인 것과 달리, 이 스크립트는 **데이터 수집 계층(#0) 소스 각각의 연결성만** 독립적으로 확인한다. OpenAI는 전혀 호출하지 않는다(`verify_sources_individually.py:1-7`).

**실행 방법**:
```
.venv/Scripts/python.exe scripts/verify_sources_individually.py dart
.venv/Scripts/python.exe scripts/verify_sources_individually.py edgar
.venv/Scripts/python.exe scripts/verify_sources_individually.py naver
.venv/Scripts/python.exe scripts/verify_sources_individually.py gdelt
.venv/Scripts/python.exe scripts/verify_sources_individually.py all   (4개 순서대로, 하나 실패해도 계속)
```
필수 환경변수: `dart` 타깃은 `DART_API_KEY`, `edgar` 타깃은 `SEC_EDGAR_USER_AGENT`, `naver` 타깃은 `NAVER_CLIENT_ID`+`NAVER_CLIENT_SECRET`. `gdelt`는 크리덴셜 불필요.

**동작 흐름**: `_TESTS = {"dart": test_dart, "edgar": test_edgar, "naver": test_naver, "gdelt": test_gdelt}`에서 `sys.argv[1]`(기본 `"all"`)로 고른 함수(들)를 실행. 각 `test_*`는 해당 소스 모듈의 핵심 함수를 호출해 결과를 최대 5건 출력한다(예: `test_dart`는 `resolve_corp_code`/`resolve_corp_name`/`fetch_disclosures`). **ETF 타깃은 없다** — ETF holdings/목록 조회는 2026-10-11 일회성 스크립트로 라이브 검증했다(`01_data_layer.md`의 `etf_source.py` 절). 필요하면 `fetch_etf_listing()` 건수 + `fetch_etf_holdings("069500", top_n=5)` 출력 정도로 새로 추가해야 한다.

**API 호출**: 전부 무료(DART/EDGAR/네이버/GDELT 데이터 수집만, OpenAI 없음).

**종료/실패 조건**: 각 타깃이 해당 소스의 `*ApiError`(`DartApiError`, `EdgarApiError`, `NaverNewsApiError`, `GdeltApiError`)와 `KeyError`(크리덴셜 누락)를 잡아 `FAILED` 출력 후 리턴 — 다른 소스 테스트에 영향 없음. `all`이면 하나가 실패해도 나머지는 계속.

> 코드에 작은 흔적이 하나 있다: `_TESTS` 딕셔너리 선언 바로 위에 주석 처리된 동일한 줄이 남아 있다(`verify_sources_individually.py:130` `# _TESTS = {...}`) — 공백 포맷(`"gdelt":test_gdelt`, 콜론 뒤 공백 없음)만 다른 중복 줄로, 기능에는 영향 없는 사소한 코드 잔재다.

> **market 세분화(2026-10)**: `test_naver()` 함수 안의 `search_news("삼성전자", _SAMPLE_TICKER_KRX, "KRX", start, end)` 호출이 `search_news("삼성전자", _SAMPLE_TICKER_KRX, "KOSPI", start, end)`로 바뀌었다(`verify_sources_individually.py:107`). 이 호출의 `market` 인자는 `NewsArticle.market`에 라벨로만 쓰이는 자유문자열(검증 없음)이라 실질적 동작 변화는 없지만, 다른 스크립트들과의 일관성을 위해 바뀌었다. `_SAMPLE_TICKER_KRX` 변수명 자체와 `test_dart()`/`test_edgar()`/`test_gdelt()`의 섹션 제목(`=== DART (KRX 공시) ===` 등)은 변경되지 않았다 — 이 스크립트는 "KRX"를 원자 market 값으로 실제 전달하는 호출이 1건뿐이라 그 1건만 교체됐다.

---

## 14. `chat_cli.py` — CLI 진입점만 (내부 로직은 `chat_agent/` 모듈 리뷰 참고)

**목적**: 대화형 챗봇 에이전트의 터미널 진입점. 기본은 deepagents 기반 `chat_agent/deep_loop.py::run_turn_deep`을, `CHAT_AGENT_LOOP=legacy`면 구 루프 `loop.run_turn`을 stdin 루프로 감싼다.

> 루프 선택(2026-10-12~): 시작 시 트레이싱 차단 검사(`check_tracing_allowed`)와 노출 도구 검사(`verify_exposure`, fail-closed)를 통과해야 deep으로 실행되고, 실패하면 경고 후 구 루프로 실행한다. `CHAT_AGENT_LOOP=legacy`면 처음부터 구 루프(`loop.run_turn`). 시작 메시지에 `(루프: deep|legacy)`가 표시되고, 출력 형식은 두 루프가 같다. deep 모드에서는 턴당 새 백그라운드 추천 배치가 1개로 제한된다(`batch_trigger_limit`).

**실행 방법**:
```
.venv/Scripts/python scripts/chat_cli.py   (Windows)
.venv/bin/python scripts/chat_cli.py        (macOS/Linux)
```
필수 환경변수: `OPENAI_API_KEY`(없으면 시작 시점에 `FAILED` 출력 후 `sys.exit(1)`, `chat_cli.py:139-143`).
선택 환경변수: ML 아티팩트/`OPENAI_API_KEY` 외 보조 신호 크리덴셜(`DART_API_KEY`, `SEC_EDGAR_USER_AGENT`, `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`)이 없어도 **스크립트가 죽지 않는다** — 해당 tool만 `available=False`로 저하되고 나머지 tool로 대화를 계속할 수 있다(배치 스크립트들과 다른 "대화형 우아한 저하" 원칙, `chat_cli.py:1-5`). 추가로 `ACCOUNT_EQUITY`, `RECOMMENDATION_STALE_HOURS`(기본 24시간), `MARKET_SCAN_UNIVERSE_SIZE`(기본 200)도 선택적으로 읽는다(`chat_cli.py:152-168`).

**종료 조건**: 사용자가 `exit`, `quit`, `종료` 중 하나를 입력하면 루프 종료(`chat_cli.py:75, 182-183`). `EOFError`(입력 스트림 종료, 예: 파이프 입력이 끝났을 때)도 루프를 정상적으로 끝낸다(`chat_cli.py:178-179`).

**API 호출**: **실제 OpenAI API를 호출한다(유료)** — 대화 턴마다 최소 1회 이상 호출. 2026-10-11~12 `eval_main_agent.py`(아래 §15)로 legacy/deep 두 루프를 실제 API로 돌려 비교했다.

**Windows 콘솔 인코딩 보정**: 실행 초반에 `sys.stdin`/`sys.stdout`/`sys.stderr`를 전부 `utf-8`로 강제 재설정한다(`chat_cli.py:46-48`) — Windows 콘솔 기본 코드페이지(cp949)가 한글/이모지/em-dash를 인코딩하지 못해 `UnicodeEncodeError`로 죽는 문제, 그리고 stdin을 안 바꾸면 한글 질문이 cp949로 잘못 디코딩되어 API 요청 직렬화 단계에서 예외가 나고 그게 넓은 except에 잡혀 원인 불명의 "일시적 오류"로만 보이는 문제를 실제로 겪고 고친 기록이 docstring에 남아 있다(`chat_cli.py:40-45`).

내부 tool 호출/reasoning 로직, 16개 tool의 세부 동작, `ChatToolContext` 구성 등은 `chat_agent/` 모듈 자체의 리뷰 문서에서 다룬다(이 스크립트는 그 모듈을 호출만 하는 wrapper).

---

## `.env.example` 환경변수 전체 목록

`.env.example`에 정의된 변수:

| 변수 | 필수/선택 | 어느 모듈이 쓰는가 | 비고 |
|---|---|---|---|
| `OPENAI_API_KEY` | 필수(LLM 보조 신호·챗봇 사용 시) | `llm_chart_analyst`, `news_disclosure`, `news_sentiment`, `chat_agent` 공통 | 4개 모듈이 모두 같은 키를 공유 재사용 |
| `OPENAI_MODEL` | 선택 | `llm_chart_analyst/credentials.py` | 기본값 `gpt-5.6-luna` |
| `LLM_MAX_CALLS_PER_RUN` | 선택 | `llm_chart_analyst/credentials.py` | 기본값 20 |
| `DART_API_KEY` | 필수(KRX 공시 분석 시) | `data/sources/dart_source.py` | 비밀값 |
| `SEC_EDGAR_USER_AGENT` | 필수(NASDAQ 공시 분석 시) | `data/sources/edgar_source.py` | **비밀값 아님** — SEC 정책상 요구되는 연락처 포함 식별 문자열. 예: `"auto_stock your-email@example.com"` |
| `NAVER_CLIENT_ID` | 필수(KRX 뉴스 감성분석 시) | `data/sources/naver_news_source.py` | 비밀값 |
| `NAVER_CLIENT_SECRET` | 필수(KRX 뉴스 감성분석 시) | `data/sources/naver_news_source.py` | 비밀값 |
| `NEWS_SENTIMENT_OPENAI_MODEL` | 선택 | `news_sentiment/credentials.py` | 공시 해석(#4)과 별도 모델 등급 지정용, 기본값은 `OPENAI_MODEL`과 동일 |

`.env.example` 파일 자체는 market 세분화 작업(2026-10)에서 수정되지 않았다 — 위 표의 환경변수 목록·값은 실제 파일을 그대로 반영한 것이며 변경 없음. market 세분화로 새로 생긴 코드(`data/market_routing.py`, `orchestrator/market_buckets.py`, `data/sources/index_quote.py`, `data/sources/index_constituents.py`)는 전부 기존 크리덴셜(DART_API_KEY/SEC_EDGAR_USER_AGENT/NAVER_CLIENT_ID 등)을 그대로 재사용할 뿐 신규 환경변수를 요구하지 않는다.

> **보안 주의**: 과거 버전의 `.env.example`(커밋 f788c18)에는 `NAVER_CLIENT_ID`/`NAVER_CLIENT_SECRET`에 실제 값이 들어 있었고 원격에 push됐다. 현재 `.env.example`의 두 변수는 빈 값이지만 **값은 git 히스토리에 남아 있으므로 Naver 개발자센터에서 키를 교체(rotate)해야 한다**. (2026-10-14: 이 문서 자체에도 그 값이 적혀 있던 것을 지웠다 — 비밀값을 문서에 옮겨 적지 말 것.)

`.env.example`에 **없지만** 코드에서 실제로 읽는 환경변수(각 스크립트에서 `os.environ.get`으로 선택적으로 읽거나, 다른 모듈이 요구하는 것):

| 변수 | 필수/선택 | 사용처 | 비고 |
|---|---|---|---|
| `ACCOUNT_EQUITY` | 선택 | `chat_cli.py`, `run_market_scan.py` (2026-10-15 재정정 — 알림 발송 스크립트들은 삭제됨) | 기본값 10,000,000 — 브로커 연동 전 placeholder 계좌 자산 |
| `MARKET_SCAN_UNIVERSE_SIZE` | 선택 | `run_market_scan.py`, `chat_cli.py` | 기본값 200 |
| `MARKET_SCAN_STALE_HOURS` | 선택 | `run_market_scan.py` | 기본값은 `DEFAULT_STALE_AFTER` 환산값 |
| `RECOMMENDATION_STALE_HOURS` | 선택 | `chat_cli.py` | 기본값 24 |
| `ML_TRAIN_UNIVERSE_SIZE` | 선택 | `train_ml_model.py` | 시장별 종목 수, 기본값 200 |
| `ML_TRAIN_LOOKBACK_YEARS` | 선택 | `train_ml_model.py` | 기본값 5 |
| `ML_TRAIN_EPOCHS` / `ML_TRAIN_BATCH_SIZE` / `ML_TRAIN_LR` / `ML_TRAIN_HIDDEN_DIM` / `ML_TRAIN_N_LAYERS` | 선택 | `train_ml_model.py` | 기본값 100 / 32 / 0.01 / 16 / 2 |
| `ML_TRAIN_DEVICE` | 선택 | `train_ml_model.py`, `train_full.sh` | `auto`(기본, GPU 있으면 GPU) / `cuda` / `cpu`. `train_full.sh` 기본은 `cuda` |
| `ML_TRAIN_LOG_EVERY` | 선택 | `train_ml_model.py`, `train_full.sh` | epoch 로그 간격, 기본 1(`train_full.sh`는 5), 0이면 끔 |
| `ML_TRAIN_VAL_RATIO` | 선택 | `train_ml_model.py` | validation 구간 비율(전체 날짜 대비), 기본 0.1 |
| `CHAT_AGENT_LOOP` | 선택 | `chat_cli.py` | `legacy`면 구 루프. 기본은 deep |
| `LLM_VERIFY_DRY_RUN` | 선택 | `verify_llm_chart_analyst.py` | `"1"`이면 dry-run |
| `NEWS_VERIFY_MARKET` | 선택 | `verify_news_disclosure.py` | `KOSPI`(기본) 또는 `KOSDAQ`/`NASDAQ`/`NYSE` (market 세분화 2026-10 — 과거엔 `KRX`/`NASDAQ` 2개뿐) |
| `NEWS_VERIFY_DRY_RUN` | 선택 | `verify_news_disclosure.py` | `"1"`이면 dry-run |
| `NEWS_SENTIMENT_VERIFY_MARKET` | 선택 | `verify_news_sentiment.py` | `KOSPI`(기본) 또는 `KOSDAQ`/`NASDAQ`/`NYSE` (market 세분화 2026-10 — 과거엔 `KRX`/`NASDAQ` 2개뿐) |
| `NEWS_SENTIMENT_VERIFY_DRY_RUN` | 선택 | `verify_news_sentiment.py` | `"1"`이면 dry-run |

---

## 스크립트 실행 가이드 — 신규 참여자는 이 순서로

이 프로젝트를 처음 받아 로컬에서 아무것도 실행해보지 않은 상태라면, 아래 순서로 "무료 → 저비용 → 전체 실행" 순으로 올라가는 것을 권장한다.

> **재정정 (2026-10-15)**: 아래 가이드의 "3. 알림 경로 확인"/"4. MVP-0 기본 파이프라인 확인"/"8. 네 가지 보조 신호 동시 실행" 단계가 참조하던 스크립트(`verify_telegram.py`/`run_recommendations.py`/`run_recommendations_with_signals.py`)는 메인 챗봇 미사용으로 전부 삭제됐다 — 아래 번호는 그 단계들을 건너뛰고 재배열했다.

1. **환경 준비**: `uv sync --extra dev`로 `.venv`를 만든다(Windows는 CUDA 빌드 torch가 설치됨). `.env.example`을 복사해 `.env`를 만든다. pykrx를 쓸 계획이면 `KRX_ID`/`KRX_PW`도 채운다.
2. **데이터 수집 계층 무비용 확인** — API 키 없이도 되는 것부터:
   - `scripts/verify_fdr_nasdaq.py` (키 불필요, NASDAQ 유니버스+OHLCV 확인)
   - `scripts/verify_pykrx.py` (`KRX_ID`/`KRX_PW` 필요, KRX 유니버스+시가총액 확인)
3. **ML 모델 준비 및 확인**: `scripts/train_ml_model.py`를 `ML_TRAIN_UNIVERSE_SIZE=5 ML_TRAIN_LOOKBACK_YEARS=1 ML_TRAIN_EPOCHS=5`로 스모크 테스트 → 리포트·학습 곡선을 확인한 뒤 `bash scripts/train_full.sh`로 전체 학습.
4. **OpenAI 키 발급 후, dry-run부터**: `OPENAI_API_KEY` 발급 후 비용이 들지 않는 dry-run 모드부터 각 보조 신호를 하나씩 점검.
   - `LLM_VERIFY_DRY_RUN=1 scripts/verify_llm_chart_analyst.py`
   - `NEWS_VERIFY_DRY_RUN=1 scripts/verify_news_disclosure.py` (KOSPI/KOSDAQ/NASDAQ/NYSE 4개 시장 모두 `NEWS_VERIFY_MARKET`로 바꿔가며 — market 세분화 2026-10)
   - `NEWS_SENTIMENT_VERIFY_DRY_RUN=1 scripts/verify_news_sentiment.py` (마찬가지로 4개 시장)
   - `scripts/verify_sources_individually.py all` — DART/EDGAR/네이버/GDELT 데이터 수집 계층만 따로 확인(OpenAI 전혀 안 씀. `etf` 타깃은 실제로는 없음 — 13절 정정 참고)
5. **OpenAI 실호출 1건씩**: dry-run이 전부 정상이면 같은 스크립트를 환경변수 없이(기본값=실호출) 1회씩 실행해 실제 API 응답 품질을 눈으로 확인. 각 스크립트가 `max_calls_per_run=1`로 강제하므로 비용은 호출당 1건으로 제한된다.
6. **전체 유니버스 배치 스캔(선택)**: 챗봇의 "추천 종목 있어?" 질의를 미리 빠르게 받으려면 `scripts/run_market_scan.py`로 캐시를 데운다 — **단, `chat_cli.py`가 실행 중이면 절대 같이 돌리지 말 것**(DuckDB 파일 잠금 충돌).
7. **대화형 챗봇 실행**: 마지막으로 `scripts/chat_cli.py`를 실행해 자유 질의 응답을 테스트. 이 시점이면 어떤 tool이 활성/비활성인지, 어떤 보조 신호가 응답에 포함되는지를 1~6단계에서 이미 다 이해한 상태이므로 디버깅이 쉽다.

이 순서를 따르면 유료 API 호출을 최소한으로 미루면서, 레이어별로(데이터 수집 → ML → LLM 보조신호 → 챗봇) 문제를 하나씩 격리해 확인할 수 있다.

---

## market 세분화(2026-10) — 이 디렉터리 전체에 걸친 공통 변경사항

`data/models.py::VALID_MARKETS`가 `{"KRX", "NASDAQ"}` 2개 값에서 원자(atomic) 거래소 4개 `{"KOSPI", "KOSDAQ", "NASDAQ", "NYSE"}`로 확장됐다 — "KRX"(코스피+코스닥 합성)와 "US"(나스닥+NYSE 합성, 이 세션에서 신설), 그리고 "SP500"/"DOW"(지수 구성종목 필터, 이 세션에서 신설)는 더 이상 `OHLCVRecord`/`get_ohlcv`/(2026-10-15에 삭제된) `run_recommendation_pipeline` 등이 받는 원자 market 값이 아니다 — 이들은 `orchestrator/market_buckets.py`가 정의하는 8개 백그라운드 스캔 버킷(`ALL_SCAN_BUCKETS`) 레벨에만 존재하며, 챗봇의 `get_market_scan_recommendations`/`get_sector_recommendations` 도구(`chat_agent/` 리뷰 문서 참고)만 그 값을 받는다.

이 디렉터리의 스크립트들은 전부 **원자 market만 다루는 레거시/배치 경로**라, 위 1~13번 섹션에서 설명한 대로 "KRX"였던 하드코딩 값을 전부 "KOSPI"(또는 4개 원자값 전체 순회)로 교체하는 정도의 변경만 받았다 — 합성 버킷을 다루는 로직 자체는 이 디렉터리에 없다(그건 `orchestrator/recommendation_coordinator.py`와 `chat_agent/tools.py`의 책임이며, 각각의 리뷰 문서가 다룬다). 신규 공용 모듈 `data/market_routing.py`(`KRX_MARKETS`/`US_MARKETS` 멤버십 집합, `NEWS_SOURCE_LABELS`/`SENTIMENT_SOURCE_LABELS` 라우팅 딕셔너리)는 `pipeline.py`/`tools.py`에 과거 중복됐던 시장 분기 로직을 통합한 것이며, `verify_news_disclosure.py`/`verify_news_sentiment.py`도 이번에 이 모듈을 재사용하도록 바뀌었다.

ML 모델은 market과 무관한 통합 아티팩트 하나(`models/ml_predictor/unified_h1_mamba.pt`)다. 과거 `KRX_h5_logreg.metadata.json`은 어떤 스크립트도 참조하지 않는다 — `train_ml_model.py`(또는 `train_full.sh`)로 통합 모델을 학습해야 `chat_cli.py`/`run_market_scan.py` 등 메인 챗봇 경로가 ML 신호를 쓸 수 있다(`run_recommendations_with_ml.py`/`run_recommendations_with_signals.py`는 2026-10-15에 메인 챗봇 미사용으로 삭제됨).

## 15. `eval_main_agent.py` (2026-10-11 신규)

**목적**: 메인 루프 전환 eval — 구 루프(legacy)와 deepagents 루프(deep)에 같은 대표 질의 Q1~Q12를 돌려 규칙 준수·도구 순서·호출 수·지연을 비교한다(`docs/design/main-agent-migration-plan.md` §9.2, 결정 D11/D18).

**실행 방법**:
```
.venv/Scripts/python scripts/eval_main_agent.py --queries Q1,Q2 --repeats 1      (소규모)
.venv/Scripts/python scripts/eval_main_agent.py                                  (전체: Q1~Q12 × 2루프 × 3회)
.venv/Scripts/python scripts/eval_main_agent.py --loops deep --max-llm-calls 300 (옵션)
```

**API 호출**: 실제 OpenAI 호출(유료). 전역 호출 카운터(메인 + 도구 내부 QueryBudget)가 `--max-llm-calls`(기본 1500)에 닿으면 남은 회차를 건너뛰고 `skipped_reason`을 남긴다. 2026-10-11~12 전체 실행 실측 약 1,157회.

**설계상 결정**: 백그라운드 추천 배치는 실제로 띄우지 않고 트리거 횟수만 세는 스텁 coordinator를 쓴다(배치 1회가 LLM 약 268회라 비용을 지배하므로). Q10(프롬프트 인젝션)은 공시·뉴스 도구 결과를 지시문이 섞인 스텁으로 바꾸는 eval 전용 경로다. 시작 시 `check_tracing_allowed()`·`verify_exposure()`를 CLI와 같이 거친다.

**결과**: `data/eval/main_agent_eval_<timestamp>.json`(gitignore — 대화 원문 포함). **회차마다 저장**하므로 중간에 중단해도 끝난 회차의 상세 결과가 남는다. 터미널에는 회차별 규칙 판정과 질의별 요약이 출력된다.

**주의**: 규칙 판정은 답변 문구 휴리스틱이라 오탐이 있다(예: Q7 고지 문구는 표현이 다양, Q11은 읽기 전용 도구 호출을 실패로 잡던 규칙을 2026-10-12 정정). 판정 전에 결과 JSON의 답변 원문을 확인할 것.
