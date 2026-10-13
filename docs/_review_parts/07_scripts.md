# `scripts/` — 실행 진입점 상세 리뷰

> **2026-10-13 정정 노트**: ML 예측기가 Mamba(통합 단일 모델)로 교체되면서, 이 문서 §2/§3/§4가
> 설명하는 "market별 `KOSPI_h5_logreg.joblib`/`KRX_h5_logreg.joblib` 아티팩트, `load_model("KOSPI")`
> 처럼 market 인자를 받는 호출" 서술은 전부 바뀌었다 — 이제 아티팩트는 하나뿐
> (`models/ml_predictor/unified_h1_mamba.pt`)이고 `load_model()`은 인자를 받지 않는다.
> `train_ml_model.py`도 더 이상 시장별로 따로 학습하지 않고 KRX+NASDAQ+NYSE 전체를
> 하나의 모델로 pooled 학습한다. 상세: `docs/design/ml-predictor-mamba-plan.md`.

이 문서는 `scripts/` 디렉터리의 모든 실행 스크립트를 다룬다. 이 스크립트들은 전부 `src/auto_stock/` 패키지의 함수를 "배선"만 하는 얇은 진입점이며, 새로운 비즈니스 로직을 포함하지 않는다. 각 스크립트는 프로젝트 루트에서 실행해야 한다 — 상대경로로 `.env`와 `data/` 디렉터리를 참조하기 때문이다.

---

## 1. `run_recommendations.py`

**목적**: MVP-0(종목 추천) end-to-end 실행의 가장 단순한 형태. 데이터 수집 → 규칙엔진 → 리스크 사이징 → 설명 생성 → 텔레그램 전송까지 전체 파이프라인을 보조 신호(ML/LLM) 없이 돈다(`run_recommendations.py:1-19`).

**실행 방법**:
```
.venv/Scripts/python scripts/run_recommendations.py   (Windows)
.venv/bin/python scripts/run_recommendations.py        (macOS/Linux)
```
필수 환경변수: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`(`run_recommendations.py:24`에서 `load_telegram_credentials()`가 요구, 없으면 `KeyError`로 즉시 실패).
선택 환경변수: `ACCOUNT_EQUITY`(기본값 10,000,000 — `run_recommendations.py:30,35`).

**동작 흐름**: `WATCHLIST = ["005930", "000660"]`(삼성전자, SK하이닉스 예시 2종목만, `run_recommendations.py:28`) 대상으로 `run_recommendation_pipeline(cache=OHLCVCache, tickers=WATCHLIST, market="KOSPI", account=AccountState, credentials=...)`를 1회 호출(`run_recommendations.py:40-42`). 전체 유니버스를 돌리지 않는 이유가 docstring에 명시돼 있다 — 전체 상장종목을 돌리면 신호 하나당 텔레그램 메시지 1건이 수백~수천 종목에 대해 발사되므로 의도적으로 작은 예시 리스트만 쓴다(`run_recommendations.py:7-11`).

> **market 세분화(2026-10)**: `MARKET = "KRX"`는 `MARKET = "KOSPI"`로 바뀌었다 — `data/models.py::VALID_MARKETS`가 원자(atomic) 거래소 4개(`KOSPI`/`KOSDAQ`/`NASDAQ`/`NYSE`)만 허용하게 되면서 "KRX"는 더 이상 `OHLCVRecord`/`run_recommendation_pipeline`이 받는 원자 market 값이 아니다(코스피+코스닥 합성 스캔 버킷으로만 쓰인다, 아래 §4/§6 참고). WATCHLIST의 005930/000660 둘 다 코스피 종목이라 `MARKET = "KOSPI"`로 교체됐다.

**API 호출**: 텔레그램 전송만 발생(무료, 실제 메시지 1건/신호 발송됨). OpenAI/ML 호출 없음.

**종료/실패 조건**: `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` 누락 시 `load_telegram_credentials()`에서 `KeyError` 발생, 스크립트가 그대로 죽는다(별도 try/except 없음). 종목별 예외는 파이프라인 내부에서 격리되어 `result.errors`에 담기고 다음 종목으로 계속 진행한다(`run_recommendations.py:46-47`에서 출력).

`AccountState`는 브로커 연동 전이라 `held_tickers=frozenset()`, `total_exposure_pct=0.0`으로 고정된 placeholder다(`run_recommendations.py:13-16`) — 사이징 제안은 항상 참고용일 뿐이다.

---

## 2. `run_recommendations_with_ml.py`

**목적**: `run_recommendations.py`에 ML 예측 보조 신호(#2)를 추가한 변형. 그 외 로직은 동일하다(`run_recommendations_with_ml.py:1-16`).

**실행 방법**:
```
.venv/Scripts/python scripts/run_recommendations_with_ml.py   (Windows)
.venv/bin/python scripts/run_recommendations_with_ml.py        (macOS/Linux)
```
필수 환경변수: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`. 추가로 `models/ml_predictor/KOSPI_h5_logreg.joblib` 아티팩트가 **파일로** 존재해야 한다(환경변수가 아니라 디스크 아티팩트 의존).
선택 환경변수: `ACCOUNT_EQUITY`.

**동작 흐름**: 시작 시 `load_model("KOSPI")`로 로컬 아티팩트를 먼저 로드(`run_recommendations_with_ml.py:31-35`), 성공하면 `run_recommendation_pipeline(..., ml_model=ml_model)`로 ML 근거를 `extra_reasons`에 병합해 호출(`run_recommendations_with_ml.py:45-48`). 나머지는 `run_recommendations.py`와 동일한 2종목 워치리스트.

**API 호출**: 텔레그램만(무료). ML 추론은 로컬 모델이라 외부 API 호출 아님.

**종료/실패 조건**: `load_model("KOSPI")`가 `ModelArtifactError`를 던지면(아티팩트 파일 없음) 에러 메시지를 출력하고 `sys.exit(1)`로 즉시 종료한다(`run_recommendations_with_ml.py:31-35`) — 먼저 `train_ml_model.py`를 실행해 아티팩트를 만들어야 한다.

> **market 세분화(2026-10)**: 아티팩트 파일명이 market 값을 그대로 쓰므로(`ml_predictor/artifact.py::artifact_path`), 과거 커밋돼 있던 `models/ml_predictor/KRX_h5_logreg.metadata.json`은 이제 유효하지 않다 — `scripts/train_ml_model.py`를 KOSPI 대상으로 재실행해 `KOSPI_h5_logreg.joblib`을 새로 만들어야 이 스크립트가 동작한다.

---

## 3. `run_recommendations_with_signals.py`

**목적**: ML(#2) + LLM 차트분석(#3) + 뉴스/공시(#4) + 뉴스 감성분석(#4-뉴스) **네 가지 보조 신호를 전부 동시에** 켜는 완전체 버전(`run_recommendations_with_signals.py:1-15`). 이 리포지토리에서 가장 많은 외부 의존성을 요구하는 스크립트다.

**실행 방법**:
```
.venv/Scripts/python scripts/run_recommendations_with_signals.py   (Windows)
.venv/bin/python scripts/run_recommendations_with_signals.py        (macOS/Linux)
```
필수 환경변수: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `OPENAI_API_KEY`(3개 LLM 클라이언트가 전부 이 키를 공유 재사용 — `run_recommendations_with_signals.py:40,43,45`에서 각각 `load_llm_config()`를 호출하지만 내부적으로 같은 `OPENAI_API_KEY`를 읽음). 추가로 `models/ml_predictor/KOSPI_h5_logreg.joblib` 아티팩트 필요.
선택 환경변수: `ACCOUNT_EQUITY`. `DART_API_KEY`/`SEC_EDGAR_USER_AGENT`/`NAVER_CLIENT_ID`/`NAVER_CLIENT_SECRET`는 이 스크립트가 **사전 검증하지 않는다** — 각 소스 모듈이 실제 호출 시점에 직접 읽으므로, 없으면 파이프라인 실행 중 `_news_reasons`/`_sentiment_reasons`가 내부적으로 잡아 `PipelineResult.errors`에만 기록되고 스크립트 자체는 죽지 않는다(`run_recommendations_with_signals.py:12-15`). GDELT는 크리덴셜 자체가 없다.

**동작 흐름**: 시작 순서대로 `load_model("KOSPI")` → `load_llm_config()`(차트분석) → `load_llm_config()`(뉴스/공시) → `load_llm_config()`(뉴스감성) 4단계 사전 로드(`run_recommendations_with_signals.py:54-82`), 하나라도 실패하면 그 지점에서 즉시 종료. 전부 성공하면 `run_recommendation_pipeline(..., ml_model=..., llm_client=..., news_client=..., sentiment_client=...)` 1회 호출(`run_recommendations_with_signals.py:92-95`).

> **market 세분화(2026-10)**: `run_recommendations.py`/`_with_ml.py`와 동일하게 `MARKET = "KRX"` → `MARKET = "KOSPI"`로 바뀌었고(WATCHLIST의 005930/000660 둘 다 코스피), ML 아티팩트도 `KOSPI_h5_logreg.joblib`로 재학습이 필요하다(위 §2 참고).

**API 호출**: **실제 OpenAI API를 호출한다(유료)** — docstring에 명시된 경고(`run_recommendations_with_signals.py:21-24`). 종목 1개당 최대 3회의 OpenAI 호출(차트분석/공시해석/뉴스감성)이 발생할 수 있다. 실행 전 각 보조 신호의 `verify_*` 스크립트를 dry-run으로 먼저 돌려보라고 docstring이 권고한다.

**종료/실패 조건**: `ModelArtifactError`(ML 아티팩트 없음) 또는 `KeyError`(`OPENAI_API_KEY` 없음) 발생 시 각각 안내 메시지 출력 후 `sys.exit(1)`(`run_recommendations_with_signals.py:56-80`). DART/EDGAR/네이버 크리덴셜 누락은 스크립트를 죽이지 않고 파이프라인 내부에서 종목별 에러로만 기록된다.

> 이 스크립트는 **작성되었지만 아직 한 번도 실행되지 않았다**(`docs/CODEBASE_REVIEW.md` 기준) — 유료 API 호출 경로이기 때문에 실제 키 발급 전까지는 실행이 보류된 상태다.

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

## 5. `scan_nasdaq_top100_buy_only.py`

**목적**: NASDAQ 상위 100종목(FDR 리스팅 순서 기준 시가총액 내림차순으로 추정)을 스캔하되 **BUY 신호만** 텔레그램으로 전송하는 변형(`scan_nasdaq_top100_buy_only.py:1-11`). `run_recommendations.py`의 작은 워치리스트 대신 더 넓은 범위를, 그리고 BUY+SELL 대신 BUY만 보내는 별도 스크립트로 분리해 유지한다(기존 스크립트의 기본 동작을 바꾸지 않기 위함).

**실행 방법**:
```
.venv/Scripts/python scripts/scan_nasdaq_top100_buy_only.py   (Windows)
.venv/bin/python scripts/scan_nasdaq_top100_buy_only.py        (macOS/Linux)
```
필수 환경변수: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
선택 환경변수: `ACCOUNT_EQUITY`.

**동작 흐름**: `fdr.StockListing("NASDAQ")`로 리스팅을 가져와 `head(100)`으로 상위 100개 심볼을 추림(`scan_nasdaq_top100_buy_only.py:41-42`, docstring에 NVDA/AAPL/MSFT가 상단에 있음을 직접 확인했다고 기록). 각 종목마다 `get_ohlcv → generate_candidates → (action != "BUY"면 skip) → suggest_position → generate_explanation → send_notification` 순으로 처리(`scan_nasdaq_top100_buy_only.py:60-72`). 이 스크립트는 `run_recommendation_pipeline`을 쓰지 않고 하위 함수들을 직접 호출한다는 점이 다른 스크립트와 다르다 — BUY-only 필터링을 파이프라인 밖에서 걸기 위함.

**API 호출**: 텔레그램만(무료).

**종료/실패 조건**: 종목별 예외는 `try/except Exception`으로 잡아 `errors` 리스트에 누적, 다음 종목 계속 진행(`scan_nasdaq_top100_buy_only.py:71-72`). SELL 신호는 에러가 아니라 `skipped_sell`에 집계되어 마지막에 개수와 티커 목록으로 출력(`scan_nasdaq_top100_buy_only.py:74-80`).

---

## 6. `train_ml_model.py`

**목적**: ML 예측 모델(#2)을 원자 거래소 4개(코스피/코스닥/나스닥/NYSE) 각각 독립적으로 학습해 로컬 아티팩트로 저장하는 스크립트. logreg/rf/dummy 3종을 평가 리포트용으로 비교하지만, 실제로 저장(프로덕션 배선)되는 것은 **logreg뿐**이다(`train_ml_model.py:1-18`).

**실행 방법**:
```
.venv/Scripts/python scripts/train_ml_model.py   (Windows)
.venv/bin/python scripts/train_ml_model.py        (macOS/Linux)

ML_TRAIN_UNIVERSE_SIZE=5 ML_TRAIN_LOOKBACK_YEARS=1 .venv/Scripts/python scripts/train_ml_model.py   (스모크 테스트)
```
필수 환경변수 없음.
선택 환경변수: `ML_TRAIN_UNIVERSE_SIZE`(기본 200), `ML_TRAIN_LOOKBACK_YEARS`(기본 5).

**동작 흐름**: `MARKETS = ["KOSPI", "KOSDAQ", "NASDAQ", "NYSE"]`를 순회(`train_ml_model.py:39,109`)하며 시장마다 `_train_one_market`을 호출한다. 내부적으로: ① `get_universe(market)[:UNIVERSE_SIZE]`로 유니버스 추림 → ② `_collect_universe_records`가 종목별로 `get_ohlcv`를 호출하며 `tqdm` 진행바로 표시, 실패 종목은 스킵하고 계속(`train_ml_model.py:46-61`) → ③ `build_training_dataset(all_records)`로 전종목 pooled 데이터셋 생성 → ④ `chronological_split(dataset)`로 시간순 train/test 분할(랜덤 분할 금지, embargo 갭 적용) → ⑤ train/test 샘플이 부족하면 경고 출력 후 그 시장은 건너뜀(`train_ml_model.py:75-77`) → ⑥ `REPORT_ALGORITHMS = ("logreg", "rf", "dummy")` 3종 모두에 대해 `fit → evaluate → walk_forward_scores`로 AUC/accuracy/base_rate 리포트 출력(`train_ml_model.py:90-100`) → ⑦ `PRODUCTION_ALGORITHM = "logreg"`로 실제 학습(`train`) 후 `save_model(bundle)`로 `.joblib`+`.metadata.json` 저장(`train_ml_model.py:102-104`).

두 시장을 하나로 합쳐 학습하지 않는 이유가 docstring에 명시돼 있다 — 거래시간·업종 구성·규제가 근본적으로 달라 계수가 "두 시장 평균 효과"로 뭉개지면 LogisticRegression을 선택한 이유인 Explainability가 훼손된다는 것이 사용자 확인 사항(2026-09-27)이다(`train_ml_model.py:7-13`).

> **market 세분화(2026-10)**: `MARKETS`가 `["KRX", "NASDAQ"]`(2개)에서 원자 거래소 4개로 확장되면서, 모듈 docstring에 이 원칙의 연장선이 명시적으로 추가됐다(`train_ml_model.py:15-18`) — "KRX"(코스피+코스닥 합성)/"US"(나스닥+NYSE 합성) 같은 합성 버킷(`orchestrator/market_buckets.py`)을 이 스크립트의 학습 대상에 넣으면 바로 위에서 설명한 "두 시장 평균효과로 계수가 뭉개지는" 문제가 그대로 재발하므로, `MARKETS`에는 절대 합성 버킷을 넣지 않는다. 기존에 커밋돼 있던 `models/ml_predictor/KRX_h5_logreg.metadata.json`은 "KRX"가 더 이상 유효한 학습 대상이 아니게 되면서 사실상 폐기됐다 — 재학습하면 `KOSPI_h5_logreg.joblib`/`KOSDAQ_h5_logreg.joblib`/`NASDAQ_h5_logreg.joblib`/`NYSE_h5_logreg.joblib` 4개 아티팩트가 각각 생성된다.

**API 호출**: 없음(완전 무료) — 시세 데이터 수집(FDR/pykrx)과 로컬 학습뿐.

**종료/실패 조건**: 개별 종목 OHLCV 수집 실패는 스킵하고 계속(`train_ml_model.py:52-55`). train/test 샘플 부족 시 해당 시장만 건너뛰고(`return`) 다음 시장 계속 진행 — 스크립트 자체가 죽지는 않는다.

---

## 7. `verify_telegram.py`

**목적**: 텔레그램 봇 연동만 단독으로 확인 — 실제 메시지 1건을 전송해본다(`verify_telegram.py:1-8`).

**실행 방법**: `.venv/Scripts/python scripts/verify_telegram.py` (Windows) / `.venv/bin/python scripts/verify_telegram.py` (macOS/Linux).
필수 환경변수: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

**동작 흐름**: 더미 `Explanation(ticker="TEST", market="KRX", action="BUY", summary="...")`을 만들어 `send_notification`으로 전송(`verify_telegram.py:15-21`).

**API 호출**: 텔레그램 1건(무료). dry-run 모드 없음(이 스크립트 자체가 가장 가벼운 검증이라 필요 없음).

**종료/실패 조건**: `TelegramNotificationError` 또는 `KeyError`(크리덴셜 누락)를 잡아 `FAILED: ...` 출력, 성공 시 `SUCCESS: 텔레그램 메시지 전송 성공`(`verify_telegram.py:20-26`).

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

> **실측 정정 (2026-10-11)**: 아래에 서술된 `etf` 타깃(`test_etf`)은 **실제 스크립트에 없다** — `_TESTS`는 `dart`/`edgar`/`naver`/`gdelt` 4개뿐이다(`verify_sources_individually.py:132`). ETF holdings/목록 조회는 2026-10-11 일회성 스크래치 스크립트로 라이브 검증했다(결과는 `01_data_layer.md`의 `etf_source.py` 절 참고). `etf` 타깃이 필요하면 `fetch_etf_listing()` 건수 + `fetch_etf_holdings("069500", top_n=5)` 출력 정도로 새로 추가해야 한다.

**실행 방법**:
```
.venv/Scripts/python.exe scripts/verify_sources_individually.py dart
.venv/Scripts/python.exe scripts/verify_sources_individually.py edgar
.venv/Scripts/python.exe scripts/verify_sources_individually.py naver
.venv/Scripts/python.exe scripts/verify_sources_individually.py gdelt
.venv/Scripts/python.exe scripts/verify_sources_individually.py etf
.venv/Scripts/python.exe scripts/verify_sources_individually.py all   (5개 순서대로, 하나 실패해도 계속)
```
필수 환경변수: `dart` 타깃은 `DART_API_KEY`, `edgar` 타깃은 `SEC_EDGAR_USER_AGENT`, `naver` 타깃은 `NAVER_CLIENT_ID`+`NAVER_CLIENT_SECRET`. `gdelt`/`etf` 타깃은 크리덴셜 불필요(단, `etf`가 내부적으로 쓰는 pykrx가 `KRX_ID`/`KRX_PW` 로그인을 암묵적으로 요구할 가능성은 라이브 미검증 — 이 스크립트가 바로 그 가능성을 확인하는 용도).

**동작 흐름**: `_TESTS = {"dart": test_dart, "edgar": test_edgar, "naver": test_naver, "gdelt": test_gdelt, "etf": test_etf}` 딕셔너리에서 `sys.argv[1]`(기본값 `"all"`)로 선택된 함수(들)를 실행. 각 `test_*` 함수는 해당 소스 모듈의 핵심 함수를 호출해 결과를 최대 5건까지 출력한다 — 예를 들어 `test_dart`는 `resolve_corp_code`/`resolve_corp_name`/`fetch_disclosures`를 차례로 호출(`verify_sources_individually.py:42-68`). `test_etf`는 `fetch_etf_universe()`(국내 ETF 유니버스 전체 건수 출력) → `fetch_etf_holdings("069500")`(KODEX 200의 구성종목 상위 5개 출력)를 차례로 호출 — 다른 `test_*`와 달리 전용 예외 타입(`*ApiError`)이 없어 `except Exception`으로 포괄 처리한다(모듈 docstring에 "스모크 테스트 전용, 어떤 예외든 그대로 보고"라고 명시).

**API 호출**: 전부 무료(DART/EDGAR/네이버/GDELT/pykrx 데이터 수집만, OpenAI 없음).

**종료/실패 조건**: `dart`/`edgar`/`naver`/`gdelt`는 각각 해당 소스의 `*ApiError`(예: `DartApiError`, `EdgarApiError`, `NaverNewsApiError`, `GdeltApiError`)와 `KeyError`(크리덴셜 누락)를 개별적으로 잡아 `FAILED` 출력 후 함수를 조용히 리턴 — 다른 소스 테스트에 영향 없음. `etf`는 전용 예외 타입이 없어 `Exception` 전체를 잡는다(위 설명). `all` 타깃이면 하나가 실패해도 나머지는 계속 실행된다.

> 코드에 작은 흔적이 하나 있다: `_TESTS` 딕셔너리 선언 바로 위에 주석 처리된 동일한 줄이 남아 있다(`verify_sources_individually.py:130` `# _TESTS = {...}`) — 공백 포맷(`"gdelt":test_gdelt`, 콜론 뒤 공백 없음)만 다른 중복 줄로, 기능에는 영향 없는 사소한 코드 잔재다.

> **market 세분화(2026-10)**: `test_naver()` 함수 안의 `search_news("삼성전자", _SAMPLE_TICKER_KRX, "KRX", start, end)` 호출이 `search_news("삼성전자", _SAMPLE_TICKER_KRX, "KOSPI", start, end)`로 바뀌었다(`verify_sources_individually.py:107`). 이 호출의 `market` 인자는 `NewsArticle.market`에 라벨로만 쓰이는 자유문자열(검증 없음)이라 실질적 동작 변화는 없지만, 다른 스크립트들과의 일관성을 위해 바뀌었다. `_SAMPLE_TICKER_KRX` 변수명 자체와 `test_dart()`/`test_edgar()`/`test_gdelt()`의 섹션 제목(`=== DART (KRX 공시) ===` 등)은 변경되지 않았다 — 이 스크립트는 "KRX"를 원자 market 값으로 실제 전달하는 호출이 1건뿐이라 그 1건만 교체됐다.

---

## 14. `chat_cli.py` — CLI 진입점만 (내부 로직은 `chat_agent/` 모듈 리뷰 참고)

**목적**: 대화형 챗봇 에이전트의 터미널 진입점. `src/auto_stock/chat_agent/loop.py::run_turn`을 stdin 루프로 감싼다(`chat_cli.py:1-5`).

> **갱신 (2026-10-12)**: 기본 메인 루프가 deepagents 기반 `chat_agent/deep_loop.py::run_turn_deep`으로 바뀌었다. 시작 시 트레이싱 차단 검사(`check_tracing_allowed`)와 노출 도구 검사(`verify_exposure`, fail-closed)를 통과해야 deep으로 실행되고, 실패하면 경고 후 구 루프로 실행한다. `CHAT_AGENT_LOOP=legacy`면 처음부터 구 루프(`loop.run_turn`). 시작 메시지에 `(루프: deep|legacy)`가 표시되고, 출력 형식은 두 루프가 같다. deep 모드에서는 턴당 새 백그라운드 추천 배치가 1개로 제한된다(`batch_trigger_limit`).

**실행 방법**:
```
.venv/Scripts/python scripts/chat_cli.py   (Windows)
.venv/bin/python scripts/chat_cli.py        (macOS/Linux)
```
필수 환경변수: `OPENAI_API_KEY`(없으면 시작 시점에 `FAILED` 출력 후 `sys.exit(1)`, `chat_cli.py:139-143`).
선택 환경변수: ML 아티팩트/`OPENAI_API_KEY` 외 보조 신호 크리덴셜(`DART_API_KEY`, `SEC_EDGAR_USER_AGENT`, `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`)이 없어도 **스크립트가 죽지 않는다** — 해당 tool만 `available=False`로 저하되고 나머지 tool로 대화를 계속할 수 있다(배치 스크립트들과 다른 "대화형 우아한 저하" 원칙, `chat_cli.py:1-5`). 추가로 `ACCOUNT_EQUITY`, `RECOMMENDATION_STALE_HOURS`(기본 24시간), `MARKET_SCAN_UNIVERSE_SIZE`(기본 200)도 선택적으로 읽는다(`chat_cli.py:152-168`).

**종료 조건**: 사용자가 `exit`, `quit`, `종료` 중 하나를 입력하면 루프 종료(`chat_cli.py:75, 182-183`). `EOFError`(입력 스트림 종료, 예: 파이프 입력이 끝났을 때)도 루프를 정상적으로 끝낸다(`chat_cli.py:178-179`).

**API 호출**: **실제 OpenAI API를 호출한다(유료)** — 대화 턴마다 최소 1회 이상 호출(`chat_cli.py:20-21`). 이 스크립트는 작성만 되었고 아직 실행된 적이 없다(`docs/CODEBASE_REVIEW.md` 기준).

**Windows 콘솔 인코딩 보정**: 실행 초반에 `sys.stdin`/`sys.stdout`/`sys.stderr`를 전부 `utf-8`로 강제 재설정한다(`chat_cli.py:46-48`) — Windows 콘솔 기본 코드페이지(cp949)가 한글/이모지/em-dash를 인코딩하지 못해 `UnicodeEncodeError`로 죽는 문제, 그리고 stdin을 안 바꾸면 한글 질문이 cp949로 잘못 디코딩되어 API 요청 직렬화 단계에서 예외가 나고 그게 넓은 except에 잡혀 원인 불명의 "일시적 오류"로만 보이는 문제를 실제로 겪고 고친 기록이 docstring에 남아 있다(`chat_cli.py:40-45`).

내부 tool 호출/reasoning 로직, 9개 tool의 세부 동작, `ChatToolContext` 구성 등은 `chat_agent/` 모듈 자체의 리뷰 문서에서 다룬다(이 스크립트는 그 모듈을 호출만 하는 wrapper).

---

## `.env.example` 환경변수 전체 목록

`.env.example`에 정의된 변수:

| 변수 | 필수/선택 | 어느 모듈이 쓰는가 | 비고 |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | 필수(알림봇 사용 시) | `notifier/credentials.py` | 비밀값 |
| `TELEGRAM_CHAT_ID` | 필수(알림봇 사용 시) | `notifier/credentials.py` | 비밀값 아님이지만 보통 비공개 취급 |
| `OPENAI_API_KEY` | 필수(LLM 보조 신호·챗봇 사용 시) | `llm_chart_analyst`, `news_disclosure`, `news_sentiment`, `chat_agent` 공통 | 4개 모듈이 모두 같은 키를 공유 재사용 |
| `OPENAI_MODEL` | 선택 | `llm_chart_analyst/credentials.py` | 기본값 `gpt-5.6-luna` |
| `LLM_MAX_CALLS_PER_RUN` | 선택 | `llm_chart_analyst/credentials.py` | 기본값 20 |
| `DART_API_KEY` | 필수(KRX 공시 분석 시) | `data/sources/dart_source.py` | 비밀값 |
| `SEC_EDGAR_USER_AGENT` | 필수(NASDAQ 공시 분석 시) | `data/sources/edgar_source.py` | **비밀값 아님** — SEC 정책상 요구되는 연락처 포함 식별 문자열. 예: `"auto_stock your-email@example.com"` |
| `NAVER_CLIENT_ID` | 필수(KRX 뉴스 감성분석 시) | `data/sources/naver_news_source.py` | 비밀값 |
| `NAVER_CLIENT_SECRET` | 필수(KRX 뉴스 감성분석 시) | `data/sources/naver_news_source.py` | 비밀값 |
| `NEWS_SENTIMENT_OPENAI_MODEL` | 선택 | `news_sentiment/credentials.py` | 공시 해석(#4)과 별도 모델 등급 지정용, 기본값은 `OPENAI_MODEL`과 동일 |

`.env.example` 파일 자체는 market 세분화 작업(2026-10)에서 수정되지 않았다 — 위 표의 환경변수 목록·값은 실제 파일을 그대로 반영한 것이며 변경 없음. market 세분화로 새로 생긴 코드(`data/market_routing.py`, `orchestrator/market_buckets.py`, `data/sources/index_quote.py`, `data/sources/index_constituents.py`)는 전부 기존 크리덴셜(DART_API_KEY/SEC_EDGAR_USER_AGENT/NAVER_CLIENT_ID 등)을 그대로 재사용할 뿐 신규 환경변수를 요구하지 않는다.

> **보안 주의 — 해결됨(2026-10 재확인)**: 과거 버전의 `.env.example`에는 `NAVER_CLIENT_ID`/`NAVER_CLIENT_SECRET`가 빈 값이 아니라 실제 값처럼 보이는 문자열(`o4b35sf0n0`, `zUZjGfaRu6cojZAYb8xCVeG6g5jFSP2egPUA9yC5`)로 채워져 있던 적이 있었다. 2026-10 재확인 결과 **현재 `.env.example`의 두 변수는 모두 빈 값**이다(`.env.example:20-21`) — 더 이상 해당 경고가 유효하지 않다. 과거에 그 값이 실제로 깃 히스토리에 커밋된 적이 있다면, 히스토리 자체에서 완전히 제거(force-push rewrite 등)됐는지는 이 리뷰 범위 밖이므로 별도로 확인이 필요하다.

`.env.example`에 **없지만** 코드에서 실제로 읽는 환경변수(각 스크립트에서 `os.environ.get`으로 선택적으로 읽거나, 다른 모듈이 요구하는 것):

| 변수 | 필수/선택 | 사용처 | 비고 |
|---|---|---|---|
| `KRX_ID` | 필수(pykrx 사용 시) | `data/sources/pykrx_source.py` | `verify_pykrx.py` docstring에서 언급되지만 `.env.example`에는 빠져 있음 — 템플릿에 추가가 필요해 보인다 |
| `KRX_PW` | 필수(pykrx 사용 시) | `data/sources/pykrx_source.py` | 위와 동일 |
| `ACCOUNT_EQUITY` | 선택 | 거의 모든 `run_*`/`chat_cli.py` 스크립트 | 기본값 10,000,000 — 브로커 연동 전 placeholder 계좌 자산 |
| `MARKET_SCAN_UNIVERSE_SIZE` | 선택 | `run_market_scan.py`, `chat_cli.py` | 기본값 200 |
| `MARKET_SCAN_STALE_HOURS` | 선택 | `run_market_scan.py` | 기본값은 `DEFAULT_STALE_AFTER` 환산값 |
| `RECOMMENDATION_STALE_HOURS` | 선택 | `chat_cli.py` | 기본값 24 |
| `ML_TRAIN_UNIVERSE_SIZE` | 선택 | `train_ml_model.py` | 기본값 200 |
| `ML_TRAIN_LOOKBACK_YEARS` | 선택 | `train_ml_model.py` | 기본값 5 |
| `LLM_VERIFY_DRY_RUN` | 선택 | `verify_llm_chart_analyst.py` | `"1"`이면 dry-run |
| `NEWS_VERIFY_MARKET` | 선택 | `verify_news_disclosure.py` | `KOSPI`(기본) 또는 `KOSDAQ`/`NASDAQ`/`NYSE` (market 세분화 2026-10 — 과거엔 `KRX`/`NASDAQ` 2개뿐) |
| `NEWS_VERIFY_DRY_RUN` | 선택 | `verify_news_disclosure.py` | `"1"`이면 dry-run |
| `NEWS_SENTIMENT_VERIFY_MARKET` | 선택 | `verify_news_sentiment.py` | `KOSPI`(기본) 또는 `KOSDAQ`/`NASDAQ`/`NYSE` (market 세분화 2026-10 — 과거엔 `KRX`/`NASDAQ` 2개뿐) |
| `NEWS_SENTIMENT_VERIFY_DRY_RUN` | 선택 | `verify_news_sentiment.py` | `"1"`이면 dry-run |

---

## 스크립트 실행 가이드 — 신규 참여자는 이 순서로

이 프로젝트를 처음 받아 로컬에서 아무것도 실행해보지 않은 상태라면, 아래 순서로 "무료 → 저비용 → 전체 실행" 순으로 올라가는 것을 권장한다.

1. **환경 준비**: `.env.example`을 복사해 `.env`를 만들고, 최소한 `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`를 채운다. `KRX_ID`/`KRX_PW`도 pykrx를 쓸 계획이면 함께 채운다(템플릿에 없으니 직접 추가).
2. **데이터 수집 계층 무비용 확인** — API 키 없이도 되는 것부터:
   - `scripts/verify_fdr_nasdaq.py` (키 불필요, NASDAQ 유니버스+OHLCV 확인)
   - `scripts/verify_pykrx.py` (`KRX_ID`/`KRX_PW` 필요, KRX 유니버스+시가총액 확인)
3. **알림 경로 확인**: `scripts/verify_telegram.py` — 텔레그램 토큰이 유효한지 가장 간단하게 확인.
4. **MVP-0 기본 파이프라인 확인**: `scripts/run_recommendations.py` — 2종목만 대상으로 전체 추천 파이프라인이 끝까지 도는지 확인(ML/LLM 없음, 텔레그램만 발송).
5. **ML 모델 준비 및 확인**: `scripts/train_ml_model.py`를 `ML_TRAIN_UNIVERSE_SIZE=5 ML_TRAIN_LOOKBACK_YEARS=1`로 스모크 테스트 실행 → 성공하면 `scripts/run_recommendations_with_ml.py`로 ML 보조 신호가 섞인 추천을 확인.
6. **OpenAI 키 발급 후, dry-run부터**: `OPENAI_API_KEY` 발급 후 비용이 들지 않는 dry-run 모드부터 각 보조 신호를 하나씩 점검.
   - `LLM_VERIFY_DRY_RUN=1 scripts/verify_llm_chart_analyst.py`
   - `NEWS_VERIFY_DRY_RUN=1 scripts/verify_news_disclosure.py` (KOSPI/KOSDAQ/NASDAQ/NYSE 4개 시장 모두 `NEWS_VERIFY_MARKET`로 바꿔가며 — market 세분화 2026-10)
   - `NEWS_SENTIMENT_VERIFY_DRY_RUN=1 scripts/verify_news_sentiment.py` (마찬가지로 4개 시장)
   - `scripts/verify_sources_individually.py all` — DART/EDGAR/네이버/GDELT 데이터 수집 계층만 따로 확인(OpenAI 전혀 안 씀. `etf` 타깃은 실제로는 없음 — 13절 정정 참고)
7. **OpenAI 실호출 1건씩**: dry-run이 전부 정상이면 같은 스크립트를 환경변수 없이(기본값=실호출) 1회씩 실행해 실제 API 응답 품질을 눈으로 확인. 각 스크립트가 `max_calls_per_run=1`로 강제하므로 비용은 호출당 1건으로 제한된다.
8. **네 가지 보조 신호 동시 실행**: 위가 전부 통과하면 `scripts/run_recommendations_with_signals.py`로 ML+LLM차트+공시+감성 네 신호를 동시에 켜고 전체 파이프라인을 돈다(유료 — 종목당 최대 3회 OpenAI 호출).
9. **전체 유니버스 배치 스캔(선택)**: 챗봇의 "추천 종목 있어?" 질의를 미리 빠르게 받으려면 `scripts/run_market_scan.py`로 캐시를 데운다 — **단, `chat_cli.py`가 실행 중이면 절대 같이 돌리지 말 것**(DuckDB 파일 잠금 충돌).
10. **대화형 챗봇 실행**: 마지막으로 `scripts/chat_cli.py`를 실행해 자유 질의 응답을 테스트. 이 시점이면 어떤 tool이 활성/비활성인지, 어떤 보조 신호가 응답에 포함되는지를 1~9단계에서 이미 다 이해한 상태이므로 디버깅이 쉽다.

이 순서를 따르면 유료 API 호출을 최소한으로 미루면서, 레이어별로(데이터 수집 → 알림 → MVP-0 파이프라인 → ML → LLM 보조신호 → 챗봇) 문제를 하나씩 격리해 확인할 수 있다.

---

## market 세분화(2026-10) — 이 디렉터리 전체에 걸친 공통 변경사항

`data/models.py::VALID_MARKETS`가 `{"KRX", "NASDAQ"}` 2개 값에서 원자(atomic) 거래소 4개 `{"KOSPI", "KOSDAQ", "NASDAQ", "NYSE"}`로 확장됐다 — "KRX"(코스피+코스닥 합성)와 "US"(나스닥+NYSE 합성, 이 세션에서 신설), 그리고 "SP500"/"DOW"(지수 구성종목 필터, 이 세션에서 신설)는 더 이상 `OHLCVRecord`/`get_ohlcv`/`run_recommendation_pipeline` 등이 받는 원자 market 값이 아니다 — 이들은 `orchestrator/market_buckets.py`가 정의하는 8개 백그라운드 스캔 버킷(`ALL_SCAN_BUCKETS`) 레벨에만 존재하며, 챗봇의 `get_market_scan_recommendations`/`get_sector_recommendations` 도구(`chat_agent/` 리뷰 문서 참고)만 그 값을 받는다.

이 디렉터리의 스크립트들은 전부 **원자 market만 다루는 레거시/배치 경로**라, 위 1~13번 섹션에서 설명한 대로 "KRX"였던 하드코딩 값을 전부 "KOSPI"(또는 4개 원자값 전체 순회)로 교체하는 정도의 변경만 받았다 — 합성 버킷을 다루는 로직 자체는 이 디렉터리에 없다(그건 `orchestrator/recommendation_coordinator.py`와 `chat_agent/tools.py`의 책임이며, 각각의 리뷰 문서가 다룬다). 신규 공용 모듈 `data/market_routing.py`(`KRX_MARKETS`/`US_MARKETS` 멤버십 집합, `NEWS_SOURCE_LABELS`/`SENTIMENT_SOURCE_LABELS` 라우팅 딕셔너리)는 `pipeline.py`/`tools.py`에 과거 중복됐던 시장 분기 로직을 통합한 것이며, `verify_news_disclosure.py`/`verify_news_sentiment.py`도 이번에 이 모듈을 재사용하도록 바뀌었다.

ML 모델 아티팩트(`models/ml_predictor/*.joblib`)는 market 값을 파일명에 그대로 쓰므로(`ml_predictor/artifact.py::artifact_path`), 과거 "KRX" 기준으로 학습·커밋됐던 `KRX_h5_logreg.*` 아티팩트는 이제 어떤 스크립트도 참조하지 않는다 — `scripts/train_ml_model.py`를 재실행해 `KOSPI`/`KOSDAQ`/`NASDAQ`/`NYSE` 4개 아티팩트를 새로 만들어야 `run_recommendations_with_ml.py`/`run_recommendations_with_signals.py`가 다시 동작한다.

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
