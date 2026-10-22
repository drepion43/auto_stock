# `src/auto_stock/ml_predictor/` — ML 예측 모듈 (#2, 보조 신호) 상세 리뷰

> **2026-10-14 전면 재작성**: 이 문서는 현재 코드(Mamba SSM, 통합 단일 모델, 다음 거래일 수익률
> regression)를 기준으로 다시 썼다. 2026-10-13 이전의 scikit-learn `LogisticRegression` 설계(5거래일
> 방향 분류, market별 모델, 계수 기반 설명가능성)는 §12 "교체 이력"에 요약만 남겼다. 설계 결정의
> 근거는 `docs/design/ml-predictor-mamba-plan.md`. 줄 번호는 적지 않는다(자주 바뀌어 금방 stale해짐) —
> `파일::이름`으로 찾을 것.

## 0. 이 모듈의 위치와 역할

`ml_predictor`는 종목의 과거 시퀀스로 **다음 거래일 수익률**(`close[t+1]/close[t] - 1`)을 예측한다.
출력은 확률이 아니라 예측 수익률 하나(`MLPrediction.predicted_return`)다.

> **삭제 노트 (2026-10-15)**: 과거 이 값을 쓰던 두 경로 중 "`orchestrator/pipeline.py::run_recommendation_pipeline`(수동 스크립트 경로)"는 메인 챗봇(deepagents 루프)이 전혀 쓰지 않는 코드였다 — `pipeline.py` 자체와 그걸 호출하던 `scripts/run_recommendations_with_ml.py`/`run_recommendations_with_signals.py`를 포함한 수동 스크립트 5개, 그리고 유일한 소비자였던 `notifier/`(텔레그램 발송 모듈)를 전부 삭제했다. 지금 ML 예측 수익률을 쓰는 경로는 아래 하나뿐이다.

**`orchestrator/shortlist.py::build_shortlist`** (챗봇 추천 경로) — 전체 유니버스 종목마다 규칙엔진과 ML을
**독립적으로** 돌리고, 둘 중 하나라도 방향성 신호를 내면(OR) ①단계 숏리스트에 올린다. ML만 잡는 종목도
숏리스트에 들어가고, 숏리스트에 든 종목만 ②단계(LLM 심층분석, 비용 발생)로 넘어간다.
`RecommendationCoordinator`가 이 경로를 쓴다(`05_orchestrator_and_synthesis.md`).

챗봇 도구 `analyze_ml_prediction`(`chat_agent/tools.py`)도 `predict`를 직접 호출한다.

공통 전제: **ML 예측이 실패하거나 모델이 없어도 상위 파이프라인은 막히지 않는다.** 모델 파일이 없으면
`load_model()`이 `ModelArtifactError`를 던지고, 호출부(`scripts/chat_cli.py::_load_ml_model` 등)가 이를 잡아
ML 신호만 끈 채 계속 진행한다. `predict`도 데이터 부족 시 예외 대신 `None`을 반환한다.

```
src/auto_stock/ml_predictor/
├── models.py        # frozen dataclass + slots 도메인 모델
├── features.py      # OHLCV(+국내 펀더멘털) -> 17개 피처
├── fundamentals.py  # 국내 종목 날짜별 펀더멘털 조립(pykrx + DART)
├── labeling.py      # 다음 거래일 수익률 레이블
├── dataset.py       # 종목별 시퀀스 구성, 시간순 분할(embargo), 패딩 배치
├── mamba_block.py   # Mamba(S6) 블록 — MambaStock/mamba.py 원본 그대로
├── pscan.py         # 병렬 스캔(Blelloch) — 원본 그대로
├── training.py      # MambaRegressor, 학습 루프, device 선택, 평가 지표
├── reporting.py     # 학습 곡선 PNG + history CSV
├── artifact.py      # torch .pt(state_dict) + metadata.json 저장/로드
└── predictor.py     # 추론 + 근거 문구 생성
```

---

## 1. lookahead bias 방어 — 이 모듈을 관통하는 설계 축

모델이 학습·검증 시점에 미래 정보를 참조하면 검증 지표만 좋고 실전에서는 쓸모없는 모델이 된다.
현재 코드의 방어 축은 다음과 같다(LogReg 시절의 "스케일러를 Pipeline에 넣기" 축은 Mamba에 스케일러가
없어 사라졌다 — 대신 피처를 처음부터 스케일 프리로 설계했다).

| # | 방어 축 | 구현 위치 | 핵심 |
|---|---|---|---|
| 1 | 피처는 후행(trailing) 윈도우만 사용 | `features.py`, `rule_engine/indicators.py` 재사용 | 시점 `t`의 피처는 `t` 이하 데이터만으로 계산(인과성). 미래 봉을 추가해도 과거 피처가 바뀌지 않음을 `test_features.py`가 검증 |
| 2 | 레이블은 `close[t+1]` 하나만 참조, 레이블 없는 마지막 행 제거 | `labeling.py`, `dataset.py::build_ticker_sequence` | 경계는 `test_labeling.py`가 고정 |
| 3 | 시간순 분할 + embargo (랜덤 분할 금지) | `dataset.py::chronological_split`, `train_val_test_split` | `train_test_split`/`KFold`/`shuffle`을 쓰지 않음(정적 가드 테스트 있음). 구간 경계마다 `LABEL_HORIZON_DAYS`만큼 비움 |
| 4 | 다종목 데이터는 **전역 날짜** 기준으로 분할 | `dataset.py::chronological_split` | 종목별로 따로 자르면 종목 간 시간 누출이 숨는다 |
| 5 | 펀더멘털은 그 시점에 공시돼 있던 값만 사용 | `fundamentals.py`, `data/sources/dart_financials_source.py::fetch_latest_revenue` | DART 매출액은 사업연도 종료 후 공시 유예를 반영해 날짜별로 "그때 알 수 있던 값"만 반환 |
| 6 | 학습 중에는 test 구간을 보지 않음 | `training.py::train` | epoch 선택은 validation으로만 한다(§6). test 지표는 학습이 끝난 뒤 한 번만 계산 |

---

## 2. `models.py` — 도메인 모델

전부 `@dataclass(frozen=True, slots=True)`.

| 클래스 | 필드 | 비고 |
|---|---|---|
| `FeatureVector` | `ticker, market, date, values: dict[str, float]` | `values` 키는 `features.FEATURE_NAMES`(17개) |
| `LabeledSample` | `feature, label: float` | `label` = 다음 거래일 수익률 |
| `TickerSequence` | `ticker, market, samples: list[LabeledSample]` | Mamba 배치의 원자 단위(배치 축 = 종목). 종목마다 길이가 다름 |
| `SequenceDataset` | `sequences, feature_names` | |
| `SequenceDatasetSplit` | `train, test, validation: SequenceDataset \| None = None` | 2분할이면 `validation=None` |
| `EpochRecord` | `epoch, train_loss, val_loss \| None, learning_rate, elapsed_seconds` | 학습 곡선 한 행(2026-10-14 추가) |
| `ModelMetadata` | `market="UNIFIED", algorithm="mamba", horizon_days, feature_names, universe_size, train/test 기간, 표본 수, test_mse/rmse/mae/r2, model_config, trained_at` | `model_config` = `{in_dim, hidden_dim, n_layers}`(모델 재구성용) |
| `ModelBundle` | `estimator, metadata, history: tuple[EpochRecord, ...] = ()` | `estimator`는 **CPU 텐서 state_dict**. `history`는 학습 직후에만 채워지고 아티팩트에는 저장되지 않음 |
| `MLPrediction` | `ticker, market, date, predicted_return` | 확률화(sigmoid)하지 않은 수익률 그대로 |

---

## 3. `features.py` — 17개 피처

`FEATURE_NAMES = CORE_FEATURE_NAMES + FUNDAMENTAL_FEATURE_NAMES`.

**코어 11개(스케일 프리, all-or-nothing)**: `return_1d`, `return_5d`, `return_20d`, `rsi_14`(0~1로 정규화),
`macd_hist_norm`(MACD 히스토그램/종가), `close_to_sma20`, `close_to_sma60`, `sma20_to_sma60`, `atr_14_pct`(ATR/종가),
`volume_ratio_20`, `channel_position_20`(20일 고저 채널 내 위치). 전부 비율이라 원화·달러 종목을 한 모델에 섞어도
스케일 왜곡이 없다. 하나라도 워밍업 미충족(최장 SMA60)이면 그 시점 전체가 `None`이다.

**펀더멘털 6개(국내 전용, 결측 허용)**: `earnings_yield`(1/PER), `pbr`, `dividend_yield`, `turnover_rate`(거래량/상장주식수),
`sales_yield`(매출액/시가총액, 1/PSR), `fundamental_available`(플래그). 미국 종목이나 조회 실패 날짜는 중립값
(`pbr=1.0`, 나머지 0.0)과 `fundamental_available=0.0`으로 채운다. 국내 종목은 일부 필드만 빠져도 플래그는 1.0이고
빠진 필드만 중립값이다. PER/PSR은 손실기업의 음수·발산을 줄이려고 역수로 쓴다.

> 참고: `fundamental_available`은 미국 종목에서 항상 0이라 사실상 국내/미국 구분 신호 역할도 한다. 다만 국내 종목이
> 조회에 실패한 날도 0이 되어 두 의미가 섞인다. 별도 `is_domestic` 피처는 아직 없다(논의만 됨).

- `build_feature_vectors(records, fundamentals_by_date=None) -> list[FeatureVector | None]` — records와 같은 길이.
- `latest_feature_vector(records, fundamentals_by_date=None)` — 마지막 원소만. Mamba 추론은 시퀀스 전체를 쓰므로
  이 함수를 쓰지 않고, `llm_chart_analyst/snapshot.py`가 단일 시점 지표 스냅샷용으로 쓴다.

## 4. `fundamentals.py` — 국내 펀더멘털 조립

`build_fundamentals_by_date(ticker, market, start, end) -> dict[date, dict] | None`. market이 KOSPI/KOSDAQ이
아니면 즉시 `None`. 국내면 pykrx(`get_market_fundamental_range`, `get_market_cap_detail_range`)와 DART 매출액
(`dart_financials_source.fetch_latest_revenue`, 사업연도 단위로 메모이즈)을 날짜별로 합친다. 두 소스는 독립이라
DART 실패(키 없음 등)는 매출액만 `None`으로 만들고 나머지는 유지한다. pykrx 결과는 캐시하지 않아 학습할 때마다
다시 조회한다(DART 매출액은 `data/dart_revenue_cache.json`에 디스크 캐시).

## 5. `labeling.py` / `dataset.py`

- `LABEL_HORIZON_DAYS = 1`, `forward_returns(closes, horizon)` — 마지막 `horizon`개는 `None`.
- `build_ticker_sequence(records, fundamentals_by_date)` — 워밍업 구간과 레이블 없는 꼬리를 뺀 `TickerSequence`
  (표본이 없으면 `None`).
- `build_sequence_dataset(records_by_ticker, fundamentals_by_date_by_ticker)` — 종목별 시퀀스 묶음. 펀더멘털 리스트는
  records와 1:1(미국 종목은 `None`).
- `chronological_split(dataset, test_ratio=0.2, embargo_days=None)` — 전역 날짜 컷오프로 train/test, 경계에 embargo.
- `train_val_test_split(dataset, val_ratio=0.1, test_ratio=0.2, embargo_days=None)` — `[train | embargo | val | embargo | test]`.
  **test 구간은 `chronological_split`과 정확히 같다**(validation을 추가해도 test 결과를 이전과 비교 가능). validation은
  test 앞쪽 구간을 같은 규칙으로 한 번 더 잘라 만든다.
- `collate_sequences(sequences, feature_names)` — 배치 내 최대 길이로 0-패딩하고 `mask`로 실제 시점만 표시.
- `to_xy_flat(dataset)` — 모든 시점을 flat하게 펼친 평가용 헬퍼.

분할 후 각 구간의 시퀀스는 그 구간 날짜만 남긴 조각이다. 즉 validation/test 시퀀스는 구간 시작일부터 시작해
그 이전 히스토리를 상태로 이어받지 않는다(추론 때는 `ML_LOOKBACK_DAYS`(3년) 히스토리 전체를 넣는 것과 조건이 다름).

---

## 6. `training.py` — 모델, 학습 루프, 평가

### 모델: `MambaRegressor`
`nn.Linear(in_dim, hidden) -> Mamba(n_layers) -> nn.Linear(hidden, 1)`. MambaStock `main.py::Net`과 같은 구조이되
마지막 `Tanh()`는 없다. 입력 `(B, L, in_dim)` → 출력 `(B, L)` 시점별 예측 수익률. `Mamba`/`MambaBlock`/`pscan`은
`mamba_block.py`·`pscan.py`에 원본 로직 그대로 있다(`MambaConfig.pscan=True`로 병렬 스캔).

기본 하이퍼파라미터: `DEFAULT_HIDDEN_DIM=16`, `DEFAULT_N_LAYERS=2`, `DEFAULT_EPOCHS=100`, `DEFAULT_LEARNING_RATE=0.01`,
`DEFAULT_WEIGHT_DECAY=1e-5`, `DEFAULT_BATCH_SIZE=32`, `RANDOM_STATE=0`. 옵티마이저는 Adam, 학습률 스케줄러는 없다.

### device: `resolve_device(device)`
`None`/`"auto"`면 CUDA가 있으면 GPU, 없으면 CPU. `"cuda"`를 명시했는데 GPU를 못 쓰면 `ValueError`(CPU로 조용히
떨어지지 않음). Windows에서는 pyproject의 `[tool.uv.sources]`가 PyTorch `cu130` 빌드를 설치한다(RTX 50/Blackwell 지원).
2026-10-14 측정(B=32, L=1200, 17 피처, hidden 16): GPU 54ms vs CPU 738ms/step.

### `train(split, universe_size, ..., device=None, log_every=0) -> ModelBundle`
1. 시드 고정, device 결정, 모델을 device로 이동.
2. 매 epoch `_train_one_epoch`: 종목 단위 미니배치(`_batches`, 셔플 없음) → `_to_tensors` → `_masked_mse_loss`
   (패딩 시점 제외 MSE) → 역전파. epoch 평균 train loss는 GPU에서 누적해 epoch 끝에 한 번만 꺼낸다.
3. `split.validation`이 있으면 매 epoch `_masked_loss_on`으로 val loss 계산(train loss와 같은 정의라 같은 척도).
4. 매 epoch `EpochRecord` 기록. `log_every > 0`이면 그 간격과 마지막 epoch에 INFO 로그 한 줄
   (`epoch  N/M  loss=…  val_loss=…  lr=…  time=…  elapsed=…`).
5. 학습이 끝난 뒤 `evaluate(model, split.test)`로 test 지표를 한 번 계산해 메타데이터에 넣는다.
6. state_dict를 **CPU 텐서로 복사**해 반환(추론 환경이 CPU 전용이어도 로드 가능). `history`도 함께 반환.

~~**저장되는 가중치는 마지막 epoch의 것**이다.~~ **갱신 (2026-10-22)**: 매 epoch validation 구간의 날짜별 순위 IC(`EpochRecord.val_ic`, history CSV `val_ic` 열)를 계산하고, **val IC가 가장 높았던 epoch의 가중치**를 저장한다(`train(select_by_val_ic=True)` 기본). 학습 타깃은 기본으로 같은 날·같은 지역 평균을 뺀 수익률(`dataset.cross_sectional_demean`, 스크립트가 train·validation에만 적용, test는 원래 레이블로 평가). 진단·근거: `docs/design/ml-predictor-mamba-plan.md` "2026-10-22 진단".

### 평가 함수
- `evaluate(model, dataset, feature_names)` — **종목별 마지막 시점 예측만** 평가(추론 경로와 같은 조건). 반환 키:
  `mse, rmse, mae, r2, hit_rate`. 표본(종목)이 2개 미만이면 전부 `nan`.
- `persistence_baseline_metrics(dataset, feature_names)` — "내일 수익률 = 오늘 수익률(`return_1d`)" 베이스라인, 같은 키.
- `hit_rate`(`_hit_rate`) — 예측과 실제 수익률의 **부호가 같은 비율**(방향 적중률). 실제 수익률이 정확히 0인 표본은
  분모에서 빼고, 예측 0은 오답으로 친다.
- `split_by_region(dataset)` — `{"domestic": KOSPI/KOSDAQ, "us": 그 외}`로 나눈다(시장별 리포트용).

평가 표본은 종목당 1개라 종목 600개면 표본도 600개 정도로, 지표가 흔들릴 수 있다.

## 7. `reporting.py` — 학습 곡선

- `best_val_epoch(history)` — val loss 최저 epoch(동률이면 앞선 것), validation이 없으면 `None`.
- `save_history_csv(history, path)` — 열: `epoch, train_loss, val_loss, learning_rate, elapsed_seconds`(val 없으면 빈칸).
- `plot_training_history(history, path)` — matplotlib(Agg) PNG. 위 패널 train/val loss(양수면 로그 척도, best val epoch
  점선+표시), 아래 패널 학습률(0부터). loss와 학습률은 척도가 달라 이중 y축 대신 패널을 나눴다. 색은 dataviz 기본
  팔레트 1·2번(train `#2a78d6`, val `#eb6834`, CVD 검증 통과). 로그 눈금은 U+2212 글리프 문제를 피하려고 일반 텍스트
  (`1e-03`)로 표시한다.

## 8. `scripts/train_ml_model.py` / `scripts/collect_ml_dataset.py` / `scripts/train_full.sh` — 학습 실행

> **갱신 (2026-10-15)**: 데이터 수집(OHLCV+펀더멘털 → `SequenceDataset`)이 `ml_predictor/collection.py`(신규)로 추출됐다 — `universe_for`/`atomic_market_for`/`collect_universe_records`(과거엔 `train_ml_model.py`의 `_universe_for`/`_atomic_market_for`/`_collect_universe_records` 사설 헬퍼였음) + `save_dataset`/`load_dataset`(신규, pickle 기반). 신규 `scripts/collect_ml_dataset.py`가 수집만 해서 `data/ml_training_dataset.pkl`(기본 경로, `ML_TRAIN_DATASET_PATH`로 변경 가능)에 저장하고 종료한다. `train_ml_model.py`는 그 파일이 있으면 재수집 없이 로드해서 학습만 하고, 없으면(또는 `ML_TRAIN_FORCE_COLLECT=1`이면) 예전처럼 직접 수집한다 — 즉 "수집+학습 한 번에" 모드와 "수집 따로, 학습 따로" 모드를 둘 다 지원한다.

1. `resolve_device(ML_TRAIN_DEVICE)` — 데이터 수집/로드 **전에** device 확인.
2. `ML_TRAIN_DATASET_PATH`에 파일이 있고 `ML_TRAIN_FORCE_COLLECT`가 아니면 `load_dataset`으로 즉시 로드. 아니면
   `collection.collect_training_records`로 수집 후 `save_dataset`으로 저장. **갱신 (2026-10-22)**: 원자 시장
   `TRAINING_MARKETS = ("KOSPI", "KOSDAQ", "NASDAQ", "NYSE")`마다 `universe.select_universe(market, 수집 시작일, UNIVERSE_SIZE)` —
   국내는 **수집 시작일 시점 시가총액 상위 보통주**(우선주·거래정지 제외, `pykrx_source.get_market_cap_table`, 휴장일의 "전부 0인 표"는
   건너뜀), 미국은 현재 FDR 목록(시가총액 순). 이전 `universe_for("KRX")[:200]`은 KOSPI 뒤에 KOSDAQ을 붙인 목록의 앞 200개라
   **KOSDAQ이 통째로 빠졌고**, KOSPI도 가나다순(우선주 포함) 임의 200개였다. `collect_universe_records`가 OHLCV(`data/ohlcv.duckdb` 캐시)와
   국내 펀더멘털을 수집(종목 단위 예외 격리)하는 것은 같다.
3. `filter_regions(dataset, ML_TRAIN_REGIONS)` → `train_val_test_split(dataset, VAL_RATIO, TEST_RATIO=0.2)` → 구간별 기간·종목 수 출력 →
   하이퍼파라미터 출력 → `train(training_split(split))`(타깃이 `demeaned`면 train·validation 레이블을 같은 날·같은 지역 평균 차감).
4. `_print_evaluation_report` — **validation/test × 지역(KR/US)**마다 Mamba·0 예측·persistence의 MSE/방향적중률과, Mamba·persistence·
   **ridge 기준선**(`baseline.fit_ridge`, 같은 피처·타깃)의 순위 IC·상위−하위 스프레드·**회전율·왕복비용 차감 스프레드**를 출력하고,
   validation IC로 "Mamba 사용 가능 / 선형 기준선을 이기지 못함 → 배포하지 말 것"을 판정한다.
5. `save_model` → `unified_h1_mamba.history.{csv,png}` 저장 → 저장된 가중치의 epoch(val IC 최고)와 val loss 최저 epoch 출력.

환경변수: `ML_TRAIN_UNIVERSE_SIZE`(원자 시장별, 200), `ML_TRAIN_LOOKBACK_YEARS`(5), `ML_TRAIN_EPOCHS`(100), `ML_TRAIN_BATCH_SIZE`(32),
`ML_TRAIN_LR`(0.01), `ML_TRAIN_HIDDEN_DIM`(16), `ML_TRAIN_N_LAYERS`(2), `ML_TRAIN_DEVICE`(auto), `ML_TRAIN_LOG_EVERY`(1),
`ML_TRAIN_VAL_RATIO`(0.1), `ML_TRAIN_DATASET_PATH`(`data/ml_training_dataset.pkl`), `ML_TRAIN_FORCE_COLLECT`(1이면 저장된 파일 무시),
**신규(2026-10-22)** `ML_TRAIN_TARGET`(`demeaned` 기본 | `raw`), `ML_TRAIN_REGIONS`(`KR,US` 기본), `ML_EVAL_COST_KR`(0.0025)·`ML_EVAL_COST_US`(0.0005)
(순 스프레드용 왕복비용 — 가정값).

`scripts/collect_ml_dataset.py`는 위 1·2단계만 수행하고 종료한다(학습 없음) — 환경변수는 `ML_TRAIN_UNIVERSE_SIZE`/`ML_TRAIN_LOOKBACK_YEARS`/`ML_TRAIN_DATASET_PATH`를 `train_ml_model.py`와 공유한다(같은 유니버스/기간으로 수집해야 나중에 그대로 로드 가능).

`scripts/train_full.sh` — 전체 학습 래퍼. 프로젝트 루트로 이동, `.venv` 파이썬 탐색(Windows/Unix), 기본
`ML_TRAIN_DEVICE=cuda`·`ML_TRAIN_LOG_EVERY=5`·`PYTHONUTF8=1`, 출력을 `logs/ml_training/train_full_<시각>.log`에 `tee`
(`pipefail`로 학습 실패 시 비0 종료). `logs/`는 gitignore.

## 9. `artifact.py` — 저장/로드

- `artifact_path(model_dir)` → `models/ml_predictor/unified_h1_mamba.pt`(market 인자 없음, 통합 모델 하나).
- `save_model(bundle)` — `torch.save(state_dict)` + `asdict(metadata)`를 `.metadata.json`으로(날짜는 ISO 문자열).
  `history`는 저장하지 않는다(학습 스크립트가 CSV/PNG로 따로 남김).
- `load_model(model_dir)` — `.pt`/`.metadata.json` 중 하나라도 없으면 `ModelArtifactError`. `torch.load(path, weights_only=True)`.
  `.pt`는 gitignore(`models/**/*.pt`), 메타데이터 JSON은 커밋 대상.

## 10. `predictor.py` — 추론과 근거 문구

- `ML_LOOKBACK_DAYS = 365 * 3` — 추론 호출부는 규칙엔진용 120일이 아니라 이 기간을 따로 조회해야 한다
  (`shortlist.py`, `chat_agent/tools.py`가 그렇게 함 — 과거엔 `pipeline.py::_ml_reasons`도 그랬으나 2026-10-15에 삭제됨).
- `build_model(bundle)` — `model_config`로 `MambaRegressor`를 만들고 state_dict 로드, `eval()`. CPU에서 동작.
- `predict(bundle, records, fundamentals_by_date=None) -> MLPrediction | None` — 시퀀스 전체를 넣고 **마지막 시점 예측**을 쓴다.
  시퀀스를 못 만들면 `None`.
- `ML_AGREE_RETURN_THRESHOLD = 0.003`, `ML_CONFLICT_RETURN_THRESHOLD = -0.003` — ±0.3% 이상을 방향성 있는 예측으로 본다.
- `_stance(prediction, action)` — 규칙엔진 action(BUY/SELL)과 동의/상충/중립 문구(예상 수익률 % 표기).
- `to_reasons(prediction, action)` — `[_stance 문구, DISCLAIMER]`. 기여 피처 문구는 없다(계수 분해 불가, 사용자 승인).
  `DISCLAIMER = "(ML 신호는 백테스트 검증 전 참고용 보조 지표입니다)"`는 항상 붙는다.

숏리스트(`shortlist.py::_ml_has_directional_opinion`)는 같은 ±0.3% 기준으로 "방향성 의견 있음"을 판정하고,
`RecommendationCoordinator._shortlist_score`는 `abs(predicted_return)`을 점수에 더한다.

---

## 11. 현재 한계 (2026-10-14)

- **실사용할 학습 모델이 아직 없다.** 2026-10-14 23:01 스모크 실행으로 `unified_h1_mamba.pt`(+ metadata, history)가 로컬에
  생겼지만 14종목·1년·5 epoch짜리라(test R² −72.7, `.pt`는 gitignore) 의미 있는 신호가 아니다. 이 파일이 있으면 ML 신호
  경로가 **켜지므로** 실사용 전에 전체 학습으로 교체하거나 지워야 한다. 같은 날 23:03에 시작한 전체 학습(`train_full.sh`)은
  KRX 43/200 종목(종목당 약 44초)에서 로그가 멈췄다. 교체 전 `KRX_h5_logreg.metadata.json`도 남아 있다(지금 코드는 읽지 않음).
- **국내 펀더멘털 조회 실패(2026-10-14 로그)** — pykrx `get_market_fundamental_by_date`가 `None of [Index(['TRD_DD','BPS','PER',…])]
  are in the [columns]` 에러를 내며 빈 결과를 돌려준다. 이 경우 `fundamentals.py`는 예외 없이 중립값으로 채우므로 국내 종목의
  PER/PBR/배당 피처가 조용히 전부 중립값(`fundamental_available=0`)이 될 수 있다. 원인(pykrx/KRX 쪽 변경, 로그인 필요 여부)은 미확인.
- ~~**early stopping 없음**~~ — 2026-10-22부터 validation 순위 IC 최고 epoch 가중치를 저장한다.
- ~~**평가는 종목당 마지막 시점 1개**~~ — 구간 내 모든 시점을 채점한다(evaluation.py). 2026-10-22부터 회전율·왕복비용 차감 스프레드도
  나온다(비용은 가정값). 누적손익·MDD 같은 전략 백테스트는 여전히 없다.
- **생존편향** — 국내는 2026-10-22부터 수집 시작일 시점 상장 종목으로 골라 줄었다. 미국은 과거 시점 목록 소스가 없어 "오늘 상장된 종목" 그대로다.
- ~~**유니버스 순서 미검증**~~ — 확인 결과 국내는 가나다순이었고 KOSDAQ이 빠져 있었다(2026-10-22 수정, 위 2단계).
- **예측력 미확인** — 2026-10-22 진단에서 기존 Mamba는 같은 피처의 선형 ridge보다 못했다. 새 유니버스 데이터(`data/ml_training_dataset_v2.pkl`, 수집 중)로
  재학습해 리포트의 판정을 본 뒤에 ML 신호 사용 여부를 정한다.
- **설명가능성 없음** — 블랙박스 모델이라 "왜 그 예측인지"를 피처 단위로 설명하지 못한다(사용자 승인 트레이드오프).
- 4계층 신호(규칙엔진/ML/LLM차트/뉴스)의 가중 결합은 PRD §5/§11에서 미확정.

## 12. 교체 이력 (LogReg → Mamba, 2026-10-13)

교체 전에는 scikit-learn `StandardScaler + LogisticRegression` 파이프라인으로 5거래일 후 상승 확률을 분류했고,
market별(KRX/NASDAQ…) 아티팩트 `{market}_h5_logreg.joblib`를 따로 학습했으며, 계수×표준화값으로 상위 3개 기여 피처를
보여줬다. 2026-09-28 KRX 200종목 학습 결과는 test ROC-AUC 0.536(무작위 0.5보다 약간 높음). 교체로 바뀐 것: 레이블(5일 방향 →
1일 수익률), 모델(LogReg → Mamba), 단위(market별 → 통합 1개), 출력(`probability_up`/`top_features` → `predicted_return`),
아티팩트(`.joblib` → `.pt`), `data/market_routing.py::ML_MODEL_KEYS` 삭제. 상세: `docs/design/ml-predictor-mamba-plan.md`,
교체 전 설계 기록은 `docs/design/ml-predictor.md`(역사적 기록 노트 있음).

## 13. 테스트 (`tests/ml_predictor/`, 2026-10-14 기준 75개)

| 파일 | 테스트 수 | 검증 내용 |
|---|---|---|
| `conftest.py` | — | 결정론적 합성 OHLCV, `synthetic_sequence_dataset`(길이 다른 2종목) |
| `test_features.py` | 15 | 피처 계산, 워밍업 `None`, 인과성, 펀더멘털 중립값·플래그 |
| `test_labeling.py` | 5 | `close[t+1]` 경계, 꼬리 `None` |
| `test_dataset.py` | 15 | 시퀀스 구성, 전역 날짜 분할, embargo, 3분할 순서·embargo·test 구간 동일성, 패딩/마스크, 셔플 금지 가드 |
| `test_training.py` | 21 | forward/backward, device 선택(CUDA 없을 때 실패 포함), CPU state_dict, 로그 형식·간격, train/val history, `hit_rate`, `split_by_region` |
| `test_reporting.py` | 6 | best epoch, CSV 열·빈 val, PNG 생성 |
| `test_artifact.py` | 5 | 저장/로드 왕복, 파일 없을 때 `ModelArtifactError` |
| `test_predictor.py` | 8 | `predict`의 `None` 조건, `_stance` 3분기, `DISCLAIMER` 항상 포함 |

`slow` 마커가 붙은 학습 테스트는 `-m "not slow"`로 뺄 수 있다. CUDA 테스트는 GPU가 없으면 skip.
