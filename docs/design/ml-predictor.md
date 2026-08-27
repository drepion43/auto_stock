# ML 예측 모듈 (#2) 구현 가이드

> `docs/design/ml-predictor-plan.md`(구현 착수 전 계획)를 실제로 구현한 뒤 작성한 결과 문서. `docs/design/rule-engine.md`, `docs/design/orchestrator.md`와 같은 위치·형식.

## 목적

규칙엔진(#1) 후보에 대해 OHLCV 기반 해석 가능한 ML 분류기(scikit-learn `LogisticRegression`)로 "N거래일 후 상승 확률"을 계산해 **보조 근거**로 추가한다. PRD §11 "ML 모델의 구체적 알고리즘/프레임워크 선정" TBD를 해소한다.

## 모듈 구조

```
src/auto_stock/ml_predictor/
├── __init__.py
├── models.py       # FeatureVector, LabeledSample, TrainingDataset, DatasetSplit,
│                   # ModelMetadata, ModelBundle, MLPrediction — 전부 frozen dataclass + slots
├── features.py     # FEATURE_NAMES(11종) + build_feature_vectors / latest_feature_vector
├── labeling.py     # forward_returns / to_label — 5거래일 후 상승(1)/하락(0)
├── dataset.py      # build_training_dataset, chronological_split(embargo), to_xy
├── training.py     # build_estimator(logreg/rf/dummy), train, evaluate, walk_forward_scores
├── artifact.py     # save_model/load_model — joblib + metadata.json 사이드카
└── predictor.py    # predict(bundle, records) -> MLPrediction | None, to_reasons(...)
```

`tests/ml_predictor/`에 각 모듈당 테스트 파일 + `conftest.py`(결정론적 합성 데이터셋 fixture). 총 50개 신규 테스트, 기존 76개 포함 전체 126개 통과, 커버리지 98%.

## 모델 선정: `LogisticRegression` (L2, scikit-learn 기본값)

계획대로 해석 가능성을 최우선 조건으로 `StandardScaler` + `LogisticRegression`을 `Pipeline`으로 묶어 채택했다. `penalty` 인자는 **명시적으로 지정하지 않는다** — scikit-learn 1.8+에서 `penalty="l2"` 같은 문자열 지정이 deprecated되어 1.9.0 설치 환경에서 이미 `FutureWarning`이 발생하고 1.10에서 제거될 예정이기 때문이다(코드 리뷰에서 발견). L2가 이미 기본값이므로 인자를 생략해도 동작은 동일하다. `RandomForestClassifier`/`DummyClassifier(strategy="prior")`는 `training.py`의 `build_estimator`에 함께 구현돼 있지만 `scripts/train_ml_model.py`의 평가 리포트 전용이며 프로덕션(`predictor.py`)에는 배선되지 않는다.

## lookahead bias 방지

- `features.py`: 모든 피처는 시점 `t` 이하 데이터의 후행 윈도우만 사용. `test_feature_at_index_is_unchanged_when_future_bars_are_appended`로 인과성을 자동 검증.
- `labeling.py`: `close[t+horizon]` 기준 레이블, 마지막 `horizon`개 행은 레이블 없음(`test_last_horizon_rows_have_no_label`).
- `dataset.py::chronological_split`: 랜덤 분할이 아니라 시간순 분할 + `embargo_days`(기본값 = horizon) 갭. 다종목 pooled 데이터셋은 종목별이 아니라 전역 날짜 기준으로 분할(`test_multi_ticker_dataset_is_split_by_global_date_not_per_ticker`).
- `training.py`: `StandardScaler`를 `Pipeline` 안에 넣어 fold 밖 통계 누출을 구조적으로 차단.

## 오케스트레이터 통합

`orchestrator/pipeline.py`에 `ml_model: ModelBundle | None = None` 선택 인자를 추가했다. `ml_model=None`(기본값)이면 `generate_explanation(candidate, sizing)`를 **기존과 동일한 2개 위치 인자만으로** 호출해 `extra_reasons` 키워드 인자 자체를 전달하지 않는다 — 기존 회귀 테스트가 2-인자 `side_effect`로 모킹돼 있어 이를 그대로 만족시키기 위한 결정이며, 이로써 기존 동작과 100% 동일함을 보장한다(`test_pipeline_without_ml_model_behaves_identically_to_before`).

`ml_model`이 주어지면 `_ml_reasons(ml_model, records, action)` 헬퍼가 `predict`/`to_reasons`를 호출해 `extra_reasons`로 전달한다. 이 헬퍼는 **어떤 예외도 밖으로 던지지 않는다** — ML 실패는 `PipelineResult.errors`에 `(ticker, "ML 예측 실패: ...")`로 기록되지만 알림 발송은 계속 진행된다(`test_ml_prediction_failure_still_sends_notification_and_records_error`). `explainer/generator.py`는 한 줄도 수정하지 않았다.

## 추론 문구 (`predictor.py`)

`ML_AGREE_THRESHOLD=0.55`, `ML_CONFLICT_THRESHOLD=0.45`를 기준으로 규칙엔진 신호와의 동의/상충/중립 문구를 생성한다. 선형 파이프라인이면 `coef_ * 표준화값`으로 top-3 기여 피처를 뽑아 근거에 포함한다. 모든 문구 뒤에 "(ML 신호는 백테스트 검증 전 참고용 보조 지표입니다)" 고지를 항상 붙인다(PRD §10 준수).

## 모델 아티팩트

`models/ml_predictor/{market}_h{horizon}_logreg.joblib` + 동명의 `.metadata.json` 사이드카. `.gitignore`에 `models/**/*.joblib`을 추가해 바이너리는 제외하고 메타데이터(JSON)만 커밋 대상으로 남긴다.

> **코드 리뷰에서 발견/수정한 버그**: 최초 구현의 `.gitignore` 패턴은 `models/*.joblib`이었는데, git의 glob은 `/`를 넘어가지 않으므로 실제 저장 경로인 `models/ml_predictor/*.joblib`을 무시하지 못해 바이너리가 커밋될 뻔했다. `models/**/*.joblib`로 수정해 실제로 무시되는지 `git check-ignore`로 확인했다.

이번 저장소에는 **실제 모델 아티팩트를 커밋하지 않았다.** 배선 검증용으로 8개 종목·2년치 데이터를 직접 넣어 스모크 테스트했고(`chronological_split` → `train` → `save_model`/`load_model` 왕복까지 정상 동작 확인, AUC 0.46~0.53 수준으로 8종목 소표본에서 나올 법한 노이즈 범위), 그 산출물은 실제 배포 모델이 아니므로 커밋 전에 삭제했다.

전체 유니버스(200종목) 학습은 아직 실행하지 않았다(사용자 판단으로 보류 중). **원인은 이미 찾아 고쳤다** — `pykrx`(설치 버전 1.2.8)는 `KRX_ID`/`KRX_PW` 환경변수로 로그인 세션을 만들어 데이터를 요청한다(`.venv/Lib/site-packages/pykrx/website/comm/auth.py`). 문제는 `pykrx.website.comm.webio` 모듈이 **import 시점에** `_session = build_krx_session()`을 호출하는데, `.env`를 로드하는 `load_dotenv()`가 그보다 먼저 실행된 적이 없어 `os.getenv("KRX_ID")`가 항상 빈 값이었던 것이다. 콘솔의 "KRX 로그인 실패" 메시지가 바로 이 증상이었다(참고로 `docs/design/orchestrator.md`가 언급하는 "무해한 KRX 로그인 경고"는 `FinanceDataReader`가 내부적으로 찍는 **별개의** 메시지로, 이번 pykrx 로그인 실패와는 다른 원인이다 — 메시지 문구가 우연히 비슷해 초기 진단에 혼선이 있었다).

**수정**: `src/auto_stock/data/sources/pykrx_source.py`의 `get_ticker_list`/`get_market_cap` 각각에 `load_dotenv()`를 호출문 앞에 추가했다. `pykrx`의 세션 재시도 로직(`get_auth_session()`)이 매 호출 시점에 `os.getenv()`를 다시 읽으므로, import 시점의 최초 로그인 시도가 실패하더라도 실제 요청 직전에 `.env`가 로드돼 있으면 재로그인에 성공한다 — `.env`에 `KRX_ID`/`KRX_PW`를 설정한 뒤 3~5종목 스모크 테스트로 로그인 성공과 정상 데이터 수신을 직접 확인했다. `notifier/credentials.py`가 이미 쓰던 "각 진입점이 필요한 시점에 `load_dotenv()`를 직접 호출" 패턴을 그대로 따른 것이다.

## 테스트 전략

`rule_engine`의 패턴(계산 함수는 수학적 극단 케이스로, 의사결정 로직은 계산을 모킹해서)을 확장했다. 학습 로직은 단위테스트에서 실제 시장 데이터를 쓰지 않고 `conftest.py`의 결정론적 합성 데이터셋(레이블이 특정 피처의 결정론적 함수)으로 배선만 검증한다. `RANDOM_STATE=0` + `lbfgs`(결정론적 solver)로 재현성을 보장한다.

## 실행 진입점

- `scripts/train_ml_model.py` — 유니버스 조회 → 학습 → logreg/rf/dummy 3종 평가 리포트 출력 → logreg만 저장. 환경변수 `ML_TRAIN_UNIVERSE_SIZE`/`ML_TRAIN_LOOKBACK_YEARS`로 스모크 테스트 규모 조정 가능.
- `scripts/run_recommendations_with_ml.py` — `run_recommendations.py`의 변형. `load_model("KRX")` 실패 시(아티팩트 없음) 안내 메시지 출력 후 종료. 기존 `run_recommendations.py`는 무변경.

## API 키

새로 필요한 것 없음(scikit-learn/joblib/numpy는 로컬 라이브러리). 다만 `scripts/train_ml_model.py`로 실제 KRX 유니버스를 학습하려면 기존에 필요했던 `KRX_ID`/`KRX_PW`(pykrx 로그인)가 설정돼 있어야 한다.

## 이번 스코프가 아닌 것

`docs/design/ml-predictor-plan.md`의 "이번 스코프가 아닌 것" 절 그대로 유효하다 — 전략 레벨 백테스트, 생존편향 제거, 4계층 앙상블 가중치 확정, ML 단독 후보 생성/필터링, 딥러닝 고도화, 전 종목 실시간 스캔, 자동 재학습 등은 모두 이연됐다.
