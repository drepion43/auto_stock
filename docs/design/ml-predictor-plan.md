# ML 예측 모듈 (#2) 구현 계획

> `docs/IMPLEMENTATION_PLAN.md`의 "신호 소스 확장" §1(ML #2)을 실행 가능한 수준으로 구체화한 사전 계획 문서. `docs/design/rule-engine.md`, `docs/design/orchestrator.md`와 같은 위치·형식을 따르되, 이 문서는 **구현 착수 전 계획**이다 — 구현이 끝난 뒤에는 실제로 무엇을 어떻게 만들었는지 기록하는 `docs/design/ml-predictor.md`를 별도로 작성한다 (Phase 4 참고).

## 배경

`auto_stock`의 MVP-0(데이터수집 #0, 규칙엔진 #1, 리스크사이징 #5, 설명생성기 #6, 텔레그램알림 #7, 오케스트레이터)는 이미 구현 완료되어 커밋(`60523cd`)되었고 테스트 76개 전부 통과 상태다.

다음 단계로 브로커 주문 실행(#8, MVP-1)은 배제하고, `IMPLEMENTATION_PLAN.md`가 정의한 "신호 소스 확장" 3종(ML #2 / LLM 차트분석 #3 / 뉴스분석 #4) 중 **ML 예측 모듈(#2)부터** 진행한다. PRD §5는 "해석 가능한 전통 ML(scikit-learn 등)로 시작 → 검증 후 딥러닝으로 고도화"를 이미 방향으로 확정했고, §11은 "ML 모델의 구체적 알고리즘/프레임워크 선정"을 이번 구현 단계에서 확정할 TBD로 남겨뒀다. `explainer/generator.py`는 이미 `extra_reasons` 파라미터로 이 확장을 염두에 두고 설계돼 있다 — ML #2, LLM 차트 #3, 뉴스 #4의 reason 문자열을 그대로 통과시킬 수 있도록 만들어졌다는 docstring이 있다.

**목표**: OHLCV만으로 학습하는 해석 가능한 ML 분류기를 만들어, 규칙엔진 후보에 대한 **보조 근거**로 파이프라인에 배선한다. 완전한 전략 백테스트나 4계층 앙상블 가중치 확정은 PRD §5/§11이 이미 TBD로 남긴 영역이므로 이번 스코프에서 제외한다.

## 핵심 설계 결정

1. **모델: `scikit-learn`의 `LogisticRegression`(L2 정규화)**
   해석 가능성이 최우선 조건(PRD §5)이기 때문이다. 계수 기반으로 개별 예측의 기여 피처를 분해할 수 있어 PRD §5.1 Explainability("왜 63%인지")를 만족하는 유일한 후보이고, 추론 비용이 사실상 0이라 전종목 스캔에 적합하며, 11개 피처 선형모델+L2는 금융 시계열의 낮은 신호대잡음비에서 가장 안전한 출발점이다. `RandomForestClassifier`와 `DummyClassifier(strategy="prior")`는 학습 스크립트의 **평가 리포트 전용 벤치마크**로만 사용하고 프로덕션에는 배선하지 않는다. RandomForest가 walk-forward AUC 중앙값 기준 LogReg 대비 +0.02 이상 우세할 때만 승격을 재검토한다는 결정 규칙을 미리 못박아 사후 정당화(결과를 보고 기준을 만드는 것)를 방지한다.

2. **오케스트레이터 통합: `extra_reasons` 경유 보조 신호**
   rule_engine처럼 자체 `Candidate`를 만드는 대신, 이미 존재하는 `extra_reasons` 확장 포인트를 통해 **규칙엔진 후보가 이미 있는 티커에 대해서만** 보조 문장을 추가한다. ML이 알림 건수를 늘리거나 후보를 필터링하지 않는다 — PRD §5의 앙상블 방식 TBD, §10의 "보조 시그널로만 사용" 원칙과 일치한다. `explainer/generator.py`는 **한 줄도 수정하지 않는다.**

3. **lookahead bias 방지 (5중 방어)**
   피처는 후행(trailing) 윈도우만 사용, 레이블은 `close[t+H]` 기준으로 마지막 H개 행 절단, 학습/테스트 분할은 **시간순 + embargo 갭**(랜덤 분할 금지), 스케일러는 `Pipeline` 안에 넣어 fold 밖 통계 누출을 구조적으로 차단한다. "미래 봉을 추가해도 과거 시점 피처값이 바뀌지 않는다"는 인과성(causality) 속성 테스트로 이를 자동 검증한다.

4. **모델 아티팩트는 커밋하지 않는다**
   `.joblib`은 `data/*.duckdb`와 같은 이유로 `.gitignore`에 추가한다. 대신 학습 스크립트 + 하이퍼파라미터 상수 + 메타데이터 JSON(성능 지표 포함)을 커밋해 "어떤 모델이 어떤 성능으로 배포됐는지"를 git 히스토리에 남긴다.

5. **ML 실패는 추천 발송을 막지 않는다**
   ML 예측이 실패해도 발송은 계속된다. 실패는 `PipelineResult.errors`에 기록하되(`orchestrator.md`의 "silent failure 금지" 원칙), 해당 `Explanation`은 `sent`에도 정상적으로 담긴다.

## 신규 모듈 구조

`rule_engine/` 패턴(models.py + 순수계산 + engine.py 분리)을 그대로 따른다.

```
src/auto_stock/ml_predictor/
├── __init__.py
├── models.py       # FeatureVector, LabeledSample, TrainingDataset, DatasetSplit,
│                   # ModelMetadata, ModelBundle, MLPrediction — 전부 frozen dataclass + slots
├── features.py     # OHLCV -> 스케일 프리 피처 11종. rule_engine.indicators(sma/rsi/macd/atr) 재사용
├── labeling.py     # forward_returns / to_label — N거래일 후 상승(1)/하락(0) 이진 레이블
├── dataset.py      # build_training_dataset, chronological_split(embargo 포함), to_xy
├── training.py     # build_estimator(logreg/rf/dummy), train, evaluate, walk_forward_scores
├── artifact.py     # save_model/load_model — joblib + metadata.json 사이드카
└── predictor.py    # predict(bundle, records) -> MLPrediction, to_reasons(prediction, action)
```

`tests/ml_predictor/`에 각 모듈당 대응 테스트 파일(`conftest.py` 포함)을 둔다.

### 피처 (전부 시점 t 이하 데이터만, 전부 스케일 프리)

| 피처명 | 정의 | 왜 |
|---|---|---|
| `return_1d` | `close[t]/close[t-1] - 1` | 단기 모멘텀/반전 |
| `return_5d` | `close[t]/close[t-5] - 1` | 주간 모멘텀 |
| `return_20d` | `close[t]/close[t-20] - 1` | 중기 추세 |
| `rsi_14` | `rsi(closes)[t] / 100` | 과매수/과매도 |
| `macd_hist_norm` | `macd(closes)[t] / close[t]` | 추세 가속도(가격 정규화) |
| `close_to_sma20` | `close[t]/sma20[t] - 1` | 단기 이격도 |
| `close_to_sma60` | `close[t]/sma60[t] - 1` | 중기 이격도 |
| `sma20_to_sma60` | `sma20[t]/sma60[t] - 1` | 골든/데드크로스의 연속값 표현 |
| `atr_14_pct` | `atr(...)[t] / close[t]` | 변동성(사이징과 동일 지표 재사용) |
| `volume_ratio_20` | `volume[t] / mean(volume[t-19..t])` | 거래량 이상치 |
| `channel_position_20` | 20일 채널 내 종가 위치(0~1) | 박스권 내 위치 |

모든 피처를 비율/정규화값으로 만들어 KRX(원)와 NASDAQ(달러) 종목을 하나의 pooled 모델에 섞어도 스케일 왜곡이 없게 한다. 최장 워밍업은 SMA60(60거래일)이므로 오케스트레이터의 기존 `DEFAULT_LOOKBACK_DAYS = 120`을 그대로 쓸 수 있다 — 추론 경로 변경 불필요.

### 레이블 및 분할

- `LABEL_HORIZON_DAYS = 5`, `label[t] = 1 if close[t+5]/close[t] - 1 > 0 else 0`. 레이블이 없는 마지막 5개 행은 데이터셋에서 제거한다.
- 분할은 `test_ratio=0.2`의 시간 기준 분할 + `embargo_days=horizon` 갭. train 내부 검증은 `TimeSeriesSplit(n_splits=5)` walk-forward. **`train_test_split`/`KFold`/`shuffle=True`는 이 모듈에서 사용 금지.**
- 다종목 pooled 데이터셋은 종목별이 아니라 **전역 날짜 기준**으로 분할한다 (종목별 분할은 시간적 정보 누출을 일으킨다).
- 학습 유니버스는 상위 200종목 · 5년치로 제한한다(전 종목 학습은 별도 단계로 이연 — PRD §7.3/§11 TBD와 직결).

## 기존 파일 변경 (최소 침습)

| 파일 | 변경 내용 |
|---|---|
| `src/auto_stock/orchestrator/pipeline.py` | `run_recommendation_pipeline`에 `ml_model: ModelBundle \| None = None` 선택 인자 추가. 내부에 `_ml_reasons(ml_model, records, action)` 헬퍼(절대 raise 안 함)를 추가해 `generate_explanation(..., extra_reasons=...)`로 전달. 기본값 `None`이면 기존 동작과 완전히 동일해야 한다(회귀 없음) |
| `pyproject.toml` | `dependencies`에 `scikit-learn>=1.4`, `joblib`, `numpy` 추가 |
| `.gitignore` | `models/*.joblib` 추가 |
| `docs/PRD.md` | §5 "(구체 알고리즘/프레임워크 선정은 TBD)" 제거, §11 해당 체크박스를 확정 목록으로 이동 |
| `docs/IMPLEMENTATION_PLAN.md` | "신호 소스 확장" §1(ML #2)에 확정 내용 반영 |

**변경하지 않는 것**: `explainer/generator.py`, `rule_engine/*`, `risk_sizing/*`, `notifier/*`, `data/*`, `orchestrator/models.py`(`PipelineResult` 형태 유지).

## 재사용할 기존 코드/패턴

- `rule_engine/indicators.py`의 `sma/rsi/macd/atr` — pandas-ta 래퍼, `list[float | None]` 인덱스 정렬 반환 관례를 피처 계산에 그대로 재사용
- `data/models.py`의 `OHLCVRecord` — 유일한 입력 데이터 형태
- `data/service.py`의 `get_ohlcv(cache, ticker, start, end, market)` — 학습 스크립트의 데이터 수집에 재사용
- `orchestrator/pipeline.py`의 `DEFAULT_LOOKBACK_DAYS = 120` — 변경 불필요
- 테스트 패턴: `test_indicators.py`(수학적 극단 케이스로 계산 검증) + `test_engine.py`(계산을 모킹해 로직만 검증) 스타일을 `test_features.py`/`test_predictor.py`에 그대로 적용
- `scripts/run_recommendations.py`, `scan_nasdaq_top100_buy_only.py` 패턴 — 학습/실행 스크립트를 파이프라인 기본 동작과 분리하는 관례

## 구현 순서 (Phase)

**Phase 0 — 의존성 게이트**: `scikit-learn`/`joblib`/`numpy` 추가 후 기존 76개 테스트 전량 재실행. `pandas-ta` 0.3.x가 numpy 2.x와 충돌할 수 있는 알려진 리스크가 있어(`from numpy import NaN` 의존), 코드를 쓰기 전에 먼저 회귀를 확인한다. 깨지면 `numpy<2` 상한을 추가한다.

**Phase 1 — 데이터셋 기반** (모델 없이 단독으로 테스트 가능): `models.py` → `features.py`(+ 인과성/스케일불변성/워밍업 테스트) → `labeling.py` → `dataset.py`(시간순 분할 + embargo).

**Phase 2 — 학습 + 아티팩트**: `training.py`(logreg/rf/dummy 3종, walk-forward) → `artifact.py`(joblib+metadata.json) → `scripts/train_ml_model.py`(3종 평가 리포트 출력, logreg만 저장).

**Phase 3 — 추론 + 배선**: `predictor.py`(계수 기반 top-3 기여 피처, 동의/상충 문구 + "백테스트 검증 전 참고용" 고지) → `pipeline.py` 확장(`ml_model=None` 시 완전 무동작 보장) → `scripts/run_recommendations_with_ml.py`.

**Phase 4 — 문서화 + 리뷰**: 구현 완료 후 실제 결과를 기록하는 `docs/design/ml-predictor.md` 작성 → PRD/IMPLEMENTATION_PLAN.md 갱신 → `code-reviewer` 서브에이전트 리뷰(CRITICAL/HIGH 이슈 해소까지).

## 이번 스코프가 아닌 것 (명확한 경계)

이번엔 **모델 레벨 walk-forward 평가**(ROC-AUC, 베이스라인 대비, 5-fold)만 하고 **전략 레벨 백테스트**(거래비용/슬리피지/누적손익/MDD)는 하지 않는다.

| 이연 항목 | 왜 이번이 아닌가 |
|---|---|
| 5년 이상 전략 백테스트 | 손익 시뮬레이터가 아직 없다 |
| 백테스트 데이터 정합성 검증 (PRD §11 TBD) | 수정주가·액면분할·상장폐지 처리 검증 필요 |
| 생존편향 제거 | `get_universe()`가 오늘 시점 상장목록만 반환 — 학습셋이 낙관 편향돼 있음을 메타데이터에 명시만 함 |
| 4계층 앙상블 가중치 확정 (PRD §5 TBD) | 백테스트 없이 가중치를 정하면 근거 없는 숫자가 됨 |
| ML 단독 후보 생성/필터링 | 아래 승격 규칙을 모두 충족해야 재검토 |
| 딥러닝 고도화 | PRD 원칙상 "성능이 확인된 부분만 교체" — 확인 절차가 이번 단계 |
| 전 종목 실시간 스캔, 자동 재학습/드리프트 모니터링, 뉴스/펀더멘털 피처 | 각각 PRD §11 TBD / 별도 단계(#4) |

**승격 규칙 (사전 명문화)**: ML 신호가 보조 근거에서 후보 생성·필터링 권한으로 승격되려면 (1) 서로 겹치지 않는 3개 이상의 홀드아웃 기간에서 AUC가 일관되게 베이스라인 초과, (2) 거래비용을 반영한 전략 레벨 백테스트에서 규칙엔진 단독 대비 개선 입증, (3) 백테스트 데이터 정합성 검증(TBD) 해소, (4) 앙상블/가중치 방식 확정 — 이 네 가지를 **모두** 충족해야 한다.

## 검증 (Verification)

- Phase 0 직후: `pytest -q` (기존 76개 전량 통과 확인, 회귀 없음)
- Phase 1: `pytest tests/ml_predictor/test_features.py tests/ml_predictor/test_labeling.py tests/ml_predictor/test_dataset.py -v` — 인과성 속성 테스트, embargo 갭 테스트, 전역 날짜 분할 테스트 통과 확인
- Phase 2: `python scripts/train_ml_model.py` 실행해 3종 모델 평가 리포트(AUC, base rate)와 `models/ml_predictor/KRX_h5_logreg.joblib` + `.metadata.json` 생성 확인
- Phase 3: `pytest tests/ml_predictor/test_predictor.py tests/orchestrator/test_pipeline.py -v` — 특히 `ml_model=None` 시 기존 동작과 완전 동일함을 검증하는 회귀 테스트, ML 실패가 발송을 막지 않는 테스트
- 전체: `pytest --cov=src --cov-report=term-missing` (커버리지 80%+)
- 수동: `.venv/Scripts/python scripts/run_recommendations_with_ml.py` 실행해 텔레그램 메시지에 확률/기여피처/고지 문구가 포함되는지 육안 확인
- 마지막: `code-reviewer` 서브에이전트로 `pipeline.py` 변경분 포함 전체 diff 리뷰

## API 키

새로 필요한 것은 없다. scikit-learn/joblib/numpy 모두 로컬 라이브러리로 API 키가 필요 없다.
