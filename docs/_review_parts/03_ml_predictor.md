# `src/auto_stock/ml_predictor/` — ML 예측 모듈 (#2, 보조 신호) 상세 리뷰

## 0. 이 모듈의 위치와 역할

`ml_predictor`는 규칙엔진(`rule_engine`)이 이미 만들어낸 BUY/SELL `Candidate`에 대해 "앞으로 5거래일 뒤 종가가 오를 확률"을 계산해 **보조 근거 문장**을 하나 더 추가하는 모듈이다. 중요한 전제 두 가지를 먼저 짚어야 한다.

1. **ML은 자체적으로 매수/매도 후보를 만들지 않는다.** 규칙엔진이 이미 후보로 판단한 종목에 대해서만 ML 예측이 실행된다(`orchestrator/pipeline.py`가 `generate_candidates`로 후보를 만든 "다음"에 ML을 호출하는 순서로 배선돼 있다). ML이 알림 발송 여부를 바꾸지 않는다 — 이것은 PRD §10("보조 시그널로만 사용")과 직결된 설계 제약이다.
2. **ML 예측이 실패해도 추천 발송은 막히지 않는다.** 오케스트레이터의 `_ml_reasons` 헬퍼(`orchestrator/pipeline.py`)가 예외를 흡수하고 `PipelineResult.errors`에만 기록한다. `predictor.py` 자신도 데이터 부족 시 예외 대신 `None`을 반환하는 보수적 관례를 따른다.

모듈 구조는 `rule_engine/`과 동일한 패턴(models.py + 순수 계산 함수들 + 오케스트레이션 지점)을 따른다.

```
src/auto_stock/ml_predictor/
├── models.py       # 전부 frozen dataclass + slots
├── features.py     # OHLCV -> 11종 스케일 프리 피처
├── labeling.py     # N거래일 후 상승/하락 이진 레이블
├── dataset.py       # 피처+레이블 결합, 다종목 pooled, 시간순 분할(embargo)
├── training.py      # 추정기 팩토리 + 학습/평가/walk-forward
├── artifact.py       # joblib + metadata.json 저장/로드
└── predictor.py       # 추론 + 근거 문구 생성
```

참고 설계 문서: `docs/design/ml-predictor-plan.md`(구현 전 계획), `docs/design/ml-predictor.md`(구현 후 실제 결과 기록). 아래 설명은 이 두 문서와 실제 코드(2026-10-07 기준)를 직접 대조해서 작성했다.

---

## 1. lookahead bias(미래 데이터 누출) — 이 모듈 전체를 관통하는 핵심 설계 축

금융 시계열 ML에서 가장 흔하고 치명적인 실수가 lookahead bias다. 모델이 학습·검증 시점에 "미래에만 알 수 있는 정보"를 몰래 참조하면, 백테스트 성능은 비현실적으로 좋게 나오지만 실전에서는 전혀 작동하지 않는다. 이 모듈은 설계 문서(`ml-predictor-plan.md` 핵심 설계 결정 3)에서 "5중 방어"라고 명명한 다섯 겹의 구조적 방어를 두고 있다. 각 축이 어느 파일의 어느 코드에 대응하는지 미리 정리한다.

| # | 방어 축 | 구현 위치 | 핵심 아이디어 |
|---|---|---|---|
| 1 | 피처는 후행(trailing) 윈도우만 사용 | `features.py` 전체, `rule_engine/indicators.py` 재사용 | 시점 `t`의 피처값은 `t` 이하 데이터만으로 계산되며, `t+1` 이후 데이터를 추가해도 `t` 시점 피처값이 바뀌지 않는다("인과성(causality)" 속성) |
| 2 | 레이블은 `close[t+H]` 하나만 참조, 레이블 없는 꼬리 행 제거 | `labeling.py`, `dataset.py::build_samples` | 레이블 산출에 쓰는 미래 데이터는 정확히 "H거래일 후 종가" 하나뿐이고, 레이블을 매길 수 없는 마지막 H개 행은 데이터셋에서 완전히 제거된다 |
| 3 | 시간순 분할 + embargo 갭 (랜덤 분할 금지) | `dataset.py::chronological_split` | `train_test_split`/`KFold`/`shuffle=True`를 이 모듈에서 전혀 쓰지 않는다. 학습 구간은 항상 테스트 구간보다 과거이고, 그 경계에 레이블 호라이즌만큼의 완충 구간(embargo)을 둔다 |
| 4 | 다종목 데이터셋은 종목별이 아니라 **전역 날짜 기준**으로 분할 | `dataset.py::chronological_split` | 종목별로 따로 분할하면 "A종목의 미래 데이터가 B종목의 과거 데이터와 같은 학습 세트에 섞이는" 식의 교차 종목 시간 누출이 숨어버린다 |
| 5 | 스케일러를 `Pipeline` 안에 넣어 fold 밖 통계 누출 차단 | `training.py::build_estimator` | `StandardScaler`를 전체 데이터셋에 먼저 `fit`하고 나중에 train/test를 나누면, 테스트 구간의 평균/분산 정보가 학습 과정에 이미 새어 들어간 것이 된다. `Pipeline`으로 묶으면 매 분할·매 fold마다 "그 fold의 학습 데이터만으로" 스케일러가 다시 `fit`된다 |

이 다섯 축을 하나라도 깨면 조용히(에러 없이) 미래 정보가 새어 들어가고, 그 결과 모델이 실제로는 쓸모없는데도 검증 지표만 좋게 나오는 상황이 벌어진다. 아래 각 파일 설명에서 이 표의 번호를 다시 인용한다.

---

## 2. `models.py` — 도메인 모델

`rule_engine/models.py`와 동일한 관례로 전부 `@dataclass(frozen=True, slots=True)`다. 불변(frozen)으로 만드는 이유는 피처 벡터·레이블·메타데이터가 한번 계산되면 파이프라인을 통과하는 동안 절대 변경되지 않아야 하는 값이기 때문이다(실수로 다른 곳에서 값을 바꿔버리는 버그를 타입 시스템 차원에서 막는다).

### `FeatureVector`
```python
@dataclass(frozen=True, slots=True)
class FeatureVector:
    ticker: str
    market: str
    date: date
    values: dict[str, float]  # keyed by ml_predictor.features.FEATURE_NAMES
```
한 종목·한 날짜 시점의 11개 피처값을 담는다. `values`는 `FEATURE_NAMES`(아래 §3)에 정의된 이름을 키로 쓰는 딕셔너리다.

### `LabeledSample`
```python
@dataclass(frozen=True, slots=True)
class LabeledSample:
    feature: FeatureVector
    label: int  # 0 | 1
```
피처 벡터 하나 + 그 시점의 정답 레이블(1=상승, 0=하락) 쌍. 학습 데이터셋의 원자 단위.

### `TrainingDataset`
```python
@dataclass(frozen=True, slots=True)
class TrainingDataset:
    samples: list[LabeledSample]
    feature_names: list[str]
```
`LabeledSample`의 리스트 + 그 피처들의 순서가 보장된 이름 리스트. `feature_names`를 별도로 들고 다니는 이유는 `to_xy`(§5)가 딥셔너리 순회 순서에 의존하지 않고 항상 동일한 순서로 피처를 벡터화하기 위해서다.

### `DatasetSplit`
```python
@dataclass(frozen=True, slots=True)
class DatasetSplit:
    train: TrainingDataset
    test: TrainingDataset
```
`chronological_split`의 반환형. 학습용/테스트용 두 `TrainingDataset`을 묶는다.

### `ModelMetadata`
```python
@dataclass(frozen=True, slots=True)
class ModelMetadata:
    market: str
    algorithm: str
    horizon_days: int
    feature_names: list[str]
    universe_size: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_samples: int
    test_samples: int
    test_base_rate: float
    test_roc_auc: float
    test_accuracy: float
    trained_at: datetime
```
학습이 끝난 시점에 "이 모델이 언제, 어떤 데이터로, 어떤 성능으로 만들어졌는지"를 완전히 기록하는 감사(audit) 레코드다. 이 필드들은 그대로 `.metadata.json`으로 직렬화되어 git에 커밋된다(§6). 즉 "실제 바이너리 모델 파일은 git에 없지만, 그 모델이 어떤 조건에서 어떤 성능을 냈는지는 git 히스토리에 영구히 남는다"는 설계다.

### `ModelBundle`
```python
@dataclass(frozen=True, slots=True)
class ModelBundle:
    estimator: Any  # fitted sklearn Pipeline/estimator — untyped by design (3rd-party object)
    metadata: ModelMetadata
```
학습이 끝난 scikit-learn 추정기(또는 `Pipeline`) 객체 + 그 메타데이터를 하나로 묶는다. `estimator`의 타입을 `Any`로 둔 것은 의도적이다 — scikit-learn의 다양한 추정기 타입(logreg Pipeline, RandomForest, Dummy)을 하나의 타입으로 추상화하려 하지 않고, 서드파티 라이브러리 객체를 그대로 감싸는 쪽을 택했다.

### `MLPrediction`
```python
@dataclass(frozen=True, slots=True)
class MLPrediction:
    ticker: str
    market: str
    date: date
    probability_up: float
    top_features: list[tuple[str, float]]  # (feature_name, contribution), sorted by |contribution| desc
```
`predictor.predict()`의 반환형. 상승 확률 하나와, 그 확률에 가장 큰 영향을 준 피처 top-3(이름, 기여도)를 담는다.

---

## 3. `features.py` — OHLCV → 11종 스케일 프리 피처

### 설계 의도 (파일 docstring, `features.py:1-6`)

> "전부 비율/정규화값이라 KRX(원)와 NASDAQ(달러) 종목을 하나의 pooled 모델에 섞어도 스케일 왜곡이 없다."

이것이 이 파일 전체의 설계 원칙이다. 원화로 거래되는 삼성전자(수십만 원 단위)와 달러로 거래되는 애플(수백 달러 단위)의 "종가" 자체는 전혀 비교할 수 없는 숫자지만, "이격도(%)"나 "RSI(0~100)" 같은 비율/정규화 지표는 통화 단위와 무관하게 같은 스케일로 비교 가능하다. 이 설계 덕분에 KRX와 NASDAQ 데이터를 하나의 pooled 모델에 함께 넣어도 한쪽 시장의 절대 가격대가 모델을 왜곡시키지 않는다.

또한 `rule_engine.indicators`의 `sma`/`rsi`/`macd`/`atr`가 이미 **후행(trailing) 윈도우만 쓰는 causal 구현**이라는 점을 그대로 재사용한다(`indicators.py` 참고, §1의 방어 축 1). 즉 지표 계산 자체를 새로 만들지 않고 기존에 lookahead-safe하다고 검증된 코드를 재사용함으로써 새로운 버그가 들어올 여지를 줄였다.

### `FEATURE_NAMES` (상수, `features.py:12-24`)

```python
FEATURE_NAMES = [
    "return_1d", "return_5d", "return_20d",
    "rsi_14", "macd_hist_norm",
    "close_to_sma20", "close_to_sma60", "sma20_to_sma60",
    "atr_14_pct", "volume_ratio_20", "channel_position_20",
]
```

11개 피처의 전체 목록이며, 순서가 고정돼 있다는 점이 중요하다 — `to_xy`(dataset.py)가 이 리스트의 순서 그대로 피처 벡터를 만들기 때문에, 학습 시점과 추론 시점에 항상 같은 순서로 피처가 모델에 들어간다.

| 피처명 | 정의(수식) | 의미 |
|---|---|---|
| `return_1d` | `close[t]/close[t-1] - 1` | 1일 수익률(단기 모멘텀/반전) |
| `return_5d` | `close[t]/close[t-5] - 1` | 5일 수익률(주간 모멘텀) |
| `return_20d` | `close[t]/close[t-20] - 1` | 20일 수익률(중기 추세) |
| `rsi_14` | `RSI(14)[t] / 100` | 0~1로 정규화한 RSI(과매수/과매도) |
| `macd_hist_norm` | `MACD_hist[t] / close[t]` | 가격으로 정규화한 MACD 히스토그램(추세 가속도) |
| `close_to_sma20` | `close[t]/SMA20[t] - 1` | 20일 이동평균 대비 이격도 |
| `close_to_sma60` | `close[t]/SMA60[t] - 1` | 60일 이동평균 대비 이격도 |
| `sma20_to_sma60` | `SMA20[t]/SMA60[t] - 1` | 골든/데드크로스를 연속값으로 표현 |
| `atr_14_pct` | `ATR(14)[t] / close[t]` | 가격 대비 변동성(리스크사이징과 동일 지표 재사용) |
| `volume_ratio_20` | `volume[t] / mean(volume[t-19..t])` | 20일 평균 대비 당일 거래량 비율(거래량 이상치 탐지) |
| `channel_position_20` | 20일 채널(최고가~최저가) 내 종가 위치, 0~1 | 박스권 내 상대적 위치 |

### 헬퍼 함수

#### `_returns(closes: list[float], k: int) -> list[float | None]` (`features.py:30-36`)
`k`일 전 종가 대비 현재 종가의 변화율을 계산한다. 인덱스 `i < k`인 구간(워밍업 미충족)은 `None`. 분모(`prior`)가 0이면 `ZeroDivisionError` 대신 `None`을 반환하는 가드가 있다.

#### `_channel_position(highs, lows, closes, window) -> list[float | None]` (`features.py:39-48`)
윈도우 내 최고가(`channel_high`)와 최저가(`channel_low`) 사이에서 현재 종가가 상대적으로 어디에 위치하는지를 0~1 스칼라로 환산한다. `span == 0`(윈도우 내 가격이 전혀 움직이지 않은 극단적 경우)일 때는 `0.5`(중간)를 반환해 `ZeroDivisionError`를 피한다.

### `build_feature_vectors(records: list[OHLCVRecord]) -> list[FeatureVector | None]` (`features.py:51-96`)

**이 파일의 핵심 함수.** OHLCV 레코드 리스트를 입력받아 **인덱스가 입력과 정렬된(index-aligned)** `FeatureVector | None` 리스트를 반환한다.

동작 순서:
1. `records`가 빈 리스트면 즉시 빈 리스트 반환.
2. closes/highs/lows/volumes를 각각 리스트로 추출.
3. 11개 피처 각각을 전체 시계열에 대해 벡터화 계산(`_returns`, `rsi`, `macd`, `sma`(20/60), `atr`, `volume_ratio`, `_channel_position`). 이 단계에서 `rsi`/`macd`/`sma`/`atr`는 `rule_engine.indicators`를 그대로 호출한다.
4. 레코드 하나하나(인덱스 `i`)를 순회하며 11개 피처값을 딕셔너리로 모은다. 이때 각 피처의 분모가 0이거나(`sma_20[i] == 0` 등) 선행 지표 자체가 `None`(워밍업 미충족)이면 해당 피처값도 `None`이 된다.
5. **11개 피처 중 단 하나라도 `None`이면 그 인덱스 전체를 `None`으로 처리한다**(`if any(v is None for v in raw.values())`). 즉 "일부 피처는 있고 일부는 없는" 반쪽짜리 벡터는 절대 만들지 않는다 — 전부 있거나 전부 없거나(all-or-nothing).

가장 긴 워밍업 요구 지표가 SMA60(60거래일)이므로, 레코드가 60개 미만이면 모든 인덱스가 `None`이 된다. 이는 오케스트레이터의 기존 `DEFAULT_LOOKBACK_DAYS = 120`으로 충분히 커버되는 범위라서 추론 경로의 lookback 설정을 바꿀 필요가 없었다(설계 결정, `ml-predictor-plan.md` 참고).

**lookahead 방어와의 연결(방어 축 1)**: 인덱스 `i`의 피처값은 `records[0..i]` 범위의 데이터만으로 계산된다. `records[i+1]` 이후를 아무리 추가해도 인덱스 `i`의 피처값은 바뀌지 않는다 — 이것이 테스트(`test_feature_at_index_is_unchanged_when_future_bars_are_appended`, `ml-predictor.md:32` 인용)로 자동 검증되는 "인과성(causality)" 속성이다.

### `latest_feature_vector(records: list[OHLCVRecord]) -> FeatureVector | None` (`features.py:99-102`)

추론(실시간 예측) 경로 전용 함수. 내부적으로 `build_feature_vectors`를 그대로 호출하고 마지막 원소만 꺼낸다. **학습용 함수를 그대로 재사용**하는 이유는 "학습 시점의 피처 계산 로직"과 "추론 시점의 피처 계산 로직"이 갈라지면 그 둘 사이의 미묘한 불일치(train-serving skew)가 생길 수 있기 때문이다 — 코드를 공유함으로써 이 위험을 구조적으로 제거했다.

---

## 4. `labeling.py` — N거래일 후 상승/하락 이진 레이블

파일 docstring(`labeling.py:1-5`)이 명시하듯, "레이블은 반드시 `close[t+H]` 하나의 값만 참조해야 한다 — 인덱스 경계를 한 칸이라도 잘못 잡으면 조용한 lookahead가 되므로" 테스트로 경계를 못박아 검증한다(방어 축 2).

### 상수
- `LABEL_HORIZON_DAYS = 5` — "5거래일 후"를 예측 지평으로 고정.
- `LABEL_THRESHOLD = 0.0` — 수익률이 정확히 0을 초과하면 상승(1), 그 외(0 이하)는 하락(0).

### `forward_returns(closes: list[float], horizon: int = LABEL_HORIZON_DAYS) -> list[float | None]` (`labeling.py:11-17`)

```python
def forward_returns(closes, horizon=LABEL_HORIZON_DAYS):
    n = len(closes)
    out = [None] * n
    for i in range(n - horizon):
        base = closes[i]
        out[i] = None if base == 0 else closes[i + horizon] / base - 1
    return out
```

인덱스 `i`에 대해 "`horizon`일 후 수익률" `closes[i+horizon]/closes[i] - 1`을 계산한다. 루프 범위가 `range(n - horizon)`이므로 **마지막 `horizon`개 인덱스는 애초에 루프에 들어가지 않아 자동으로 `None`으로 남는다** — "레이블 없는 마지막 5개 행"이 이 함수 레벨에서부터 구조적으로 보장된다. `base == 0`일 때 `ZeroDivisionError`를 피하는 가드도 있다.

### `to_label(forward_return: float | None, threshold: float = LABEL_THRESHOLD) -> int | None` (`labeling.py:20-23`)

`forward_return`이 `None`이면 `None`을 그대로 전달(레이블 없음을 전파). 그 외에는 `forward_return > threshold`면 1(상승), 아니면 0(하락). 즉 **"보합(0% 변화)"은 하락(0)으로 분류**된다는 점에 유의 — 임계값이 "초과"(`>`)이지 "이상"(`>=`)이 아니다.

---

## 5. `dataset.py` — 피처+레이블 결합, 다종목 pooled 데이터셋, 시간순 분할

파일 docstring(`dataset.py:1-7`)이 이 파일의 존재 이유를 명확히 말한다:

> "`train_test_split`/`KFold`/`shuffle=True`는 이 모듈에서 절대 사용하지 않는다 — 랜덤 분할은 미래 정보가 학습 세트로 새는 것을 감출 수 있어 금융 시계열에서 금지된다. 다종목 데이터셋은 종목별이 아니라 전역 날짜 기준으로 분할한다: 종목별 분할은 한 종목의 미래가 다른 종목의 과거와 뒤섞여도 감지되지 않는 시간적 정보 누출을 낳는다."

### `build_samples(records: list[OHLCVRecord]) -> list[LabeledSample]` (`dataset.py:15-30`)

단일 종목 레코드 리스트를 받아 "워밍업 구간(피처가 `None`인 구간)"과 "레이블 없는 마지막 `horizon`개 행"을 **모두** 제거한 `LabeledSample` 리스트를 만든다. `build_feature_vectors`와 `forward_returns`를 각각 호출한 뒤 `zip`으로 짝지어, 둘 중 하나라도 `None`인 인덱스는 건너뛴다(`continue`). 레코드 수가 `LABEL_HORIZON_DAYS`보다 적으면 빈 리스트가 된다.

### `build_training_dataset(records_by_ticker: list[list[OHLCVRecord]]) -> TrainingDataset` (`dataset.py:33-40`)

여러 종목의 레코드 리스트(리스트의 리스트)를 받아 각 종목에 `build_samples`를 적용한 뒤 전부 하나의 리스트로 합친다(pooled). 그 다음 **`(날짜, 티커)` 튜플 기준으로 정렬**한다 — 날짜가 1차 키이므로 결과적으로 전체 데이터셋이 시간순으로 정렬된 상태가 된다. 이 정렬이 바로 뒤의 `chronological_split`이 "앞쪽은 과거, 뒤쪽은 미래"라는 전제를 가질 수 있게 하는 전제조건이다.

### `chronological_split(dataset, test_ratio=0.2, embargo_days=None) -> DatasetSplit` (`dataset.py:43-71`)

**이 모듈에서 가장 중요한 함수.** 동작을 단계별로 보면:

1. `embargo_days`가 명시되지 않으면 `LABEL_HORIZON_DAYS`(5)로 기본값을 채운다. 즉 기본 embargo는 레이블 호라이즌과 항상 같다 — 이것이 왜 중요한지는 바로 아래에서 설명한다.
2. 데이터셋에 등장하는 **고유 날짜**(`unique_dates`)를 정렬해 뽑는다. 분할은 "표본 개수" 기준이 아니라 "날짜" 기준으로 이루어진다 — 같은 날짜에 여러 종목의 표본이 있어도 그 날짜는 통째로 train 또는 test 한쪽에만 속한다(방어 축 4, 전역 날짜 분할).
3. `split_idx = int(n_dates * (1 - test_ratio))`로 분할 지점을 계산하고, 최소 1개/최대 `n_dates-1`개로 clamp한다.
4. `embargo_start_idx = split_idx - embargo_days`로 **embargo가 시작되는 지점**을 분할 지점보다 `embargo_days`만큼 앞으로 당긴다.
5. `train_dates`는 `unique_dates[:embargo_start_idx]`(즉 분할 지점보다 훨씬 이전의 날짜들만), `test_dates`는 `unique_dates[split_idx:]`(분할 지점 이후). **`embargo_start_idx`와 `split_idx` 사이의 날짜들은 train에도 test에도 들어가지 않고 완전히 버려진다** — 이게 "embargo 갭"이다.

**왜 embargo가 필요한가(방어 축 3)**: 레이블이 `close[t+5]`를 참조하므로, 날짜 `t`의 표본은 사실상 "5일 뒤의 미래 정보"를 레이블 안에 담고 있다. 만약 train의 마지막 날짜가 test의 첫 날짜 바로 하루 전이라면, train의 마지막 표본들의 레이블이 참조하는 미래 종가가 test 구간 안에 있는 날짜일 수 있다 — 이는 train 레이블이 test 구간의 가격 정보를 암묵적으로 포함하는 셈이 되어 누출이다. `embargo_days`를 레이블 호라이즌과 같게 둠으로써 "train의 가장 늦은 표본이 참조하는 미래 날짜가 test 시작 이전에 끝나도록" 보장한다.

6. train/test 각각을 다시 `(날짜, 티커)`로 정렬해 `TrainingDataset`으로 감싸 반환.

### `to_xy(dataset: TrainingDataset) -> tuple[list[list[float]], list[int]]` (`dataset.py:74-77`)

`dataset.feature_names`(즉 `FEATURE_NAMES`) 순서대로 각 표본의 피처값을 뽑아 2차원 리스트 `X`를 만들고, 레이블 리스트 `y`를 만든다. scikit-learn의 `estimator.fit(X, y)` 호출에 바로 넣을 수 있는 형태다.

---

## 6. `training.py` — 추정기 팩토리 + 학습/평가/walk-forward 검증

파일 docstring(`training.py:1-7`)의 핵심 문장: "`StandardScaler`는 반드시 `Pipeline` 안에 넣는다 — fold/split 밖에서 전체 데이터로 스케일러를 fit하면 테스트 구간의 분포 정보가 학습에 새어 들어간다(lookahead 방어 5중 축 중 하나)."

### `build_estimator(algorithm: Literal["logreg", "rf", "dummy"]) -> Any` (`training.py:30-47`)

세 가지 추정기 중 하나를 생성하는 팩토리.

- **`"logreg"`**: `Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE))])`. **이것이 프로덕션에 실제로 배선되는 유일한 추정기다.** 코드 주석(`training.py:36-38`)에 scikit-learn 버전 호환성 메모가 있다: "L2는 이미 scikit-learn의 기본값이다. `penalty="l2"`를 명시적으로 넘기면 1.8+에서 `FutureWarning`이 뜨고 1.10에서는 제거된다" — 그래서 `penalty` 인자를 **의도적으로 생략**했다(명시하지 않아도 동작은 동일, 다만 미래 버전과의 호환성을 미리 확보).
- **`"rf"`**: `RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE)`. **평가 리포트 전용 벤치마크** — `scripts/train_ml_model.py`가 세 알고리즘을 모두 돌려 성능을 비교 출력하지만, `predictor.py`가 실제로 로드해서 쓰는 모델은 logreg뿐이다.
- **`"dummy"`**: `DummyClassifier(strategy="prior")`. 데이터의 클래스 비율만으로 예측하는 베이스라인 — "아무 패턴도 안 쓰고 그냥 다수 클래스로만 찍었을 때" 수준을 보여주는 하한선 역할.
- 알 수 없는 알고리즘 문자열이 들어오면 `ValueError`.

**RandomForest를 프로덕션에 쓰지 않는 이유**: `ml-predictor-plan.md`(핵심 설계 결정 1)에 명문화된 승격 규칙이 있다 — "RandomForest가 walk-forward AUC 중앙값 기준 LogReg 대비 +0.02 이상 우세할 때만 승격을 재검토한다"는 기준을 **미리** 못박아, 나중에 결과를 보고 나서 기준을 만드는 사후 정당화(post-hoc rationalization)를 방지했다. LogisticRegression이 선택된 근본 이유는 PRD §5가 요구하는 Explainability(설명 가능성)다 — 계수(coefficient) 기반으로 "왜 이 예측이 나왔는지"를 분해할 수 있는 건 선형모델뿐이다(§8의 `_top_contributing_features` 참고).

### `evaluate(estimator: Any, dataset: TrainingDataset) -> dict[str, float]` (`training.py:50-65`)

주어진(이미 학습된) 추정기를 데이터셋에 대해 평가해 `{"roc_auc": ..., "accuracy": ..., "base_rate": ...}`를 반환한다.

- `base_rate`: 데이터셋에서 레이블이 1(상승)인 비율. "아무 모델도 없이 그냥 항상 다수쪽을 찍었을 때"의 기준점.
- `accuracy`: `estimator.predict(X)`와 실제 `y`의 일치율.
- `roc_auc`: `estimator.predict_proba(X)[:, 1]`(상승 확률)과 `y`로 계산. **단일 클래스만 존재하는 홀드아웃(degenerate holdout)에서는 `roc_auc_score`가 `ValueError`를 던지는데, 이 경우를 잡아서 `float("nan")`으로 대체**한다 — 매우 작은 홀드아웃(예: 소규모 스모크 테스트)에서 흔히 발생하는 엣지케이스다.
- 데이터셋이 비어 있으면(`X`가 빈 리스트) 셋 다 `nan`을 즉시 반환.

### `train(split: DatasetSplit, market: str, universe_size: int, algorithm: Algorithm = "logreg") -> ModelBundle` (`training.py:68-94`)

이 모듈의 "진짜 학습" 함수. 흐름:
1. `build_estimator(algorithm)`으로 추정기 생성.
2. `split.train`을 `to_xy`로 변환해 `estimator.fit(X_train, y_train)` 호출.
3. `split.test`에 대해 `evaluate` 호출해 성능 지표 획득.
4. `split.train`/`split.test` 각각의 최소/최대 날짜를 뽑아 `train_start`/`train_end`/`test_start`/`test_end`로 기록.
5. 위 모든 정보를 `ModelMetadata`로 조립하고, `trained_at=datetime.now()`로 학습 시각을 타임스탬프.
6. `ModelBundle(estimator=estimator, metadata=metadata)`를 반환.

이 함수가 `scripts/train_ml_model.py`에서 호출되어 실제로 저장될 모델을 만들어낸다(§7).

### `walk_forward_scores(dataset, n_splits=5, algorithm="logreg") -> list[float]` (`training.py:97-122`)

**"train의 성능이 우연이 아니었는지"를 추가로 검증하는 보조 평가.** `chronological_split`이 만든 train 데이터셋 **내부**를 다시 `TimeSeriesSplit(n_splits=5)`로 5개의 fold로 나눠, 매 fold마다 "그 fold 이전 데이터로 학습 → 그 fold로 평가"하는 식으로 AUC를 5번 측정한다.

핵심 포인트:
- `TimeSeriesSplit`은 scikit-learn이 제공하는 시계열 전용 분할기로, 항상 "과거로 학습, 미래로 평가"를 보장한다(일반 `KFold`와 달리 랜덤 셔플이 없다) — 방어 축 3의 연장선.
- 각 fold 평가 전에 다시 `build_estimator(algorithm)`으로 **새 추정기를 생성**한다 — 즉 5개 fold가 서로 다른, 독립적으로 학습된 모델이다. 이는 `Pipeline` 안의 `StandardScaler`도 매 fold마다 그 fold의 학습 데이터로만 다시 `fit`된다는 뜻이다(방어 축 5가 여기서도 다시 적용).
- 홀드아웃(`y_test`)이 단일 클래스뿐인 fold는 AUC가 정의되지 않으므로 건너뛴다(`continue`).
- 데이터 길이가 `n_splits`보다 적으면 애초에 분할이 불가능하므로 빈 리스트를 즉시 반환.
- `tqdm`으로 진행 상황을 출력(배치 스크립트에서 수백 종목·수년치 데이터를 다룰 때 체감 대기시간이 길기 때문).

이 함수의 결과는 모델 저장에 쓰이지 않고 `scripts/train_ml_model.py`가 콘솔에 리포트만 출력하는 용도다 — "test 구간 단일 분할 AUC"뿐 아니라 "학습 구간 내부에서의 반복 검증 AUC"까지 함께 보여줘서 성능이 운이 아닌지 가늠하게 해준다.

---

## 7. `scripts/train_ml_model.py` — 학습 실행 진입점

이 모듈의 유일한 실행 스크립트. 파일 docstring(`train_ml_model.py:1-18`)에 중요한 아키텍처 결정이 적혀 있다.

> "2026-09-27부로 KRX/NASDAQ 둘 다 순회하며 시장별로 독립된 아티팩트를 학습·저장한다 ... 두 시장을 하나로 합쳐 학습하지 않는다: 거래시간·업종 구성·규제가 근본적으로 달라 계수(coefficient)가 '두 시장 평균 효과'로 뭉개지면 PRD §5가 LogisticRegression을 고른 이유인 Explainability(피처별 기여도 해석)가 훼손된다(사용자 확인, 2026-09-27)."

즉, "피처는 통화 단위에 무관하게 스케일 프리로 설계해 KRX+NASDAQ을 하나로 pool할 수 있다"는 `features.py`의 설계와는 별도로, **실제 운영상의 선택은 "시장별로 독립된 모델을 학습한다"는 쪽으로 확정됐다.** 이유는 두 시장의 평균 효과가 뭉개지면 계수 해석(이 모델을 쓰는 핵심 이유)이 무의미해지기 때문이다. 이 결정은 `save_model`/`load_model`이 이미 시장별로 아티팩트 경로를 분리하는 구조였기 때문에(§8), 학습 스크립트만 시장을 순회하도록 바꾸면 충분했다.

**market 세분화 확장(이번 세션) 이후에도 이 원칙은 그대로 유지되고, 오히려 한 번 더 명문화됐다.** 파일 docstring(`train_ml_model.py:15-18`)에 다음이 추가됐다:

> "market 세분화(KOSPI/KOSDAQ/NASDAQ/NYSE) 확장 후에도 이 원칙은 그대로 유지된다 — `MARKETS`는 원자 거래소 4개만 순회한다. 'KRX'(코스피+코스닥)/'US'(나스닥+NYSE) 같은 합성 버킷을 학습하면 바로 위에서 설명한 '두 시장 평균효과로 계수가 뭉개지는' 문제가 그대로 재발하므로, 합성 버킷은 절대 이 스크립트의 학습 대상에 들어가면 안 된다."

즉 이번 세션의 market 세분화 작업(`data/models.py::VALID_MARKETS`가 `{"KRX","NASDAQ"}` 2개 원자값에서 `{"KOSPI","KOSDAQ","NASDAQ","NYSE"}` 4개 원자값으로 바뀐 것, 상세는 `docs/_review_parts/01_data_layer.md`/`05_orchestrator_and_synthesis.md` 참고)는 "두 시장을 독립 학습한다"는 기존 원칙을 뒤집은 게 아니라, **독립 학습 단위를 2개에서 4개로 늘렸을 뿐**이다 — "합성해서 학습하지 않는다"는 핵심 설계 원칙은 전혀 바뀌지 않았고, 오히려 "KRX"/"US" 같은 합성 scan 버킷이 새로 생기면서 그걸 학습에 섞으면 안 된다는 금지 규칙이 명시적으로 추가됐다.

ml_predictor 패키지 자체(`models.py`/`features.py`/`labeling.py`/`dataset.py`/`training.py`/`predictor.py`)의 코드는 이번 market 세분화 작업에서 **전혀 수정되지 않았다** — 이 모듈들은 market을 아티팩트 경로 분리용 문자열로만 다루고 그 값의 집합이 몇 개인지는 신경 쓰지 않으므로, 원자 market이 2개에서 4개로 늘어나도 코드 변경이 필요 없었다. 이 문서의 §1~§6, §9(피처 설계·lookahead 방어·학습·추론)는 이번 세션과 무관하게 그대로 유효하다.

### 상수
- `MARKETS = ["KOSPI", "KOSDAQ", "NASDAQ", "NYSE"]`(이번 세션에서 `["KRX", "NASDAQ"]`에서 변경 — 원자 거래소 4개를 각각 독립 학습) — `train_ml_model.py:39`
- `UNIVERSE_SIZE = int(os.environ.get("ML_TRAIN_UNIVERSE_SIZE", 200))` — 환경변수로 재정의 가능, 기본 상위 200종목.
- `LOOKBACK_YEARS = int(os.environ.get("ML_TRAIN_LOOKBACK_YEARS", 5))` — 기본 5년치.
- `REPORT_ALGORITHMS = ("logreg", "rf", "dummy")` — 평가 리포트에 셋 다 출력.
- `PRODUCTION_ALGORITHM = "logreg"` — 실제로 저장되는 건 이것 하나뿐.

### `_collect_universe_records(cache, tickers, start, end, market) -> list[list]` (`train_ml_model.py:41-56`)

유니버스(종목 리스트)를 순회하며 `get_ohlcv`로 각 종목의 OHLCV를 수집한다. **종목 단위로 예외를 격리**한다 — 한 종목에서 예외가 나도(`except Exception`) 그 종목만 스킵하고 나머지는 계속 수집한다(오케스트레이터의 종목 단위 격리 원칙과 동일, `train_ml_model.py:52` 주석에 명시). `tqdm` progress bar로 로드/스킵 개수를 실시간 표시.

### `_train_one_market(cache, market) -> None` (`train_ml_model.py:59-99`)

한 시장에 대한 전체 학습 파이프라인:
1. `get_universe(market)[:UNIVERSE_SIZE]`로 대상 종목 추출.
2. 오늘부터 `LOOKBACK_YEARS`년 전까지의 기간을 계산.
3. `_collect_universe_records`로 전 종목 OHLCV 수집.
4. `build_training_dataset` → `chronological_split`으로 데이터셋 구성.
5. train/test 어느 쪽이든 비어 있으면("표본이 부족합니다") 경고 출력 후 그 시장은 건너뛴다(`return`) — 이 경우 **그 시장의 기존 아티팩트는 덮어쓰지 않고 그대로 남는다.**
6. `REPORT_ALGORITHMS` 3종 각각에 대해 `build_estimator` → `fit` → `evaluate` → `walk_forward_scores`를 돌려 `test AUC / accuracy / base_rate / walk-forward AUC(5-fold 평균)`를 콘솔에 출력.
7. `PRODUCTION_ALGORITHM`(logreg)로 `train()` 호출 후 `save_model()`로 저장.

### `main()` (`train_ml_model.py:102-109`)
`OHLCVCache("data/ohlcv.duckdb")`를 열고 `MARKETS`를 순회하며 `_train_one_market`을 호출한다.

**스모크 테스트 실행법**(스크립트 docstring 인용): `ML_TRAIN_UNIVERSE_SIZE=5 ML_TRAIN_LOOKBACK_YEARS=1 .venv/Scripts/python scripts/train_ml_model.py`로 소규모·단기간 데이터로 빠르게 전체 파이프라인을 왕복 검증할 수 있다.

---

## 8. `artifact.py` — 모델 저장/로드 (joblib + metadata.json)

파일 docstring(`artifact.py:1-6`): "`.joblib`은 `data/*.duckdb`와 같은 이유로 커밋하지 않는다(.gitignore). 대신 메타데이터 JSON은 커밋해 '어떤 모델이 어떤 성능으로 배포됐는지'를 git 히스토리에 남긴다."

### `artifact_path(market: str, model_dir=DEFAULT_MODEL_DIR) -> Path` (`artifact.py:25-28`)

파일명 패턴은 `{market}_h{LABEL_HORIZON_DAYS}_logreg.joblib`(예: `KRX_h5_logreg.joblib`). 코드 주석에 "알고리즘(logreg)과 호라이즌(5)은 파라미터화하지 않고 파일명에 고정했다 — 프로덕션에 저장되는 건 오직 logreg뿐이기 때문"이라는 설명이 있다. `DEFAULT_MODEL_DIR = Path("models/ml_predictor")`.

### `save_model(bundle: ModelBundle, model_dir=DEFAULT_MODEL_DIR) -> Path` (`artifact.py:35-48`)

1. 모델 디렉터리를 (없으면) 생성.
2. `joblib.dump(bundle.estimator, path)`로 추정기 바이너리 저장.
3. `bundle.metadata`를 `asdict()`로 딕셔너리화하고, `date`/`datetime` 타입 필드는 `.isoformat()` 문자열로 변환(JSON은 날짜 타입을 직접 지원하지 않으므로).
4. 메타데이터를 `.metadata.json`(같은 파일명에서 확장자만 바꾼 경로)로 pretty-print(`indent=2`) 저장.

### `load_model(market: str, model_dir=DEFAULT_MODEL_DIR) -> ModelBundle` (`artifact.py:51-86`)

1. `.joblib`/`.metadata.json` 둘 다 존재하는지 확인 — 하나라도 없으면 `ModelArtifactError`를 던지며 "`scripts/train_ml_model.py`를 먼저 실행하세요"라는 안내 메시지를 포함한다.
2. `joblib.load(path)`로 추정기 복원. 코드 주석(`artifact.py:64-66`)에 보안 관련 설명이 있다: "`joblib.load`는 pickle 기반 역직렬화를 실행하는데, 이 아티팩트가 오직 `scripts/train_ml_model.py`에 의해 같은 로컬 저장소/파일시스템에서만 생성되고 네트워크 다운로드나 사용자 업로드를 거치지 않기 때문에 여기서는 안전하다." (pickle 역직렬화는 일반적으로 임의 코드 실행 위험이 있는 패턴이지만, 신뢰할 수 있는 로컬 생성 아티팩트에만 적용되므로 여기서는 허용 가능하다는 명시적 판단이다.)
3. JSON을 읽어 `ModelMetadata`로 재구성(날짜 문자열을 `date.fromisoformat`/`datetime.fromisoformat`으로 역변환).
4. `ModelBundle(estimator=estimator, metadata=metadata)` 반환.

`ModelArtifactError`를 던지는 지점이 바로 상위 호출자(`scripts/chat_cli.py`의 `_load_ml_models`, `scripts/run_recommendations_with_ml.py` 등)가 "모델이 없으면 해당 신호만 꺼진 채로 계속 진행"하는 우아한 저하(graceful degradation) 로직의 트리거가 된다.

---

## 9. `predictor.py` — 추론 + 근거 문구 생성

파일 docstring(`predictor.py:1-6`): "`predict`는 데이터 부족 시 예외 대신 `None`을 반환한다 — rule_engine의 '후보 없음' 관례와 동일한 보수적 원칙이다. `to_reasons`가 만드는 문구는 PRD §10 '보조 시그널로만 사용' 원칙에 따라 항상 백테스트 미검증 고지를 포함한다."

### 상수
- `TOP_FEATURE_COUNT = 3` — 기여 피처 상위 3개만 노출.
- `ML_AGREE_THRESHOLD = 0.55` — 상승확률이 이 값 이상이면 "상승 예측"으로 간주.
- `ML_CONFLICT_THRESHOLD = 0.45` — 상승확률이 이 값 이하이면 "하락 예측"으로 간주.
- (0.45~0.55 사이는 "중립" — 이 구간에서는 ML이 뚜렷한 방향성을 제시하지 않는다고 처리)
- `DISCLAIMER = "(ML 신호는 백테스트 검증 전 참고용 보조 지표입니다)"`

### `_top_contributing_features(estimator, feature_names, vector) -> list[tuple[str, float]]` (`predictor.py:20-40`)

**Explainability(설명 가능성)의 실제 구현.** 선형모델(`StandardScaler`+`LogisticRegression`을 담은 `Pipeline`)에서만 작동한다.

1. `estimator.named_steps`가 없으면(비선형 추정기, 예: RandomForest) 즉시 빈 리스트 반환 — "해석 가능한 분해가 없다"는 뜻.
2. `named_steps["scaler"]`와 `named_steps["clf"]`를 꺼내고, `clf`에 `coef_` 속성이 없으면(선형모델이 아님) 역시 빈 리스트.
3. 입력 피처값을 `scaler.transform()`으로 **표준화(standardize)**한다 — 로지스틱 회귀의 계수(`coef_`)는 표준화된 입력 기준으로 학습됐으므로, 기여도를 올바르게 분해하려면 원본 피처값이 아니라 표준화된 값을 써야 한다.
4. 각 피처에 대해 `기여도 = coef_[i] * 표준화된_값[i]`를 계산 — 이것이 로지스틱 회귀의 선형 결합(logit) 안에서 그 피처가 실제로 차지하는 수치적 비중이다.
5. `abs(기여도)` 기준 내림차순 정렬 후 상위 3개만 반환.

이 함수가 바로 `models.py`의 docstring이 가리키는 "선형모델이면 `coef_ × 표준화값`으로 top-3 기여 피처 산출"의 실제 구현이며, PRD §5.1이 요구하는 "왜 63%인지"에 대한 답을 제공하는 지점이다.

### `predict(bundle: ModelBundle, records: list[OHLCVRecord]) -> MLPrediction | None` (`predictor.py:43-59`)

1. `latest_feature_vector(records)`로 가장 최근 시점 피처를 뽑는다. `None`이면(워밍업 미충족) 즉시 `None` 반환 — 여기서 **예외를 던지지 않는다.**
2. `bundle.metadata.feature_names`(즉 저장 시점의 `FEATURE_NAMES` 순서)를 그대로 사용해 피처값을 벡터화 — 모델이 학습된 순서와 추론 시 순서가 항상 일치하도록 보장.
3. `estimator.predict_proba(row)[0][1]`로 "상승(클래스 1)" 확률을 얻는다.
4. `_top_contributing_features`로 기여 피처 계산.
5. `MLPrediction`으로 묶어 반환.

### `_stance(prediction: MLPrediction, action: str) -> str` (`predictor.py:62-74`)

규칙엔진이 내린 `action`(BUY/SELL)과 ML 예측이 **동의**하는지, **상충**하는지, 아니면 **중립**인지를 판단해 한국어 문구를 만든다.

- `predicts_up = probability_up >= 0.55`, `predicts_down = probability_up <= 0.45`
- `agrees`: (BUY이고 상승예측) 또는 (SELL이고 하락예측) → `"ML 모델(상승확률 {pct}%)도 {action} 신호에 동의합니다"`
- `conflicts`: (BUY인데 하락예측) 또는 (SELL인데 상승예측) → `"ML 모델(상승확률 {pct}%)은 {action} 신호와 상충됩니다 — 주의"`
- 그 외(중립 구간, 0.45 < p < 0.55) → `"ML 모델 상승확률 {pct}% (중립)"`

### `to_reasons(prediction: MLPrediction | None, action: str) -> list[str]` (`predictor.py:77-86`)

오케스트레이터가 `extra_reasons`로 쓸 최종 문구 리스트를 만든다.

1. `prediction`이 `None`이면 빈 리스트(ML 근거 없음).
2. `_stance`로 동의/상충/중립 문구 1줄.
3. `top_features`가 있으면 `"주요 기여 피처: a, b, c"` 1줄 추가.
4. **항상 마지막에** `DISCLAIMER`를 추가 — 이 고지 문구는 조건 없이 항상 붙는다(PRD §10 "보조 시그널로만 사용" 원칙의 코드 레벨 강제).

---

## 10. `explainer/generator.py`와의 연결 (오케스트레이터 통합 요약)

`ml_predictor` 자체는 오케스트레이터 코드를 포함하지 않는다 — 실제 호출은 `orchestrator/pipeline.py`에서 일어난다(상세는 orchestrator 섹션 참고). 여기서는 ML 모듈 입장에서 알아야 할 계약만 요약한다.

- `pipeline.py`는 `ml_model: ModelBundle | None = None` 선택 인자를 받는다.
- `ml_model=None`(기본값)이면 ML 코드 경로를 **전혀 타지 않는다** — `generate_explanation(candidate, sizing)`를 기존과 동일하게 2개 위치 인자만으로 호출한다. 이는 ML 도입 전 코드와 100% 동일한 동작을 보장하기 위한 회귀 방지 장치다.
- `ml_model`이 주어지면 내부 헬퍼 `_ml_reasons(ml_model, records, action)`가 `predict`/`to_reasons`를 호출해 결과를 `extra_reasons`로 전달한다. 이 헬퍼는 **절대 예외를 밖으로 던지지 않는다** — ML 예측이 실패해도 `(ticker, "ML 예측 실패: ...")` 형태로 `PipelineResult.errors`에 기록될 뿐, 알림 발송 자체는 계속된다.
- `explainer/generator.py`는 ML 도입 과정에서 **한 줄도 수정되지 않았다** — 애초에 `extra_reasons`라는 확장 포인트를 염두에 두고 설계된 덕분이다.

---

## 11. 현재 한계 (2026-10-07 기준, market 세분화 반영)

- **`MARKETS`는 이제 `["KOSPI", "KOSDAQ", "NASDAQ", "NYSE"]` 원자 거래소 4개를 순회하지만, 실제로 학습이 완료되어 커밋된 아티팩트는 `models/ml_predictor/KRX_h5_logreg.metadata.json` 하나뿐이다.** `models/ml_predictor/` 디렉터리를 Glob으로 직접 확인한 결과 이 파일 외에는 아무것도 없다.
- **중요: 이 "KRX" 아티팩트는 market 세분화 이후 더 이상 유효한 market 이름이 아니다.** `data/models.py::VALID_MARKETS`가 `{"KOSPI","KOSDAQ","NASDAQ","NYSE"}` 원자값으로 바뀌면서 "KRX"는 `OHLCVRecord.market`에 쓸 수 없는 합성 scan 버킷 전용 값이 됐다(`orchestrator/market_buckets.py`). `ml_predictor/artifact.py::artifact_path(market)`는 `{market}_h{LABEL_HORIZON_DAYS}_logreg.joblib` 패턴으로 market 이름을 파일명에 그대로 쓰므로, `KRX_h5_logreg.metadata.json`은 이제 `scripts/train_ml_model.py`의 신규 `MARKETS` 리스트(`["KOSPI","KOSDAQ","NASDAQ","NYSE"]`)에 더 이상 등장하지 않는 레거시 아티팩트가 됐다 — **KOSPI와 KOSDAQ 각각에 대한 재학습이 필요하며, 아직 실행되지 않았다**(이번 세션 범위 밖). `chat_agent/tools.py`/`scripts/chat_cli.py`가 `ml_models.get(market)`으로 조회할 때 market이 "KOSPI"/"KOSDAQ"이면 이 레거시 파일을 찾지 못해 `None`(ML 신호 비활성)으로 저하된다 — 에러는 아니지만 재학습 전까지는 KRX 계열 종목에 대한 ML 보조신호가 조용히 꺼진 상태라는 뜻이다.
- NASDAQ/NYSE 모델도 마찬가지로 아직 학습된 적이 없다(이 한계는 세분화 이전부터 있던 것과 동일 — NASDAQ 쪽 유니버스/데이터 소스가 정상 동작하는 환경에서 `scripts/train_ml_model.py`를 실행하면 된다, 코드 추가 작업은 필요 없음).
- **실제 학습 결과 수치** (`models/ml_predictor/KRX_h5_logreg.metadata.json`, 2026-09-28 학습):
  - `market: KRX`, `algorithm: logreg`, `horizon_days: 5`
  - `universe_size: 200`종목
  - `train_start: 2021-12-22` ~ `train_end: 2025-09-24` (`train_samples: 166034`건)
  - `test_start: 2025-10-02` ~ `test_end: 2026-09-16` (`test_samples: 44369`건)
  - `test_base_rate: 0.4678`(테스트 구간에서 실제 상승 비율 약 46.8%)
  - `test_roc_auc: 0.5362`
  - `test_accuracy: 0.5342`

  AUC 0.536은 완전 무작위(0.5)보다 약간 높은 수준으로, "약한 신호는 있지만 강한 예측력은 없다"는 뜻이다. 이 수치 자체가 `predictor.py`의 `DISCLAIMER`("백테스트 검증 전 참고용 보조 지표")가 왜 항상 붙는지에 대한 실증적 근거이기도 하다 — 이 모델은 아직 "승격 규칙"(§6, AUC가 여러 홀드아웃에서 일관되게 베이스라인을 넘고, 전략 레벨 백테스트로 개선이 입증되는 등 4개 조건을 모두 충족)을 만족하지 못했으므로, 보조 신호 지위를 유지해야 한다.
- **전략 레벨 백테스트(거래비용/슬리피지/누적손익/MDD)는 아직 없다.** 이 모듈이 하는 평가는 "모델 레벨"(ROC-AUC, accuracy, walk-forward)뿐이고, 실제 매매 전략으로서의 손익 시뮬레이션은 범위 밖이다(`ml-predictor-plan.md` "이번 스코프가 아닌 것" 표).
- **생존편향(survivorship bias) 미보정.** `get_universe()`가 "오늘 시점 상장종목 목록"만 반환하므로, 과거 데이터로 학습하더라도 그 유니버스 자체는 "지금까지 살아남은 종목들"로 구성돼 있다 — 상장폐지된 종목은 애초에 데이터셋에 들어오지 않는다. 메타데이터에 이 사실이 명시만 돼 있을 뿐 보정 로직은 없다.
- **4계층 앙상블(규칙엔진/ML/LLM차트/뉴스) 가중치는 미확정**이다. 지금은 각 신호원이 독립된 문구를 `extra_reasons`에 나열하는 수준이고, 신호 간 가중 결합 로직은 PRD §5/§11에 TBD로 남아 있다.
- **ML 단독 후보 생성·필터링 권한은 없다.** §6에서 설명한 승격 규칙(4개 조건 모두 충족) 전부를 만족해야 재검토 대상이 되는데, 현재 AUC 수준(0.536)으로는 조건 1("서로 겹치지 않는 3개 이상의 홀드아웃에서 AUC가 일관되게 베이스라인 초과")조차 아직 입증된 바 없다.
- **딥러닝으로의 고도화도 범위 밖**이다 — PRD 원칙상 "성능이 확인된 부분만 교체"가 우선이고, 지금 단계는 그 확인 절차 자체이기 때문이다.

---

## 12. 테스트 구조 요약 (`tests/ml_predictor/`)

| 파일 | 줄수 | 검증 내용 |
|---|---|---|
| `conftest.py` | 77 | `make_records`/`random_walk_prices` 등 결정론적 합성 OHLCV 데이터 생성 fixture |
| `test_features.py` | 113 | 11개 피처의 수학적 정확성, 워밍업 미충족 시 `None`, **인과성(causality) 속성**(미래 봉 추가해도 과거 피처 불변) |
| `test_labeling.py` | 45 | `close[t+H]` 경계값, 마지막 `horizon`개 행에 레이블 없음 |
| `test_dataset.py` | 125 | 워밍업/꼬리 제거, 다종목 pooling, **전역 날짜 분할**(종목별 분할 아님을 직접 검증, `test_multi_ticker_dataset_is_split_by_global_date_not_per_ticker`), **embargo 갭 길이**, `train_test_split`/`KFold`가 모듈에 import조차 안 됐음을 확인하는 정적 가드(`test_chronological_split_never_shuffles`) |
| `test_training.py` | 100 | `build_estimator` 3종 생성, `evaluate`의 단일클래스 AUC `nan` 처리, `walk_forward_scores`의 fold 분할 |
| `test_artifact.py` | 79 | 저장/로드 왕복, 아티팩트 없을 때 `ModelArtifactError` |
| `test_predictor.py` | 117 | `predict`의 `None` 반환 조건, `_top_contributing_features`의 선형/비선형 분기, `_stance`의 동의/상충/중립 3분기, `to_reasons`의 `DISCLAIMER` 항상 포함 |

전체 656줄, 모두 결정론적 합성 데이터로 동작하며 실제 시장 데이터나 외부 API를 전혀 호출하지 않는다 — 빠르고 재현 가능한 단위테스트를 유지하기 위한 설계다.
