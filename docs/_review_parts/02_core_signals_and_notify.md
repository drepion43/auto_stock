# 코드 리뷰: 규칙엔진(#1) · 리스크·포지션 사이징(#5) · 추천 설명 생성기(#6) · 알림봇(#7)

> 이 문서는 신규 참여자가 코드만 보고 추론하기보다 빠르게 전체 그림과 세부 동작을 파악할 수 있도록, 4개 모듈의 파일/함수/클래스를 하나씩 상세히 설명한다. 서술은 코드(`src/auto_stock/rule_engine/`, `risk_sizing/`, `explainer/`, `notifier/`)와 각 모듈의 테스트, 그리고 `docs/design/rule-engine.md`, `docs/design/risk-position-sizing.md`, `docs/design/recommendation-explainer.md`, `docs/design/telegram-notifier.md`를 직접 읽고 확인한 내용만 담았다.

이 4개 모듈은 파이프라인에서 다음 순서로 연결된다:

```
OHLCVRecord(데이터 수집 계층) 
  → rule_engine.generate_candidates() → Candidate
  → risk_sizing.suggest_position() → SizingSuggestion
  → explainer.generate_explanation() → Explanation
  → notifier.send_notification() → 텔레그램 메시지 발송
```

각 단계는 순수 함수(또는 그에 준하는 얇은 I/O 래퍼)이고, 오케스트레이터(`src/auto_stock/orchestrator/pipeline.py`)가 이 순서로 호출을 엮는다. 이 문서는 그 각 단계의 내부를 다룬다.

---

## 1. `src/auto_stock/rule_engine/` — 규칙엔진 (#1)

### 역할

데이터 수집 계층(#0)이 제공하는 `OHLCVRecord` 리스트를 받아 기술적 지표를 계산하고, 이를 근거로 "지금 이 종목을 사거나 팔 신호가 있는가"에 대한 1차 후보(`Candidate`)를 만든다. 외부 API나 LLM에 전혀 의존하지 않는, 파이프라인에서 가장 먼저 만들어진 신호원이다. API 키가 필요 없다.

### `models.py` — `Candidate`

```python
@dataclass(frozen=True, slots=True)
class Candidate:
    ticker: str
    market: str
    action: str  # "BUY" | "SELL"
    reasons: list[str]
```

규칙엔진(및 이후 다른 신호원 확장 시에도 재사용되는) 출력 자료형이다. `frozen=True`로 생성 후 불변, `slots=True`로 메모리 효율을 높였다. 필드 검증 로직(`__post_init__`)은 없다 — `action` 값이 "BUY"/"SELL" 외의 문자열이어도 타입 레벨에서는 막지 않는다(실제로 생성하는 코드는 `engine.py` 한 곳뿐이라 실무상 문제는 없다).

### `indicators.py` — 기술적 지표 계산

파일 docstring에 명시된 설계 의도: "직접 수식을 구현하지 않고, 검증된 `pandas-ta` 라이브러리를 얇게 감싼다." 애초에는 SMA/RSI/MACD를 직접 구현했으나, "검증된 라이브러리를 재사용하라"는 개발 원칙에 따라 pandas-ta로 교체했다(`docs/design/rule-engine.md` §"라이브러리 채택" 참고). 입력과 출력은 전부 순수 `list[float]` / `list[float | None]`이라 호출자(`engine.py`)는 pandas-ta의 존재를 몰라도 된다.

#### `_to_list(series, length) -> list[float | None]` (내부 헬퍼, indicators.py:11-16)

pandas-ta가 반환하는 `pandas.Series`(또는 입력이 지표의 최소 길이보다 짧을 때는 `None` 자체)를 평평한 `list[float | None]`로 변환한다. `pd.isna(v)`인 값은 `None`으로 치환한다. **주의할 점**: pandas-ta는 "입력이 지표 계산에 필요한 최소 길이보다 짧을 때" `NaN으로 채워진 Series`가 아니라 `None` 그 자체를 돌려준다 — 이 함수가 그 두 경우(짧아서 `None`, 충분히 길어서 `Series`지만 앞부분이 `NaN`)를 모두 같은 포맷으로 정규화한다.

#### `sma(closes: list[float], window: int) -> list[float | None]` (19-20)

단순이동평균. `ta.sma()` 호출을 `_to_list`로 감싼 것뿐이다.

#### `ema(values: list[float], period: int) -> list[float | None]` (23-24)

지수이동평균. 현재 `engine.py`에서는 쓰이지 않지만 향후 확장을 위해 제공된 유틸리티 함수다.

#### `rsi(closes: list[float], period: int = 14) -> list[float | None]` (27-28)

상대강도지수(RSI). 기본 기간 14일(표준값). **직접 구현과의 차이점**: pandas-ta는 EWM(지수가중이동평균) 기반으로 계산해서, 워밍업 구간이 인덱스 0에서만 `NaN`이고 이후부터는 바로 값이 나온다. 손으로 구현한 버전은 `period`만큼 기다려야 값이 나왔다. 이 차이는 라이브러리 교체로 인한 정상적인 동작 변화다(design 문서에 명시됨).

#### `atr(highs, lows, closes, period: int = 14) -> list[float | None]` (31-38)

평균진폭(Average True Range) — 변동성 지표. `risk_sizing/sizing.py`에서 ATR 기반 손절/익절·포지션 사이징 계산에 재사용된다(§2 참고). `rule_engine`에서 이미 ATR을 pandas-ta로 제공해둔 덕분에 `risk_sizing`은 새 계산 코드를 만들지 않고 이 함수를 그대로 import한다.

#### `volume_ratio(volumes: list[int], window: int = 20) -> list[float | None]` (41-54)

최신 거래량을 트레일링 `window`일 평균 거래량으로 나눈 비율. docstring에 명시된 대로 **pandas-ta에 직접 대응하는 함수가 없어서** 수작업으로 구현한 유일한 지표다. `ml_predictor.features`와 `llm_chart_analyst.snapshot`이 이 함수를 공유해서 쓰기 때문에, 두 보조 신호가 "거래량비"라는 숫자에 대해 서로 다른 계산식을 쓰는 일이 없도록 보장한다. `mean_volume == 0`인 구간은 `None`(0으로 나누기 방지).

#### `macd(closes, fast=12, slow=26, signal=9) -> tuple[list, list, list]` (57-68)

MACD선/시그널선/히스토그램 3개를 튜플로 반환. `ta.macd()`가 `None`을 반환하면(입력이 너무 짧음) 세 리스트 모두 `[None] * n`으로 채운다. **현재 `engine.py`의 매수/매도 판단 규칙에는 사용되지 않는다** — 지표 함수로만 제공되어 있고, 향후 규칙 확장 여지로 남겨져 있다.

### 신규 지표 7종(2026-10 작업) — `analyze_technical_context` 챗봇 tool 전용

사용자가 "RSI/SMA20/60만으로는 부족하지 않냐"고 지적해 추가된 지표들. **`engine.generate_candidates`의 BUY/SELL 판단 규칙에는 쓰이지 않는다** — 전부 `rule_engine/technical_context.py`(아래 별도 절)가 조립해 챗봇의 `analyze_technical_context` tool로만 노출되고, 해석은 LLM(`stock_analyst`/메인 챗봇)에게 넘긴다. `macd`와 동일하게 "지표 함수만 제공, 판단은 다른 계층"이라는 역할 분리를 그대로 따른다.

#### `adx(highs, lows, closes, length=14) -> tuple[list, list, list]` (71-91)

`(adx, plus_di, minus_di)` — 추세 강도(ADX)·방향(+DI/-DI). pandas-ta의 `ta.adx()`가 반환하는 컬럼명은 `ADX_{length}`/`DMP_{length}`/`DMN_{length}`다 — **"DMP"/"DMN"이 "+DI"/"-DI"를 가리킨다는 점을 설치된 pandas-ta 소스를 직접 읽어 확인**했다(옛 버전/포크의 문서로 추측하지 않음, 함수 docstring에 명시).

#### `bbands(closes, length=20, std=2.0) -> tuple[list, list, list, list]` (94-122)

`(percent_b, bandwidth, lower, upper)` — 볼린저밴드 %B(가격이 밴드 내 어디 위치하는지)와 밴드폭(변동성 레짐). **중요한 함정(이 세션에서 직접 발견·수정)**: 설치된 pandas-ta의 `ta.bbands()`는 단일 `std` 파라미터를 받지 않고 `lower_std`/`upper_std`로 분리돼 있다 — `std=std`로만 호출하면 `**kwargs`에 조용히 흡수되어 무시되고 라이브러리 자체 기본값(2.0)으로 돌아간다. 이 래퍼는 반드시 `lower_std=std, upper_std=std` 둘 다 명시해서 호출한다. 컬럼명도 `BBP_{length}_{std}_{std}`처럼 두 std 값을 모두 포함한다(설치된 소스를 직접 읽어 확인).

#### `stoch(highs, lows, closes, k=14, d=3, smooth_k=3) -> tuple[list, list]` (125-142)

`(percent_k, percent_d)` — 스토캐스틱. 컬럼명 `STOCHk_{k}_{d}_{smooth_k}`/`STOCHd_{k}_{d}_{smooth_k}`.

#### `obv(closes, volumes) -> list[float | None]` (145-149)

On-Balance Volume. 다른 다중 컬럼 지표(macd/adx/bbands/stoch)와 달리 `ta.obv()`는 파라미터화된 접미사 없이 `"OBV"`라는 단일 Series만 반환한다 — 이 함수가 유일하게 튜플이 아니라 단일 리스트를 돌려주는 신규 지표.

#### `aroon(highs, lows, length=14) -> tuple[list, list, list]` (152-163)

`(aroon_up, aroon_down, aroon_oscillator)`. 컬럼명 `AROONU_{length}`/`AROOND_{length}`/`AROONOSC_{length}`.

#### `squeeze(highs, lows, closes, bb_length=20, kc_length=20, kc_scalar=1.5) -> list[float | None]` (166-190)

볼린저밴드가 켈트너채널 안으로 수축했는지(`SQZ_ON` == 1.0, 저변동성 — 변동성 폭발 임박 신호)를 반환한다. **중요한 함정(bbands와 연쇄된 버그, 이 세션에서 발견)**: `ta.squeeze()`는 내부적으로 `bbands(..., std=bb_std)`를 호출하는데, 이 리포의 `bbands()` 래퍼(및 설치된 pandas-ta 자체)는 `std`가 아니라 `lower_std`/`upper_std`를 받으므로 `std=bb_std` 키워드가 `**kwargs`에 흡수돼 조용히 버려진다 — 즉 `bb_std`를 아무리 바꿔도 내부 밴드폭 계산은 항상 라이브러리 기본값(2.0)을 쓴다. 이 래퍼는 그래서 애초에 `bb_std` 파라미터를 노출하지 않는다 — 동작하지 않는 노브를 있는 것처럼 보여주지 않기 위함(docstring에 이 판단 근거 명시).

#### `rsi_divergence(closes, rsi_values, lookback=20) -> list[str | None]` (193-237)

**pandas-ta에 대응 함수가 없어 `volume_ratio`와 같은 선례로 직접 구현한 유일한 신규 지표.** 각 인덱스에서 `lookback`(기본 20) 구간을 전/후반으로 나눠, 가격 저점이 낮아지는데 RSI 저점은 높아지면(가격은 더 떨어졌지만 모멘텀은 반등 — 전형적 상승 반전 경고) `"bullish"`, 가격 고점이 높아지는데 RSI 고점은 낮아지면 `"bearish"`. 둘 다 아니거나 워밍업 미충족이면 `None`. **의도적으로 단순화된 2구간 비교**(완전한 스윙포인트 탐지기가 아님) — 이 리포의 "결정론적이고 손으로 검증 가능한 극단 케이스로 테스트" 원칙을 지표 자체의 복잡도에서부터 유지하기 위한 선택이다(docstring에 명시).

### `technical_context.py` — 7개 지표를 종목 1개 단위로 조립하는 레이어(신규)

`rule_engine/indicators.py`는 순수 계산(단일 책임)만 하고, 이 모듈이 "종목 1개 최신 시점"의 7개 지표를 하나의 결과로 묶는다 — `ml_predictor.features`가 `rule_engine.indicators`를 조립해 `FeatureVector`를 만드는 것과 동일한 레이어 구조. `engine.generate_candidates`와 마찬가지로 **BUY/SELL 판단을 내리지 않는다** — 해석은 `chat_agent.stock_analyst`(LLM)에게 그대로 넘긴다.

`build_technical_context(records: list[OHLCVRecord]) -> TechnicalContext | None`가 유일한 공개 함수다. `TechnicalContext`(frozen dataclass)는 `ticker`/`market` + ADX/+DI/-DI, 볼린저밴드 %B·밴드폭, 스토캐스틱 %K/%D, OBV 추세(`obv_trend: "RISING"|"FALLING"|"FLAT"`, 최근 5개 OBV 값의 **방향**만 봄 — OBV는 임의 기준점에서 시작하는 누적값이라 절대 레벨 자체는 의미가 없기 때문), Aroon Up/Down/Oscillator, `squeeze_on: bool`, `rsi_divergence: str | None` 전부를 담는다.

**설계상 비자명한 결정(all-or-nothing)**: 7개 지표 중 **하나라도** 최신 시점에 워밍업 미충족(`None`)이면 `TechnicalContext` 전체를 `None`으로 반환한다 — `ml_predictor.features.build_feature_vectors`의 all-or-nothing 관례와 동일하게, 일부만 채워진 반쪽 결과를 만들지 않는다.

### `engine.py` — `generate_candidates(records: list[OHLCVRecord]) -> list[Candidate]`

#### 상수

```python
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
SMA_SHORT_WINDOW = 20
SMA_LONG_WINDOW = 60
```

#### 동작 흐름 (engine.py:11-48)

1. `records`가 비어있으면 즉시 `[]` 반환.
2. 종가(`close`)만 추출해 `rsi()`, `sma(20)`, `sma(60)` 세 지표를 계산.
3. **RSI 판정**: 최신 RSI 값이 `None`이 아니고, 30 미만이면 "RSI(14)=xx.x 과매도 구간 진입"을 BUY 근거에, 70 초과면 같은 포맷의 SELL 근거를 추가.
4. **SMA 크로스 판정**: 레코드가 2개 이상 있을 때만 평가. 직전 봉과 최신 봉의 SMA20/SMA60 네 값이 모두 `None`이 아니어야 한다. `short_prev <= long_prev and short_now > long_now`면 골든크로스(SMA20이 SMA60을 상향 돌파) → BUY 근거. 반대로 `short_prev >= long_prev and short_now < long_now`면 데드크로스 → SELL 근거.
5. **상충 시 후보 없음**(보수적 원칙): `buy_reasons`와 `sell_reasons`가 둘 다 있으면 — 즉 RSI는 매수를, SMA크로스는 매도를 가리키는 식으로 신호가 엇갈리면 — 어떤 `Candidate`도 생성하지 않고 `[]`를 반환한다. 이 설계의 근거는 design 문서에 명시돼 있다: "규칙엔진 단계는 보수적으로 판단하고, 최종 판단은 이후 리스크 에이전트·설명 생성기·사용자 승인 단계에서 종합한다."
6. BUY 근거만 있으면 `action="BUY"`인 `Candidate` 1개, SELL 근거만 있으면 `action="SELL"`인 `Candidate` 1개를 리스트에 담아 반환. 둘 다 없으면(또는 상충이면) `[]`.

**엣지케이스**:
- 레코드 0~1개, 또는 지표 워밍업 미충족(SMA60이면 최소 60개 종가 필요) → 해당 지표는 `None`으로 남아 그 조건은 그냥 평가되지 않는다. 모든 조건이 `None`이면 근거가 비어 `[]` 반환.
- `Candidate`는 항상 **최대 1개**만 반환된다 — 여러 티커를 한번에 넣어도 이 함수는 각 호출당 "하나의 종목(가장 최근 레코드가 가리키는 ticker/market)"에 대한 판단만 내린다. 여러 종목을 스캔하려면 호출자가 종목마다 이 함수를 반복 호출해야 한다(실제로 `orchestrator/pipeline.py`가 그렇게 한다).

---

## 2. `src/auto_stock/risk_sizing/` — 리스크·포지션 사이징 (#5, 참고용)

### 역할과 범위의 핵심 결정

규칙엔진이 만든 `Candidate`를 받아 "이 종목은 몇 주를 사는 게 적절한가, 손절/익절가는 어디인가"를 PRD §6의 확정 리스크 한도 안에서 계산한다. **단, 이 모듈의 범위는 "참고용 제안"까지만이다 — 실제 주문을 막는 집행(enforcement) 로직은 없다.** `sizing.py` 모듈 docstring에 명시: "No enforcement (no order blocking, no daily/monthly loss-limit halts) — that's MVP-1 (#8) scope." 이 모듈이 한도 초과를 감지해도(`limit_check` 필드) 그건 어디까지나 사용자에게 보여주는 **표시**일 뿐, 주문을 차단하지 않는다. 이유는 단순하다 — 아직 주문 실행 에이전트(#8) 자체가 존재하지 않으므로 "차단"이라는 개념이 작동할 대상이 없다. 일일 손실 2%/월간 손실 10% 한도의 "도달 시 자동매매 중단" 집행 로직도 같은 이유로 아직 없다(PRD §9, MVP-1 범위).

### `models.py`

#### `AccountState(equity, held_tickers, total_exposure_pct)` (models.py:4-14)

호출자가 채워서 넘기는 계좌 상태 — 이 모듈 자신은 브로커 API를 호출하지 않으므로, 실제 계좌 조회는 상위 호출자(향후 브로커 연동 또는 지금은 placeholder) 책임이다.

`__post_init__`에서 두 가지를 검증한다:
- `equity < 0`이면 `ValueError`.
- `total_exposure_pct`가 `[0, 1]` 범위를 벗어나면 `ValueError`.

이 검증은 `risk-policy-guardian` 서브에이전트 리뷰에서 지적된 MEDIUM 항목을 반영해 추가된 것이다(design 문서 §"risk-policy-guardian 리뷰 결과 반영"). 생성 시점에 막아두면, 잘못된 계좌 상태가 사이징 계산 끝까지 흘러가 "음수 수량" 같은 터무니없는 제안을 만드는 사고를 원천적으로 방지한다.

#### `SizingSuggestion` (models.py:17-28)

```python
ticker: str
market: str
action: str  # "BUY" | "SELL"
suggested_quantity: int | None
suggested_allocation_pct: float | None
stop_loss_price: float | None
take_profit_price: float | None
reference_price: float | None  # 사이징 계산에 쓰인 최신 종가 — 실시간 시세도 지정가도 아님
limit_check: str  # "PASS" | "EXCEEDS_MAX_POSITIONS" | "EXCEEDS_EXPOSURE_CAP" | "NOT_APPLICABLE"
notes: list[str]
```

`reference_price` 필드의 주석이 중요하다 — 이건 "지금 호가"가 아니라 사이징 계산에 쓴 **최신 종가**일 뿐이다. 실시간 체결가나 주문에 쓸 지정가로 오인해서는 안 된다(이 모듈이 참고용이라는 성격을 다시 한번 보여주는 디테일).

### `sizing.py`

#### 상수 (sizing.py:16-22)

| 상수 | 값 | PRD §6 대응 | 확정 여부 |
|---|---|---|---|
| `MAX_POSITION_PCT` | 0.05 (5%) | 종목당 최대 투자 비중 | **확정치** |
| `MAX_CONCURRENT_POSITIONS` | 10 | 최대 동시 보유 종목 수 | **확정치** |
| `MAX_TOTAL_EXPOSURE_PCT` | 0.50 (50%) | (10종목×5%, 나머지 50%는 항상 현금) | **확정치** |
| `STOP_LOSS_ATR_MULT` | 1.5 | ATR 기반 손절 | 잠정치 — 백테스트로 확정 예정(PRD §11 TBD) |
| `TAKE_PROFIT_ATR_MULT` | 3.0 | ATR 기반 익절 | 잠정치 — 백테스트로 확정 예정(PRD §11 TBD) |
| `BASELINE_ATR_PCT` | 0.02 (2%) | ATR 기반 포지션 사이징의 기준 변동성 | 잠정치 — 백테스트로 확정 예정(PRD §11 TBD) |
| `_MIN_PRICE` | 0.01 | — | 손절가가 0/음수로 가는 걸 막는 하한(내부용) |

앞 세 개는 PRD가 제품 기획 레벨에서 이미 **확정**한 수치이고, ATR 관련 세 개(배수·기준변동성)는 백테스트 인프라가 아직 없어서 **잠정값**으로 구현해둔 것이다 — 코드를 수정할 때 이 구분을 혼동하면 안 된다.

#### `_not_applicable(candidate, reason) -> SizingSuggestion` (내부 헬퍼, 25-37)

수량/배분/손절/익절을 전부 `None`으로, `limit_check="NOT_APPLICABLE"`로, `notes=[reason]`으로 채운 `SizingSuggestion`을 만든다. 아래 세 가지 "사이징이 성립하지 않는" 경우에 공통으로 쓰인다.

#### `suggest_position(candidate, records, account) -> SizingSuggestion` (40-87)

**1단계 — SELL은 사이징 대상이 아님** (43-44): `candidate.action != "BUY"`이면 즉시 `_not_applicable`로 "매도 후보는 기존 보유분 청산 개념이라 신규 매수 사이징 대상이 아닙니다."를 반환한다. SELL 후보에 대한 사이징이라는 개념 자체가 "지금 얼마나 팔아야 하는가"가 아니라 "이미 들고 있는 걸 얼마나 파는가"의 문제이기 때문에, 이 모듈 범위 밖이다.

**2단계 — ATR 계산 불가** (46-51): `rule_engine.indicators.atr()`로 최신 ATR을 구한다. 레코드가 부족해 ATR이 `None`이면 역시 `_not_applicable`.

**3단계 — 종가 비정상** (53-55): 최신 종가가 0 이하이면 `_not_applicable`. 이 체크는 `risk-policy-guardian` 리뷰에서 추가된 안전장치다 — 검증 없이 진행하면 뒤의 나눗셈·사이징 계산이 "배분 제안이 나왔는데 수량은 0이고 손절가는 음수" 같은 앞뒤가 안 맞는 결과를 만들 수 있었다.

**4단계 — 포지션 사이징 (ATR 역변동성)** (57-60):
```python
atr_pct = latest_atr / close
allocation_pct = MAX_POSITION_PCT if atr_pct <= 0 else min(
    MAX_POSITION_PCT, MAX_POSITION_PCT * BASELINE_ATR_PCT / atr_pct
)
```
변동성(`atr_pct`)이 기준치(`BASELINE_ATR_PCT`=2%)보다 낮으면 배분은 상한(5%)에 그대로 머물고, 기준치보다 높으면 반비례로 배분이 줄어든다. 예를 들어 `atr_pct`가 4%(기준의 2배)면 `allocation_pct = 0.05 * 0.02 / 0.04 = 0.025`(2.5%)로 줄어든다. `atr_pct <= 0`인 예외적 경우(변동성이 전혀 없는 데이터)는 상한을 그대로 적용한다 — 0으로 나누기를 피하는 분기이기도 하다.

**5단계 — 손절/익절가** (61-62):
```python
stop_loss_price = max(_MIN_PRICE, close - STOP_LOSS_ATR_MULT * latest_atr)   # 종가 - 1.5×ATR, 0.01 하한
take_profit_price = close + TAKE_PROFIT_ATR_MULT * latest_atr                # 종가 + 3×ATR
```
ATR이 종가 대비 매우 크면 `close - 1.5*ATR`이 음수가 될 수 있어(고변동성 종목에서 실제로 발생 가능) `max(0.01, ...)`로 클램프한다. 이 역시 `risk-policy-guardian` 리뷰에서 지적된 HIGH 항목을 반영한 결과다.

**6단계 — 수량** (63): `int(account.equity * allocation_pct // close)` — 계좌 자산에 배분비중을 곱한 금액을 종가로 나눠 내림한 정수 주식 수.

**7단계 — 한도 검증 (표시만)** (65-74):
```python
is_new_position = candidate.ticker not in account.held_tickers
if is_new_position and len(account.held_tickers) >= MAX_CONCURRENT_POSITIONS:
    limit_check = "EXCEEDS_MAX_POSITIONS"
elif account.total_exposure_pct + allocation_pct > MAX_TOTAL_EXPOSURE_PCT:
    limit_check = "EXCEEDS_EXPOSURE_CAP"
else:
    limit_check = "PASS"
```
- 이 종목이 **신규 종목**(현재 미보유)인데 보유 종목 수가 이미 10개 이상이면 `EXCEEDS_MAX_POSITIONS`.
- 그렇지 않고 "현재 총 익스포저 + 이번 배분비중"이 50%를 넘으면 `EXCEEDS_EXPOSURE_CAP`.
- 둘 다 아니면 `PASS`.

**design 문서에 기록된 알려진 한계(의도적 보류)**: 10종목 초과와 50% 익스포저 초과가 동시에 해당하는 경우, `elif` 구조상 `EXCEEDS_MAX_POSITIONS`만 표시되고 익스포저 초과 사실은 `notes`에서 빠진다. 이건 LOW 심각도로 지적됐고 "표시 로직이라 차단 기능이 아니므로 우선순위가 낮다"는 판단으로 아직 고치지 않았다 — 다음에 손댈 때는 두 조건을 모두 평가해 `notes`에 합치는 방식으로 개선할 수 있다.

**엣지케이스 요약**:
- SELL 후보 → `NOT_APPLICABLE`
- ATR 계산 불가(레코드 부족) → `NOT_APPLICABLE`
- 종가 ≤ 0 → `NOT_APPLICABLE`
- ATR이 종가보다 커도 손절가는 절대 0 이하로 내려가지 않는다(`_MIN_PRICE` 클램프)
- 변동성이 전혀 없어도(`atr_pct <= 0`) 사이징은 계산되고(상한 그대로), 다른 경로로 넘어가지 않는다

---

## 3. `src/auto_stock/explainer/` — 추천 설명 생성기 (#6)

### 역할과 "왜 LLM이 아닌가"

규칙엔진의 `Candidate`와 리스크사이징의 `SizingSuggestion`을 사람이 승인 여부를 판단할 수 있는 자연어 문장으로 종합한다. 알림봇(#7)은 이 문장을 그대로 메시지에 실어 보낸다.

PRD §5.1이 그리는 최종 그림은 "규칙엔진+ML+LLM차트+뉴스" 네 계층의 근거를 LLM으로 종합하는 모습이다. 하지만 이 모듈이 구현된 시점에는 ML(#2)·LLM 차트분석(#3)·뉴스/공시(#4)가 아직 없었고, 이미 구조화된 두 계층(문자열 리스트 + 숫자)을 나열하는 데는 LLM이 필요 없었다. 그래서 **결정론적 템플릿 조합 함수**로 구현했다. 장점은 design 문서에 명시돼 있다: API 키/네트워크 호출/비용/지연시간이 없고, 출력이 결정론적이라 테스트가 쉽고 환각 위험이 없다.

중요한 건 **확장성이 애초에 설계에 들어가 있었다는 점**이다 — `extra_reasons: list[str] | None` 파라미터 하나로 향후 ML/LLM차트/뉴스 신호가 만든 근거 문자열을 그대로 받아 넘길 수 있게 해뒀다. 실제로 ML 예측 모듈(#2)이 나중에 추가됐을 때 이 파일은 **한 줄도 수정되지 않았다**(`docs/CODEBASE_REVIEW.md`에 기록된 사실). 신호가 더 늘어나 템플릿 나열만으로는 자연스러운 문장이 안 나오는 시점이 오면, 그때 이 함수를 LLM 기반으로 교체하는 걸 재검토하면 된다.

### `models.py` — `Explanation(ticker, market, action, summary)`

최종 산출물. `summary` 하나의 문자열에 모든 근거/제안/경고가 담긴다.

### `generator.py` — `generate_explanation(candidate, sizing, extra_reasons=None) -> Explanation` (15-37)

조합 순서:

1. **근거 문장**: `candidate.reasons + (extra_reasons or [])`를 `" + "`로 이어붙인다. 둘 다 비면 `"신호 없음"`.
2. **수량/배분 문장**: `sizing.suggested_quantity`와 `suggested_allocation_pct`가 **둘 다** `None`이 아닐 때만 추가 — 즉 BUY이고 사이징 계산이 성공한 경우에만. 예: `"참고용 매수 제안: 50주 (계좌 자산의 5.0%)."`
3. **손절/익절 문장**: `stop_loss_price`/`take_profit_price`가 둘 다 있을 때만 추가. 예: `"손절 48,500 / 익절 55,000 (ATR 기반, 참고용)."`
4. **`sizing.notes`**: 있으면 그대로 이어붙인다.

여기서 핵심 설계 결정이 하나 있다: **4번은 `sizing.limit_check` 값으로 분기하지 않고 `notes`가 비었는지만 본다.** 리스크사이징(#5) 쪽이 이미 모든 특수 케이스(한도 초과 경고든, `NOT_APPLICABLE`의 사유든)를 사람이 읽을 문장 형태로 `notes`에 채워 반환하도록 설계돼 있기 때문에, 설명 생성기는 그 문장을 신뢰하고 그대로 이어붙이기만 하면 된다. 두 모듈의 책임이 깨끗하게 분리돼 있다 — 설명 생성기는 "limit_check가 뭘 의미하는지" 전혀 몰라도 된다.

최종 포맷: `f"{candidate.ticker}: " + " ".join(parts)`.

**예시 (BUY, PASS)**: `"005930: RSI(14)=25.3 과매도 구간 진입. 참고용 매수 제안: 50주 (계좌 자산의 5.0%). 손절 48,500 / 익절 55,000 (ATR 기반, 참고용)."`

**예시 (SELL)**: `"005930: RSI(14)=75.1 과매수 구간 진입. 매도 후보는 기존 보유분 청산 개념이라 신규 매수 사이징 대상이 아닙니다."` — 수량/손절 문장은 아예 등장하지 않는다(2·3단계 조건이 거짓이므로).

---

## 4. `src/auto_stock/notifier/` — 알림봇 (#7, 알림 전용)

### 역할과 범위

추천 설명 생성기(#6)가 만든 `Explanation`을 텔레그램 메시지로 사용자에게 보낸다. **승인/거부 버튼이나 콜백 처리는 하지 않는다** — 승인해도 트리거될 주문 자체가 없기 때문이다(주문 실행 에이전트 #8이 아직 없음). 그 흐름은 #8과 함께 MVP-1에서 추가될 예정(현재는 PRD §9 결정에 따라 보류 중).

### 왜 `python-telegram-bot`이 아니라 `requests` 직접 호출인가

`python-telegram-bot` v20 이상은 async 전용 API다. 이 프로젝트의 나머지 모듈은 전부 동기 함수로 작성돼 있어서, 이 한 곳 때문에 asyncio 이벤트 루프를 들여올 이유가 없었다(YAGNI). 텔레그램 Bot API 자체는 단순 REST라서, 이미 다른 데이터 소스를 통해 간접 의존성으로 들어와 있던 `requests`로 `sendMessage` 엔드포인트를 직접 호출하는 쪽이 더 가볍고 일관됐다.

### `models.py` — `TelegramCredentials(bot_token, chat_id)`

```python
@dataclass(frozen=True, slots=True)
class TelegramCredentials:
    bot_token: str = field(repr=False)  # never let dataclass repr/str echo the secret
    chat_id: str
```

`bot_token`에 `field(repr=False)`를 붙여서, 이 객체를 로그에 찍거나 `repr()`/`str()`을 호출해도 토큰 값이 출력되지 않는다. 토큰 유출 방지를 코드 레벨에서 강제하는 작은 디테일이다.

### `credentials.py` — `load_telegram_credentials() -> TelegramCredentials`

```python
def load_telegram_credentials() -> TelegramCredentials:
    load_dotenv()
    return TelegramCredentials(
        bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
        chat_id=os.environ["TELEGRAM_CHAT_ID"],
    )
```

`load_dotenv()`를 호출 시점에 실행해 `.env` 파일을 읽은 뒤, `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`를 환경변수에서 읽는다. 둘 중 하나라도 없으면 `KeyError`를 던진다 — "필수 시크릿은 값이 없으면 즉시 실패"하는 패턴이다(이 리포 전체에 반복되는 관례). `send_notification()` 자체는 크리덴셜을 내부에서 직접 읽지 않고 인자로 받는다 — 그래서 테스트할 때 `requests.post`만 모킹하면 실제 `.env` 파일이나 환경변수와 무관하게 순수 함수처럼 검증할 수 있다.

이 함수가 design 문서에서 "이 리포에서 '진입점이 필요 시점에 직접 `load_dotenv()`를 호출'하는 패턴의 원조"로 언급된다 — 이후 추가된 `llm_chart_analyst/credentials.py`, `news_disclosure/credentials.py` 등도 동일한 패턴을 따른다.

### `telegram_bot.py`

#### 상수 (13-14)
```python
_ACTION_LABELS = {"BUY": "매수", "SELL": "매도"}
_REQUEST_TIMEOUT_SECONDS = 10
```

#### `TelegramNotificationError(Exception)` (17-18)

이 모듈이 던지는 유일한 예외 타입. 네트워크 오류, HTTP 실패, 텔레그램 API 자체 오류를 모두 이 타입으로 통일한다.

#### `_format_message(explanation) -> str` (내부 헬퍼, 21-23)

`f"[{label}] {explanation.summary}"` — `action`을 한국어 라벨("매수"/"매도")로 바꿔 접두어로 붙인다. `_ACTION_LABELS.get(explanation.action, explanation.action)`이라서, 혹시 알 수 없는 action 문자열이 들어와도 그 문자열을 그대로 라벨로 쓸 뿐 예외를 던지지 않는다(방어적).

#### `send_notification(explanation, credentials) -> None` (26-42)

```python
try:
    response = requests.post(
        f"https://api.telegram.org/bot{credentials.bot_token}/sendMessage",
        json={"chat_id": credentials.chat_id, "text": _format_message(explanation)},
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
except requests.RequestException as exc:
    raise TelegramNotificationError(f"텔레그램 요청 중 네트워크 오류: {exc}") from exc

if not response.ok:
    raise TelegramNotificationError(f"텔레그램 요청 실패 (status={getattr(response, 'status_code', '?')})")

payload = response.json()
if not payload.get("ok"):
    raise TelegramNotificationError(f"텔레그램 API 오류: {payload.get('description', 'unknown')}")
```

3단계 실패 처리:
1. `requests.RequestException`(타임아웃·연결 실패 등 전송 자체의 실패) → `TelegramNotificationError`로 래핑.
2. HTTP 레벨 실패(`response.ok`가 거짓, 즉 4xx/5xx) → `TelegramNotificationError`.
3. HTTP는 200이어도 텔레그램 API 자체가 `{"ok": false}`를 반환하는 경우(예: `Bad Request: chat not found`) → `payload["description"]`을 담아 `TelegramNotificationError`.

**타임아웃이 중요한 이유**: `security-reviewer` 리뷰에서 LOW 항목으로 지적되어 추가됐다 — 타임아웃이 없으면 `api.telegram.org`가 응답하지 않을 때 알림 전송이 무한정 멈춰 있을 수 있었다(자체 가용성 리스크). 동시에 그 리뷰 전에는 `requests.RequestException`이 그대로 호출자에게 새어나갔는데, 지금은 전부 `TelegramNotificationError`로 통일됐다.

**보안 설계**: 어떤 예외 메시지에도 `bot_token`이나 `chat_id` 값 자체는 포함되지 않는다 — 상태 코드와 텔레그램이 돌려준 설명 문자열만 담는다. (단, 그 `description` 문자열 자체에 `chat_id`가 간접적으로 포함될 가능성은 리뷰에서 "정보성"으로만 지적되고 코드 변경은 보류됐다 — 당장 긴급한 문제는 아니라는 판단.)

### 수동 검증 스크립트

`scripts/verify_telegram.py` — `.env`에 실제 토큰/chat_id를 채운 뒤 프로젝트 루트에서 실행하면 실제 텔레그램 메시지 1건을 전송해 연동을 확인할 수 있다. 토큰/chat_id 값 자체는 절대 출력하지 않는다. `TELEGRAM_CHAT_ID`를 모를 때는 봇에게 먼저 메시지를 보낸 뒤 `https://api.telegram.org/bot<TOKEN>/getUpdates`로 `result[0].message.chat.id`를 확인하면 된다.

---

## 테스트 커버리지 메모

- `tests/rule_engine/test_indicators.py`는 지표 계산을 "교과서 기준값" 하드코딩이 아니라 수학적으로 자명한 극단 케이스(단조증가 가격→RSI 100, 단조감소→RSI 0, 지속 상승 추세→MACD선 양수)로 검증한다. 이렇게 하면 지표 계산 라이브러리가 바뀌어도(실제로 pandas-ta로 교체된 적이 있다) 테스트가 깨지지 않는다.
- `tests/rule_engine/test_engine.py`는 `indicators.rsi`/`sma`를 모킹해서 "지표 계산이 맞는가"와 "지표 결과를 올바르게 해석하는가"를 독립적으로 검증한다.
- `tests/risk_sizing/test_sizing.py`도 같은 패턴으로 `auto_stock.risk_sizing.sizing.atr`을 모킹해 계산식과 한도 분기 로직을 ATR 계산 자체와 분리해서 검증한다(저변동성/고변동성 배분, 손절/익절 산술, 10종목 한도, 50% 익스포저 한도, PASS, SELL/ATR-불가/종가≤0의 `NOT_APPLICABLE`, 손절가 하한 클램프 등).
- `tests/risk_sizing/`에서 `AccountState`의 `__post_init__` 검증(음수 equity, 범위 밖 `total_exposure_pct`)도 별도로 테스트된다.
- `tests/explainer/test_generator.py`는 BUY+PASS(근거·수량·배분·손절·익절 전부 포함), SELL(`NOT_APPLICABLE`, 수량/손절 문장 부재), BUY+`EXCEEDS_MAX_POSITIONS`(경고 포함), `extra_reasons` 전달(확장 포인트 동작 확인) 네 가지 경로를 검증한다.
- `tests/notifier/test_credentials.py`는 `monkeypatch.setenv`/`delenv`로 환경변수 유무에 따른 정상 로드·`KeyError`를 검증한다(실제 `.env`나 진짜 토큰 불필요). `tests/notifier/test_telegram_bot.py`는 `requests.post`를 모킹해 URL/payload, BUY/SELL 라벨, 실패 응답 시 예외를 검증한다 — 실제 네트워크 호출은 테스트에서 전혀 발생하지 않는다.
- 실제 텔레그램 서버로의 전송(수동 확인)은 자동화 테스트 범위 밖이며, `scripts/verify_telegram.py`로 사용자가 직접 확인해야 한다.

## 다음 작업과의 연결 지점

- `risk_sizing.sizing`에 10종목 초과와 50% 익스포저 초과가 동시에 발생하는 경우의 `notes` 보강(§2 "알려진 한계" 참고)은 아직 미해결 상태로 남아있다.
- ATR 관련 세 상수(`STOP_LOSS_ATR_MULT`/`TAKE_PROFIT_ATR_MULT`/`BASELINE_ATR_PCT`)는 PRD §11에 "백테스트로 확정 예정"이라고 명시된 TBD 항목이다. 백테스트 인프라가 구축되면 이 값들을 재검토해야 한다.
- 알림봇의 승인/거부 콜백, 리스크사이징의 일일/월간 손실한도 "집행" 로직은 PRD §9 결정에 따라 MVP-1(주문 실행 에이전트 #8)과 함께 보류 중이다 — 사용자가 이 결정을 번복하기 전까지 착수하지 않는다.
