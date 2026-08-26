# 규칙엔진 (#1) 구현 가이드

> `docs/IMPLEMENTATION_PLAN.md` §1을 구현한 뒤 작성한 가이드 문서. `docs/design/data-collection-layer.md`와 같은 위치·형식.

## 목적

데이터 수집 계층(#0)이 제공하는 OHLCV를 입력받아 기술적 지표를 계산하고, 이를 근거로 1차 매수/매도 후보를 생성한다 (PRD §4, §5).

## 모듈 구조

```
src/auto_stock/rule_engine/
├── __init__.py
├── models.py       # Candidate(ticker, market, action, reasons)
├── indicators.py   # sma / ema / rsi / macd — pandas-ta 래퍼
└── engine.py        # generate_candidates(records: list[OHLCVRecord]) -> list[Candidate]
```

## 라이브러리 채택: pandas-ta (직접 구현 대체)

처음에는 SMA/RSI/MACD를 직접 구현했으나, 사용자 요청으로 "직접 구현 대신 쓸 수 있는 게 있는지" 조사한 결과 **[pandas-ta](https://github.com/twopirllc/pandas-ta)** 라이브러리가 130개 이상의 지표(SMA, RSI, MACD, ATR 등)를 이미 구현해 제공하며 **API 키가 필요 없는 로컬 라이브러리**임을 확인했다. 검증된 라이브러리를 재사용하는 것이 원칙이므로 (`~/.claude/rules` development-workflow.md) `indicators.py`를 pandas-ta 기반으로 교체했다.

- 조사 결과 **브로커/데이터 API 중 RSI/MACD 값 자체를 제공하는 곳은 없다** — 모두 원시 시세만 주고 지표 계산은 클라이언트 몫이다. 따라서 "API를 쓰는 게 더 효율적"이라는 취지는 원격 API가 아니라 **로컬의 검증된 라이브러리(pandas-ta)를 쓰는 것**으로 충족했다.
- `indicators.py`의 공개 함수 시그니처(입력: `list[float]`, 출력: 인덱스 정렬된 `list[float | None]`)는 유지했으므로 `engine.py`는 pandas-ta를 직접 알 필요가 없다.
- **pandas-ta 동작상 주의점**: 입력 길이가 지표에 필요한 최소 길이보다 짧으면 `None`을 반환한다(NaN 시리즈가 아님) — `indicators.py`에서 이 케이스를 감지해 `[None] * len(입력)`으로 변환한다. RSI의 워밍업 구간은 직접 구현과 달리 인덱스 0에서만 `NaN`이고 이후부터 바로 값이 나온다(직접 구현은 `period`만큼 기다렸음) — 이 차이는 pandas-ta가 EWM 기반으로 계산하기 때문이며, 라이브러리 채택 후의 정상 동작이다.
- **향후 확장**: 리스크·포지션 사이징 에이전트(#5)에 필요한 ATR(Average True Range, PRD §6)도 pandas-ta에 `ta.atr()`로 이미 구현돼 있어 재사용 가능하다.

## 지표 파라미터

| 지표 | 파라미터 | 용도 |
|---|---|---|
| SMA | 단기 20일 / 장기 60일 | 골든크로스/데드크로스 판단 |
| RSI | 14일 (표준) | 과매도(<30) / 과매수(>70) 판단 |
| MACD | 12/26/9 (표준) | (현재 엔진 규칙에는 직접 사용하지 않음, 지표 함수로만 제공 — 향후 확장 여지) |

## 의사결정 규칙 (`engine.generate_candidates`)

1. 최신 RSI가 30 미만이면 "RSI 과매도" BUY 근거, 70 초과면 "RSI 과매수" SELL 근거
2. 직전 봉 대비 SMA20이 SMA60을 상향 돌파하면 "골든크로스" BUY 근거, 하향 돌파하면 "데드크로스" SELL 근거
3. BUY 근거와 SELL 근거가 **동시에** 존재하면 (상충) **후보를 만들지 않는다** — 규칙엔진 단계는 보수적으로 판단하고, 최종 판단은 이후 리스크 에이전트·설명 생성기·사용자 승인 단계에서 종합한다
4. 데이터가 부족(레코드 0~1개, 워밍업 미충족)하면 후보 없음

## 테스트 전략과 이유

- **`test_indicators.py`**: 지표 계산 자체를 검증하되, 임의의 "교과서 기준값"을 하드코딩하지 않고 수학적으로 자명한 극단 케이스로 검증한다 — 단조 증가 가격 → RSI 100, 단조 감소 → RSI 0, 지속 상승 추세 → MACD 라인 양수. 이렇게 하면 지표 계산 라이브러리가 바뀌어도(이번처럼) 테스트가 여전히 유효하다.
- **`test_engine.py`**: 의사결정 로직을 지표 계산과 분리해서 검증하기 위해 `indicators.rsi`/`sma`를 모킹해 원하는 지표값을 직접 주입한다. 이렇게 하면 "지표 계산이 맞는가"와 "지표 결과를 올바르게 해석하는가"를 독립적으로 테스트할 수 있다.
- 두 테스트 스위트 모두 회귀 없이 통과함을 pandas-ta 교체 전후로 확인했다.

## API 키

**필요 없음.** FinanceDataReader, pykrx, pandas-ta 모두 API 키 없이 동작한다.
