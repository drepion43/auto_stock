# 리스크·포지션 사이징 에이전트 (#5) 구현 가이드

> `docs/IMPLEMENTATION_PLAN.md` §5를 구현한 뒤 작성한 가이드 문서. `docs/design/rule-engine.md`와 같은 위치·형식.

## 목적

규칙엔진(#1)이 생성한 매수/매도 후보(`Candidate`)를 입력받아, PRD §6에서 확정한 리스크 한도 내에서 "이 종목은 이 정도 수량이 적절합니다"라는 참고용 제안을 계산한다.

## MVP-0 범위 (참고용 제안까지만)

PRD §9·IMPLEMENTATION_PLAN.md §5에 따라 **1차 목표는 참고용 사이징 제안까지만**이다. 일일 손실 2%/월간 손실 10% 한도 도달 시 자동매매를 중단하는 "집행(enforcement)" 로직은 구현하지 않는다 — 실제 주문이 나가야 의미가 있는 로직이라, 주문 실행 에이전트(#8)와 함께 2차 목표(MVP-1)에서 구현한다. 이 모듈이 반환하는 한도 검증 결과(`limit_check`)도 **표시(정보 제공)일 뿐 주문을 차단하지 않는다**.

## 모듈 구조

```
src/auto_stock/risk_sizing/
├── __init__.py
├── models.py    # AccountState, SizingSuggestion
└── sizing.py    # suggest_position(candidate, records, account) -> SizingSuggestion
```

- `rule_engine/indicators.py`에 `atr()` 함수를 추가해 재사용한다 (`docs/design/rule-engine.md`에서 이미 "ATR도 pandas-ta로 재사용 가능"이라고 예고한 대로) — 새 ATR 계산 코드를 별도로 만들지 않는다.

## 계산 로직 (`suggest_position`)

1. **BUY만 사이징 대상** — SELL 후보는 기존 보유분 청산 개념이라 신규 매수 사이징과 무관하다. `limit_check="NOT_APPLICABLE"`, 수량/배분/손절/익절 모두 `None`을 반환하고 `notes`에 사유를 남긴다.
2. **ATR 계산 불가 시**(레코드 부족 등, `atr()`가 `None` 반환) 동일하게 `NOT_APPLICABLE` 처리.
2-1. **종가가 0 이하인 경우**(비정상 데이터)도 동일하게 `NOT_APPLICABLE` 처리 — `risk-policy-guardian` 리뷰에서 지적된 사항으로, 검증 없이 진행하면 "배분 제안 + 수량 0 + 음수 손절가"처럼 앞뒤가 안 맞는 결과가 나올 수 있어 계산 전에 차단한다.
3. **포지션 사이징 (ATR 역변동성)**:
   ```
   atr_pct = ATR / 종가
   배분비중 = min(MAX_POSITION_PCT, MAX_POSITION_PCT × BASELINE_ATR_PCT / atr_pct)
   ```
   변동성(`atr_pct`)이 기준(`BASELINE_ATR_PCT`)보다 낮으면 상한(`MAX_POSITION_PCT`=5%)까지 배분하고, 높으면 반비례로 배분을 줄인다. `atr_pct <= 0`(변동성 없음)인 예외적 경우는 상한을 그대로 적용한다.
4. **손절/익절 (ATR 배수)**: `손절가 = max(0.01, 종가 - STOP_LOSS_ATR_MULT × ATR)`, `익절가 = 종가 + TAKE_PROFIT_ATR_MULT × ATR` — ATR이 종가 대비 매우 크면 손절가가 음수가 될 수 있어(`risk-policy-guardian` 리뷰에서 지적) 0에 가까운 하한(`0.01`)으로 클램프한다.
5. **수량**: `equity × 배분비중 / 종가`를 내림한 정수 주식 수
6. **한도 검증 (표시만, 차단 없음)**:
   - 신규 종목이면서 현재 보유 종목 수가 이미 `MAX_CONCURRENT_POSITIONS`(10) 이상 → `EXCEEDS_MAX_POSITIONS`
   - 현재 총 익스포저 + 이번 배분비중이 `MAX_TOTAL_EXPOSURE_PCT`(50%) 초과 → `EXCEEDS_EXPOSURE_CAP`
   - 둘 다 아니면 `PASS`

## 리스크 한도 상수

| 상수 | 값 | PRD 근거 | 비고 |
|---|---|---|---|
| `MAX_POSITION_PCT` | 5% | §6 종목당 최대 투자 비중 | 확정치 |
| `MAX_CONCURRENT_POSITIONS` | 10종목 | §6 최대 동시 보유 종목 수 | 확정치 |
| `MAX_TOTAL_EXPOSURE_PCT` | 50% | §6 (10종목×5%) | 확정치 |
| `STOP_LOSS_ATR_MULT` | 1.5 | §6 ATR 기반 손절 | **잠정치** — 백테스트로 확정 예정 (PRD §11 TBD) |
| `TAKE_PROFIT_ATR_MULT` | 3.0 | §6 ATR 기반 익절 | **잠정치** — 백테스트로 확정 예정 (PRD §11 TBD) |
| `BASELINE_ATR_PCT` | 2% | §6 ATR 기반 포지션 사이징 | **잠정치** — 사이징 계산식의 기준 변동성. 백테스트로 확정 예정 (PRD §11 TBD) |

`MAX_POSITION_PCT`/`MAX_CONCURRENT_POSITIONS`/`MAX_TOTAL_EXPOSURE_PCT`는 PRD가 이미 확정한 값이지만, ATR 배수·기준변동성 3개는 PRD §11에 "백테스트로 확정 예정"이라고 명시된 TBD 항목이다. 백테스트 인프라가 아직 없는 현재 단계에서는 이 문서에 명시된 잠정값으로 구현하고, 이후 백테스트 결과에 따라 조정한다.

## 테스트 전략과 이유

- `test_engine.py`가 지표 계산(`rsi`/`sma`)을 모킹해 의사결정 로직만 검증한 것과 동일한 방식으로, `test_sizing.py`도 `auto_stock.risk_sizing.sizing.atr`을 모킹해 사이징/한도 로직을 ATR 계산 자체와 분리해 검증한다.
- 저변동성 → 상한(5%) 근접, 고변동성 → 배분 축소, 손절/익절 가격이 ATR 배수대로 계산되는지, 동시보유 10종목 한도 초과, 총 익스포저 50% 한도 초과, 정상 통과, SELL/ATR-불가 시 `NOT_APPLICABLE`, 종가 0 이하 시 `NOT_APPLICABLE`, ATR이 종가를 초과할 때 손절가가 0보다 큰지 — 10개 케이스로 계산식과 한도 검증 분기를 모두 커버한다.
- `AccountState`도 `OHLCVRecord`와 동일하게 `__post_init__`으로 경계값을 검증한다 (`tests/risk_sizing/test_account_state.py`) — `equity` 음수, `total_exposure_pct`가 `[0, 1]` 범위 밖인 경우를 생성 시점에 차단해, 잘못된 계좌 상태가 사이징 계산까지 흘러가 음수 수량 같은 안전하지 않은 제안을 만드는 것을 막는다.

## risk-policy-guardian 리뷰 결과 반영

구현 직후 `risk-policy-guardian` 서브에이전트 리뷰에서 5%/10종목/50% 한도 산술과 ATR 방향성(저변동성→상한 근접, 고변동성→축소)은 정확하다고 확인됐다. 다만 아래 두 가지 안전성 이슈(HIGH)가 지적되어 반영했다:

1. **음수 손절가** — ATR이 종가 대비 크면 `종가 - 1.5×ATR`이 음수가 될 수 있음 → `max(0.01, ...)`으로 클램프 (위 §계산 로직 4번)
2. **`종가 <= 0` 미검증** — `ZeroDivisionError`만 막고 있었을 뿐, 비정상 데이터가 그대로 사이징 계산에 들어가 앞뒤가 안 맞는 결과를 만들 수 있었음 → SELL/ATR-불가와 동일하게 `NOT_APPLICABLE` 처리 (위 §계산 로직 2-1번)

MEDIUM으로 지적된 `AccountState.equity` 음수 미검증도 함께 반영했다(`__post_init__` 추가). LOW로 지적된 "동시 10종목 초과와 50% 익스포저 초과가 동시에 해당하면 `EXCEEDS_MAX_POSITIONS`만 표시되고 익스포저 초과 사실이 `notes`에서 누락되는 문제"는 이번 라운드에서는 보류했다 — `limit_check`은 참고용 표시일 뿐 차단 로직이 아니라 우선순위는 낮으며, 필요 시 이후 라운드에서 두 조건을 모두 평가해 `notes`에 합치는 방식으로 개선할 수 있다.

## API 키

**필요 없음.** 이 모듈은 순수 계산 로직이며 외부 API를 호출하지 않는다. 계좌 상태(`AccountState`)는 호출자가 (향후 브로커 API 또는 로컬 포트폴리오 추적 로직에서) 채워서 전달한다.

## 다음 단계와의 연결

- **추천 설명 생성기(#6)**: 이 모듈의 `SizingSuggestion`(수량·배분·손절/익절·한도 검증 결과)을 규칙엔진의 `Candidate.reasons`와 함께 종합해 자연어 설명을 생성한다.
- **주문 실행 에이전트(#8, MVP-1)**: 이 모듈에 일일/월간 손실한도 도달 시 자동매매를 중단하는 집행 로직을 추가하고, `limit_check` 결과를 실제 주문 차단에 사용하도록 확장한다.
