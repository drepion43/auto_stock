# 추천 설명 생성기 (#6) 구현 가이드

> `docs/IMPLEMENTATION_PLAN.md` §6을 구현한 뒤 작성한 가이드 문서. `docs/design/risk-position-sizing.md`와 같은 위치·형식.

## 목적

규칙엔진(#1)의 매수/매도 후보와 리스크·포지션 사이징(#5)의 참고용 제안을, 사용자가 승인 여부를 판단할 수 있는 자연어 요약으로 종합한다 (PRD §5.1, Explainability). 이 요약은 알림봇(#7)이 텔레그램/디스코드 메시지에 그대로 실어 보낸다.

## 모듈 구조

```
src/auto_stock/explainer/
├── __init__.py
├── models.py      # Explanation(ticker, market, action, summary)
└── generator.py   # generate_explanation(candidate, sizing, extra_reasons=None) -> Explanation
```

## 왜 LLM이 아니라 템플릿 조합인가

PRD §5.1의 예시("RSI 과매도 구간 진입 + 실적 서프라이즈 기사 확인 + LLM 차트 분석상 반등 패턴 감지...")는 네 계층(규칙엔진+ML+LLM차트+뉴스)의 근거를 LLM으로 종합하는 그림이다. 하지만 **현재 구현된 신호 소스는 규칙엔진(#1)과 리스크·사이징(#5)뿐**이다 — ML(#2)/LLM 차트분석(#3)/뉴스·공시(#4)는 아직 없다. 이미 구조화된 두 계층의 데이터(문자열 리스트 + 숫자)를 자연어로 나열하는 데는 LLM 호출이 필요 없으므로, MVP-0에서는 **결정론적 템플릿 조합 함수**로 구현했다. 장점:

- API 키/네트워크 호출/비용/지연시간이 없다
- 출력이 결정론적이라 테스트하기 쉽다(LLM 환각 위험 없음)
- IMPLEMENTATION_PLAN.md §6의 설계 원칙("확장 가능한 구조")은 `extra_reasons: list[str] | None` 파라미터로 충족한다 — 이후 ML/LLM차트/뉴스 신호가 추가되면, 그 신호가 만든 근거 문자열을 이 파라미터로 전달하기만 하면 되고 `generate_explanation` 자체를 재설계할 필요가 없다. 신호가 늘어나 템플릿 나열만으로는 자연스러운 문장이 안 나오는 시점이 오면, 그때 이 함수를 LLM 기반으로 교체하는 것을 재검토한다.

## 조합 로직 (`generate_explanation`)

1. **근거 문장**: `candidate.reasons + extra_reasons`를 `" + "`로 결합
2. **수량/배분 문장**: `sizing.suggested_quantity`와 `suggested_allocation_pct`가 모두 있을 때만 추가 (BUY이고 사이징 계산에 성공한 경우) — 예: "참고용 매수 제안: 50주 (계좌 자산의 5.0%)."
3. **손절/익절 문장**: `stop_loss_price`/`take_profit_price`가 모두 있을 때만 추가
4. **`sizing.notes`**: 있으면 그대로 이어붙인다 — 한도 초과 경고(`EXCEEDS_MAX_POSITIONS`/`EXCEEDS_EXPOSURE_CAP`)든, SELL/ATR-불가로 인한 `NOT_APPLICABLE` 사유든 **`limit_check` 값으로 분기하지 않고 `notes`의 유무만 본다** — 리스크·사이징(#5)이 이미 모든 특수 케이스를 `notes`에 사람이 읽을 문장으로 채워 반환하도록 설계했으므로, 설명 생성기는 그 문장을 그대로 신뢰하고 이어붙이기만 하면 된다. 두 모듈의 책임을 분리하는 방식이다.

## 테스트 전략

- BUY + `PASS`: 근거·수량·배분·손절·익절 문장이 모두 포함되는지
- SELL(`NOT_APPLICABLE`): 근거 + `notes` 사유만 포함되고, 수량/손절 문장은 나타나지 않는지 ("참고용 매수 제안", "ATR 기반" 문자열 부재로 확인)
- BUY + `EXCEEDS_MAX_POSITIONS`: 한도 초과 경고가 요약에 포함되는지
- `extra_reasons` 전달: 향후 신호 확장 시 근거 문장에 그대로 결합되는지 (확장 가능한 구조의 실제 동작 검증)

## API 키

**필요 없음.** LLM을 호출하지 않는 순수 문자열 조합 로직이다.
