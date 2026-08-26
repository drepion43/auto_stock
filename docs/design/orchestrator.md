# 오케스트레이터 구현 가이드 — MVP-0 end-to-end

> `docs/IMPLEMENTATION_PLAN.md`의 1차 목표(MVP-0) 마지막 항목을 구현한 뒤 작성한 가이드 문서. 이전 컴포넌트 가이드들과 같은 위치·형식.

## 목적

데이터 수집(#0) → 규칙엔진(#1) → 리스크·사이징(#5) → 설명 생성기(#6) → 텔레그램 알림(#7)을 실제로 연결해, 종목 리스트를 넣으면 매수/매도 신호가 있는 종목에 대해 리스크 계산 + 자연어 설명 + 텔레그램 알림까지 자동으로 나가는 파이프라인을 완성한다 (PRD §3 흐름의 앞부분, "추천까지"). 새로운 계산 로직은 없다 — 이미 각자 테스트된 기존 함수들을 순서대로 호출하는 배선(wiring) 코드다.

## 모듈 구조

```
src/auto_stock/orchestrator/
├── __init__.py
├── models.py     # PipelineResult(sent: list[Explanation], errors: list[tuple[str, str]])
└── pipeline.py   # run_recommendation_pipeline(cache, tickers, market, account, credentials, lookback_days=120)
```

## 왜 종목 리스트와 계좌 상태를 오케스트레이터가 스스로 정하지 않는가

- **종목 리스트**: `docs/design/data-collection-layer.md` §7에 이미 "전체 종목 실시간 스캔의 배치/병렬화·API 호출 한도 설계는 TBD"로 남겨뒀다. 오케스트레이터가 `get_universe()`로 전체 상장종목을 자동으로 훑게 만들면, 검증되지 않은 상태에서 수백~수천 종목에 대해 실제 텔레그램 메시지가 나갈 위험이 있다. 그래서 `tickers: list[str]`를 인자로 받게 해 스캔 범위를 호출자가 결정하게 했다.
- **계좌 상태**: 아직 브로커 연동(#8, MVP-1)이 없어 실제 계좌 잔고/보유종목을 조회할 방법이 없다. `AccountState`를 호출자가 채워서 넘긴다 — `scripts/run_recommendations.py`는 임시로 `held_tickers=frozenset()`, `total_exposure_pct=0.0`, `equity`만 환경변수로 받는 자리표시자(placeholder) 값을 쓴다.

## 처리 흐름 (`run_recommendation_pipeline`)

티커마다:
1. `get_ohlcv(cache, ticker, start, end, market)` — `data/service.py`
2. `generate_candidates(records)` — `rule_engine/engine.py`. 후보가 없으면 다음 티커로 넘어간다.
3. 각 후보에 대해 `suggest_position` → `generate_explanation` → `send_notification`을 순서대로 호출
4. 성공한 `Explanation`은 `PipelineResult.sent`에 쌓인다

## 종목 단위 에러 격리

한 티커의 처리(위 1~3단계 어디서든) 중 예외가 발생하면 `(ticker, str(exception))`을 `PipelineResult.errors`에 담고 **다음 티커 처리를 계속한다**. 전체 상장종목 스캔을 목표로 하는 시스템에서 한 종목의 일시적 데이터 오류나 네트워크 실패로 전체 배치가 죽으면 안 되기 때문이다. `except Exception`으로 넓게 잡지만, 실패를 조용히 삼키지 않고 반드시 `errors`에 기록해 반환한다 — silent failure가 아니다.

## `lookback_days=120`인 이유

규칙엔진의 SMA60 워밍업과 리스크사이징의 ATR(14) 계산에 최소 60거래일 이상의 데이터가 필요하다. 주말·공휴일을 감안해 캘린더 기준 120일을 기본값으로 뒀다 — 대략 80여 거래일이 확보돼 60거래일 요구치에 여유가 있다.

## 테스트 전략

기존 세션 패턴(엔진 테스트가 지표 계산을, 사이징 테스트가 ATR을 모킹한 것)과 동일하게, `get_ohlcv`/`generate_candidates`/`suggest_position`/`generate_explanation`/`send_notification` 5개를 모두 모킹해 **배선 순서와 에러 격리만** 검증한다 — 각 모듈 내부 계산 로직은 이미 각자의 테스트 스위트에서 검증됐으므로 여기서 다시 검증하지 않는다.

- BUY 후보 1개 → 5단계가 순서대로 호출되고 결과가 `sent`에 담기는지
- 후보 없는 티커 → 사이징/설명/알림이 전혀 호출되지 않는지
- 여러 티커 중 하나가 데이터 조회 단계에서 예외 → 그 티커만 `errors`에 담기고 나머지는 정상 처리되는지
- 텔레그램 전송 실패(`TelegramNotificationError`) → `errors`에 담기고 파이프라인이 멈추지 않는지

## 실행 진입점: `scripts/run_recommendations.py`

`scripts/verify_telegram.py`와 같은 패턴. 로컬 DuckDB 캐시(`data/ohlcv.duckdb`, `.gitignore`에 이미 `data/*.duckdb` 포함)와 텔레그램 크리덴셜을 로드하고, 예시 워치리스트(`005930`, `000660`)에 대해 파이프라인을 한 번 실행한 뒤 전송/에러 건수만 출력한다. 전체 유니버스 스캔은 기본 동작이 아니다 — 검증 전 대량 알림을 막기 위한 의도적 설계다.

## 알려진 무해한 경고 메시지

`scripts/run_recommendations.py` 실행 시 `FinanceDataReader`가 `"KRX 로그인 실패: KRX_ID 또는 KRX_PW 환경 변수가 설정되지 않았습니다."`라는 메시지를 표준출력에 찍을 수 있다. 이는 `fdr.DataReader()`가 특정 KRX 데이터 경로를 먼저 시도하다 실패한 뒤 다른 소스로 자동 폴백하면서 남기는 라이브러리 자체의 경고이며, **데이터 조회 자체는 정상적으로 성공한다** (실제로 확인: `005930` 기준 81건의 OHLCV 정상 수신). `PipelineResult.errors`에 잡히지 않는 한 무시해도 된다.

## API 키

새로 필요한 것은 없다. 이미 확보된 것들(FinanceDataReader/pykrx/pandas-ta는 키 불필요, 텔레그램 봇 토큰/chat_id는 #7에서 이미 발급·설정 완료)을 조합할 뿐이다.
