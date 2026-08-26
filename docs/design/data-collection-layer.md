# 데이터 수집 계층 (#0) 개발 계획

> `docs/IMPLEMENTATION_PLAN.md`의 #0 항목을 실제 스캐폴딩이 가능한 수준으로 구체화한 컴포넌트 설계 문서.

## 목적

`docs/PRD.md` §7.2·§7.3에 정의된 시세/차트 데이터 수집을 담당하는 공용 인프라 모듈. 규칙엔진(#1), ML 예측 모듈(#2), 차트분석 에이전트(#3) 등 대부분의 서브에이전트가 이 계층에 의존한다.

## 핵심 설계 결정

### 1. 캐시 스토리지: DuckDB

이전 논의에서 "SQLite/DuckDB"로 병기됐던 것을 여기서 DuckDB로 확정한다. 전체 상장종목(수천 개) × 5년치 OHLCV에 대한 컬럼 단위 집계·백테스트 쿼리가 주 워크로드이므로, 컬럼형 저장과 분석 쿼리에 강한 DuckDB가 SQLite보다 이 용도에 적합하다. 로컬 파일 하나(`data/cache.duckdb`)로 운영되어 SQLite와 마찬가지로 별도 서버 없이 가볍게 운영된다.

### 2. 라이브러리 역할 분담

- **FinanceDataReader (FDR)**: 국내(KRX) + 나스닥 개별종목 OHLCV 조회 — 시세 데이터의 주 소스
- **pykrx**: 국내 시가총액, 전체 상장종목 리스트 등 FDR이 약한 부분을 보조

### 3. 모듈 구조

```
src/auto_stock/data/
├── models.py            # OHLCV 레코드 스키마
├── sources/
│   ├── fdr_source.py    # FinanceDataReader 래퍼 (KRX + 나스닥 OHLCV)
│   └── pykrx_source.py  # pykrx 래퍼 (국내 시가총액·종목리스트)
├── cache.py              # DuckDB 읽기/쓰기, 캐시 히트/미스 판단
├── scheduler.py          # APScheduler 폴링 잡 (시장별 운영시간 인지)
└── service.py            # 공개 인터페이스
```

### 4. 데이터 스키마 (DuckDB)

**`ohlcv` 테이블**

| 컬럼 | 타입 | 설명 |
|---|---|---|
| ticker | VARCHAR | 종목코드 |
| market | VARCHAR | `KRX` \| `NASDAQ` |
| date | DATE | 거래일 |
| open, high, low, close | DOUBLE | 가격 |
| volume | BIGINT | 거래량 |
| updated_at | TIMESTAMP | 캐시 갱신 시각 |

**`universe` 테이블** (전체 상장종목 리스트, pykrx/FDR 상장목록 기반)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| ticker | VARCHAR | 종목코드 |
| name | VARCHAR | 종목명 |
| market | VARCHAR | `KRX` \| `NASDAQ` |
| market_cap | BIGINT (nullable) | 시가총액 (pykrx, 국내만) |
| updated_at | TIMESTAMP | 갱신 시각 |

### 5. 공개 인터페이스 (`service.py`)

```python
def get_ohlcv(ticker: str, start: date, end: date, market: Literal["KRX", "NASDAQ"]) -> pd.DataFrame:
    """캐시 우선 조회, 미스(구간 부족) 시 소스 API 호출 후 캐시에 적재."""

def get_universe(market: Literal["KRX", "NASDAQ"]) -> list[str]:
    """전체 상장종목 리스트. KRX는 pykrx, NASDAQ은 FDR 상장목록 기반."""

def refresh_recent(tickers: list[str]) -> None:
    """폴링 잡이 호출하는 근접-실시간 갱신 함수."""
```

### 6. 스케줄링 (실시간에 가까운 반복 수집)

한국장(09:00–15:30 KST)과 나스닥(현지 09:30–16:00 ET, 한국시간 기준 야간)은 운영시간이 다르므로, **시장별로 별도 APScheduler job**을 등록하고 각 시장의 장 시간 외에는 폴링을 정지한다. 동시 호출은 세마포어로 제한한다 (asyncio 기반).

- Kafka 등 메시지 브로커는 채택하지 않는다 — 1인 사용자·단일 파이프라인 규모에 과도한 인프라 (`docs/IMPLEMENTATION_PLAN.md` #0 참고, YAGNI)
- 폴링 주기·동시성 제한의 정확한 수치는 스캐폴딩 시점에 FDR/pykrx의 실제 응답 시간과 호출 제한을 관찰하며 튜닝한다 (지금 임의로 고정하지 않음)

### 7. 확장 시 재검토 (Kafka 등)

향후 사용자가 많아지면 이 계층을 Kafka 같은 메시지 브로커로 전환해야 하는지에 대한 결론: **아니오, 사용자 수 증가만으로는 전환 사유가 되지 않는다.**

- 시세 데이터 수집 자체는 사용자 수와 거의 무관하게 스케일된다 — 모든 사용자가 같은 시장 데이터(코스피/코스닥+나스닥 전체 상장종목 시세)를 공유하기 때문에, 사용자가 늘어도 이 계층의 부하는 거의 늘지 않는다.
- 사용자 증가로 실제 부하가 커지는 곳은 **사용자별 추천 계산, 알림 발송** 같은 컴퓨팅/서빙 계층이며, 이는 Kafka가 아니라 경량 작업 큐(Celery+Redis, RQ 등)로 해결하는 것이 적절하다.
- **Kafka 전환이 정당화되는 실제 트리거**: 여러 개의 독립적인 다운스트림 서비스(예: 실시간 리스크 엔진, ML 피처스토어, 감사로그, 알림서비스 등)가 같은 고빈도 이벤트 스트림을 각자 내구성 있게 소비하고 재생(replay)까지 해야 하는 진짜 스트리밍 아키텍처로 전환할 때. 즉 트리거는 "사용자 수"가 아니라 "내부 서비스 개수·이벤트 재처리 필요성"이다.

### 8. 에러 처리

- 소스 API 호출 실패 시 지수 백오프로 재시도 (최대 재시도 횟수는 스캐폴딩 시점에 결정)
- 재시도 소진 시 캐시에 있는 마지막 값을 유지하고 에러를 로깅한다 — 조용히 실패시키지 않는다

### 9. 테스트 전략

- FDR/pykrx 실제 API를 호출하지 않고 고정 fixture(CSV/DataFrame)로 모킹
- 캐시 히트/미스 분기 테스트
- 국내/나스닥 혼합 조회 테스트
- 스케줄러는 시각을 모킹해 "장 시간에만 트리거되는지" 검증 (실제 폴링 대기 없이)
- API 실패 → 재시도 → 캐시 폴백 경로 테스트
