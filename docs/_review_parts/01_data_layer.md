# `src/auto_stock/data/` — 데이터 수집 계층 (#0) 상세 리뷰

이 문서는 PRD #0 데이터 수집 계층을 구성하는 모든 파일의 공개 함수/클래스를 함수 단위로 상세히 정리한 것이다. 설계 배경 전반은 `docs/design/data-collection-layer.md`에 있으며, 이 문서는 그 설계가 실제 코드에서 어떻게 구현됐는지, 그리고 설계 문서 작성 이후 실제 구현 과정에서 추가/변경된 부분(DART/EDGAR/네이버/GDELT 소스, 업종 분류, market 세분화)까지 포함해 다룬다.

> **실측 정정 (2026-10-11)**: 아래 market 세분화 서술은 계획 상태이며 **아직 코드에 반영되지 않았다** — `data/models.py::VALID_MARKETS`는 여전히 `{"KRX","NASDAQ"}`이고, `data/market_routing.py`·`orchestrator/market_buckets.py`·`sources/index_*`·`sources/investor_flow.py`는 untracked 상태로 연결되지 않았다. `sources/etf_source.py` 절만 2026-10-11 구현·라이브 검증 기준으로 갱신했다.

**market 세분화(2026-10 작업) 개요**: 이 계층 전체가 "KRX"/"NASDAQ" 2분류에서 "KOSPI"/"KOSDAQ"/"NASDAQ"/"NYSE" 원자(atomic) 거래소 4분류로 바뀌었다. 핵심 설계 원칙은 **"원자 market(레코드·캐시 PK에 쓰임)"과 "합성 scan_key(백그라운드 스캔 버킷에만 쓰임)"를 분리**하는 것이다 — "KRX"(코스피+코스닥)/"US"(나스닥+NYSE)/"SP500"/"DOW" 같은 합성 개념은 더 이상 `OHLCVRecord.market`에 들어가지 않고, `data/service.py::get_universe_union`과 `orchestrator/market_buckets.py`(이 문서 범위 밖, `orchestrator` 리뷰 참고) 레벨에서만 존재한다. 이렇게 분리한 이유는 합성값을 레코드/캐시 PK에 허용하면 "코스피 종목을 KRX로 캐싱했다가 KOSPI로 재조회하면 캐시미스가 나는" 식의 애매함이 생기기 때문이다.

이 계층은 **에이전트가 아니라 순수 인프라 모듈**이다. 시세·공시·뉴스 원본 데이터를 외부 API에서 가져와 다른 모든 모듈(규칙엔진, ML 예측, LLM 분석, 챗봇)에 공급하는 역할만 하고, 그 데이터를 해석하거나 판단하는 로직은 전혀 포함하지 않는다.

**신규 보조신호(2026-10 작업) 개요**: 챗봇에 두 가지 신규 보조신호가 추가되면서 이 계층에도 소스 모듈 2개가 추가됐다 — `sources/investor_flow.py`(외국인/기관 순매수·공매도, 국내 전용)와 `sources/etf_source.py`(ETF 구성종목 집계 분석을 위한 holdings/유니버스 조회, 국내 전용). 둘 다 챗봇의 신규 tool(`analyze_supply_demand`/`analyze_etf_composition`, `06_chat_agent.md` 참고)이 소비한다.

전체 구조:

```
src/auto_stock/data/
├── models.py                       # OHLCV/공시/뉴스 레코드 스키마
├── cache.py                        # DuckDB 기반 OHLCV 캐시
├── service.py                      # 공개 인터페이스 (캐시 우선 조회 등)
├── scheduler.py                    # APScheduler 기반 장중 폴링
├── market_routing.py                # 원자 market -> 공시/뉴스 소스 라벨 매핑(신규)
└── sources/
    ├── fdr_source.py                # FinanceDataReader — KOSPI/KOSDAQ/NASDAQ/NYSE OHLCV
    ├── pykrx_source.py              # pykrx — KOSPI/KOSDAQ 종목리스트·시가총액·소속시장 역조회
    ├── dart_source.py               # DART(국내 전자공시) OpenAPI
    ├── edgar_source.py              # SEC EDGAR(미국 전자공시) OpenAPI — 거래소 구분 포함
    ├── naver_news_source.py         # 네이버 뉴스 검색 API
    ├── gdelt_source.py              # GDELT DOC 2.0 API
    ├── sector_classification.py     # pykrx 지수 API — 국내(코스피/코스닥) 업종 분류
    ├── index_quote.py               # 지수 자체 시세(VIX/DJI/IXIC/S&P500/RUT) 조회(신규)
    ├── index_constituents.py        # S&P500/다우존스 구성종목 조회(신규)
    ├── investor_flow.py             # pykrx — 외국인/기관 순매수, 공매도 비중(신규, 국내 전용)
    └── etf_source.py                # pykrx+FDR — 국내 ETF 구성종목(holdings)·유니버스 조회(신규)
```

참고: `src/auto_stock/data/__init__.py`와 `src/auto_stock/data/sources/__init__.py`는 둘 다 빈 파일이다(네임스페이스 패키지 선언 역할만).

---

## `models.py` — 공용 데이터 스키마

이 계층 전체, 그리고 이 계층을 소비하는 다른 모든 모듈이 공유하는 유일한 데이터 형태를 정의한다. 세 가지 `frozen=True, slots=True` dataclass가 있다.

### `OHLCVRecord`

```python
@dataclass(frozen=True, slots=True)
class OHLCVRecord:
    ticker: str
    market: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
```

시스템 전체에서 쓰는 유일한 시세 데이터 형태(`models.py:11-26`). `__post_init__`에서 두 가지를 검증한다:

- `market`이 `VALID_MARKETS = frozenset({"KOSPI", "KOSDAQ", "NASDAQ", "NYSE"})`에 속하지 않으면 `ValueError`(`models.py:8`).
- `high < low`이면 `ValueError`(가격 데이터 자체의 논리적 무결성 검증).

`frozen=True`이므로 생성 후 수정 불가능 — 캐시에 넣고 뺄 때도 항상 새 인스턴스를 만든다. 이 검증 덕분에 "잘못된 market 문자열이 들어있는 레코드가 하류 모듈까지 전파되는" 사고를 생성 시점에 차단한다.

**market 세분화(2026-10)**: `VALID_MARKETS`는 원래 `{"KRX", "NASDAQ"}` 2개였으나 코스피/코스닥 분리, 나스닥/NYSE 분리 요구에 따라 원자 거래소 4개로 확장됐다(`models.py:4-8` 주석에 설계 근거 명시). **"KRX"/"US"는 이제 `OHLCVRecord.market`의 유효값이 아니다** — 이 둘은 각각 "코스피+코스닥 합성", "나스닥+NYSE 합성"을 가리키는 scan_key 전용 라벨로 의미가 바뀌었고, 레코드 레벨에서 쓰려고 하면 `ValueError`가 난다. 이 선택이 왜 중요한지는 `cache.py`의 `(ticker, market, date)` PRIMARY KEY와 직결된다 — 합성값을 레코드에 허용하면 "코스피 종목을 KRX로 캐싱한 뒤 KOSPI로 재조회하면 캐시미스" 같은 모순이 생기므로, 원자값만 레코드/캐시에 들어가도록 강제한다.

### `DisclosureRecord`

```python
@dataclass(frozen=True, slots=True)
class DisclosureRecord:
    corp_code: str
    ticker: str
    report_name: str  # DART report_nm
    filed_date: date
    remark: str  # DART rm (정정/첨부 등 플래그, 없으면 "")
```

DART(`dart_source.py`)와 EDGAR(`edgar_source.py`)가 공통으로 생성하는 공시 레코드 형태. `corp_code` 필드는 DART에서는 실제 corp_code, EDGAR에서는 CIK가 들어간다 — 두 소스가 서로 다른 식별자 체계를 쓰지만 하류(뉴스/공시 분석 모듈)에서는 "그 공시 시스템이 쓰는 식별자" 하나로만 다루면 되므로 필드명을 통일했다. 검증 로직은 없다(생성자에 `__post_init__` 없음).

### `NewsArticle`

```python
@dataclass(frozen=True, slots=True)
class NewsArticle:
    ticker: str
    market: str
    title: str
    url: str
    published_at: date
    source: str  # 네이버: 언론사 도메인(originallink에서 추출), GDELT: domain 필드
```

네이버(`naver_news_source.py`)와 GDELT(`gdelt_source.py`)가 공통으로 생성하는 뉴스 기사 레코드. `source` 필드의 의미가 두 소스에서 서로 다른 방식으로 채워진다는 점에 유의 — 네이버는 원문 링크의 도메인을 `urlparse`로 추출한 값, GDELT는 API 응답의 `domain` 필드를 그대로 쓴 값이다.

---

## `cache.py` — DuckDB 기반 OHLCV 캐시

### `OHLCVCache` 클래스

생성자: `__init__(self, db_path: str | Path)` (`cache.py:28-31`)

**목표/역할**: 시세 데이터를 로컬 DuckDB 파일(`data/ohlcv.duckdb`)에 저장해, 같은 종목·기간을 반복 조회할 때 외부 API(FDR)를 다시 호출하지 않도록 하는 캐시 계층. `docs/design/data-collection-layer.md` §1에서 DuckDB를 선택한 이유는 "전체 상장종목 × 5년치 OHLCV에 대한 컬럼 단위 집계·백테스트 쿼리가 주 워크로드"이기 때문이며, SQLite보다 컬럼형 저장·분석 쿼리에 강하다는 설계 근거가 있다.

**상세 동작**: 생성 시 `duckdb.connect(str(db_path))`로 커넥션을 열고, `_SCHEMA`(`cache.py:9-22`)를 실행해 `ohlcv` 테이블이 없으면 생성한다. 스키마는 `(ticker, market, date)` 복합 PRIMARY KEY를 가지며, `updated_at` 컬럼이 `DEFAULT current_timestamp`로 자동 채워진다. `self._write_lock = threading.Lock()`을 인스턴스 속성으로 둔다.

**설계상 비자명한 결정**: 실제 운영 환경(설계 당시 문서의 `data/cache.duckdb`가 아니라 실제로는 `data/ohlcv.duckdb` 경로가 쓰임 — `scripts/chat_cli.py:146` 참고)에서 멀티스레드 접근을 고려해 쓰기 락을 둔다.

#### `cursor(self) -> "OHLCVCache"`

**목표/역할**: 같은 DuckDB 파일을 가리키는 스레드 전용 커넥션을 가진 새 `OHLCVCache` 인스턴스를 반환한다(`cache.py:33-43`). DuckDB 공식 패턴인 `Connection.cursor()`를 활용 — 새 DB를 만드는 게 아니라 같은 DB에 대한 별도 커넥션이다.

**상세 동작**: `object.__new__(OHLCVCache)`로 `__init__`을 거치지 않는 clone을 만들고, `clone._con = self._con.cursor()`로 스레드 전용 커넥션을 할당한다. 핵심은 `clone._write_lock = self._write_lock`(원본과 **같은** Lock 객체를 공유) — 새 Lock을 만들지 않는다. 이렇게 해야 원본 인스턴스와 clone이 동시에 같은 파일에 쓰기 작업을 할 때 서로를 막을 수 있다. 주석에 따르면 이 메서드는 "온디맨드 배치 스캔 트리거"(`majestic-waddling-breeze.md` — 리포에는 이 파일이 없으나 주석에 언급됨) 계획에서, 백그라운드 스레드가 메인 스레드와 동시에 안전하게 캐시를 읽고 쓰기 위해 쓰이도록 설계됐다.

#### `put(self, records: list[OHLCVRecord]) -> None`

**목표/역할**: 레코드 리스트를 캐시에 적재(upsert)한다(`cache.py:45-61`).

**상세 동작**: `self._write_lock`으로 전체 루프를 감싼 뒤, 레코드마다 `INSERT ... ON CONFLICT (ticker, market, date) DO UPDATE SET ...` 쿼리를 실행한다. 즉 같은 (ticker, market, date) 조합이 이미 있으면 OHLCV 값과 `updated_at`을 덮어쓰고, 없으면 새로 삽입한다. 레코드 수만큼 개별 `execute`를 호출하는 방식이라, 대량 데이터(예: 200종목×5년치)를 적재할 때는 배치 insert보다 느릴 수 있다는 점은 명시적으로 최적화되어 있지 않다.

#### `get(self, ticker: str, market: str, start: date, end: date) -> list[OHLCVRecord]`

**목표/역할**: 지정 구간의 캐시된 레코드를 날짜순으로 조회한다(`cache.py:63-79`).

**상세 동작**: `WHERE ticker = ? AND market = ? AND date BETWEEN ? AND ? ORDER BY date` 쿼리 결과를 `OHLCVRecord` 리스트로 변환해 반환한다. 캐시에 아무것도 없으면 빈 리스트.

#### `covers(self, ticker: str, market: str, start: date, end: date) -> bool`

**목표/역할**: 요청한 구간을 캐시가 이미 완전히 포함하는지 판단해, `service.get_ohlcv`가 소스 API를 다시 호출할지 결정하는 데 쓰인다(`cache.py:81-90`).

**상세 동작**: `SELECT MIN(date), MAX(date) FROM ohlcv WHERE ticker = ? AND market = ?`로 그 종목의 캐시 범위를 가져온 뒤, `min_date <= start and max_date >= end`인지 확인한다. 캐시가 비어 있으면(`min_date is None`) `False`.

**엣지케이스/주의점(근사치 판정)**: 이 메서드는 docstring에서도 "근사치"라고 명시하듯, 캐시의 최소/최대 날짜만 보고 **구간 중간에 빠진 날짜(거래 공백일 등)가 없는지는 검증하지 않는다**. 예를 들어 캐시에 2024-01-01과 2024-01-31 레코드만 있고 그 사이가 전부 비어 있어도, `start=2024-01-01, end=2024-01-31` 요청에 대해 `covers`는 `True`를 반환한다. 실제 거래일 사이의 공백은 보통 공휴일 때문에 자연스럽게 발생하므로 큰 문제가 되진 않지만, 데이터 소스 쪽 결함으로 중간 구간이 통째로 빠진 경우를 이 로직은 탐지하지 못한다는 점은 알아둬야 한다.

---

## `service.py` — 공개 인터페이스

데이터 계층을 쓰는 다른 모듈(오케스트레이터, 챗봇 tool 등)이 직접 호출하는 창구. 세 함수 모두 `cache.py`/`sources/`를 조합해 더 단순한 인터페이스를 제공한다.

#### `get_ohlcv(cache: OHLCVCache, ticker: str, start: date, end: date, market: str) -> list[OHLCVRecord]`

**목표/역할**: "캐시 우선 조회, 미스 시 소스 호출 후 캐시 적재"라는 전체 데이터 계층의 핵심 계약을 구현한다(`service.py:10-15`).

**상세 동작**: `cache.covers(...)`가 `False`면 `fdr_source.fetch_ohlcv(...)`를 호출해 가져온 레코드를 `cache.put(...)`으로 적재한 뒤, 항상 `cache.get(...)`으로 최종 결과를 반환한다. 즉 캐시 히트 여부와 관계없이 **반환값은 항상 캐시에서 읽은 값**이다 — 캐시 미스 시 방금 받아온 API 응답을 그대로 반환하지 않고, 한 번 캐시를 거쳐서 반환한다는 점에 유의. 이 설계 덕분에 반환되는 레코드 형태(순서, 중복 제거 등)가 항상 `cache.get`의 보장(날짜순 정렬)을 따른다.

#### `get_universe(market: str, as_of: date | None = None) -> list[str]`

**목표/역할**: 원자(atomic) 거래소 하나의 전체 상장종목 코드 리스트를 반환한다(`service.py:18-29`).

**상세 동작**: `market in ("KOSPI", "KOSDAQ")`이면 `pykrx_source.get_ticker_list(as_of or date.today(), market=market)`를 호출(각 서브마켓을 명시적으로 지정 — 과거엔 `market="ALL"`로 KOSPI+KOSDAQ을 한 번에 받아왔으나, 이제는 호출자가 KOSPI/KOSDAQ 중 어느 쪽인지 지정한다). `market in ("NASDAQ", "NYSE")`이면 `fdr.StockListing(market)`을 직접 호출해 `"Symbol"` 컬럼을 리스트로 반환(이 경로는 `fdr_source.py`를 거치지 않고 `service.py`가 `FinanceDataReader`를 직접 import해서 쓴다 — `service.py:3`). FDR 소스코드를 확인한 결과 `StockListing("NASDAQ")`와 `StockListing("NYSE")`는 둘 다 동일한 `NaverStockListing` 클래스로 라우팅되는 동형 스키마다. **"KRX"/"US" 같은 합성 scan_key는 이 함수가 받지 않는다** — 그 외 market 값이면 `ValueError(f"unsupported market: {market!r}")`.

**설계상 비자명한 점**: KOSPI/KOSDAQ과 NASDAQ/NYSE의 유니버스 조회 경로가 비대칭적이다 — 국내는 전용 소스 모듈(`pykrx_source`)을 거치지만 미국은 `fdr` 라이브러리를 바로 호출한다. 이는 `fdr_source.py`에 미국용 유니버스 조회 함수가 따로 없기 때문으로 보인다(해당 파일에는 `fetch_ohlcv`만 있음).

#### `get_universe_union(markets: list[str], as_of: date | None = None) -> list[str]`

**목표/역할**: 여러 원자 market의 유니버스를 합집합으로 반환한다(`service.py:32-43`) — "KRX"(코스피+코스닥)/"US"(나스닥+NYSE) 같은 **합성 scan_key가 조립되는 지점**이다. `orchestrator/market_buckets.py`가 이 함수를 호출해 합성 버킷의 유니버스를 만든다(이 문서 범위 밖, `orchestrator` 리뷰 참고).

**상세 동작**: `markets` 리스트를 순회하며 각 원자 market에 `get_universe(market, as_of)`를 호출하고, `seen` set으로 중복을 걸러내며 **처음 등장한 순서를 보존**해 하나의 리스트로 합친다 — `sector_classification.py::fetch_constituents`가 쓰는 "순서 보존 합집합" 패턴과 동일한 관용구다.

#### `refresh_recent(cache: OHLCVCache, tickers: list[str], market: str, lookback_days: int = 5) -> None`

**목표/역할**: 폴링 잡(`scheduler.py`)이 주기적으로 호출해 최근 N일치 시세를 근접-실시간으로 갱신하는 함수(`service.py:46-52`).

**상세 동작**: `end = date.today()`, `start = end - timedelta(days=lookback_days)`로 구간을 정하고, 티커마다 `fdr_source.fetch_ohlcv(...)`를 **캐시 커버리지 확인 없이 무조건 호출**한 뒤 `cache.put(...)`한다. `get_ohlcv`와 달리 캐시 우선 조회 로직을 거치지 않는 이유는, 이 함수의 목적 자체가 "최근 데이터는 항상 최신으로 갱신"이기 때문이다(캐시에 이미 있어도 최신값으로 덮어써야 함).

---

## `scheduler.py` — 장중 폴링 인프라

**중요 — 현재 상태**: `docs/CODEBASE_REVIEW.md`에 명시된 대로, 이 파일은 **정의만 되어 있고 어떤 스크립트/오케스트레이터에서도 실제로 호출되지 않는다**. 즉 지금 코드베이스에서 데이터는 수동 트리거(스크립트 실행, 챗봇 tool 호출 등)로만 갱신되며, 장중 자동 폴링은 아직 실행 경로에 연결되지 않았다.

#### `is_market_open(market: str, now: datetime | None = None) -> bool`

**목표/역할**: 시장별 정규장 운영 여부를 판단한다(`scheduler.py:16-24`).

**상세 동작**: `MARKET_HOURS` 딕셔너리(`scheduler.py:14-19`)에서 시장별 타임존·개장/폐장 시각을 가져온다 — **원자 market 4개**로 키가 늘었다(과거엔 KRX/NASDAQ 2개):
- KOSPI/KOSDAQ: `Asia/Seoul`, 09:00–15:30 (둘 다 한국거래소 정규장이라 동일한 값을 중복해서 가짐)
- NASDAQ/NYSE: `America/New_York`, 09:30–16:00 (둘 다 미국 정규장이라 동일한 값을 중복해서 가짐)

`scheduler.py:10-13` 주석에 이 중복이 의도적임이 명시돼 있다 — `is_market_open(market)`이 원자 market 하나만 받는 계약을 유지하기 위해, 거래시간이 같아도 각 거래소를 별도 키로 둔다.

`now`가 주어지지 않으면 해당 타임존의 현재 시각을 쓰고, 주어졌다면 `.astimezone(tz)`로 해당 타임존으로 변환한다(타임존 인지 — 호출자가 naive datetime을 넘겨도 올바르게 변환된다는 전제 하에 동작). `weekday() >= 5`(토/일)면 바로 `False`. 그 외에는 `open <= local_now.time() <= close` 비교. **공휴일은 고려하지 않는다** — 설날/추석 등 평일 휴장일에도 이 함수는 `True`를 반환한다는 한계가 있다.

#### `poll_market(cache: OHLCVCache, market: str, get_tickers: Callable[[str], list[str]], now: datetime | None = None) -> None`

**목표/역할**: 장 시간에만 대상 종목 시세를 갱신하는 폴링 잡의 실제 작업 함수(`scheduler.py:27-37`).

**상세 동작**: `is_market_open`이 `False`면 즉시 반환(아무 일도 안 함). `True`면 `get_tickers(market)`로 대상 티커 목록을 얻고 `service.refresh_recent(cache, tickers, market=market)`를 호출. `get_tickers`가 콜백으로 주입되는 이유는 "어떤 종목을 폴링할지"(전체 유니버스? 워치리스트? 보유 종목?)를 이 모듈이 알 필요가 없게 분리하기 위함으로 보인다.

#### `register_polling_jobs(scheduler: BackgroundScheduler, cache: OHLCVCache, get_tickers: Callable[[str], list[str]], interval_minutes: int = 5) -> None`

**목표/역할**: `MARKET_HOURS`에 정의된 각 시장마다 별도의 APScheduler interval job을 등록한다(`scheduler.py:40-55`).

**상세 동작**: `MARKET_HOURS`를 순회하며 `scheduler.add_job(poll_market, trigger="interval", minutes=interval_minutes, args=[cache, market, get_tickers], id=f"poll_{market}", replace_existing=True)`를 호출. `id`를 시장별로 고정하고 `replace_existing=True`를 줘서, 이 함수를 여러 번 호출해도 같은 시장의 job이 중복 등록되지 않고 교체된다. `interval_minutes` 기본값 5분은 docstring에서 "실제 API 응답을 보며 튜닝할 값"이라고 명시 — 확정값이 아니다. market 세분화 이후 `MARKET_HOURS`가 4개 키를 가지므로, 이 함수를 호출하면 자동으로 4개(KOSPI/KOSDAQ/NASDAQ/NYSE) 폴링 job이 등록된다 — 함수 자체는 수정되지 않았고 `MARKET_HOURS` 딕셔너리 확장만으로 범위가 넓어졌다.

---

## `sources/fdr_source.py` — FinanceDataReader 래퍼

#### `fetch_ohlcv(ticker: str, start: date, end: date, market: str) -> list[OHLCVRecord]`

**목표/역할**: 국내(KOSPI/KOSDAQ)·미국(NASDAQ/NYSE) 4개 원자 거래소 전부의 시세 데이터를 동일한 방식으로 조회하는 시세 데이터의 주 소스(`fdr_source.py:8-26`). `market` 분기 로직이 전혀 없다는 점이 특징 — `FinanceDataReader`가 종목코드 포맷만으로 내부적으로 시장을 식별하기 때문에, 이 함수는 `market` 파라미터를 받아 결과 `OHLCVRecord`에 그대로 채워넣기만 하고 조회 로직 자체에는 쓰지 않는다. market 세분화로 원자값이 2개에서 4개로 늘었어도 이 함수는 **한 글자도 수정되지 않았다** — `market` 파라미터를 그대로 투과시키는 설계였기 때문.

**상세 동작**: `fdr.DataReader(ticker, start, end)`로 pandas DataFrame을 받아, 각 행을 `OHLCVRecord`로 변환한다. `index.date()`로 날짜를 추출하고, Open/High/Low/Close는 `float()`, Volume은 `int()`로 캐스팅한다.

**엣지케이스**: FDR이 빈 DataFrame을 반환하면(예: 상장 전 기간 조회) 빈 리스트를 반환한다 — 예외를 던지지 않음. 반대로 FDR 자체가 네트워크 오류 등으로 예외를 던지면 이 함수는 그 예외를 그대로 전파한다(별도 예외 타입으로 감싸지 않음 — DART/EDGAR/네이버/GDELT 소스들과 달리 이 파일에는 전용 에러 클래스가 없다).

---

## `sources/pykrx_source.py` — pykrx 래퍼

파일 최상단(`pykrx_source.py:13-14`)에서 `with contextlib.redirect_stdout(io.StringIO()): from pykrx import stock`로 `pykrx` 모듈을 import한다.

**설계상 비자명한 결정(중요 버그 수정 이력)**: `pykrx`는 모듈이 **import되는 순간** `website/comm/webio.py`의 모듈 최상위 코드가 KRX 로그인을 시도한다. `KRX_ID`/`KRX_PW`가 아직 로드되지 않은 상태(즉 import 시점에는 `.env`가 읽히기 전)에서 로그인에 실패하면 "KRX 로그인 실패..." 메시지가 stdout에 그대로 출력된다. 이 프로젝트는 그 로그인이 필요한 프리미엄 엔드포인트를 쓰지 않고 공개 OHLCV/티커 목록만 쓰기 때문에 실제 동작에는 영향이 없지만, 사용자가 챗봇 실행 중 이 메시지를 보고 혼란을 겪어 문의한 적이 있다(2026-09-06). 그래서 **import 시점의 출력만** `redirect_stdout`로 억제하고, 함수 호출 시점의 출력(있다면)은 그대로 둔다.

#### `get_ticker_list(as_of: date, market: str = "ALL") -> list[str]`

**목표/역할**: 전체 상장종목 코드 리스트(국내)를 반환한다(`pykrx_source.py:24-27`). `market`은 pykrx 자체의 파라미터값(`"ALL"`/`"KOSPI"`/`"KOSDAQ"`/`"KONEX"`)을 그대로 투과시킨다 — 이 프로젝트의 market 세분화 작업으로 `data/service.py::get_universe`가 `market="KOSPI"`/`market="KOSDAQ"`를 명시적으로 지정해서 호출하게 됐다(과거엔 `"ALL"` 기본값으로 코스피+코스닥을 한꺼번에 받았음).

**상세 동작**: 함수 호출 시점에 `load_dotenv()`를 명시적으로 다시 호출한다 — import 시점에는 아직 `.env`가 로드되지 않은 상태라 최초 로그인이 늘 실패하지만, 함수가 실제로 호출되는 시점(앱 진입점에서 `.env`가 이미 로드된 뒤)에는 재로그인이 성공한다는 것을 전제로 한 패턴이다. `stock.get_market_ticker_list(as_of.strftime("%Y%m%d"), market=market)`를 그대로 반환.

#### `get_market_cap(ticker: str, as_of: date) -> int | None`

**목표/역할**: 특정 종목·특정일자의 시가총액을 조회(`pykrx_source.py:30-37`).

**상세 동작**: `stock.get_market_cap(date_str, date_str, ticker)`로 DataFrame을 받아, 비어 있으면 `None`, 아니면 `df.iloc[-1]["시가총액"]`을 `int()`로 캐스팅해 반환. 컬럼명이 한글(`"시가총액"`)이라는 점에 유의 — pykrx 라이브러리 자체의 응답 스키마를 그대로 쓰고 있다.

#### `resolve_market(ticker: str, cache_path=TICKER_MARKET_CACHE_PATH, refresh_days=TICKER_MARKET_REFRESH_DAYS) -> str | None` (신규, market 세분화 작업)

**목표/역할**: 국내 종목코드가 KOSPI/KOSDAQ 중 어디에 상장돼 있는지 역조회한다(`pykrx_source.py:73-82`) — `chat_agent/ticker_resolution.py`가 "6자리 숫자"라는 패턴만으로는 코스피/코스닥을 구분할 수 없어서 신설됐다.

**상세 동작**: `_download_ticker_market_map()`(`pykrx_source.py:47-57`)이 `("KOSPI", "KOSDAQ")` 각 서브마켓에 대해 `stock.get_market_ticker_list(오늘날짜, market=sub_market)`를 호출해, 그 결과를 `{ticker: sub_market}` 형태로 뒤집은 역방향 매핑을 만든다 — 이 역조회가 가능한 전제는 "KOSPI와 KOSDAQ에 동시 상장되는 종목은 없다"는 KRX 상장 규정이다(함수 docstring에 명시). `_load_or_refresh_ticker_market_map`이 이 매핑을 `data/krx_ticker_market.json`에 7일 TTL로 캐싱한다.

**설계상 비자명한 결정(원자적 쓰기 패턴 재사용)**: 캐시 저장은 DART/EDGAR 캐시와 동일한 "임시파일(`.tmp`) 쓰기 → `os.replace()`로 교체" 패턴을 쓴다(`pykrx_source.py:65-69` 주석에 "DART/EDGAR 캐시와 동일한 원자적 쓰기 패턴"이라고 명시) — 이 역조회 캐시가 세 번째로 이 패턴을 채택한 사례다. 캐시에 없는 티커(코넥스, 상장폐지 등)는 에러가 아니라 `None`을 반환한다 — `chat_agent/ticker_resolution.py`가 이 `None`을 받으면 거래소를 단정하지 않고 합성 라벨 `"KRX"`로 폴백 처리한다(해당 모듈 리뷰 참고).

---

## `sources/dart_source.py` — DART(국내 전자공시) OpenAPI 연동

DART API는 KRX 종목코드가 아니라 자체 `corp_code`로 종목을 식별하므로, 이 모듈은 전체 매핑(`corpCode.xml`, zip 압축)을 다운로드해 로컬 JSON 캐시(`data/dart_corp_codes.json`, 기본 7일 갱신)로 관리하고 종목코드로 역조회하는 역할을 한다(`docs/design/news-disclosure-plan.md` 핵심 설계 결정 3 — 해당 파일은 리포에 존재하지 않으나 소스 주석에 참조됨).

`DART_API_KEY`는 `notifier/credentials.py`와 동일한 패턴으로 필요 시점에 직접 `load_dotenv()`를 호출해 로드한다(`_load_dart_api_key`, `dart_source.py:40-42`).

### 캐시 다운로드/관리 내부 함수

#### `_download_corp_code_map() -> dict[str, dict[str, str]]`

**목표/역할**: DART의 `corpCode.xml`(zip)을 다운로드해 `{stock_code: {"corp_code": ..., "corp_name": ...}}` 매핑으로 파싱한다(`dart_source.py:45-83`). `corp_name`은 뉴스 감성분석(#4-뉴스) 확장에서 뉴스 검색 쿼리 생성에 쓰이는데, 애초에는 파싱만 하고 버리던 필드를 나중에 재활용하게 된 것이다.

**상세 동작**: `requests.get(CORP_CODE_URL, params={"crtfc_key": api_key}, timeout=30)` → zip 응답을 `zipfile.ZipFile`로 열어 첫 엔트리를 읽고 → `defusedxml.ElementTree.fromstring`으로 파싱(표준 `xml.etree` 대신 `defusedxml`을 쓰는 이유는 XXE(XML External Entity) 공격 방어). `<list>` 항목마다 `stock_code`/`corp_code`/`corp_name`을 추출하되, `stock_code`가 빈 문자열인 항목(비상장 등)은 매핑에서 제외한다 — 종목코드로 역조회할 일이 없기 때문.

**에러 처리**:
- `requests.exceptions.RequestException`이 발생하면 `DartApiError(f"DART API 요청 실패: {type(exc).__name__}")`로 다시 던진다(`from exc`로 원인 체인은 유지). **원본 예외 메시지를 그대로 노출하지 않고 타입명만 남기는 이유**: `requests`/`urllib3`는 예외 문자열에 요청 URL을 그대로 포함시키는데, 이 URL의 쿼리스트링에는 `crtfc_key`(API 키)가 담겨 있다 — 보안 리뷰에서 CRITICAL로 지적된 사항.
- DART가 HTTP 200으로 응답했지만 XML의 `<status>`가 `"000"`(정상) 이외의 값이면 `DartApiError`를 던진다(`dart_source.py:67-74`). **이 분기가 중요한 이유**: 잘못되거나 폐기된 API 키를 넣으면 DART는 HTTP 오류를 내지 않고 `<list>` 항목이 없는 "정상처럼 보이는" zip을 돌려준다. 이를 빈 매핑으로 오인하고 그대로 캐싱하면, 뉴스/공시 신호가 `CORP_CODE_REFRESH_DAYS`(7일)간 **에러 기록 하나 없이 조용히 무력화**된다 — 코드 리뷰에서 HIGH로 지적되어 수정된 사항이다.

#### `_is_cache_fresh(cache_path: Path, refresh_days: int) -> bool`

파일이 없으면 `False`, 있으면 `time.time() - mtime < refresh_days * 86400`으로 신선도 판단(`dart_source.py:86-90`).

#### `_load_or_refresh_corp_map(cache_path: Path, refresh_days: int) -> dict`

캐시가 신선하면 JSON 파일을 그대로 읽어 반환, 아니면 `_download_corp_code_map()`을 호출해 새로 받은 매핑을 저장한 뒤 반환(`dart_source.py:93-104`).

**설계상 비자명한 결정(원자적 쓰기)**: 캐시 저장을 `cache_path.write_text(...)`로 직접 하지 않고, 임시 파일(`.tmp`)에 먼저 쓴 뒤 `os.replace(tmp_path, cache_path)`로 교체한다. `write_text()` 도중 프로세스가 죽으면(OOM, 컨테이너 강제종료 등) 절반만 쓰인 손상된 캐시 파일이 남아, `refresh_days` 동안 자동으로 복구되지 않고 매번 그 손상된 JSON을 읽으려다 실패하는 문제가 생길 수 있다 — 코드 리뷰 MEDIUM #1로 지적되어 이 패턴으로 수정됐다. `edgar_source.py`의 동일 함수도 처음부터 이 패턴으로 구현됐다(DART 쪽 리뷰에서 배운 교훈을 EDGAR 구현 시 선제적으로 적용).

#### `_corp_code_and_name(entry) -> tuple[str | None, str | None]`

**목표/역할**: 캐시 항목에서 `(corp_code, corp_name)`을 추출하되, 신/구 스키마를 모두 지원하는 하위호환 헬퍼(`dart_source.py:107-116`).

**상세 동작**: `entry`가 `None`이면 `(None, None)`. `entry`가 문자열이면(뉴스 감성분석 확장 전, `corp_name`이 없던 시절에 저장된 구 스키마 캐시) `(entry, None)`을 반환 — 즉 그 문자열 자체가 corp_code였던 시절의 캐시 포맷. `entry`가 dict면 `(entry.get("corp_code"), entry.get("corp_name"))`을 반환. `refresh_days`가 지나 캐시가 재다운로드되면 자연스럽게 새 스키마로 바뀐다.

### 공개 함수

#### `resolve_corp_code(ticker: str, cache_path=CORP_CODE_CACHE_PATH, refresh_days=CORP_CODE_REFRESH_DAYS) -> str | None`

`corpCode.xml` 캐시에서 티커로 corp_code를 역조회한다(`dart_source.py:119-130`). 캐시가 오래되면 재다운로드. **나스닥 등 DART에 등록되지 않은 티커는 에러가 아니라 `None`을 반환**한다 — 이는 "DART에 없는 종목"이 정상적으로 발생할 수 있는 상황이기 때문(KRX/NASDAQ을 같은 파이프라인에서 다루는 이 시스템의 특성).

#### `resolve_corp_name(ticker: str, ...) -> str | None`

같은 캐시에서 회사명을 역조회한다(`dart_source.py:133-145`). 뉴스 검색 API는 티커가 아니라 회사명 문자열로 질의해야 하므로, 그 쿼리 입력을 만드는 데 쓰인다. `resolve_corp_code`와 캐시(다운로드/저장)를 공유하므로, 같은 티커에 대해 두 함수를 순서대로 호출해도 DART API를 두 번 부르지 않는다(둘 다 내부적으로 `_load_or_refresh_corp_map`을 호출하고, 두 번째 호출 시점엔 이미 캐시가 신선하므로 네트워크 요청이 발생하지 않음).

#### `resolve_ticker_by_name(name_query: str, ...) -> list[tuple[str, str]]`

**목표/역할**: 회사명 문자열로 종목코드를 역조회한다(`resolve_corp_name`의 반대 방향) — 대화형 챗봇이 사용자가 입력한 회사명을 종목코드로 바꿀 때 쓴다(`dart_source.py:148-173`). 새로운 DART API 호출 없이 기존 캐시를 재사용.

**상세 동작**:
1. 캐시의 모든 항목 중 `corp_name`이 있는 것만 `named_entries`로 수집(구 스키마의 문자열 항목은 제외).
2. 대소문자를 무시한(`casefold()`) **완전일치**가 하나라도 있으면 그 목록만 반환.
3. 완전일치가 없으면 **부분문자열 포함** 여부로 폴백해 반환.

**설계 의도**: 완전일치를 우선하는 이유는, "삼성전자"로 질의했을 때 부분일치 후보("삼성전자우" 등)까지 섞여 들어가는 걸 막기 위함(`dart_source.py:157-159` 주석).

#### `fetch_disclosures(corp_code: str, start: date, end: date) -> list[DisclosureRecord]`

**목표/역할**: DART 공시검색(`list.json`) API를 호출해 지정 기간의 공시 목록을 가져온다(`dart_source.py:180-221`).

**상세 동작**: `crtfc_key`/`corp_code`/`bgn_de`/`end_de`(YYYYMMDD)/`page_no=1`/`page_count=100`을 파라미터로 GET 요청. 응답 JSON의 `status`가 `"013"`(데이터 없음)이면 빈 리스트 반환(에러 아님). `"000"`(정상)이 아닌 다른 status면 `DartApiError`. 정상이면 `list` 배열의 각 항목을 `DisclosureRecord`로 변환.

**엣지케이스(JSON null 방어)**: `report_name=item.get("report_nm") or ""`와 `remark=item.get("rm") or ""`처럼 `or ""`를 붙인 이유가 명시돼 있다(`dart_source.py:213-215`) — `.get(key, "")`는 키가 **아예 없을 때만** 기본값을 쓰고, DART가 명시적으로 JSON `null`을 보내면 `None`을 그대로 반환해 `DisclosureRecord`의 `str` 타입 계약을 조용히 깨뜨린다. `or ""`는 두 경우(키 없음, null 값) 모두 방어한다 — 코드 리뷰 MEDIUM #2로 지적되어 수정된 사항.

---

## `sources/edgar_source.py` — SEC EDGAR(미국 전자공시) OpenAPI 연동

`dart_source.py`와 대칭 구조. EDGAR는 CIK(Central Index Key, 10자리 0-패딩)로 종목을 식별하며, 전체 매핑(`company_tickers.json`)을 다운로드해 로컬 캐시(`data/edgar_cik_map.json`, 기본 7일 갱신)로 관리한다. DART와 달리 **API 키가 아니라 SEC 정책상 필수인 `User-Agent`**(연락처 포함 식별 문자열)가 필요하다 — 비밀 정보는 아니지만, 없으면 즉시 실패하는 계약은 DART의 API 키와 동일하게 유지한다.

### `_download_cik_map() -> dict[str, dict[str, str]]`

`company_tickers.json`을 다운로드해 `{ticker: {"cik": "0000320193", "title": "Apple Inc.", "exchange": "NASDAQ"}}` 형태로 파싱한다(`edgar_source.py:72-101`). CIK는 `f"{int(cik):010d}"`로 10자리 0-패딩된다. `title`은 `resolve_corp_name`과 동일한 목적(뉴스 검색 쿼리 생성)으로 쓰인다.

**market 세분화(2026-10) — `exchange` 필드 추가**: 캐시 항목에 `exchange` 필드가 새로 추가됐다. SEC가 보내는 원본 `exchange` 값(`"nasdaq"`/`"nyse"`/`"nyse mkt"`/`"batsglobal"` 등 소문자 자유 문자열, 2026-10 실제 응답으로 확인됨)을 `_normalize_exchange()`(`edgar_source.py:62-69`)가 `_EXCHANGE_TO_MARKET = {"nasdaq": "NASDAQ", "nyse": "NYSE"}` 매핑으로 정규화한다. 매핑에 없는 거래소명(ETF 발행사 등)이면 `None`을 반환하고, 캐시에는 빈 문자열로 저장된다(`entry.get("exchange") or ""`, `edgar_source.py:98`) — **"NASDAQ으로 임의 단정하지 않는다"**는 것이 이 설계의 핵심이다(`_normalize_exchange` docstring에 명시).

**에러 처리**: `requests.exceptions.RequestException`을 `EdgarApiError(f"EDGAR API 요청 실패: {type(exc).__name__}")`로 재던짐 — DART와 동일한 "원본 예외 비노출" 원칙.

### `_is_cache_fresh`/`_load_or_refresh_cik_map`/`_cik_title_and_exchange`

DART의 대응 함수와 동일한 역할·동일한 원자적 쓰기 패턴(임시파일+`os.replace`)을 쓴다. 주석(`edgar_source.py:116-118`)에 "DART 쪽 리뷰에서 지적받았던 것과 같은 문제를 처음부터 방지한다"고 명시돼 있다 — DART에서 발견된 교훈이 EDGAR 구현에 선제 반영된 사례.

**market 세분화로 이름이 바뀜**: 과거 `_cik_and_title`(2-tuple 반환)이 `_cik_title_and_exchange`(`edgar_source.py:125-136`, 3-tuple 반환)로 바뀌었다. `entry`가 `None`이면 `(None, None, None)`, 문자열이면(구 스키마) `(entry, None, None)`, dict인데 `exchange` 키가 없으면(exchange 필드 추가 **이전** 시점에 저장된 중간 스키마) `entry.get("exchange") or None`이 `None`을 반환 — 즉 "구 스키마(문자열)"와 "exchange 없는 중간 스키마(dict)"를 모두 하위호환으로 처리한다. 둘 다 `refresh_days`(7일)가 지나면 자연스럽게 새 스키마로 갱신된다.

### `resolve_cik(ticker, ...) -> str | None` / `resolve_company_title(ticker, ...) -> str | None`

DART의 `resolve_corp_code`/`resolve_corp_name`과 동형(同形)이다(`edgar_source.py:139-165`). 한국 상장사 등 EDGAR에 없는 티커는 `None`(에러 아님).

### `resolve_exchange(ticker, cache_path=CIK_MAP_CACHE_PATH, refresh_days=CIK_MAP_REFRESH_DAYS) -> str | None` (신규, market 세분화 작업)

**목표/역할**: 영문 심볼이 실제로 NASDAQ인지 NYSE인지 역조회한다(`edgar_source.py:168-180`) — `chat_agent/ticker_resolution.py`가 "영문 1~5자"라는 패턴만으로는 NASDAQ/NYSE를 구분할 수 없어서 신설됐다. `resolve_cik`/`resolve_company_title`과 동일한 캐시를 공유하므로 추가 API 호출이 들지 않는다.

**상세 동작**: `_load_or_refresh_cik_map` → `_cik_title_and_exchange`에서 `exchange` 성분만 꺼내 반환. SEC가 모르는 거래소명을 보냈거나 캐시가 구/중간 스키마라 exchange 정보가 없으면 `None` — 호출부가 "NASDAQ/NYSE 둘 다 후보"로 보수적으로 처리해야 한다는 신호다.

**설계상 비자명한 결정(이 작업 중 발견된 과거 버그)**: `resolve_exchange`를 신설하고 실제 역조회를 해보기 전까지, `chat_agent/ticker_resolution.py`의 한글 별칭표(`_NASDAQ_KOREAN_ALIASES`라는 이름이었다)는 "애플"/"테슬라" 같은 통칭을 전부 `market="NASDAQ"`으로 하드코딩하고 있었다. 그런데 JPM(제이피모건)/V(비자)/GS(골드만삭스)/BA(보잉)/DIS(디즈니)/NKE(나이키)/JNJ(존슨앤존슨)/PFE(화이자)/KO(코카콜라)/WMT(월마트)/XOM(엑슨모빌)/BRK-B(버크셔해서웨이) 등은 실제로는 전부 NYSE 상장 종목이다 — 즉 그동안 챗봇이 이 종목들을 사용자에게 "NASDAQ 종목"으로 잘못 라벨링해 보여주고 있었다는 뜻이다. `resolve_exchange`로 EDGAR의 실제 `exchange` 필드를 역조회해보고서야 이 오분류가 드러났다. `chat_agent/ticker_resolution.py` 리뷰에서 이 수정의 상세(별칭표 자체는 티커만 매핑하도록 남기고, market은 매번 `resolve_exchange`로 조회하게 바꾼 경위)를 다룬다.

### `resolve_ticker_by_name(name_query, ...) -> list[tuple[str, str]]`

DART의 `resolve_ticker_by_name`과 동일한 매칭 규칙(완전일치 우선, 없으면 부분일치 폴백)을 그대로 따른다(`edgar_source.py:183-208`).

### `fetch_filings(cik: str, start: date, end: date) -> list[DisclosureRecord]`

**목표/역할**: SEC의 `submissions` API(`https://data.sec.gov/submissions/CIK{cik}.json`)를 호출해 공시 목록을 가져온다(`edgar_source.py:215-260`).

**상세 동작**: 이 API는 날짜 범위 파라미터를 지원하지 않고 최근 1년/1000건을 **컬럼형 배열**(`forms`와 `filingDates`가 각각 평행한 리스트)로 돌려주므로, 기간 필터링을 함수 내부에서 클라이언트 사이드로 수행한다. `zip(forms, filing_dates)`로 순회하며:

1. `form.split("/")[0] in _EXCLUDED_FORM_TYPES`(`{"3", "4", "5"}`)이면 건너뛴다 — Form 3/4/5(임원·주요주주 지분변동 신고)는 대형주 기준 주 단위로 여러 건씩 쏟아져 `MAX_DISCLOSURES_PER_QUERY` 윈도우를 노이즈로 채우기 때문에 기본 제외한다. `.split("/")[0]`을 쓰는 이유는 정정판(예: `"4/A"`)이 원본 폼 코드와 정확히 일치하지 않아 단순 `in` 체크로는 걸러지지 않기 때문 — 코드 리뷰 MEDIUM으로 지적되어 "/" 앞부분만 비교하도록 수정됨(`edgar_source.py:202-205`).
2. `date.fromisoformat(filing_date_str)`로 날짜를 파싱하되, 실패 시(`ValueError`) 원본 예외를 노출하지 않고 `EdgarApiError(f"EDGAR 응답 파싱 실패: {type(exc).__name__}")`로 재던짐(보안 리뷰 LOW — SEC가 예상 밖 날짜 형식을 보낼 가능성에 대한 방어).
3. `start <= filed_date <= end` 범위를 벗어나면 건너뛴다.
4. 통과한 항목을 `DisclosureRecord(corp_code=cik, ticker=ticker, report_name=_label_form_type(form), filed_date=filed_date, remark="")`로 변환.

`_label_form_type`은 `_FORM_TYPE_LABELS` 딕셔너리(`edgar_source.py:29-40`)로 10-K/10-Q/8-K 등 주요 폼 코드를 한국어 라벨로 변환하고, 매핑에 없는 폼은 원본 코드를 그대로 쓴다.

---

## `sources/naver_news_source.py` — 네이버 뉴스 검색 API 연동

뉴스 감성분석(#4-뉴스)의 국내 소스. DART/EDGAR가 "기업이 직접 제출하는 공식 문서(공시)"를 다루는 것과 달리, 이 모듈은 **진짜 뉴스 기사**를 다룬다. 네이버 뉴스 검색 API는 종목코드가 아니라 **회사명 문자열**로 검색해야 하므로, 이 함수는 이미 해석된 회사명을 인자로 받는다 — DART의 corp_code 같은 자체 식별자 개념이 없다.

**이용약관 관련 미해결 사항(중요)**: 파일 최상단 docstring(`naver_news_source.py:12-15`)에 명시되어 있다 — 이 API는 "검색 스니펫 제공"이 목적이며 상업적/자동화 대량 수집에 대한 이용약관 제약 가능성이 있다. 작성 당시 `developers.naver.com` 접근이 차단되어 재확인하지 못했다는 미해결 항목이 남아 있다. **실제 프로덕션 스케줄러에 이 소스를 넣기 전에는 반드시 이용약관을 직접 확인해야 한다.**

#### `_clean_html(text: str) -> str`

**목표/역할**: 네이버 응답의 `title`/`description`에 포함된 HTML 태그와 엔티티를 정제한다(`naver_news_source.py:46-53`).

**상세 동작**: `unescape(text)`(HTML 엔티티 언이스케이프)를 **먼저** 수행한 뒤 `_HTML_TAG_RE.sub("", ...)`(태그 제거)를 수행한다.

**설계상 비자명한 결정(순서가 중요한 보안 수정)**: 네이버는 검색어를 `<b>` 태그로 강조하고 특수문자를 HTML 엔티티로 이스케이프해서 응답한다(예: `&quot;`). 만약 순서를 반대로 해서 태그 제거를 먼저 하면, `&lt;script&gt;`처럼 **엔티티로 이스케이프된 가짜 태그**가 태그 제거 정규식을 통과한 뒤, 그다음 언이스케이프 단계에서 **진짜 `<script>` 태그로 되살아난다**. 엔티티 언이스케이프를 먼저 하면 되살아난 태그도 바로 다음 단계에서 함께 제거된다 — 코드 리뷰 MEDIUM으로 지적되어 수정된 사항.

#### `search_news(company_name: str, ticker: str, market: str, start: date, end: date) -> list[NewsArticle]`

**목표/역할**: 네이버 뉴스 검색(`news.json`)을 호출해 지정 기간의 기사를 가져온다(`naver_news_source.py:60-104`).

**상세 동작**: 크리덴셜(`NAVER_CLIENT_ID`/`NAVER_CLIENT_SECRET`)은 **HTTP 헤더**(`X-NCP-APIGW-API-KEY-ID`/`X-NCP-APIGW-API-KEY`)로 전달한다(URL 쿼리스트링이 아님 — DART의 `crtfc_key`와 대조적인 설계). 쿼리 파라미터는 `query=company_name`, `display=100`(건당 최대값), `sort="date"`.

네이버 API는 날짜 범위 파라미터를 지원하지 않으므로(정렬만 가능), 최신순으로 최대 건수를 가져온 뒤 **클라이언트 사이드로 기간 필터링**을 수행한다(DART/EDGAR와 동일 패턴). `item["pubDate"]`를 `parsedate_to_datetime`(RFC 822 형식 파싱)으로 변환하고, 그 변환 중 발생하는 `(KeyError, ValueError, TypeError)`를 `NaverNewsApiError`로 재던짐(에러 메시지는 타입명만).

각 기사의 `source` 필드는 `urlparse(original_link).netloc`로 채워진다. `original_link`는 `item.get("originallink") or item.get("link") or ""` — 원문 링크가 없으면 네이버 자체 링크로 폴백.

**엣지케이스**: `ticker`/`market`은 네이버 응답에 없는 정보이므로, 호출자가 명시적으로 넘긴 값을 그대로 각 `NewsArticle`에 채운다 — 이 함수 자체는 종목 식별을 전혀 하지 않는다.

---

## `sources/gdelt_source.py` — GDELT DOC 2.0 API 연동

뉴스 감성분석(#4-뉴스)의 미국 소스. GDELT는 API 키가 필요 없는 무료 공개 API — 크리덴셜 로딩 코드가 전혀 없다.

**검증 미완료 사항(중요)**: 파일 상단 docstring(`gdelt_source.py:7-12`)에 `seendate` 필드 형식(`%Y%m%dT%H%M%SZ`)이 "GDELT 공식 문서·커뮤니티 자료로 교차 확인한 값"이지만, 작성 당시 환경의 네트워크 제약(`api.gdeltproject.org`로의 아웃바운드 연결이 WebFetch·curl 모두 타임아웃)으로 **실제 라이브 응답을 한 번도 직접 검증하지 못했다**고 명시돼 있다. 실제 사용 전 반드시 라이브 호출로 재확인해야 하는 항목.

#### `search_articles(query: str, ticker: str, market: str, start: date, end: date) -> list[NewsArticle]`

**목표/역할**: GDELT DOC 2.0 기사 검색(`mode=artlist`)을 호출한다(`gdelt_source.py:36-88`).

**상세 동작**: `query` 파라미터는 **호출자가 완성한 검색 질의 문자열**을 그대로 받는다 — 이 함수는 질의 구성 로직(정확 구문 검색을 위한 따옴표 처리 등)을 모른다. GDELT는 `startdatetime`/`enddatetime` 파라미터를 지원하지만, 이 함수는 타임존/경계 처리의 불확실성을 피하기 위해 그 파라미터를 쓰지 않고 `sort="datedesc"`로 최신순 전체를 가져온 뒤 클라이언트 사이드로 기간 필터링한다(DART/EDGAR/네이버와 동일한 선택).

`item["seendate"]`를 `datetime.strptime(value, "%Y%m%dT%H%M%SZ")`로 파싱한다. 파싱 실패 시 `(KeyError, ValueError, TypeError)`를 잡아 `GdeltApiError`로 재던짐.

**엣지케이스(네이버와 대칭적인 버그 수정)**: `seendate`가 JSON `null`이면 `datetime.strptime`은 `ValueError`가 아니라 **`TypeError`**를 던진다. `naver_news_source.py`의 `pubDate` 파싱(동일 클래스 실패지만 거기선 `parsedate_to_datetime`이라 다른 예외 타입)과 대칭을 맞추기 위해 `TypeError`도 명시적으로 잡는다 — 코드 리뷰 MEDIUM으로 지적되어 수정된 사항(`gdelt_source.py:72-75`).

> **갱신 (2026-10-12, 라이브 실측 반영)**
> - **라이브 검증 완료**: 실제 호출에서 기사 50건이 `_SEENDATE_FORMAT`으로 정상 파싱됐다 — 위 "검증 미완료 사항"의 `seendate` 형식 가정은 맞았다.
> - **요청 제한**: GDELT는 과다 요청 시 HTTP 429("Please limit requests to one every 5 seconds")를 돌려준다. `search_articles`는 이제 내부 `_get`을 거쳐 프로세스 전체 요청을 5.5초 간격으로 직렬화하고(스레드 공유 잠금, `_next_allowed_at`), 429를 받으면 다음 허용 시각을 15초 뒤로 미룬 다음 1회 재시도한다. 그래도 실패하면 기존처럼 `GdeltApiError`. 실측상 이 IP에는 공지보다 엄격해 재시도 후에도 실패하는 경우가 있다.
> - **`company_phrase(name)`(신규)**: 회사명 → 정확구문 검색어. 따옴표 제거 + 끝의 SEC 법인 접미사 제거. GDELT 호출부 4곳이 모두 쓴다.
> - 현황·선택지·결정 대기 항목: `docs/design/us-news-source-reliability.md`.

`source`는 `item.get("domain") or ""`로 채워진다(네이버의 `urlparse` 추출과 달리 API가 직접 도메인 필드를 제공).

---

## `sources/sector_classification.py` — 공식 업종 분류(국내 전용)

`recommendation-synthesis-plan.md` §5의 "1차 매칭 경로"로 쓰이는 모듈(해당 설계 문서는 현재 리포에 파일로 존재하지 않으나 소스 주석에 참조됨). pykrx의 지수 API로 업종명↔코드 매핑과 그 업종의 구성종목을 조회한다.

**설계상 비자명한 결정(실제 호출로 확인된 로그인 요구사항)**: 2026-10-05 실제 pykrx 호출로, `KRX_ID`/`KRX_PW` 로그인이 선행되어야 하며 그걸 빼먹으면 모든 호출이 `"Expecting value"`라는 모호한 에러로 실패한다는 사실이 확인됐다(`sector_classification.py:5-8`) — 라이브러리 자체의 버그가 아니라 **로그인 선행 누락이 원인**이었다. 이 때문에 모든 공개 함수가 호출 직전에 `load_dotenv()`를 부른다(`pykrx_source.py`와 동일한 "함수 호출 시점 재로그인" 패턴).

**중요한 범위 제한**: **미국 시장은 지원하지 않는다.** pykrx는 한국거래소 전용 스크래퍼라 미국 종목의 공식 업종 분류를 제공하지 않는다(사용자 확인, 2026-10-05). NASDAQ/NYSE 섹터 질의는 이 모듈을 거치지 않고 전부 2차 경로(LLM 제안+뉴스검증)로만 처리된다 — 즉 이 모듈을 수정해서 미국 시장을 지원하게 만드는 건 애초에 불가능하다는 뜻이다.

**market 세분화(2026-10) — 원자/합성 양쪽을 받도록 확장**: `fetch_sector_names`의 `market` 인자가 과거엔 `"KRX"` 하나(코스피+코스닥 통째)만 받았는데, 이제는 `_MARKET_SUB_MARKETS = {"KRX": ("KOSPI", "KOSDAQ"), "KOSPI": ("KOSPI",), "KOSDAQ": ("KOSDAQ",)}` 매핑(`sector_classification.py:32-36`)으로 "KRX"(합성, 둘 다 조회)/"KOSPI"(단독)/"KOSDAQ"(단독) 세 입력을 구분해서 처리한다. 호출부(`chat_agent/sector_resolution.py`)가 사용자 질의 범위에 맞는 값을 그대로 넘긴다 — "코스닥 로봇관련주"면 `"KOSDAQ"`, "로봇관련주"(국내 전체)면 `"KRX"`. 이 모듈은 애초에 `_SUB_MARKETS = ("KOSPI", "KOSDAQ")` 내부 개념을 이미 갖고 있었기 때문에, 세분화 요구에 비교적 쉽게 대응할 수 있었다(모듈 docstring에도 명시).

#### `fetch_sector_names(market: str) -> dict[str, list[str]]`

**목표/역할**: 업종명을 지수코드 목록으로 매핑한다(`sector_classification.py:39-53`).

**상세 동작**: `_MARKET_SUB_MARKETS.get(market)`이 `None`(즉 market이 "KRX"/"KOSPI"/"KOSDAQ" 중 하나가 아님 — NASDAQ/NYSE/US 등)이면 빈 dict를 즉시 반환 — pykrx 대상이 아닌 시장은 에러 없이 조용히 스킵해, 호출부가 2차 LLM 폴백 경로로 자연스럽게 넘어갈 수 있게 한다. 매칭되면 그 튜플(1개 또는 2개 서브마켓)을 순회하며 `stock.get_index_ticker_list(market=sub_market)`로 지수코드를 얻고, `stock.get_index_ticker_name(code)`로 업종명을 얻어 `names.setdefault(name, []).append(code)`로 쌓는다.

**설계상 비자명한 결정(동명 업종)**: 반환 타입이 `dict[str, list[str]]`(값이 리스트)인 이유가 명시돼 있다 — 코스피와 코스닥에 **동명 업종**이 존재할 수 있다(실제 확인된 사례: KOSPI 1013번과 KOSDAQ 2072번 지수가 둘 다 "전기전자"). 하나의 업종명 키에 두 시장의 코드가 모두 들어갈 수 있어야 하므로 값을 리스트로 둔다. market이 `"KOSPI"`/`"KOSDAQ"` 단독이면 당연히 그 시장의 코드만 쌓이므로 동명 업종이 생길 일이 없다 — 이 "리스트" 반환 타입이 실질적으로 의미를 갖는 건 합성 버킷(`"KRX"`) 입력일 때뿐이다.

#### `fetch_constituents(index_codes: list[str]) -> list[str]`

**목표/역할**: 여러 지수 코드(예: 동명 업종의 코스피+코스닥 코드 둘 다)의 구성종목을 중복 없이 합집합으로 반환한다(`sector_classification.py:56-70`). `fetch_sector_names`가 반환한 코드 리스트를 그대로 이 함수에 넘기면, (market이 합성 버킷이었을 때는) 양쪽 시장의 구성종목을 하나로 합쳐 받을 수 있다.

**상세 동작**: `index_codes`가 빈 리스트면 빈 리스트 반환(네트워크 호출 없음). 아니면 코드마다 `stock.get_index_portfolio_deposit_file(code)`로 구성종목을 받아, `seen` set으로 중복을 걸러내며 **처음 등장한 순서를 보존**해 리스트에 쌓는다. 이 "순서 보존 합집합" 패턴은 `data/service.py::get_universe_union`에서도 동일하게 재사용된다.

---

## `sources/index_quote.py` — 지수 자체 시세 조회(신규, market 세분화 작업)

종목(ticker)이 아니라 지수 자체를 묻는 챗봇 질의("VIX 지금 어때") 전용 데이터 소스. `SUPPORTED_INDICES = frozenset({"VIX", "DJI", "IXIC", "S&P500", "RUT"})`(`index_quote.py:18`) — 챗봇 tool의 enum(`chat_agent/tool_schemas.py`의 `get_index_quote`)과 반드시 동기화해야 하는 상수다.

### `IndexQuote` (frozen dataclass)

```python
index: str
date: date
close: float
change_pct: float | None  # 직전 거래일 대비 변화율(%). 직전 데이터가 없으면 None
```

`OHLCVRecord`와 달리 `market` 필드가 없다 — 지수는 "종목"이 아니고 `VALID_MARKETS` 검증과 무관한 별개의 데이터 형태이기 때문(모듈 docstring에 명시).

### `fetch_index_quote(index: str) -> IndexQuote | None`

**목표/역할**: 지수의 최신 종가와 전일대비 변화율을 반환한다(`index_quote.py:35-57`).

**상세 동작**: `index`가 `SUPPORTED_INDICES`에 없으면 `UnsupportedIndexError`. 그 외에는 `fdr.DataReader(index, start, end)`를 호출(`_LOOKBACK_DAYS = 10`로 최소 2거래일+공휴일 여유를 확보) — FDR 소스코드를 직접 확인한 결과, 이 호출은 `data.py`의 `yahoo_index_symbol_map`을 거쳐 `YahooDailyReader`로 라우팅되어 Yahoo Finance의 OHLCV+Adj Close를 그대로 돌려준다. `df.empty`면 `None`(에러 아님 — 데이터가 전혀 없는 상황은 정상적으로 발생 가능). 그렇지 않으면 마지막 행의 종가를 `close`로, 마지막 두 행이 있으면 `(최신종가/직전종가 - 1) * 100`을 `change_pct`로 계산한다. **엣지케이스**: 데이터가 1행뿐이면 `change_pct`는 `None`. 직전 종가가 0이면(이론상 극단적 케이스) `ZeroDivisionError`를 피하기 위해 그 경우도 `change_pct = None`으로 둔다.

**설계상 비자명한 점**: `OHLCVRecord`로 감싸지 않고 전용 경량 dataclass를 쓴다 — 지수는 종목 레코드 캐시(`cache.py`)에 들어갈 이유가 없고(챗봇이 즉석에서 조회해 바로 보여주는 용도), market 검증 체계와도 무관하기 때문에 별도 타입으로 분리했다.

---

## `sources/index_constituents.py` — S&P500/다우존스 구성종목 조회(신규, market 세분화 작업)

`orchestrator/market_buckets.py`의 `INDEX_FILTER_BUCKETS`("SP500"/"DOW" 스캔 버킷, 이 문서 범위 밖)가 유니버스를 지수 구성종목으로 필터링할 때 쓰는 모듈.

**설계상 비자명한 결정(S&P500은 FDR 재사용, 다우존스는 자체 구현)**: S&P500은 FDR이 이미 지원한다 — `fdr.StockListing("S&P500")`은 Wikipedia `List_of_S%26P_500_companies` 페이지를 스크래핑하는 기존 `WikipediaStockListing` 클래스로 라우팅된다(FDR 소스코드로 직접 확인). 반면 **다우존스는 FDR에 없다** — FDR의 `WikipediaStockListing` 클래스는 S&P500 위키 URL이 클래스 내부에 하드코딩돼 있어, market 인자를 받아도 무시하고 항상 같은 S&P500 페이지만 긁는 구조라 재사용이 불가능하다(모듈 docstring에 이 조사 결과가 명시됨). 그래서 이 모듈은 다우존스 위키 페이지(`List_of_Dow_Jones_Industrial_Average_companies`)를 FDR과 동일한 패턴(`requests` + `pandas.read_html`)으로 **직접 스크래핑**하는 자체 구현을 둔다. 다우존스는 30개뿐이라 스크래핑이 실패해도(위키 페이지 구조 변경 등) 영향 범위가 작다는 점도 docstring에서 명시적으로 언급한다. 러셀2000(RUT) 구성종목은 무료 정확한 소스가 없어 범위에서 제외됐다(지수 자체 시세는 `index_quote.py`로 지원하지만 구성종목 리스트는 없음).

둘 다 7일 TTL 로컬 캐시를 쓴다(`SP500_CACHE_PATH = Path("data/sp500_constituents.json")`, `DOW_CACHE_PATH = Path("data/dow_constituents.json")`) — DART/EDGAR/pykrx 역조회 캐시와 동일한 "임시파일+`os.replace`" 원자적 쓰기 패턴을 `_load_or_refresh()`(`index_constituents.py:50-59`) 헬퍼로 공유한다. 이 헬퍼는 `download: Callable[[], list[str]]`을 인자로 받는 범용 함수라 S&P500/다우존스 양쪽이 재사용한다.

### `fetch_sp500_constituents() -> set[str]`

**상세 동작**: `_download_sp500_tickers()`(`index_constituents.py:62-64`)가 `fdr.StockListing("S&P500")["Symbol"]`을 리스트로 뽑고, `_load_or_refresh`가 캐싱한다. 반환형이 `list`가 아니라 `set`인 이유는 호출부(`market_buckets.py`)가 "이 티커가 구성종목에 포함되는가"만 빠르게 멤버십 체크하면 되기 때문이다.

### `fetch_dow_constituents() -> set[str]`

**상세 동작**: `_download_dow_tickers()`(`index_constituents.py:76-96`)가 다우존스 위키 페이지를 `requests.get`으로 가져온 뒤 `pandas.read_html`로 모든 테이블을 파싱하고, 컬럼명에 `"symbol"`/`"ticker"`가 포함된(대소문자 무시) 첫 테이블을 찾는다. 심볼 컬럼의 값에서 `"."`을 제거(FDR의 S&P500 스크래퍼와 동일한 정제, 예: `BRK.B` → `BRKB`류 표기 차이 방지)하고, **그 테이블의 행 개수가 20~40 사이**인지 검증한다(`index_constituents.py:94`) — 다우존스가 정확히 30개이므로, 이 범위 밖이면 위키 페이지에서 엉뚱한 테이블(예: 역사적 변경 이력 테이블)을 집었을 가능성이 높다고 보고 다음 테이블을 계속 탐색한다. 끝까지 못 찾으면 `IndexConstituentsError`.

**에러 처리**: `requests.exceptions.RequestException`은 다른 소스 모듈과 동일하게 원본 예외를 노출하지 않고 타입명만 남긴 `IndexConstituentsError`로 재던진다. 적합한 테이블을 끝내 찾지 못하는 경우도 같은 예외 타입으로 승격된다 — 호출부(백그라운드 스캔 스레드)의 "절대 전파 안 함" catch-all이 이 실패를 흡수한다.

---

## `sources/investor_flow.py` — 수급 데이터(외국인/기관 순매수, 공매도) 조회(신규, 2026-10 작업)

규칙엔진(RSI/SMA)과 완전히 무관한 **수급 주체 동향**을 보여주는 보조 신호. pykrx 기반이며 `pykrx_source.py`와 같은 "함수 호출 시점 `load_dotenv()`" 패턴을 쓴다(모듈 docstring에 `sector_classification.py`의 "로그인 없이 호출하면 'Expecting value'로 실패" 선례를 그대로 인용).

**국내(KOSPI/KOSDAQ) 전용** — 미국은 기관 보유현황이 13F(분기 공시)뿐이라 실시간성·파싱 복잡도가 완전히 다르다는 이유로 범위에서 제외됐다. `market`이 그 외 값이면 `_SUPPORTED_MARKETS = frozenset({"KOSPI", "KOSDAQ"})` 체크로 조용히 `None`을 반환한다(`sector_classification.py`의 "pykrx 미지원 시장은 에러 아님" 관례와 동일).

**설계상 비자명한 결정(캐싱하지 않음)**: DART/EDGAR corp_code 캐시나 `pykrx_source.resolve_market` 캐시는 "거의 안 바뀌는 참조 매핑"이라 7일 TTL이 맞지만, 외국인/기관 순매수와 공매도 잔고는 매일(심지어 일중) 바뀌는 시계열 데이터다 — 그 캐싱 패턴을 그대로 가져오면 "오늘 수급"을 묻는 질의에 최대 6일 묵은 값을 돌려주는 버그가 된다. 그래서 매 호출마다 pykrx를 직접 조회한다(이 모듈이 이 리포에서 처음으로 "캐싱하지 않는" 소스 모듈이라는 점이 다른 소스들과의 핵심 차이).

#### `fetch_net_purchases(ticker: str, market: str, start: date, end: date) -> dict[str, int] | None`

**목표/역할**: 지정 기간의 외국인/기관 순매수 거래량·거래대금을 반환(`investor_flow.py:40-66`).

**상세 동작**: pykrx의 `get_market_net_purchases_of_equities_by_ticker`는 1회 호출당 1개 `investor`만 조회할 수 있다(동시조회 옵션 없음) — 그래서 `investor="외국인"`/`investor="기관합계"`로 **두 번** 호출해 각각의 DataFrame에서 해당 티커 행을 찾는다. 둘 중 하나라도 `ticker`가 인덱스에 없으면 `None`(데이터 없음, 에러 아님). 있으면 `{"foreign_net_volume": ..., "foreign_net_value": ..., "institution_net_volume": ..., "institution_net_value": ...}`로 반환 — 전부 pykrx의 한글 컬럼명(`"순매수거래량"`/`"순매수거래대금"`)을 `int()`로 캐스팅.

#### `fetch_shorting_balance(ticker: str, market: str, as_of: date) -> dict | None`

**목표/역할**: 특정일자의 공매도 잔고·시가총액 대비 비중을 반환(`investor_flow.py:69-87`).

**상세 동작**: `stock.get_shorting_balance_by_ticker(date_str, market=market)` 호출 — `market` 파라미터가 `"KOSPI"`/`"KOSDAQ"`/`"KONEX"`만 받는다는 pykrx 자체의 제약(이 모듈의 `_SUPPORTED_MARKETS`와 자연히 맞아떨어짐). 티커가 응답 인덱스에 없으면(비상장·휴장일 등) `None`. 있으면 `{"balance_shares": ..., "balance_value": ..., "market_cap_ratio_pct": ...}` — 비중은 `float()`로 캐스팅(나머지 둘은 `int()`).

**챗봇 쪽 소비 지점**: `chat_agent/tools.py::tool_analyze_supply_demand`(해당 모듈 리뷰 참고)가 이 두 함수를 그대로 호출해 하나의 tool 결과로 묶는다. LLM을 쓰지 않으므로 호출부도 `QueryBudget`을 소비하지 않는다.

---

## `sources/etf_source.py` — 국내 ETF 구성종목(holdings)·유니버스 조회(신규, 2026-10 ETF 지원 작업)

ETF 가격은 구성종목들의 가중평균 움직임을 그대로 반영한다는 전제(사용자가 제시한 통찰) — "ETF 자체의 공시/뉴스"를 찾는 대신 "구성종목들의 공시/뉴스를 집계"하면 "이 ETF가 왜 움직였는지"에 답할 수 있다는 설계가 이 모듈과 `chat_agent/etf_composition.py`(챗봇 리뷰에서 다룸)의 출발점이다.

**설계상 비자명한 결정(API 패밀리 혼동 주의)**: `pykrx.stock.get_etf_portfolio_deposit_file(etf_ticker)`는 `sector_classification.py`가 쓰는 `get_index_portfolio_deposit_file(index_code)`와 이름이 비슷하지만 **완전히 다른 API 패밀리**다 — 전자는 ETF 티커를 직접 받아 구성종목+비중을 한 번에 반환하므로, `fetch_sector_names`→`fetch_constituents`처럼 "업종명→지수코드→구성종목" 2단계를 거칠 필요가 없다. 모듈 docstring이 이 혼동 가능성을 명시적으로 경고한다.

**국내(KOSPI/KOSDAQ) 전용** — 미국 ETF(SPY/QQQ 등) holdings는 SEC EDGAR의 Form N-PORT(분기 공시, 구조화 API 없이 raw XML 직접 파싱 필요, 신선도도 분기 단위로 국내 pykrx보다 훨씬 떨어짐)로만 무료 조회 가능하다는 조사 결과에 따라 사용자 확인 후 범위에서 명시적으로 제외됐다 — 다음 라운드 과제로 남아 있다.

#### `fetch_etf_holdings(etf_ticker: str, top_n: int = DEFAULT_TOP_N, as_of: date | None = None) -> list[tuple[str, float]]`

**목표/역할**: ETF 구성종목을 비중(%) 내림차순으로 상위 `top_n`(기본 10)개 반환.

**상세 동작**: `pykrx_source.py`와 동일하게 함수 호출 시점에 `load_dotenv()`를 호출해 KRX 로그인(`KRX_ID`/`KRX_PW`)을 보장한다. `stock.get_etf_portfolio_deposit_file(etf_ticker, "YYYYMMDD")`가 `None`이거나 빈 DataFrame이면 빈 리스트(ETF가 아니거나 holdings 데이터가 없음, 에러 아님 — ETF가 아닌 코드 예: 005930은 pykrx가 내부 에러 로그를 찍고 빈 결과를 준다). 아니면 `금액` 컬럼으로 비중(%)을 직접 계산해 내림차순 상위 N개를 `(종목코드, 비중)`(소수 둘째 자리 반올림)로 변환. `금액` 합이 0 이하면 빈 리스트.

**2026-10-11 라이브 실측으로 발견해 수정한 두 가지 함정**:
- **날짜를 반드시 명시한다**(`as_of`, 기본 오늘) — 생략하면 pykrx가 `get_index_ohlcv_by_date`로 최근 영업일을 찾는데, 이 환경에서 그 지수 API가 "Expecting value"로 실패해 `IndexError`가 난다. 명시 날짜(주말 포함)를 넘기면 KRX가 그 시점의 최신 PDF를 반환한다.
- **pykrx의 `비중` 컬럼을 쓰지 않는다** — 실제 컬럼은 `구성종목명/계약수/금액/시가총액/비중`(인덱스 `티커`)인데, `비중`이 `시가총액` 기준으로 계산되고 주식 행의 `시가총액`은 전부 0이라 현금(`010010` 원화예금) 행만 100, 나머지는 0으로 깨져 나온다. 1CU당 평가금액인 `금액`으로 계산하면 KODEX 200 기준 삼성전자 34.49%, SK하이닉스 26.96%로 상식과 일치한다(현금 행은 `금액`이 0이라 자연히 맨 뒤로 빠진다).

**설계상 비자명한 결정(`top_n` 기본값)**: `DEFAULT_TOP_N = 10`은 `recommendation_coordinator.DEFAULT_SHORTLIST_CAP`(20)보다 작게 잡았다 — ETF가 200개 종목을 담고 있어도 전부 분석하면 LLM 비용이 종목 수만큼 선형으로 커지기 때문에, "ETF 분석은 참고용 요약"이라는 성격을 반영해 더 적게 잡았다(모듈 docstring에 명시).

#### `fetch_etf_universe() -> list[str]`

**목표/역할**: 국내 ETF 전체 종목코드 리스트(`etf_source.py:50-55`).

**상세 동작**: `fdr.StockListing("ETF/KR")`의 `"Symbol"` 컬럼을 그대로 리스트로 반환. FDR 전체를 확인한 결과 ETF holdings 조회 기능 자체가 없다는 것이 docstring에 명시돼 있다 — 유니버스 조회에만 FDR을 쓰고 holdings는 이 모듈이 pykrx로 별도 조회.

#### `fetch_etf_listing() -> list[tuple[str, str]]`

**목표/역할**: 국내 ETF `(종목코드, ETF명)` 전체 리스트(`etf_source.py` 추가분) — `chat_agent/ticker_resolution.py`가 ETF 이름/코드를 해석할 때 보조 경로로 쓴다.

**상세 동작**: `fdr.StockListing("ETF/KR")`의 `Symbol`/`Name` 컬럼을 `zip`으로 묶어 반환(2026-10-11 실측 1,172개, `069500`→"KODEX 200" 확인). **신설 배경**: DART의 `corpCode.xml`은 상장"법인"(기업) 전용이라 ETF(운용사가 발행하는 펀드, 법인이 아님)를 포함하지 않을 가능성이 높아, `ticker_resolution.py`가 DART 역조회 실패 시 이 함수로 폴백한다 — 해당 모듈 리뷰(`06_chat_agent.md`)에서 상세히 다룬다.

**캐싱**: 이 모듈 자체는 캐싱하지 않고 매 호출마다 FDR을 호출한다. 대신 소비자인 `ticker_resolution.py`가 `lru_cache`로 프로세스 수명 동안 1회만 호출한다(`resolve_ticker`는 매 대화 턴 호출되므로) — `index_constituents.py`의 7일 TTL 파일 캐시와 달리 메모리 캐시라 챗봇 프로세스를 재시작하면 갱신된다.

---

## `market_routing.py` — 원자 market → 공시/뉴스 소스 라벨 매핑(신규, market 세분화 작업)

**중요**: 이 파일은 `data/sources/` 디렉터리가 아니라 `data/` 바로 아래(`src/auto_stock/data/market_routing.py`)에 있다 — 특정 외부 API를 감싸는 "소스" 모듈이 아니라, 여러 모듈이 공유하는 순수 매핑 테이블이기 때문이다.

**신설 배경**: `orchestrator/pipeline.py`와 `chat_agent/tools.py`가 과거 각자 `_NEWS_SOURCE_LABELS`/`_SENTIMENT_SOURCE_LABELS` 딕셔너리를 독립적으로 중복 정의하고 있었다(코드는 공유하지 않는다는 이 리포의 일반 원칙과는 별개로, 이 매핑 테이블만큼은 두 파일이 완전히 똑같은 값을 들고 있었다). market을 KOSPI/KOSDAQ으로 세분화하는 작업에서 "두 파일을 매번 동시에 고쳐야 하는" 번거로움과 누락 위험을 없애기 위해 공용 모듈로 추출했다.

**내용**(`market_routing.py:12-30`):
- `NEWS_SOURCE_LABELS = {"KOSPI": "DART", "KOSDAQ": "DART", "NASDAQ": "EDGAR", "NYSE": "EDGAR"}`
- `SENTIMENT_SOURCE_LABELS = {"KOSPI": "네이버뉴스", "KOSDAQ": "네이버뉴스", "NASDAQ": "GDELT", "NYSE": "GDELT"}`
- `KRX_MARKETS = frozenset({"KOSPI", "KOSDAQ"})`, `US_MARKETS = frozenset({"NASDAQ", "NYSE"})` — "국내냐 미국이냐"만 판단하면 되는 분기(공시 ID 해석, 이름 해석 등)가 반복돼서 멤버십 집합도 함께 둔다.

**설계상 비자명한 점**: `.get(market)`이 `None`을 반환하면(즉 지원하지 않는 시장) 호출부가 조용히 스킵하도록 설계돼 있다 — "향후 시장 확장 대비"라는 주석이 모듈 docstring에 있다. 이 파일을 쓰는 두 소비자(`orchestrator/pipeline.py`의 `_news_reasons`/`_sentiment_reasons`, `chat_agent/tools.py`의 `tool_analyze_disclosures`/`tool_analyze_news_sentiment`)는 각각 다른 문서(`05_orchestrator_and_synthesis.md`, `06_chat_agent.md`)에서 상세히 다룬다 — 이 문서는 매핑 테이블 자체의 정의만 다룬다.

---

## 데이터 계층 전체를 관통하는 공통 패턴

이 계층의 외부 연동 소스(FDR 제외하면 pykrx/DART/EDGAR/네이버/GDELT/index_constituents 여섯)를 리뷰하며 반복적으로 나타나는 설계 패턴을 정리하면:

1. **크리덴셜은 모듈 import 시점이 아니라 함수 호출 시점에 로드한다.** `_load_dart_api_key`/`_load_edgar_user_agent`/`_load_naver_credentials`가 모두 함수 안에서 `load_dotenv()`를 호출한다. 이는 `notifier/credentials.py`에서 처음 확립된 패턴이고, `pykrx_source.py`/`sector_classification.py`도 (크리덴셜이 아니라 로그인 트리거를 위해) 동일 패턴을 따른다 — 모듈 import 순서와 무관하게 `.env`가 항상 반영되도록 하기 위함.

2. **원본 예외를 절대 그대로 노출하지 않고 타입명만 남긴다.** DART/EDGAR/네이버/GDELT 모두 `requests.exceptions.RequestException`을 잡아 `f"{API명} API 요청 실패: {type(exc).__name__}"` 형태로 재던진다. 이는 `requests`/`urllib3`의 예외 문자열에 요청 URL(크리덴셜이 쿼리스트링에 담긴 경우)이나 헤더 정보가 그대로 포함될 수 있기 때문이며, 보안 리뷰에서 CRITICAL/HIGH로 지적되어 전체 소스 모듈에 일관되게 적용됐다.

3. **"데이터 없음"과 "에러"를 명확히 구분한다.** DART의 status `"013"`, 네이버/GDELT의 빈 결과 배열 등은 모두 빈 리스트를 반환할 뿐 예외를 던지지 않는다. 반면 API가 명시적으로 오류 상태를 응답하거나(DART status 이상), 응답 자체가 파싱 불가능한 형태면 예외로 승격시킨다.

4. **날짜 범위 필터링은 대부분 클라이언트 사이드.** DART만 서버 사이드 날짜 파라미터(`bgn_de`/`end_de`)가 있고, EDGAR/네이버/GDELT는 모두 "최신순 전체를 가져온 뒤 로컬에서 `start <= date <= end`로 걸러내는" 동일한 패턴을 쓴다.

5. **역조회/구성종목 캐시는 모두 원자적 파일 쓰기(`.tmp` + `os.replace`)로 저장된다.** DART corp_code, EDGAR CIK, pykrx의 KOSPI/KOSDAQ 소속시장 역조회(`resolve_market`), S&P500/다우존스 구성종목(`index_constituents.py`)까지 전부 같은 7일 TTL + 원자적 쓰기 패턴을 공유한다 — 쓰기 도중 프로세스가 죽어도 손상된 캐시가 남지 않도록 설계됐다. market 세분화 작업에서 신설된 두 캐시(`pykrx_source.resolve_market`, `index_constituents.py`)도 처음부터 이 패턴을 그대로 재사용했다. **예외**: `investor_flow.py`는 이 패턴을 의도적으로 쓰지 않는다 — 수급 데이터는 참조 매핑이 아니라 매일(심지어 일중) 바뀌는 시계열이라 캐싱 자체가 신선도 버그가 되기 때문이다(해당 모듈 절 참고).

6. **구/중간 스키마 하위호환.** DART/EDGAR 캐시 모두 "값이 문자열이면 구 스키마(이름 필드 없음), dict면 신 스키마"를 구분하는 헬퍼(`_corp_code_and_name`/`_cik_title_and_exchange`)를 둬서, 캐시가 자연 갱신될 때까지 과거에 저장된 캐시 파일도 깨지지 않고 읽힌다. EDGAR 쪽은 market 세분화로 `exchange` 필드가 추가되면서 하위호환 단계가 하나 더 늘었다 — "문자열(구 스키마)" / "dict인데 exchange 없음(exchange 필드 추가 이전의 중간 스키마)" / "dict이고 exchange 있음(신 스키마)" 세 단계를 모두 처리한다.

---

## 참고: 범위 밖 — `data/scan_cache.py`

`src/auto_stock/data/` 디렉터리에는 이 문서가 다루는 파일들 외에 `scan_cache.py`(254줄, `MarketScanCache` 클래스)가 추가로 존재한다. 이 파일은 이번 리뷰 범위(시세/공시/뉴스 원본 데이터 수집)에 포함되지 않았다 — 시장 스캔 결과를 캐싱하는 역할로 보이며(오케스트레이터 쪽에서 쓰임), 별도 리뷰가 필요하다.
