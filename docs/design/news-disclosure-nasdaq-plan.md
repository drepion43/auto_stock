# 뉴스/공시 분석 나스닥 확장 (#4-NASDAQ) 구현 계획

> `docs/design/news-disclosure-plan.md`(DART/KRX 전용, 1차 범위로 확정됐던 문서)를 확장하는 후속 계획. 그 문서의 핵심 설계 결정(1~7)은 그대로 유지하며, 이 문서는 "같은 신호원을 나스닥에도 어떻게 붙이는가"만 다룬다 — 중복 서술하지 않는다.

## 배경

현재 뉴스/공시분석(#4)은 DART(한국 전자공시)만 다룬다. DART는 한국 상장사 전용이라 `resolve_corp_code`가 나스닥 티커에 대해 `None`을 반환하고 `_news_reasons`가 이를 정상 흐름으로 조용히 스킵하도록 **의도적으로** 설계돼 있다(`tests/orchestrator/test_pipeline.py::test_nasdaq_ticker_skips_dart_lookup_entirely`). 즉 지금까지 나스닥 종목은 규칙엔진+ML+LLM차트 3개 신호로만 추천이 나갔다. 사용자가 나스닥에도 동등한 신호를 붙이길 원해 이 계획을 작성한다.

## 데이터 소스 확정: SEC EDGAR

미국 증권거래위원회(SEC)의 EDGAR(전자공시)가 DART의 미국 대응물이다. 이 세션에서 실제 엔드포인트를 조회해 확인했다(추측 아님):

- **티커→CIK(Central Index Key) 매핑**: `https://www.sec.gov/files/company_tickers.json` — 정적 파일, 무료, DART의 `corpCode.xml`과 같은 역할.
- **기업별 최근 제출 이력**: `https://data.sec.gov/submissions/CIK##########.json`(CIK를 10자리 0-패딩) — `filings.recent`가 컬럼형 배열로 `form`(폼 종류, 예: `"10-K"`/`"10-Q"`/`"8-K"`/`"4"`)/`filingDate`/`reportDate`/`accessionNumber` 등을 담는다. 실제 응답(Apple, CIK 0000320193)으로 필드명 확인 완료.
- **인증**: API 키 불필요. 대신 SEC 정책상 **`User-Agent` 헤더에 연락처 포함이 필수**(예: `"auto_stock contact@example.com"`) — 없거나 정체불명이면 차단 대상이 될 수 있다. 이건 "비밀"이 아니라 SEC의 공정이용 정책이 요구하는 식별 정보다.
- **요청 한도**: 초당 10건. 이 파이프라인의 실사용 패턴(실행당 티커 몇 개)에서는 문제되지 않아 별도 쓰로틀링은 이번 스코프에 넣지 않는다.

다른 후보(예: 3rd-party EDGAR 래퍼 API)는 검토하지 않았다 — DART를 SDK 없이 직접 REST 호출한 것과 동일한 원칙(계획 문서 핵심 설계 결정 7)으로 SEC 공식 API를 그대로 쓴다.

## 핵심 설계 결정

### 1. 크리덴셜 모델이 DART와 다르다 — "키가 없다"가 아니라 "식별정보가 필수다"

`_load_dart_api_key()`는 진짜 비밀키를 요구했지만, EDGAR는 비밀키가 없다. 대신 `_load_edgar_user_agent()`가 `SEC_EDGAR_USER_AGENT` 환경변수(없으면 `KeyError` — 크리덴셜 부재 패턴은 그대로 유지)를 요구한다. 값 자체는 비밀이 아니지만(연락처 문자열), "설정하지 않으면 즉시 실패"라는 계약은 SEC 정책 준수를 강제하기 위해 동일하게 유지한다.

### 2. `DisclosureRecord`를 재사용한다 — 필드 의미를 시장별로 재해석

새 데이터클래스를 만들지 않는다. `DisclosureRecord.corp_code`는 KRX면 DART `corp_code`, NASDAQ이면 SEC `CIK`(10자리 0-패딩 문자열)를 담는 것으로 의미를 확장한다 — `OHLCVRecord`가 이미 KRX/NASDAQ을 하나의 스키마로 다루는 것과 같은 선례를 따른다. `report_name`은 EDGAR의 `form` 코드(`"8-K"` 등)를 그대로 넣지 않고, **데이터 수집 계층에서 사람이 읽을 수 있는 라벨로 변환**해서 채운다(핵심 설계 결정 3). 이렇게 하면 해석 계층(`news_disclosure/`)은 시장이 KRX든 NASDAQ이든 완전히 동일한 코드로 동작한다 — 한 줄도 수정하지 않는다.

### 3. 폼 종류 라벨링 + 저신호 폼 필터링이 새로 필요하다

DART의 `report_nm`은 이미 한국어 설명형 문자열이라 그대로 LLM에 넣을 수 있었다. EDGAR의 `form`은 `"10-K"`/`"8-K"`/`"4"` 같은 코드라 LLM이 오독하거나(예: `"4"`를 숫자로 착각) 맥락을 놓칠 위험이 있다. `edgar_source.py` 안에 최소한의 매핑 테이블을 둔다:

```python
_FORM_TYPE_LABELS = {
    "10-K": "연차보고서(10-K)",
    "10-Q": "분기보고서(10-Q)",
    "8-K": "주요사항보고(8-K)",
    "S-1": "증권신고서(S-1)",
    "DEF 14A": "위임장권유서류(DEF 14A)",
    "SC 13D": "지분보고(SC 13D, 5% 이상 취득)",
    "SC 13G": "지분보고(SC 13G, 수동적 투자자)",
}
# 매핑에 없는 폼은 원본 코드를 그대로 사용(예: "SC TO-T (원문)")
```

**필터링**: Form 3/4/5(임원·주요주주 지분변동 신고)는 대형주 기준 주 단위로 여러 건씩 쏟아져 `MAX_DISCLOSURES_PER_QUERY`(10건) 윈도우를 노이즈로 채운다 — DART에는 없는 EDGAR 고유의 문제다. 기본적으로 Form 3/4/5는 제외하고 "중요 이벤트성" 폼(10-K/10-Q/8-K/S-1/DEF 14A/SC 13D/SC 13G 등)만 남긴다. 제외 목록은 `edgar_source.py`에 상수로 두어 나중에 조정 가능하게 한다. **이견 있으면 알려달라** — 임원 지분변동도 유의미한 신호로 보고 포함시키고 싶다면 이 필터링 자체를 옵션으로 뺄 수 있다.

### 4. 오케스트레이터의 `_news_reasons`가 시장별로 분기한다

세 번째 "구조는 같지만 코드는 공유하지 않는다"(#3와 #4가 이미 지킨 원칙)가 아니라, 이번엔 **같은 함수 안에서 시장별로 분기**하는 것이 맞다 — `analyze()`/`to_reasons()`(해석 계층)는 완전히 동일하게 재사용되고, 차이는 오직 "어떻게 원시 공시 목록을 가져오는가"뿐이기 때문이다(핵심 설계 결정 2가 이미 이 차이를 데이터 수집 계층에 가둬놨다). 새 함수를 또 만들면 오히려 code-reviewer가 Phase 4에서 승인한 "헬퍼 3개 분리" 판단과 모순된다 — 그 판단은 "서로 다른 신호원"에 대한 것이었지, "같은 신호원의 다른 시장"에 대한 것이 아니다.

```python
def _news_reasons(
    news_client: DisclosureReader | None, ticker: str, market: str, action: str
) -> tuple[list[str], str | None]:
    if news_client is None:
        return [], None

    source_label = {"KRX": "DART", "NASDAQ": "EDGAR"}.get(market)
    if source_label is None:
        return [], None  # 지원하지 않는 시장 — 조용히 스킵(향후 시장 확장 대비)

    try:
        if market == "KRX":
            corp_id = resolve_corp_code(ticker)
        else:  # NASDAQ
            corp_id = resolve_cik(ticker)
        if corp_id is None:
            return [], None  # 미등록 종목 — 에러 아님

        end = date.today()
        start = end - timedelta(days=DISCLOSURE_LOOKBACK_DAYS)
        disclosures = fetch_disclosures(corp_id, start, end) if market == "KRX" else fetch_filings(corp_id, start, end)
    except Exception as exc:
        return [], f"{source_label} 조회 실패: {exc}"

    try:
        analysis = analyze_disclosures(news_client, disclosures, ticker, market, end)
        return to_disclosure_reasons(analysis, action), None
    except Exception as exc:
        return [], f"LLM 공시해석 실패: {exc}"
```

**기존 테스트 영향**: `test_nasdaq_ticker_skips_dart_lookup_entirely`(나스닥은 DART를 아예 안 부른다는 것을 검증하던 테스트)는 이제 틀린 전제다 — 나스닥도 EDGAR로 실제 조회가 일어난다. 이 테스트를 "나스닥 티커는 `resolve_corp_code`/`fetch_disclosures`(DART)가 아니라 `resolve_cik`/`fetch_filings`(EDGAR)를 호출한다"로 대체해야 한다.

### 5. 캐시 파일도 시장별로 분리한다

`data/dart_corp_codes.json`과 별개로 `data/edgar_cik_map.json`을 둔다(같은 7일 갱신 정책 재사용). `.gitignore`에 추가 필요.

### 6. 요청 실패 처리는 처음부터 DART Phase 4의 교훈을 반영한다

DART 쪽은 "요청 실패 시 원본 예외를 그대로 노출하면 안 된다"는 것을 보안 리뷰에서 CRITICAL로 지적받고 사후 수정했다(`docs/design/news-disclosure.md` 참고). EDGAR는 URL에 비밀키가 없어 그 정도의 유출 위험은 아니지만, 원본 `requests` 예외를 문자열화하지 않고 `EdgarApiError(f"EDGAR API 요청 실패: {type(exc).__name__}")`로 감싸는 패턴은 **처음부터** 적용한다 — 이번엔 사후 리뷰로 발견하는 게 아니라 설계 단계에서 반영.

## 신규/변경 모듈 구조

```
src/auto_stock/data/sources/edgar_source.py   # 신규
├── EDGAR_CIK_MAP_URL / EDGAR_SUBMISSIONS_URL_TEMPLATE
├── CIK_MAP_CACHE_PATH = "data/edgar_cik_map.json"
├── CIK_MAP_REFRESH_DAYS = 7
├── _FORM_TYPE_LABELS / _EXCLUDED_FORM_TYPES(Form 3/4/5)
├── _load_edgar_user_agent() -> str          # SEC_EDGAR_USER_AGENT, 없으면 KeyError
├── resolve_cik(ticker) -> str | None        # company_tickers.json 캐싱+역조회
├── fetch_filings(cik, start, end) -> list[DisclosureRecord]  # submissions API, 폼 필터링+라벨링
└── EdgarApiError
```

```
src/auto_stock/orchestrator/pipeline.py   # 변경
└── _news_reasons — market 분기 추가(위 §4 코드 스케치)
```

**변경하지 않는 것**: `news_disclosure/`(해석 계층) 전체 — `analyze()`/`to_reasons()`/`prompt.py`/`schema.py`/`client.py`/`credentials.py` 전부 시장 무관하게 그대로 재사용. `data/models.py::DisclosureRecord`의 필드 구조도 무변경(의미만 확장, §핵심 설계 결정 2).

## 에러 처리 · 실패 격리

DART와 동일한 계층 구조를 그대로 따른다(`news-disclosure-plan.md`의 에러 처리 절 참고) — 미등록 종목/조회 결과 없음은 조용히 스킵, EDGAR 응답·전송 오류는 `EdgarApiError`로 승격 후 `_news_reasons`가 잡아 `"EDGAR 조회 실패: "` 접두사로 격리, LLM 해석 실패는 기존과 동일하게 `"LLM 공시해석 실패: "`. `SEC_EDGAR_USER_AGENT` 미설정은 진입점에서 `KeyError`로 즉시 실패.

## 테스트 전략

**절대 원칙: 실제 EDGAR API를 호출하지 않는다.**

- `tests/data/test_edgar_source.py`: `requests` 모킹 — CIK 캐시 히트/미스/만료, 매핑 실패 시 `None`, 폼 필터링(Form 4 제외 확인), 폼 라벨링, 필터링 후 결과 없음(빈 리스트), EDGAR 응답 오류, `SEC_EDGAR_USER_AGENT` 미설정 시 `KeyError`, **요청 실패 시 원본 예외 문자열이 노출되지 않음**(DART Phase 4 교훈을 회귀 테스트로 처음부터 포함).
- `tests/orchestrator/test_pipeline.py`: 기존 `test_nasdaq_ticker_skips_dart_lookup_entirely`를 대체하는 신규 케이스(나스닥 → EDGAR 호출, DART 함수는 호출 안 됨), KRX는 여전히 DART만 호출됨(회귀), EDGAR 실패해도 ML/LLM 근거 유지.
- 나머지 회귀: `news_disclosure/`, `dart_source.py` 테스트는 전부 무수정 통과.

## 의존성 · 설정 변경

| 파일 | 변경 |
|---|---|
| `pyproject.toml` | 신규 의존성 없음 — `requests` 재사용 |
| `.env.example` | `SEC_EDGAR_USER_AGENT=`(형식 예시 주석 포함) 추가 |
| `.gitignore` | `data/edgar_cik_map.json` 추가 |
| `docs/design/news-disclosure.md` | "나스닥은 DART 미등록이라 자동 스킵" 서술을 "나스닥은 EDGAR로 커버됨"으로 갱신(Phase 3 단계에서) |
| `docs/IMPLEMENTATION_PLAN.md`/`docs/CODEBASE_REVIEW.md` | #4 항목에 나스닥 확장 완료 반영(Phase 3 단계에서) |

## 구현 순서 (Phase)

**Phase 0 — API 게이트 (비용 0)**: EDGAR 엔드포인트·필드명·요청정책 재확인 — **이미 이 문서 작성 중 완료**(WebFetch로 실제 응답 확인). `.env.example`에 `SEC_EDGAR_USER_AGENT=` 추가. 사용자가 자신의 연락처로 값을 채우는 것이 이 Phase의 완료 조건(비밀키 아니라 즉시 채울 수 있음).

**Phase 1 — 데이터 수집 계층 (API 호출 0, 모킹)**: `data/sources/edgar_source.py` 신규 구현 + `tests/data/test_edgar_source.py`. 독립 머지 가능, `news_disclosure/`나 `pipeline.py`를 건드리지 않는다.

**Phase 2 — 오케스트레이터 배선**: `_news_reasons` 시장 분기 추가, `test_nasdaq_ticker_skips_dart_lookup_entirely` 대체, 신규 통합 테스트. `scripts/verify_news_disclosure.py`에 나스닥 샘플 티커(예: AAPL) 검증 모드 추가(또는 시장 인자 추가).

**Phase 3 — 문서화 + 리뷰**: `code-reviewer`+`security-reviewer` 병렬 리뷰(EDGAR 쪽은 비밀키가 없어 DART와 위협 모델이 다르다는 점을 리뷰어에게 명시 — `User-Agent` 헤더 처리, CIK 캐시, 폼 필터링 로직에 집중). `docs/design/news-disclosure.md`/`IMPLEMENTATION_PLAN.md`/`CODEBASE_REVIEW.md` 갱신.

**Phase 4 (선택, 보류) — 실제 API 최초 호출 검증**: `SEC_EDGAR_USER_AGENT` 설정 후 `scripts/verify_news_disclosure.py` 나스닥 모드 실행 → 텔레그램 메시지 육안 확인.

## 이번 스코프가 아닌 것

| 이연 항목 | 왜 이번이 아닌가 |
|---|---|
| EDGAR 전문(全文) 검색(Full-Text Search API) | 이번엔 `submissions` API의 메타데이터(폼 종류·날짜)만으로 시작 — DART도 원문(`document.xml`)은 스코프 밖이었던 것과 동일 원칙 |
| Form 3/4/5(임원 지분변동) 신호화 | 기본 제외(§핵심 설계 결정 3) — 별도로 원하면 후속 확장 |
| XBRL 재무데이터(companyfacts API) | 정량 재무데이터는 이 신호원의 "공시 영향 해석"과 다른 문제 — ML예측(#2)의 피처 확장 쪽에 더 가까움, 별도 검토 필요 |
| 다른 시장(런던·홍콩 등) 확장 | 사용자가 요청한 범위는 나스닥까지 |

## 검증 (Verification)

- Phase 0: `.env.example` 갱신 확인, `SEC_EDGAR_USER_AGENT` 값 채우기(사용자 액션)
- Phase 1: `pytest tests/data/test_edgar_source.py -v`
- Phase 2: `pytest tests/orchestrator/test_pipeline.py -v`(기존+신규)
- 전체: `pytest --cov=src --cov-report=term-missing`(80%+, 현재 99% 기준선 유지)
- Phase 4(선택): `scripts/verify_news_disclosure.py` 나스닥 모드 1회 실행 → 텔레그램 육안 확인
- 마지막: `code-reviewer` + `security-reviewer`

## API 키 / 크리덴셜

**비밀키 불필요.** `SEC_EDGAR_USER_AGENT`(연락처 포함 식별 문자열, 즉시 발급 아니라 즉시 설정 가능)만 필요 — `DART_API_KEY`/`OPENAI_API_KEY`와 달리 별도 신청 절차가 없다.

## 결정 현황

| 항목 | 상태 |
|---|---|
| 데이터 소스 | **확정 — SEC EDGAR**(이 세션에서 실제 API 조회로 검증) |
| `DisclosureRecord` 재사용 여부 | **확정 — 재사용**(필드 의미만 시장별 확장) |
| Form 3/4/5 필터링 | 권고안대로 진행(기본 제외). 이견 있으면 알려달라 |
| `_news_reasons` 시장 분기 vs 별도 헬퍼 | **확정 — 같은 함수 안에서 분기**(해석 계층이 동일하기 때문, §핵심 설계 결정 4) |
| CIK 캐시 갱신 주기(7일) | 권고안대로 진행(DART와 동일 정책). 이견 있으면 알려달라 |
