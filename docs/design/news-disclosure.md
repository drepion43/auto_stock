# 뉴스/공시 분석 에이전트 (#4) — 구현 결과

> `docs/design/news-disclosure.md`(사전 계획, DART/KRX)과 `docs/design/news-disclosure.md`(EDGAR/NASDAQ 확장 계획)의 실제 구현 결과 기록. Phase 0~4(DART) + 나스닥 확장 Phase 1~3(EDGAR) 전체 완료.

## 목적·입출력

종목의 최근 공시 목록(보고서명·접수일자·비고) — KRX는 DART, NASDAQ은 SEC EDGAR — 을 OpenAI GPT(`gpt-5.6-luna`)에 입력해 시장 영향(호재/악재/중립)을 구조화된 형태로 해석하고, 규칙엔진 후보에 대한 **보조 근거**로만 추가한다. ML예측(#2)·LLM차트분석(#3)과 동일하게 자체 후보를 만들지도, 알림을 필터링하지도 않는다.

## 모듈 구조 (계획 그대로 구현)

### 데이터 수집 계층 확장 (`src/auto_stock/data/`)

| 파일 | 역할 |
|---|---|
| `models.py` | `DisclosureRecord(corp_code, ticker, report_name, filed_date, remark)` — KRX/NASDAQ 공용, `corp_code` 필드는 시장에 따라 DART corp_code 또는 SEC CIK를 담는다 |
| `sources/dart_source.py` | `resolve_corp_code(ticker)`(corpCode.xml 캐싱+역조회), `fetch_disclosures(corp_code, start, end)`, `DartApiError` |
| `sources/edgar_source.py` | `resolve_cik(ticker)`(company_tickers.json 캐싱+역조회, CIK 10자리 0-패딩), `fetch_filings(cik, start, end)`(submissions API, Form 3/4/5 및 그 정정판 `"N/A"` 제외, 폼 코드를 한국어 라벨로 변환), `EdgarApiError`. API 키가 아니라 SEC 정책상 필수인 `SEC_EDGAR_USER_AGENT`(연락처 포함 식별 문자열, 비밀 아님)를 사용 |

### 해석 계층 (`src/auto_stock/news_disclosure/`)

| 파일 | 역할 |
|---|---|
| `models.py` | `DisclosureSummary`/`DisclosureAnalysis`/`LLMConfig`/`DisclosureReader`(Protocol) |
| `schema.py` | `DisclosureRead` — `responses.parse`의 `text_format` |
| `prompt.py` | `SYSTEM_PROMPT`, `build_user_prompt(summary)` — 공시 접수일은 D-N 상대 오프셋으로만 렌더링 |
| `credentials.py` | `load_llm_config()`, `DISCLOSURE_LOOKBACK_DAYS=90`, `MAX_DISCLOSURES_PER_QUERY=10` |
| `client.py` | `OpenAIDisclosureClient` — `openai`를 import하는 유일한 파일 |
| `analyst.py` | `analyze(client, disclosures, ticker, market, as_of)`, `to_reasons(analysis, action)` |

계획서의 7개 핵심 설계 결정("감성분석"이 아닌 "공시 영향 해석", 보고서명+메타데이터만 입력, corp_code 매핑을 데이터 수집 계층 확장으로 분류, `extra_reasons` 경유, 공시 없음은 정상, DART/LLM 독립 실패격리, SDK 직접 호출)은 코드로 그대로 구현됐다 — 상세 근거는 `news-disclosure.md` 참고, 여기서는 중복 서술하지 않는다.

## 오케스트레이터 통합

`orchestrator/pipeline.py`에 `news_client: DisclosureReader | None = None` 파라미터 추가. 분기 조건은 `ml_model is None and llm_client is None and news_client is None`(3항)일 때만 기존 2-위치인자 `generate_explanation` 호출을 보존(회귀 잠금). `_news_reasons` 헬퍼는 절대 raise하지 않고 `([], str(exc))`를 반환하되, 공시 조회 실패와 LLM 해석 실패를 별도 `try/except`로 감싸 각각 `"DART 조회 실패: "`/`"EDGAR 조회 실패: "`/`"LLM 공시해석 실패: "` 접두사를 스스로 붙인다 — `_ml_reasons`/`_llm_reasons`(호출부에서 접두사를 붙임)와 다른 이 함수만의 특징이다(두 단계 외부 호출을 체이닝하기 때문).

**시장별 소스 분기**: `_news_reasons`는 `_NEWS_SOURCE_LABELS = {"KRX": "DART", "NASDAQ": "EDGAR"}`로 market에 따라 `resolve_corp_code`/`fetch_disclosures`(KRX) 또는 `resolve_cik`/`fetch_filings`(NASDAQ)를 호출한다 — 해석 계층(`analyze_disclosures`/`to_disclosure_reasons`)은 시장과 무관하게 완전히 동일한 코드가 재사용된다. 별도 헬퍼로 쪼개지 않고 이 함수 안에서 분기한 이유는 `_ml_reasons`/`_llm_reasons`/`_news_reasons`를 애초에 3개로 분리했던 것과 다른 판단이기 때문이다 — 저 셋은 서로 다른 신호원이지만, DART/EDGAR는 같은 신호원(공시)의 시장별 소스 차이일 뿐이다(`docs/design/news-disclosure.md` 핵심 설계 결정 4). 미등록 종목(예: DART에 없는 나스닥 티커, EDGAR에 없는 한국 티커)은 `resolve_corp_code`/`resolve_cik`가 `None`을 반환해 조회 함수 자체가 호출되지 않고 조용히 스킵된다(에러 아님). `extra_reasons` 병합 순서는 ML → LLM차트 → 뉴스/공시(항상 마지막).

`scripts/verify_news_disclosure.py`(dry-run/실호출 2모드, `NEWS_VERIFY_MARKET`으로 KRX/NASDAQ 선택)와 `scripts/run_recommendations_with_signals.py`(`news_client` 배선 추가)도 함께 구현됐다 — 각각 `docs/design/news-disclosure.md` §5/§6, `docs/design/news-disclosure.md` 참고.

## `code-reviewer` + `security-reviewer` 리뷰 결과 반영 (Phase 4)

CRITICAL 1건 + HIGH 1건 + MEDIUM 2건, 전부 TDD로 수정. LOW 3건은 낮은 실질 리스크로 판단해 이번 라운드에서는 보류(사유 아래 기술). 리뷰 반영 전 258개 → 반영 후 **265개**로 회귀 확인.

**CRITICAL**
1. **`DART_API_KEY`가 요청 실패 시 예외 메시지로 노출될 수 있었음** — DART API는 인증키를 URL 쿼리스트링(`crtfc_key`)으로 받는데(DART의 API 계약, 이 코드의 선택이 아님), `_download_corp_code_map()`/`fetch_disclosures()`가 `requests.get()`/`response.raise_for_status()`를 감싸지 않아, 네트워크 오류·요청 제한·잘못된 키 등 어떤 전송 실패든 `requests`/`urllib3`가 예외 문자열에 그대로 담는 요청 URL(키 포함)이 `PipelineResult.errors`를 거쳐 `scripts/run_recommendations_with_signals.py`가 stdout에 그대로 출력할 수 있었다 — 그 스크립트 자신의 독스트링("절대 DART_API_KEY를 출력하지 않는다")과 정면으로 모순되는 상태였다. 두 함수 모두 `try/except requests.exceptions.RequestException`으로 감싸 `DartApiError(f"DART API 요청 실패: {type(exc).__name__}")`로 정제했다(원본 예외 객체는 절대 문자열화하지 않음) — `news_disclosure/client.py`가 OpenAI SDK 예외를 처리하는 것과 동일한 원칙.

**HIGH**
2. **잘못된 `DART_API_KEY`가 빈 매핑을 7일간 조용히 캐싱해 뉴스 신호를 무력화** — DART는 유효하지 않은/폐기된 키에 대해 HTTP 오류가 아니라 `<list>` 항목 없이 `<status>`/`<message>`만 담긴 "정상" zip을 반환한다. 기존 코드는 이를 빈 매핑(`{}`)으로 오인해 `data/dart_corp_codes.json`에 그대로 캐싱했고, 이후 `CORP_CODE_REFRESH_DAYS`(7일) 동안 모든 티커에 대해 `resolve_corp_code`가 `None`을 반환 — `_news_reasons`는 이를 "나스닥 등 미등록 티커"와 구분하지 못하고 에러 없이 조용히 스킵했다. `_download_corp_code_map()`이 이제 파싱된 XML 루트의 `status` 필드를 확인해, `fetch_disclosures`와 동일하게 오류 status면 캐싱 전에 `DartApiError`로 승격시킨다.

**MEDIUM**
3. **캐시 쓰기가 비원자적이었음** — `cache_path.write_text(...)`가 중간에 프로세스가 죽으면(OOM, 컨테이너 재시작 등) 손상된 캐시가 남고, `refresh_days` 동안 자연 복구되지 않을 수 있었다. 임시파일(`.tmp`) 쓰기 + `os.replace()`로 원자적 교체하도록 수정.
4. **`.get(key, "")`이 DART의 명시적 JSON `null`을 방어하지 못함** — 키가 아예 없을 때만 기본값이 적용되고, DART가 `"rm": null`/`"report_nm": null`을 보내면 `None`이 그대로 `DisclosureRecord`에 담겨 `str` 타입 계약을 조용히 깨고 있었다. `item.get(key) or ""`로 두 경우 모두 방어.

**보류(LOW, 이번 라운드 미수정)**
- DART `report_nm`/`rm` 필드에 길이 상한이 없어 LLM 프롬프트에 무제한 길이로 유입될 수 있음 — DART는 공신력 있는 정부 API(공격자 제어 불가)이고 LLM 출력 스키마(`DisclosureRead`)가 이미 `max_length`로 방어돼 있어 실질 위험은 낮다고 판단.
- `_download_corp_code_map()`의 zip 압축 해제에 크기 상한이 없음(zip-bomb 방어) — 신뢰된 정부 HTTPS 엔드포인트가 소스라 실질 위험은 낮다고 판단.
- 다중 프로세스 동시 실행 시 캐시 쓰기 경합 — 현재 단일 프로세스/순차 스케줄 사용 전제(MVP-0)라 해당 없음.

**리뷰가 확인만 하고 이상 없다고 판단한 항목**: 크리덴셜 로딩 패턴(`KeyError` 즉시실패, 로그 미노출), `LLMConfig.api_key`의 `field(repr=False)`(Phase 2 구현 시점부터 반영돼 있었음 — #3처럼 별도 수정이 필요하지 않았음), `defusedxml.ElementTree` 사용, `.gitignore`의 캐시 파일 제외, `client.py`의 `openai.APIError` catch-all(Phase 2부터 포함).

**`_ml_reasons`/`_llm_reasons`/`_news_reasons` 구조적 중복에 대한 결론**: 리팩터링하지 않기로 결정. 세 함수는 "절대 raise하지 않고 `(reasons, error)`를 반환한다"는 계약만 공유할 뿐, `_news_reasons`는 DART→OpenAI 2단계 외부 호출을 체이닝하며 중간에 조기 반환 분기(`corp_code is None`)까지 있어 다른 두 함수와 실질적으로 다르다. 공유 추상화를 만들면 체이닝 단계 수·조기종료 분기·에러 접두사 위치를 전부 파라미터화해야 해 코드량을 줄이는 것보다 복잡도를 늘릴 가능성이 크다 — 이 프로젝트의 YAGNI/DRY 원칙("반복이 실재할 때만 추상화")과도 맞지 않는다는 code-reviewer의 독립적 판단에 동의.

## EDGAR(나스닥) 확장 리뷰 결과 반영

DART 완료 후 `docs/design/news-disclosure.md`에 따라 `edgar_source.py` + `_news_reasons` market 분기를 구현하고 동일하게 `code-reviewer`+`security-reviewer` 병렬 리뷰를 거쳤다. CRITICAL/HIGH 없음. MEDIUM 1건 + LOW 3건(코드 리뷰 2건 + 보안 리뷰 공통 지적 1건, 중복 제외) 발견, MEDIUM과 이번에 반영하기로 한 LOW 2건은 TDD로 수정. 리뷰 반영 전 283개 → 반영 후 **284개**로 회귀 확인.

**MEDIUM**
1. **Form 3/4/5 정정판(`"3/A"`/`"4/A"`/`"5/A"`)이 필터를 우회했음** — `_EXCLUDED_FORM_TYPES`가 정확 문자열 일치(`form in _EXCLUDED_FORM_TYPES`)라 정정판은 걸러지지 않아, 애초에 필터가 막으려던 노이즈(임원 지분변동 신고)가 정정판 형태로 다시 새어 들어올 수 있었다. `form.split("/")[0] in _EXCLUDED_FORM_TYPES`로 접두사 비교하도록 수정.

**LOW (이번에 반영)**
2. **`fetch_filings`의 `date.fromisoformat`이 다른 실패 경로와 다르게 정제되지 않았음** — SEC가 예상 밖 형식의 `filingDate`를 보내면 원본 `ValueError`가 `requests.exceptions.RequestException`과 달리 `f"EDGAR API 요청 실패: {type(exc).__name__}"` 패턴을 거치지 않고 그대로 `_news_reasons`의 광범위한 `except Exception`까지 전파됐다(크래시는 아니었지만 — 파이프라인 격리는 유지됨 — 원본 예외를 문자열화하지 않는다는 이 모듈의 일관된 원칙에서 벗어남). `date.fromisoformat` 호출을 try/except로 감싸 `EdgarApiError(f"EDGAR 응답 파싱 실패: {type(exc).__name__}")`로 승격.
3. **연결 오류 테스트가 원본 예외 텍스트 미노출을 확인하지 않았음** — `test_resolve_cik_wraps_connection_error_as_edgar_api_error`/`test_fetch_filings_wraps_connection_error_as_edgar_api_error`가 예외 타입만 확인하고, DART의 대응 테스트(`test_resolve_corp_code_wraps_connection_error_without_leaking_api_key`)처럼 원본 메시지 문자열이 최종 에러에 없다는 assertion은 없었다. 두 테스트 모두 assertion 추가.

**보류(LOW, 이번 라운드 미수정)** — code-reviewer 판단, 사유는 각 항목 참고
- `fetch_filings`의 `tickers[0]` 기반 ticker 귀속이 다중 클래스 발행사(예: 복수 티커를 가진 CIK)에서 호출자가 실제 조회한 티커와 다를 수 있음 — 다운스트림(`DisclosureSummary.ticker`)이 이 필드를 읽지 않아 현재는 잠재적 함정일 뿐 실사용 버그 아님으로 확인됨.
- `_NEWS_SOURCE_LABELS.get(market)`이 `None`인 "미지원 시장" 분기가 테스트되지 않음 — 현재 KRX/NASDAQ 외 시장이 없어 도달 불가능한 코드, 세 번째 시장이 추가될 때 재검토.
- `_is_cache_fresh`가 `dart_source.py`에서 그대로 복제됨 — 2회 중복은 추출 기준(YAGNI)에 못 미침, 세 번째 소스가 생기면 공용 캐시 유틸로 추출 검토.
- `submissions` API의 병렬 배열(`form`/`filingDate`) 길이가 어긋나면 `zip()`이 조용히 짧은 쪽에 맞춰 자름 — SEC 자체 계약상 길이가 항상 같으므로 현재는 이론적 위험.

**리뷰가 확인만 하고 이상 없다고 판단한 항목**: CIK 10자리 zero-padding이 다운로드→캐시→URL 조립 전 구간에서 일관됨, 빈 `filings.recent`(공시 0건 회사) 처리, 캐시 스키마가 DART와 동형(`{ticker: cik}` vs `{stock_code: corp_code}`), 원자적 캐시 쓰기(임시파일+`os.replace()`)가 DART 수정분을 정확히 포팅, KRX↔NASDAQ 양방향 market 분기 테스트 커버리지, `verify_news_disclosure.py`의 시장별 디스패치, `.env.example`(`SEC_EDGAR_USER_AGENT`가 비밀 아님이라고 주석에 명시)/`.gitignore`(`data/edgar_cik_map.json`) 반영, zip 압축 해제 단계 자체가 없어 DART의 zip-bomb 우려가 해당 없음, 현재 호출 패턴(실행당 티커 소수)에서 SEC 10req/s 한도에 전혀 근접하지 않음.

## 테스트 현황

**284개 통과** (DART Phase 4 완료 시점 265개 + EDGAR 나스닥 확장 Phase 1~2 구현 18개 + EDGAR 리뷰 반영 1개 신규). `news_disclosure/`·`dart_source.py`·`edgar_source.py`·`orchestrator/pipeline.py` 전부 100% 커버리지. `requests`/`openai` 전부 모킹, 실제 API 호출 없음.

## 남은 작업 (사용자 액션 필요)

- `DART_API_KEY` 발급 후: `NEWS_VERIFY_MARKET=KRX`(기본값) `scripts/verify_news_disclosure.py` dry-run(비용 0, DART만) → 실호출(DART+OpenAI, 소액 비용) 순서로 1회 검증
- `SEC_EDGAR_USER_AGENT` 설정 후(무료, 즉시 가능 — 비밀키 아님): `NEWS_VERIFY_MARKET=NASDAQ scripts/verify_news_disclosure.py`로 동일하게 dry-run → 실호출 검증
- `scripts/run_recommendations_with_signals.py`로 ML+LLM차트+뉴스공시 3개 블록이 텔레그램에 함께 뜨는지 육안 확인(KRX/NASDAQ 각각)

이 항목들은 두 크리덴셜(`DART_API_KEY`, `SEC_EDGAR_USER_AGENT`) 중 미설정된 것이 있어 이번 리뷰 범위에 포함하지 않았다.
