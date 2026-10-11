# 뉴스/공시 분석 (#4) — Phase 1 완료 기록

> `docs/design/news-disclosure-plan.md`의 Phase 1(데이터 수집 계층 확장)을 실제로 구현한 결과. DART 인증키 없이 전부 모킹 기반으로 진행했다.

## 무엇을 했는가

DART(전자공시시스템) API 연동의 기반이 되는 **데이터 수집 계층**을 구현했다. 이 계층은 아직 LLM 해석은 하지 않고, "종목코드로 DART 공시 목록을 가져오는 것"까지만 담당한다.

## 변경/추가된 파일

| 파일 | 내용 |
|---|---|
| `src/auto_stock/data/models.py` | `DisclosureRecord(corp_code, ticker, report_name, filed_date, remark)` 신규 dataclass 추가 |
| `src/auto_stock/data/sources/dart_source.py` | **신규 모듈**. `resolve_corp_code(ticker)`, `fetch_disclosures(corp_code, start, end)`, `DartApiError` |
| `tests/data/test_models.py` | `DisclosureRecord` 생성 테스트 1건 추가 |
| `tests/data/test_dart_source.py` | **신규 파일**, 10개 테스트 |
| `pyproject.toml` | `openai>=3.5,<4`(버전 고정), `defusedxml`(신규 의존성) 추가 |
| `.env.example` | `DART_API_KEY=` 추가 |
| `.gitignore` | `data/dart_corp_codes.json`(캐시 파일) 추가 |

## 핵심 구현 내용

### `resolve_corp_code(ticker) -> str | None`
DART는 KRX 종목코드(`005930`)가 아니라 자체 `corp_code`(8자리)로 종목을 식별한다. 이 함수는:
1. 로컬 캐시(`data/dart_corp_codes.json`)가 있고 7일 이내면 그대로 읽어 역조회
2. 없거나 오래됐으면 DART의 `corpCode.xml`(zip 압축된 전체 상장·비상장법인 매핑)을 다운로드 → 압축 해제 → 파싱 → `{종목코드: corp_code}` 딕셔너리로 캐시 저장
3. 나스닥 등 DART에 등록되지 않은 티커는 예외가 아니라 `None`을 반환(정상 흐름)

### `fetch_disclosures(corp_code, start, end) -> list[DisclosureRecord]`
DART 공시검색(`list.json`) API를 호출해 해당 기간의 공시 목록을 가져온다.
- 응답 `status`가 `"013"`(조회된 데이터 없음)이면 빈 리스트 반환(예외 아님) — 공시가 없는 종목이 대다수일 것이므로 정상 케이스로 처리
- 그 외 오류 status는 `DartApiError`로 변환

### 보안 하드닝 (계획에 없던 추가 조치)
XML 파싱에 파이썬 표준 `xml.etree.ElementTree`를 쓰려다, 세션의 보안 가이드 훅이 "표준 XML 파서는 XXE(외부 엔티티 주입)·billion-laughs 공격에 기본적으로 취약하다"고 경고해 **`defusedxml`로 교체**했다. DART는 우리가 직접 호출하는 신뢰된 API지만, 방어적 코딩 차원에서 비용이 거의 없는(drop-in 대체) 조치라 바로 반영했다. `pyproject.toml`에 신규 의존성으로 추가.

## 테스트 현황

`tests/data/test_dart_source.py` 10건 + `test_models.py` 1건 = 11건 신규, 전체 스위트 **202개 전부 통과**(리뷰 전 191개 + 이번 11개). 전부 `requests`를 모킹해 실제 DART API를 호출하지 않는다.

검증한 케이스: 캐시 미스 시 다운로드+캐시 저장, `stock_code` 없는(비상장) 항목 제외, DART 미등록 티커 → `None`, 캐시 신선할 때 API 미호출, 캐시 만료(7일) 시 재다운로드, 공시 정상 조회, 공시 없음(`013`) → 빈 리스트, DART 오류 status → 예외, `DART_API_KEY` 누락 → `KeyError`(양쪽 함수 모두).

## 아직 하지 않은 것 (다음 단계)

- **Phase 2**: `news_disclosure/` 해석 계층(`models.py`/`schema.py`/`prompt.py`/`credentials.py`/`client.py`/`analyst.py`) — OpenAI로 공시 목록을 해석해 호재/악재/중립 판단. 이것도 DART 키 없이 모킹으로 진행 가능(`OPENAI_API_KEY`는 #3에서 이미 확보).
- **Phase 3**: 오케스트레이터(`pipeline.py`) 배선 + 실제 API 최초 호출 — `DART_API_KEY` 발급 필요.
- **Phase 4**: 문서화 + `code-reviewer`/`security-reviewer` 리뷰.

이번 Phase 1 산출물(`dart_source.py`)은 실제 DART 응답 형식에 대한 가정(공식 문서 기준)으로 작성됐고, 아직 실제 API로 검증된 적은 없다 — `DART_API_KEY` 발급 후 Phase 3에서 최초로 실호출 검증이 이뤄진다.
