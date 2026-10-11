# ETF 구성종목 집계 분석 — 구현 가이드

> 챗봇에 ETF 지원을 추가한 뒤 작성한 가이드 문서. 이전 컴포넌트 가이드들과 같은 위치·형식. 상세 코드 리뷰는 `docs/_review_parts/01_data_layer.md`(`sources/etf_source.py`)와 `docs/_review_parts/06_chat_agent.md`(`etf_composition.py`, `tool_analyze_etf_composition`, `ticker_resolution.py`의 ETF 폴백)를 참고.

## 목적

사용자가 "이 ETF 왜 올랐어/추천해줘" 같은 질의를 했을 때, 지금까지 전혀 다루지 않던 ETF를 분석할 수 있게 한다. 유니버스 조회(`get_universe`)가 일반 주식만 보고 있었고, 공시/뉴스/수급 분석(`analyze_disclosures`/`analyze_news_sentiment`/`analyze_supply_demand`)은 "그 회사가 직접 내는 공시·뉴스"를 전제로 하는데, ETF는 운용사가 발행하는 펀드라 이 전제가 맞지 않았다.

## 핵심 설계 결정: 구성종목 집계로 ETF 자체 분석을 대체한다

사용자가 제시한 통찰이 출발점이다 — **ETF 가격은 구성종목들의 가중평균 움직임을 그대로 반영한다.** 그래서 "ETF 자체의 뉴스/공시"를 찾는 대신 "구성종목들의 공시/뉴스를 집계"하면 "이 ETF가 왜 움직였는지"에 답할 수 있다.

조사 결과 이 설계가 실제로 저비용으로 가능하다는 게 확인됐다 — `pykrx.stock.get_etf_portfolio_deposit_file(etf_ticker)` 하나로 ETF 구성종목+비중을 직접 얻을 수 있고(지수 역조회 등 중간 단계 불필요), 이미 있는 "여러 종목 종합" 파이프라인의 구조(아래 "선례" 절)를 거의 그대로 재사용할 수 있었다.

## 조사로 확인된 사실

1. **ETF 구성종목 조회**: `pykrx.stock.get_etf_portfolio_deposit_file(ticker, date=None) -> DataFrame`(인덱스=구성종목 티커, 컬럼에 `구성종목명`/`비중`/`금액` 등 포함) — ETF 티커 하나만 넣으면 끝. `sector_classification.fetch_sector_names`→`fetch_constituents`의 2단계 패턴과 달리 1단계로 충분하다. **다만 로그인(`KRX_ID`/`KRX_PW`) 필요 여부는 소스 레벨로 확정되지 않았다** — `sector_classification.py`가 지수 API에 대해 겪은 선례(로그인 없이 호출하면 "Expecting value"로 실패)와 유사할 가능성이 높아 동일한 `load_dotenv()` 패턴을 적용했지만, 실제 라이브 호출로 검증이 필요하다(`scripts/verify_sources_individually.py etf`로 확인할 것).
2. **ETF 유니버스/시세**: `fdr.StockListing("ETF/KR")`(컬럼: Symbol/Name/Price/NAV/EarningRate/Volume/MarCap/Category)로 국내 ETF 전체 목록을 얻을 수 있다. OHLCV는 일반 종목과 동일하게 `fdr_source.fetch_ohlcv`로 그대로 조회 가능(ETF도 거래소에 상장된 종목코드라 포맷이 같다) — 단 이번 범위는 holdings 집계 분석까지만이라 OHLCV 경로는 실제로 연결하지 않았다.
3. **FDR은 holdings를 제공하지 않는다** — ETF 구성종목 조회는 pykrx 단독 경로다.
4. **"여러 종목 종합" 기존 패턴 재사용**: `orchestrator/recommendation_coordinator.py`의 "①숏리스트 생략, 외부에서 받은 종목 리스트를 그대로 변환 → 신호 수집 → LLM 종합" 모양(섹터 경로가 이미 쓰는 패턴)이 ETF 경로의 직접적 선례였다. 다만 ETF는 배치 파이프라인이 아니라 **챗봇 대화형 질의 1회성 tool**로 구현했다(아래 "범위" 절).

## 구현

### A. 데이터소스 — `data/sources/etf_source.py`

- `fetch_etf_holdings(etf_ticker, top_n=10, as_of=None) -> list[tuple[str, float]]` — 비중 내림차순 상위 N개 `(종목코드, 비중%)`. 데이터 없음(ETF 아님/비상장)이면 빈 리스트(에러 아님). **2026-10-11 라이브 실측으로 수정된 두 가지**: (1) 날짜를 생략하면 pykrx가 `get_index_ohlcv_by_date`로 최근 영업일을 찾다가 이 환경에서 "Expecting value"로 실패하므로 항상 명시 날짜(기본 오늘, 주말도 최신 PDF 반환 확인)를 넘긴다. (2) pykrx의 `비중` 컬럼은 `시가총액`(주식 행은 전부 0) 기준이라 현금 행만 100, 나머지 0으로 깨져 있다 — 1CU당 평가금액 `금액` 컬럼으로 비중을 직접 계산한다(KODEX 200: 삼성전자 34.49%, SK하이닉스 26.96%로 확인). 실제 컬럼: `구성종목명/계약수/금액/시가총액/비중`, 인덱스 `티커`. ETF가 아닌 종목코드(예: 005930)는 pykrx가 내부 에러를 로그로 찍고 빈 결과 → 빈 리스트.
- `fetch_etf_universe() -> list[str]` — `fdr.StockListing("ETF/KR")`의 `Symbol` 컬럼.
- `fetch_etf_listing() -> list[tuple[str, str]]` — `(종목코드, ETF명)` 전체 — `ticker_resolution.py`의 ETF 이름 역조회 폴백용(아래 D).

`top_n` 기본값 10은 `recommendation_coordinator.DEFAULT_SHORTLIST_CAP`(20)보다 작게 — ETF가 200개 종목을 담고 있어도 전부 분석하면 LLM 비용이 선형으로 커지므로, "참고용 요약"이라는 성격을 반영했다.

### B. 챗봇 tool — `tool_analyze_etf_composition` (`chat_agent/tools.py`)

흐름: `market != "KRX"`면 즉시 미지원 → 남은 예산으로 감당 가능한 구성종목 수 계산 → `fetch_etf_holdings(ticker, top_n=50)`로 넉넉히 받음 → 구성종목마다 `resolve_corp_name`으로 이름 확인(현금/채권/선물 등 이름 없는 항목은 공시·뉴스 신호가 있을 수 없으므로 제외) → **제외한 뒤에** 감당 가능한 수로 자름(이름 없는 행이 분석 자리를 차지하지 않게 — 코드 리뷰 발견사항) → 남는 게 없으면 `{"available": True, "holdings": None}` → **이미 있는** `tool_analyze_disclosures`/`tool_analyze_news_sentiment`를 그대로 재호출(새 분석 로직 없음, `ThreadPoolExecutor`로 병렬, 비중 순서 보존) → `etf_composition._holding_to_payload`로 조립 → `run_etf_composition_synthesis`에 종합 위임 → 성공 시 분석한 `holdings`(코드/이름/비중) 목록을 결과에 덧붙인다.

> 초안은 KOSPI/KOSDAQ 시장 세분화(`KRX_MARKETS`/`resolve_krx_sub_market`)를 전제했지만, `data/models.py::VALID_MARKETS`가 아직 `{"KRX","NASDAQ"}`라 현재 구현은 기존 `"KRX"` 관례를 그대로 쓴다. 시장 세분화 작업 때 이 tool의 market 판정도 함께 바꿔야 한다.

구성종목 수 × 2개 신호(공시+뉴스)만큼 LLM 호출이 추가로 들 수 있어 `context.budget`을 그대로 소비한다(새 budget을 만들지 않음 — 다른 모든 서브에이전트와 동일 원칙). 신호를 다 모으고 종합 단계만 예산 부족으로 실패하는 낭비를 막기 위해, 남은 예산에서 종합 몫(`ETF_COMPOSITION_LLM_CALL_ESTIMATE`)을 먼저 떼고 `(남은 예산) // 2`개까지만(최대 10) 분석한다 — 그마저 0이면 holdings 조회 없이 즉시 예산 소진 응답. 같은 턴에 다른 도구를 이미 많이 쓴 경우 분석 종목 수가 줄 수 있으며, 그래서 결과에 실제 분석한 `holdings`를 담아 답변에 명시하게 한다.

### C. 종합 — `chat_agent/etf_composition.py` (네 번째 deepagents 서브에이전트)

입력: ETF 티커 + 구성종목별 신호(종목명/비중/공시요약/뉴스감성) 리스트. 시스템 프롬프트: "이 ETF가 최근 왜 움직였는지, 상위 비중 종목 중 어떤 종목의 신호가 두드러지는지 요약하라" + 프롬프트 인젝션 방어(기존 모든 서브에이전트와 동일 문구). `budget` 공유는 `recommendation_synthesis/synthesizer.py`와 동일 원칙(새 `QueryBudget`을 만들지 않음). `provenance` 라벨("ETF 구성종목 집계 분석")은 결정론적으로 부착하되, 기존 "서브에이전트 자율 조사 결과"와 **의도적으로 다른 문자열**을 쓴다 — 이 agent는 신호를 자율 조사하는 게 아니라 이미 수집된 신호를 종합만 하기 때문. **재조사(reinvestigate) tool이 없다** — ETF holdings 자체가 이미 상위 N개로 캡돼 있어 "추가로 조사할 다른 종목"이라는 개념이 없다. 실패 격리는 기존 전원과 동일: 절대 raise 안 함.

`ETF_COMPOSITION_RECURSION_LIMIT`/`ETF_COMPOSITION_LLM_CALL_ESTIMATE`(둘 다 6)는 `recommendation_synthesis`의 값과 동일하게 설정했다 — "여러 종목 판단을 비교·종합"한다는 복잡도가 비슷하다고 판단해서다.

### D. 등록 지점

- `tool_schemas.py`: `_ticker_market_schema`로 `analyze_etf_composition` 스키마 추가.
- `TOOL_DISPATCH`에 등록(메인 대화 루프에서 직접 호출 가능).
- `_STOCK_ANALYST_TOOL_SPECS`에는 **추가하지 않았다** — ETF 분석은 종목 1개 전용 `stock_analyst`의 보조 신호가 아니라 별도 질의 유형이므로, `find_related_companies`처럼 메인 챗봇이 직접 호출하는 독립 tool로 둔다.
- `prompt.py`: "ETF 왜 올랐어/추천해줘" 질의면 `analyze_etf_composition`을 호출하라는 안내 + 결과 인용 시 "비중 상위 N개만 분석했다"는 점과 "ETF 구성종목 집계 분석" 출처를 함께 밝히라는 답변 표기 규칙.
- `ticker_resolution.py`: DART의 `corpCode.xml`은 상장"법인"(기업) 전용이라 ETF(펀드, 법인이 아님)를 포함하지 않을 가능성이 높다는 추정에 따라 `fetch_etf_listing()` 보조 경로를 추가했다(코드/이름 양방향, 결과 market은 `"KRX"`). 6자리 코드는 `resolve_corp_name`이 `None`일 때만, 이름 검색은 **DART 매치가 0건일 때만** ETF 목록을 본다 — 항상 섞으면 "삼성전자" 질의에 "KODEX 삼성전자채권혼합" 등이 끼어들어 이미 확정되던 일반 종목 질의가 모호해지기 때문. 이름 매칭 규칙은 DART와 동일(대소문자 무시 완전일치 우선, 없으면 부분일치). ETF 목록(약 1,170개)은 `lru_cache`로 프로세스 수명 동안 1회만 조회한다. **ETF 목록 조회 실패는 경고 로그 후 "ETF 매치 없음"으로 취급한다** — 보조 경로의 장애가 ETF와 무관한 질의(DART에 없는 나스닥 회사명 → EDGAR 매치)까지 `available=false`로 만들면 안 되기 때문(코드 리뷰 발견사항). 실패는 캐시되지 않아 다음 턴에 재시도된다.

## 범위에서 제외

- **미국 ETF(SPY, QQQ 등) holdings** — 사용자 확인 후 명시적으로 다음 라운드로 분리. 조사 결과: SEC EDGAR의 Form N-PORT(월간/분기 공시)가 유일한 무료 공식 경로이지만 (a) 구조화 API가 없어 raw XML을 직접 파싱해야 하고(`invstOrSecs.cusip`/`valUSD`/`pctVal` 등 N-PORT 전용 스키마, 지금 쓰는 EDGAR submissions API와 다른 포맷), (b) 공시 주기가 분기 단위(2개월 지연)라 국내 pykrx(거의 일별)보다 데이터 신선도가 훨씬 떨어진다. FDR도 미국 ETF holdings를 지원하지 않는다(확인됨). 유료 대안(sec-api.io 등)은 이 프로젝트의 "무료 소스 전용" 원칙과 맞지 않아 배제. 다음 라운드에서 별도로 `data/sources/edgar_nport_source.py`(가명) 신규 + XML 파서 + "분기 데이터"라는 신선도 디스클레이머 추가가 필요하다.
- **배치 추천 파이프라인(`recommendation_coordinator`)에 ETF를 자동 포함시키는 것** — 이번엔 챗봇 대화형 질의(`analyze_etf_composition` tool)로만 지원. "ETF 추천해줘" 같은 전체 스캔형 질의(유니버스에 ETF를 넣는 것 자체)는 범위 밖 — 지금 범위는 "사용자가 ETF 티커를 지정했을 때의 심층분석"까지다.
- **ETF 자체의 추적오차/괴리율**(`get_etf_tracking_error`/`get_etf_price_deviation`) — 가치 있는 추가 정보지만 이번 라운드는 "구성종목 집계" 하나에 집중했다.

## 테스트

- `tests/data/sources/test_etf_source.py` — `fetch_etf_holdings`(pykrx 모킹, 라이브 실측 컬럼 형태로 `금액` 기반 비중 계산·명시 날짜 전달·상위 N개 절삭·빈 결과 처리), `fetch_etf_universe`/`fetch_etf_listing`(FDR 모킹).
- `tests/chat_agent/test_etf_composition.py` — `run_etf_composition_synthesis`(`create_deep_agent` 모킹) — budget 공유, 예산 소진, invoke 실패, structured_response 손상 케이스(`synthesizer.py` 테스트와 동일 패턴), `_holding_to_payload`.
- `tests/chat_agent/test_tools.py`에 `tool_analyze_etf_composition` 추가(미지원 시장, holdings 조회 실패 격리, holdings 없음, 이름 없는 항목 제외·자리 비차지, 정상 흐름이 기존 두 tool에 위임 + 비중 순서 유지, 예산 기반 종목 수 캡, 종합 예산 부족 시 조회 생략, 종합 실패 전달).
- `tests/chat_agent/test_tool_schemas.py` 도구 개수 갱신.
- `tests/chat_agent/test_ticker_resolution.py`에 ETF 폴백 경로 테스트 추가(코드/이름 매칭, DART 매치 시 ETF 미조회, 부분일치 모호 처리, 목록 1회 조회, **목록 조회 실패해도 EDGAR 매치 유지·코드는 not found**). 테스트 중 실제 FDR 호출을 막는 autouse fixture가 있다.
- `tests/chat_agent/test_prompt.py`에 `analyze_etf_composition`·"ETF 구성종목 집계 분석" 문구 존재 확인 추가.

**라이브 검증 (2026-10-11)**: 프로젝트 `.venv`(실제 pandas-ta 포함)에서 전체 테스트 통과, 그리고 라이브 호출로 확인 — `fetch_etf_holdings("069500")`이 KRX 로그인(`.env`의 `KRX_ID`/`KRX_PW`) 후 실제 비중을 반환, ETF가 아닌 코드는 빈 리스트, `fetch_etf_listing()`은 1,172개 반환. 위 §A의 날짜/비중 컬럼 문제는 이 검증에서 발견해 수정했다. 아직 미검증: LLM을 실제로 태운 end-to-end 대화(`scripts/chat_cli.py`에서 "KODEX 200 왜 올랐어?").
