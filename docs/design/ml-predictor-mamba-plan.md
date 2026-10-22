# ML 예측기 교체 — scikit-learn LogisticRegression → MambaStock 기반 PyTorch Mamba SSM (+ 신규 펀더멘털 피처)

**결정일: 2026-10-13.** 이 문서는 그 교체 작업의 설계 결정과 구현 범위를 기록한다. 과거 LogisticRegression 설계의 상세는 `docs/design/ml-predictor.md`에 역사적 참고용으로 남아 있다.

## Context

당초 "피처만 추가"가 아니라 **production ML 모델 자체를 MambaStock(Mamba 선택적 상태공간모델, PyTorch) 구조로 교체**하는 것이 목적으로 바뀌었다. `src/auto_stock/MambaStock-main/`(외부 참고 레포, 패키지 네임스페이스 밖에 방치돼 있던 벤더 코드)가 쓰는 피처를 가능한 맞추면서, 모델 자체도 그 레포가 쓰는 시계열 구조로 바꿔 성능을 올리는 것이 목표.

교체 전 구조:
- `predictor.py::predict`는 **최근 1개 시점의 11개 스칼라 피처**(`(1, 11)`)만 받아 `estimator.predict_proba()` 호출 — sklearn `Pipeline(StandardScaler, LogisticRegression)` 전제.
- `_top_contributing_features`(predictor.py:20-40)는 `named_steps`/`coef_`에 전적으로 의존하는 **선형모델 전용 설명가능성** 메커니즘 — PRD §5/§5.1이 명시적으로 "계수 기반 분해가 가능한 유일한 후보라서 LogisticRegression 확정"이라고 적어둔 바로 그 근거.
- `artifact.py`는 `joblib.dump(estimator)` + `_logreg` 하드코딩 파일명, `training.py::build_estimator`는 `.fit(X,y)` 1회 호출(epoch/배치 없음).
- MambaStock 원본(`main.py`)은 **종목 1개의 전체 히스토리를 길이 L 시퀀스 하나**(`(1, L, D)`, batch=1)로 통째로 넣고, 매 시점마다 **다음날 수익률(연속값)**을 regression — 기존의 "최근 1시점만 보고 5일 뒤 방향을 분류"와 구조적으로 다름.

## 사용자 확정 결정

1. **완전 교체, 설명가능성 포기** — production 모델을 Mamba로 완전히 바꾸고, 계수 기반 top_features/설명가능성 메커니즘은 유지하지 않는다.
2. **모델 교체 + 신규 펀더멘털 피처 추가를 한 플랜으로 같이 진행** — PER/PBR/PSR/배당수익률/회전율/시가총액 등(pykrx `get_market_fundamental`/`get_market_cap` + DART `fnlttSinglAcntAll.json` 매출액, "매출액/영업수익/이자수익" 폴백 매칭) 전부 신규 피처 벡터에 포함. float_share/circ_mv(유통주식)는 데이터소스 없어 범위 제외.
3. **시퀀스 구조는 MambaStock 원본 방식** — 종목별 전체 히스토리를 길이 L 시퀀스 하나로, 매 시점 다음날 수익률을 regression.
4. **출력값은 예측 수익률 단순 매핑** — sigmoid로 유사-확률화하지 않고, 모델이 출력한 "예측 다음날 수익률"(float, 음수 가능)을 그대로 쓴다.
5. **시장별/종목별 개별 모델이 아니라 통합(unified) 단일 모델** — KRX/NASDAQ 구분도, 종목별 개별 모델도 두지 않고 전체 유니버스(국내+미국)를 하나의 Mamba 모델로 pooled 학습.

## 핵심 설계 리스크와 해결

### A. 국내전용 피처(PER/PBR/PSR/회전율/배당)와 pooled 전체-유니버스 설계의 충돌
새 펀더멘털 피처는 전부 국내(KOSPI/KOSDAQ)에서만 나온다. 통합 단일 모델이 미국 종목도 같이 학습하므로, 미국 종목 시퀀스에서는 이 채널들이 항상 결측.

**해결**: 피처 채널을 두 그룹으로 나눈다 — `CORE_FEATURE_NAMES`(기존 11개, all-or-nothing 유지)와 `FUNDAMENTAL_FEATURE_NAMES`(국내전용 5개 + 결측 플래그 1개, 결측 시 중립값 폴백). 중립값: `earnings_yield`/`dividend_yield`/`turnover_rate`/`sales_yield`=0.0, `pbr`=1.0, `fundamental_available`=0.0(또는 결측 시). 결측 플래그 1채널로 "실제 신호 0인지, 그냥 비어서 0인지"를 모델이 구분할 수 있게 한다.

PER/PSR은 역수(`earnings_yield`=1/PER, `sales_yield`=1/PSR)로 저장 — 손실기업의 음수/발산값을 완화하는 퀀트 투자 관례(Earnings Yield/Sales Yield). PBR은 발산 위험이 적어 역수를 쓰지 않는다.

### B. "종목 전체 히스토리 = 시퀀스 하나" + "여러 종목을 하나의 pooled 모델"의 결합
MambaStock 원본은 종목 1개 = 배치 1개(`unsqueeze(0)`)였지만, 여기선 종목이 수백~수천 개라 **배치 축 = 종목**이 되어야 한다. 종목마다 길이가 다르므로, 배치 내 **가변 길이 패딩 + 손실 마스킹**이 필요(`dataset.py::collate_sequences`, `training.py::_masked_mse_loss`).

### C. 학습 데이터 규모와 메모리
미니배치로 종목을 묶어서(기본 32종목/배치) 여러 step에 걸쳐 학습 — `training.py::train`이 PyTorch epoch/배치 학습 루프를 직접 구현(MambaStock `main.py::PredictWithData` 구조 기반).

### D. Lookahead bias — 신규 펀더멘털 채널도 기존 5중 방어축을 지켜야 함
DART 매출액(연간)은 "공시 시점"이 있다. `dart_financials_source.py::_accounting_year_as_of`가 `as_of` 월이 4월 이전이면 전전년도까지 내려가는 보수적 유예를 적용해, 공시 전 기간에 미래 매출액이 섞이지 않도록 한다. pykrx PER/PBR/시가총액/회전율은 일별 종가 기준이라 lookahead 문제 없음.

## 구현

### 1. Mamba 모듈 편입 — `ml_predictor/mamba_block.py` + `ml_predictor/pscan.py`
`MambaStock-main/mamba.py`/`pscan.py`의 `Mamba`/`MambaConfig`/`pscan`을 패키지 안으로 가져왔다(로직 그대로). `src/auto_stock/MambaStock-main/` 디렉터리는 삭제했다.

### 2. `pyproject.toml` — `torch`(CPU) 의존성 추가

### 3. `data/sources/pykrx_source.py` 확장
`get_market_fundamental`/`get_market_cap_detail`(단일일자) + `get_market_fundamental_range`/`get_market_cap_detail_range`(기간조회, 시퀀스 전체 구간용) 추가. 기존 `get_market_cap`은 `get_market_cap_detail`을 재사용하도록 리팩토링(하위호환 유지).

### 4. `data/sources/dart_financials_source.py` — 신규 모듈
`fnlttSinglAcntAll.json`, 매출액 폴백 리스트(`["매출액", "영업수익", "이자수익"]`), 연간 보고서(`reprt_code="11011"`) 기준, CFS→OFS 재시도, status 013/금융업 제외는 `None`. 사업연도 단위 영구 캐싱(`data/dart_revenue_cache.json`, TTL 1년, 빈 결과는 캐싱 안 함).

### 5. `ml_predictor/fundamentals.py` — 신규 모듈
`build_fundamentals_by_date(ticker, market, start, end)` — pykrx+DART를 날짜별 dict로 합친다. 국내(KOSPI/KOSDAQ) 아니면 즉시 None.

### 6. `ml_predictor/features.py` — 피처 채널 재설계
`CORE_FEATURE_NAMES`(11개) + `FUNDAMENTAL_FEATURE_NAMES`(5개+플래그 1개) = `FEATURE_NAMES`(17개). `build_feature_vectors(records, fundamentals_by_date=None)` — 코어는 all-or-nothing, 펀더멘털은 개별 중립값 폴백.

### 7. `ml_predictor/labeling.py` — regression 타깃
`LABEL_HORIZON_DAYS=1`(5일 방향분류 → 1일 수익률), `forward_returns`만 유지(`to_label`/`LABEL_THRESHOLD` 삭제).

### 8. `ml_predictor/dataset.py` — 시퀀스 데이터셋
`TickerSequence`/`SequenceDataset`/`SequenceDatasetSplit`(models.py). `build_ticker_sequence`/`build_sequence_dataset`/`chronological_split`(전역 날짜 기준, 동일하게 유지)/`collate_sequences`(가변길이 패딩+마스크)/`to_xy_flat`(평가용 flat 변환).

### 9. `ml_predictor/models.py` — 데이터클래스 갱신
`MLPrediction.probability_up`→`predicted_return`, `top_features` 필드 삭제. `ModelMetadata.market`은 항상 `"UNIFIED"`, 분류 지표(`test_roc_auc` 등) → regression 지표(`test_mse`/`test_rmse`/`test_mae`/`test_r2`) + `model_config`(MambaRegressor 재구성용 하이퍼파라미터).

### 10. `ml_predictor/artifact.py` — torch 직렬화
`joblib.dump/load` → `torch.save/load`(state_dict). 파일명 `unified_h1_mamba.pt`(market별 분기 없음, 통합 단일 아티팩트).

### 11. `ml_predictor/predictor.py` — 추론 인터페이스
`predict(bundle, records, fundamentals_by_date=None)` — 종목의 과거 전체 시퀀스를 받아 모델에 통과시키고 마지막 시점 출력을 `predicted_return`에 담는다. `_top_contributing_features` 삭제. `to_reasons()`는 "예측 수익률 X.XX%" 문구로 자연어화. `ML_AGREE_RETURN_THRESHOLD=0.003`/`ML_CONFLICT_RETURN_THRESHOLD=-0.003`(기존 0.55/0.45 확률 임계값 대체). `ML_LOOKBACK_DAYS=365*3` — Mamba는 짧은 lookback(120일, SMA60 워밍업용)으로는 부족해 ML 추론 전용 긴 기간을 따로 조회.

### 12. `ml_predictor/training.py` — PyTorch 학습 루프
`MambaRegressor`(`nn.Linear → Mamba → nn.Linear`, MambaStock `main.py::Net`과 동일 구조) + `train()`(미니배치 epoch 루프, Adam, masked MSE loss) + `evaluate()`(시퀀스 마지막 시점만 평가) + `persistence_baseline_metrics()`(regression 하한선 베이스라인, 기존 rf/dummy 벤치마크 대체).

### 13. 다운스트림 전파
- `data/market_routing.py::ML_MODEL_KEYS` 삭제(통합 단일 모델이라 market별 매핑 불필요).
- `chat_agent/tools.py::ChatToolContext.ml_model`(dict→단일 필드), `tool_analyze_ml_prediction`이 `_fetch_ml_records`(ML 전용 긴 lookback)+`_fetch_fundamentals` 사용.
- `orchestrator/shortlist.py::build_shortlist` — ML 전용 긴 lookback으로 별도 `get_ohlcv` 재조회 + `build_fundamentals_by_date` 호출.
- `orchestrator/recommendation_coordinator.py::RecommendationCoordinator` — `ml_models: dict` → `ml_model: ModelBundle | None`.
- `orchestrator/pipeline.py::_ml_reasons` — 동일하게 ML 전용 lookback 재조회.
- `recommendation_synthesis/synthesizer.py::_judgment_to_payload` — `probability_up`/`top_features` → `predicted_return`.
- `scripts/train_ml_model.py` — KRX+NASDAQ+NYSE 전체를 하나의 통합 모델로 학습(시장별 루프 제거), persistence baseline 리포트.
- `scripts/chat_cli.py`/`run_recommendations_with_ml.py`/`run_recommendations_with_signals.py`/`eval_main_agent.py` — `_load_ml_models()`(dict) → `_load_ml_model()`(단일), `load_model(market)` → `load_model()`(인자 없음).

## 검증
- `.venv/Scripts/python -m pytest -q -m "not slow" tests/ml_predictor/ tests/data/sources/ tests/orchestrator/ tests/chat_agent/ tests/recommendation_synthesis/` — 로직 변경 전제인 pandas-ta 관련 환경 실패(Python 3.11 venv, numba 미지원 스텁) 외 전부 통과.
- Mamba forward/backward smoke test(실제 PyTorch 연산) 통과 — `tests/ml_predictor/test_training.py`(`@pytest.mark.slow` 포함, 전체 학습 루프 end-to-end).

## 2026-10-22 진단 — 예측력이 없던 이유와 수정

진단(읽기 전용, 기존 `data/ml_training_dataset.pkl`): 같은 17개 피처의 **선형 ridge가 국내 test IC +0.025(원래 레이블)~+0.046(t=4.1, 시장평균 차감 레이블)**을 낸 반면 Mamba는 ≈0. 신호는 데이터에 있는데 Mamba가 꺼내지 못했고, 그 위에 아래 문제가 겹쳐 있었다.

| # | 문제 | 근거 | 수정 |
|---|---|---|---|
| 1 | 유니버스: `universe_for("KRX")[:200]`이 KOSPI 뒤에 KOSDAQ을 붙인 리스트의 앞 200개 → **KOSDAQ 0종목**. pykrx가 가나다순이라 KOSPI도 우선주 섞인 임의 200개. 현재 상장 종목만(생존 편향) | 데이터셋 시장별 종목 수, 첫 종목들(AJ네트웍스, AK홀딩스, BYC우…) | `ml_predictor/universe.py`: 원자 시장별로 **수집 시작일 시점 시가총액 상위 보통주**(우선주·거래정지 제외). 국내는 그 시점 상장 종목이라 이후 상폐 종목 포함. 미국은 과거 시점 목록 소스가 없어 현재 목록 유지(한계). `pykrx_source.get_market_cap_table`은 휴장일에 pykrx가 **전부 0인 표**를 주는 것을 실측으로 확인해 이를 건너뛴다 |
| 2 | 타깃: 원래 다음날 수익률 + MSE → 같은 날 시장 평균이 라벨 분산의 9~25%, 모델은 평균 드리프트만 학습(예측 74% 양수) | 날짜 고정효과 분산 비율, 예측 분포 | `dataset.cross_sectional_demean`(같은 날·같은 지역 평균 차감)을 학습·검증 타깃 기본값으로(`ML_TRAIN_TARGET=demeaned`). 평가는 원래 레이블. `filter_regions` + `ML_TRAIN_REGIONS`로 KR/US 분리 학습 가능 |
| 3 | Mamba가 선형보다 못함, 저장 가중치가 마지막 epoch | ridge vs Mamba IC, val loss 최저 58 vs 저장 100 | `ml_predictor/baseline.py`(ridge, 같은 전처리) 리포트에 항상 병기 + **validation IC로 판정**("Mamba가 선형 기준선을 이기지 못함 → 배포하지 말 것"). `training.train`은 매 epoch val IC를 재고 **최고 val IC epoch 가중치**를 저장(history CSV `val_ic` 열) |
| 4 | 평가가 MSE·비용 전 스프레드 | — | `evaluation.daily_rank_ic`에 상위·하위 20% **회전율**과 **왕복비용 차감 스프레드**(`ML_EVAL_COST_KR` 0.25%, `ML_EVAL_COST_US` 0.05% — 가정값). 리포트는 지역별 |
| 5 | 미국 극단 라벨(MRNA +177% 등) 데이터 오류 의심 | 미국 \|r\|>40% 28건 전수 점검: **모두 당일 거래량 3배 이상(최대 37배)** — 실제 사건 | 수정 없음. 기존 ±20% 학습 레이블 자르기로 충분(진단 때의 "오류 가능성" 언급은 틀렸음) |

**새 리포트로 본 기존 모델(재학습 전, 기존 데이터셋, `ML_TRAIN_EVAL_ONLY`)**

| 구간 | Mamba IC | ridge IC (시장평균 차감 타깃) | ridge 비용 차감 스프레드 |
|---|---|---|---|
| validation KR | −0.003 | +0.023 (t=2.1) | −0.19%/일 (비용 전 −0.03%) |
| validation US | +0.006 | +0.024 (t=1.8) | +0.16%/일 |
| test KR | −0.025 | **+0.045 (t=4.0)** | +0.08%/일 (비용 전 +0.25%, 회전율 34%) |
| test US | +0.002 | +0.003 | +0.07%/일 |

- 판정: 두 지역 모두 Mamba가 선형 기준선을 못 이김 → 배포 금지.
- ridge도 확정이 아니다: 국내 IC는 두 구간 모두 양수지만 validation에서는 상위−하위 스프레드가 음수(꼬리 종목이 순위와 반대로 움직임), 미국 IC는 test에서 사라진다. 단기 반전(persistence)은 회전율 77%라 비용 차감 후 크게 음수.
- 아직 안 한 일: 새 유니버스로 **데이터 재수집**(KOSPI·KOSDAQ 각 200 + 미국 400, 국내 펀더멘털 조회 때문에 수 시간), 재학습(타깃 demeaned, 지역별), 그 결과로 Mamba 유지 여부 결정.
