# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project status

**Code exists (updated 2026-10-12)**: Python 3.12 package under `src/auto_stock/` (setuptools src layout, `pyproject.toml`, `uv.lock`). Active work is on branch `feat/main-agent-registry` (pushed to origin, not yet merged to `master`). ~760 pytest tests under `tests/`; run fast tests with `.venv/Scripts/python -m pytest -q -m "not slow" tests/` (`slow` = model-fitting smoke tests) — all pass as of 2026-10-13. LLM provider is OpenAI (`openai`, `langchain-openai`, `deepagents`). Chat entry point: `scripts/chat_cli.py`; legacy-vs-deep eval: `scripts/eval_main_agent.py` (real API calls). `docs/` is git-tracked (removed from `.gitignore` in 7b9533d).

`docs/PRD.md` defines the product: a Python system that recommends and semi-automatically executes stock trades (코스피/코스닥 + 나스닥), gated by a confirmed risk policy (§6) and a paper-trading-first broker strategy (§7). Read `docs/PRD.md` before writing any code — it is the source of truth for scope, architecture, risk limits, and broker roles. Remaining open items are tracked in its §11.

**Architecture (updated 2026-10-12)**: PRD §4's original all-`deepagents` proposal has been refined — see `docs/_review_parts/06_chat_agent.md` and `docs/design/main-agent-migration-plan.md`. The analysis modules (rule engine #1, ML predictor #2, LLM chart analyst #3, news/disclosure #4, risk sizing #5) are plain functions. The **main conversational agent runs on `deepagents` `create_deep_agent`** (default since 2026-10-12, see "Main loop migration" below). Besides the main agent, `deepagents` is used for four subagents, all implemented:
- `find_related_companies` and `stock_analyst(ticker)` — exposed to the main agent as agent-as-tool.
- `recommendation_synthesis` — invoked by the background batch coordinator (`orchestrator/recommendation_coordinator.py`), not the main agent; see `docs/design/recommendation-synthesis.md`.
- `etf_composition` (`chat_agent/etf_composition.py`, 2026-10-11) — synthesis-only, no tools; called from the plain tool `analyze_etf_composition` after it collects disclosure/news signals for an ETF's top holdings — domestic ETFs via pykrx, US ETFs via SEC Form N-PORT (`data/sources/edgar_nport_source.py`, quarterly data up to ~2 months stale); see `docs/design/etf-composition-analysis.md`.

**Markets (committed in 34e433c, 2026-10-13)**: `data/models.py::VALID_MARKETS` = atomic exchanges `KOSPI`/`KOSDAQ`/`NASDAQ`/`NYSE`. Per-ticker tools take an atomic market (`chat_agent/tool_args.py::AtomicMarket`); scan/sector tools also accept composite buckets `KRX`/`US`/`SP500`/`DOW` (`ScanMarket`, must stay in sync with `orchestrator/market_buckets.ALL_SCAN_BUCKETS`). Ticker resolution picks the real exchange via `pykrx_source.resolve_market` / `edgar_source.resolve_exchange` (live check: `scripts/verify_market_resolution.py`).

**ML predictor model swap (2026-10-13)**: production model replaced scikit-learn `LogisticRegression` with a Mamba (selective state-space model, PyTorch, ported from the MambaStock reference repo) — `ml_predictor/mamba_block.py` + `training.py::MambaRegressor`. One **unified** model now (no per-market/per-ticker split; `data/market_routing.py::ML_MODEL_KEYS` removed), trained on the whole KRX+NASDAQ+NYSE universe pooled. Label changed from 5-day-ahead direction classification to next-trading-day return regression (`MLPrediction.predicted_return`, no `probability_up`/`top_features` — coefficient-based explainability was dropped, user-approved tradeoff). New domestic-only fundamental features (PER/PBR/dividend yield/turnover/PSR via pykrx + DART `fnlttSinglAcntAll.json`) sit alongside the 11 core scale-free features with neutral-value fallback for non-KRX tickers. Artifact format is `torch.save`/`.pt` (`models/ml_predictor/unified_h1_mamba.pt`), not `.joblib`. `torch` is now a dependency; on Windows uv installs the CUDA 13.0 build (`[tool.uv.sources]` → PyTorch `cu130` index, supports RTX 50/Blackwell). Training picks the device via `training.py::resolve_device` (`ML_TRAIN_DEVICE=auto|cuda|cpu`, auto = GPU if available; explicit `cuda` without a GPU fails before data collection); the saved state_dict is always CPU tensors so inference stays CPU-only. `train_ml_model.py` splits train/validation/test chronologically (`dataset.py::train_val_test_split`, ~70/10/20, test window identical to the 2-way split), logs train/val loss + lr per epoch (`ML_TRAIN_LOG_EVERY`), prints a post-training report (validation/test, test split domestic vs US, Mamba vs persistence baseline incl. direction hit rate) and saves `unified_h1_mamba.history.{png,csv}` (`ml_predictor/reporting.py`). The saved weights are from the LAST epoch, not the best-val epoch. Details: `docs/design/ml-predictor-mamba-plan.md`.

**Main loop migration (2026-10-11, branch `feat/main-agent-registry`)**: tools are defined once in `chat_agent/tool_registry.py` (`TOOL_SPECS`) — adding a tool means one `ToolSpec` + its `tool_*` function in `tools.py`; schemas, dispatch and stock_analyst bindings are derived, and golden fixtures in `tests/chat_agent/fixtures/` pin what the model receives. The deepagents-based main loop (`chat_agent/deep_loop.py` + `agent_middleware.py`) is the default since 2026-10-12 (eval passed); `CHAT_AGENT_LOOP=legacy` rolls back to `loop.py`, which is removed after a 2-week trial (decision D12). Plan and decisions D1–D19: `docs/design/main-agent-migration-plan.md`.

**Current priority (2026-10-05)**: MVP-1 (semi-automated order execution) and everything after it are excluded from current work — the user confirmed the near-term goal is finishing the stock-recommendation chatbot only. See the dated decision update in `docs/PRD.md` §9. Do not start broker integration, order execution, or execution-side risk enforcement unless the user explicitly reverses this.

Key facts from the PRD that code MUST respect:
- **Risk policy (§6, confirmed)**: 종목당 5% 이내, 최대 동시 10종목, 일일 손실 2%/월간 10% 중단, ATR 기반 손절익절·포지션 사이징.
- **Broker roles (§7, confirmed)**: 토스증권 Open API = 실전계좌 전용 (모의투자 미지원, 공식 확인됨). 한국투자증권/키움증권 = 모의투자 검증의 필수 경로. Never use 토스 for paper-trading validation.
- Credentials: environment variables or a secret manager only, never hardcoded.

## Project-specific tooling

- **`risk-policy-guardian`** subagent — invoke (or let it auto-trigger) when writing/modifying risk management, position sizing, or order execution code, to check it against PRD §6.
- **`broker-safety-reviewer`** subagent — invoke (or let it auto-trigger) when writing/modifying broker API integration, credential handling, or trading-mode switching code, to check it against PRD §7.
- **`/risk-check`** skill — validate a proposed strategy/order plan against PRD §6 risk limits, usable even before implementation code exists.

## Next steps

- Still unfinished: the unified Mamba model artifact has not been trained/committed yet (only the training/inference code exists, `scripts/train_ml_model.py` needs an actual run with real data).
- Main-loop trial: the deepagents loop is the default since 2026-10-12; remove legacy `loop.py`, its tests and the `CHAT_AGENT_LOOP` flag after ~2026-10-26 (decision D12). Still open from the plan: global concurrent-batch cap (second half of D17).
- US news (GDELT) is heavily rate-limited (HTTP 429) and slow; throttling/retry and legal-suffix-free queries were added but it remains unreliable — options and pending decisions in `docs/design/us-news-source-reliability.md`.
- Rotate the Naver Open API client secret: a real client ID was committed in `.env.example` in history (commit f788c18, now blanked).
- `docs/_review_parts/*.md` partly describe planned (not implemented) state — each file has a dated correction note at the top of the affected section.
