"""Manual verification: confirms the LLM 차트분석 에이전트(#3) wiring end-to-end.

Run from the project root so `.env` and `data/` resolve correctly:
    .venv/Scripts/python scripts/verify_llm_chart_analyst.py           (Windows, real API call)
    LLM_VERIFY_DRY_RUN=1 .venv/Scripts/python scripts/verify_llm_chart_analyst.py  (dry-run, no API call)
    .venv/bin/python scripts/verify_llm_chart_analyst.py                (macOS/Linux, real API call)

Two modes:

- `LLM_VERIFY_DRY_RUN=1` (cost: $0): fetches OHLCV for one sample ticker (005930),
  builds a `ChartSnapshot` via `build_snapshot`, and prints the exact system/user
  prompts that would be sent to OpenAI — without ever constructing an OpenAI client
  or making a network call. Use this to eyeball prompt content (docs/design/
  llm-chart-analyst-plan.md §검증 "수동(dry-run, 비용 0)").
- Otherwise: loads `LLMConfig` via `load_llm_config()` (requires `OPENAI_API_KEY` in
  `.env`), builds an `OpenAIChartClient` with `max_calls_per_run` forced to 1 (this
  script only ever needs a single call), and runs `analyze()` against the sample
  ticker's real OHLCV, printing the structured result.

Prints only counts/labels/rationale text — never prints OPENAI_API_KEY or any other
credential.
"""

import dataclasses
import os
import sys
from datetime import date, timedelta

from auto_stock.data.cache import OHLCVCache
from auto_stock.data.service import get_ohlcv
from auto_stock.llm_chart_analyst.analyst import analyze
from auto_stock.llm_chart_analyst.credentials import load_llm_config
from auto_stock.llm_chart_analyst.prompt import SYSTEM_PROMPT, build_user_prompt
from auto_stock.llm_chart_analyst.snapshot import build_snapshot
from auto_stock.orchestrator.pipeline import DEFAULT_LOOKBACK_DAYS

SAMPLE_TICKER = "005930"  # 삼성전자 — same smoke-test target as verify_pykrx.py
SAMPLE_MARKET = "KRX"


def _fetch_sample_records() -> list:
    cache = OHLCVCache("data/ohlcv.duckdb")
    end = date.today()
    start = end - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    return get_ohlcv(cache, SAMPLE_TICKER, start, end, SAMPLE_MARKET)


def _run_dry_run() -> None:
    records = _fetch_sample_records()
    snapshot = build_snapshot(records)
    if snapshot is None:
        print("FAILED: build_snapshot이 None을 반환했습니다 (워밍업 기간 부족 — SMA60 확보 필요)")
        sys.exit(1)

    print("SUCCESS: ChartSnapshot 생성 완료 (API 호출 없음, 비용 0)")
    print(f"봉 개수: {len(snapshot.bars)}, 지표 개수: {len(snapshot.indicators)}")
    print()
    print("--- SYSTEM PROMPT ---")
    print(SYSTEM_PROMPT)
    print("--- USER PROMPT ---")
    print(build_user_prompt(snapshot))


def _run_real_call() -> None:
    try:
        config = load_llm_config()
    except KeyError:
        print("FAILED: OPENAI_API_KEY가 .env에 설정되어 있지 않습니다")
        sys.exit(1)

    # 이 스크립트는 검증용 단일 호출만 필요하므로, 사용자의 LLM_MAX_CALLS_PER_RUN 설정과
    # 무관하게 예산을 1로 강제한다 — 실수로 여러 번 호출되는 사고를 방지한다.
    config = dataclasses.replace(config, max_calls_per_run=1)

    from auto_stock.llm_chart_analyst.client import LLMChartAnalystError, OpenAIChartClient

    client = OpenAIChartClient(config)
    records = _fetch_sample_records()

    try:
        analysis = analyze(client, records)
    except LLMChartAnalystError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)

    if analysis is None:
        print("FAILED: analyze()가 None을 반환했습니다 (워밍업 기간 부족 — SMA60 확보 필요)")
        sys.exit(1)

    print("SUCCESS: LLM 차트분석 응답 수신")
    print(f"모델: {analysis.model}")
    print(f"direction: {analysis.direction}")
    print(f"confidence: {analysis.confidence}")
    print(f"pattern_name: {analysis.pattern_name}")
    print(f"rationale: {analysis.rationale}")
    print(f"caveat: {analysis.caveat}")


if __name__ == "__main__":
    if os.environ.get("LLM_VERIFY_DRY_RUN") == "1":
        _run_dry_run()
    else:
        _run_real_call()
