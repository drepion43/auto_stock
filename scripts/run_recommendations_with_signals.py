"""MVP-0 + ML 보조 신호(#2) + LLM 차트분석 보조 신호(#3) 동시 실행 스크립트
— scripts/run_recommendations_with_ml.py의 변형(docs/design/llm-chart-analyst-plan.md
"오케스트레이터 통합" / Phase 3).

KRX 로컬 ML 모델 아티팩트(models/ml_predictor/KRX_h5_logreg.joblib)와 OpenAI LLM
크리덴셜(.env의 OPENAI_API_KEY)을 모두 로드해 run_recommendation_pipeline에
ml_model/llm_client로 함께 전달한다. 둘 중 하나라도 없으면 안내 메시지를 출력하고
종료한다 — ML 아티팩트가 없으면 먼저 scripts/train_ml_model.py를, OPENAI_API_KEY가
없으면 .env에 발급받은 키를 채워야 한다.

Run from the project root so `.env` and `data/` resolve correctly:
    .venv/Scripts/python scripts/run_recommendations_with_signals.py   (Windows)
    .venv/bin/python scripts/run_recommendations_with_signals.py        (macOS/Linux)

이 스크립트는 실제 OpenAI API를 호출한다(비용 발생) — OPENAI_API_KEY 발급 전에는
실행하지 말 것. 실행 전 scripts/verify_llm_chart_analyst.py의 dry-run(무료) /
실호출(유료, 1회) 모드로 먼저 배선을 확인하는 것을 권장한다.

WATCHLIST below is a small example list, not the full market universe (see
run_recommendations.py for the same caveat). AccountState is a placeholder for
the same reason as run_recommendations.py (no broker integration yet).

Prints only counts — never prints TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
OPENAI_API_KEY, or ACCOUNT_EQUITY.
"""

import os
import sys

from auto_stock.data.cache import OHLCVCache
from auto_stock.llm_chart_analyst.client import OpenAIChartClient
from auto_stock.llm_chart_analyst.credentials import load_llm_config
from auto_stock.ml_predictor.artifact import ModelArtifactError, load_model
from auto_stock.notifier.credentials import load_telegram_credentials
from auto_stock.orchestrator.pipeline import run_recommendation_pipeline
from auto_stock.risk_sizing.models import AccountState

WATCHLIST = ["005930", "000660"]  # 삼성전자, SK하이닉스 — example only
MARKET = "KRX"
DEFAULT_ACCOUNT_EQUITY = 10_000_000.0

try:
    ml_model = load_model(MARKET)
except ModelArtifactError as exc:
    print(str(exc))
    sys.exit(1)

try:
    llm_config = load_llm_config()
except KeyError:
    print("FAILED: OPENAI_API_KEY가 .env에 설정되어 있지 않습니다 — LLM 차트분석 보조 신호를 켜려면 먼저 발급/설정하세요.")
    sys.exit(1)

llm_client = OpenAIChartClient(llm_config)

cache = OHLCVCache("data/ohlcv.duckdb")
credentials = load_telegram_credentials()
account = AccountState(
    equity=float(os.environ.get("ACCOUNT_EQUITY", DEFAULT_ACCOUNT_EQUITY)),
    held_tickers=frozenset(),
    total_exposure_pct=0.0,
)

result = run_recommendation_pipeline(
    cache=cache, tickers=WATCHLIST, market=MARKET, account=account, credentials=credentials,
    ml_model=ml_model, llm_client=llm_client,
)

print(f"전송된 추천: {len(result.sent)}건")
print(f"에러: {len(result.errors)}건")
for ticker, message in result.errors:
    print(f"  - {ticker}: {message}")
