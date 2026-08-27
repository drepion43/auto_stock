"""MVP-0 + ML 보조 신호(#2) 실행 스크립트 — scripts/run_recommendations.py의 변형.

KRX 로컬 모델 아티팩트(models/ml_predictor/KRX_h5_logreg.joblib)를 로드해
run_recommendation_pipeline에 ml_model로 전달한다. 아티팩트가 없으면 안내
메시지를 출력하고 종료한다 — 먼저 scripts/train_ml_model.py를 실행해야 한다.

Run from the project root so `.env` and `data/` resolve correctly:
    .venv/Scripts/python scripts/run_recommendations_with_ml.py   (Windows)
    .venv/bin/python scripts/run_recommendations_with_ml.py        (macOS/Linux)

WATCHLIST below is a small example list, not the full market universe (see
run_recommendations.py for the same caveat). AccountState is a placeholder for
the same reason as run_recommendations.py (no broker integration yet).

Prints only counts — never prints TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, or ACCOUNT_EQUITY.
"""

import os
import sys

from auto_stock.data.cache import OHLCVCache
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

cache = OHLCVCache("data/ohlcv.duckdb")
credentials = load_telegram_credentials()
account = AccountState(
    equity=float(os.environ.get("ACCOUNT_EQUITY", DEFAULT_ACCOUNT_EQUITY)),
    held_tickers=frozenset(),
    total_exposure_pct=0.0,
)

result = run_recommendation_pipeline(
    cache=cache, tickers=WATCHLIST, market=MARKET, account=account, credentials=credentials,
    ml_model=ml_model,
)

print(f"전송된 추천: {len(result.sent)}건")
print(f"에러: {len(result.errors)}건")
for ticker, message in result.errors:
    print(f"  - {ticker}: {message}")
