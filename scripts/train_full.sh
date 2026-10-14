#!/usr/bin/env bash
# Full ML training run (Mamba, unified model) with a timestamped log file.
#
# Defaults are the full-scale settings of scripts/train_ml_model.py
# (200 tickers per market, 5 years, 100 epochs). Any ML_TRAIN_* variable can be
# overridden from the caller, e.g.:
#   scripts/train_full.sh
#   ML_TRAIN_EPOCHS=40 scripts/train_full.sh
#   ML_TRAIN_DEVICE=cpu scripts/train_full.sh
#
# Output: logs/ml_training/train_full_<YYYYmmdd_HHMM>.log, plus the model,
# metadata and learning-curve files under models/ml_predictor/.
set -euo pipefail

# Run from the project root so data/ and models/ resolve regardless of the caller's cwd.
cd "$(dirname "$0")/.."

if [[ -x .venv/Scripts/python ]]; then
    PYTHON=.venv/Scripts/python   # Windows (Git Bash)
elif [[ -x .venv/bin/python ]]; then
    PYTHON=.venv/bin/python       # macOS/Linux
else
    echo "No .venv found. Create it first: uv sync --extra dev" >&2
    exit 1
fi

export ML_TRAIN_DEVICE="${ML_TRAIN_DEVICE:-cuda}"   # fails fast before data collection if no GPU
export ML_TRAIN_LOG_EVERY="${ML_TRAIN_LOG_EVERY:-5}"
export PYTHONUTF8=1                                  # keep Korean output intact when piped through tee

LOG_DIR=logs/ml_training
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_full_$(date +%Y%m%d_%H%M).log"

echo "Log file: $LOG_FILE"
# pipefail makes the script exit non-zero if training fails, even though tee succeeds.
"$PYTHON" -u scripts/train_ml_model.py 2>&1 | tee "$LOG_FILE"
