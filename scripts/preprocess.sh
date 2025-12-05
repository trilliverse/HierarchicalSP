#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Preprocess datasets for node-classification tasks.
#
# Usage:
#   ./preprocess.sh <DATASET>
#
# Example:
#   ./preprocess.sh arxiv
#
# Notes:
#   - This script calls `preprocess_node_data.py` with a single dataset argument.
#   - Logs are stored in `logs/preprocess/` with timestamped filenames.
#   - Use `tail -f <log_file>` to monitor progress in real time.
# -----------------------------------------------------------------------------

# set -Eeuo pipefail

# ----------------------------- configuration ---------------------------------
PYTHON_BIN="python"
MAIN_FILE="preprocess.py"
LOG_DIR="logs/preprocess"

# ----------------------------- argument check --------------------------------
if [ $# -lt 1 ]; then
  echo "Usage: $0 <DATASET>"
  echo "Example: $0 arxiv"
  exit 1
fi

DATASET=$1

# ----------------------------- logging setup ---------------------------------
mkdir -p "$LOG_DIR"
TIME_STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${DATASET}_${TIME_STAMP}.log"

# ----------------------------- run preprocessing -----------------------------
echo "------------------------------------------------------------"
echo "Starting preprocessing..."
echo "Python    : $PYTHON_BIN"
echo "Main file : $MAIN_FILE"
echo "Dataset   : $DATASET"
echo "Log file  : $LOG_FILE"
echo "------------------------------------------------------------"

nohup "$PYTHON_BIN" -u "$MAIN_FILE" --dataset "$DATASET" > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo "Use 'tail -f $LOG_FILE' to follow the log."