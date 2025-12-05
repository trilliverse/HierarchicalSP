#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Main training launcher for node-classification tasks.
#
# Usage:
#   ./main.sh <GPU_ID> <DATASET>
#
# Example:
#   ./main.sh 0 cora
#
# Notes:
#   - This script launches main_node.py with the given GPU and dataset.
#   - Logs are stored under logs/results/ with timestamped filenames.
#   - Use `tail -f <log_file>` to monitor progress.
# -----------------------------------------------------------------------------

# set -Eeuo pipefail

# --------------------------- argument parsing --------------------------------
GPU_ID=$1        # GPU device ID (e.g., 0 or 1)
DATASET=$2       # Dataset name (e.g., cora, arxiv, products)

if [ -z "$GPU_ID" ] || [ -z "$DATASET" ]; then
  echo "Usage: $0 <GPU_ID> <DATASET>"
  echo "Example: $0 0 cora"
  exit 1
fi

# --------------------------- environment setup -------------------------------
export CUDA_VISIBLE_DEVICES=$GPU_ID
PYTHON_BIN="python"
MAIN_FILE="main.py"

# --------------------------- logging setup -----------------------------------
TIME_STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/main"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${DATASET}_${TIME_STAMP}.log"

# --------------------------- launch training ---------------------------------
echo "------------------------------------------------------------"
echo "Starting training..."
echo "GPU device : $CUDA_VISIBLE_DEVICES"
echo "Main file  : $MAIN_FILE"
echo "Dataset    : $DATASET"
echo "Log file   : $LOG_FILE"
echo "Start time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "------------------------------------------------------------"

nohup "$PYTHON_BIN" -u "$MAIN_FILE" --dataset "$DATASET" > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo "Use 'tail -f $LOG_FILE' to view training logs."