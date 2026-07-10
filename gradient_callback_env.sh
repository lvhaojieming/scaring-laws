#!/usr/bin/env bash
set -Eeuo pipefail

# Gradient callback runtime settings.
# Add these exports to the existing GPT-W2560 training script.

LOG_DIR="${LOG_DIR:-/datadisk_2/logs/gpt-w2560-l32-h2560-ffn8960-2p7265b-5btok}"

export GRAD_CB_ENABLED="${GRAD_CB_ENABLED:-1}"
export GRAD_CB_LOG_INTERVAL="${GRAD_CB_LOG_INTERVAL:-10}"
export GRAD_CB_TOPK="${GRAD_CB_TOPK:-20}"

# Absolute thresholds
export GRAD_CB_MAX_GLOBAL_NORM="${GRAD_CB_MAX_GLOBAL_NORM:-10000}"
export GRAD_CB_MAX_ABS_GRAD="${GRAD_CB_MAX_ABS_GRAD:-1000}"

# Relative spike threshold
export GRAD_CB_EMA_BETA="${GRAD_CB_EMA_BETA:-0.98}"
export GRAD_CB_SPIKE_FACTOR="${GRAD_CB_SPIKE_FACTOR:-8}"
export GRAD_CB_MIN_STEPS="${GRAD_CB_MIN_STEPS:-50}"

# Trigger switches
export GRAD_CB_TRIGGER_NONFINITE="${GRAD_CB_TRIGGER_NONFINITE:-1}"
export GRAD_CB_TRIGGER_SPIKE="${GRAD_CB_TRIGGER_SPIKE:-1}"
export GRAD_CB_TRIGGER_ABS="${GRAD_CB_TRIGGER_ABS:-1}"
export GRAD_CB_TRIGGER_GLOBAL_NORM="${GRAD_CB_TRIGGER_GLOBAL_NORM:-1}"
export GRAD_CB_TRIGGER_NONFINITE_LOSS="${GRAD_CB_TRIGGER_NONFINITE_LOSS:-1}"

# Callback actions
export GRAD_CB_SAVE_REPORT="${GRAD_CB_SAVE_REPORT:-1}"
export GRAD_CB_SAVE_SNAPSHOT="${GRAD_CB_SAVE_SNAPSHOT:-1}"
export GRAD_CB_EMERGENCY_CKPT="${GRAD_CB_EMERGENCY_CKPT:-1}"
export GRAD_CB_SKIP_STEP="${GRAD_CB_SKIP_STEP:-1}"
export GRAD_CB_ABORT="${GRAD_CB_ABORT:-0}"

export GRAD_CB_OUTPUT_DIR="${GRAD_CB_OUTPUT_DIR:-$LOG_DIR/gradient_anomalies}"

mkdir -p "$GRAD_CB_OUTPUT_DIR"

echo "Gradient anomaly callback configuration:"
env | grep '^GRAD_CB_' | sort
