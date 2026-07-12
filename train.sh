#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# GPT-W2560 / L32 / H2560 / FFN8960 / GQA 20:5
# Approx. parameters: 3.3831B total with untied embeddings
# Transformer body only: 2.7265B
#
# Dataset mixture:
#   C4              35%
#   FineWeb-Edu     30%
#   SlimPajama      20%
#   OpenWebText2    10%
#   Wikipedia        1.67%
#   StackExchange    1.67%
#   ArXiv            1.66%
#
# Checkpoint policy:
#   - Persistent: exactly 1B, 3B, and 5B token milestones only.
#   - Persistent contents: model weights + required checkpoint metadata only;
#     optimizer, LR scheduler, and RNG state are excluded.
#   - Rollback: one full restart checkpoint only, refreshed every 250 steps.
#
# Default behavior: dry run.
# Launch:
#   RUN_TRAIN=1 bash pretraining.sh
# ============================================================

# ----------------------------
# 0. User-configurable paths
# ----------------------------
DATASET_ROOT="${DATASET_ROOT:-/root/scaling-laws}"
MODEL_STORAGE_ROOT="${MODEL_STORAGE_ROOT:-$DATASET_ROOT/training_artifacts}"
PROJECT_DIR="${PROJECT_DIR:-/datadisk_1/projects}"
MEGATRON_DIR="${MEGATRON_DIR:-$PROJECT_DIR/Megatron-LM}"
DATA_DIR="${DATA_DIR:-$DATASET_ROOT/megatron_llama3_by_source}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$DATASET_ROOT/tokenizer/llama3}"

RUN_NAME="${RUN_NAME:-gpt-w2560-l32-h2560-ffn8960-2p7265b-5btok}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$MODEL_STORAGE_ROOT/checkpoints}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$CHECKPOINT_ROOT/$RUN_NAME}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$MODEL_STORAGE_ROOT/tensorboard/$RUN_NAME}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-$MODEL_STORAGE_ROOT/data_cache/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$DATASET_ROOT/logs/$RUN_NAME}"
WANDB_DIR="${WANDB_DIR:-$MODEL_STORAGE_ROOT/wandb/$RUN_NAME}"
GRAD_ANOMALY_DIR="${GRAD_ANOMALY_DIR:-$MODEL_STORAGE_ROOT/gradient_anomalies/$RUN_NAME}"
ROLLBACK_CURRENT_DIR="${ROLLBACK_CURRENT_DIR:-$CHECKPOINT_DIR/rollback_current}"
RUNTIME_PATCH_DIR="${RUNTIME_PATCH_DIR:-$LOG_DIR/runtime_patch}"
CHECKPOINT_POLICY_WRAPPER="${CHECKPOINT_POLICY_WRAPPER:-$RUNTIME_PATCH_DIR/pretrain_gpt_checkpoint_policy.py}"

mkdir -p \
  "$CHECKPOINT_DIR" \
  "$TENSORBOARD_DIR" \
  "$DATA_CACHE_DIR" \
  "$LOG_DIR" \
  "$WANDB_DIR" \
  "$GRAD_ANOMALY_DIR" \
  "$ROLLBACK_CURRENT_DIR" \
  "$RUNTIME_PATCH_DIR"

# ----------------------------
# 1. Runtime environment
# ----------------------------
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-$(command -v python3)}"
export WANDB_DIR
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONPATH="/root:$MEGATRON_DIR:${PYTHONPATH:-}"

# ----------------------------
# 2. Distributed configuration
# ----------------------------
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6000}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-4}"
CP_SIZE="${CP_SIZE:-1}"

# ----------------------------
# 3. Training configuration
# ----------------------------
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-240}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"

# 240 * 2048 = 491,520 tokens/iteration
# 5B / 491,520 ≈ 10172.53
TRAIN_ITERS="${TRAIN_ITERS:-10173}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-10173}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-2000}"

MAX_LR="${MAX_LR:-3.0e-4}"
MIN_LR="${MIN_LR:-3.0e-5}"

# Persistent checkpoint scheduling is handled by a small runtime wrapper so
# Megatron can save model-only checkpoints at non-uniform milestone iterations.
# --save-interval is still required by Megatron argument validation; setting it
# to TRAIN_ITERS also prevents an extra non-milestone persistent save.
PERSISTENT_SAVE_INTERVAL="${PERSISTENT_SAVE_INTERVAL:-$TRAIN_ITERS}"
ROLLBACK_SAVE_INTERVAL="${ROLLBACK_SAVE_INTERVAL:-250}"
DROP_OLD_ROLLBACK_BEFORE_SAVE="${DROP_OLD_ROLLBACK_BEFORE_SAVE:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
EVAL_ITERS="${EVAL_ITERS:-20}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-none}"
ENABLE_MANUAL_GC="${ENABLE_MANUAL_GC:-0}"
EMPTY_UNUSED_MEMORY_LEVEL="${EMPTY_UNUSED_MEMORY_LEVEL:-0}"

# Only these persistent milestones are allowed. They are mapped to the nearest
# optimizer iteration, while the final 5B point is fixed to TRAIN_ITERS.
KEEP_TOKEN_MILESTONES="${KEEP_TOKEN_MILESTONES:-1000000000 3000000000 5000000000}"


# ----------------------------
# 4. Monitoring configuration
# ----------------------------
ENABLE_SYSTEM_MONITOR="${ENABLE_SYSTEM_MONITOR:-1}"
SYSTEM_MONITOR_INTERVAL="${SYSTEM_MONITOR_INTERVAL:-10}"
ENABLE_RUN_MANIFEST="${ENABLE_RUN_MANIFEST:-1}"
ENABLE_GRADIENT_ANOMALY_CALLBACK="${ENABLE_GRADIENT_ANOMALY_CALLBACK:-1}"
ENABLE_ANOMALY_WATCHER="${ENABLE_ANOMALY_WATCHER:-1}"
ANOMALY_WATCH_INTERVAL="${ANOMALY_WATCH_INTERVAL:-15}"

WANDB_PROJECT="${WANDB_PROJECT:-width-scaling-5btoken}"
WANDB_NAME="${WANDB_NAME:-gpt-w2560-h2560-5btok}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-${WANDB_NAME}}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
VERIFY_WANDB_LOGIN="${VERIFY_WANDB_LOGIN:-0}"
WANDB_LOGIN_VERIFY_TIMEOUT="${WANDB_LOGIN_VERIFY_TIMEOUT:-20}"
FRESH_START_NEW_WANDB_RUN="${FRESH_START_NEW_WANDB_RUN:-1}"

RUN_TRAIN="${RUN_TRAIN:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
AUTO_RESUME_ROLLBACK="${AUTO_RESUME_ROLLBACK:-1}"
ALLOW_GPU_MISMATCH="${ALLOW_GPU_MISMATCH:-0}"
ALLOW_LOW_CHECKPOINT_SPACE="${ALLOW_LOW_CHECKPOINT_SPACE:-1}"
MIN_CHECKPOINT_FREE_GB="${MIN_CHECKPOINT_FREE_GB:-50}"
REQUIRE_GRADIENT_CALLBACK_INSTALLED="${REQUIRE_GRADIENT_CALLBACK_INSTALLED:-0}"

# Gradient/data anomaly runtime configuration. The Megatron training loop reads
# these values through /root/gradient_anomaly_callback.py.
export GRAD_CB_ENABLED="${GRAD_CB_ENABLED:-$ENABLE_GRADIENT_ANOMALY_CALLBACK}"
export GRAD_CB_LOG_INTERVAL="${GRAD_CB_LOG_INTERVAL:-$LOG_INTERVAL}"
export GRAD_CB_TOPK="${GRAD_CB_TOPK:-20}"
export GRAD_CB_MAX_GLOBAL_NORM="${GRAD_CB_MAX_GLOBAL_NORM:-10000}"
export GRAD_CB_MAX_ABS_GRAD="${GRAD_CB_MAX_ABS_GRAD:-1000}"
export GRAD_CB_EMA_BETA="${GRAD_CB_EMA_BETA:-0.98}"
export GRAD_CB_SPIKE_FACTOR="${GRAD_CB_SPIKE_FACTOR:-8}"
export GRAD_CB_MIN_STEPS="${GRAD_CB_MIN_STEPS:-50}"
export GRAD_CB_TRIGGER_NONFINITE="${GRAD_CB_TRIGGER_NONFINITE:-1}"
export GRAD_CB_TRIGGER_SPIKE="${GRAD_CB_TRIGGER_SPIKE:-1}"
export GRAD_CB_TRIGGER_ABS="${GRAD_CB_TRIGGER_ABS:-1}"
export GRAD_CB_TRIGGER_GLOBAL_NORM="${GRAD_CB_TRIGGER_GLOBAL_NORM:-1}"
export GRAD_CB_TRIGGER_NONFINITE_LOSS="${GRAD_CB_TRIGGER_NONFINITE_LOSS:-1}"
export GRAD_CB_SAVE_REPORT="${GRAD_CB_SAVE_REPORT:-1}"
export GRAD_CB_SAVE_SNAPSHOT="${GRAD_CB_SAVE_SNAPSHOT:-1}"
export GRAD_CB_EMERGENCY_CKPT="${GRAD_CB_EMERGENCY_CKPT:-0}"
export GRAD_CB_SKIP_STEP="${GRAD_CB_SKIP_STEP:-1}"
export GRAD_CB_ABORT="${GRAD_CB_ABORT:-0}"
export GRAD_CB_OUTPUT_DIR="${GRAD_CB_OUTPUT_DIR:-$GRAD_ANOMALY_DIR}"
export DATA_ANOMALY_CHECK_ENABLED="${DATA_ANOMALY_CHECK_ENABLED:-1}"
export DATA_ANOMALY_OUTPUT_DIR="${DATA_ANOMALY_OUTPUT_DIR:-$GRAD_CB_OUTPUT_DIR}"

# ----------------------------
# 5. Model constants
# ----------------------------
MODEL_NUM_LAYERS=32
MODEL_HIDDEN_SIZE=2560
MODEL_FFN_HIDDEN_SIZE=8960
MODEL_NUM_HEADS=20
MODEL_NUM_QUERY_GROUPS=5
MODEL_KV_CHANNELS=128
MODEL_APPROX_PARAMS="3.3831B total"
MODEL_TRANSFORMER_BODY_PARAMS="2.7265B"

# ----------------------------
# 6. Dataset mixture
# ----------------------------
# Total weight = 100.00
data_path=(
  35            "${C4_PREFIX:-$DATA_DIR/c4/c4_text_document}"
  30            "${FINEWEB_EDU_PREFIX:-$DATA_DIR/fineweb_edu/fineweb_edu_text_document}"
  20            "${SLIMPAJAMA_PREFIX:-$DATA_DIR/slimpajama/slimpajama_text_document}"
  10            "${OPENWEBTEXT2_PREFIX:-$DATA_DIR/openwebtext2/openwebtext2_text_document}"
  1.6666666667  "${WIKIPEDIA_PREFIX:-$DATA_DIR/wikipedia/wikipedia_text_document}"
  1.6666666667  "${STACKEXCHANGE_PREFIX:-$DATA_DIR/stackexchange/stackexchange_text_document}"
  1.6666666666  "${ARXIV_PREFIX:-$DATA_DIR/arxiv/arxiv_text_document}"
)

# ----------------------------
# 7. Validation helpers
# ----------------------------
fail() {
  echo "ERROR: $*" >&2
  exit 1
}

command -v "$PYTHON" >/dev/null 2>&1 || fail "Python not found: $PYTHON"
command -v torchrun >/dev/null 2>&1 || fail "torchrun not found in PATH."
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found."
command -v wandb >/dev/null 2>&1 || fail "wandb CLI not found."

[[ -f "$MEGATRON_DIR/pretrain_gpt.py" ]] \
  || fail "Missing $MEGATRON_DIR/pretrain_gpt.py"

[[ -d "$TOKENIZER_DIR" ]] \
  || fail "Missing tokenizer directory: $TOKENIZER_DIR"

[[ -f /root/gradient_anomaly_callback.py ]] \
  || fail "Missing /root/gradient_anomaly_callback.py"

if [[ "$ENABLE_GRADIENT_ANOMALY_CALLBACK" == "1" ]]; then
  if grep -R "GradientAnomalyCallback\|gradient_anomaly_callback" "$MEGATRON_DIR" >/dev/null 2>&1; then
    GRADIENT_CALLBACK_INSTALLED=1
  else
    GRADIENT_CALLBACK_INSTALLED=0
  fi

  if [[ "$REQUIRE_GRADIENT_CALLBACK_INSTALLED" == "1" \
        && "$GRADIENT_CALLBACK_INSTALLED" != "1" ]]; then
    fail "Gradient anomaly callback is not wired into $MEGATRON_DIR. Set REQUIRE_GRADIENT_CALLBACK_INSTALLED=0 to launch anyway."
  fi
else
  GRADIENT_CALLBACK_INSTALLED=0
fi

for ((i=1; i<${#data_path[@]}; i+=2)); do
  prefix="${data_path[$i]}"
  [[ -f "${prefix}.bin" ]] || fail "Missing dataset file: ${prefix}.bin"
  [[ -f "${prefix}.idx" ]] || fail "Missing dataset file: ${prefix}.idx"
done

detected_gpus="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
checkpoint_free_gb="$(df -BG "$CHECKPOINT_ROOT" | awk 'NR==2 {gsub(/G/,"",$4); print $4}')"

if [[ "$RUN_TRAIN" == "1" && "$ALLOW_GPU_MISMATCH" != "1" \
      && "$detected_gpus" != "$GPUS_PER_NODE" ]]; then
  fail "Detected ${detected_gpus} GPU(s), but GPUS_PER_NODE=${GPUS_PER_NODE}."
fi

if [[ "$RUN_TRAIN" == "1" && "$checkpoint_free_gb" -lt "$MIN_CHECKPOINT_FREE_GB" ]]; then
  fail "Checkpoint disk has only ${checkpoint_free_gb}G free; minimum safety floor is ${MIN_CHECKPOINT_FREE_GB}G: $CHECKPOINT_ROOT"
fi

if [[ "$RUN_TRAIN" == "1" && "$checkpoint_free_gb" -lt 250 ]]; then
  if [[ "$ALLOW_LOW_CHECKPOINT_SPACE" == "1" ]]; then
    echo "WARNING: Checkpoint disk has only ${checkpoint_free_gb}G free; continuing because ALLOW_LOW_CHECKPOINT_SPACE=1."
  else
    fail "Checkpoint disk has only ${checkpoint_free_gb}G free: $CHECKPOINT_ROOT"
  fi
fi

if [[ "$WORLD_SIZE" -ne $((TP_SIZE * PP_SIZE * DP_SIZE)) ]]; then
  fail "WORLD_SIZE=${WORLD_SIZE}, but TP*PP*DP=$((TP_SIZE * PP_SIZE * DP_SIZE))."
fi

if [[ $((GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DP_SIZE))) -ne 0 ]]; then
  fail "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by MICRO_BATCH_SIZE*DP_SIZE=$((MICRO_BATCH_SIZE * DP_SIZE))."
fi

if [[ "$PERSISTENT_SAVE_INTERVAL" -ne "$TRAIN_ITERS" ]]; then
  fail "For milestone-only persistent checkpoints, PERSISTENT_SAVE_INTERVAL must equal TRAIN_ITERS (${TRAIN_ITERS}); got ${PERSISTENT_SAVE_INTERVAL}."
fi

if [[ $((TRAIN_ITERS % PERSISTENT_SAVE_INTERVAL)) -ne 0 ]]; then
  fail "TRAIN_ITERS=${TRAIN_ITERS} must be divisible by PERSISTENT_SAVE_INTERVAL=${PERSISTENT_SAVE_INTERVAL}."
fi

if [[ $((MODEL_HIDDEN_SIZE % MODEL_NUM_HEADS)) -ne 0 ]]; then
  fail "hidden_size must be divisible by num_attention_heads."
fi

if [[ $((MODEL_NUM_HEADS % MODEL_NUM_QUERY_GROUPS)) -ne 0 ]]; then
  fail "num_attention_heads must be divisible by num_query_groups."
fi

if [[ $((MODEL_HIDDEN_SIZE / MODEL_NUM_HEADS)) -ne "$MODEL_KV_CHANNELS" ]]; then
  fail "hidden_size/num_attention_heads must equal kv_channels."
fi

# Tensor-parallel divisibility checks. With GQA, query groups must also be
# divisible by TP. GPT-W2560 uses five query groups, so TP=2/4 is invalid.
if [[ $((MODEL_HIDDEN_SIZE % TP_SIZE)) -ne 0 ]]; then
  fail "hidden_size=${MODEL_HIDDEN_SIZE} must be divisible by TP_SIZE=${TP_SIZE}."
fi
if [[ $((MODEL_FFN_HIDDEN_SIZE % TP_SIZE)) -ne 0 ]]; then
  fail "ffn_hidden_size=${MODEL_FFN_HIDDEN_SIZE} must be divisible by TP_SIZE=${TP_SIZE}."
fi
if [[ $((MODEL_NUM_HEADS % TP_SIZE)) -ne 0 ]]; then
  fail "num_attention_heads=${MODEL_NUM_HEADS} must be divisible by TP_SIZE=${TP_SIZE}."
fi
if [[ $((MODEL_NUM_QUERY_GROUPS % TP_SIZE)) -ne 0 ]]; then
  fail "num_query_groups=${MODEL_NUM_QUERY_GROUPS} must be divisible by TP_SIZE=${TP_SIZE}."
fi
if [[ $((MODEL_NUM_LAYERS % PP_SIZE)) -ne 0 ]]; then
  fail "num_layers=${MODEL_NUM_LAYERS} must be divisible by PP_SIZE=${PP_SIZE} for this non-interleaved setup."
fi
if [[ "$CP_SIZE" -ne 1 ]]; then
  fail "This 2048-token configuration is tuned for CP_SIZE=1."
fi

GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * DP_SIZE)))
RUN_TOPOLOGY="tp${TP_SIZE}_pp${PP_SIZE}_dp${DP_SIZE}_cp${CP_SIZE}"

if [[ "$RUN_TRAIN" == "1" && "$VERIFY_WANDB_LOGIN" == "1" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" ]] \
      && ! timeout "${WANDB_LOGIN_VERIFY_TIMEOUT}s" wandb login --verify >/dev/null 2>&1; then
    fail "W&B login verification failed or timed out. Run: wandb login"
  fi
fi

# Reuse the same W&B run only when there is a matching rollback checkpoint to
# resume. Fresh starts need a new run id; otherwise W&B rejects low step numbers
# after an older interrupted run has already logged later steps.
WANDB_RUN_ID_FILE="${WANDB_RUN_ID_FILE:-$WANDB_DIR/run_id.txt}"
ROLLBACK_TOPOLOGY_FILE="$ROLLBACK_CURRENT_DIR/topology.txt"
CAN_REUSE_WANDB_RUN_ID=0
if [[ -s "$ROLLBACK_CURRENT_DIR/latest_checkpointed_iteration.txt" \
      && -s "$ROLLBACK_TOPOLOGY_FILE" \
      && "$(tr -d '[:space:]' < "$ROLLBACK_TOPOLOGY_FILE")" == "$RUN_TOPOLOGY" ]]; then
  CAN_REUSE_WANDB_RUN_ID=1
fi

if [[ -z "${WANDB_RUN_ID:-}" && -s "$WANDB_RUN_ID_FILE" \
      && "$CAN_REUSE_WANDB_RUN_ID" == "1" ]]; then
  WANDB_RUN_ID="$(tr -d '[:space:]' < "$WANDB_RUN_ID_FILE")"
fi
if [[ "$RUN_TRAIN" == "1" && -z "${WANDB_RUN_ID:-}" ]]; then
  WANDB_RUN_ID="$("$PYTHON" - <<'PY_WANDB'
import wandb
print(wandb.util.generate_id())
PY_WANDB
)"
  printf '%s\n' "$WANDB_RUN_ID" > "$WANDB_RUN_ID_FILE"
fi
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  export WANDB_RUN_ID
  if [[ "$CAN_REUSE_WANDB_RUN_ID" == "1" ]]; then
    export WANDB_RESUME="${WANDB_RESUME:-allow}"
  elif [[ "$FRESH_START_NEW_WANDB_RUN" == "1" ]]; then
    export WANDB_RESUME="${WANDB_RESUME:-never}"
  else
    export WANDB_RESUME="${WANDB_RESUME:-allow}"
  fi
fi

# ----------------------------
# 8. Checkpoint policy
# ----------------------------
TOKENS_PER_ITER=$((GLOBAL_BATCH_SIZE * SEQ_LENGTH))
TARGET_TOKENS=$((TOKENS_PER_ITER * TRAIN_ITERS))
KEEP_ITERATIONS=()
KEEP_MILESTONE_TOKENS=()

# Map 1B and 3B to the nearest optimizer iteration. Map the terminal 5B
# milestone to TRAIN_ITERS so the final checkpoint is always preserved.
for milestone in $KEEP_TOKEN_MILESTONES; do
  raw_iter=$(((milestone + TOKENS_PER_ITER / 2) / TOKENS_PER_ITER))

  if [[ "$milestone" -ge $((TARGET_TOKENS - TOKENS_PER_ITER)) ]]; then
    save_iter="$TRAIN_ITERS"
  else
    save_iter="$raw_iter"
  fi

  if [[ "$save_iter" -ge 1 && "$save_iter" -le "$TRAIN_ITERS" ]]; then
    KEEP_ITERATIONS+=("$save_iter")
    KEEP_MILESTONE_TOKENS+=("$milestone")
  fi
done

if [[ "${#KEEP_ITERATIONS[@]}" -ne 3 ]]; then
  fail "Expected exactly three persistent milestones (1B/3B/5B); got iterations: ${KEEP_ITERATIONS[*]}"
fi

export MODEL_ONLY_MILESTONE_ITERATIONS="${KEEP_ITERATIONS[*]}"
export MODEL_ONLY_MILESTONE_TOKENS="${KEEP_MILESTONE_TOKENS[*]}"
export ROLLBACK_CURRENT_DIR
export CHECKPOINT_DIR
export ROLLBACK_SAVE_INTERVAL
export DROP_OLD_ROLLBACK_BEFORE_SAVE
export RUN_TOPOLOGY
export MEGATRON_PRETRAIN_ENTRY="$MEGATRON_DIR/pretrain_gpt.py"

is_keep_iteration() {
  local candidate="$1"
  local keep_iter

  for keep_iter in "${KEEP_ITERATIONS[@]}"; do
    if [[ "$candidate" == "$keep_iter" ]]; then
      return 0
    fi
  done

  return 1
}

find_iter_dir() {
  local root="$1"
  local iteration="$2"
  local candidate

  for candidate in \
    "$root/iter_$(printf '%07d' "$iteration")" \
    "$root/iter_$(printf '%06d' "$iteration")" \
    "$root/iter_$iteration"; do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

refresh_persistent_latest_marker() {
  local latest=0
  local keep_iter

  for keep_iter in "${KEEP_ITERATIONS[@]}"; do
    if find_iter_dir "$CHECKPOINT_DIR" "$keep_iter" >/dev/null 2>&1 \
       && [[ "$keep_iter" -gt "$latest" ]]; then
      latest="$keep_iter"
    fi
  done

  if [[ "$latest" -gt 0 ]]; then
    printf '%s\n' "$latest" > "$CHECKPOINT_DIR/.latest_checkpointed_iteration.txt.tmp"
    mv -f "$CHECKPOINT_DIR/.latest_checkpointed_iteration.txt.tmp" \
      "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
  else
    rm -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
  fi
}

prune_non_milestone_persistent_checkpoints() {
  local dir base iter

  shopt -s nullglob
  for dir in "$CHECKPOINT_DIR"/iter_*; do
    [[ -d "$dir" ]] || continue
    base="$(basename "$dir")"
    iter="${base#iter_}"
    iter=$((10#$iter))

    if is_keep_iteration "$iter"; then
      continue
    fi

    echo "Removing non-milestone persistent checkpoint: $dir"
    rm -rf -- "$dir"
  done
  shopt -u nullglob

  # Remove obsolete rollback-copy storage created by older script revisions.
  rm -rf -- "$CHECKPOINT_DIR/rollback_keep"
  refresh_persistent_latest_marker
}

prune_rollback_to_latest_shell() {
  local latest_file="$ROLLBACK_CURRENT_DIR/latest_checkpointed_iteration.txt"
  local keep_iter=""
  local dir base iter

  if [[ -s "$latest_file" ]]; then
    keep_iter="$(tr -dc '0-9' < "$latest_file" || true)"
    keep_iter="${keep_iter#0}"
    keep_iter="${keep_iter:-0}"
  fi

  if [[ -n "$keep_iter" && "$keep_iter" != "0" ]]; then
    if ! find_iter_dir "$ROLLBACK_CURRENT_DIR" "$keep_iter" >/dev/null 2>&1; then
      keep_iter="0"
    fi
  fi

  if [[ -z "$keep_iter" || "$keep_iter" == "0" ]]; then
    shopt -s nullglob
    for dir in "$ROLLBACK_CURRENT_DIR"/iter_*; do
      [[ -d "$dir" ]] || continue
      base="$(basename "$dir")"
      iter="${base#iter_}"
      iter=$((10#$iter))
      if [[ -z "$keep_iter" || "$iter" -gt "$keep_iter" ]]; then
        keep_iter="$iter"
      fi
    done
    shopt -u nullglob
  fi

  shopt -s nullglob
  for dir in "$ROLLBACK_CURRENT_DIR"/iter_*; do
    [[ -d "$dir" ]] || continue
    base="$(basename "$dir")"
    iter="${base#iter_}"
    iter=$((10#$iter))
    if [[ "$iter" != "${keep_iter:-0}" ]]; then
      echo "Removing stale rollback checkpoint: $dir"
      rm -rf -- "$dir"
    fi
  done
  shopt -u nullglob

  if [[ -n "$keep_iter" && "$keep_iter" != "0" ]]; then
    printf '%s\n' "$keep_iter" > "$latest_file"
  else
    rm -f "$latest_file"
  fi
}

write_checkpoint_policy_wrapper() {
  cat > "$CHECKPOINT_POLICY_WRAPPER" <<'PY_POLICY'
#!/usr/bin/env python3
"""Runtime checkpoint policy for GPT width-scaling pretraining.

Persistent checkpoints are model-only and are written only at configured
1B/3B/5B milestones. Non-persistent checkpoints retain optimizer/scheduler/RNG
state and only the newest rollback checkpoint is kept.
"""

from __future__ import annotations

import os
import runpy
import shutil
from pathlib import Path
from typing import Any

import torch

from megatron.training import checkpointing as _checkpointing
from megatron.training.global_vars import get_args as _get_args


_MILESTONES = {
    int(value)
    for value in os.environ.get("MODEL_ONLY_MILESTONE_ITERATIONS", "").split()
    if value.strip()
}
_ROLLBACK_DIR = Path(os.environ["ROLLBACK_CURRENT_DIR"])
_PERSISTENT_DIR = Path(os.environ["CHECKPOINT_DIR"])
_PRETRAIN_ENTRY = os.environ["MEGATRON_PRETRAIN_ENTRY"]
_DROP_OLD_ROLLBACK_BEFORE_SAVE = os.environ.get("DROP_OLD_ROLLBACK_BEFORE_SAVE", "1") == "1"
_RUN_TOPOLOGY = os.environ["RUN_TOPOLOGY"]

if not _MILESTONES:
    raise RuntimeError("MODEL_ONLY_MILESTONE_ITERATIONS is empty")


_original_save_checkpoint = _checkpointing.save_checkpoint


def _save_checkpoint_with_policy(*args: Any, **kwargs: Any):
    """Save persistent milestones without optimizer/scheduler/RNG state."""
    non_persistent = bool(kwargs.get("non_persistent_ckpt", False))
    global_args = _get_args()

    if non_persistent:
        # Full rollback checkpoint: model + optimizer + scheduler + RNG.
        return _original_save_checkpoint(*args, **kwargs)

    old_no_save_optim = bool(getattr(global_args, "no_save_optim", False))
    old_no_save_rng = bool(getattr(global_args, "no_save_rng", False))
    global_args.no_save_optim = True
    global_args.no_save_rng = True

    try:
        return _original_save_checkpoint(*args, **kwargs)
    finally:
        global_args.no_save_optim = old_no_save_optim
        global_args.no_save_rng = old_no_save_rng


# training.py imports save_checkpoint by value, so patch checkpointing before
# importing the training module.
_checkpointing.save_checkpoint = _save_checkpoint_with_policy

from megatron.training import training as _training  # noqa: E402


def _read_tracker(root: Path) -> int:
    tracker = root / "latest_checkpointed_iteration.txt"
    try:
        return int(tracker.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return 0


def _iter_dir(root: Path, iteration: int) -> Path:
    return root / f"iter_{iteration:07d}"


def _persistent_complete(iteration: int) -> bool:
    return _read_tracker(_PERSISTENT_DIR) >= iteration and _iter_dir(
        _PERSISTENT_DIR, iteration
    ).is_dir()


def _prepare_persistent_target(iteration: int) -> None:
    """Delete a partial milestone directory left by an interrupted save."""
    target = _iter_dir(_PERSISTENT_DIR, iteration)
    if torch.distributed.get_rank() == 0 and target.exists() and not _persistent_complete(iteration):
        shutil.rmtree(target)
    torch.distributed.barrier()


def _prune_rollback_to_latest() -> None:
    """Guarantee one and only one full rollback checkpoint after each save."""
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        latest = _read_tracker(_ROLLBACK_DIR)
        dirs = []
        if _ROLLBACK_DIR.exists():
            for path in _ROLLBACK_DIR.glob("iter_*"):
                if not path.is_dir():
                    continue
                try:
                    iteration = int(path.name.removeprefix("iter_"))
                except ValueError:
                    continue
                dirs.append((iteration, path))

        available_iterations = {iteration for iteration, _ in dirs}
        if latest not in available_iterations:
            latest = max(available_iterations) if available_iterations else 0

        for iteration, path in dirs:
            if iteration != latest:
                shutil.rmtree(path)

        tracker = _ROLLBACK_DIR / "latest_checkpointed_iteration.txt"
        if latest > 0:
            tracker.parent.mkdir(parents=True, exist_ok=True)
            tracker.write_text(f"{latest}\n", encoding="utf-8")
        elif tracker.exists():
            tracker.unlink()
    torch.distributed.barrier()


def _drop_existing_rollback_before_save(iteration: int) -> None:
    """Free disk before writing the next full rollback checkpoint.

    A full rollback checkpoint is large enough that this machine cannot hold
    two copies at once. Keeping the previous checkpoint until the new one is
    finalized is safer, but it requires substantially more disk.
    """
    if not _DROP_OLD_ROLLBACK_BEFORE_SAVE:
        return

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0 and _ROLLBACK_DIR.exists():
        for path in _ROLLBACK_DIR.glob("iter_*"):
            if path.is_dir() and path != _iter_dir(_ROLLBACK_DIR, iteration):
                shutil.rmtree(path)
        tracker = _ROLLBACK_DIR / "latest_checkpointed_iteration.txt"
        if tracker.exists():
            tracker.unlink()
        topology = _ROLLBACK_DIR / "topology.txt"
        if topology.exists():
            topology.unlink()
    torch.distributed.barrier()


def _save_full_rollback(
    iteration,
    model,
    optimizer,
    opt_param_scheduler,
    num_floating_point_operations_so_far,
    checkpointing_context,
    train_data_iterator,
):
    _drop_existing_rollback_before_save(iteration)
    _training.save_checkpoint_and_time(
        iteration,
        model,
        optimizer,
        opt_param_scheduler,
        num_floating_point_operations_so_far,
        checkpointing_context,
        non_persistent_ckpt=True,
        train_data_iterator=train_data_iterator,
    )
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        (_ROLLBACK_DIR / "topology.txt").write_text(f"{_RUN_TOPOLOGY}\n", encoding="utf-8")
    torch.distributed.barrier()
    _prune_rollback_to_latest()


def _checkpoint_and_decide_exit(
    model,
    optimizer,
    opt_param_scheduler,
    iteration,
    num_floating_point_operations_so_far,
    checkpointing_context,
    train_data_iterator,
):
    """Enforce milestone-only persistent saves and latest-only rollback saves."""
    args = _training.get_args()
    saved_checkpoint = False

    # On termination signals, save a full rollback checkpoint rather than an
    # off-policy persistent checkpoint.
    if args.exit_signal_handler:
        signal_handler = _training.get_signal_handler()
        if any(signal_handler.signals_received()):
            if args.save:
                _save_full_rollback(
                    iteration,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far,
                    checkpointing_context,
                    train_data_iterator,
                )
            _training.print_datetime("exiting program after receiving SIGTERM.")
            return True

    # Persistent model-only milestones have priority over periodic rollback.
    if args.save and iteration in _MILESTONES:
        if not _persistent_complete(iteration):
            _prepare_persistent_target(iteration)
            _training.save_checkpoint_and_time(
                iteration,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                checkpointing_context,
                train_data_iterator=train_data_iterator,
            )
        saved_checkpoint = True
    elif (
        args.save
        and args.non_persistent_save_interval
        and iteration % args.non_persistent_save_interval == 0
    ):
        _save_full_rollback(
            iteration,
            model,
            optimizer,
            opt_param_scheduler,
            num_floating_point_operations_so_far,
            checkpointing_context,
            train_data_iterator,
        )
        saved_checkpoint = True

    if args.exit_duration_in_mins:
        train_time = (_training.time.time() - _training._TRAIN_START_TIME) / 60.0
        done_cuda = torch.tensor(
            [train_time > args.exit_duration_in_mins], dtype=torch.int, device="cuda"
        )
        torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
        if done_cuda.item():
            if args.save and not saved_checkpoint:
                _save_full_rollback(
                    iteration,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far,
                    checkpointing_context,
                    train_data_iterator,
                )
            _training.print_datetime(f"exiting program after {train_time} minutes")
            return True

    if (
        args.exit_interval and iteration % args.exit_interval == 0
    ) or (
        args.phase_transition_iterations
        and iteration in args.phase_transition_iterations
    ):
        if args.save and not saved_checkpoint:
            _save_full_rollback(
                iteration,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                checkpointing_context,
                train_data_iterator,
            )
        _training.print_datetime(f"exiting program at iteration {iteration}")
        return True

    return False


_training.checkpoint_and_decide_exit = _checkpoint_and_decide_exit

# Execute the unmodified Megatron entry point under the patched policy.
runpy.run_path(_PRETRAIN_ENTRY, run_name="__main__")
PY_POLICY
  chmod 0755 "$CHECKPOINT_POLICY_WRAPPER"
}

write_checkpoint_policy_wrapper

# ----------------------------
# 9. Independent system monitoring
# ----------------------------
system_monitor_loop() {
  local train_pid="$1"
  local output_file="$LOG_DIR/system_metrics_$(date +%Y%m%d_%H%M%S).csv"

  echo "timestamp,gpu_index,gpu_util_percent,memory_used_mb,memory_total_mb,memory_util_percent,temperature_c,power_w,power_limit_w,sm_clock_mhz,memory_clock_mhz,cpu_load_1m,ram_used_mb,ram_total_mb,checkpoint_free_gb" \
    > "$output_file"

  while kill -0 "$train_pid" >/dev/null 2>&1; do
    local timestamp cpu_load ram_used ram_total disk_free
    timestamp="$(date --iso-8601=seconds)"
    cpu_load="$(awk '{print $1}' /proc/loadavg)"
    ram_used="$(free -m | awk '/Mem:/ {print $3}')"
    ram_total="$(free -m | awk '/Mem:/ {print $2}')"
    disk_free="$(df -BG "$CHECKPOINT_ROOT" | awk 'NR==2 {gsub(/G/,"",$4); print $4}')"

    nvidia-smi \
      --query-gpu=index,utilization.gpu,memory.used,memory.total,utilization.memory,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem \
      --format=csv,noheader,nounits \
      | while IFS= read -r line; do
          echo "${timestamp},${line},${cpu_load},${ram_used},${ram_total},${disk_free}" \
            >> "$output_file"
        done

    sleep "$SYSTEM_MONITOR_INTERVAL"
  done
}

# ----------------------------
# 10. Distributed and model arguments
# ----------------------------
DISTRIBUTED_ARGS=(
  --nproc_per_node "$GPUS_PER_NODE"
  --nnodes "$NUM_NODES"
  --node_rank "$NODE_RANK"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
)

MODEL_ARGS=(
  --use-mcore-models

  --num-layers "$MODEL_NUM_LAYERS"
  --hidden-size "$MODEL_HIDDEN_SIZE"
  --ffn-hidden-size "$MODEL_FFN_HIDDEN_SIZE"

  --num-attention-heads "$MODEL_NUM_HEADS"
  --group-query-attention
  --num-query-groups "$MODEL_NUM_QUERY_GROUPS"
  --kv-channels "$MODEL_KV_CHANNELS"

  --seq-length "$SEQ_LENGTH"
  --max-position-embeddings "$SEQ_LENGTH"

  --position-embedding-type rope
  --rotary-base 500000
  --rotary-percent 1.0
  --use-rope-scaling

  --transformer-impl transformer_engine
  --attention-backend "$ATTENTION_BACKEND"
  --use-flash-attn

  --attention-dropout 0.0
  --hidden-dropout 0.0

  --swiglu
  --normalization RMSNorm
  --init-method-std 0.02

  --untie-embeddings-and-output-weights
  --disable-bias-linear
)

PARALLEL_ARGS=(
  --tensor-model-parallel-size "$TP_SIZE"
  --pipeline-model-parallel-size "$PP_SIZE"
  --context-parallel-size "$CP_SIZE"

  --use-distributed-optimizer
  --overlap-grad-reduce
  --overlap-param-gather
)

TRAINING_ARGS=(
  --micro-batch-size "$MICRO_BATCH_SIZE"
  --global-batch-size "$GLOBAL_BATCH_SIZE"

  --train-iters "$TRAIN_ITERS"
  --lr-decay-iters "$LR_DECAY_ITERS"
  --lr-warmup-iters "$LR_WARMUP_ITERS"

  --lr "$MAX_LR"
  --min-lr "$MIN_LR"
  --lr-decay-style cosine

  --weight-decay 0.1
  --clip-grad 1.0

  --adam-beta1 0.9
  --adam-beta2 0.95
  --adam-eps 1.0e-8

  --bf16
  --grad-reduce-in-bf16
  --attention-softmax-in-fp32

  --cross-entropy-loss-fusion
  --cross-entropy-fusion-impl native
  --calculate-per-token-loss

)

if [[ "$RECOMPUTE_GRANULARITY" != "none" ]]; then
  TRAINING_ARGS+=(--recompute-granularity "$RECOMPUTE_GRANULARITY")
fi
if [[ "$ENABLE_MANUAL_GC" == "1" ]]; then
  TRAINING_ARGS+=(--manual-gc)
fi
if [[ "$EMPTY_UNUSED_MEMORY_LEVEL" -gt 0 ]]; then
  TRAINING_ARGS+=(--empty-unused-memory-level "$EMPTY_UNUSED_MEMORY_LEVEL")
fi

DATA_ARGS=(
  --data-path "${data_path[@]}"
  --split 99,1,0

  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model "$TOKENIZER_DIR"
  --vocab-size 128256

  --data-cache-path "$DATA_CACHE_DIR"
  --no-create-attention-mask-in-dataloader
  --num-workers 2
)

# Automatically resume only from the full rolling checkpoint. Persistent
# 1B/3B/5B checkpoints are model-only and are not exact training-resume points.
LOAD_DIR="${LOAD_DIR:-}"
LOAD_MODEL_ONLY="${LOAD_MODEL_ONLY:-0}"
if [[ -z "$LOAD_DIR" && "$AUTO_RESUME_ROLLBACK" == "1" \
      && -s "$ROLLBACK_CURRENT_DIR/latest_checkpointed_iteration.txt" ]]; then
  if [[ -s "$ROLLBACK_TOPOLOGY_FILE" \
        && "$(tr -d '[:space:]' < "$ROLLBACK_TOPOLOGY_FILE")" == "$RUN_TOPOLOGY" ]]; then
    LOAD_DIR="$ROLLBACK_CURRENT_DIR"
  else
    echo "WARNING: Existing rollback checkpoint has missing/mismatched topology marker; starting fresh for ${RUN_TOPOLOGY}."
    echo "         Existing rollback dir will be replaced at the first rollback save: $ROLLBACK_CURRENT_DIR"
  fi
fi

LOGGING_ARGS=(
  --log-interval "$LOG_INTERVAL"

  --eval-interval "$EVAL_INTERVAL"
  --eval-iters "$EVAL_ITERS"

  # Megatron writes only the final 5B persistent checkpoint. The 1B/3B
  # checkpoints are promoted from the rolling checkpoint stream by this script.
  --save "$CHECKPOINT_DIR"
  --save-interval "$PERSISTENT_SAVE_INTERVAL"

  --non-persistent-save-interval "$ROLLBACK_SAVE_INTERVAL"
  --non-persistent-ckpt-type global
  --non-persistent-global-ckpt-dir "$ROLLBACK_CURRENT_DIR"

  --ckpt-format torch_dist
  --dist-ckpt-workers 4

  --tensorboard-dir "$TENSORBOARD_DIR"
  --log-throughput
  --log-timers-to-tensorboard
  --log-memory-to-tensorboard
  --log-validation-ppl-to-tensorboard

  --distributed-timeout-minutes 60

  --wandb-project "$WANDB_PROJECT"
  --wandb-exp-name "$WANDB_EXP_NAME"
  --wandb-save-dir "$WANDB_DIR"
)

if [[ -n "$LOAD_DIR" ]]; then
  LOGGING_ARGS+=(--load "$LOAD_DIR")
  if [[ "$LOAD_MODEL_ONLY" == "1" ]]; then
    LOGGING_ARGS+=(--no-load-optim --no-load-rng)
  fi
fi

if [[ -n "$WANDB_ENTITY" ]]; then
  LOGGING_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi

CMD=(
  torchrun "${DISTRIBUTED_ARGS[@]}"
  "$CHECKPOINT_POLICY_WRAPPER"
  "${MODEL_ARGS[@]}"
  "${PARALLEL_ARGS[@]}"
  "${TRAINING_ARGS[@]}"
  "${DATA_ARGS[@]}"
  "${LOGGING_ARGS[@]}"
)

# ----------------------------
# 11. Run manifest
# ----------------------------
write_run_manifest() {
  local manifest="$LOG_DIR/run_manifest_$(date +%Y%m%d_%H%M%S).txt"

  {
    echo "============================================================"
    echo "Run metadata"
    echo "============================================================"
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "user=$(whoami)"
    echo "run_name=$RUN_NAME"
    echo "model_storage_root=$MODEL_STORAGE_ROOT"
    echo "project_dir=$PROJECT_DIR"
    echo "megatron_dir=$MEGATRON_DIR"
    echo "dataset_root=$DATASET_ROOT"
    echo "data_dir=$DATA_DIR"
    echo "tokenizer_dir=$TOKENIZER_DIR"
    echo "load_dir=${LOAD_DIR:-fresh_start}"
    echo "wandb_run_id=${WANDB_RUN_ID:-unset}"
    echo

    echo "============================================================"
    echo "Model"
    echo "============================================================"
    echo "model=GPT-W2560"
    echo "approx_params=$MODEL_APPROX_PARAMS"
    echo "transformer_body_params=$MODEL_TRANSFORMER_BODY_PARAMS"
    echo "num_layers=$MODEL_NUM_LAYERS"
    echo "hidden_size=$MODEL_HIDDEN_SIZE"
    echo "ffn_hidden_size=$MODEL_FFN_HIDDEN_SIZE"
    echo "num_attention_heads=$MODEL_NUM_HEADS"
    echo "num_query_groups=$MODEL_NUM_QUERY_GROUPS"
    echo "kv_channels=$MODEL_KV_CHANNELS"
    echo "head_dim=$((MODEL_HIDDEN_SIZE / MODEL_NUM_HEADS))"
    echo "sequence_length=$SEQ_LENGTH"
    echo

    echo "============================================================"
    echo "Dataset mixture"
    echo "============================================================"
    echo "C4=35%"
    echo "FineWeb-Edu=30%"
    echo "SlimPajama=20%"
    echo "OpenWebText2=10%"
    echo "Wikipedia=1.67%"
    echo "StackExchange=1.67%"
    echo "ArXiv=1.66%"
    echo

    echo "============================================================"
    echo "Training"
    echo "============================================================"
    echo "micro_batch_size=$MICRO_BATCH_SIZE"
    echo "global_batch_size=$GLOBAL_BATCH_SIZE"
    echo "train_iters=$TRAIN_ITERS"
    echo "tokens_per_iter=$TOKENS_PER_ITER"
    echo "target_tokens=$((TOKENS_PER_ITER * TRAIN_ITERS))"
    echo "warmup_iters=$LR_WARMUP_ITERS"
    echo "max_lr=$MAX_LR"
    echo "min_lr=$MIN_LR"
    echo "persistent_megatron_save_interval=$PERSISTENT_SAVE_INTERVAL"
    echo "rolling_rollback_save_interval=$ROLLBACK_SAVE_INTERVAL"
    echo "drop_old_rollback_before_save=$DROP_OLD_ROLLBACK_BEFORE_SAVE"
    echo "auto_resume_rollback=$AUTO_RESUME_ROLLBACK"
    echo "run_topology=$RUN_TOPOLOGY"
    echo "eval_interval=$EVAL_INTERVAL"
    echo "eval_iters=$EVAL_ITERS"
    echo "persistent_token_milestones=$KEEP_TOKEN_MILESTONES"
    echo "persistent_checkpoint_iterations=${KEEP_ITERATIONS[*]}"
    echo "persistent_checkpoint_contents=model_weights_plus_required_metadata_no_optimizer_no_scheduler_no_rng"
    echo "gradient_anomaly_callback_enabled=$ENABLE_GRADIENT_ANOMALY_CALLBACK"
    echo "gradient_anomaly_callback_installed=$GRADIENT_CALLBACK_INSTALLED"
    echo "gradient_anomaly_output_dir=$GRAD_CB_OUTPUT_DIR"
        echo "rollback_current_dir=$ROLLBACK_CURRENT_DIR"
        echo

    echo "============================================================"
    echo "Parallelism"
    echo "============================================================"
    echo "num_nodes=$NUM_NODES"
    echo "gpus_per_node=$GPUS_PER_NODE"
    echo "world_size=$WORLD_SIZE"
    echo "tensor_parallel=$TP_SIZE"
    echo "pipeline_parallel=$PP_SIZE"
    echo "data_parallel=$DP_SIZE"
    echo "context_parallel=$CP_SIZE"
    echo "gradient_accumulation_steps=$GRAD_ACCUM_STEPS"
    echo

    echo "============================================================"
    echo "Software"
    echo "============================================================"
    "$PYTHON" --version 2>&1 || true
    "$PYTHON" - <<'PY' 2>/dev/null || true
import torch
print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
PY

    if [[ -d "$MEGATRON_DIR/.git" ]]; then
      echo
      echo "Megatron git commit:"
      git -C "$MEGATRON_DIR" rev-parse HEAD 2>/dev/null || true
      git -C "$MEGATRON_DIR" status --short 2>/dev/null || true
    fi

    echo
    echo "============================================================"
    echo "GPU"
    echo "============================================================"
    nvidia-smi 2>/dev/null || true
    nvidia-smi topo -m 2>/dev/null || true

    echo
    echo "============================================================"
    echo "Storage"
    echo "============================================================"
    df -h "$CHECKPOINT_ROOT" 2>/dev/null || true

    echo
    echo "============================================================"
    echo "Command"
    echo "============================================================"
    printf '%q ' "${CMD[@]}"
    printf '\n'
  } > "$manifest"

  echo "Run manifest: $manifest"
}

# ----------------------------
# 12. Configuration summary
# ----------------------------
cat <<EOF

============================================================
GPT-W2560 pretraining configuration
============================================================

Model:
  Name: GPT-W2560
  Approximate parameters: ${MODEL_APPROX_PARAMS}
  Transformer body parameters: ${MODEL_TRANSFORMER_BODY_PARAMS}
  Layers: ${MODEL_NUM_LAYERS}
  Hidden size: ${MODEL_HIDDEN_SIZE}
  FFN hidden size: ${MODEL_FFN_HIDDEN_SIZE}
  Attention heads: ${MODEL_NUM_HEADS}
  Query groups: ${MODEL_NUM_QUERY_GROUPS}
  Head dimension: $((MODEL_HIDDEN_SIZE / MODEL_NUM_HEADS))
  GQA ratio: $((MODEL_NUM_HEADS / MODEL_NUM_QUERY_GROUPS)):1

Dataset mixture:
  C4: 35%
  FineWeb-Edu: 30%
  SlimPajama: 20%
  OpenWebText2: 10%
  Wikipedia: 1.67%
  StackExchange: 1.67%
  ArXiv: 1.66%

Training:
  Detected GPUs: ${detected_gpus}
  Requested GPUs: ${GPUS_PER_NODE}
  Parallelism: TP=${TP_SIZE}, PP=${PP_SIZE}, DP=${DP_SIZE}, CP=${CP_SIZE}
  Micro/global batch: ${MICRO_BATCH_SIZE}/${GLOBAL_BATCH_SIZE}
  Gradient accumulation steps: ${GRAD_ACCUM_STEPS}
  Sequence length: ${SEQ_LENGTH}
  Tokens per iteration: ${TOKENS_PER_ITER}
  Train iterations: ${TRAIN_ITERS}
  Target tokens: $((TOKENS_PER_ITER * TRAIN_ITERS))
  Warmup iterations: ${LR_WARMUP_ITERS}
  Maximum LR: ${MAX_LR}
  Minimum LR: ${MIN_LR}
  Recompute granularity: ${RECOMPUTE_GRANULARITY}
  Manual GC: ${ENABLE_MANUAL_GC}
  Empty-unused-memory level: ${EMPTY_UNUSED_MEMORY_LEVEL}
  Attention backend: ${ATTENTION_BACKEND}

Evaluation and checkpoints:
  Evaluation interval: ${EVAL_INTERVAL}
  Evaluation iterations: ${EVAL_ITERS}
  Persistent policy: only 1B / 3B / 5B, model weights only
  Megatron validation save interval: ${PERSISTENT_SAVE_INTERVAL}; runtime policy writes milestones only
  Rolling rollback save interval: ${ROLLBACK_SAVE_INTERVAL}; keep exactly latest one
  Drop old rollback before writing next: ${DROP_OLD_ROLLBACK_BEFORE_SAVE}
  Auto-resume rollback: ${AUTO_RESUME_ROLLBACK}
  Persistent token milestones only: ${KEEP_TOKEN_MILESTONES}
  Persistent checkpoint iterations only: ${KEEP_ITERATIONS[*]}

Monitoring:
  W&B project: ${WANDB_PROJECT}
  W&B run: ${WANDB_EXP_NAME}
  W&B run ID: ${WANDB_RUN_ID:-will_be_created_at_launch}
  TensorBoard: ${TENSORBOARD_DIR}
  System monitor enabled: ${ENABLE_SYSTEM_MONITOR}
  System monitor interval: ${SYSTEM_MONITOR_INTERVAL}s
  Run manifest enabled: ${ENABLE_RUN_MANIFEST}

Paths:
  Megatron: ${MEGATRON_DIR}
  Dataset root: ${DATASET_ROOT}
  Data prefixes: ${DATA_DIR}
  Tokenizer: ${TOKENIZER_DIR}
  Checkpoints: ${CHECKPOINT_DIR}
  Rolling checkpoint (latest full state only): ${ROLLBACK_CURRENT_DIR}
  Resume load directory: ${LOAD_DIR:-fresh_start}
  Logs: ${LOG_DIR}
  Gradient/data anomalies: ${GRAD_CB_OUTPUT_DIR}
  W&B local directory: ${WANDB_DIR}
  Checkpoint free space: ${checkpoint_free_gb}G
  Gradient callback installed: ${GRADIENT_CALLBACK_INSTALLED}

Offline theory metrics:
  d_parallel, d_2, d_align, train-test cosine, and sign-flip
  are intentionally NOT calculated in the training loop.
  Calculate them offline from the 1B / 3B / 5B checkpoints.

============================================================

EOF

printf 'Command:\n'
printf '%q ' "${CMD[@]}"
printf '\n\n'

# ----------------------------
# 13. Dry-run guard
# ----------------------------
if [[ "$RUN_TRAIN" != "1" ]]; then
  echo "Dry run only."
  echo "Launch with:"
  echo "  RUN_TRAIN=1 bash $0"
  exit 0
fi

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "Preflight checks passed; training launch skipped because PREFLIGHT_ONLY=1."
  exit 0
fi

# ----------------------------
# 14. Launch
# ----------------------------
cd "$MEGATRON_DIR"

log_file="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
echo "Launching training. Log: $log_file"

if [[ "$ENABLE_RUN_MANIFEST" == "1" ]]; then
  write_run_manifest
fi

# Remove checkpoints left by older policy revisions before launch.
prune_non_milestone_persistent_checkpoints
prune_rollback_to_latest_shell

"${CMD[@]}" > >(tee "$log_file") 2>&1 &
train_pid="$!"

monitor_pid=""
if [[ "$ENABLE_SYSTEM_MONITOR" == "1" ]]; then
  system_monitor_loop "$train_pid" &
  monitor_pid="$!"
fi

anomaly_watcher_pid=""
if [[ "$ENABLE_ANOMALY_WATCHER" == "1" ]]; then
  anomaly_watcher_loop "$train_pid" &
  anomaly_watcher_pid="$!"
fi

cleanup_background_jobs() {
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
  fi

  if [[ -n "$anomaly_watcher_pid" ]]; then
    kill "$anomaly_watcher_pid" >/dev/null 2>&1 || true
  fi
}

trap cleanup_background_jobs EXIT INT TERM

set +e
wait "$train_pid"
train_status="$?"
set -e

prune_non_milestone_persistent_checkpoints || true
prune_rollback_to_latest_shell || true

echo "Training finished with exit status: $train_status"
exit "$train_status"
