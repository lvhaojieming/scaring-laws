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
# Default behavior: dry run.
# Launch:
#   RUN_TRAIN=1 bash pretraining.sh
# ============================================================

# ----------------------------
# 0. User-configurable paths
# ----------------------------
CODE_DATA_ROOT="${CODE_DATA_ROOT:-/datadisk_1}"
MODEL_STORAGE_ROOT="${MODEL_STORAGE_ROOT:-/datadisk_2}"
PROJECT_DIR="${PROJECT_DIR:-$CODE_DATA_ROOT/projects}"
MEGATRON_DIR="${MEGATRON_DIR:-$PROJECT_DIR/Megatron-LM}"
DATASET_ROOT="${DATASET_ROOT:-$CODE_DATA_ROOT/balanced_web_edu_mix_5B}"
DATA_DIR="${DATA_DIR:-$DATASET_ROOT/megatron_llama3_by_source}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$DATASET_ROOT/tokenizers/llama3}"

RUN_NAME="${RUN_NAME:-gpt-w2560-l32-h2560-ffn8960-2p7265b-5btok}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$MODEL_STORAGE_ROOT/checkpoints}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$CHECKPOINT_ROOT/$RUN_NAME}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$MODEL_STORAGE_ROOT/tensorboard/$RUN_NAME}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-$MODEL_STORAGE_ROOT/data_cache/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$MODEL_STORAGE_ROOT/logs/$RUN_NAME}"
WANDB_DIR="${WANDB_DIR:-$MODEL_STORAGE_ROOT/wandb/$RUN_NAME}"
GRAD_ANOMALY_DIR="${GRAD_ANOMALY_DIR:-$MODEL_STORAGE_ROOT/gradient_anomalies/$RUN_NAME}"
ROLLBACK_KEEP_DIR="${ROLLBACK_KEEP_DIR:-$CHECKPOINT_DIR/rollback_keep}"

mkdir -p \
  "$CHECKPOINT_DIR" \
  "$TENSORBOARD_DIR" \
  "$DATA_CACHE_DIR" \
  "$LOG_DIR" \
  "$WANDB_DIR" \
  "$GRAD_ANOMALY_DIR" \
  "$ROLLBACK_KEEP_DIR"

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
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-2}"

# ----------------------------
# 3. Training configuration
# ----------------------------
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"

# 256 * 2048 = 524,288 tokens/iteration
# 5B / 524,288 ≈ 9536.74
TRAIN_ITERS="${TRAIN_ITERS:-9537}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-9537}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-2000}"

MAX_LR="${MAX_LR:-3.0e-4}"
MIN_LR="${MIN_LR:-3.0e-5}"

SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
EVAL_ITERS="${EVAL_ITERS:-20}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-selective}"

# Persist checkpoints closest to 1B, 3B, and 5B tokens.
KEEP_TOKEN_MILESTONES="${KEEP_TOKEN_MILESTONES:-1000000000 3000000000 5000000000}"
CHECKPOINT_PRUNE_INTERVAL_SECONDS="${CHECKPOINT_PRUNE_INTERVAL_SECONDS:-60}"
CHECKPOINT_PRUNE_GRACE_SECONDS="${CHECKPOINT_PRUNE_GRACE_SECONDS:-1800}"

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
WANDB_NAME="${WANDB_NAME:-$RUN_NAME}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-$WANDB_NAME}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

RUN_TRAIN="${RUN_TRAIN:-0}"
ALLOW_GPU_MISMATCH="${ALLOW_GPU_MISMATCH:-0}"
ALLOW_LOW_CHECKPOINT_SPACE="${ALLOW_LOW_CHECKPOINT_SPACE:-0}"
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
export GRAD_CB_EMERGENCY_CKPT="${GRAD_CB_EMERGENCY_CKPT:-1}"
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
  35    "${C4_PREFIX:-$DATA_DIR/c4_text_document}"
  30    "${FINEWEB_EDU_PREFIX:-$DATA_DIR/fineweb_edu_text_document}"
  20    "${SLIMPAJAMA_PREFIX:-$DATA_DIR/slimpajama_text_document}"
  10    "${OPENWEBTEXT2_PREFIX:-$DATA_DIR/openwebtext2_text_document}"
  1.67  "${WIKIPEDIA_PREFIX:-$DATA_DIR/wikipedia_text_document}"
  1.67  "${STACKEXCHANGE_PREFIX:-$DATA_DIR/stackexchange_text_document}"
  1.66  "${ARXIV_PREFIX:-$DATA_DIR/arxiv_text_document}"
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

if [[ "$RUN_TRAIN" == "1" && "$ALLOW_LOW_CHECKPOINT_SPACE" != "1" \
      && "$checkpoint_free_gb" -lt 250 ]]; then
  fail "Checkpoint disk has only ${checkpoint_free_gb}G free: $CHECKPOINT_ROOT"
fi

if [[ "$WORLD_SIZE" -ne $((TP_SIZE * PP_SIZE * DP_SIZE)) ]]; then
  fail "WORLD_SIZE=${WORLD_SIZE}, but TP*PP*DP=$((TP_SIZE * PP_SIZE * DP_SIZE))."
fi

if [[ $((GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DP_SIZE))) -ne 0 ]]; then
  fail "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by MICRO_BATCH_SIZE*DP_SIZE=$((MICRO_BATCH_SIZE * DP_SIZE))."
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

if [[ "$RUN_TRAIN" == "1" ]]; then
  if ! wandb login --verify >/dev/null 2>&1 && [[ -z "${WANDB_API_KEY:-}" ]]; then
    fail "W&B is not logged in. Run: wandb login"
  fi
fi

# ----------------------------
# 8. Checkpoint milestone logic
# ----------------------------
TOKENS_PER_ITER=$((GLOBAL_BATCH_SIZE * SEQ_LENGTH))
KEEP_ITERATIONS=()

for milestone in $KEEP_TOKEN_MILESTONES; do
  iter=$(((milestone + TOKENS_PER_ITER / 2) / TOKENS_PER_ITER))
  save_iter=$((((iter + SAVE_INTERVAL / 2) / SAVE_INTERVAL) * SAVE_INTERVAL))

  if [[ "$save_iter" -lt "$SAVE_INTERVAL" ]]; then
    save_iter="$SAVE_INTERVAL"
  fi

  if [[ "$save_iter" -le "$TRAIN_ITERS" ]]; then
    KEEP_ITERATIONS+=("$save_iter")
  fi
done

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

prune_checkpoints_once() {
  local latest_file="$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
  local latest_iter=""
  local dir base iter now mtime age

  if [[ -f "$latest_file" ]]; then
    latest_iter="$(tr -dc '0-9' < "$latest_file" || true)"
    latest_iter="${latest_iter#0}"
    latest_iter="${latest_iter:-0}"
  fi

  now="$(date +%s)"
  shopt -s nullglob

  for dir in "$CHECKPOINT_DIR"/iter_*; do
    [[ -d "$dir" ]] || continue

    base="$(basename "$dir")"
    iter="${base#iter_}"
    iter="${iter#0}"
    iter="${iter:-0}"

    if [[ "$iter" == "$latest_iter" ]] || is_keep_iteration "$iter"; then
      continue
    fi

    mtime="$(stat -c %Y "$dir")"
    age=$((now - mtime))

    if [[ "$age" -lt "$CHECKPOINT_PRUNE_GRACE_SECONDS" ]]; then
      continue
    fi

    echo "Pruning non-persistent checkpoint: $dir"
    rm -rf -- "$dir"
  done

  shopt -u nullglob
}

checkpoint_pruner_loop() {
  while true; do
    prune_checkpoints_once || true
    sleep "$CHECKPOINT_PRUNE_INTERVAL_SECONDS"
  done
}

preserve_latest_checkpoint() {
  local reason="${1:-unknown}"
  local latest_file="$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
  local latest_iter=""
  local source_dir target_dir metadata_file

  mkdir -p "$ROLLBACK_KEEP_DIR"

  if [[ ! -f "$latest_file" ]]; then
    echo "No latest checkpoint marker found; rollback preserve skipped for reason=$reason"
    return 0
  fi

  latest_iter="$(tr -dc '0-9' < "$latest_file" || true)"
  latest_iter="${latest_iter#0}"
  latest_iter="${latest_iter:-0}"

  if [[ "$latest_iter" == "0" ]]; then
    echo "Latest checkpoint marker is empty; rollback preserve skipped for reason=$reason"
    return 0
  fi

  source_dir="$CHECKPOINT_DIR/iter_$(printf '%07d' "$latest_iter")"
  if [[ ! -d "$source_dir" ]]; then
    source_dir="$CHECKPOINT_DIR/iter_$(printf '%06d' "$latest_iter")"
  fi
  if [[ ! -d "$source_dir" ]]; then
    source_dir="$CHECKPOINT_DIR/iter_$latest_iter"
  fi

  if [[ ! -d "$source_dir" ]]; then
    echo "Checkpoint directory for iter $latest_iter not found; rollback preserve skipped for reason=$reason"
    return 0
  fi

  target_dir="$ROLLBACK_KEEP_DIR/${reason}_iter_${latest_iter}_$(date +%Y%m%d_%H%M%S)"
  metadata_file="$target_dir/rollback_metadata.txt"
  mkdir -p "$target_dir"

  # Prefer hard links to avoid duplicating multi-GB checkpoint shards. Fall back
  # to a normal copy if the filesystem does not allow hard links.
  if ! cp -al "$source_dir/." "$target_dir/" 2>/dev/null; then
    cp -a "$source_dir/." "$target_dir/"
  fi

  cp -a "$latest_file" "$target_dir/latest_checkpointed_iteration.txt" 2>/dev/null || true

  {
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "reason=$reason"
    echo "source_dir=$source_dir"
    echo "target_dir=$target_dir"
    echo "latest_iteration=$latest_iter"
    echo "run_name=$RUN_NAME"
    echo "checkpoint_dir=$CHECKPOINT_DIR"
    echo "gradient_anomaly_dir=$GRAD_CB_OUTPUT_DIR"
  } > "$metadata_file"

  echo "Preserved rollback checkpoint: $target_dir"
}

anomaly_watcher_loop() {
  local train_pid="$1"
  local seen_file="$LOG_DIR/anomaly_watcher_seen.txt"
  local report report_key

  touch "$seen_file"

  while kill -0 "$train_pid" >/dev/null 2>&1; do
    shopt -s nullglob
    for report in "$GRAD_CB_OUTPUT_DIR"/*.json; do
      report_key="$(basename "$report")"
      if grep -Fxq "$report_key" "$seen_file"; then
        continue
      fi

      echo "$report_key" >> "$seen_file"
      echo "Gradient/data anomaly report detected: $report"
      preserve_latest_checkpoint "anomaly" || true
    done
    shopt -u nullglob

    sleep "$ANOMALY_WATCH_INTERVAL"
  done
}

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

  --cross-entropy-loss-fusion
  --cross-entropy-fusion-impl native
  --calculate-per-token-loss

  --manual-gc
  --empty-unused-memory-level 1
)

if [[ "$RECOMPUTE_GRANULARITY" != "none" ]]; then
  TRAINING_ARGS+=(--recompute-granularity "$RECOMPUTE_GRANULARITY")
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

LOGGING_ARGS=(
  --log-interval "$LOG_INTERVAL"

  --eval-interval "$EVAL_INTERVAL"
  --eval-iters "$EVAL_ITERS"

  --save-interval "$SAVE_INTERVAL"
  --save "$CHECKPOINT_DIR"
  --load "$CHECKPOINT_DIR"

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

if [[ -n "$WANDB_ENTITY" ]]; then
  LOGGING_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi

CMD=(
  torchrun "${DISTRIBUTED_ARGS[@]}"
  "$MEGATRON_DIR/pretrain_gpt.py"
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
    echo "code_data_root=$CODE_DATA_ROOT"
    echo "model_storage_root=$MODEL_STORAGE_ROOT"
    echo "project_dir=$PROJECT_DIR"
    echo "megatron_dir=$MEGATRON_DIR"
    echo "dataset_root=$DATASET_ROOT"
    echo "data_dir=$DATA_DIR"
    echo "tokenizer_dir=$TOKENIZER_DIR"
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
    echo "save_interval=$SAVE_INTERVAL"
    echo "eval_interval=$EVAL_INTERVAL"
    echo "eval_iters=$EVAL_ITERS"
    echo "persistent_token_milestones=$KEEP_TOKEN_MILESTONES"
    echo "persistent_checkpoint_iterations=${KEEP_ITERATIONS[*]}"
    echo "gradient_anomaly_callback_enabled=$ENABLE_GRADIENT_ANOMALY_CALLBACK"
    echo "gradient_anomaly_callback_installed=$GRADIENT_CALLBACK_INSTALLED"
    echo "gradient_anomaly_output_dir=$GRAD_CB_OUTPUT_DIR"
    echo "rollback_keep_dir=$ROLLBACK_KEEP_DIR"
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
  Parallelism: TP=${TP_SIZE}, PP=${PP_SIZE}, DP=${DP_SIZE}
  Micro/global batch: ${MICRO_BATCH_SIZE}/${GLOBAL_BATCH_SIZE}
  Sequence length: ${SEQ_LENGTH}
  Tokens per iteration: ${TOKENS_PER_ITER}
  Train iterations: ${TRAIN_ITERS}
  Target tokens: $((TOKENS_PER_ITER * TRAIN_ITERS))
  Warmup iterations: ${LR_WARMUP_ITERS}
  Maximum LR: ${MAX_LR}
  Minimum LR: ${MIN_LR}
  Recompute granularity: ${RECOMPUTE_GRANULARITY}
  Attention backend: ${ATTENTION_BACKEND}

Evaluation and checkpoints:
  Evaluation interval: ${EVAL_INTERVAL}
  Evaluation iterations: ${EVAL_ITERS}
  Save interval: ${SAVE_INTERVAL}
  Persistent token milestones: ${KEEP_TOKEN_MILESTONES}
  Persistent checkpoint iterations: ${KEEP_ITERATIONS[*]}

Monitoring:
  W&B project: ${WANDB_PROJECT}
  W&B run: ${WANDB_EXP_NAME}
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
  Rollback keep: ${ROLLBACK_KEEP_DIR}
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

# ----------------------------
# 14. Launch
# ----------------------------
cd "$MEGATRON_DIR"

log_file="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
echo "Launching training. Log: $log_file"

if [[ "$ENABLE_RUN_MANIFEST" == "1" ]]; then
  write_run_manifest
fi

checkpoint_pruner_loop &
pruner_pid="$!"

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
  kill "$pruner_pid" >/dev/null 2>&1 || true

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

prune_checkpoints_once || true

if [[ "$train_status" -ne 0 ]]; then
  preserve_latest_checkpoint "train_exit_${train_status}" || true
fi

echo "Training finished with exit status: $train_status"
exit "$train_status"

