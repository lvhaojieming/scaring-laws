#!/usr/bin/env bash
set -euo pipefail

# Dry-run by default. Set RUN_TRAIN=1 only after reviewing the printed config.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="${PROJECT_DIR:-/datadisk_1/megatron_pretrain_setup}"
MEGATRON_DIR="${MEGATRON_DIR:-/datadisk_1/projects/Megatron-LM}"
DATA_DIR="${DATA_DIR:-/datadisk_1/balanced_web_edu_mix_10B/megatron_data}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${PROJECT_DIR}/tokenizers/Meta-Llama-3.1-8B}"

RUN_NAME="${RUN_NAME:-llama3_1p745b_balanced_web_edu_mix_5B}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/datadisk_1/megatron_pretrain_runs/checkpoints}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${CHECKPOINT_ROOT}/${RUN_NAME}}"
RUN_ROOT="${RUN_ROOT:-/datadisk_1/megatron_pretrain_runs/${RUN_NAME}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${RUN_ROOT}/tensorboard}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${RUN_ROOT}/data_cache}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
WANDB_DIR="${WANDB_DIR:-/datadisk_1/wandb}"

mkdir -p "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}" "${DATA_CACHE_DIR}" "${LOG_DIR}" "${WANDB_DIR}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHON="${PYTHON:-/usr/bin/python}"
export WANDB_DIR

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6000}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-2}"

MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
TRAIN_ITERS="${TRAIN_ITERS:-9537}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-9537}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-2000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
EVAL_ITERS="${EVAL_ITERS:-20}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
KEEP_TOKEN_MILESTONES="${KEEP_TOKEN_MILESTONES:-1000000000 3000000000 5000000000}"
CHECKPOINT_PRUNE_INTERVAL_SECONDS="${CHECKPOINT_PRUNE_INTERVAL_SECONDS:-60}"
CHECKPOINT_PRUNE_GRACE_SECONDS="${CHECKPOINT_PRUNE_GRACE_SECONDS:-1800}"

WANDB_PROJECT="${WANDB_PROJECT:-balanced_web_edu_mix_10B}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-${RUN_NAME}}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

RUN_TRAIN="${RUN_TRAIN:-0}"
ALLOW_GPU_MISMATCH="${ALLOW_GPU_MISMATCH:-0}"
ALLOW_LOW_CHECKPOINT_SPACE="${ALLOW_LOW_CHECKPOINT_SPACE:-0}"

data_path=(
  35 "${DATA_DIR}/balanced_web_edu_mix_10B_llama31_c4_text_document"
  30 "${DATA_DIR}/balanced_web_edu_mix_10B_llama31_fineweb_edu_text_document"
  20 "${DATA_DIR}/balanced_web_edu_mix_10B_llama31_slimpajama_text_document"
  10 "${DATA_DIR}/balanced_web_edu_mix_10B_llama31_openwebtext2_text_document"
  1.67 "${DATA_DIR}/balanced_web_edu_mix_10B_llama31_wikipedia_text_document"
  1.67 "${DATA_DIR}/balanced_web_edu_mix_10B_llama31_stackexchange_text_document"
  1.66 "${DATA_DIR}/balanced_web_edu_mix_10B_llama31_arxiv_text_document"
)

for ((i=1; i<${#data_path[@]}; i+=2)); do
  prefix="${data_path[$i]}"
  if [[ ! -f "${prefix}.bin" || ! -f "${prefix}.idx" ]]; then
    echo "Missing Megatron dataset files for prefix: ${prefix}" >&2
    exit 1
  fi
done

[[ -d "${TOKENIZER_DIR}" ]] || { echo "Missing tokenizer dir: ${TOKENIZER_DIR}" >&2; exit 1; }
[[ -f "${MEGATRON_DIR}/pretrain_gpt.py" ]] || { echo "Missing ${MEGATRON_DIR}/pretrain_gpt.py" >&2; exit 1; }

detected_gpus=0
if command -v nvidia-smi >/dev/null 2>&1; then
  detected_gpus="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
fi

checkpoint_free_gb="$(df -BG "${CHECKPOINT_ROOT}" | awk 'NR==2 {gsub(/G/,"",$4); print $4}')"

if [[ "${RUN_TRAIN}" == "1" && "${ALLOW_GPU_MISMATCH}" != "1" && "${detected_gpus}" != "${GPUS_PER_NODE}" ]]; then
  echo "Detected ${detected_gpus} GPU(s), but this config requests GPUS_PER_NODE=${GPUS_PER_NODE}, TP=${TP_SIZE}." >&2
  echo "Adjust GPUS_PER_NODE/parallelism or attach the requested GPUs before launch." >&2
  exit 1
fi

if [[ "${RUN_TRAIN}" == "1" && "${ALLOW_LOW_CHECKPOINT_SPACE}" != "1" && "${checkpoint_free_gb}" -lt 250 ]]; then
  echo "Checkpoint target has only ${checkpoint_free_gb}G free: ${CHECKPOINT_ROOT}" >&2
  echo "Training checkpoints with optimizer can be large. Move CHECKPOINT_ROOT to a larger disk or set ALLOW_LOW_CHECKPOINT_SPACE=1 explicitly." >&2
  exit 1
fi

if [[ "${WORLD_SIZE}" -ne $((TP_SIZE * PP_SIZE * DP_SIZE)) ]]; then
  echo "Invalid parallelism: WORLD_SIZE=${WORLD_SIZE}, but TP*PP*DP=$((TP_SIZE * PP_SIZE * DP_SIZE))." >&2
  exit 1
fi

TOKENS_PER_ITER=$((GLOBAL_BATCH_SIZE * SEQ_LENGTH))
KEEP_ITERATIONS=()
for milestone in ${KEEP_TOKEN_MILESTONES}; do
  iter=$(((milestone + TOKENS_PER_ITER / 2) / TOKENS_PER_ITER))
  save_iter=$((((iter + SAVE_INTERVAL / 2) / SAVE_INTERVAL) * SAVE_INTERVAL))
  if [[ "${save_iter}" -lt "${SAVE_INTERVAL}" ]]; then
    save_iter="${SAVE_INTERVAL}"
  fi
  if [[ "${save_iter}" -le "${TRAIN_ITERS}" ]]; then
    KEEP_ITERATIONS+=("${save_iter}")
  fi
done

is_keep_iteration() {
  local candidate="$1"
  local keep_iter
  for keep_iter in "${KEEP_ITERATIONS[@]}"; do
    if [[ "${candidate}" == "${keep_iter}" ]]; then
      return 0
    fi
  done
  return 1
}

prune_checkpoints_once() {
  local latest_file="${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
  local latest_iter=""
  local dir base iter now mtime age

  if [[ -f "${latest_file}" ]]; then
    latest_iter="$(tr -dc '0-9' < "${latest_file}" || true)"
    latest_iter="${latest_iter#0}"
    latest_iter="${latest_iter:-0}"
  fi

  now="$(date +%s)"
  shopt -s nullglob
  for dir in "${CHECKPOINT_DIR}"/iter_*; do
    [[ -d "${dir}" ]] || continue
    base="$(basename "${dir}")"
    iter="${base#iter_}"
    iter="${iter#0}"
    iter="${iter:-0}"

    if [[ "${iter}" == "${latest_iter}" ]] || is_keep_iteration "${iter}"; then
      continue
    fi

    mtime="$(stat -c %Y "${dir}")"
    age=$((now - mtime))
    if [[ "${age}" -lt "${CHECKPOINT_PRUNE_GRACE_SECONDS}" ]]; then
      continue
    fi

    echo "Pruning non-persistent checkpoint: ${dir}"
    rm -rf -- "${dir}"
  done
  shopt -u nullglob
}

checkpoint_pruner_loop() {
  while true; do
    prune_checkpoints_once || true
    sleep "${CHECKPOINT_PRUNE_INTERVAL_SECONDS}"
  done
}

if [[ "${RUN_TRAIN}" == "1" ]]; then
  if ! wandb login --verify >/dev/null 2>&1 && [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "W&B is not logged in. Run: wandb login" >&2
    echo "Or: WANDB_API_KEY=... ${PROJECT_DIR}/scripts/setup_wandb.sh" >&2
    exit 1
  fi
fi

DISTRIBUTED_ARGS=(
  --nproc_per_node "${GPUS_PER_NODE}"
  --nnodes "${NUM_NODES}"
  --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
)

MODEL_ARGS=(
  --use-mcore-models
  --num-layers 32
  --hidden-size 2048
  --ffn-hidden-size 7168
  --num-attention-heads 16
  --group-query-attention
  --num-query-groups 4
  --kv-channels 128
  --seq-length "${SEQ_LENGTH}"
  --max-position-embeddings "${SEQ_LENGTH}"
  --position-embedding-type rope
  --rotary-base 500000
  --rotary-percent 1.0
  --use-rope-scaling
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --swiglu
  --normalization RMSNorm
  --init-method-std 0.02
  --untie-embeddings-and-output-weights
  --disable-bias-linear
)

PARALLEL_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --pipeline-model-parallel-size "${PP_SIZE}"
  --use-distributed-optimizer
  --overlap-grad-reduce
  --overlap-param-gather
)

TRAINING_ARGS=(
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --train-iters "${TRAIN_ITERS}"
  --lr-decay-iters "${LR_DECAY_ITERS}"
  --lr-warmup-iters "${LR_WARMUP_ITERS}"
  --lr 3.0e-4
  --min-lr 3.0e-5
  --lr-decay-style cosine
  --weight-decay 0.1
  --clip-grad 1.0
  --adam-beta1 0.9
  --adam-beta2 0.95
  --adam-eps 1.0e-8
  --bf16
  --grad-reduce-in-bf16
  --cross-entropy-loss-fusion
  --calculate-per-token-loss
  --manual-gc
  --empty-unused-memory-level 1
)

DATA_ARGS=(
  --data-path "${data_path[@]}"
  --split 99,1,0
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model "${TOKENIZER_DIR}"
  --vocab-size 128256
  --data-cache-path "${DATA_CACHE_DIR}"
  --no-create-attention-mask-in-dataloader
  --num-workers 2
)

LOGGING_ARGS=(
  --log-interval "${LOG_INTERVAL}"
  --eval-interval "${EVAL_INTERVAL}"
  --eval-iters "${EVAL_ITERS}"
  --save-interval "${SAVE_INTERVAL}"
  --save "${CHECKPOINT_DIR}"
  --load "${CHECKPOINT_DIR}"
  --ckpt-format torch_dist
  --dist-ckpt-workers 4
  --tensorboard-dir "${TENSORBOARD_DIR}"
  --log-throughput
  --distributed-timeout-minutes 60
  --wandb-project "${WANDB_PROJECT}"
  --wandb-exp-name "${WANDB_EXP_NAME}"
  --wandb-save-dir "${WANDB_DIR}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  LOGGING_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

CMD=(
  torchrun "${DISTRIBUTED_ARGS[@]}"
  "${MEGATRON_DIR}/pretrain_gpt.py"
  "${MODEL_ARGS[@]}"
  "${PARALLEL_ARGS[@]}"
  "${TRAINING_ARGS[@]}"
  "${DATA_ARGS[@]}"
  "${LOGGING_ARGS[@]}"
)

cat <<EOF
Run name: ${RUN_NAME}
Detected GPUs: ${detected_gpus}
Requested GPUs: ${GPUS_PER_NODE}
Parallelism: TP=${TP_SIZE}, PP=${PP_SIZE}, DP=${DP_SIZE}
Micro/global batch: ${MICRO_BATCH_SIZE}/${GLOBAL_BATCH_SIZE}
Sequence length: ${SEQ_LENGTH}
Tokens per iteration: $((GLOBAL_BATCH_SIZE * SEQ_LENGTH))
Train iterations: ${TRAIN_ITERS}
Target tokens: $((GLOBAL_BATCH_SIZE * SEQ_LENGTH * TRAIN_ITERS))
Persistent token milestones: ${KEEP_TOKEN_MILESTONES}
Persistent checkpoint iterations: ${KEEP_ITERATIONS[*]}

Dataset weights:
  C4: 35%
  FineWeb-Edu: 30%
  SlimPajama: 20%
  OpenWebText2: 10%
  Wikipedia/StackExchange/arXiv: 5% total
    Wikipedia: 1.67%
    StackExchange: 1.67%
    arXiv: 1.66%

Train/valid/test split: 99/1/0, applied by Megatron to each indexed dataset.
Validation callback: every ${EVAL_INTERVAL} iterations, ${EVAL_ITERS} eval iterations each time.
Checkpoint callback: every ${SAVE_INTERVAL} iterations.
Checkpoint retention: keep latest checkpoint dynamically, plus persistent milestone checkpoints.
Training log callback: every ${LOG_INTERVAL} iterations.

Checkpoint dir: ${CHECKPOINT_DIR}
Checkpoint free space: ${checkpoint_free_gb}G
TensorBoard dir: ${TENSORBOARD_DIR}
W&B project: ${WANDB_PROJECT}
W&B run: ${WANDB_EXP_NAME}
W&B local dir: ${WANDB_DIR}

W&B web visibility has been validated with 'wandb login --verify' before launch.
d_align is not implemented in this Megatron training loop yet; validation loss
will be logged now, and d_align needs a separate validation hook/job.

EOF

printf 'Command:\n'
printf '%q ' "${CMD[@]}"
printf '\n'

if [[ "${RUN_TRAIN}" != "1" ]]; then
  echo
  echo "Dry run only. To launch after inspection:"
  echo "  RUN_TRAIN=1 /root/pretrain.sh"
  exit 0
fi

cd "${MEGATRON_DIR}"
log_file="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "Launching training. Log: ${log_file}"
checkpoint_pruner_loop &
pruner_pid="$!"
trap 'kill "${pruner_pid}" >/dev/null 2>&1 || true' EXIT

set +e
"${CMD[@]}" 2>&1 | tee "${log_file}"
train_status="${PIPESTATUS[0]}"
set -e
prune_checkpoints_once || true
exit "${train_status}"
