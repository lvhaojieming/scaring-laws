#!/usr/bin/env bash
set -euo pipefail

MEGATRON_DIR="/datadisk_1/projects/Megatron-LM"
DATA_DIR="/datadisk_1/balanced_web_edu_mix_5B"
INPUT_DIR="${DATA_DIR}/cleaned_v1/jsonl"
TOKENIZER_DIR="${DATA_DIR}/tokenizers/llama3"
OUTPUT_DIR="${DATA_DIR}/megatron_llama3_by_source"
WORKERS="${WORKERS:-12}"
LOG_INTERVAL="${LOG_INTERVAL:-50000}"

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${MEGATRON_DIR}:${PYTHONPATH:-}"

run_one() {
  local source_name="$1"
  local input_file="$2"
  local output_prefix="${OUTPUT_DIR}/${source_name}"

  echo "==== ${source_name} ===="
  echo "input: ${input_file}"
  echo "output-prefix: ${output_prefix}"

  python3 "${MEGATRON_DIR}/tools/preprocess_data.py" \
    --input "${input_file}" \
    --json-keys text \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "${TOKENIZER_DIR}" \
    --append-eod \
    --output-prefix "${output_prefix}" \
    --workers "${WORKERS}" \
    --log-interval "${LOG_INTERVAL}"
}

run_one arxiv        "${INPUT_DIR}/cleaned_00000.jsonl"
run_one c4           "${INPUT_DIR}/cleaned_00001.jsonl"
run_one fineweb_edu  "${INPUT_DIR}/cleaned_00002.jsonl"
run_one openwebtext2 "${INPUT_DIR}/cleaned_00003.jsonl"
run_one slimpajama   "${INPUT_DIR}/cleaned_00004.jsonl"
run_one stackexchange "${INPUT_DIR}/cleaned_00005.jsonl"
run_one wikipedia    "${INPUT_DIR}/cleaned_00006.jsonl"
