#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

model_dir="${EIDER_MODEL_DIR:-$repo_dir/models/qwen3-30b-a3b-nvfp4}"
vllm_model="${VLLM_MODEL:-$model_dir}"
vllm_url="${VLLM_URL:-http://127.0.0.1:8000/v1/completions}"
decode_tokens="${DECODE_TOKENS:-200}"
spark_warmup_repeats="${SPARK_WARMUP_REPEATS:-1}"
repeats="${REPEATS:-3}"
prompt="${PROMPT:-Hello world, this is a benchmark.}"

cat >&2 <<EOF
eider vs vLLM-compatible benchmark
  spark model dir: $model_dir
  vLLM model: $vllm_model
  vLLM URL: $vllm_url
  decode tokens: $decode_tokens
  spark warmup repeats: $spark_warmup_repeats
  repeats: $repeats

To run the eugr/spark-vllm Docker image and benchmark it in one step:
  scripts/bench-eider-vllm-docker.sh

If you already have a vLLM OpenAI-compatible server running, this script only
benchmarks against VLLM_URL.

Override with EIDER_MODEL_DIR, VLLM_MODEL, VLLM_URL, PROMPT,
DECODE_TOKENS, SPARK_WARMUP_REPEATS, and REPEATS.
EOF

echo "eider_begin"
cargo run --release -p infer --bin qwen-bench -- \
  --model "$model_dir" \
  --prompt "$prompt" \
  --decode-tokens "$decode_tokens" \
  --warmup-repeats "$spark_warmup_repeats" \
  --repeats "$repeats" \
  --temperature 0
echo "eider_end"

echo "vllm_begin"
"$repo_dir/scripts/bench-vllm-compat.py" \
  --url "$vllm_url" \
  --model "$vllm_model" \
  --prompt "$prompt" \
  --decode-tokens "$decode_tokens" \
  --repeats "$repeats" \
  --temperature 0 \
  --warmup
echo "vllm_end"
