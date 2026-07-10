#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

image="${SPARK_VLLM_IMAGE:-eugr/spark-vllm:latest}"
container_name="${SPARK_VLLM_CONTAINER:-infer-vllm-bench}"
host_port="${VLLM_PORT:-8000}"
model_dir="${EIDER_MODEL_DIR:-$repo_dir/models/qwen3-30b-a3b-nvfp4}"
container_model_dir="${VLLM_CONTAINER_MODEL_DIR:-/models/qwen3-30b-a3b-nvfp4}"
vllm_model="${VLLM_MODEL:-$container_model_dir}"
decode_tokens="${DECODE_TOKENS:-200}"
spark_warmup_repeats="${SPARK_WARMUP_REPEATS:-1}"
repeats="${REPEATS:-3}"
prompt="${PROMPT:-Hello world, this is a benchmark.}"
gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.65}"
max_model_len="${VLLM_MAX_MODEL_LEN:-4096}"
load_format="${VLLM_LOAD_FORMAT:-}"
extra_vllm_args="${EXTRA_VLLM_ARGS:-}"
keep_container="${KEEP_VLLM_CONTAINER:-0}"

if [[ ! -d "$model_dir" ]]; then
  echo "model directory does not exist: $model_dir" >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "$container_name"; then
  echo "removing existing container: $container_name" >&2
  docker rm -f "$container_name" >/dev/null
fi

cleanup() {
  if [[ "$keep_container" != "1" ]]; then
    docker rm -f "$container_name" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "pulling $image" >&2
docker pull "$image"

load_format_arg=()
if [[ -n "$load_format" ]]; then
  load_format_arg=(--load-format "$load_format")
fi

echo "starting $container_name from $image" >&2
docker run -d \
  --name "$container_name" \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  -p "$host_port:8000" \
  -v "$model_dir:$container_model_dir:ro" \
  --entrypoint bash \
  "$image" \
  -lc "vllm serve '$vllm_model' \
    --port 8000 --host 0.0.0.0 \
    --gpu-memory-utilization '$gpu_memory_utilization' \
    --max-model-len '$max_model_len' \
    ${load_format_arg[*]} \
    $extra_vllm_args"

echo "waiting for vLLM on http://127.0.0.1:$host_port/v1/models" >&2
"$repo_dir/scripts/wait-openai-compatible.py" \
  --url "http://127.0.0.1:$host_port/v1/models" \
  --container "$container_name"

VLLM_URL="http://127.0.0.1:$host_port/v1/completions" \
VLLM_MODEL="$vllm_model" \
EIDER_MODEL_DIR="$model_dir" \
DECODE_TOKENS="$decode_tokens" \
REPEATS="$repeats" \
PROMPT="$prompt" \
SPARK_WARMUP_REPEATS="$spark_warmup_repeats" \
  "$repo_dir/scripts/compare-vllm.sh"

if [[ "$keep_container" == "1" ]]; then
  echo "leaving container running: $container_name" >&2
fi
