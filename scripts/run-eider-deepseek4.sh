#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cache_root="${XDG_CACHE_HOME:-$HOME/.cache}"
revision="${DEEPSEEK4_REVISION:-e3cd60e7de98e9867116860d522499a728de1cf9}"
model_root="${DEEPSEEK4_MODEL_ROOT:-$cache_root/eider/models/nvidia--DeepSeek-V4-Flash-NVFP4/$revision}"
model_dir="${DEEPSEEK4_THIN_DIR:-$model_root/deepseek4-thin-nvfp4-v1}"
artifact_dir="${DEEPSEEK4_ARTIFACT_DIR:-$model_root/deepseek4-experts-q2-v3}"
hot_cache_dir="${DEEPSEEK4_HOT_CACHE_DIR:-$model_root/deepseek4-hot-nvfp4-v1}"
served_model="${EIDER_SERVED_MODEL:-eider-deepseek-v4}"
max_context_tokens="${EIDER_MAX_CONTEXT_TOKENS:-32768}"
prefill_token_capacity="${EIDER_PREFILL_TOKEN_CAPACITY:-2048}"
hot_expert_capacity="${DEEPSEEK4_HOT_EXPERT_CAPACITY:-8}"
hotset_plan_output="${DEEPSEEK4_HOTSET_PLAN_OUTPUT:-}"

if [[ ! -f "$model_dir/model.safetensors.index.json" ]]; then
  echo "DeepSeek V4 thin checkpoint is not prepared: $model_dir" >&2
  echo "run scripts/prepare-deepseek4-experts-streaming.sh first" >&2
  exit 1
fi
if [[ ! -f "$artifact_dir/manifest.json" ]]; then
  echo "DeepSeek V4 expert artifacts are not prepared: $artifact_dir" >&2
  echo "run scripts/prepare-deepseek4-experts-streaming.sh first" >&2
  exit 1
fi

deepseek_args=(
  --deepseek-hot-expert-capacity "$hot_expert_capacity"
  --deepseek-hot-expert-cache-dir "$hot_cache_dir"
)
if [[ -n "$hotset_plan_output" ]]; then
  deepseek_args+=(--deepseek-hotset-plan-output "$hotset_plan_output")
fi

exec cargo run --release \
  --manifest-path "$repo_dir/Cargo.toml" \
  -p eider-api \
  --bin eider-serve \
  -- \
  --model-dir "$model_dir" \
  --artifact-dir "$artifact_dir" \
  --served-model-name "$served_model" \
  --max-context-tokens "$max_context_tokens" \
  --prefill-token-capacity "$prefill_token_capacity" \
  "${deepseek_args[@]}" \
  "$@"
