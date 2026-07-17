#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

model_dir="${EIDER_MODEL_DIR:-$repo_dir/models/agents-a1-nvfp4}"
listen="${EIDER_LISTEN:-127.0.0.1:8080}"
served_model="${EIDER_SERVED_MODEL:-eider-agents-a1}"
max_context_tokens="${EIDER_MAX_CONTEXT_TOKENS:-262144}"
bf16_attention="${EIDER_QWEN_BF16_ATTENTION:-nvfp4}"
bf16_lm_head="${EIDER_QWEN_BF16_LM_HEAD:-nvfp4}"
export EIDER_API_KEY="${EIDER_API_KEY:-local-eider}"

dogstatsd_args=()
if [[ -n "${EIDER_DOGSTATSD_ENDPOINT:-}" ]]; then
  dogstatsd_args+=(--dogstatsd-endpoint "$EIDER_DOGSTATSD_ENDPOINT")
  interval="${EIDER_DOGSTATSD_INTERVAL_SECS:-1}"
  dogstatsd_args+=(--dogstatsd-interval-secs "$interval")
fi

if [[ ! -d "$model_dir" ]]; then
  echo "model directory does not exist: $model_dir" >&2
  exit 1
fi

echo "Building eider-serve..." >&2
cargo build --release \
  --manifest-path "$repo_dir/Cargo.toml" \
  -p eider-api \
  --bin eider-serve

exec "$repo_dir/target/release/eider-serve" \
  "$model_dir" \
  --listen "$listen" \
  --served-model-name "$served_model" \
  --max-context-tokens "$max_context_tokens" \
  --qwen-bf16-attention "$bf16_attention" \
  --qwen-bf16-lm-head "$bf16_lm_head" \
  "${dogstatsd_args[@]}" \
  "$@"
