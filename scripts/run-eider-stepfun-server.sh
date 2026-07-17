#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

model_dir="${EIDER_MODEL_DIR:-$repo_dir/models/step-3.7-flash-nvfp4}"
listen="${EIDER_LISTEN:-127.0.0.1:8080}"
served_model="${EIDER_SERVED_MODEL:-eider-step3.7}"
expert_capacity="${EIDER_STEP_EXPERT_CAPACITY:-240}"
bf16_attention="${EIDER_STEP_BF16_ATTENTION:-nvfp4}"
bf16_dense_mlp="${EIDER_STEP_BF16_DENSE_MLP:-nvfp4}"
bf16_shared_expert="${EIDER_STEP_BF16_SHARED_EXPERT:-nvfp4}"
bf16_lm_head="${EIDER_STEP_BF16_LM_HEAD:-nvfp4}"
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

echo "Building Step-3.7 expert preparer..." >&2
cargo build --release \
  --manifest-path "$repo_dir/Cargo.toml" \
  -p infer \
  --bin step37-experts

echo "Preparing or validating the Step-3.7 expert cache..." >&2
"$repo_dir/target/release/step37-experts" prepare "$model_dir"

echo "Building eider-serve..." >&2
cargo build --release \
  --manifest-path "$repo_dir/Cargo.toml" \
  -p eider-api \
  --bin eider-serve

exec "$repo_dir/target/release/eider-serve" \
  "$model_dir" \
  --listen "$listen" \
  --served-model-name "$served_model" \
  --step-expert-capacity "$expert_capacity" \
  --step-bf16-attention "$bf16_attention" \
  --step-bf16-dense-mlp "$bf16_dense_mlp" \
  --step-bf16-shared-expert "$bf16_shared_expert" \
  --step-bf16-lm-head "$bf16_lm_head" \
  "${dogstatsd_args[@]}" \
  "$@"
