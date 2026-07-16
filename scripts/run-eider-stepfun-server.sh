#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

model_dir="${EIDER_MODEL_DIR:-$repo_dir/models/step-3.7-flash-nvfp4}"
listen="${EIDER_LISTEN:-127.0.0.1:8080}"
served_model="${EIDER_SERVED_MODEL:-eider-step3.7}"
expert_capacity="${EIDER_STEP_EXPERT_CAPACITY:-240}"
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
  "${dogstatsd_args[@]}" \
  "$@"
