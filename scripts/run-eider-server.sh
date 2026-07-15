#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

model_dir="${EIDER_MODEL_DIR:-$repo_dir/models/qwen3.6-35b-a3-nvfp4}"
listen="${EIDER_LISTEN:-127.0.0.1:8080}"
served_model="${EIDER_SERVED_MODEL:-eider-qwen3.6}"
export EIDER_API_KEY="${EIDER_API_KEY:-local-eider}"

if [[ ! -d "$model_dir" ]]; then
  echo "model directory does not exist: $model_dir" >&2
  exit 1
fi

echo "building eider-serve" >&2
cargo build --release \
  --manifest-path "$repo_dir/Cargo.toml" \
  -p eider-api \
  --bin eider-serve

echo "launching $served_model; the API will become available after model loading" >&2
exec "$repo_dir/target/release/eider-serve" \
  "$model_dir" \
  --listen "$listen" \
  --served-model-name "$served_model" \
  "$@"
