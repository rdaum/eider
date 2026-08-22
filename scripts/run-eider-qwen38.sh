#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model="${EIDER_MODEL:-qwen3.8-27b}"
server_args=("$@")

speculative_drafts_set=false
for arg in "${server_args[@]}"; do
  case "$arg" in
    --speculative-drafts | --speculative-drafts=*)
      speculative_drafts_set=true
      ;;
  esac
done

if [[ "$speculative_drafts_set" == false ]]; then
  server_args+=(--speculative-drafts 2)
fi

exec cargo run --release \
  --manifest-path "$repo_dir/Cargo.toml" \
  -p eider-api \
  --bin eider-serve \
  -- "$model" "${server_args[@]}"
