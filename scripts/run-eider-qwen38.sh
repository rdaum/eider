#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model="${EIDER_MODEL:-qwen3.8-27b}"
server_args=("$@")

attention_storage_set=false
lm_head_storage_set=false
speculative_drafts_set=false
for arg in "${server_args[@]}"; do
  case "$arg" in
    --qwen-bf16-attention | --qwen-bf16-attention=*)
      attention_storage_set=true
      ;;
    --qwen-bf16-lm-head | --qwen-bf16-lm-head=*)
      lm_head_storage_set=true
      ;;
    --speculative-drafts | --speculative-drafts=*)
      speculative_drafts_set=true
      ;;
  esac
done

if [[ "$attention_storage_set" == false ]]; then
  server_args+=(--qwen-bf16-attention bf16)
fi
if [[ "$lm_head_storage_set" == false ]]; then
  server_args+=(--qwen-bf16-lm-head bf16)
fi
if [[ "$speculative_drafts_set" == false ]]; then
  server_args+=(--speculative-drafts 2)
fi

exec cargo run --release \
  --manifest-path "$repo_dir/Cargo.toml" \
  -p eider-api \
  --bin eider-serve \
  -- "$model" "${server_args[@]}"
