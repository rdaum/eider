#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model="${EIDER_MODEL:-qwen3.8-27b}"
server_args=("$@")

speculative_drafts_set=false
max_context_tokens_set=false
max_active_sequences_set=false
decode_capacity_set=false
prefill_sequence_capacity_set=false
for arg in "${server_args[@]}"; do
  case "$arg" in
    --speculative-drafts | --speculative-drafts=*)
      speculative_drafts_set=true
      ;;
    --max-context-tokens | --max-context-tokens=*)
      max_context_tokens_set=true
      ;;
    --max-active-sequences | --max-active-sequences=*)
      max_active_sequences_set=true
      ;;
    --decode-capacity | --decode-capacity=*)
      decode_capacity_set=true
      ;;
    --prefill-sequence-capacity | --prefill-sequence-capacity=*)
      prefill_sequence_capacity_set=true
      ;;
  esac
done

if [[ "$speculative_drafts_set" == false ]]; then
  server_args+=(--speculative-drafts 2)
fi
if [[ "$max_context_tokens_set" == false ]]; then
  server_args+=(--max-context-tokens 262144)
fi
if [[ "$max_active_sequences_set" == false ]]; then
  server_args+=(--max-active-sequences 1)
fi
if [[ "$decode_capacity_set" == false ]]; then
  server_args+=(--decode-capacity 1)
fi
if [[ "$prefill_sequence_capacity_set" == false ]]; then
  server_args+=(--prefill-sequence-capacity 1)
fi

exec cargo run --release \
  --manifest-path "$repo_dir/Cargo.toml" \
  -p eider-api \
  --bin eider-serve \
  -- "$model" "${server_args[@]}"
