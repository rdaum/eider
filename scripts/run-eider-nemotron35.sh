#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
hf_home="${HF_HOME:-${XDG_CACHE_HOME:-${HOME:?HOME must be set}/.cache}/huggingface}"
hf_cache="${HF_HUB_CACHE:-$hf_home/hub}"
revision="7d4bc747b6bdc2cff64c58d65da0d08dd9caa25c"
model_dir="${NEMOTRON35_MODEL:-$hf_cache/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/$revision}"
server_args=("$@")

# Explicit server arguments take precedence over launcher defaults.
add_default() {
  local option="$1" value="$2" arg
  for arg in "${server_args[@]}"; do
    if [[ "$arg" == "$option" || "$arg" == "$option="* ]]; then
      return
    fi
  done
  server_args+=("$option" "$value")
}

add_default --model-dir "$model_dir"
add_default --served-model-name "${EIDER_SERVED_MODEL:-eider-nemotron3.5-super}"
add_default --max-context-tokens "${EIDER_MAX_CONTEXT_TOKENS:-1048576}"
add_default --max-active-sequences 1
add_default --decode-capacity 1
add_default --prefill-sequence-capacity 1
add_default --prefill-token-capacity "${EIDER_PREFILL_TOKEN_CAPACITY:-128}"
add_default --retained-prefix-gib 1
add_default --speculative-drafts "${EIDER_SPECULATIVE_DRAFTS:-0}"
add_default --nemotron-bf16-storage bf16
add_default --nemotron-fp8-storage fp8

exec cargo run --release \
  --manifest-path "$repo_dir/Cargo.toml" \
  -p eider-api \
  --bin eider-serve \
  -- "${server_args[@]}"
