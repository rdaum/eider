#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
hf_home="${HF_HOME:-${XDG_CACHE_HOME:-${HOME:?HOME must be set}/.cache}/huggingface}"
hf_cache="${HF_HUB_CACHE:-$hf_home/hub}"
revision="c6cc2e8ffa353403213d6b9b18aaf8eb684d705a"
model_dir="${NEMOTRON35_BOLT_MODEL:-$hf_cache/models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB/snapshots/$revision}"
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
add_default --served-model-name "${EIDER_SERVED_MODEL:-eider-nemotron3.5-bolt}"
add_default --max-context-tokens "${EIDER_MAX_CONTEXT_TOKENS:-262144}"
add_default --max-active-sequences 1
add_default --decode-capacity 1
add_default --prefill-sequence-capacity 1
add_default --prefill-token-capacity "${EIDER_PREFILL_TOKEN_CAPACITY:-2048}"
add_default --retained-prefix-gib 1
# Bolt has no MTP block; generation uses its released LiLiCorr companion.
speculative_drafts=0
for arg in "${server_args[@]}"; do
  if [[ "$arg" == --lilicorr-dir || "$arg" == --lilicorr-dir=* ]]; then
    speculative_drafts=3
    break
  fi
done
if [[ "${EIDER_LILICORR:-1}" == 1 ]]; then
  lilicorr_revision="4c6a5613b6d887194a3a5493e3a18a5b3930bb86"
  lilicorr_dir="${LILICORR_DIR:-$hf_cache/models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr/snapshots/$lilicorr_revision}"
  add_default --lilicorr-dir "$lilicorr_dir"
  speculative_drafts=3
fi
add_default --speculative-drafts "${EIDER_SPECULATIVE_DRAFTS:-$speculative_drafts}"
add_default --nemotron-bf16-storage bf16
add_default --nemotron-fp8-storage fp8

exec cargo run --release \
  --manifest-path "$repo_dir/Cargo.toml" \
  -p eider-api \
  --bin eider-serve \
  -- "${server_args[@]}"
