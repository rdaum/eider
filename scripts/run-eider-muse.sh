#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${MUSE_GLIMMER_MODEL:-$repo_dir/models/muse-glimmer-30b-nvfp4}"
hf_home="${HF_HOME:-${XDG_CACHE_HOME:-${HOME:?HOME must be set}/.cache}/huggingface}"
dflash_revision="93769bc7ab5ad1e9cd22d857e3138cf5d977ae81"
dflash_default="$hf_home/hub/models--meta-models--Muse-Glimmer-30B-GGUF/snapshots/$dflash_revision/dflash-kquant.gguf"
dflash_gguf="${MUSE_GLIMMER_DFLASH:-$dflash_default}"
served_model="${EIDER_SERVED_MODEL:-eider-muse-glimmer-30b}"
max_context_tokens="${EIDER_MAX_CONTEXT_TOKENS:-131072}"
prefill_token_capacity="${EIDER_PREFILL_TOKEN_CAPACITY:-64}"

if [[ ! -f "$model_dir/model.safetensors.index.json" ]]; then
  echo "Muse Glimmer NVFP4 checkpoint not found: $model_dir" >&2
  echo "set MUSE_GLIMMER_MODEL to the downloaded checkpoint directory" >&2
  exit 1
fi
if [[ ! -f "$dflash_gguf" ]]; then
  echo "Muse Glimmer DFlash companion not found: $dflash_gguf" >&2
  echo "set MUSE_GLIMMER_DFLASH to Meta's dflash-kquant.gguf" >&2
  exit 1
fi

exec cargo run --release \
  --manifest-path "$repo_dir/Cargo.toml" \
  -p eider-api \
  --bin eider-serve \
  -- \
  --model-dir "$model_dir" \
  --dflash-gguf "$dflash_gguf" \
  --served-model-name "$served_model" \
  --max-context-tokens "$max_context_tokens" \
  --prefill-token-capacity "$prefill_token_capacity" \
  "$@"
