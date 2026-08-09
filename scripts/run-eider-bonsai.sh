#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_models_dir="$(cd "$repo_dir/../llm-learn/models" 2>/dev/null && pwd || true)"
gguf="${EIDER_BONSAI_GGUF:-${default_models_dir:+$default_models_dir/Ternary-Bonsai-8B-Q2_0_g64.gguf}}"
tokenizer="${EIDER_BONSAI_TOKENIZER:-${default_models_dir:+$default_models_dir/qwen3-tokenizer.json}}"
runtime_dir="${EIDER_BONSAI_RUNTIME_DIR:-${XDG_CACHE_HOME:-${HOME:?HOME must be set}/.cache}/eider/bonsai-local}"

if [[ -z "$gguf" || ! -f "$gguf" ]]; then
  echo "Bonsai GGUF not found; set EIDER_BONSAI_GGUF" >&2
  exit 1
fi
if [[ -z "$tokenizer" || ! -f "$tokenizer" ]]; then
  echo "Qwen3 tokenizer not found; set EIDER_BONSAI_TOKENIZER" >&2
  exit 1
fi

install -d "$runtime_dir"
ln -sfn "$gguf" "$runtime_dir/Ternary-Bonsai-8B-Q2_0_g64.gguf"
ln -sfn "$tokenizer" "$runtime_dir/tokenizer.json"
ln -sfn "$repo_dir/scripts/bonsai/config.json" "$runtime_dir/config.json"
ln -sfn "$repo_dir/scripts/bonsai/generation_config.json" "$runtime_dir/generation_config.json"
ln -sfn "$repo_dir/scripts/bonsai/tokenizer_config.json" "$runtime_dir/tokenizer_config.json"

exec cargo run --release \
  --manifest-path "$repo_dir/Cargo.toml" \
  -p eider-api \
  --bin eider-serve \
  -- --model-dir "$runtime_dir" \
  --served-model-name eider-ternary-bonsai-8b \
  --prefix-cache-gib 0 \
  "$@"
