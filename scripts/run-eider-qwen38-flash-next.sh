#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model="${EIDER_MODEL:-qwen3.8-flash-next}"

exec cargo run --release \
  --manifest-path "$repo_dir/Cargo.toml" \
  -p eider-api \
  --bin eider-serve \
  -- \
  "$model" \
  --offline \
  --max-context-tokens 262144 \
  --prefill-token-capacity 64 \
  --max-active-sequences 1 \
  --decode-capacity 1 \
  --prefill-sequence-capacity 1 \
  --speculative-drafts 0 \
  "$@"
