#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
server_args=("$@")
add_default() {
  local option="$1" value="$2" arg
  for arg in "${server_args[@]}"; do
    if [[ "$arg" == "$option" || "$arg" == "$option="* ]]; then
      return
    fi
  done
  server_args+=("$option" "$value")
}
add_default --max-active-sequences 1
add_default --decode-capacity 1
add_default --prefill-sequence-capacity 1
# The catalogue resolves the pinned snapshot in the standard Hugging Face cache.
# Pass --dory-loops 1 to trade reasoning depth for latency.
exec cargo run --release --manifest-path "$repo_dir/Cargo.toml" \
  -p eider-api --bin eider-serve -- dory1.0-4b-nvfp4 --offline "${server_args[@]}"
