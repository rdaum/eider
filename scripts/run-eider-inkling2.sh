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
# Each retained prefix copies about 172 MB of local KV and convolution history
# and shares its global KV pages.
add_default --retained-prefix-gib 8

exec cargo run --release --manifest-path "$repo_dir/Cargo.toml" \
  -p eider-api --bin eider-serve -- inkling-2-mini-text --offline "${server_args[@]}"
