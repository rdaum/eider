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
# Retain one decision allocation and fit shared prefixes plus suffix waves.
add_default --max-active-sequences 1
add_default --decode-capacity 1
add_default --prefill-sequence-capacity 1
add_default --prefill-token-capacity 2048
exec env EIDER_MODEL=lfm3-1.7b-nvfp4 "$repo_dir/scripts/run-eider" "${server_args[@]}"
