#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec env EIDER_MODEL=flint-nvfp4 "$repo_dir/scripts/run-eider" "$@"
