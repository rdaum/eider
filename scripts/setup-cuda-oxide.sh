#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"

toolchain="nightly-2026-08-28"
revision="97f8b2b7882f0c15ad9ce9b53abed5553920caa8"
deps_dir="${EIDER_DEPS_DIR:-$repo_dir/.deps}"
install_root="${CARGO_OXIDE_ROOT:-$deps_dir/cuda-oxide}"
cargo_oxide="$install_root/bin/cargo-oxide"

if [[ -x "$cargo_oxide" ]]; then
  printf '%s\n' "$cargo_oxide"
  exit 0
fi

rustup toolchain install "$toolchain" \
  --profile minimal \
  --component rust-src \
  --component rustc-dev \
  --component llvm-tools

cargo "+$toolchain" install \
  --root "$install_root" \
  --git https://github.com/NVlabs/cuda-oxide \
  --rev "$revision" \
  cargo-oxide

printf '%s\n' "$cargo_oxide"
