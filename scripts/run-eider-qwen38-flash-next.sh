#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model="${EIDER_MODEL:-qwen3.8-flash-next}"
server_args=()
cuda_oxide=false

offline_set=false
speculative_drafts_set=false
max_context_tokens_set=false
max_active_sequences_set=false
decode_capacity_set=false
prefill_token_capacity_set=false
prefill_sequence_capacity_set=false
for arg in "$@"; do
  case "$arg" in
    --cuda-oxide)
      cuda_oxide=true
      continue
      ;;
    --offline)
      offline_set=true
      ;;
    --speculative-drafts | --speculative-drafts=*)
      speculative_drafts_set=true
      ;;
    --max-context-tokens | --max-context-tokens=*)
      max_context_tokens_set=true
      ;;
    --max-active-sequences | --max-active-sequences=*)
      max_active_sequences_set=true
      ;;
    --decode-capacity | --decode-capacity=*)
      decode_capacity_set=true
      ;;
    --prefill-token-capacity | --prefill-token-capacity=*)
      prefill_token_capacity_set=true
      ;;
    --prefill-sequence-capacity | --prefill-sequence-capacity=*)
      prefill_sequence_capacity_set=true
      ;;
  esac
  server_args+=("$arg")
done

if [[ "$offline_set" == false ]]; then
  server_args+=(--offline)
fi
if [[ "$speculative_drafts_set" == false ]]; then
  server_args+=(--speculative-drafts "${EIDER_SPECULATIVE_DRAFTS:-1}")
fi
if [[ "$max_context_tokens_set" == false ]]; then
  server_args+=(--max-context-tokens 262144)
fi
if [[ "$max_active_sequences_set" == false ]]; then
  server_args+=(--max-active-sequences 2)
fi
if [[ "$decode_capacity_set" == false ]]; then
  server_args+=(--decode-capacity 2)
fi
if [[ "$prefill_token_capacity_set" == false ]]; then
  server_args+=(--prefill-token-capacity 64)
fi
if [[ "$prefill_sequence_capacity_set" == false ]]; then
  server_args+=(--prefill-sequence-capacity 2)
fi

if [[ "$cuda_oxide" == true ]]; then
  if [[ -z "${CARGO_OXIDE:-}" ]]; then
    if cargo_oxide_path="$(command -v cargo-oxide)"; then
      export CARGO_OXIDE="$cargo_oxide_path"
    else
      export CARGO_OXIDE="$repo_dir/.deps/cuda-oxide/bin/cargo-oxide"
      if [[ ! -x "$CARGO_OXIDE" ]]; then
        "$repo_dir/scripts/setup-cuda-oxide.sh"
      fi
    fi
  elif ! command -v "$CARGO_OXIDE" >/dev/null 2>&1; then
    printf 'cargo-oxide is not executable: %s\n' "$CARGO_OXIDE" >&2
    exit 1
  fi
  cargo_oxide_path="$(command -v "$CARGO_OXIDE")"
  export CARGO_OXIDE="$(realpath "$cargo_oxide_path")"
fi

cargo_args=(
  run
  --release
  --manifest-path "$repo_dir/Cargo.toml"
  -p eider-api
  --bin eider-serve
)
if [[ "$cuda_oxide" == true ]]; then
  cargo_args+=(--features cuda-oxide)
fi

exec cargo "${cargo_args[@]}" -- "$model" "${server_args[@]}"
