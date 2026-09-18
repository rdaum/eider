#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model="${EIDER_MODEL:-gemma-4-26b-a4b-nvfp4}"
server_args=()

offline_set=false
speculative_drafts_set=false
max_context_tokens_set=false
max_active_sequences_set=false
decode_capacity_set=false
decision_branch_capacity_set=false
prefill_token_capacity_set=false
prefill_sequence_capacity_set=false
retained_prefix_gib_set=false
for arg in "$@"; do
  case "$arg" in
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
    --decision-branch-capacity | --decision-branch-capacity=*)
      decision_branch_capacity_set=true
      ;;
    --prefill-token-capacity | --prefill-token-capacity=*)
      prefill_token_capacity_set=true
      ;;
    --prefill-sequence-capacity | --prefill-sequence-capacity=*)
      prefill_sequence_capacity_set=true
      ;;
    --retained-prefix-gib | --retained-prefix-gib=*)
      retained_prefix_gib_set=true
      ;;
  esac
  server_args+=("$arg")
done

if [[ "$offline_set" == false && "${EIDER_OFFLINE:-1}" != 0 ]]; then
  server_args+=(--offline)
fi
if [[ "$speculative_drafts_set" == false ]]; then
  server_args+=(--speculative-drafts "${EIDER_SPECULATIVE_DRAFTS:-0}")
fi
if [[ "$max_context_tokens_set" == false ]]; then
  server_args+=(--max-context-tokens "${EIDER_MAX_CONTEXT_TOKENS:-65536}")
fi
if [[ "$max_active_sequences_set" == false ]]; then
  server_args+=(--max-active-sequences "${EIDER_MAX_ACTIVE_SEQUENCES:-4}")
fi
if [[ "$decode_capacity_set" == false ]]; then
  server_args+=(--decode-capacity "${EIDER_DECODE_CAPACITY:-4}")
fi
if [[ "$decision_branch_capacity_set" == false ]]; then
  server_args+=(--decision-branch-capacity "${EIDER_DECISION_BRANCH_CAPACITY:-3}")
fi
if [[ "$prefill_token_capacity_set" == false ]]; then
  server_args+=(--prefill-token-capacity "${EIDER_PREFILL_TOKEN_CAPACITY:-1536}")
fi
if [[ "$prefill_sequence_capacity_set" == false ]]; then
  server_args+=(--prefill-sequence-capacity "${EIDER_PREFILL_SEQUENCE_CAPACITY:-4}")
fi
if [[ "$retained_prefix_gib_set" == false ]]; then
  server_args+=(--retained-prefix-gib "${EIDER_RETAINED_PREFIX_GIB:-0}")
fi

exec cargo run --release \
  --manifest-path "$repo_dir/Cargo.toml" \
  -p eider-api \
  --bin eider-serve \
  -- "$model" "${server_args[@]}"
