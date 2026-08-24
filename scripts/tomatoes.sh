#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base_url="${EIDER_BASE_URL:-http://127.0.0.1:8080}"
max_tokens="${EIDER_TOMATO_MAX_TOKENS:-4000}"

if [[ -t 2 ]]; then
    interactive=1
else
    interactive=0
fi
if ((interactive)) && [[ -z "${NO_COLOR:-}" ]]; then
    reset=$'\033[0m'
    bold=$'\033[1m'
    red=$'\033[38;5;196m'
    green=$'\033[38;5;82m'
    yellow=$'\033[38;5;220m'
    cyan=$'\033[38;5;45m'
else
    reset=""
    bold=""
    red=""
    green=""
    yellow=""
    cyan=""
fi

# API model name | catalogue/local deployment | recommended launcher.
model_rows=(
    "eider-bitnet-b1.58-2b|bitnet-b1.58-2b-4t|scripts/run-eider-bitnet.sh"
    "eider-muse-glimmer-30b|muse-glimmer-30b-nvfp4|EIDER_MODEL=muse-glimmer-30b-nvfp4 scripts/run-eider"
    "eider-ling-3.0-tiny|ling-3.0-tiny-nvfp4|scripts/run-ling3-tiny"
    "eider-qwen3.6|qwen3.6-35b-a3b|EIDER_MODEL=qwen3.6-35b-a3b scripts/run-eider"
    "eider-ornith-1.5-35b-a3b|ornith-1.5-35b-a3b|EIDER_MODEL=ornith-1.5-35b-a3b scripts/run-eider"
    "eider-qwen3.8|qwen3.8-27b|scripts/run-eider-qwen38.sh"
    "eider-agents-a1|agents-a1|EIDER_MODEL=agents-a1 scripts/run-eider"
    "eider-laguna-s-2.1|laguna-s-2.1|EIDER_MODEL=laguna-s-2.1 scripts/run-eider"
    "eider-step3.7|step-3.7-flash|EIDER_MODEL=step-3.7-flash scripts/run-eider"
    "eider-gemma4-26b|gemma-4-26b-a4b-nvfp4|EIDER_MODEL=gemma-4-26b-a4b-nvfp4 scripts/run-eider"
    "eider-nemotron3-puzzle|nemotron-3-puzzle-75b-a9b|EIDER_MODEL=nemotron-3-puzzle-75b-a9b scripts/run-eider"
    "eider-nemotron3-super|nemotron-3-super-120b-a12b|EIDER_MODEL=nemotron-3-super-120b-a12b scripts/run-eider"
    "eider-ternary-bonsai-8b|local Bonsai Q2_0|scripts/run-eider-bonsai.sh"
    "eider-deepseek-v4|prepared local DeepSeek V4|scripts/run-eider-deepseek4.sh"
)

model_record_for() {
    local wanted="$1" row api_model deployment launcher
    for row in "${model_rows[@]}"; do
        IFS='|' read -r api_model deployment launcher <<<"$row"
        if [[ "$api_model" == "$wanted" || "$deployment" == "$wanted" ]]; then
            printf '%s\t%s\t%s\n' "$api_model" "$deployment" "$launcher"
            return 0
        fi
    done
    return 1
}

list_models() {
    printf 'Valid API models:\n'
    local row api_model deployment launcher
    for row in "${model_rows[@]}"; do
        IFS='|' read -r api_model deployment launcher <<<"$row"
        printf '  %-29s  %-34s  %s\n' "$api_model" "$deployment" "$launcher"
    done
}

if [[ $# -gt 1 ]]; then
    printf 'Usage: %s [API_MODEL_OR_CATALOGUE_ID]\n\n' "${0##*/}" >&2
    list_models >&2
    exit 2
fi

model=""
deployment=""
launcher=""
if [[ $# -eq 1 ]]; then
    requested_model="$1"
    if ! model_record="$(model_record_for "$requested_model")"; then
        printf 'Unknown model: %s\n\n' "$requested_model" >&2
        list_models >&2
        exit 2
    fi
    IFS=$'\t' read -r model deployment launcher <<<"$model_record"
fi

if ! command -v curl >/dev/null 2>&1; then
    printf 'curl is required but was not found in PATH.\n' >&2
    exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
    printf 'jq is required but was not found in PATH.\n' >&2
    exit 1
fi
auth_headers=()
if [[ -n "${EIDER_API_KEY:-}" ]]; then
    auth_headers=(-H "Authorization: Bearer ${EIDER_API_KEY}")
fi

server_ready=0
served_model=""
if curl --fail --silent --show-error "$base_url/healthz" >/dev/null 2>&1; then
    server_ready=1
    models_json="$(curl --fail --silent --show-error \
        "${auth_headers[@]}" "$base_url/v1/models" 2>/dev/null || true)"
    served_model="$(jq --raw-output '.data[0].id // empty' <<<"$models_json" 2>/dev/null || true)"
fi

if [[ $# -eq 0 ]]; then
    if ((server_ready == 0)); then
        list_models
        exit 0
    fi
    if [[ -z "$served_model" ]]; then
        printf 'Eider is responding at %s, but /v1/models did not report a model.\n' \
            "$base_url" >&2
        printf 'If the server requires authentication, set EIDER_API_KEY.\n' >&2
        exit 1
    fi
    model="$served_model"
    if model_record="$(model_record_for "$model")"; then
        IFS=$'\t' read -r model deployment launcher <<<"$model_record"
    else
        deployment="running Eider deployment"
    fi
fi

if [[ ! "$max_tokens" =~ ^[1-9][0-9]*$ ]]; then
    printf 'EIDER_TOMATO_MAX_TOKENS must be a positive integer, got: %s\n' \
        "$max_tokens" >&2
    exit 2
fi

if ((server_ready == 0)); then
    printf 'Eider is not responding at %s.\n' "$base_url" >&2
    printf 'Launch %s (%s) with:\n' "$model" "$deployment" >&2
    printf '  cd %q\n' "$repo_dir" >&2
    printf '  %s\n' "$launcher" >&2
    exit 1
fi

if [[ -n "$served_model" && "$served_model" != "$model" ]]; then
    printf 'Eider at %s is serving %s, not %s.\n' \
        "$base_url" "$served_model" "$model" >&2
    printf 'Launch the requested deployment with:\n' >&2
    printf '  cd %q\n' "$repo_dir" >&2
    printf '  %s\n' "$launcher" >&2
    exit 1
fi

printf '%b🍅  I’m asking %s about tomatoes!  🍅%b\n' \
    "$bold$red" "$model" "$reset" >&2
printf '%b✓%b %bServer ready%b at %s\n' \
    "$green" "$reset" "$bold" "$reset" "$base_url" >&2
printf '%b→%b Requesting a detailed twenty-point guide, up to %s tokens.\n' \
    "$cyan" "$reset" "$max_tokens" >&2

request="$(jq --null-input \
    --arg model "$model" \
    --arg content 'Write a detailed twenty-point guide to growing tomatoes.' \
    --arg max_tokens "$max_tokens" \
    '{
        model: $model,
        messages: [{role: "user", content: $content}],
        reasoning_effort: "none",
        temperature: 0,
        max_tokens: ($max_tokens | tonumber),
        stream: false
    }')"

prometheus_value() {
    local metrics="$1" metric="$2"
    awk -v metric="$metric" '
        $1 == metric { print $2; found = 1; exit }
        END { if (!found) print 0 }
    ' <<<"$metrics"
}

number_or_zero() {
    if [[ "$1" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]]; then
        printf '%s\n' "$1"
    else
        printf '0\n'
    fi
}

baseline_metrics="$(curl --silent --show-error "$base_url/metrics" 2>/dev/null || true)"
baseline_generated="$(number_or_zero "$(prometheus_value "$baseline_metrics" eider_infer_generated_tokens)")"
baseline_prefill_sum="$(number_or_zero "$(prometheus_value "$baseline_metrics" eider_server_prefill_tokens_per_second_sum)")"
baseline_prefill_count="$(number_or_zero "$(prometheus_value "$baseline_metrics" eider_server_prefill_tokens_per_second_count)")"
baseline_decode_sum="$(number_or_zero "$(prometheus_value "$baseline_metrics" eider_server_decode_tokens_per_second_sum)")"
baseline_decode_count="$(number_or_zero "$(prometheus_value "$baseline_metrics" eider_server_decode_tokens_per_second_count)")"
baseline_ttft_sum="$(number_or_zero "$(prometheus_value "$baseline_metrics" eider_infer_ttft_us_sum)")"
baseline_ttft_count="$(number_or_zero "$(prometheus_value "$baseline_metrics" eider_infer_ttft_us_count)")"

work_dir="$(mktemp -d)"
response_file="$work_dir/response.json"
curl_error_file="$work_dir/curl.err"
curl_pid=""
cleanup() {
    if [[ -n "$curl_pid" ]] && kill -0 "$curl_pid" 2>/dev/null; then
        kill "$curl_pid" 2>/dev/null || true
        wait "$curl_pid" 2>/dev/null || true
    fi
    rm -f "$response_file" "$curl_error_file"
    rmdir "$work_dir" 2>/dev/null || true
}
interrupted() {
    if ((interactive)); then
        printf '\r\033[2K' >&2
    fi
    printf '%b✗ Tomato request cancelled.%b\n' "$yellow$bold" "$reset" >&2
    exit 130
}
trap cleanup EXIT
trap interrupted INT TERM

curl --silent --show-error \
    "$base_url/v1/chat/completions" \
    "${auth_headers[@]}" \
    -H 'Content-Type: application/json' \
    --data "$request" \
    >"$response_file" 2>"$curl_error_file" &
curl_pid=$!

spinner=("🍅" "·🍅" "··🍅" "···🍅" "··🍅" "·🍅")
frame=0
started=$SECONDS
decoded=0
if ((!interactive)); then
    printf 'Waiting for the non-streaming response; progress comes from /metrics...\n' >&2
fi
while kill -0 "$curl_pid" 2>/dev/null; do
    elapsed=$((SECONDS - started))
    metrics="$(curl --max-time 1 --silent "$base_url/metrics" 2>/dev/null || true)"
    current_generated="$(number_or_zero "$(prometheus_value "$metrics" eider_infer_generated_tokens)")"
    current_decode_rate="$(number_or_zero "$(prometheus_value "$metrics" eider_server_current_decode_tokens_per_second)")"
    current_prefill_rate="$(number_or_zero "$(prometheus_value "$metrics" eider_server_current_prefill_tokens_per_second)")"
    if [[ "$current_generated" =~ ^[0-9]+$ ]] && ((current_generated >= baseline_generated)); then
        decoded=$((current_generated - baseline_generated))
    fi
    minutes=$((elapsed / 60))
    seconds=$((elapsed % 60))
    if ((decoded > 0)); then
        phase="decoded ${decoded} tokens · $(printf '%.1f' "$current_decode_rate") tok/s"
    elif [[ "$current_prefill_rate" != "0" ]]; then
        phase="prefilling · $(printf '%.1f' "$current_prefill_rate") tok/s"
    else
        phase="prefilling / waiting for the first token"
    fi
    if ((interactive)); then
        printf '\r\033[2K%b%s%b  %s  %02d:%02d elapsed' \
            "$yellow" "${spinner[frame]}" "$reset" "$phase" "$minutes" "$seconds" >&2
    fi
    frame=$(((frame + 1) % ${#spinner[@]}))
    sleep 0.5
done

curl_status=0
wait "$curl_pid" || curl_status=$?
curl_pid=""
elapsed=$((SECONDS - started))
if ((interactive)); then
    printf '\r\033[2K' >&2
fi
if ((curl_status != 0)); then
    printf '%b✗ Request failed%b after %ss.\n' "$red$bold" "$reset" "$elapsed" >&2
    sed -n '1,8p' "$curl_error_file" >&2
    exit "$curl_status"
fi

if ! completion_tokens="$(jq --exit-status --raw-output '
    if .error then error(.error.message) else .usage.completion_tokens // 0 end
' "$response_file")"; then
    printf '%b✗ Eider returned an invalid response.%b\n' "$red$bold" "$reset" >&2
    sed -n '1,20p' "$response_file" >&2
    exit 1
fi
prompt_tokens="$(jq --raw-output '.usage.prompt_tokens // 0' "$response_file")"
cached_prompt_tokens="$(jq --raw-output '.usage.prompt_tokens_details.cached_tokens // 0' "$response_file")"
[[ "$prompt_tokens" =~ ^[0-9]+$ ]] || prompt_tokens=0
[[ "$cached_prompt_tokens" =~ ^[0-9]+$ ]] || cached_prompt_tokens=0
if ! answer="$(jq --exit-status --raw-output '
    .choices[0].message
    | if (.content // "") != "" then .content else .reasoning_content // "" end
' "$response_file")"; then
    printf '%b✗ The response did not contain an answer.%b\n' "$red$bold" "$reset" >&2
    exit 1
fi

final_metrics="$(curl --silent --show-error "$base_url/metrics" 2>/dev/null || true)"
final_prefill_sum="$(number_or_zero "$(prometheus_value "$final_metrics" eider_server_prefill_tokens_per_second_sum)")"
final_prefill_count="$(number_or_zero "$(prometheus_value "$final_metrics" eider_server_prefill_tokens_per_second_count)")"
final_decode_sum="$(number_or_zero "$(prometheus_value "$final_metrics" eider_server_decode_tokens_per_second_sum)")"
final_decode_count="$(number_or_zero "$(prometheus_value "$final_metrics" eider_server_decode_tokens_per_second_count)")"
final_ttft_sum="$(number_or_zero "$(prometheus_value "$final_metrics" eider_infer_ttft_us_sum)")"
final_ttft_count="$(number_or_zero "$(prometheus_value "$final_metrics" eider_infer_ttft_us_count)")"

metrics_report="$(jq --null-input --raw-output \
    --argjson prompt_tokens "$prompt_tokens" \
    --argjson cached_prompt_tokens "$cached_prompt_tokens" \
    --argjson completion_tokens "$completion_tokens" \
    --argjson prefill_sum_before "$baseline_prefill_sum" \
    --argjson prefill_sum_after "$final_prefill_sum" \
    --argjson prefill_count_before "$baseline_prefill_count" \
    --argjson prefill_count_after "$final_prefill_count" \
    --argjson decode_sum_before "$baseline_decode_sum" \
    --argjson decode_sum_after "$final_decode_sum" \
    --argjson decode_count_before "$baseline_decode_count" \
    --argjson decode_count_after "$final_decode_count" \
    --argjson ttft_sum_before "$baseline_ttft_sum" \
    --argjson ttft_sum_after "$final_ttft_sum" \
    --argjson ttft_count_before "$baseline_ttft_count" \
    --argjson ttft_count_after "$final_ttft_count" '
    def nonnegative: if . < 0 then 0 else . end;
    def delta($before; $after): ($after - $before) | nonnegative;
    def one_decimal: (. * 10 | round) / 10;
    (delta($prefill_sum_before; $prefill_sum_after)) as $prefill_sum
    | (delta($prefill_count_before; $prefill_count_after)) as $prefill_count
    | (delta($decode_sum_before; $decode_sum_after)) as $decode_sum
    | (delta($decode_count_before; $decode_count_after)) as $decode_count
    | (delta($ttft_sum_before; $ttft_sum_after)) as $ttft_sum
    | (delta($ttft_count_before; $ttft_count_after)) as $ttft_count
    | (if $prefill_count > 0 then $prefill_sum / $prefill_count else 0 end) as $prefill_rate
    | (if $decode_count > 0 then $decode_sum / $decode_count else 0 end) as $decode_rate
    | (($prompt_tokens - $cached_prompt_tokens) | nonnegative) as $uncached_prompt_tokens
    | (($completion_tokens - 1) | nonnegative) as $timed_decode_tokens
    | [
        $prompt_tokens,
        $cached_prompt_tokens,
        (if $prefill_rate > 0 then $uncached_prompt_tokens / $prefill_rate else 0 end),
        ($prefill_rate | one_decimal),
        $completion_tokens,
        (if $decode_rate > 0 then $timed_decode_tokens / $decode_rate else 0 end),
        ($decode_rate | one_decimal),
        (if $ttft_count > 0 then $ttft_sum / $ttft_count / 1000000 else 0 end)
    ]
    | @tsv
')"
IFS=$'\t' read -r metric_prompt metric_cached metric_prefill_secs metric_prefill_rate \
    metric_decode metric_decode_secs metric_decode_rate metric_ttft <<<"$metrics_report"

printf '%b✓ Tomato guide complete!%b\n' "$green$bold" "$reset" >&2
printf '%b╭─ 🍅  Performance harvest  ─────────────────────────────╮%b\n' \
    "$bold$cyan" "$reset" >&2
printf '%b│%b  Prefill  %b%4s prompt tokens%b · %7.3fs · %6.1f tok/s  %b│%b\n' \
    "$cyan" "$reset" "$bold" "$metric_prompt" "$reset" \
    "$metric_prefill_secs" "$metric_prefill_rate" "$cyan" "$reset" >&2
if ((metric_cached > 0)); then
    printf '%b│%b  Cache    %4s prompt tokens reused                     %b│%b\n' \
        "$cyan" "$reset" "$metric_cached" "$cyan" "$reset" >&2
fi
printf '%b│%b  Decode   %b%4s output tokens%b · %7.3fs · %6.1f tok/s  %b│%b\n' \
    "$cyan" "$reset" "$bold" "$metric_decode" "$reset" \
    "$metric_decode_secs" "$metric_decode_rate" "$cyan" "$reset" >&2
printf '%b│%b  TTFT      %7.3fs             · wall %4ss          %b│%b\n' \
    "$cyan" "$reset" "$metric_ttft" "$elapsed" "$cyan" "$reset" >&2
printf '%b╰────────────────────────────────────────────────────────╯%b\n\n' \
    "$bold$cyan" "$reset" >&2
printf '%s\n' "$answer"
