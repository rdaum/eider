#!/usr/bin/env bash
set -euo pipefail

base_url="${EIDER_BASE_URL:-http://127.0.0.1:8080}"
base_url="${base_url%/}"
max_questions="${1:-${EIDER_QUESTIONS_MAX:-64}}"
requested_model="${2:-${EIDER_QUESTIONS_MODEL:-}}"
runs="${EIDER_QUESTIONS_RUNS:-3}"
warmups="${EIDER_QUESTIONS_WARMUPS:-1}"
state="${EIDER_QUESTIONS_STATE:-Help! My payouts have been failing for three days and I need to know who can fix them.}"

usage() {
    printf 'Usage: %s [MAX_QUESTIONS] [MODEL]\n' "${0##*/}" >&2
    printf '\nEnvironment:\n' >&2
    printf '  EIDER_BASE_URL           Server URL (default: http://127.0.0.1:8080)\n' >&2
    printf '  EIDER_API_KEY            Optional bearer token\n' >&2
    printf '  EIDER_QUESTIONS_RUNS     Timed runs per count (default: 3)\n' >&2
    printf '  EIDER_QUESTIONS_WARMUPS  Warm-up runs per count (default: 1)\n' >&2
    printf '  EIDER_QUESTIONS_STATE    Shared decision state\n' >&2
}

if [[ $# -gt 2 ]]; then
    usage
    exit 2
fi
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    usage
    exit 0
fi
if [[ ! "$max_questions" =~ ^[1-9][0-9]*$ ]] || ((max_questions > 64)); then
    printf 'MAX_QUESTIONS must be an integer in 1..64, got: %s\n' "$max_questions" >&2
    exit 2
fi
if [[ ! "$runs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'EIDER_QUESTIONS_RUNS must be a positive integer, got: %s\n' "$runs" >&2
    exit 2
fi
if [[ ! "$warmups" =~ ^[0-9]+$ ]]; then
    printf 'EIDER_QUESTIONS_WARMUPS must be a nonnegative integer, got: %s\n' "$warmups" >&2
    exit 2
fi
for command in curl jq awk; do
    if ! command -v "$command" >/dev/null 2>&1; then
        printf '%s is required but was not found in PATH.\n' "$command" >&2
        exit 1
    fi
done

auth_headers=()
if [[ -n "${EIDER_API_KEY:-}" ]]; then
    auth_headers=(-H "Authorization: Bearer ${EIDER_API_KEY}")
fi

if ! curl --fail --silent --show-error \
    "${auth_headers[@]}" "$base_url/healthz" >/dev/null; then
    printf 'Eider is not responding at %s.\n' "$base_url" >&2
    exit 1
fi
models_json="$(curl --fail --silent --show-error \
    "${auth_headers[@]}" "$base_url/v1/models")"
served_model="$(jq --raw-output '.data[0].id // empty' <<<"$models_json")"
if [[ -z "$served_model" ]]; then
    printf '%s/v1/models did not report a model.\n' "$base_url" >&2
    exit 1
fi
if [[ -n "$requested_model" && "$requested_model" != "$served_model" ]]; then
    printf 'Eider is serving %s, not requested model %s.\n' \
        "$served_model" "$requested_model" >&2
    exit 1
fi
model="$served_model"

counts=(1)
if ((max_questions >= 2)); then
    counts+=(2)
fi
previous=1
current=2
while ((previous + current <= max_questions)); do
    next=$((previous + current))
    counts+=("$next")
    previous=$current
    current=$next
done
last_index=$((${#counts[@]} - 1))
if ((counts[last_index] != max_questions)); then
    counts+=("$max_questions")
fi

work_dir="$(mktemp -d)"
request_file="$work_dir/request.json"
response_file="$work_dir/response.json"
cleanup() {
    rm -f "$request_file" "$response_file"
    rmdir "$work_dir" 2>/dev/null || true
}
trap cleanup EXIT

build_request() {
    local count="$1"
    jq --null-input \
        --arg model "$model" \
        --arg state "$state" \
        --argjson count "$count" \
        '{
            model: $model,
            state: $state,
            questions: (reduce range(1; $count + 1) as $number ({};
                .["question_\($number)"] = {
                    type: "noul",
                    instructions: "Question \($number): Does this support request need prompt attention?",
                    criteria: {
                        "true": "The request needs prompt attention.",
                        "false": "The request can wait."
                    }
                }
            ))
        }' >"$request_file"
}

run_request() {
    local expected_answers="$1" result status elapsed actual_answers
    if ! result="$(curl --silent --show-error \
        "${auth_headers[@]}" \
        -H 'Content-Type: application/json' \
        --output "$response_file" \
        --write-out '%{http_code} %{time_total}' \
        --data-binary "@$request_file" \
        "$base_url/v1/decisions")"; then
        printf 'Request with %s questions failed.\n' "$expected_answers" >&2
        return 1
    fi
    read -r status elapsed <<<"$result"
    if [[ "$status" != 200 ]]; then
        printf 'Request with %s questions returned HTTP %s:\n' \
            "$expected_answers" "$status" >&2
        jq . "$response_file" >&2 2>/dev/null || sed -n '1,40p' "$response_file" >&2
        return 1
    fi
    actual_answers="$(jq '.answers | length' "$response_file")"
    if [[ "$actual_answers" != "$expected_answers" ]]; then
        printf 'Expected %s answers but received %s.\n' \
            "$expected_answers" "$actual_answers" >&2
        return 1
    fi
    awk -v seconds="$elapsed" 'BEGIN { printf "%.3f\n", seconds * 1000 }'
}

printf 'Eider decision question latency\n'
printf '  server: %s\n' "$base_url"
printf '  model:  %s\n' "$model"
printf '  runs:   %s timed + %s warm-up per count\n\n' "$runs" "$warmups"
printf '%10s %8s %12s %12s %12s\n' 'questions' 'runs' 'min_ms' 'mean_ms' 'max_ms'

for count in "${counts[@]}"; do
    build_request "$count"
    for ((run = 0; run < warmups; run++)); do
        run_request "$count" >/dev/null
    done

    latencies=()
    for ((run = 0; run < runs; run++)); do
        latencies+=("$(run_request "$count")")
    done
    stats="$(printf '%s\n' "${latencies[@]}" | awk '
        NR == 1 { minimum = maximum = $1 }
        { sum += $1; if ($1 < minimum) minimum = $1; if ($1 > maximum) maximum = $1 }
        END { printf "%.3f %.3f %.3f", minimum, sum / NR, maximum }
    ')"
    read -r minimum mean maximum <<<"$stats"
    printf '%10d %8d %12.3f %12.3f %12.3f\n' \
        "$count" "$runs" "$minimum" "$mean" "$maximum"
done
