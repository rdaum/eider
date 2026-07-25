#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

pi_bin="${PI_BIN:-pi}"
provider="${PI_EIDER_PROVIDER:-eider}"
model="${EIDER_SERVED_MODEL:-eider-deepseek-v4}"
thinking="${PI_EIDER_THINKING:-low}"
health_url="${EIDER_HEALTH_URL:-http://127.0.0.1:8080/healthz}"
export EIDER_API_KEY="${EIDER_API_KEY:-local-eider}"
export PI_CODING_AGENT_DIR="${PI_CODING_AGENT_DIR:-$repo_dir/pi/agent}"
export PI_OFFLINE="${PI_OFFLINE:-1}"

if ! command -v "$pi_bin" >/dev/null 2>&1; then
  echo "Pi executable not found: $pi_bin" >&2
  exit 1
fi

if command -v curl >/dev/null 2>&1 && ! curl --fail --silent --show-error "$health_url" >/dev/null; then
  echo "Eider is not ready at $health_url" >&2
  echo "start it first with scripts/run-eider-deepseek4.sh" >&2
  exit 1
fi

exec "$pi_bin" \
  --provider "$provider" \
  --model "$model" \
  --thinking "$thinking" \
  "$@"
