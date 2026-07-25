#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 <hotset-plan.json>" >&2
    echo 'plan format: {"0":[4,17,93],"1":[8,42]}' >&2
    exit 1
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cache_root="${XDG_CACHE_HOME:-$HOME/.cache}"
repository="${DEEPSEEK4_REPOSITORY:-nvidia/DeepSeek-V4-Flash-NVFP4}"
revision="${DEEPSEEK4_REVISION:-e3cd60e7de98e9867116860d522499a728de1cf9}"
capacity="${DEEPSEEK4_HOT_CAPACITY:-8}"
staging_dir="${DEEPSEEK4_STAGING_DIR:-$cache_root/eider/staging/deepseek-v4-flash-nvfp4-$revision}"
hot_cache_dir="${DEEPSEEK4_HOT_CACHE_DIR:-$cache_root/eider/models/nvidia--DeepSeek-V4-Flash-NVFP4/$revision/deepseek4-hot-nvfp4-v1}"
plan="$(realpath -- "$1")"
binary="$repo_root/target/release/deepseek4-experts"

for command in hf jq; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "required command not found: $command" >&2
        exit 1
    fi
done
if ! [[ "$capacity" =~ ^[1-9][0-9]*$ ]]; then
    echo "DEEPSEEK4_HOT_CAPACITY must be a positive integer" >&2
    exit 1
fi
jq -e 'type == "object" and all(.[]; type == "array")' "$plan" >/dev/null

mkdir -p "$staging_dir" "$hot_cache_dir"
cargo build \
    --manifest-path "$repo_root/Cargo.toml" \
    --release \
    -p infer \
    --bin deepseek4-experts

hf download "$repository" \
    --revision "$revision" \
    --local-dir "$staging_dir" \
    --include config.json \
    --include model.safetensors.index.json

config="$staging_dir/config.json"
index="$staging_dir/model.safetensors.index.json"
layers="$(jq -er '.num_hidden_layers' "$config")"

for ((layer = 0; layer < layers; layer++)); do
    mapfile -t experts < <(jq -r --arg layer "$layer" '.[$layer] // [] | .[]' "$plan")
    if [[ "${#experts[@]}" -eq 0 ]]; then
        "$binary" prepare-hot-layer \
            "$staging_dir" \
            "$hot_cache_dir" \
            "$capacity" \
            "$layer"
        continue
    fi
    if [[ "${#experts[@]}" -gt "$capacity" ]]; then
        echo "layer $layer selects ${#experts[@]} experts, capacity is $capacity" >&2
        exit 1
    fi
    prefix="layers.$layer.ffn.experts."
    mapfile -t shards < <(
        jq -r --arg prefix "$prefix" '
            .weight_map
            | to_entries[]
            | select(.key | startswith($prefix))
            | .value
        ' "$index" | sort -u
    )
    if [[ "${#shards[@]}" -ne 1 ]]; then
        echo "layer $layer expert tensors span ${#shards[@]} shards; expected exactly one" >&2
        exit 1
    fi
    shard="${shards[0]}"

    echo "caching ${#experts[@]} DeepSeek V4 hot experts for layer $layer from $shard"
    hf download "$repository" \
        --revision "$revision" \
        --local-dir "$staging_dir" \
        --include "$shard"
    "$binary" prepare-hot-layer \
        "$staging_dir" \
        "$hot_cache_dir" \
        "$capacity" \
        "$layer" \
        "${experts[@]}"
    unlink -- "$staging_dir/$shard"
done

"$binary" inspect-hot "$staging_dir" "$hot_cache_dir" "$capacity"
echo "DeepSeek V4 bounded NVFP4 hot-expert cache: $hot_cache_dir"
