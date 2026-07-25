#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cache_root="${XDG_CACHE_HOME:-$HOME/.cache}"
repository="${DEEPSEEK4_REPOSITORY:-nvidia/DeepSeek-V4-Flash-NVFP4}"
revision="${DEEPSEEK4_REVISION:-e3cd60e7de98e9867116860d522499a728de1cf9}"
staging_dir="${DEEPSEEK4_STAGING_DIR:-$cache_root/eider/staging/deepseek-v4-flash-nvfp4-$revision}"
artifact_dir="${DEEPSEEK4_ARTIFACT_DIR:-$cache_root/eider/models/nvidia--DeepSeek-V4-Flash-NVFP4/$revision/deepseek4-experts-q2-v2}"
thin_dir="${DEEPSEEK4_THIN_DIR:-$cache_root/eider/models/nvidia--DeepSeek-V4-Flash-NVFP4/$revision/deepseek4-thin-nvfp4-v1}"
binary="$repo_root/target/release/deepseek4-experts"

for command in hf jq; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "required command not found: $command" >&2
        exit 1
    fi
done

mkdir -p "$staging_dir" "$artifact_dir" "$thin_dir"

cargo build \
    --manifest-path "$repo_root/Cargo.toml" \
    --release \
    -p infer \
    --bin deepseek4-experts

hf download "$repository" \
    --revision "$revision" \
    --local-dir "$staging_dir" \
    --include config.json \
    --include generation_config.json \
    --include tokenizer.json \
    --include tokenizer_config.json \
    --include chat_template.jinja \
    --include model.safetensors.index.json

config="$staging_dir/config.json"
index="$staging_dir/model.safetensors.index.json"
layers="$(jq -er '.num_hidden_layers' "$config")"
"$binary" preflight "$staging_dir" "$artifact_dir"

for metadata in config.json generation_config.json tokenizer.json tokenizer_config.json chat_template.jinja; do
    if [[ -f "$staging_dir/$metadata" ]]; then
        install -m 0644 "$staging_dir/$metadata" "$thin_dir/$metadata"
    fi
done

prepare_thin_shard() {
    local shard="$1"
    if "$binary" inspect-thin-shard "$staging_dir" "$thin_dir" "$shard" >/dev/null 2>&1; then
        return
    fi
    hf download "$repository" \
        --revision "$revision" \
        --local-dir "$staging_dir" \
        --include "$shard"
    "$binary" prepare-thin-shard "$staging_dir" "$thin_dir" "$shard"
    unlink -- "$staging_dir/$shard"
}

embed_shard="$(jq -er '.weight_map["embed.weight"]' "$index")"
prepare_thin_shard "$embed_shard"

for ((layer = 0; layer < layers; layer++)); do
    gate_up="$artifact_dir/layer-$(printf '%02d' "$layer")-gate-up.q2t"
    down="$artifact_dir/layer-$(printf '%02d' "$layer")-down.q2t"
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
        printf '  %s\n' "${shards[@]}" >&2
        exit 1
    fi
    shard="${shards[0]}"
    q2_ready=false
    if [[ -f "$gate_up" && -f "$down" ]] &&
        "$binary" prepare-layer "$staging_dir" "$artifact_dir" "$layer"; then
        q2_ready=true
    fi
    thin_ready=false
    if "$binary" inspect-thin-shard "$staging_dir" "$thin_dir" "$shard" >/dev/null 2>&1; then
        thin_ready=true
    fi
    if [[ "$q2_ready" == true && "$thin_ready" == true ]]; then
        continue
    fi

    echo "preparing DeepSeek V4 expert layer $layer/$((layers - 1)) from $shard"
    hf download "$repository" \
        --revision "$revision" \
        --local-dir "$staging_dir" \
        --include "$shard"
    if [[ "$q2_ready" != true ]]; then
        "$binary" prepare-layer "$staging_dir" "$artifact_dir" "$layer"
    fi
    if [[ "$thin_ready" != true ]]; then
        "$binary" prepare-thin-shard "$staging_dir" "$thin_dir" "$shard"
    fi

    # This is a disposable staging tree, not an immutable Hugging Face
    # snapshot. Retaining completed source shards would defeat bounded-disk
    # preparation.
    unlink -- "$staging_dir/$shard"
done

head_shard="$(jq -er '.weight_map["head.weight"]' "$index")"
prepare_thin_shard "$head_shard"

# Every layer now validates without its source shard, so this only publishes
# the complete artifact manifest and reports the exact result size.
"$binary" prepare "$staging_dir" "$artifact_dir"
"$binary" inspect "$staging_dir" "$artifact_dir"
"$binary" finalise-thin "$staging_dir" "$thin_dir"
"$binary" inspect-thin "$thin_dir"

echo "DeepSeek V4 Q2 experts: $artifact_dir"
echo "DeepSeek V4 thin serving checkpoint: $thin_dir"
