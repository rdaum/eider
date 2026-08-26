#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cache_root="${XDG_CACHE_HOME:-$HOME/.cache}"
repository="${DEEPSEEK4_REPOSITORY:-nvidia/DeepSeek-V4-Flash-NVFP4}"
revision="${DEEPSEEK4_REVISION:-e3cd60e7de98e9867116860d522499a728de1cf9}"
template_repository="${DEEPSEEK4_TEMPLATE_REPOSITORY:-deepseek-ai/DeepSeek-V4-Flash}"
template_revision="${DEEPSEEK4_TEMPLATE_REVISION:-014a5cfe6d1349d3d1096b2f8c15faaaa11819d5}"
staging_dir="${DEEPSEEK4_STAGING_DIR:-$cache_root/eider/staging/deepseek-v4-flash-nvfp4-$revision}"
artifact_dir="${DEEPSEEK4_ARTIFACT_DIR:-$cache_root/eider/models/nvidia--DeepSeek-V4-Flash-NVFP4/$revision/deepseek4-experts-nvfp4-v2}"
legacy_q3_dir="${DEEPSEEK4_LEGACY_Q3_DIR:-$cache_root/eider/models/nvidia--DeepSeek-V4-Flash-NVFP4/$revision/deepseek4-experts-q3-v1}"
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

if [[ ! -f "$staging_dir/chat_template.jinja" ]]; then
    hf download "$template_repository" \
        --revision "$template_revision" \
        --local-dir "$staging_dir" \
        --include chat_template.jinja
fi

config="$staging_dir/config.json"
index="$staging_dir/model.safetensors.index.json"
layers="$(jq -er '.num_hidden_layers' "$config")"
mtp_layers="$(jq -er '.num_nextn_predict_layers' "$config")"
experts="$(jq -er '.n_routed_experts' "$config")"
hidden="$(jq -er '.hidden_size' "$config")"
intermediate="$(jq -er '.moe_intermediate_size' "$config")"

weights_per_matrix=$((hidden * intermediate))
record_bytes=$((3 * (weights_per_matrix / 2 + weights_per_matrix / 16)))
required_bytes=$(((layers + mtp_layers) * (8192 + experts * record_bytes)))
prepared_bytes="$(du -sb "$artifact_dir" 2>/dev/null | awk '{print $1}')"
prepared_bytes="${prepared_bytes:-0}"
if [[ -d "$legacy_q3_dir" ]]; then
    reclaimable_bytes="$(du -sb "$legacy_q3_dir" | awk '{print $1}')"
else
    reclaimable_bytes=0
fi
available_bytes="$(df --output=avail -B1 "$artifact_dir" | tail -1 | tr -d ' ')"
reserve_bytes=$((5 * 1024 * 1024 * 1024))
missing_bytes=$((required_bytes > prepared_bytes ? required_bytes - prepared_bytes : 0))
if ((missing_bytes + reserve_bytes > available_bytes + reclaimable_bytes)); then
    echo "insufficient disk for exact DeepSeek V4 experts:" >&2
    echo "  missing:      $missing_bytes bytes" >&2
    echo "  available:    $available_bytes bytes" >&2
    echo "  reclaimable:  $reclaimable_bytes bytes" >&2
    echo "  reserve:      $reserve_bytes bytes" >&2
    exit 1
fi

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
    nvfp4_ready=false
    if "$binary" inspect-nvfp4-layer "$staging_dir" "$artifact_dir" "$layer" >/dev/null 2>&1; then
        nvfp4_ready=true
    fi
    thin_ready=false
    if "$binary" inspect-thin-shard "$staging_dir" "$thin_dir" "$shard" >/dev/null 2>&1; then
        thin_ready=true
    fi
    if [[ "$nvfp4_ready" == true && "$thin_ready" == true ]]; then
        rm -f \
            "$legacy_q3_dir/layer-$(printf '%02d' "$layer")-gate-up.q3t" \
            "$legacy_q3_dir/layer-$(printf '%02d' "$layer")-down.q3t"
        continue
    fi

    echo "preparing exact DeepSeek V4 NVFP4 layer $layer/$((layers - 1)) from $shard"
    hf download "$repository" \
        --revision "$revision" \
        --local-dir "$staging_dir" \
        --include "$shard"
    if [[ "$nvfp4_ready" != true ]]; then
        "$binary" prepare-nvfp4-layer "$staging_dir" "$artifact_dir" "$layer"
        "$binary" inspect-nvfp4-layer "$staging_dir" "$artifact_dir" "$layer"
    fi
    if [[ "$thin_ready" != true ]]; then
        "$binary" prepare-thin-shard "$staging_dir" "$thin_dir" "$shard"
    fi

    # Exact NVFP4 has now replaced this layer's approximate Q3 tables. Delete
    # those only after the exact layer validates, then release the disposable
    # source shard.
    rm -f \
        "$legacy_q3_dir/layer-$(printf '%02d' "$layer")-gate-up.q3t" \
        "$legacy_q3_dir/layer-$(printf '%02d' "$layer")-down.q3t"
    unlink -- "$staging_dir/$shard"
done

if ((mtp_layers == 1)); then
    mtp_prefix="mtp.0.ffn.experts."
    mapfile -t mtp_shards < <(
        jq -r --arg prefix "$mtp_prefix" '
            .weight_map
            | to_entries[]
            | select(.key | startswith($prefix))
            | .value
        ' "$index" | sort -u
    )
    if [[ "${#mtp_shards[@]}" -ne 1 ]]; then
        echo "MTP expert tensors span ${#mtp_shards[@]} shards; expected exactly one" >&2
        printf '  %s\n' "${mtp_shards[@]}" >&2
        exit 1
    fi
    mtp_shard="${mtp_shards[0]}"
    if ! "$binary" inspect-nvfp4-mtp "$staging_dir" "$artifact_dir" >/dev/null 2>&1 \
        || ! "$binary" inspect-thin-shard "$staging_dir" "$thin_dir" "$mtp_shard" >/dev/null 2>&1; then
        echo "preparing exact DeepSeek V4 MTP NVFP4 experts from $mtp_shard"
        hf download "$repository" \
            --revision "$revision" \
            --local-dir "$staging_dir" \
            --include "$mtp_shard"
        "$binary" prepare-nvfp4-mtp "$staging_dir" "$artifact_dir"
        "$binary" inspect-nvfp4-mtp "$staging_dir" "$artifact_dir"
        "$binary" prepare-thin-shard "$staging_dir" "$thin_dir" "$mtp_shard"
        unlink -- "$staging_dir/$mtp_shard"
    fi
fi

head_shard="$(jq -er '.weight_map["head.weight"]' "$index")"
prepare_thin_shard "$head_shard"

"$binary" inspect-nvfp4 "$staging_dir" "$artifact_dir"
if ((mtp_layers == 1)); then
    "$binary" inspect-nvfp4-mtp "$staging_dir" "$artifact_dir"
fi
"$binary" finalise-thin "$staging_dir" "$thin_dir"
"$binary" inspect-thin "$thin_dir"
rm -f "$legacy_q3_dir/manifest.json"
rmdir "$legacy_q3_dir" 2>/dev/null || true

echo "DeepSeek V4 exact NVFP4 experts: $artifact_dir"
echo "DeepSeek V4 thin serving checkpoint: $thin_dir"
