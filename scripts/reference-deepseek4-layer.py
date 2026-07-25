#!/usr/bin/env python3
"""Generate an independent reference for one DeepSeek V4 decoder layer."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("source_shard", type=Path)
    parser.add_argument("layer", type=int)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--token-ids",
        default="0,128803,19905,418,9045,28,44388,128804,128821",
    )
    return parser.parse_args()


def load(shard: Path, name: str) -> torch.Tensor:
    with safe_open(shard, framework="pt", device="cpu") as source:
        return source.get_tensor(name)


def main() -> None:
    args = parse_args()
    config = json.loads((args.model_dir / "config.json").read_text())
    if not 0 <= args.layer < config["num_hidden_layers"]:
        raise ValueError(f"layer {args.layer} is outside the checkpoint")
    device = torch.device("cuda")
    hidden = config["hidden_size"]
    streams_count = config["hc_mult"]
    eps = config["rms_norm_eps"]
    hc_eps = config["hc_eps"]
    sinkhorn_iters = config["hc_sinkhorn_iters"]
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    rope_dim = config["qk_rope_head_dim"]
    groups = config["o_groups"]
    o_rank = config["o_lora_rank"]
    top_k = config["num_experts_per_tok"]
    route_scale = config["routed_scaling_factor"]
    swiglu_limit = config["swiglu_limit"]
    ratio = config["compress_ratios"][args.layer]
    prefix = f"layers.{args.layer}"
    token_ids = torch.tensor(
        [int(token) for token in args.token_ids.split(",")],
        dtype=torch.long,
        device=device,
    )
    streams = np.fromfile(args.input, dtype="<f4")
    expected = token_ids.numel() * streams_count * hidden
    if streams.size != expected:
        raise ValueError(f"input contains {streams.size} floats; expected {expected}")
    streams = torch.from_numpy(
        streams.reshape(token_ids.numel(), streams_count, hidden).copy()
    ).to(device)

    def fp8_linear(name: str, x: torch.Tensor) -> torch.Tensor:
        weight = load(args.source_shard, name + ".weight").float().to(device)
        scale = load(args.source_shard, name + ".scale").float().to(device)
        weight *= scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
        return x.float() @ weight.T

    def bf16_linear(name: str, x: torch.Tensor) -> torch.Tensor:
        return x.float() @ load(args.source_shard, name).float().to(device).T

    e2m1 = torch.tensor(E2M1_VALUES, device=device)

    def nvfp4_linear(name: str, x: torch.Tensor) -> torch.Tensor:
        packed = load(args.source_shard, name + ".weight").to(device)
        codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(1)
        weight = e2m1[codes.long()]
        scale = load(args.source_shard, name + ".weight_scale").float().to(device)
        global_scale = load(args.source_shard, name + ".weight_scale_2").item()
        weight *= scale.repeat_interleave(16, 1) * global_scale
        return x.float() @ weight.T

    def rms_norm(x: torch.Tensor, name: str) -> torch.Tensor:
        weight = load(args.source_shard, name).float().to(device)
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight

    def hyper_prepare(
        x: torch.Tensor, name: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = x.flatten(1)
        function = load(args.source_shard, name + "_fn").float().to(device)
        base = load(args.source_shard, name + "_base").float().to(device)
        scale = load(args.source_shard, name + "_scale").float().to(device)
        mixed = F.linear(flat, function) * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + eps
        )
        pre = (
            torch.sigmoid(
                mixed[:, :streams_count] * scale[0] + base[:streams_count]
            )
            + hc_eps
        )
        post = 2 * torch.sigmoid(
            mixed[:, streams_count : 2 * streams_count] * scale[1]
            + base[streams_count : 2 * streams_count]
        )
        combination = (
            mixed[:, 2 * streams_count :] * scale[2]
            + base[2 * streams_count :]
        ).view(-1, streams_count, streams_count)
        combination = combination.softmax(-1) + hc_eps
        combination /= combination.sum(-2, keepdim=True) + hc_eps
        for _ in range(1, sinkhorn_iters):
            combination /= combination.sum(-1, keepdim=True) + hc_eps
            combination /= combination.sum(-2, keepdim=True) + hc_eps
        collapsed = torch.sum(pre.unsqueeze(-1) * x, dim=1)
        return collapsed, post, combination

    def hyper_apply(
        residual: torch.Tensor,
        sublayer: torch.Tensor,
        post: torch.Tensor,
        combination: torch.Tensor,
    ) -> torch.Tensor:
        return post.unsqueeze(-1) * sublayer.unsqueeze(1) + torch.einsum(
            "sij,sid->sjd", combination, residual
        )

    def inv_freq(base: float, yarn: bool) -> torch.Tensor:
        frequencies = 1.0 / (
            base
            ** (
                torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device)
                / rope_dim
            )
        )
        if not yarn:
            return frequencies
        factor = config["rope_scaling"]["factor"]
        original = config["rope_scaling"]["original_max_position_embeddings"]

        def correction_dim(rotations: int) -> float:
            return rope_dim * math.log(original / (rotations * 2 * math.pi)) / (
                2 * math.log(base)
            )

        low = max(math.floor(correction_dim(config["rope_scaling"]["beta_fast"])), 0)
        high = min(
            math.ceil(correction_dim(config["rope_scaling"]["beta_slow"])),
            rope_dim - 1,
        )
        ramp = (
            (
                torch.arange(rope_dim // 2, dtype=torch.float32, device=device)
                - low
            )
            / max(high - low, 0.001)
        ).clamp(0, 1)
        extrapolation = 1 - ramp
        return frequencies / factor * (1 - extrapolation) + frequencies * extrapolation

    frequencies = inv_freq(
        config["rope_theta"] if ratio == 0 else config["compress_rope_theta"],
        ratio != 0,
    )

    def rope(
        x: torch.Tensor, positions: torch.Tensor, inverse: bool = False
    ) -> torch.Tensor:
        angles = torch.outer(positions.float(), frequencies)
        if inverse:
            angles = -angles
        trailing = x[..., -rope_dim:].reshape(
            x.shape[0], *x.shape[1:-1], rope_dim // 2, 2
        )
        real = trailing[..., 0]
        imag = trailing[..., 1]
        view_shape = [x.shape[0]] + [1] * (real.ndim - 2) + [rope_dim // 2]
        cosine = angles.cos().view(view_shape)
        sine = angles.sin().view(view_shape)
        rotated = torch.stack(
            (real * cosine - imag * sine, real * sine + imag * cosine), dim=-1
        ).flatten(-2)
        result = x.clone()
        result[..., -rope_dim:] = rotated
        return result

    positions = torch.arange(token_ids.numel(), device=device)

    def compressor(
        x: torch.Tensor, name: str, compressed_width: int
    ) -> torch.Tensor:
        projected_width = 2 * compressed_width
        kv = bf16_linear(name + ".wkv.weight", x)
        gate = bf16_linear(name + ".wgate.weight", x)
        position_bias = load(args.source_shard, name + ".ape").float().to(device)
        windows = x.shape[0] // ratio
        cutoff = windows * ratio
        kv = kv[:cutoff].view(windows, ratio, projected_width)
        gate = gate[:cutoff].view(windows, ratio, projected_width) + position_bias
        assembled_kv = torch.zeros(windows, 2 * ratio, compressed_width, device=device)
        assembled_gate = torch.full_like(assembled_kv, float("-inf"))
        assembled_kv[:, ratio:] = kv[..., compressed_width:]
        assembled_gate[:, ratio:] = gate[..., compressed_width:]
        assembled_kv[1:, :ratio] = kv[:-1, ..., :compressed_width]
        assembled_gate[1:, :ratio] = gate[:-1, ..., :compressed_width]
        compressed = (assembled_kv * assembled_gate.softmax(dim=1)).sum(dim=1)
        compressed = rms_norm(compressed, name + ".norm.weight")
        return rope(compressed, torch.arange(windows, device=device) * ratio)

    attention_input, attention_post, attention_combination = hyper_prepare(
        streams, prefix + ".hc_attn"
    )
    attention_input = rms_norm(attention_input, prefix + ".attn_norm.weight")
    q_a = fp8_linear(prefix + ".attn.wq_a", attention_input)
    q_residual = rms_norm(q_a, prefix + ".attn.q_norm.weight")
    query = fp8_linear(prefix + ".attn.wq_b", q_residual).view(
        -1, heads, head_dim
    )
    query *= torch.rsqrt(query.square().mean(-1, keepdim=True) + eps)
    query = rope(query, positions)
    kv = fp8_linear(prefix + ".attn.wkv", attention_input)
    kv = rms_norm(kv, prefix + ".attn.kv_norm.weight")
    kv = rope(kv, positions)

    selected = [torch.empty(0, dtype=torch.long, device=device)] * token_ids.numel()
    main_compressed = None
    if ratio == 4:
        index_heads = config["index_n_heads"]
        index_dim = config["index_head_dim"]
        main_compressed = compressor(
            attention_input, prefix + ".attn.compressor", head_dim
        )
        index_query = fp8_linear(
            prefix + ".attn.indexer.wq_b", q_residual
        ).view(-1, index_heads, index_dim)
        index_query = rope(index_query, positions)
        index_compressed = compressor(
            attention_input, prefix + ".attn.indexer.compressor", index_dim
        )
        head_weights = bf16_linear(
            prefix + ".attn.indexer.weights_proj.weight", attention_input
        )
        selected = []
        for position in range(token_ids.numel()):
            causal_length = min(
                index_compressed.shape[0], (position + 1) // ratio
            )
            if causal_length == 0:
                selected.append(
                    torch.empty(0, dtype=torch.long, device=device)
                )
                continue
            dots = torch.einsum(
                "hd,td->ht",
                index_query[position],
                index_compressed[:causal_length],
            )
            scores = (
                F.relu(dots / math.sqrt(index_dim))
                * head_weights[position].unsqueeze(-1)
                / math.sqrt(index_heads)
            ).sum(dim=0)
            selected.append(
                scores.argsort(descending=True)[
                    : min(config["index_topk"], causal_length)
                ]
            )

    sink = load(args.source_shard, prefix + ".attn.attn_sink").float().to(device)
    attended = torch.empty_like(query)
    for position in range(token_ids.numel()):
        begin = 0 if ratio != 0 else max(0, position + 1 - config["sliding_window"])
        visible = kv[begin : position + 1]
        if main_compressed is not None and selected[position].numel() != 0:
            visible = torch.cat(
                (visible, main_compressed[selected[position]]), dim=0
            )
        scores = torch.einsum("hd,td->ht", query[position], visible) / math.sqrt(
            head_dim
        )
        maximum = torch.maximum(scores.max(-1).values, sink)
        exponentials = torch.exp(scores - maximum.unsqueeze(-1))
        denominator = exponentials.sum(-1) + torch.exp(sink - maximum)
        attended[position] = torch.einsum(
            "ht,td->hd", exponentials, visible
        ) / denominator.unsqueeze(-1)
    attended = rope(attended, positions, inverse=True)
    group_width = heads * head_dim // groups
    attended_groups = attended.flatten(1).view(-1, groups, group_width)
    o_a_weight = load(args.source_shard, prefix + ".attn.wo_a.weight").float().to(
        device
    )
    o_a_scale = load(args.source_shard, prefix + ".attn.wo_a.scale").float().to(
        device
    )
    o_a_weight *= o_a_scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    o_a_weight = o_a_weight.view(groups, o_rank, group_width)
    grouped = torch.einsum(
        "sgd,grd->sgr", attended_groups, o_a_weight
    ).flatten(1)
    attention_output = fp8_linear(prefix + ".attn.wo_b", grouped)
    after_attention = hyper_apply(
        streams, attention_output, attention_post, attention_combination
    )

    ffn_input, ffn_post, ffn_combination = hyper_prepare(
        after_attention, prefix + ".hc_ffn"
    )
    ffn_input = rms_norm(ffn_input, prefix + ".ffn_norm.weight")
    gate_logits = bf16_linear(prefix + ".ffn.gate.weight", ffn_input)
    gate_scores = torch.sqrt(F.softplus(gate_logits))
    if args.layer < config["num_hash_layers"]:
        token_to_expert = load(
            args.source_shard, prefix + ".ffn.gate.tid2eid"
        ).long().to(device)
        indices = token_to_expert[token_ids]
    else:
        gate_bias = load(args.source_shard, prefix + ".ffn.gate.bias").float().to(
            device
        )
        indices = (gate_scores + gate_bias).topk(top_k, dim=-1).indices
    route_weights = gate_scores.gather(1, indices)
    route_weights = route_weights / route_weights.sum(-1, keepdim=True) * route_scale
    routed = torch.zeros_like(ffn_input)
    for expert in indices.unique().tolist():
        token_rows, route_slots = torch.where(indices == expert)
        expert_input = ffn_input[token_rows]
        expert_prefix = f"{prefix}.ffn.experts.{expert}"
        gate = nvfp4_linear(expert_prefix + ".w1", expert_input)
        up = nvfp4_linear(expert_prefix + ".w3", expert_input)
        activated = F.silu(gate.clamp(max=swiglu_limit)) * up.clamp(
            min=-swiglu_limit, max=swiglu_limit
        )
        down = nvfp4_linear(expert_prefix + ".w2", activated)
        routed.index_add_(
            0,
            token_rows,
            down * route_weights[token_rows, route_slots].unsqueeze(-1),
        )
    shared_gate = fp8_linear(prefix + ".ffn.shared_experts.w1", ffn_input)
    shared_up = fp8_linear(prefix + ".ffn.shared_experts.w3", ffn_input)
    shared = fp8_linear(
        prefix + ".ffn.shared_experts.w2", F.silu(shared_gate) * shared_up
    )
    final_streams = hyper_apply(
        after_attention, routed + shared, ffn_post, ffn_combination
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        final_streams=final_streams.cpu().numpy(),
        indices=indices.cpu().numpy(),
        route_weights=route_weights.cpu().numpy(),
    )
    final_streams.cpu().numpy().astype("<f4").tofile(
        args.output.with_suffix(".bin")
    )
    print(
        json.dumps(
            {
                "layer": args.layer,
                "ratio": ratio,
                "experts": sorted(indices.unique().tolist()),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
