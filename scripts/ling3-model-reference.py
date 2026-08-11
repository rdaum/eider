#!/usr/bin/env python3
"""Generate an independent complete Ling 3 Tiny CPU decode reference.

The equations follow the checkpoint's published model implementation, while
block-FP8 weights are explicitly dequantised. The script deliberately avoids
Eider kernels and FLA/Triton so it runs on the host CPU of a DGX Spark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


HIDDEN = 1536
LAYERS = 24
HEADS = 16
HEAD_DIM = 128
PROJECTION = HEADS * HEAD_DIM
Q_RANK = 256
KV_RANK = 512
NOPE = 128
ROPE = 64
QK = 192
VALUE = 128
EXPERTS = 128
TOP_K = 8
GROUPS = 8
TOP_GROUPS = 4
ROUTED_SCALE = 2.5
EPS = 1.0e-6
LOWER_BOUND = -5.0
ROPE_THETA = 6_000_000.0


class Checkpoint:
    def __init__(self, root: Path):
        self.root = root
        self.weight_map = json.loads(
            (root / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.handles = {
            filename: safe_open(root / filename, framework="pt", device="cpu")
            for filename in set(self.weight_map.values())
        }

    def tensor(self, name: str) -> torch.Tensor:
        return self.handles[self.weight_map[name]].get_tensor(name)

    def linear(self, value: torch.Tensor, prefix: str) -> torch.Tensor:
        weight = self.tensor(f"{prefix}.weight")
        if weight.dtype == torch.bfloat16:
            return F.linear(value.to(weight.dtype), weight)
        scales = self.tensor(f"{prefix}.weight_scale_inv").float()
        scales = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
        dequantised = weight.float() * scales[: weight.shape[0], : weight.shape[1]]
        return F.linear(value.float(), dequantised)


def rms_norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    variance = value.float().square().mean(dim=-1, keepdim=True)
    return value.float() * torch.rsqrt(variance + EPS) * weight.float()


def mlp(checkpoint: Checkpoint, prefix: str, value: torch.Tensor) -> torch.Tensor:
    gate = checkpoint.linear(value, f"{prefix}.gate_proj")
    up = checkpoint.linear(value, f"{prefix}.up_proj")
    return checkpoint.linear(F.silu(gate) * up, f"{prefix}.down_proj")


def moe(checkpoint: Checkpoint, prefix: str, value: torch.Tensor) -> torch.Tensor:
    logits = F.linear(value.float(), checkpoint.tensor(f"{prefix}.gate.weight").float())
    probabilities = logits.sigmoid()
    routing = probabilities + checkpoint.tensor(f"{prefix}.gate.expert_bias").float()
    group_scores = routing.reshape(GROUPS, EXPERTS // GROUPS).topk(2, dim=-1).values.sum(-1)
    selected_groups = group_scores.topk(TOP_GROUPS, sorted=False).indices
    group_mask = torch.zeros(GROUPS, dtype=torch.bool)
    group_mask[selected_groups] = True
    masked = routing.masked_fill(
        ~group_mask.repeat_interleave(EXPERTS // GROUPS), float("-inf")
    )
    indices = masked.topk(TOP_K).indices
    weights = probabilities[indices]
    weights = weights / (weights.sum() + 1.0e-20) * ROUTED_SCALE
    output = torch.zeros(HIDDEN, dtype=torch.float32)
    for index, weight in zip(indices.tolist(), weights.tolist()):
        output += mlp(checkpoint, f"{prefix}.experts.{index}", value).float() * weight
    output += mlp(checkpoint, f"{prefix}.shared_experts", value).float()
    return output


class KdaState:
    def __init__(self):
        self.conv = torch.zeros(3, PROJECTION, 3, dtype=torch.float32)
        self.recurrent = torch.zeros(HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float32)


def causal_conv(
    value: torch.Tensor, weight: torch.Tensor, state: torch.Tensor
) -> torch.Tensor:
    mixed = (state * weight[:, 0, :3].float()).sum(-1)
    mixed += value.float() * weight[:, 0, 3].float()
    state[:, :2] = state[:, 1:].clone()
    state[:, 2] = value.float()
    return F.silu(mixed)


def kda_attention(
    checkpoint: Checkpoint, prefix: str, value: torch.Tensor, state: KdaState
) -> torch.Tensor:
    query = causal_conv(
        checkpoint.linear(value, f"{prefix}.q_proj"),
        checkpoint.tensor(f"{prefix}.q_conv1d.weight"),
        state.conv[0],
    ).reshape(HEADS, HEAD_DIM)
    key = causal_conv(
        checkpoint.linear(value, f"{prefix}.k_proj"),
        checkpoint.tensor(f"{prefix}.k_conv1d.weight"),
        state.conv[1],
    ).reshape(HEADS, HEAD_DIM)
    recurrent_value = causal_conv(
        checkpoint.linear(value, f"{prefix}.v_proj"),
        checkpoint.tensor(f"{prefix}.v_conv1d.weight"),
        state.conv[2],
    ).reshape(HEADS, HEAD_DIM)
    query = query / torch.sqrt(query.square().sum(-1, keepdim=True) + 1.0e-6)
    key = key / torch.sqrt(key.square().sum(-1, keepdim=True) + 1.0e-6)
    raw_gate = checkpoint.linear(value, f"{prefix}.f_proj").reshape(HEADS, HEAD_DIM)
    beta = checkpoint.linear(value, f"{prefix}.b_proj").float().sigmoid()
    a_log = checkpoint.tensor(f"{prefix}.A_log").float()
    dt_bias = checkpoint.tensor(f"{prefix}.dt_bias").float().reshape(HEADS, HEAD_DIM)
    gate = LOWER_BOUND * torch.sigmoid(a_log.exp().unsqueeze(-1) * (raw_gate + dt_bias))
    state.recurrent *= gate.exp().unsqueeze(-1)
    prediction = torch.einsum("hkv,hk->hv", state.recurrent, key)
    delta = beta.unsqueeze(-1) * (recurrent_value - prediction)
    state.recurrent += torch.einsum("hk,hv->hkv", key, delta)
    output = torch.einsum(
        "hk,hkv->hv", query * HEAD_DIM**-0.5, state.recurrent
    )
    output_weight = checkpoint.tensor(f"{prefix}.o_norm.weight").float()
    output = rms_norm(output, output_weight)
    output_gate = checkpoint.linear(value, f"{prefix}.g_proj").reshape(HEADS, HEAD_DIM)
    output = output * output_gate.float().sigmoid()
    return checkpoint.linear(output.flatten(), f"{prefix}.o_proj")


class MlaState:
    def __init__(self):
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []


def rotate_interleaved(value: torch.Tensor, position: int) -> torch.Tensor:
    result = value.clone()
    inv_freq = ROPE_THETA ** (-2.0 * torch.arange(ROPE // 2).float() / ROPE)
    angle = position * inv_freq
    cosine = angle.cos()
    sine = angle.sin()
    even = value[..., -ROPE::2]
    odd = value[..., -ROPE + 1 :: 2]
    result[..., -ROPE::2] = even * cosine - odd * sine
    result[..., -ROPE + 1 :: 2] = odd * cosine + even * sine
    return result


def mla_attention(
    checkpoint: Checkpoint,
    prefix: str,
    value: torch.Tensor,
    position: int,
    state: MlaState,
) -> torch.Tensor:
    query_a = checkpoint.linear(value, f"{prefix}.q_a_proj")
    query_a = rms_norm(query_a, checkpoint.tensor(f"{prefix}.q_a_layernorm.weight"))
    query = checkpoint.linear(query_a, f"{prefix}.q_b_proj").reshape(HEADS, QK)
    kv_a = checkpoint.linear(value, f"{prefix}.kv_a_proj_with_mqa")
    compressed = rms_norm(
        kv_a[:KV_RANK], checkpoint.tensor(f"{prefix}.kv_a_layernorm.weight")
    )
    shared_rope = kv_a[KV_RANK:]
    kv = checkpoint.linear(compressed, f"{prefix}.kv_b_proj").reshape(
        HEADS, NOPE + VALUE
    )
    key = torch.cat((kv[:, :NOPE], shared_rope.expand(HEADS, ROPE)), dim=-1)
    recurrent_value = kv[:, NOPE:]
    query = rotate_interleaved(query, position)
    key = rotate_interleaved(key, position)
    state.keys.append(key.float())
    state.values.append(recurrent_value.float())
    keys = torch.stack(state.keys, dim=1)
    values = torch.stack(state.values, dim=1)
    scores = torch.einsum("hd,htd->ht", query.float(), keys) * QK**-0.5
    probabilities = scores.softmax(dim=-1)
    output = torch.einsum("ht,htd->hd", probabilities, values)
    gate = checkpoint.linear(value, f"{prefix}.g_proj").float().sigmoid()
    output *= gate.unsqueeze(-1)
    return checkpoint.linear(output.flatten(), f"{prefix}.dense")


def generate(checkpoint: Checkpoint, tokens: list[int]) -> dict[str, torch.Tensor]:
    kda_states = {layer: KdaState() for layer in range(LAYERS) if (layer + 1) % 4}
    mla_states = {layer: MlaState() for layer in range(LAYERS) if not (layer + 1) % 4}
    all_logits: list[torch.Tensor] = []
    all_layers: list[torch.Tensor] = []
    embeddings = checkpoint.tensor("model.word_embeddings.weight")
    for position, token in enumerate(tokens):
        hidden = embeddings[token].float()
        layer_outputs: list[torch.Tensor] = []
        for layer in range(LAYERS):
            prefix = f"model.layers.{layer}"
            normed = rms_norm(
                hidden, checkpoint.tensor(f"{prefix}.input_layernorm.weight")
            )
            if (layer + 1) % 4:
                attention = kda_attention(
                    checkpoint, f"{prefix}.attention", normed, kda_states[layer]
                )
            else:
                attention = mla_attention(
                    checkpoint,
                    f"{prefix}.attention",
                    normed,
                    position,
                    mla_states[layer],
                )
            hidden = hidden + attention.float()
            ffn_input = rms_norm(
                hidden, checkpoint.tensor(f"{prefix}.post_attention_layernorm.weight")
            )
            if layer == 0:
                ffn_output = mlp(checkpoint, f"{prefix}.mlp", ffn_input)
            else:
                ffn_output = moe(checkpoint, f"{prefix}.mlp", ffn_input)
            hidden = hidden + ffn_output.float()
            layer_outputs.append(hidden.float().clone())
        final = rms_norm(hidden, checkpoint.tensor("model.norm.weight"))
        logits = checkpoint.linear(final, "lm_head")
        all_logits.append(logits.float())
        all_layers.append(torch.stack(layer_outputs))
        print(
            f"position={position} token={token} top={int(logits.argmax())}",
            flush=True,
        )
    return {
        "tokens": torch.tensor(tokens, dtype=torch.float32),
        "layers": torch.stack(all_layers).contiguous(),
        "logits": torch.stack(all_logits).contiguous(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("tokens", nargs="+", type=int)
    args = parser.parse_args()
    tensors = generate(Checkpoint(args.model_dir), args.tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata={"format": "eider-ling3-model-reference-v1"})
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
