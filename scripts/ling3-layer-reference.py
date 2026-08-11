#!/usr/bin/env python3
"""Generate a deterministic Ling 3 Tiny BF16 layer-zero reference.

The equations follow the checkpoint's ``modeling_bailing_moe_v3.py`` and the
installed FLA KDA naive recurrence. They are written directly with PyTorch CPU
operations so the artifact remains independent of Eider's CUDA kernels and of
FLA/Triton support for the local GPU architecture.
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
HEADS = 16
HEAD_DIM = 128
PROJECTION = HEADS * HEAD_DIM
INTERMEDIATE = 4608
EPS = 1.0e-6
LOWER_BOUND = -5.0


class Checkpoint:
    def __init__(self, root: Path):
        self.root = root
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]

    def tensor(self, name: str) -> torch.Tensor:
        shard = self.root / self.weight_map[name]
        with safe_open(shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def linear(self, value: torch.Tensor, prefix: str) -> torch.Tensor:
        weight = self.tensor(f"{prefix}.weight")
        if weight.dtype == torch.bfloat16:
            return F.linear(value.to(weight.dtype), weight)
        scale = self.tensor(f"{prefix}.weight_scale_inv").float()
        scale = scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        dequantized = weight.float() * scale[: weight.shape[0], : weight.shape[1]]
        return F.linear(value.float(), dequantized)


def rms_norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    dtype = value.dtype
    variance = value.float().square().mean(dim=-1, keepdim=True)
    normalized = value.float() * torch.rsqrt(variance + EPS)
    return weight * normalized.to(dtype)


def short_conv_first_token(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # A zero initial cache leaves only the newest (last) causal-convolution tap.
    mixed = value.float() * weight[:, 0, -1].float()
    return F.silu(mixed).to(value.dtype)


def recurrent_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query = query.reshape(HEADS, HEAD_DIM).float()
    key = key.reshape(HEADS, HEAD_DIM).float()
    value_dtype = value.dtype
    value = value.reshape(HEADS, HEAD_DIM).float()
    raw_gate = raw_gate.reshape(HEADS, HEAD_DIM).float()
    query = query / torch.sqrt(query.square().sum(dim=-1, keepdim=True) + 1.0e-6)
    key = key / torch.sqrt(key.square().sum(dim=-1, keepdim=True) + 1.0e-6)
    gate = LOWER_BOUND * torch.sigmoid(
        a_log.float().exp().unsqueeze(-1)
        * (raw_gate + dt_bias.float().reshape(HEADS, HEAD_DIM))
    )
    state = torch.zeros(HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float32)
    state *= gate.exp().unsqueeze(-1)
    prediction = torch.einsum("hkv,hk->hv", state, key)
    delta = beta.float().unsqueeze(-1) * (value - prediction)
    state += torch.einsum("hk,hv->hkv", key, delta)
    output = torch.einsum("hk,hkv->hv", query * HEAD_DIM**-0.5, state)
    return output.to(value_dtype), query, key, gate


def sigmoid_gated_rms_norm(
    value: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    value = value.reshape(HEADS, HEAD_DIM)
    gate = gate.reshape(HEADS, HEAD_DIM)
    variance = value.float().square().mean(dim=-1, keepdim=True)
    normalized = value.float() * torch.rsqrt(variance + EPS)
    output = normalized * weight.float() * torch.sigmoid(gate.float())
    return output.to(value.dtype).reshape(PROJECTION)


def generate(checkpoint: Checkpoint) -> dict[str, torch.Tensor]:
    prefix = "model.layers.0"
    attention = f"{prefix}.attention"
    input_value = torch.tensor(
        [((index % 97) - 48) / 96.0 for index in range(HIDDEN)],
        dtype=torch.bfloat16,
    )
    normed = rms_norm(
        input_value, checkpoint.tensor(f"{prefix}.input_layernorm.weight")
    )
    query_raw = checkpoint.linear(normed, f"{attention}.q_proj")
    key_raw = checkpoint.linear(normed, f"{attention}.k_proj")
    value_raw = checkpoint.linear(normed, f"{attention}.v_proj")
    query_conv = short_conv_first_token(
        query_raw, checkpoint.tensor(f"{attention}.q_conv1d.weight")
    )
    key_conv = short_conv_first_token(
        key_raw, checkpoint.tensor(f"{attention}.k_conv1d.weight")
    )
    value_conv = short_conv_first_token(
        value_raw, checkpoint.tensor(f"{attention}.v_conv1d.weight")
    )
    raw_gate = checkpoint.linear(normed, f"{attention}.f_proj")
    beta = checkpoint.linear(normed, f"{attention}.b_proj").float().sigmoid()
    recurrent, query, key, gate = recurrent_kda(
        query_conv,
        key_conv,
        value_conv,
        raw_gate,
        beta,
        checkpoint.tensor(f"{attention}.A_log"),
        checkpoint.tensor(f"{attention}.dt_bias"),
    )
    output_gate = checkpoint.linear(normed, f"{attention}.g_proj")
    gated = sigmoid_gated_rms_norm(
        recurrent, output_gate, checkpoint.tensor(f"{attention}.o_norm.weight")
    )
    attention_output = checkpoint.linear(gated, f"{attention}.o_proj")
    post_attention = input_value + attention_output
    ffn_input = rms_norm(
        post_attention, checkpoint.tensor(f"{prefix}.post_attention_layernorm.weight")
    )
    mlp_gate = checkpoint.linear(ffn_input, f"{prefix}.mlp.gate_proj")
    mlp_up = checkpoint.linear(ffn_input, f"{prefix}.mlp.up_proj")
    mlp_output = checkpoint.linear(F.silu(mlp_gate) * mlp_up, f"{prefix}.mlp.down_proj")
    output = post_attention + mlp_output
    return {
        "input": input_value.float().contiguous(),
        "normed": normed.float().contiguous(),
        "query": query.flatten().contiguous(),
        "key": key.flatten().contiguous(),
        "value": value_conv.float().contiguous(),
        "gate": gate.flatten().contiguous(),
        "beta": beta.flatten().contiguous(),
        "recurrent_output": recurrent.float().flatten().contiguous(),
        "gated_output": gated.float().contiguous(),
        "attention_output": attention_output.float().contiguous(),
        "post_attention": post_attention.float().contiguous(),
        "ffn_input": ffn_input.float().contiguous(),
        "mlp_output": mlp_output.float().contiguous(),
        "output": output.float().contiguous(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tensors = generate(Checkpoint(args.model_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata={"format": "eider-ling3-layer-reference-v1"})
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
