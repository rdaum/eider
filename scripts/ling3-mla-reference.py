#!/usr/bin/env python3
"""Generate a multi-token Ling 3 Tiny MLA attention reference on CPU."""

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
Q_RANK = 256
KV_RANK = 512
NOPE = 128
ROPE = 64
QK = 192
VALUE = 128
EPS = 1.0e-6
ROPE_THETA = 6_000_000.0


class Checkpoint:
    def __init__(self, root: Path):
        self.root = root
        self.weight_map = json.loads(
            (root / "model.safetensors.index.json").read_text()
        )["weight_map"]

    def tensor(self, name: str) -> torch.Tensor:
        with safe_open(
            self.root / self.weight_map[name], framework="pt", device="cpu"
        ) as handle:
            return handle.get_tensor(name)

    def linear(self, value: torch.Tensor, prefix: str) -> torch.Tensor:
        weight = self.tensor(f"{prefix}.weight")
        if weight.dtype == torch.bfloat16:
            return F.linear(value.to(weight.dtype), weight)
        scales = self.tensor(f"{prefix}.weight_scale_inv").float()
        scales = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
        return F.linear(value.float(), weight.float() * scales)


def rms_norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    variance = value.float().square().mean()
    return value.float() * torch.rsqrt(variance + EPS) * weight.float()


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


def generate(checkpoint: Checkpoint, layer: int, tokens: int) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer}.attention"
    inputs = torch.stack(
        [
            torch.tensor(
                [((index * 17 + token * 11) % 101 - 50) / 100.0 for index in range(HIDDEN)],
                dtype=torch.bfloat16,
            )
            for token in range(tokens)
        ]
    )
    key_cache: list[torch.Tensor] = []
    value_cache: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    attentions: list[torch.Tensor] = []
    for position, hidden in enumerate(inputs):
        q_a = checkpoint.linear(hidden, f"{prefix}.q_a_proj")
        q_a = rms_norm(q_a, checkpoint.tensor(f"{prefix}.q_a_layernorm.weight"))
        query = checkpoint.linear(q_a, f"{prefix}.q_b_proj").reshape(HEADS, QK)
        kv_a = checkpoint.linear(hidden, f"{prefix}.kv_a_proj_with_mqa")
        compressed = rms_norm(
            kv_a[:KV_RANK], checkpoint.tensor(f"{prefix}.kv_a_layernorm.weight")
        )
        shared_rope = kv_a[KV_RANK:]
        kv = checkpoint.linear(compressed, f"{prefix}.kv_b_proj").reshape(
            HEADS, NOPE + VALUE
        )
        key = torch.cat((kv[:, :NOPE], shared_rope.expand(HEADS, ROPE)), dim=-1)
        value = kv[:, NOPE:]
        query = rotate_interleaved(query, position)
        key = rotate_interleaved(key, position)
        key_cache.append(key.float())
        value_cache.append(value.float())
        keys = torch.stack(key_cache, dim=1)
        values = torch.stack(value_cache, dim=1)
        scores = torch.einsum("hd,htd->ht", query.float(), keys) * QK**-0.5
        probabilities = scores.softmax(dim=-1)
        attention = torch.einsum("ht,htd->hd", probabilities, values)
        gate = checkpoint.linear(hidden, f"{prefix}.g_proj").float().sigmoid()
        gated = attention * gate.unsqueeze(-1)
        outputs.append(checkpoint.linear(gated.flatten(), f"{prefix}.dense"))
        attentions.append(attention.flatten())
    return {
        "input": inputs.float().contiguous(),
        "attention": torch.stack(attentions).float().contiguous(),
        "output": torch.stack(outputs).float().contiguous(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=3)
    args = parser.parse_args()
    tensors = generate(Checkpoint(args.model_dir), args.layer, args.tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata={"format": "eider-ling3-mla-reference-v1"})
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
