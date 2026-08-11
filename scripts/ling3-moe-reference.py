#!/usr/bin/env python3
"""Generate an independent Ling 3 Tiny layer-1 MoE reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


HIDDEN = 1536
EXPERTS = 128
TOP_K = 8
GROUPS = 8
TOP_GROUPS = 4
SCALE = 2.5


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


def expert(checkpoint: Checkpoint, prefix: str, value: torch.Tensor) -> torch.Tensor:
    gate = checkpoint.linear(value, f"{prefix}.gate_proj")
    up = checkpoint.linear(value, f"{prefix}.up_proj")
    return checkpoint.linear(F.silu(gate) * up, f"{prefix}.down_proj")


def generate(checkpoint: Checkpoint, layer: int) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer}.mlp"
    value = torch.tensor(
        [((index * 17) % 101 - 50) / 100.0 for index in range(HIDDEN)],
        dtype=torch.bfloat16,
    )
    router_weight = checkpoint.tensor(f"{prefix}.gate.weight")
    logits = F.linear(value.float(), router_weight.float())
    probabilities = logits.sigmoid()
    routing_scores = probabilities + checkpoint.tensor(f"{prefix}.gate.expert_bias")
    group_scores = routing_scores.reshape(GROUPS, EXPERTS // GROUPS).topk(2, dim=-1).values.sum(-1)
    selected_groups = group_scores.topk(TOP_GROUPS, sorted=False).indices
    mask = torch.zeros(GROUPS, dtype=torch.bool)
    mask[selected_groups] = True
    masked = routing_scores.masked_fill(
        ~mask.repeat_interleave(EXPERTS // GROUPS), float("-inf")
    )
    indices = masked.topk(TOP_K).indices
    weights = probabilities[indices]
    weights = weights / (weights.sum() + 1.0e-20) * SCALE
    routed = torch.zeros(HIDDEN, dtype=torch.float32)
    for index, weight in zip(indices.tolist(), weights.tolist()):
        routed += expert(checkpoint, f"{prefix}.experts.{index}", value).float() * weight
    shared = expert(checkpoint, f"{prefix}.shared_experts", value).float()
    return {
        "input": value.float().contiguous(),
        "logits": logits.float().contiguous(),
        "indices": indices.float().contiguous(),
        "weights": weights.float().contiguous(),
        "output": (routed + shared).float().contiguous(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, default=1)
    args = parser.parse_args()
    tensors = generate(Checkpoint(args.model_dir), args.layer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata={"format": "eider-ling3-moe-reference-v1"})
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
