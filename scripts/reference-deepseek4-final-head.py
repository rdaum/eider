#!/usr/bin/env python3
"""Generate an independent DeepSeek V4 final-head reference.

The input is a headerless little-endian f32 file containing one or more
consecutive [4, hidden] mHC rows. Only the final row is projected, matching the
serving path.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def load_tensor(model_dir: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    shard = model_dir / weight_map[name]
    with safe_open(shard, framework="pt", device="cpu") as checkpoint:
        return checkpoint.get_tensor(name)


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    tensor.detach().float().cpu().numpy().astype("<f4").tofile(path)


def main() -> None:
    args = parse_args()
    config = json.loads((args.model_dir / "config.json").read_text())
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    hidden = config["hidden_size"]
    streams = np.fromfile(args.input, dtype="<f4")
    row_width = config["hc_mult"] * hidden
    if streams.size == 0 or streams.size % row_width != 0:
        raise ValueError(
            f"input contains {streams.size} floats; expected a non-zero multiple of {row_width}"
        )
    streams = torch.from_numpy(streams.reshape(-1, config["hc_mult"], hidden)[-1:].copy())

    function = load_tensor(args.model_dir, weight_map, "hc_head_fn").float()
    base = load_tensor(args.model_dir, weight_map, "hc_head_base").float()
    scale = load_tensor(args.model_dir, weight_map, "hc_head_scale").float()
    norm_weight = load_tensor(args.model_dir, weight_map, "norm.weight").float()
    head_weight = load_tensor(args.model_dir, weight_map, "head.weight").float()

    flat = streams.flatten(1).float()
    inverse_rms = torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + config["rms_norm_eps"]
    )
    mixes = F.linear(flat, function) * inverse_rms
    pre = torch.sigmoid(mixes * scale + base) + config["hc_eps"]
    collapsed = torch.sum(pre.unsqueeze(-1) * streams.float(), dim=1)
    normed = (
        collapsed
        * torch.rsqrt(
            collapsed.square().mean(-1, keepdim=True) + config["rms_norm_eps"]
        )
        * norm_weight
    )
    logits = F.linear(normed, head_weight)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_f32(args.output_dir / "input.bin", streams)
    write_f32(args.output_dir / "collapsed.bin", collapsed)
    write_f32(args.output_dir / "normed.bin", normed)
    write_f32(args.output_dir / "logits.bin", logits)
    top = torch.topk(logits[0], 10)
    print(
        json.dumps(
            {
                "input_rows": streams.shape[0],
                "top_token_ids": top.indices.tolist(),
                "top_logits": top.values.tolist(),
            }
        )
    )


if __name__ == "__main__":
    main()
