#!/usr/bin/env python3
"""Generate LFM3 fixtures or evaluate a checkpoint with independent PyTorch equations.

Mirrors LiquidAI/LFM3-1.7B's modeling_lfm3.py architecture. Uses BF16 operator boundaries,
F32 accumulation, and blockwise W4A4 activation quantization.
Requires torch and safetensors. Writes only to --output, never to a checkpoint.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


def fixture(root, nvfp4):
    torch.manual_seed(347)
    config = dict(model_type="lfm3", hidden_size=256, intermediate_size=512,
                  num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                  head_dim=128, vocab_size=256, max_position_embeddings=128,
                  norm_eps=1e-5, tie_word_embeddings=True, conv_L_cache=7,
                  conv_bias=False, use_sandwich_norm=True, sliding_window=5,
                  layer_types=["conv", "full_attention", "sliding_attention", "conv"],
                  rope_parameters={"full_attention": dict(rope_type="default", rope_theta=1e6, partial_rotary_factor=.5),
                                   "sliding_attention": dict(rope_type="default", rope_theta=10000)})
    weights = {}

    # Discrete dyadic weights keep projection sums reproducible across BLAS
    # reduction orders, isolating architecture and quantizer rounding errors.
    def bf(name, shape, norm=False):
        weights[name + ".weight"] = ((1 + .1 * torch.randn(shape)) if norm else torch.randint(-3, 4, shape).float() / 64).bfloat16()

    def linear(name, rows, cols, scale):
        if not nvfp4:
            bf(name, (rows, cols))
            return
        codes = torch.randint(0, 16, (rows, cols), dtype=torch.uint8)
        weights[name + ".weight"] = codes[:, ::2] | (codes[:, 1::2] << 4)
        weights[name + ".weight_scale"] = (.5 + torch.rand(rows, cols // 16)).to(torch.float8_e4m3fn)
        weights[name + ".weight_scale_2"] = torch.tensor(scale)
        weights[name + ".input_scale"] = torch.tensor(.013 + scale)

    bf("model.embed_tokens", (256, 256))
    bf("model.embedding_norm", (256,), True)
    for i, kind in enumerate(config["layer_types"]):
        p = f"model.layers.{i}"
        for name in ["operator_norm", "ffn_norm", "post_operator_norm", "post_ffn_norm"]:
            bf(p + "." + name, (256,), True)
            # Small learned sandwich scales resemble the released checkpoint.
            scale = .02 if name.startswith("post_") else .25
            weights[p + "." + name + ".weight"] *= scale
        if kind == "conv":
            bf(p + ".conv.in_proj", (768, 256))
            bf(p + ".conv.out_proj", (256, 256))
            bf(p + ".conv.conv", (256, 1, 7))
        else:
            for name, rows in [("q_proj", 256), ("k_proj", 128), ("v_proj", 128), ("o_proj", 256), ("gate_proj", 2)]:
                bf(p + ".self_attn." + name, (rows, 256))
            for name in ["q_norm", "k_norm"]:
                bf(p + ".self_attn." + name, (128,), True)
        for j, (name, rows, cols) in enumerate([("w1", 512, 256), ("w3", 512, 256), ("w2", 256, 512)]):
            linear(p + ".feed_forward." + name, rows, cols, .012 + .003*j + .001*i)
    save_file(weights, root / "model.safetensors")
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")


TABLE = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])


def quantize(x, global_scale):
    blocks = (x / global_scale).reshape(*x.shape[:-1], -1, 16)
    scales = (blocks.abs().amax(-1, keepdim=True) / 6).clamp(max=448).to(torch.float8_e4m3fn).float()
    z = blocks / torch.where(scales == 0, torch.ones_like(scales), scales)
    # E2M1 rounds ties to the even encoded value.
    magnitude = z.abs().clamp(max=6)
    levels = TABLE[:8]
    distance = (magnitude[..., None] - levels).abs()
    minimum = distance.amin(-1, keepdim=True)
    ranks = torch.arange(8) + (torch.arange(8) % 2) * 8
    code = torch.where(distance == minimum, ranks, 100).argmin(-1)
    return (levels[code] * z.sign() * scales).reshape_as(x)


def bf(x):
    return x.bfloat16().float()


def evaluate(root, tokens):
    config = json.loads((root / "config.json").read_text())
    shard = safe_open(root / "model.safetensors", framework="pt", device="cpu")

    def tensor(name):
        return shard.get_tensor(name)

    def linear(x, name):
        w = tensor(name + ".weight")
        alpha = 1.0
        if w.dtype == torch.uint8:
            codes = torch.stack([w & 15, w >> 4], -1).flatten(-2)
            scales = tensor(name + ".weight_scale").float().repeat_interleave(16, -1)
            w = TABLE[codes.long()] * scales
            input_scale = tensor(name + ".input_scale").float()
            x = quantize(x, input_scale)
            # NVFP4 MMA accumulates block-scaled products before the tensor scales.
            alpha = input_scale * tensor(name + ".weight_scale_2").float()
        return bf(F.linear(x, w.float()) * alpha)

    def norm(x, name):
        return bf(bf(x * torch.rsqrt(x.square().mean(-1, keepdim=True) + config["norm_eps"])) * tensor(name + ".weight").float())

    h, dim = config["hidden_size"], config["head_dim"]
    heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
    length = len(tokens)
    position = torch.arange(length)
    x = tensor("model.embed_tokens.weight")[tokens].float()
    for i, kind in enumerate(config["layer_types"]):
        p = f"model.layers.{i}"
        z = norm(x, p + ".operator_norm")
        if kind == "conv":
            b, c, u = linear(z, p + ".conv.in_proj").chunk(3, -1)
            conv = F.conv1d(bf(b*u).T[None], tensor(p + ".conv.conv.weight").float(), padding=config["conv_L_cache"]-1, groups=h)
            y = linear(bf(c * bf(conv[0, :, :length].T)), p + ".conv.out_proj")
        else:
            a = p + ".self_attn"
            q = norm(linear(z, a + ".q_proj").reshape(length, heads, dim), a + ".q_norm")
            k = norm(linear(z, a + ".k_proj").reshape(length, kv_heads, dim), a + ".k_norm")
            v = linear(z, a + ".v_proj").reshape(length, kv_heads, dim).transpose(0, 1)
            r = config["rope_parameters"][kind]
            freq = r["rope_theta"] ** (-torch.arange(0, dim, 2).float() / dim)
            freq[int(r.get("partial_rotary_factor", 1)*dim)//2:] = 0
            angle = position[:, None] * freq
            angle = torch.cat([angle, angle], -1)[:, None, :]

            def rope(t):
                a, b = t.chunk(2, -1)
                return bf(t*bf(angle.cos()) + torch.cat([-b, a], -1)*bf(angle.sin())).transpose(0, 1)

            q, k = rope(q), rope(k).repeat_interleave(heads//kv_heads, 0)
            v = v.repeat_interleave(heads//kv_heads, 0)
            scores = q @ k.transpose(-1, -2) / dim**.5
            mask = position[None, :] > position[:, None]
            if kind == "sliding_attention":
                mask |= position[None, :] <= position[:, None] - config["sliding_window"]
            scores.masked_fill_(mask, -torch.inf)
            attended = bf(scores.softmax(-1) @ v).transpose(0, 1)
            gate = bf(linear(z, a + ".gate_proj").sigmoid())[..., None]
            y = linear(bf(attended*gate).reshape(length, h), a + ".o_proj")
        if config["use_sandwich_norm"]:
            y = norm(y, p + ".post_operator_norm")
        x = bf(x + y)
        z = norm(x, p + ".ffn_norm")
        f = p + ".feed_forward"
        y = linear(bf(bf(F.silu(linear(z, f + ".w1")))*linear(z, f + ".w3")), f + ".w2")
        if config["use_sandwich_norm"]:
            y = norm(y, p + ".post_ffn_norm")
        x = bf(x + y)
    return linear(norm(x, "model.embedding_norm"), "model.embed_tokens")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--nvfp4", action="store_true")
    parser.add_argument("--tokens", type=int, nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.checkpoint or args.output
    if args.checkpoint is None:
        fixture(root, args.nvfp4)
    tokens = args.tokens or [13, 2, 97, 255, 6, 31, 77, 14, 53, 11, 9, 8, 19, 71, 55, 8, 31, 62, 4]
    with torch.inference_mode():
        logits = evaluate(root, tokens)
    (args.output / "reference.json").write_text(json.dumps(dict(tokens=tokens, logits=logits.tolist())) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
