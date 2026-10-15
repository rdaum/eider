#!/usr/bin/env python3
"""Evaluate a ModelOpt LFM2 checkpoint on CPU, one layer/expert at a time.

Usage: python3 scripts/lfm2-checkpoint-reference.py CHECKPOINT OUTPUT_JSON TOKEN...
No downloads or checkpoint writes. Requires torch and safetensors.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.set_num_threads(8)
root, destination = Path(sys.argv[1]), Path(sys.argv[2])
tokens = [int(token) for token in sys.argv[3:]]
assert tokens, "provide token IDs"
config = json.loads((root / "config.json").read_text())
index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
shards = {name: safe_open(root / name, framework="pt", device="cpu") for name in set(index.values())}
table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])


def tensor(name):
    return shards[index[name]].get_tensor(name)


def linear(x, prefix):
    w = tensor(prefix + ".weight")
    if w.dtype == torch.uint8:
        codes = torch.stack([w & 15, w >> 4], dim=-1).flatten(-2)
        scales = tensor(prefix + ".weight_scale").float().repeat_interleave(16, dim=-1)
        w = table[codes.long()] * scales * tensor(prefix + ".weight_scale_2").float()
    return F.linear(x, w.float())


def norm(x, prefix):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + config["norm_eps"]) * tensor(prefix + ".weight").float()


h = config["hidden_size"]
heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
dim = h // heads
length = len(tokens)
x = tensor("model.embed_tokens.weight")[tokens].float()
for i, kind in enumerate(config["layer_types"]):
    p = f"model.layers.{i}"
    z = norm(x, p + ".operator_norm")
    if kind == "conv":
        b, c, u = linear(z, p + ".conv.in_proj").chunk(3, dim=-1)
        conv = F.conv1d((b * u).T[None], tensor(p + ".conv.conv.weight").float(), padding=config["conv_L_cache"] - 1, groups=h)
        y = linear(c * conv[0, :, :length].T, p + ".conv.out_proj")
    else:
        q = norm(linear(z, p + ".self_attn.q_proj").reshape(length, heads, dim), p + ".self_attn.q_layernorm")
        k = norm(linear(z, p + ".self_attn.k_proj").reshape(length, kv_heads, dim), p + ".self_attn.k_layernorm")
        v = linear(z, p + ".self_attn.v_proj").reshape(length, kv_heads, dim).transpose(0, 1)
        freq = torch.arange(length)[:, None] / (config["rope_parameters"]["rope_theta"] ** (torch.arange(0, dim, 2).float() / dim))
        angle = torch.cat([freq, freq], dim=-1)[:, None, :]
        def rope(t):
            a, b = t.chunk(2, dim=-1)
            return (t * angle.cos() + torch.cat([-b, a], dim=-1) * angle.sin()).transpose(0, 1)
        q, k = rope(q), rope(k)
        k = k.repeat_interleave(heads // kv_heads, dim=0)
        v = v.repeat_interleave(heads // kv_heads, dim=0)
        scores = q @ k.transpose(-1, -2) / dim**0.5
        scores.masked_fill_(torch.ones(length, length, dtype=torch.bool).triu(1), -torch.inf)
        y = linear((scores.softmax(-1) @ v).transpose(0, 1).reshape(length, h), p + ".self_attn.out_proj")
    x = x + y
    z = norm(x, p + ".ffn_norm")
    f = p + ".feed_forward"
    if i < config["num_dense_layers"]:
        y = linear(F.silu(linear(z, f + ".w1")) * linear(z, f + ".w3"), f + ".w2")
    else:
        probability = linear(z, f + ".gate").sigmoid()
        scores = probability + tensor(f + ".expert_bias") if config["use_expert_bias"] else probability
        indices = scores.topk(config["num_experts_per_tok"], dim=-1).indices
        routes = probability.gather(1, indices)
        if config["norm_topk_prob"]:
            routes = routes / (routes.sum(-1, keepdim=True) + 1e-6)
        routes *= config["routed_scaling_factor"]
        outputs = torch.zeros(length, config["num_experts_per_tok"], h)
        for expert in indices.unique().tolist():
            rows, slots = torch.where(indices == expert)
            e = f + f".experts.{expert}"
            outputs[rows, slots] = linear(F.silu(linear(z[rows], e + ".w1")) * linear(z[rows], e + ".w3"), e + ".w2")
        y = (outputs * routes[..., None]).sum(1)
    x = x + y
    print(f"layer {i + 1}/{len(config['layer_types'])}", flush=True)
logits = norm(x, "model.embedding_norm") @ tensor("model.embed_tokens.weight").float().T
destination.write_text(json.dumps(dict(tokens=tokens, logits=logits.tolist())))
print(destination)
