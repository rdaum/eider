#!/usr/bin/env python3
"""Write a small LFM2 NVFP4 checkpoint and independent CPU reference logits.

Uses the equations in Transformers' modeling_lfm2_moe.py. No model download.
Run with an output directory, then set LFM2_REFERENCE_DIR for the ignored Rust
lfm2_reference test. Add --bf16 for BF16 matrix inputs with F32 accumulation.
Requires torch and safetensors.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

torch.set_num_threads(1)
torch.manual_seed(143)
use_bf16 = "--bf16" in sys.argv[2:]
root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
config = dict(
    model_type="lfm2_moe", hidden_size=128, intermediate_size=256,
    moe_intermediate_size=128, num_hidden_layers=4, num_dense_layers=1,
    num_attention_heads=2, num_key_value_heads=1, num_experts=8,
    num_experts_per_tok=2, vocab_size=256, max_position_embeddings=128,
    norm_eps=1e-5, norm_topk_prob=True, routed_scaling_factor=1.3,
    use_expert_bias=True, tie_word_embeddings=True, conv_L_cache=3,
    conv_bias=False, layer_types=["conv", "conv", "full_attention", "conv"],
    rope_parameters=dict(rope_type="default", rope_theta=1e7),
)
weights = {}
reference = {}


def bf16(name, shape, norm=False):
    value = ((1 + 0.05 * torch.randn(shape)) if norm else torch.randn(shape) * 0.07).bfloat16()
    weights[name] = value
    reference[name] = value.float()


def nvfp4(prefix, rows, cols, global_scale):
    # Exercise all nibbles, nonuniform K16 UE4M3 scales, and independent
    # tensor-wide scales for gate/up/down and every expert.
    codes = torch.randint(0, 16, (rows, cols), dtype=torch.uint8)
    table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
    scales = (0.5 + torch.rand(rows, cols // 16)).to(torch.float8_e4m3fn)
    weights[prefix + ".weight"] = codes[:, ::2] | (codes[:, 1::2] << 4)
    weights[prefix + ".weight_scale"] = scales
    weights[prefix + ".weight_scale_2"] = torch.tensor(global_scale)
    weights[prefix + ".input_scale"] = torch.tensor(0.01)
    reference[prefix + ".weight"] = table[codes.long()] * scales.float().repeat_interleave(16, dim=1) * global_scale


bf16("model.embed_tokens.weight", (256, 128))
bf16("model.embedding_norm.weight", (128,), True)
for i, kind in enumerate(config["layer_types"]):
    p = f"model.layers.{i}"
    for name in ["operator_norm", "ffn_norm"]:
        bf16(f"{p}.{name}.weight", (128,), True)
    if kind == "conv":
        bf16(f"{p}.conv.in_proj.weight", (384, 128))
        bf16(f"{p}.conv.out_proj.weight", (128, 128))
        bf16(f"{p}.conv.conv.weight", (128, 1, 3))
    else:
        for name, rows in [("q_proj", 128), ("k_proj", 64), ("v_proj", 64), ("out_proj", 128)]:
            bf16(f"{p}.self_attn.{name}.weight", (rows, 128))
        for name in ["q_layernorm", "k_layernorm"]:
            bf16(f"{p}.self_attn.{name}.weight", (64,), True)
    f = f"{p}.feed_forward"
    if i == 0:
        for name, shape in [("w1", (256, 128)), ("w3", (256, 128)), ("w2", (128, 256))]:
            bf16(f"{f}.{name}.weight", shape)
    else:
        bf16(f"{f}.gate.weight", (8, 128))
        weights[f"{f}.expert_bias"] = torch.linspace(-0.3, 0.3, 8)
        reference[f"{f}.expert_bias"] = weights[f"{f}.expert_bias"]
        for expert in range(8):
            for j, name in enumerate(["w1", "w3", "w2"]):
                nvfp4(f"{f}.experts.{expert}.{name}", 128, 128, 0.012 + expert * 0.001 + j * 0.003)

save_file(weights, root / "model.safetensors")
(root / "config.json").write_text(json.dumps(config))


def linear(x, name):
    return F.linear(x.bfloat16().float() if use_bf16 else x, reference[name + ".weight"])


def norm(x, name):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + config["norm_eps"]) * reference[name + ".weight"]


tokens = [13, 2, 97, 255, 6, 31, 77, 14, 53, 11, 9, 8]
x = reference["model.embed_tokens.weight"][tokens]
length = len(tokens)
for i, kind in enumerate(config["layer_types"]):
    p = f"model.layers.{i}"
    z = norm(x, p + ".operator_norm")
    if kind == "conv":
        b, c, u = linear(z, p + ".conv.in_proj").chunk(3, dim=-1)
        conv = F.conv1d((b * u).T[None], reference[p + ".conv.conv.weight"], padding=2, groups=128)
        y = linear(c * conv[0, :, :length].T, p + ".conv.out_proj")
    else:
        q = norm(linear(z, p + ".self_attn.q_proj").reshape(length, 2, 64), p + ".self_attn.q_layernorm")
        k = norm(linear(z, p + ".self_attn.k_proj").reshape(length, 1, 64), p + ".self_attn.k_layernorm")
        v = linear(z, p + ".self_attn.v_proj").reshape(length, 1, 64).transpose(0, 1)
        freq = torch.arange(length)[:, None] / (1e7 ** (torch.arange(0, 64, 2).float() / 64))
        angle = torch.cat([freq, freq], dim=-1)[:, None, :]
        def rope(t):
            a, b = t.chunk(2, dim=-1)
            return (t * angle.cos() + torch.cat([-b, a], dim=-1) * angle.sin()).transpose(0, 1)
        q, k = rope(q), rope(k)
        scores = q @ k.transpose(-1, -2) / 8
        scores.masked_fill_(torch.ones(length, length, dtype=torch.bool).triu(1), -torch.inf)
        y = linear((scores.softmax(-1) @ v).transpose(0, 1).reshape(length, 128), p + ".self_attn.out_proj")
    x = x + y
    z = norm(x, p + ".ffn_norm")
    f = p + ".feed_forward"
    if i == 0:
        y = linear(F.silu(linear(z, f + ".w1")) * linear(z, f + ".w3"), f + ".w2")
    else:
        probability = linear(z, f + ".gate").sigmoid()
        indices = (probability + reference[f + ".expert_bias"]).topk(2, dim=-1).indices
        routes = probability.gather(1, indices)
        routes = routes / (routes.sum(-1, keepdim=True) + 1e-6) * 1.3
        y = torch.zeros_like(z)
        for row in range(length):
            for slot in range(2):
                e = f + f".experts.{indices[row, slot]}"
                y[row] += routes[row, slot] * linear(F.silu(linear(z[row], e + ".w1")) * linear(z[row], e + ".w3"), e + ".w2")
    x = x + y
logits = linear(norm(x, "model.embedding_norm"), "model.embed_tokens")
(root / "reference.json").write_text(json.dumps(dict(tokens=tokens, logits=logits.tolist(), bf16=use_bf16)))
print(root)
