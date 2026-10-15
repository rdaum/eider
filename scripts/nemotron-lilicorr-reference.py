#!/usr/bin/env python3
"""CPU reference for the released LiLiCorr drafter; no downloads or snapshot writes.

Usage: python3 scripts/nemotron-lilicorr-reference.py DRAFT TARGET OUTPUT [--slots 7]
Requires torch and safetensors. Uses deterministic synthetic target residuals and
the real target embedding. The output is a small safetensors correctness fixture.
Reference: vLLM 73a78e6f1f38e280986b81e0f2a9aa5e1ee6fe47, lilicorr.py,
qwen3_dflash.py, qwen3_dflash2.py, and LiLiCorrSpeculator.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("draft", type=Path)
parser.add_argument("target", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--slots", type=int, default=7, choices=range(1, 8))
parser.add_argument("--features", type=Path, help="target feature JSON from capture_lilicorr_reference_context")
args = parser.parse_args()
torch.set_num_threads(4)
torch.manual_seed(4182)
checkpoint = safe_open(args.draft / "model.safetensors", framework="pt", device="cpu")
config = json.loads((args.draft / "config.json").read_text())
index = json.loads((args.target / "model.safetensors.index.json").read_text())["weight_map"]
embedding_name = next(n for n in index if n.endswith("backbone.embeddings.weight"))
target_shard = safe_open(args.target / index[embedding_name], framework="pt", device="cpu")
embedding = target_shard.get_tensor(embedding_name)
table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
eps = config["rms_norm_eps"]


def tensor(name):
    return checkpoint.get_tensor(name)


def linear(x, prefix):
    weight = tensor(prefix + ".weight")
    bias = tensor(prefix + ".bias") if prefix + ".bias" in checkpoint.keys() else None
    if weight.dtype != torch.uint8:
        return F.linear(x.float(), weight.float(), None if bias is None else bias.float()).bfloat16()
    scales = tensor(prefix + ".weight_scale").float()
    global_scale = tensor(prefix + ".weight_scale_2").float()
    outputs = []
    # Bound expansion scratch, including the 131072-row candidate LM head.
    for begin in range(0, weight.shape[0], 1024):
        packed = weight[begin:begin + 1024]
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
        # W4A16 reconstructs the complete scaled weight in BF16 before GEMM.
        # Moving the global scale after GEMM changes this rounding boundary.
        decoded = (table[codes.long()] * scales[begin:begin + 1024].repeat_interleave(16, -1)
                   * global_scale).bfloat16().float()
        y = F.linear(x.float(), decoded).bfloat16()
        if bias is not None:
            y = y + bias[begin:begin + 1024]
        outputs.append(y)
    return torch.cat(outputs, dim=-1)


def norm(x, prefix):
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
            * tensor(prefix + ".weight").float()).bfloat16()


def rope(x, start):
    positions = torch.arange(start, start + x.shape[0]).float()
    angle = positions[:, None] / config["rope_theta"] ** (torch.arange(0, 128, 2).float() / 128)
    angle = torch.cat((angle, angle), dim=-1)[:, None, :]
    left, right = x.float().chunk(2, dim=-1)
    return (x.float() * angle.cos() + torch.cat((-right, left), dim=-1) * angle.sin()).bfloat16()


def convolve(x, coefficients, base):
    # CUDA reference accumulates the two taps in F32 before storing BF16.
    y = (base[0].float() + coefficients[:, 0].float().repeat_interleave(16, -1)) * x.float()
    y[1:] += ((base[1].float() + coefficients[1:, 1].float().repeat_interleave(16, -1)) * x[:-1].float())
    return y.bfloat16()


def conv_prepare(x, prefix):
    coeff = linear(x, prefix + ".kernel_projection").reshape(x.shape[0], 2, 2, 256)
    base = tensor(prefix + ".base_kernel")
    return convolve(x, coeff[:, 0], base[0]), coeff[:, 1], base[1]


traces = {}
context_rows, anchor_token = 17, 4242
aux = torch.randn(context_rows, 20480).bfloat16()
if args.features:
    captured = json.loads(args.features.read_text())
    context_rows, anchor_token = captured["rows"], captured["anchor_token"]
    aux = torch.tensor(captured["features"]).reshape(context_rows, 20480).bfloat16()
context = norm(linear(aux, "fc"), "hidden_norm")
rows = args.slots + 1
x = embedding[torch.tensor([anchor_token] + [990] * args.slots)]
for i in range(5):
    p = f"layers.{i}"
    z, coeff, base = conv_prepare(norm(x, p + ".input_layernorm"), p + ".attention_conv")
    q = rope(norm(linear(z, p + ".self_attn.q_proj").reshape(rows, 32, 128), p + ".self_attn.q_norm"), context_rows)
    k = rope(norm(linear(z, p + ".self_attn.k_proj").reshape(rows, 8, 128), p + ".self_attn.k_norm"), context_rows)
    v = linear(z, p + ".self_attn.v_proj").reshape(rows, 8, 128)
    ck = rope(norm(linear(context, p + ".self_attn.k_proj").reshape(context_rows, 8, 128), p + ".self_attn.k_norm"), 0)
    cv = linear(context, p + ".self_attn.v_proj").reshape(context_rows, 8, 128)
    k = torch.cat((ck, k)).repeat_interleave(4, dim=1).transpose(0, 1).float()
    v = torch.cat((cv, v)).repeat_interleave(4, dim=1).transpose(0, 1).float()
    attention = F.scaled_dot_product_attention(q.transpose(0, 1).float(), k, v).transpose(0, 1).reshape(rows, 4096).bfloat16()
    x = x + convolve(linear(attention, p + ".self_attn.o_proj"), coeff, base)
    z, coeff, base = conv_prepare(norm(x, p + ".post_attention_layernorm"), p + ".mlp_conv")
    gate, up = linear(z, p + ".mlp.gate_proj"), linear(z, p + ".mlp.up_proj")
    activated = (F.silu(gate.float()) * up.float()).bfloat16()
    x = x + convolve(linear(activated, p + ".mlp.down_proj"), coeff, base)
    traces[f"layer_{i}_hidden"] = x.float().clone()
    traces[f"layer_{i}_context_key"] = ck.float().clone()
    traces[f"layer_{i}_context_value"] = cv.float().clone()
    print(f"draft layer {i+1}/5", flush=True)
pass_hidden = norm(x, "norm")[1:]
logits = linear(pass_hidden, "lm_head").float()
log_probs, ids = logits.log_softmax(-1).topk(8, dim=-1)
anchor = context[-1:]

p = "lilicorr"
features = torch.stack((log_probs, log_probs.exp(), log_probs-log_probs.max(-1, keepdim=True).values,
                        torch.arange(8).float().div(7).expand_as(log_probs),
                        (torch.arange(8) == 0).float().expand_as(log_probs)), dim=-1).bfloat16()
features = F.layer_norm(features, (5,), tensor(p + ".feature_norm.weight"), tensor(p + ".feature_norm.bias"))
features = linear(F.silu(linear(features, p + ".feature_mlp.up_proj")), p + ".feature_mlp.down_proj")
x = linear(embedding[ids], p + ".token_proj") + linear(pass_hidden, p + ".pass_hidden_proj")[:, None]
x = x + features
x = x + tensor(p + ".slot_embedding")[0, 0, :args.slots]
x = x + tensor(p + ".rank_embedding")[0, 0]
x = x.flatten(0, 1)
slot_ids = torch.arange(args.slots).repeat_interleave(8)
relative = slot_ids[:, None]-slot_ids[None, :]
attention_bias = tensor(p + ".relative_slot_bias")[:, relative+7]
attention_bias = attention_bias + (relative == 0).bfloat16()[None]*tensor(p + ".same_slot_bias")[:, None, None]
for i in range(2):
    layer = p + f".layers.{i}"
    z = norm(x, layer + ".attn_norm")
    qkv = F.linear(z.float(), tensor(layer + ".attn.in_proj_weight").float(), tensor(layer + ".attn.in_proj_bias").float()).bfloat16()
    q, k, v = [t.reshape(-1, 8, 128).transpose(0, 1).float() for t in qkv.chunk(3, -1)]
    attention = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_bias.float()).transpose(0, 1).flatten(1).bfloat16()
    x = x + linear(attention, layer + ".attn.out_proj")
    z = norm(x, layer + ".mlp_norm")
    x = x + linear(F.silu(linear(z, layer + ".mlp.up_proj")), layer + ".mlp.down_proj")
x = norm(x, p + ".output_norm").reshape(args.slots, 8, 1024)
a = norm(linear(anchor, p + ".context_proj"), p + ".anchor_norm")
w = tensor(p + ".factor_input_proj.weight")
pre = F.linear(x.float(), w[:, :1024].float(), tensor(p + ".factor_input_proj.bias").float()).bfloat16()
pre = pre + F.linear(a.float(), w[:, 1024:2048].float()).bfloat16()
pre = pre + F.linear((x*a).float(), w[:, 2048:].float()).bfloat16()
factor = F.silu(pre)
outgoing = F.normalize(linear(factor, p + ".out_head"), dim=-1, eps=0.0001)
incoming = F.normalize(linear(factor, p + ".in_head"), dim=-1, eps=0.0001)
anchor_out = F.normalize(linear(a, p + ".anchor_out_head"), dim=-1, eps=0.0001)
start = (anchor_out*incoming[0]).sum(-1).float()*8
pairs = (outgoing[:-1].float() @ incoming[1:].float().transpose(-1, -2)).bfloat16().float()*8
scores = torch.cat((start[None, None, :].expand(1, 8, 8), pairs))
walk, previous = [], 0
for step in range(args.slots):
    previous = scores[step, previous].argmax().item()
    walk.append(ids[step, previous].item())
args.output.parent.mkdir(parents=True, exist_ok=True)
save_file({**traces, "aux": aux.float(), "anchor": anchor.float().contiguous(),
           "anchor_token": torch.tensor([anchor_token], dtype=torch.int32),
           "pass_hidden": pass_hidden.float().contiguous(), "logits": logits,
           "candidate_ids": ids.to(torch.int32), "candidate_log_probs": log_probs,
           "scores": scores.contiguous(), "greedy_tokens": torch.tensor(walk, dtype=torch.int32)}, str(args.output),
          metadata={"draft": str(args.draft), "target": str(args.target), "anchor_token": str(anchor_token)})
print(args.output, flush=True)
