#!/usr/bin/env python3
"""Compare Eider logits with independent PyTorch Dory nGPT arithmetic.

Uses the local NVFP4 snapshot only. Equations follow vLLM Dory at
55076950b6c155f384d6766d63563c3bc20a0e3f, with F32 W4A16 accumulation.
Run after dory_generate has written its final positional LOGITS_JSON argument.
"""
import argparse
import json
import math
from pathlib import Path
import torch
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("eider_logits", type=Path)
    parser.add_argument("--loops", type=int, default=2)
    args = parser.parse_args()
    config = json.loads((args.model_dir / "config.json").read_text())
    actual = json.loads(args.eider_logits.read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    with safe_open(args.model_dir / "model.safetensors", framework="pt", device="cpu") as f:
        weights = {key: f.get_tensor(key).cuda() for key in f.keys()}
    dtype = torch.bfloat16
    e2m1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")

    def norm(x):
        y = x.float()
        return (y * torch.rsqrt(y.square().sum(-1, keepdim=True) + 1e-6)).to(x.dtype)

    def linear(x, prefix):
        packed = weights[prefix + ".weight"]
        ids = torch.stack((packed & 15, packed >> 4), -1).flatten(1)
        w = (e2m1[ids.long()] * weights[prefix + ".weight_scale"].float().repeat_interleave(16, -1)).to(dtype)
        return (torch.mm(x, w.T, out_dtype=torch.float32) * weights[prefix + ".weight_scale_2"]).to(dtype)

    ids = torch.tensor(actual["tokens"], device="cuda")
    positions = torch.arange(len(ids), device="cuda")
    h = weights["backbone.embeddings.weight"][ids]
    std = config["initializer_range"]
    nq, nk, dim = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]

    def layer(h, i):
        prefix = f"backbone.layers.{i}"
        mixer = prefix + ".mixer"
        kind = config["layer_types"][i]
        if kind == "mlp":
            scale = math.sqrt(config["hidden_size"])
            g = (linear(h, mixer + ".gate_proj").float() * (weights[mixer + ".suv_gate"].float() * scale)).to(dtype)
            u = (linear(h, mixer + ".up_proj").float() * (weights[mixer + ".suv_up"].float() * scale)).to(dtype)
            out = linear(norm((torch.nn.functional.silu(g.float()) * u.float()).to(dtype)), mixer + ".down_proj")
            alpha = weights[prefix + ".mlp_alpha"]
        else:
            theta = config["rope_parameters"][kind]["rope_theta"]
            angles = positions.float()[:, None] * (theta ** (-torch.arange(dim // 2, device="cuda").float() / (dim // 2)))[None, :]
            cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]

            def rope(x):
                a, b = x.float().chunk(2, -1)
                return torch.cat((a * cos - b * sin, b * cos + a * sin), -1).to(dtype)

            q = norm(rope(linear(h, mixer + ".q_proj").reshape(-1, nq, dim)))
            k = norm(rope(linear(h, mixer + ".k_proj").reshape(-1, nk, dim)))
            q = (q.float() * (weights[mixer + ".sqk"].float() * (math.sqrt(dim) / std)).square().reshape(nq, dim)).to(dtype)
            v = linear(h, mixer + ".v_proj").reshape(-1, nk, dim)
            k, v = k.repeat_interleave(nq // nk, 1), v.repeat_interleave(nq // nk, 1)
            scores = torch.einsum("thd,shd->hts", q.float(), k.float()) / math.sqrt(dim)
            mask = positions[:, None] >= positions[None, :]
            if kind == "sliding_attention":
                mask &= positions[:, None] - positions[None, :] < config["sliding_window"]
            probs = scores.masked_fill(~mask, float("-inf")).softmax(-1)
            out = torch.einsum("hts,shd->thd", probs, v.float()).reshape(-1, nq * dim).to(dtype)
            gate = linear(h, mixer + ".g_proj")
            g0 = config.get("attention_output_gate_logit_scale_init") or math.sqrt(config["hidden_size"])
            sg = (weights[mixer + ".s_gate"].float() * (math.sqrt(g0) / std)).square()
            out = norm((out.float() * torch.sigmoid(gate.float() * sg)).to(dtype))
            out = linear(out, mixer + ".o_proj")
            alpha = weights[prefix + ".attn_alpha"]
        a, b = norm(h).float(), norm(out).float()
        rate = (alpha.float() * (config["ngpt_alpha_init"] / std)).abs()
        return norm(a + rate * (b - a)).to(dtype)

    with torch.inference_mode():
        ni, nr = config["n_input_layers"], config["n_recurrent_layers"]
        for i in range(ni):
            h = layer(h, i)
        input_h = h
        for _ in range(args.loops):
            h = norm(h + input_h)
            for i in range(ni, ni + nr):
                h = layer(h, i)
            h = norm(h)
        for i in range(ni + nr, config["num_hidden_layers"]):
            h = layer(h, i)
        expected = (h[-1:] @ weights["lm_head.weight"].T).float().flatten() * (weights["lm_head.logit_scale"].float() / std)
        actual = torch.tensor(actual["logits"], device="cuda")
        err = actual - expected
        report = {"loops": args.loops, "tokens": len(ids), "max_abs": err.abs().max().item(), "rms": err.square().mean().sqrt().item(), "cosine": torch.nn.functional.cosine_similarity(actual, expected, dim=0).item(), "eider_top": actual.topk(5).indices.tolist(), "reference_top": expected.topk(5).indices.tolist()}
        print(json.dumps(report, indent=2))
        assert actual.argmax() == expected.argmax(), "top token differs"
        assert report["cosine"] > .999, "logit drift exceeds tolerance"

if __name__ == "__main__":
    main()
