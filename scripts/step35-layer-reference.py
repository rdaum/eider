#!/usr/bin/env python3
"""Generate focused Step-3.5 layer references from the checkpoint's Python model."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


LAYERS = (0, 1, 3, 4)
TOKENS = 8
HIDDEN = 4096
TOP_K = 8


def load_remote_code(model_dir: Path):
    package_name = "step35_checkpoint"
    package = types.ModuleType(package_name)
    package.__path__ = [str(model_dir)]
    sys.modules[package_name] = package
    modules = {}
    for name in ("configuration_step3p5", "modeling_step3p5"):
        spec = importlib.util.spec_from_file_location(
            f"{package_name}.{name}", model_dir / f"{name}.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import checkpoint module {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["configuration_step3p5"], modules["modeling_step3p5"]


class Checkpoint:
    def __init__(self, model_dir: Path, device: torch.device):
        self.model_dir = model_dir
        self.device = device
        with (model_dir / "model.safetensors.index.json").open() as source:
            self.weight_map = json.load(source)["weight_map"]

    def tensor(self, name: str, device: torch.device | None = None) -> torch.Tensor:
        shard = self.model_dir / self.weight_map[name]
        with safe_open(shard, framework="pt", device="cpu") as source:
            value = source.get_tensor(name)
        return value.to(device if device is not None else self.device)

    def linear(self, prefix: str, inputs: torch.Tensor) -> torch.Tensor:
        packed = self.tensor(f"{prefix}.weight_packed")
        scales = self.tensor(f"{prefix}.weight_scale").float()
        divisor = self.tensor(f"{prefix}.weight_global_scale").float().item()
        low = packed & 0x0F
        high = packed >> 4
        codes = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
        lookup = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            device=self.device,
        )
        weights = lookup[codes.long()]
        weights.mul_(scales.repeat_interleave(16, dim=1)).div_(divisor)
        output = inputs.float() @ weights.t()
        del packed, scales, low, high, codes, lookup, weights
        torch.cuda.empty_cache()
        return output


def rms_norm(inputs: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = inputs.float().square().mean(dim=-1, keepdim=True)
    return inputs.float() * torch.rsqrt(variance + eps) * (weight.float() + 1.0)


def layer_input(layer: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(TOKENS * HIDDEN, dtype=torch.int64, device=device)
    values = ((values * 29 + layer * 17) % 257).float() - 128.0
    return (values * 0.001953125).reshape(TOKENS, HIDDEN)


def rotary(config, modeling, layer: int, inputs: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
    config = copy.deepcopy(config)
    if config.layer_types[layer] not in config.yarn_only_types:
        config.rope_parameters = None
    else:
        config.rope_parameters = config.rope_scaling
    positions = torch.arange(TOKENS, device=inputs.device).unsqueeze(0)
    if config.rope_parameters is None:
        rotary_dim = int(config.head_dim * config.partial_rotary_factors[layer])
        theta = config.rope_theta[layer] if isinstance(config.rope_theta, list) else config.rope_theta
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, rotary_dim, 2, device=inputs.device).float()
                / rotary_dim
            )
        )
        freqs = torch.outer(positions[0].float(), inv_freq)
        embedding = torch.cat((freqs, freqs), dim=-1).unsqueeze(0)
        q, k = modeling.apply_rotary_pos_emb(q, k, embedding.cos(), embedding.sin())
        return q, k, inv_freq
    rotary_embedding = modeling.Step3p5RotaryEmbedding(
        config, device=inputs.device, layer_idx=layer
    ).to(inputs.device)
    cos, sin = rotary_embedding(inputs.unsqueeze(0), positions)
    q, k = modeling.apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, rotary_embedding.inv_freq.detach().float()


def attention_reference(checkpoint, config, modeling, layer: int, normed: torch.Tensor):
    prefix = f"model.layers.{layer}.self_attn"
    if config.layer_types[layer] == "sliding_attention":
        q_heads = config.attention_other_setting["num_attention_heads"]
        kv_heads = config.attention_other_setting["num_attention_groups"]
    else:
        q_heads = config.num_attention_heads
        kv_heads = config.num_attention_groups
    head_dim = config.head_dim

    q = checkpoint.linear(f"{prefix}.q_proj", normed)
    k = checkpoint.linear(f"{prefix}.k_proj", normed)
    v = checkpoint.linear(f"{prefix}.v_proj", normed)
    q_norm_weight = checkpoint.tensor(f"{prefix}.q_norm.weight")
    k_norm_weight = checkpoint.tensor(f"{prefix}.k_norm.weight")
    q = rms_norm(q.reshape(TOKENS, q_heads, head_dim), q_norm_weight, config.rms_norm_eps)
    k = rms_norm(k.reshape(TOKENS, kv_heads, head_dim), k_norm_weight, config.rms_norm_eps)
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.reshape(TOKENS, kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
    q, k, inv_freq = rotary(config, modeling, layer, normed, q, k)

    module = types.SimpleNamespace(
        num_key_value_groups=q_heads // kv_heads,
        training=False,
    )
    attention, _ = modeling.eager_attention_forward(
        module,
        q[:, :, -1:, :],
        k,
        v,
        attention_mask=None,
        scaling=head_dim**-0.5,
    )
    attention = attention.reshape(1, q_heads, head_dim)
    gate = checkpoint.linear(f"{prefix}.g_proj", normed[-1:])
    attention = attention * gate.sigmoid().unsqueeze(-1)
    attention = attention.reshape(1, q_heads * head_dim)
    return checkpoint.linear(f"{prefix}.o_proj", attention), attention, inv_freq


def dense_ffn(checkpoint: Checkpoint, prefix: str, inputs: torch.Tensor) -> torch.Tensor:
    gate = F.silu(checkpoint.linear(f"{prefix}.gate_proj", inputs))
    up = checkpoint.linear(f"{prefix}.up_proj", inputs)
    return checkpoint.linear(f"{prefix}.down_proj", gate * up)


def moe_ffn(checkpoint: Checkpoint, config, layer: int, inputs: torch.Tensor):
    prefix = f"model.layers.{layer}.moe"
    router = checkpoint.tensor(f"{prefix}.gate.weight").float()
    bias = checkpoint.tensor(f"{prefix}.router_bias").float()
    logits = inputs.float() @ router.t()
    probabilities = logits.sigmoid()
    _, indices = torch.topk(probabilities + bias, k=TOP_K, dim=-1)
    weights = probabilities.gather(1, indices)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * config.moe_router_scaling_factor

    routed = torch.zeros_like(inputs)
    for slot, expert in enumerate(indices[0].tolist()):
        expert_prefix = f"{prefix}.experts.{expert}"
        gate = F.silu(checkpoint.linear(f"{expert_prefix}.gate_proj", inputs))
        up = checkpoint.linear(f"{expert_prefix}.up_proj", inputs)
        down = checkpoint.linear(f"{expert_prefix}.down_proj", gate * up)
        routed.add_(down * weights[0, slot])

    shared = dense_ffn(
        checkpoint, f"model.layers.{layer}.share_expert", inputs
    )
    return routed + shared, logits, indices.float(), weights


def layer_reference(checkpoint, config, modeling, layer: int):
    prefix = f"model.layers.{layer}"
    inputs = layer_input(layer, checkpoint.device)
    input_norm = checkpoint.tensor(f"{prefix}.input_layernorm.weight")
    normed = rms_norm(inputs, input_norm, config.rms_norm_eps)
    attention, gated_attention, inv_freq = attention_reference(
        checkpoint, config, modeling, layer, normed
    )
    post_attention = inputs[-1:] + attention
    post_norm = checkpoint.tensor(f"{prefix}.post_attention_layernorm.weight")
    ffn_input = rms_norm(post_attention, post_norm, config.rms_norm_eps)
    result = {
        f"layer_{layer}.input": inputs.cpu(),
        f"layer_{layer}.inv_freq": inv_freq.cpu(),
        f"layer_{layer}.gated_attention": gated_attention.cpu(),
        f"layer_{layer}.attention_output": attention.cpu(),
        f"layer_{layer}.post_attention": post_attention.cpu(),
        f"layer_{layer}.ffn_input": ffn_input.cpu(),
    }
    if layer < 3:
        ffn = dense_ffn(checkpoint, f"{prefix}.mlp", ffn_input)
    else:
        ffn, logits, indices, weights = moe_ffn(checkpoint, config, layer, ffn_input)
        result[f"layer_{layer}.router_logits"] = logits.cpu()
        result[f"layer_{layer}.route_indices"] = indices.cpu()
        result[f"layer_{layer}.route_weights"] = weights.cpu()
    output = post_attention + ffn
    result[f"layer_{layer}.ffn_output"] = ffn.cpu()
    result[f"layer_{layer}.output"] = output.cpu()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the layer reference")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    configuration, modeling = load_remote_code(args.model_dir.resolve())
    config = configuration.Step3p5Config.from_pretrained(args.model_dir)
    checkpoint = Checkpoint(args.model_dir, device)

    tensors = {}
    with torch.inference_mode():
        for layer in LAYERS:
            print(f"generating Python reference for layer {layer}", flush=True)
            tensors.update(layer_reference(checkpoint, config, modeling, layer))
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata={"format": "eider-step35-layer-reference-v1"})
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
