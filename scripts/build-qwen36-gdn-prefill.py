#!/usr/bin/env python3
"""Build the GB10 Triton cubins used by Qwen3.6 chunked GDN prefill."""

import argparse
from pathlib import Path

import torch

from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule
from vllm.model_executor.layers.fla.ops.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)
from vllm.model_executor.layers.fla.ops.chunk_o import chunk_fwd_kernel_o
from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd_kernel,
)
from vllm.model_executor.layers.fla.ops.cumsum import (
    chunk_local_cumsum_scalar_kernel,
)
from vllm.model_executor.layers.fla.ops.solve_tril import (
    merge_16x16_to_64x64_inverse_kernel,
)
from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd_kernel


KERNELS = {
    "cumsum": chunk_local_cumsum_scalar_kernel,
    "kkt": chunk_scaled_dot_kkt_fwd_kernel,
    "solve": merge_16x16_to_64x64_inverse_kernel,
    "wu": recompute_w_u_fwd_kernel,
    "h": chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    "output": chunk_fwd_kernel_o,
}


def selected_compilation(kernel):
    autotuner = kernel.fn
    config = next(iter(autotuner.cache.values()))
    jit = autotuner.fn
    device_cache = next(iter(jit.device_caches.values()))
    compiled = [
        candidate
        for candidate in device_cache[0].values()
        if candidate.metadata.num_warps == config.num_warps
        and candidate.metadata.num_stages == config.num_stages
        and all(
            candidate.src.constants.get((candidate.src.fn.arg_names.index(name),))
            == value
            for name, value in config.kwargs.items()
        )
    ]
    if len(compiled) != 1:
        raise RuntimeError(f"expected one selected compilation, found {len(compiled)}")
    return compiled[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("crates/nvfp4/native"),
    )
    args = parser.parse_args()

    tokens = 3_328
    heads = 32
    head_dim = 128
    q = torch.randn(
        (1, tokens, heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn_like(q)
    gate = torch.nn.functional.logsigmoid(
        torch.randn((1, tokens, heads), device="cuda", dtype=torch.bfloat16)
    )
    beta = torch.sigmoid(torch.randn_like(gate))
    state = torch.zeros(
        (1, heads, head_dim, head_dim), device="cuda", dtype=torch.float32
    )
    cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    chunk_gated_delta_rule(
        q,
        k,
        v,
        gate,
        beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    torch.cuda.synchronize()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, kernel in KERNELS.items():
        compiled = selected_compilation(kernel)
        if compiled.metadata.arch != "sm121":
            raise RuntimeError(f"expected sm121 for {label}, got {compiled.metadata.arch}")
        output = args.output_dir / f"qwen36_gdn_{label}_sm121.cubin"
        cubin = compiled.asm["cubin"]
        output.write_bytes(cubin)
        print(
            f"wrote {output} ({len(cubin)} bytes, "
            f"warps={compiled.metadata.num_warps}, shared={compiled.metadata.shared})"
        )


if __name__ == "__main__":
    main()
