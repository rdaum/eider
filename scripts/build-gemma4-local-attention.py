#!/usr/bin/env python3
"""Build the GB10 Triton cubin used by Gemma 4 local prefill attention."""

import argparse
import math
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "query_tokens",
        "key_tokens",
        "start_position",
        "stride_qt",
        "stride_qh",
        "stride_kt",
        "stride_kh",
        "stride_vt",
        "stride_vh",
        "stride_vd",
        "stride_ot",
        "stride_oh",
    ]
)
def gemma4_local_attention(
    query,
    key,
    value,
    output,
    softmax_scale,
    query_tokens,
    key_tokens,
    start_position,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KV_GROUPS: tl.constexpr,
    WINDOW_TOKENS: tl.constexpr,
):
    query_head = tl.program_id(1)
    query_block = tl.program_id(2)
    key_head = query_head // KV_GROUPS

    query_rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    dimensions = tl.arange(0, HEAD_DIM)
    keys = tl.arange(0, BLOCK_N)
    absolute_queries = start_position + query_rows

    query_offsets = (
        query_head * stride_qh
        + query_rows[:, None] * stride_qt
        + dimensions[None, :]
    )
    q = tl.load(
        query + query_offsets,
        mask=query_rows[:, None] < query_tokens,
        other=0.0,
    )

    row_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
    row_sum = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    block_query_start = start_position + query_block * BLOCK_M
    key_start = tl.maximum(0, block_query_start + 1 - WINDOW_TOKENS)
    end_key = tl.minimum(key_tokens, start_position + (query_block + 1) * BLOCK_M)
    key_span = end_key - key_start

    for key_offset in range(0, key_span, BLOCK_N):
        key_positions = key_start + key_offset + keys
        key_offsets = (
            key_head * stride_kh
            + key_positions[None, :] * stride_kt
            + dimensions[:, None]
        )
        k = tl.load(
            key + key_offsets,
            mask=key_positions[None, :] < end_key,
            other=0.0,
        )
        scores = tl.dot(q, k)
        mask = (key_positions[None, :] < end_key) & (
            key_positions[None, :] <= absolute_queries[:, None]
        )
        mask &= absolute_queries[:, None] - key_positions[None, :] < WINDOW_TOKENS
        scores = tl.where(mask, scores * softmax_scale, -1.0e8)

        block_max = tl.maximum(row_max, tl.max(scores, axis=1))
        probabilities = tl.math.exp2(scores - block_max[:, None])
        correction = tl.math.exp2(row_max - block_max)
        row_sum = row_sum * correction + tl.sum(probabilities, axis=1)
        accumulator *= correction[:, None]

        value_offsets = (
            key_head * stride_vh
            + key_positions[:, None] * stride_vt
            + dimensions[None, :] * stride_vd
        )
        v = tl.load(
            value + value_offsets,
            mask=key_positions[:, None] < end_key,
            other=0.0,
        )
        accumulator = tl.dot(probabilities.to(v.dtype), v, accumulator)
        row_max = block_max

    accumulator /= row_sum[:, None]
    output_offsets = (
        query_head * stride_oh
        + query_rows[:, None] * stride_ot
        + dimensions[None, :]
    )
    tl.store(
        output + output_offsets,
        accumulator,
        mask=query_rows[:, None] < query_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("crates/nvfp4/native/gemma4_local_attention_sm121.cubin"),
    )
    args = parser.parse_args()

    heads = 16
    kv_heads = 8
    head_dim = 256
    block_m = 64
    block_n = 64
    tokens = 128
    query = torch.empty((heads, tokens, head_dim), device="cuda", dtype=torch.bfloat16)
    key = torch.empty((kv_heads, tokens, head_dim), device="cuda", dtype=torch.bfloat16)
    value = torch.empty_like(key)
    output = torch.empty_like(query)
    runtime_args = (
        query,
        key,
        value,
        output,
        (1.0 / math.sqrt(head_dim)) * math.log2(math.e),
        tokens,
        tokens,
        0,
        head_dim,
        tokens * head_dim,
        head_dim,
        tokens * head_dim,
        head_dim,
        tokens * head_dim,
        1,
        head_dim,
        tokens * head_dim,
    )
    compiled = gemma4_local_attention.run(
        *runtime_args,
        grid=(1, heads, triton.cdiv(tokens, block_m)),
        warmup=True,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        KV_GROUPS=heads // kv_heads,
        WINDOW_TOKENS=1024,
        num_warps=8,
        num_stages=2,
    )
    if compiled.metadata.arch != "sm121":
        raise SystemExit(f"expected sm121, got {compiled.metadata.arch}")
    if compiled.metadata.shared > 101_376:
        raise SystemExit(
            f"kernel needs {compiled.metadata.shared} shared bytes; GB10 has 101376"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(compiled.asm["cubin"])
    print(
        f"wrote {args.output} ({len(compiled.asm['cubin'])} bytes, "
        f"shared={compiled.metadata.shared})"
    )


if __name__ == "__main__":
    main()
