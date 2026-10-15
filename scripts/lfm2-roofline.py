#!/usr/bin/env python3
"""Estimate bandwidth ceilings from a local LFM2 ModelOpt configuration.

These optimistic bounds assume one read of each active matrix and KV element.
They exclude intermediate traffic, small vectors, launch costs, and contention.
No downloads, checkpoint writes, or GPU allocations occur.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--bandwidth-gbs', type=float, default=273.0)
    parser.add_argument('--contexts', type=int, nargs='+', default=[0, 4864, 14984, 32768])
    parser.add_argument('--chunk', type=int, default=256)
    args = parser.parse_args()
    if args.bandwidth_gbs <= 0 or args.chunk <= 0 or min(args.contexts) < 0:
        parser.error('bandwidth and chunk must be positive, contexts nonnegative')
    config = json.loads((args.checkpoint / 'config.json').read_text())
    if config['model_type'] != 'lfm2_moe':
        parser.error('expected an lfm2_moe configuration')
    hidden = config['hidden_size']
    head = hidden // config['num_attention_heads']
    kv = config['num_key_value_heads'] * head
    attention = config['layer_types'].count('full_attention')
    convolution = config['layer_types'].count('conv')
    dense = config['num_dense_layers']
    moe = config['num_hidden_layers'] - dense
    top_k = config['num_experts_per_tok']
    experts = config['num_experts']
    parameters = {
        'convolution': convolution * 4 * hidden**2,
        'attention': attention * (2 * hidden**2 + 2 * hidden * kv),
        'dense_ffn': dense * 3 * hidden * config['intermediate_size'],
        'router': moe * hidden * experts,
        'selected_experts': moe * top_k * 3 * hidden * config['moe_intermediate_size'],
        'lm_head': config['vocab_size'] * hidden,
    }
    # E2M1 is half a byte, plus one UE4M3 byte per 16 weights.
    matrix_bytes = {name: count * (9 / 16 if name == 'selected_experts' else 2)
                    for name, count in parameters.items()}
    weight_bytes = sum(matrix_bytes.values())
    kv_bytes_per_position = attention * 2 * kv * 4
    decode = []
    prefill = []
    for context in args.contexts:
        total = weight_bytes + context * kv_bytes_per_position
        decode.append(dict(context=context, bytes_per_token=total,
                           bandwidth_ceiling_tok_s=args.bandwidth_gbs * 1e9 / total))
        # Worst-case expert coverage, ideal weight reuse within the chunk.
        chunk_experts = min(experts, args.chunk * top_k)
        chunk_weights = (weight_bytes - matrix_bytes['selected_experts']
                         + matrix_bytes['selected_experts'] * chunk_experts / top_k)
        chunk_bytes = chunk_weights + (context + args.chunk) * kv_bytes_per_position
        prefill.append(dict(prefix=context, chunk=args.chunk, bytes_per_chunk=chunk_bytes,
                            bandwidth_ceiling_tok_s=args.chunk * args.bandwidth_gbs * 1e9 / chunk_bytes))
    print(json.dumps(dict(bandwidth_gbs=args.bandwidth_gbs,
                          active_projection_parameters=parameters,
                          projection_bytes_per_decode_token=matrix_bytes,
                          projection_flops_per_decode_token=2 * sum(parameters.values()),
                          kv_bytes_per_position=kv_bytes_per_position,
                          decode=decode, prefill=prefill), indent=2))


if __name__ == '__main__':
    main()
