# Qwen3.8 Flash-Next readiness

This note records speculative work before the public Qwen3.8 Flash-Next release.
It separates reusable work from architecture details that need a public checkpoint.

## Current evidence boundary

The provisional description names a 125-billion-parameter multimodal MoE model.
It also names 6 billion active parameters and 51 billion n-gram memory parameters.
These values are not a loader contract.

Eider does not contain a speculative Qwen4 model adapter.
The checkpoint must define tensor names, layer order, routing rules, state shapes, and attention metadata.
An adapter written without these facts creates rework and can hide correctness errors.

## Reusable work

The n-gram implementation does not depend on the Qwen4 layer design.
It follows the public contract in
[vLLM PR #47857](https://github.com/vllm-project/vllm/pull/47857).
The implementation includes these components:

- Polynomial hashes for multiple n-gram orders and split tables.
- Rolling token windows for each request.
- Chunked-prefill hashing.
- Transactional append, commit, abort, and prefix restoration.
- BF16, rowwise FP8, and rowwise NVFP4 embedding banks.
- Batched indexed gathers.
- Fused dequantization, projection, word-embedding addition, and averaging.
- Scalar CPU oracles and collision coverage.

The fused CUDA path does not create a temporary tensor for all selected rows.
Its row IDs and work buffers keep stable device addresses.
These properties make the path suitable for later CUDA Graph integration.

## GB10 memory estimate

The rowwise NVFP4 format uses 4.5 bits per parameter.
This value includes one UE4M3 scale byte for each 16 E2M1 values.

| Component | Parameters | Raw NVFP4 payload |
| --- | ---: | ---: |
| Main model | 125 billion | 65.5 GiB |
| N-gram memory | 51 billion | 26.7 GiB |
| Combined | 176 billion | 92.2 GiB |

A 128 GiB GB10 system has approximately 35.8 GiB after these raw payloads.
The remaining memory must contain other weights, CUDA workspaces, sequence state, KV pages, and staging buffers.

The estimate excludes alignment and metadata.
It also excludes any checkpoint tensors that do not use rowwise NVFP4.
Six billion active parameters can reduce token compute, but it does not reduce resident weight storage.

## Existing Qwen path

The current Qwen path already covers the main performance risks of the rumoured model.
It includes hybrid GDN layers, 256-expert top-8 MoE routing, W4A4 projections, and compact long-context KV storage.

The retained evidence gives these results:

| Path | Result | Evidence |
| --- | --- | --- |
| GDN prefill | Correctness passed, approximately 2.08 ms | `019f9d00-66dc-7ea0-8ff9-75e4757c2638` |
| Grouped W4A4 MoE decode | Correctness passed, approximately 0.122 ms | `019f9c32-f796-7a43-9484-a8c50189436f` |
| Indexed W4A4 projection | Correctness passed, approximately 0.031 ms | `019f9c28-7f73-7893-bbbd-9dfa51a6b82c` |
| Compact KV at 128K | 4.836 ms to 0.591 ms | `019f9dcf-68c9-7102-95b0-23c98aadfbaf` |
| Full-model decode at 4K | 15.962 ms to 14.756 ms | `019f9ef1-8203-7fe1-8335-c1314c7342e8` |

The direct CUTLASS W4A4 projection experiment was faster in isolation.
Small numeric changes altered expert routes and capacity decisions.
Eider rejected that path because the full serving result was not correct.

## Selected-index attention

DeepSeek V4 provides a useful preparation target for possible sparse attention.
Its score generation is separate from selected-KV attention.
No code claims that this path implements Qwen Sparse Attention.

The Lightning Indexer now scores compressed entries in parallel.
A bounded 4,096-entry scratch slab prevents a context-sized score allocation.
Each slab merges into a stable top-k index buffer.
The score kernel keeps the old per-head accumulation order.
This property prevents numeric changes in selected indices.

| Workload | Serial median | Parallel median | Change | Evidence |
| --- | ---: | ---: | ---: | --- |
| One row, 128K context, top 512 | 861.225 ms | 9.897 ms | -98.85% | `01a03bab-5a3f-7272-9977-bbdaa2271ee2` |
| 16 rows, 16K context, top 512 | 106.578 ms | 5.578 ms | -94.77% | `01a03bad-1a6b-76d3-af7a-a656df267944` |

Both comparisons passed deterministic selected-index validation.
Unit coverage also compares block selection with tokenwise selection.

## Release-day procedure

1. Record the immutable checkpoint revision and every repository file.
2. Read `config.json`, the tensor index, tokenizer files, and remote model code.
3. Compare each tensor and state shape with the current Qwen and n-gram boundaries.
4. Add a model adapter only after the tensor and execution contracts are known.
5. Keep multimodal towers outside the text path until their input contract is known.
6. Run loader-shape tests before a full checkpoint allocation.
7. Run CPU-oracle tests for hashing, routing, recurrent state, and selected indices.
8. Run focused GPU benchmarks before full-model decode measurements.
9. Measure resident memory, first-token latency, decode latency, and long-context growth on GB10.
10. Compare public llama.cpp and vLLM support with the checkpoint contract.

## Stop conditions

Do not infer Qwen4 semantics from matching names such as GDN or sparse attention.
Do not reinterpret checkpoint scale layouts as native MMA layouts.
Do not allocate the 125-billion and 51-billion banks together before the memory audit passes.
Do not retain a faster kernel if it changes routing, state, selected indices, or output semantics.
