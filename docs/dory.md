# Dory 1.0 4B NVFP4

Eider loads NVIDIA's [Dory NVFP4 checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4)
at revision `41f02335e1a1ad3289b8b32485e2265cbe97cfdf`.
The runtime uses the W4A16 weights without activation quantization.
Embeddings and the separate output head retain BF16 weights.

Dory uses nGPT L2 normalization, learned interpolation residuals, gated attention,
and input reinjection between loops. Its 72 physical sublayers contain 36 attention/MLP pairs.
Two loops execute 112 sublayers with shared weights and separate attention caches.
This matches the checkpoint's recommended `per_loop` configuration.
The runtime rejects `last_loop` cache semantics.

## Start

Start the server with the downloaded checkpoint:

```sh
scripts/run-eider-dory.sh
```

The default is two loops. For one loop, use:

```sh
scripts/run-eider-dory.sh --dory-loops 1
```

Start Pi in a second terminal:

```sh
scripts/run-pi-eider-dory.sh
```

The catalogue ID is `dory1.0-4b-nvfp4`.
API requests use `eider-dory1.0-4b`.
Both `/v1/responses` and `/v1/chat/completions` use the checkpoint template and shared output parser.
The launcher uses one active sequence and prompt chunks of 2,048 tokens.
The configured context limit is 262,144 tokens.

## Execution

Prefill uses batched BF16 matrix products with F32 accumulation.
Activation buffers are BF16, so projections do not convert or round whole F32 buffers.
The loader prepares BF16 projection weights once, which adds about 6.15 GiB of resident device memory.
Decode uses packed NVFP4 tensor-core projections and replays a CUDA graph.
Q/K/V/gate projections share one launch. MLP gate/up projections share another.
The tensor-wide ModelOpt scale applies after F32 accumulation and before BF16 rounding.
Attention uses native FlashAttention 2 kernels for causal and sliding-window layers.
Decode groups query heads by their shared KV head and splits the key sequence across blocks.
The [vendored source note](../crates/eider-cuda/native/vendor/flash-attention/README.md) records the pinned revision and licence.
Without CUTLASS headers, the build uses scalar attention.
Normalization and residual operations preserve the reference model's BF16 rounding boundaries.

KV storage is BF16. Each loop has independent causal attention history.
Request capacity includes the prompt and maximum output length.
Single-sequence serving prepares up to 32K tokens of KV storage and warms the kernels at startup.
This reserves 7 GiB at two loops, bounded by the configured context limit.
It retains one idle allocation and its decode graph for reuse.
Capacity grows in powers of two below 8K, then in 8K increments, bounded by the context limit.
At two loops, KV storage requires 224 KiB per reserved token, before workspace and prefix retention.
A full 256K allocation therefore requires 56 GiB of KV storage.
The configured limit does not imply a full-context quality or performance evaluation.

The prefix cache retains each completed prompt.
The final token stays in its prefill batch. Exact prefix hits replay one token to obtain logits.
It reuses matching buffer sizes from evicted snapshots and copies only the retained state.
Sliding attention snapshots retain two windows. Full attention snapshots retain their complete prefix.
A snapshot can restore a shorter matching prefix if it contains every key required by the sliding window.
Prefill fills each batch from the restored position, without an early split at an absolute chunk boundary.
`--retained-prefix-gib` bounds retained device memory. Zero disables retention.

## GB10 measurements

Measured on one DGX Spark on 2026-10-18, using identical streamed Chat Completions fixtures.
Both engines used the pinned NVFP4 checkpoint, two loops, BF16 compute and KV, and one active request.
Each measured request generated 128 tokens with greedy decoding and `reasoning_effort: medium`.
The benchmark context limit was 32,768 tokens; the launcher's context limit remains 262,144.
A 27-token warmup preceded the measured requests. Startup and model loading are excluded.
TTFT includes request handling, allocation, prefix restoration, prefill, and readout.
Decode throughput is `(output_tokens - 1) / (completion_time - first_token_time)`.

The comparison used [vLLM revision 5507695](https://github.com/vllm-project/vllm/commit/55076950b6c155f384d6766d63563c3bc20a0e3f)
with kernel build `153a17166e5d07bc38184dedeea542d2e8885e55`, Marlin W4A16, and FlashAttention 2.
vLLM used prefix caching, async scheduling, and a GPU memory fraction of 0.25.
Its startup reported 3.28 GiB of model memory and 25.25 GiB available for KV.
Eider used 2,048-token prefill batches and the startup allocation described above.

| Prompt tokens | Cached tokens, Eider / vLLM | Eider TTFT (ms) | vLLM TTFT (ms) | Eider decode (tok/s) | vLLM decode (tok/s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 568 | 3 / 0 | 123 | 140 | 48.9 | 49.4 |
| 6,981 | 33 / 32 | 1,236 | 1,361 | 45.1 | 44.5 |
| 10,336 | 6,973 / 32 | 685 | 2,060 | 43.4 | 42.8 |
| 17,286 | 10,328 / 10,320 | 1,523 | 1,608 | 40.2 | 40.1 |
| 21,348 | 17,278 / 10,320 | 979 | 2,583 | 39.0 | 38.5 |
| 23,714 | 21,347 / 21,344 | 690 | 667 | 37.9 | 37.6 |

Cache hits differ at 10K and 21K, so those TTFT differences include prefix reuse.
The 7K, 17K, and 24K cases process almost equal numbers of uncached tokens.
Eider is faster on the first two; its 24K TTFT is 3.3% slower in this reference run.
Across two vLLM runs, 24K TTFT ranged from 667 to 701 ms; Eider measured 690 ms.
Decode is within 1% at 568 tokens and slightly faster at the longer contexts.
These synthetic code-review prompts approximate the Pi context lengths; they are not a replay of the user's conversation.

Before these changes, Eider took 11.4 s at 7K, 24.3 s at 17K, and 12.0 s for the 24K follow-up.
Its 24K decode rate was 28.6 tok/s.
The original Eider baseline is `f6a06847f272a3142823a9b71ba4b33b0b3719f6`.

Local fixtures, pinned environment, server logs, and streaming results are under
`~/.cache/eider/bench-artifacts/dory-vllm/`; `comparison-final.json` records both final runs.
The main fixture SHA-256 is `7aa8de3b5d1e247605c37cc27c5168c863296afa4530441a3c4e6a32ed9b5c2b`.
The follow-up SHA-256 is `c1bd33d45d93c8cc519c62ecc021fe7e80b414a5a2323207c34a79e2f9e4b190`.

## Decisions

`POST /v1/decisions` supports `noul`, `choice`, and `score` questions.
The runtime processes the shared prefix once, then processes each question suffix separately.
Questions reuse the same KV allocation and cannot see sibling suffixes.
The readout projects only the 64 label rows and applies their checkpoint logit scales.
It generates no reasoning or answer text.

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "eider-dory1.0-4b",
    "state": "The parcel arrived yesterday. It contains a red book.",
    "questions": {
      "arrived": {"type": "noul", "instructions": "Has the parcel arrived?"},
      "colour": {
        "type": "choice",
        "instructions": "What colour is the book?",
        "criteria": {"red": null, "blue": null}
      }
    }
  }'
```

Decision probabilities are uncalibrated. Changing the loop count changes the scoring model.
Suffixes run sequentially in this implementation. Packed multi-question execution remains future work.

## Validation

The independent PyTorch comparison uses the local checkpoint and equations from
[vLLM's Dory implementation](https://github.com/vllm-project/vllm/blob/55076950b6c155f384d6766d63563c3bc20a0e3f/vllm/model_executor/models/dory.py).
The final two-loop comparison has logit cosine similarity 0.99919 and the same top five tokens
on a 27-token arithmetic prompt. This is numerical coverage, not a task-quality evaluation.
CUDA checks cover BF16 residual boundaries, causal attention, and sliding-window exclusion.
The checkpoint regression covers one and two loops, chunked prefill, decode graphs,
prefix restoration and eviction, reused request buffers, selected-label readout, and isolated decision suffixes.

```sh
DORY_MODEL=/path/to/snapshot cargo test --release -p eider-inference --lib \
  dory_chunk_graph_and_prefix_agree -- --ignored
cargo bench -p eider-cuda --bench dory_attention
cargo bench -p eider-cuda --bench dory_norm
DORY_MODEL=/path/to/snapshot cargo bench -p eider-cuda --bench dory_projection
```

For a direct generation run and reference comparison:

```sh
cargo run --release -p eider-inference --example dory_generate -- \
  /path/to/snapshot 2 128 2048 'What is 7 times 8? Answer briefly.' /tmp/dory-logits.json
python3 scripts/dory-checkpoint-reference.py /path/to/snapshot /tmp/dory-logits.json --loops 2
```
