# Flint NVFP4

Eider loads the official [Flint NVFP4 GGUF](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf).
The catalogue ID is `flint-nvfp4`. The API model name is `eider-flint`.
The runtime uses resident weights, shared BF16 global KV pages, and bounded local KV rings.

## Run

Start the server:

```sh
scripts/run-eider-flint.sh
```

After the weights and tokenizer are in the Hugging Face cache, start without network access:

```sh
scripts/run-eider-flint.sh --offline
```

Start Pi after the server is ready:

```sh
scripts/run-pi-eider-flint.sh
```

Both `/v1/responses` and `/v1/chat/completions` use the same Flint service.
The defaults are 32,768 context tokens, 256 prefill tokens per chunk, and one active request.
The checkpoint permits 262,144 context tokens. Larger limits increase the global KV reservation and attention scratch space. Global pages are populated as tokens arrive.

The catalogue pins two sources:

| Source | Revision | Files |
| --- | --- | --- |
| `RecursiveSI/flint-nvfp4-gguf` | `6ec43e19b565fc8d560e2d04985d7724f20673bf` | `flint-nvfp4.gguf` |
| `RecursiveSI/flint` | `2a247be0c0aec895ccb0ac7ca92229ec6df37288` | Configuration, tokenizer, generation defaults, and chat template |

Eider creates links to both snapshots under its XDG cache. It does not modify either snapshot.
The GGUF import changes the nibble layout without requantization.

## Architecture

Flint has 49 layers and a hidden width of 2,048.
The first two layers use dense MLPs with width 6,144.
The other layers select six of 256 experts, with width 512 per expert.
Each MoE layer also has an ungated shared expert with width 1,024.
Routing uses sigmoid scores and normalizes the selected weights.

Attention uses 32 query heads, four KV heads, and 128 dimensions per head.
The GGUF supplies explicit local-attention and RoPE patterns.
Local layers retain 513 tokens, including the current token, and use RoPE with theta 10,000.
Global layers do not use RoPE.
Each head has a learned sink logit that contributes to the softmax denominator without a value vector.
Serving prefill chunks of at least 64 rows share KV tiles across 16 queries.
The tiled kernel preserves the reference dot products, 256-key partitions, and sequential softmax order.
It uses 32 KiB of shared memory per block. It adds no persistent device storage.
Decode and smaller chunks use the existing attention path.

All 485 NVFP4 tensors declare `prec_a4=false`.
Eider keeps activations at BF16 precision and rejects checkpoints that request A4 execution.
The loader also validates tensor names, shapes, types, and per-expert scale counts.
Dense projections use SM121 W4A16 tensor cores. Routed projections use the existing addressed W4A16 kernels.
Prefill chunks with at least 64 tokens sort routes by expert and reuse each decoded weight across up to four routes.
Gate, up, and down share one route ordering. Short chunks and decode keep the independent-route path.
Both paths preserve the same FP32 reduction order and per-expert scale placement.

The checkpoint template retains distinct system and developer roles.
The output parser handles optional reasoning and multiple tool calls.
It preserves literal whitespace in string arguments and parses JSON values for structured arguments.
Flint defaults to parallel tool calls. An explicit `parallel_tool_calls: false` limits a turn to one call.

## Sequence cache and execution

Flint uses the shared `seqcache` manager and its ART prefix index.
Global KV uses immutable, shared 128-token page bundles. Prefix insertion and restore do not copy these pages.
Each active request owns a writable tail, a stable device page table, and its 513-token local rings.
Admission reserves future page capacity before a request runs. Cancellation releases active references and unused reservations.
Eviction preserves pages that another request still uses.

A retained prefix also stores the local rings: 36,771,840 bytes at the checkpoint's fixed geometry.
Restoring these rings copies a bounded amount of history, independent of prompt length.
The cache budget accounts for pages, local snapshots, private state, sampling counts, and page tables.
Model weights, execution scratch, and CUDA graph storage are outside this budget.
`--retained-prefix-gib 0` disables retention. The default retained budget is 2 GiB.

Prefill stops at the reusable prefix boundary and leaves at least one prompt token to evaluate after restore.
Both APIs report restored tokens in their cached-token usage fields.
The scheduler runs decode first and rotates active requests when a tick cannot serve all of them.
Two or more ready decode requests share one packed GPU execution. Prefill chunks still execute separately.

Each live sequence retains one decode graph. Its device position changes on each replay.
The graph is replaced when its attention partition count or execution workspace changes.
Packed decode uses a workspace graph with device tables for request positions and KV addresses.
Reordering or replacing requests updates these tables. A change in batch size or attention partition count replaces the graph.
Greedy selection and bounded top-k/top-p sampling stay on the GPU, including history penalties.
Unrestricted sampling uses the CPU reference. RNG and penalty counts belong to each request and are not restored from KV prefixes.

| Capability | Flint status |
| --- | --- |
| Shared prefix index, byte reservations, eviction | Implemented through `seqcache` |
| Zero-copy global prefix pages | Implemented; local history uses bounded snapshots |
| Chunked causal prefill and decode-first scheduling | Implemented |
| Tiled prefill attention | Implemented for local rings and global pages; exact reference outputs checked |
| CUDA decode graphs | Implemented; exact eager/graph replay checked |
| GPU greedy and bounded sampling | Implemented; CPU decision parity checked |
| Packed decode across active requests | Implemented; exact logits checked against isolated execution |
| Packed prefill across active requests | Open; prompt chunks execute in turn |
| Routed expert tensor-core execution | Open; dense projections use tensor cores, routed experts use W4A16 SIMT |
| Compact quantized KV | Open; current BF16 storage preserves the initial numerical path |
| Speculative decoding | Not implemented; this checkpoint has no integrated draft path |

These are capability comparisons with the Qwen and Nemotron runtimes. They do not establish throughput parity between model families.

## Validation commands

Inspect a local GGUF without allocating the model:

```sh
cargo run --release -p eider-inference --bin flint-generate -- "$FLINT_GGUF" --inspect
```

Compare chunked prefill with single-token execution across the local window:

```sh
cargo run --release -p eider-inference --example flint-parity -- "$FLINT_GGUF"
```

Check shared-prefix ownership, eviction, packed decode, graph replay, workspace replacement, and sampling:

```sh
cargo run --release -p eider-inference --example flint-cache-parity -- "$FLINT_GGUF"
```

For the API comparison, start a fresh server with two active slots and two decode slots:

```sh
scripts/run-eider-flint.sh --offline --listen 127.0.0.1:18080 \
    --max-context-tokens 8192 --max-active-sequences 2 --decode-capacity 2
python3 scripts/flint-serving-parity.py --records 320
```

The API check compares cold and warm requests, both streaming adapters, and isolated versus concurrent seeded sampling.
It also cancels one concurrent request and checks the surviving output.
Stop the server before the GPU checks below.

Run the sink-attention reference test and micromeasure separately:

```sh
cargo test -p eider-cuda sink_attention -- --test-threads=1
cargo bench -p eider-cuda --bench flint_attention
cargo bench -p eider-cuda --bench flint_routed_linear
```

Do not run these GPU checks while a model server uses the device.

Generate an independent PyTorch diagnostic, then compare its logits:

```sh
python3 scripts/flint-reference.py "$FLINT_GGUF" /tmp/flint-reference.json
cargo run --release -p eider-inference --example flint-parity -- \
    "$FLINT_GGUF" /tmp/flint-reference.json
```

The Python script requires NumPy and CUDA PyTorch. It dequantizes selected weights and evaluates a fixed seven-token prompt with eager operations.
The comparison checks the top token and reports logit error. It does not enforce full reference parity.

## Initial validation, 2026-10-21

The 521-token window-crossing replay produced identical logits with chunk sizes 1, 7, and 64.
The CUDA sink-attention test matched a CPU reference across causal chunks and ring wraps.
The dense and router projection tests passed batch-invariance and CPU numerical checks.
The GGUF, runtime, and API unit tests passed.

Live checks covered reasoning, two tool calls, tool-result history, streaming, usage, length limits, and cancellation.
Chat Completions and Responses produced the same two calls and the same token counts.
An eager PyTorch comparison agreed on the top token, with maximum logit error 0.824181 and RMS error 0.186645.
That result is an approximate numerical check. Full reference parity and broader quality evaluation remain open.
A layer trace located the first differences in six of 32,256 Q/K projection values in layer zero.
The input normalization and V projection matched exactly. FP64 dot products showed rounding differences in both implementations.
This explains the initial divergence but does not establish an acceptable final-logit error. Temporary runtime trace code was removed.

The attention micromeasure uses 32 query heads, four KV heads, 128-dimensional heads, and constant synthetic inputs.
These CUDA event medians include KV append. They do not measure whole-model performance.

| Attention | Query rows | Retained tokens | Median |
| --- | ---: | ---: | ---: |
| Local decode | 1 | 513 | 0.098 ms |
| Local prefill | 64 | 513 | 0.467 ms |
| Global decode | 1 | 4,096 | 0.108 ms |
| Global prefill | 64 | 4,096 | 3.512 ms |

## Cache and graph validation, 2026-10-21

A 1,057-token replay retained 1,024 tokens and produced identical cold/warm logits.
The next 256 tokens also matched exactly between eager and graph execution, across page and graph boundaries.
Replacing the execution workspace preserved exact logits.
Greedy, bounded sampled, and unrestricted sampled decisions matched the CPU sampler with penalties and a fixed seed.

Prefix insertion and restore added no global page bytes.
The eviction check kept an active reader valid after its retained prefix was evicted.
The backend tests covered partial-page copies, failed admission, append rollback, partial commit, and shared-page ownership.

| Fixed-token full-model replay | Eager/cold | Graph/warm |
| --- | ---: | ---: |
| 1,057 prompt tokens | 3.121 s | 0.121 s, with 1,024 cached tokens |
| 256 decode tokens | 7.900 s | 7.112 s |

The warm prompt time includes a 4.3 ms admission and restore. Decode timing includes full-logit readback for comparison and graph capture.
These are single-run diagnostics, not statistically sampled serving benchmarks.

The HTTP check used a fresh server, 8,192 context capacity, two active slots, one decode slot, and 64 output tokens.
Both APIs returned identical text, reasoning, and token counts. Concurrent seeded requests with penalties also agreed.

| HTTP request | Cached / prompt tokens | First output | Total time |
| --- | ---: | ---: | ---: |
| Cold Chat Completions | 0 / 1,810 | 5.855 s | 7.541 s |
| Warm Chat Completions | 1,792 / 1,810 | 0.131 s | 1.812 s |
| Warm Responses | 1,792 / 1,810 | 0.128 s | 1.815 s |

Measured single-request decode was approximately 37 tokens/s. Concurrent requests shared that throughput; they did not use packed GPU execution.
Closing a stream released its active state. The next request reused the retained prefix and reproduced the original output.

The final attention micromeasure measured 0.135 ms for paged global decode and 4.557 ms for 64-row paged prefill at 4,096 tokens.
The same run measured 0.115 ms and 3.823 ms with ring storage. Page indirection still has a kernel cost.
The paged prefill rows have causal lengths from 4,033 to 4,096; the full ring retains 4,096 keys per row.

## Packed decode validation, 2026-10-21

The full-model replay compared isolated execution with a packed pair across 256 continuation tokens per request.
The prompts contained 1,057 and 513 tokens. One request restored a 1,024-token prefix.
Every vocabulary logit matched exactly at every step.
The replay reversed request order each step and temporarily reduced the batch to one request.
It crossed global page boundaries, graph partition boundaries, and local-ring wraps.
Invalid-token rejection preserved both sequence positions and the cache reservation count.

The packed pair took 6.507 seconds. The two isolated eager runs took 15.896 seconds together.
These timings include logit readback and graph capture. They are diagnostics; only the packed path used graphs in this comparison.
The attention kernel also passed an FP64 reference with four different sequence lengths and reordered graph replay.

A sequential HTTP comparison used the previous server binary and the packed runtime with identical deployment limits.
Each server had two active slots, two decode slots, 8,192 context capacity, and a fresh prefix cache.
Requests used 1,810 prompt tokens and 64 output tokens. Warm requests restored 1,792 tokens.
All text, reasoning, and sampled outputs matched between implementations and API adapters.
Cancelling one member of an active pair preserved the surviving output and released the cancelled state.

| HTTP measurement | Serial runtime | Packed runtime |
| --- | ---: | ---: |
| Warm single request | 1.857 s | 1.856 s |
| Single-request decode | 36.60 tokens/s | 36.57 tokens/s |
| Each concurrent request | 3.678 s | 1.763 s |
| Combined concurrent decode | 36.97 tokens/s | 81.97 tokens/s |

These are single-run serving measurements. Combined decode throughput is the sum of the two overlapping request measurements.
They establish a gain for this two-request workload, not throughput parity with another model family.

The focused attention micromeasure compares a pair of serial launches with one packed launch, including KV append.
For 513-token local attention, CUDA event medians decreased from 0.213 ms to 0.104 ms per pair.
For 4,096-token global attention, three samples ranged from 0.269–0.368 ms serial and 0.143–0.161 ms packed.
These synthetic constant-input measurements passed their numerical checks before timing.

## Routed tensor-core experiment, 2026-10-21

An addressed BF16 MMA prototype preserved the expert scale after FP32 accumulation.
Its synthetic projection checks passed an FP64 reference before timing.
With 16 warps for gate/up and four for down, decode time matched SIMT at approximately 0.017 ms per projection.
At 64 prompt rows, gate/up increased from 0.676 ms to 0.757 ms. Down decreased from 0.847 ms to 0.752 ms.
These measurements repeatedly accessed seven experts and do not represent full-model memory traffic.
The prototype was rejected and removed. The runtime retains the existing expert path.

## Prefill weight reuse, 2026-10-21

A CUDA trace of 4,169 prompt tokens attributed 58.8% of kernel time to routed experts.
Global attention accounted for 18.2%, local attention for 8.8%, and dense projections for 10.0%.
The trace used the committed runtime and 64-token chunks. Profiling overhead is not part of the HTTP measurements.

The retained change uses the existing sorted-route W4A16 kernel. It introduces no new CUDA kernel or weight representation.
Route ordering stays on the GPU, and the workspace owns its reusable storage.
The projection benchmark includes sorting for each projection. The model sorts only once for gate, up, and down together.
All benchmark outputs matched the independent-route kernel exactly. Selected outputs also passed an FP64 reference.

| Prompt rows | Selected experts | Projection | Independent routes | Sorted weight reuse |
| --- | ---: | --- | ---: | ---: |
| 64 | 256 | Gate or up | 1.093 ms | 0.846 ms |
| 64 | 256 | Down | 1.205 ms | 1.001 ms |
| 64 | 32 | Gate or up | 0.702 ms | 0.366 ms |
| 64 | 32 | Down | 0.883 ms | 0.546 ms |
| 256 | 256 | Gate or up | 4.435 ms | 1.923 ms |
| 256 | 256 | Down | 4.824 ms | 2.927 ms |

These CUDA event medians use 256 resident synthetic experts with spread-out or concentrated routing.
They measure one projection repeatedly, not the complete model's weight traffic.

A 521-token full-model replay compared chunks of 7, 63, 64, 128, 129, and 256 tokens with single-token execution.
Every vocabulary logit matched exactly, including the three-token continuation after prefill.
The cases cross the route-reuse threshold, global pages, and the local attention window.

Sequential HTTP runs compared commit `7e0e69a` with route reuse at 64 and 256 tokens per chunk.
Each run started a fresh server with two active slots, two decode slots, and 8,192 context tokens.
The cold prompt contained 4,498 tokens. Warm requests restored 4,480 tokens.
The extended prompt contained 5,397 tokens, with 917 tokens to process after the same prefix restore.

| HTTP measurement | Original, 64 tokens | Reuse, 64 tokens | Reuse, 256 tokens |
| --- | ---: | ---: | ---: |
| Cold first output | 16.449 s | 13.102 s | 10.492 s |
| Cold complete request | 18.196 s | 14.850 s | 12.261 s |
| Extended prompt first output | 4.164 s | 3.544 s | 2.897 s |
| Extended prompt complete request | 5.912 s | 5.300 s | 4.675 s |
| Warm first output | 0.163 s | 0.162 s | 0.164 s |
| Warm decode | 36.08 tokens/s | 36.17 tokens/s | 35.74 tokens/s |

Measured completed requests generated 64 tokens. All text and reasoning matched across runs, including seeded sampling, concurrent decode, and cancellation recovery.
The extended prompt also matched between Chat Completions and Responses after prefix restoration.
These are single-run measurements. They establish lower latency for this workload, not throughput parity with another model family.

Flint now defaults to 256 prefill tokens per chunk. Larger chunks increase scratch storage and the work between scheduler ticks.
At the default 32,768-token context, attention scratch increases from 130 MiB for 64 rows to 520 MiB for 256 rows.
Other batch buffers also grow. The cache budget still excludes workspace storage.
Use `--prefill-token-capacity 64` to select the smaller workspace and scheduling chunks.

A separate experiment placed four head warps in each attention block.
Local prefill improved slightly, but paged global prefill increased from 4.760 ms to 5.139 ms.
The experiment was removed. Attention retains its existing kernels and launch geometry.

## Tiled prefill attention, 2026-10-21

After expert-weight reuse, a trace of 7,185 prompt tokens attributed 41.2% of kernel time to attention and 41.8% to routed experts.
Global attention accounted for 32.1% and local attention for 9.1%.

The selected kernel uses eight warps, two queries per warp, and a shared tile of 32 KV rows.
It preserves each query's partition boundaries, including the different starting positions within a local window.
KV append remains after attention, and the merge includes the learned sink exactly once.
The original ring kernel remains the numerical and benchmark reference.

A standalone probe compared the original and tiled kernels on identical nonzero inputs.
Every output matched exactly across cold prefill, partial tiles, local-ring wraps, and global pages at 4K and 8K context.
At 256 query rows, paged global attention decreased from 17.11 ms to 8.47 ms at 4K context.
At 8K context, it decreased from 34.71 ms to 17.19 ms. These CUDA event medians include KV append.

The repository micromeasure independently measured these candidate medians:

| Attention | Query rows | Context or window | Time |
| --- | ---: | ---: | ---: |
| Paged global prefill | 64 | 4,096 | 2.157 ms |
| Paged global prefill | 256 | 4,096 | 8.309 ms |
| Paged global prefill | 256 | 8,192 | 16.966 ms |
| Local serving prefill | 256 | 513 | 1.195 ms |

The attention regression compares every output with the original kernel across chunks of 1, 63, 64, 65, 129, and 256 tokens.
It also checks an FP64 reference and a nonzero page-layer offset. CUDA memcheck reported zero errors.
The full-model chunk replay retained exact vocabulary logits and continuation across all tested chunk sizes.

HTTP runs compared the existing 256-row expert-reuse runtime with the tiled-attention runtime.
The 7,186-token run used 16,384 context capacity, two active slots, and two decode slots.
Its extended prompt contained 8,085 tokens, with 7,168 restored tokens and 917 uncached tokens.
A separate cold request used the earlier 4,498-token prompt and 8,192 context capacity.

| HTTP measurement | Before tiling | After tiling |
| --- | ---: | ---: |
| 4,498-token cold first output | 10.492 s | 8.827 s |
| 4,498-token complete request | 12.261 s | 10.604 s |
| 7,186-token cold first output | 18.824 s | 15.293 s |
| 7,186-token complete request | 20.592 s | 17.070 s |
| Extended prompt first output | 3.493 s | 2.582 s |
| Extended prompt complete request | 5.257 s | 4.368 s |
| Warm first output | 0.193 s | 0.191 s |
| Warm decode | 35.52 tokens/s | 35.29 tokens/s |

Measured completed requests generated 64 tokens. All text and reasoning matched the previous implementation.
The comparison included cold and restored prefixes, both API adapters, concurrent seeded sampling, and cancellation recovery.
These are single-run measurements. The gain applies to these workloads and does not establish model-family throughput parity.

## Current limits

Packed prefill and faster routed-expert kernels remain performance work.
Flint does not yet have the full execution capabilities or measured throughput parity of the established model runtimes.
Tool calls use the checkpoint protocol and parser, without grammar-constrained sampling.
Long-context quality and performance still need evaluation beyond these correctness checks.
The independent PyTorch logit discrepancy above remains unresolved.

The shared batch projection APIs now keep the reduction order independent of batch size.
Focused numerical tests cover this change. Other model families still need end-to-end performance and output checks before release.
