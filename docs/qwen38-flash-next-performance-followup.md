# Qwen3.8 Flash Next performance follow-up

This note records relevant work published after Eider's first Flash Next
implementation. The review covers llama.cpp changes and new NVFP4 checkpoints.

The review date is September 8, 2026. The local llama.cpp checkout was at
`f3f1a8f2`, which matched `origin/master` during the review.

## Pre-change Eider baseline

Before this work, the repository README recorded these approximate results
from one live Pi session on GB10:

| Operation | Rate |
| --- | ---: |
| Cold prefill | 200 tokens/sec |
| Cached prefill | 170 tokens/sec |
| MTP decode | 13 tokens/sec |

The active deployment uses the
[`Inferact/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4)
checkpoint. Its 51-billion-parameter PLE table uses BF16 and stays on NVMe.

Eider has these important Flash Next paths:

- Batched QSA index projections during prefill.
- Masked sparse QSA attention over the complete logical context.
- Grouped native W4A4 routed experts during prefill.
- W4A16 routed experts during decode.
- Direct PLE reads through `O_DIRECT` and parallel `pread` workers.
- Device-resident speculative state with exact transaction rollback.
- A configurable MTP draft block of one to seven tokens.

This baseline matters because several new llama.cpp changes implement work that
Eider already contains.

## Conclusion

The new evidence justifies more Flash Next work. The current decode rate does
not appear to be the GB10 limit.

Block-FP8 side weights now improve complete GDN and QSA target layers. The
native GDN median decreased from 1.46 ms to 1.21 ms. The native QSA layer is
also 1.21 ms, down from 1.37 ms with BF16 side weights.

The cuda-oxide medians are 0.99 ms for GDN and 1.02 ms for QSA.
These results are approximately 10% faster than the prior cuda-oxide target
layers.

The improved path fuses the QKV and Z projections. It also converts two E4M3
values with one hardware instruction. The original kernels converted each
E4M3 value with scalar software operations.

A wide, small-output BF16 projection kernel also removes the cuda-oxide
hyperconnection bottleneck. It assigns a complete block to each output row.
The previous kernel assigned one warp to each row.

The native exact verifier covers target batches of two, three, and five rows.
A separate native probe also validates real MTP commit and rollback
transactions.

The routed MTP kernels also use the packed E4M3 conversion. Their combined
projection time decreased from approximately 0.15 ms to 0.07 ms per cycle.

The official MTP MoE block now runs with both CUDA backends. Its median time
was 0.42 ms with native CUDA and 0.41 ms with cuda-oxide.

The complete Inferact checkpoint now has a real-prompt depth comparison.
Target-only decode was faster than depths one, two, and four. The faster Hybrid
target still needs the same comparison when a complete checkpoint is present.

The complete cuda-oxide probe matched all output tokens and persistent state.
The target reached approximately 15 tok/s. Depth-one MTP reached approximately
14 tok/s, so target-only decode remained faster on this prompt.

Compact QSA traversal remains useful. Multi-row QSA attention and 48-head
chunked GDN failed a later full-model gate and are not production paths.

The corrected path batches QSA projections and evaluates attention one row at
a time. It uses recurrent GDN for the 48-head Flash Next shape. A 5,840-token
probe reached approximately 300 tokens/sec and matched the reference token.

The direct PLE work from llama.cpp validates Eider's existing design. The
sparse QSA work exposed a remaining problem in Eider: packed cache storage did
not make the attention computation compact. Eider now builds compact selected
block and tile lists and traverses those lists in QK, softmax, probability
quantization, and PV.

## Implementation results

This work added support for the two FP8 formats in the official NVIDIA
checkpoint. The PLE reader now accepts BF16 and per-tensor E4M3 rows. The
reader converts selected FP8 rows to F32 on the GPU.

The MTP loader now accepts E4M3 expert weights with BF16 or F32 inverse
scales. One scale applies to each 128 by 128 weight block. The routed kernels
consume F32 activations and select expert pointer tables on the device.

The implementation does not expand MTP expert weights to BF16. The native
CUDA and cuda-oxide backends pass the same routed numerical test.

The speculative path now supports one to seven draft tokens. It creates the
draft block without a host synchronization between tokens. One target call
validates the complete block.

The transaction commits the accepted state prefix. It restores the GDN, PLE,
QSA, hyperconnection, and stream state after a partial acceptance.

The grouped production W4A4 path already uses a CUTLASS TMA pipeline. The
previous proposal to add TMA to this path was based on the cuda-oxide kernel,
not the production native kernel. Therefore, this work did not add a second
TMA implementation.

The exact target verifier now accepts arbitrary row counts. It compared serial
target execution with batches of two, three, and five rows.

The cuda-oxide verifier repeats the canonical one-row BF16 projection and
top-k dispatch for each target row. This rule prevents one-bit differences from
batched arithmetic without changing normal decode or prefill.

| Target rows | Drafts | Compared rows | Result | Serial rate | Batched rate |
| ---: | ---: | ---: | --- | ---: | ---: |
| 2 | 1 | 4 | Exact | 13.5 rows/sec | 13.6 rows/sec |
| 3 | 2 | 6 | Exact | 13.5 rows/sec | 15.2 rows/sec |
| 5 | 4 | 10 | Exact | 13.6 rows/sec | 15.6 rows/sec |

All compared token IDs matched. The final residual streams also matched
bit-for-bit.

The MTP transaction probe runs separate canonical and speculative target
sequences. It compares the committed token, frontier token, target position,
PLE window, GDN state, and residual streams after each cycle.

| Drafts | Cycles | Committed tokens | Accepted drafts per cycle | Canonical rate | Speculative rate | State result |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 3 | 6 | 1.0 | 13.5 tok/s | 13.7 tok/s | Exact |
| 2 | 3 | 8 | 1.7 | 13.5 tok/s | 12.5 tok/s | Exact |
| 4 | 3 | 8 | 1.7 | 13.5 tok/s | 7.6 tok/s | Exact |

These runs are correctness evidence with a small performance sample. They do
not select the production draft depth.

A later regression run used 5,758 tokens from the repository README and 12
cycles at each depth. The target and speculative paths produced identical
tokens and state.

| Drafts | Accepted drafts per cycle | Target rate | Output rate | Cycle latency |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 13.3 tok/s | 13.3 tok/s | 75 ms/token |
| 1 | 0.83 | 13.3 tok/s | 12.4 tok/s | 148 ms |
| 2 | 1.50 | 13.3 tok/s | 11.5 tok/s | 218 ms |
| 4 | 1.67 | 13.3 tok/s | 7.4 tok/s | 359 ms |

This run used the complete Inferact checkpoint. It proves that deeper MTP does
not repay its work on this prompt. It does not measure the improved Hybrid
target because only two Hybrid target layers are present locally.

A later cuda-oxide run used the same prompt and 12 depth-one cycles. It
committed 23 tokens and accepted 0.92 drafts per cycle. The target reached
14.7 tok/s, and MTP reached 14.0 tok/s. All tokens and persistent target state
matched the canonical path.

The probe also found a workspace-capacity defect. The paged cache rounded its
capacity to 128-token pages, but the MTP QSA workspace used the raw capacity.
The workspace now uses the same page-rounded capacity.

Focused native measurements gave these median GPU times:

| Operation | Shape | Median time |
| --- | --- | ---: |
| Block-FP8 side projection | One token, 10240 by 2560 | 0.10 ms |
| BF16 side projection | One token, 10240 by 2560 | 0.30 ms |
| Block-FP8 expansion and BF16 projection | One token, 10240 by 2560 | 0.60 ms |
| Block-FP8 side projection | 512 tokens, 10240 by 2560 | 17.4 ms |
| Block-FP8 expansion and BF16 projection | 512 tokens, 10240 by 2560 | 1.0 ms |
| BF16 side projection | 512 tokens, 10240 by 2560 | 0.56 ms |
| Official MTP gate and up | Top-10, 640 by 2560 | 0.04 ms |
| Official MTP down | Top-10, 2560 by 640 | 0.03 ms |

These results define two production choices. Decode uses direct block-FP8
W8A16. Prefill expands into reusable BF16 scratch and uses cuBLASLt.

The MTP measurements cover only the routed projections. They do not predict
the speed of a complete speculative cycle.

The cuda-oxide kernels pass the same four focused numerical tests. Their
one-token side projection takes approximately 0.14 ms. The native CUDA kernel
takes approximately 0.10 ms.

A focused official-checkpoint benchmark loaded all 512 MTP experts. It ran the
router, ten routed experts, and the shared expert for one token.

| Backend | Complete MTP MoE |
| --- | ---: |
| Native CUDA | 0.42 ms |
| cuda-oxide | 0.41 ms |

This result validates the production checkpoint loader and routed block-FP8
path. It does not measure the complete MTP draft or target verification cycle.

### GDN prompt batching

The chunked tensor-core GDN path was restricted to 32 value heads. Flash Next
uses 48 value heads, so its 512-token prefill chunks used the token-serial
recurrence kernel instead.

The chunked workspace and both CUDA launch paths now use the model's value-head
count. The released-checkpoint serial comparison passed with 48 heads on both
backends. The existing 32-head regression test also passed on both backends.

These isolated comparisons did not predict full-model behaviour. The 48-head
path caused large residual-stream drift across the complete 48-layer model.
Production Flash Next therefore uses recurrent GDN. Qwen3.6 keeps its validated
32-head chunked path.

The 512-token complete GDN layer gave these results:

| Backend | Complete layer | GDN stage | Complete-layer sample range |
| --- | ---: | ---: | ---: |
| Native before | 16.40 ms | 8.40 ms | 15.71-16.84 ms |
| Native chunked | 9.72 ms | 2.34 ms | 9.67-10.24 ms |
| cuda-oxide chunked | 14.28 ms | 2.79 ms | 13.87-14.39 ms |
| cuda-oxide with shared MoE input | 13.47 ms | 2.79 ms | 13.10-13.60 ms |

The native complete layer is approximately 41% faster. Its GDN stage is
approximately 72% faster. Cuda-oxide now uses the same chunked algorithm.

The cuda-oxide grouped MoE kernel now stages one activation group in shared
memory. Four output warps reuse that group. The MoE stage decreased from
9.68 ms to 8.60 ms, and it remains the complete-layer bottleneck.

A 65-token gate also passed the released-checkpoint serial comparison. Its
complete layer took approximately 3.6 ms with native CUDA and 4.3 ms with
cuda-oxide. This gate exercises one full chunk and one partial tail chunk.

The grouped W4A4 prefill micromeasure used 512 token rows and 512 experts. Its
complete routed pipeline took approximately 6.9 ms. Gate and up took 4.3 ms,
and down took 2.3 ms.

A row-count sweep showed that smaller chunks reduce latency but waste more of
the available throughput:

| Token rows | Complete pipeline | Gate and up | Down |
| ---: | ---: | ---: | ---: |
| 64 | 4.7 ms | 3.0 ms | 1.5 ms |
| 128 | 5.9 ms | 3.8 ms | 1.9 ms |
| 256 | 6.6 ms | 4.2 ms | 2.2 ms |
| 512 | 7.0 ms | 4.3 ms | 2.3 ms |

The 512-row chunk therefore remains the best measured throughput choice. The
sweep does not identify a new production setting.

The cuda-oxide benchmark now uses the same 512 experts and 512 token rows. Its
complete routed prefill pipeline decreased from 9.63 ms to 9.38 ms. Four
output warps share each staged activation group.

The benchmark compares both grouped projections with independent W4A16
results. It also uses concentrated routes that split each active expert into
multiple 16-row groups.

Eight worker partitions improved the one-token result but regressed prefill. A
batch-dependent worker policy did not improve a complete target layer. The
production worker count remains four.

An eight-warp shared-input block exceeded the usable kernel resource limit.
The retained shared-input block uses four warps.

The benchmark now calibrates repeated GPU operations and saves its results. A
shorter 128-element K tile increased the pipeline time to 7.2 ms. A 64-element
K tile increased it to 11.0 ms.

The automatic stage count selects three stages. Two stages did not improve the
pipeline result.

A 32-column output tile and the explicit pointer-array ping-pong schedule also
did not improve the complete pipeline. Their median times were 6.9 ms and
7.0 ms.

A 16-column output tile also took 6.9 ms. Smaller output tiles do not remove
the grouped pipeline's latency limit at this shape.

Combining a 32-column output tile, a 128-element K tile, and two stages took
7.4 ms. The lower resource use did not produce an occupancy win.

Caching the CUTLASS operator initialization also did not help. The complete
pipeline remained at 7.0 ms, so the production wrapper still initializes each
invocation explicitly.

CUTLASS rejected the tested 64-row tile because its TMA scale layout was not
valid. The tested 256-row tile exceeded the usable kernel resource limit. The
original 128 by 64 by 256 tile remains the measured winner.

The winning kernel uses one persistent block on each of the 48 SMs. Each block
uses 384 threads, 115 registers per thread, and approximately 101 KiB of
dynamic shared memory.

An active-group sweep changed the logical tile-wave count while it kept 512
input rows and the production tile shape:

| Active experts | Mean rows per expert | Gate/up tile waves | Down tile waves | Pipeline median | Sample range |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 40 | 27 | 54 | 2.16 ms | 2.15-2.17 ms |
| 256 | 20 | 54 | 107 | 3.80 ms | 3.74-3.81 ms |
| 512 | 10 | 107 | 214 | 6.93 ms | 6.87-6.95 ms |

The persistent launch still uses one block per SM. The table counts useful
output tiles divided by 48 SMs and rounds up. Runtime scales with useful tiles,
so zero-sized groups and launch preparation are not the primary cost. Limiting
the active expert set is not a valid production change because model routing
selects from all 512 experts.

The measured occupancy was approximately 23%. Compute throughput was 12%, and
memory throughput was 17%. These values indicate a latency and occupancy limit,
not tensor-core or memory-bandwidth saturation.

A complete cuda-oxide probe reached approximately 15 tok/s for the target and
14 tok/s with one MTP draft. Its tokens and persistent state matched the
canonical path. The rounded README values remain unchanged because they report
live server telemetry, not probe timing.

### Complete target-layer measurement

The Hybrid-Sharp loader now accepts its paired `qwen4_exp` and
`qwen4_exp_text` model types. Partial checkpoint smoke tests loaded and ran one
GDN layer and one QSA layer through attention, hyperconnections, routed
experts, and the shared expert.

A focused benchmark then measured the same complete linear-attention layer from
the Inferact and Hybrid-Sharp checkpoints.

| Stage | Inferact BF16 | Original Hybrid | Improved native | Improved cuda-oxide |
| --- | ---: | ---: | ---: | ---: |
| Complete layer | 1.35 ms | 1.46 ms | 1.21 ms | 0.99 ms |
| Attention mix | 0.12 ms | 0.12 ms | 0.12 ms | 0.10 ms |
| Attention | 0.56 ms | 0.66 ms | 0.42 ms | 0.46 ms |
| Attention combine | 0.03 ms | 0.03 ms | 0.03 ms | 0.03 ms |
| MLP mix | 0.13 ms | 0.13 ms | 0.13 ms | 0.11 ms |
| MoE and shared expert | 0.46 ms | 0.49 ms | 0.45 ms | 0.29 ms |
| MLP combine | 0.02 ms | 0.02 ms | 0.02 ms | 0.02 ms |

The improved Hybrid-Sharp layer is approximately 17% faster than its original
path. It is approximately 10% faster than the Inferact BF16 layer.

The cuda-oxide result repeated at 0.99 ms. It is approximately 18% faster than
the improved native path in this benchmark.

The target-layer harness now accepts both attention families. A QSA layer gave
these additional medians:

| Stage | Inferact BF16 | Improved native | Improved cuda-oxide |
| --- | ---: | ---: | ---: |
| Complete layer | 1.37 ms | 1.21 ms | 1.02 ms |
| Attention mix | 0.12 ms | 0.12 ms | 0.10 ms |
| QSA attention | 0.56 ms | 0.44 ms | 0.48 ms |
| Attention combine | 0.03 ms | 0.02 ms | 0.03 ms |
| MLP mix | 0.13 ms | 0.13 ms | 0.11 ms |
| MoE and shared expert | 0.47 ms | 0.45 ms | 0.26 ms |
| MLP combine | 0.02 ms | 0.02 ms | 0.02 ms |

The native block-FP8 QSA layer is approximately 11% faster than the BF16
layer. The cuda-oxide QSA layer is approximately 26% faster.

The fused QKV and Z launch gave a small improvement by itself. The hardware
E4M3 conversion produced most of the gain.

The benchmark covers one complete target layer with fixed routing inputs. It
does not measure model loading, PLE reads, MTP acceptance, or serving overhead.

### Experimental QSA prompt batching

This section records the rejected multi-row path. Production still batches
the QSA projections, but it processes attention one row at a time.

The experimental QSA selector returns every visible token before the context
exceeds the 2,048-token indexer budget. This path bypasses scoring and
selection in this range. It still appends the raw index keys for later sparse
attention.

The experimental dense prefix uses the existing causal paged-attention kernel.
The native path processes eight rows per launch. The cuda-oxide path processes
four rows per launch because an eight-row batch exceeded the numerical gate.

The 512-token Inferact benchmark gave these median GPU times:

| Path | Native CUDA | Cuda-oxide |
| --- | ---: | ---: |
| Dense attention, one row per launch | 38.1 ms | 34.1 ms |
| Dense attention, batched rows | 9.3 ms | 12.3 ms |
| Batched row phase only | 6.6 ms | 9.9 ms |

The native complete QSA layer decreased by approximately 76%. The cuda-oxide
layer decreased by approximately 64%.

The microbenchmark compares the complete layer output with independent serial
execution before it records timing. Native CUDA and cuda-oxide passed this
gate for 512 rows. A mixed prompt range across the 2,048-token boundary also
passed.

The experimental sparse path retains 16 independent selections. It batches QSA
selection, cache updates, and attention for 16 causal rows. Selection also
builds ordered lists of selected blocks, 8-token key tiles, and 64-token value
tiles. The attention kernels traverse these lists instead of launching and
scanning over the complete logical context.

The 512-token sparse benchmark starts at position 4,096. These medians include
the complete QSA layer:

| Sparse path | Native CUDA | Cuda-oxide |
| --- | ---: | ---: |
| Original native path | 65.1 ms | — |
| Consolidated key append | 62.6 ms | 86.5 ms |
| Batched attention | 37.1 ms | 60.0 ms |
| Batched selection and attention | 25.1 ms | 27.9 ms |
| Reused scores and 16-row batches | 19.5 ms | 21.4 ms |
| Compact attention traversal | 19.0 ms | 20.6 ms |

The final experimental native path is approximately 71% faster than the
original sparse path. The cuda-oxide comparison starts after key-append
consolidation and shows an approximately 76% decrease.

The compact traversal matters more as the logical context grows. A second
512-token benchmark starts at position 32,768 with a 65,536-token workspace:

| Sparse path | Native CUDA | Cuda-oxide |
| --- | ---: | ---: |
| Original native path | 67.6 ms | — |
| Compact attention traversal | 33.6 ms | 50.0 ms |

These results include selection, cache updates, attention, and the remainder
of the complete QSA layer. Both backends compare the batched output with
independent row execution before timing.

Two cold server requests after dense-prefix batching used different 5.7K-token
README prompts. Both reached approximately 350 tokens/sec with 16-second
first-token latency.

A cold request before compact traversal included the sparse batching changes.
Its 5,800 uncached tokens reached approximately 460 tokens/sec with 13-second
first-token latency.

Two cold requests after compact traversal had 5,796 and 5,801 uncached tokens.
Both reached approximately 490 tokens/sec with 12-second first-token latency.

Two experimental requests also used 48-head chunked GDN and multi-row QSA.
Each request had 5,801 uncached tokens. They reached approximately 620
tokens/sec with approximately 9.4-second first-token latency.

The result is not a valid production claim. The complete model generated a
different frontier token than the serial reference.

This server gate used a 16K context, one active sequence, 512-token prefill
chunks, and no speculative workspace. The earlier 200-token/sec baseline used
the 262K live Pi profile.

### Full-model prefill gate

The isolated GDN and QSA tests compared one layer. Their small errors grew
across the complete model, so Eider added a full-model prefill probe.

The probe rendered `README.md` through the checkpoint chat template. It then
processed 5,840 tokens in 512-token chunks and compared the final token and
residual streams.

The `qwen38-flash-next-prefill-probe` binary implements this gate. It returns
an error when production prefill selects a different token.

The rejected production path selected token 248069. The serial reference
selected token 1596. Its final residual-stream cosine similarity was 0.434.

A run with serial GDN still selected the wrong token when multi-row QSA was
active. A run with serial QSA selected the reference token despite 48-head
chunked GDN drift. This isolated multi-row QSA as the direct token error.

The corrected production path batches QSA projections and processes attention
one row at a time. It also uses recurrent GDN for the 48-head model shape.

The corrected path and the serial reference both selected token 1596. The
corrected path processed the prompt in 19 seconds, or approximately 300
tokens/sec. The fully serial reference took approximately 163 seconds.

The corrected residual stream was not numerically identical to the serial
reference. Its cosine similarity was 0.785. The production path still uses
batched projections. Therefore, the token comparison remains a required gate.

An API regression case requested the recent Git commits through a Bash tool.
The server emitted one parsed tool call. It then consumed the tool result and
returned a normal summary. It did not repeat planning text or expose raw tool
syntax.

## llama.cpp status

llama.cpp added the initial `qwen4exp` implementation in
[#27742](https://github.com/ggml-org/llama.cpp/pull/27742). Later merged changes
reduced graph splits and reduced the QSA indexer heads by slices.

The merged changes also corrected recurrent-state rollback, sequence copies,
block positions, and recurrent-layer metadata. These changes improve the
reference implementation but do not identify a missing Eider feature.

The most relevant performance changes remain in open pull requests.

### NextN and MTP draft head

Draft pull request
[#28610](https://github.com/ggml-org/llama.cpp/pull/28610) adds a complete
NextN draft graph and supports sidecar, embedded, and draft-only tensor layouts.

Its test data reports that one draft performed best on a mixed consumer-GPU
system. Deeper draft chains reduced throughput as acceptance decreased.

This result agrees with the direction of Eider's small probe. It does not
replace an Eider measurement because the hardware, quantization, and graph
execution differ.

The pull request contains two useful correctness details. The MTP block uses
the full-attention path, and the hidden input contains all four hyperconnection
streams. Eider's implementation already follows both requirements.

### Direct PLE reads

Pull request
[#28136](https://github.com/ggml-org/llama.cpp/pull/28136) replaces lazy `mmap`
faults with concurrent direct reads. Its author reports a real-input prefill
increase from approximately 300 to 750-800 tokens/sec on GB10.

The result identifies sparse PLE access as a large prefill cost. Repeated-token
benchmarks can hide this cost because they select fewer PLE rows.

Eider already uses direct reads, parallel workers, asynchronous batches, and
duplicate-row removal. Therefore Eider does not need the llama.cpp reader.

Eider reads one aligned region for each unique BF16 row. Each region has a
minimum size of 4 KiB because `O_DIRECT` requires aligned transfers.

An FP8 row is smaller than a BF16 row. This format change does not automatically
halve the physical I/O for Eider's current reader.

### Sparse QSA decode

Pull request
[#28213](https://github.com/ggml-org/llama.cpp/pull/28213) gathers selected QSA
keys and values into a compact buffer. The current llama.cpp path otherwise
applies a mask over the complete KV cache.

The pull request reports a decode increase from 15.7 to 23.6 tokens/sec at a
130K context. The measured model used two RTX A6000 GPUs and an IQ4 checkpoint.

Eider previously stored the QSA cache in pages and produced selection masks,
but its kernels still launched and scanned across the complete logical
context. The review therefore exposed the same missing compute compaction.

The new Eider path keeps the paged cache in place. It builds ordered compact
indices for selected blocks and key/value tiles, then uses those indices in
the attention kernels. It does not gather or copy selected KV data into a
temporary cache.

At a 32K starting position, the complete 512-token native QSA layer decreased
from 67.6 ms to 33.6 ms. The current cuda-oxide result is 50.0 ms. A
production-path one-token microbenchmark at the same position takes 0.8 ms on
native CUDA and 0.8 ms on cuda-oxide. There is no matching pre-change
one-token result, so that measurement is a current cost, not a claimed speedup.

### Device-resident MTP state

Pull request
[#28118](https://github.com/ggml-org/llama.cpp/pull/28118) keeps speculative
recurrent checkpoints on the device. It removes a host round trip from each
speculative cycle.

Eider already keeps the Flash Next speculative state on the device. Its
transaction path commits accepted rows and restores rejected rows without a
host copy.

### Pipelined NVFP4 matrix operations

Pull request
[#28572](https://github.com/ggml-org/llama.cpp/pull/28572) overlaps NVFP4 tile
loads with matrix operations. It uses `cp.async`, TMA, and register-spill
reductions on Blackwell.

The pull request reports a 14% prefill increase on an RTX 5090. The workload
was dense Qwen3.8-27B NVFP4, not Flash Next on GB10.

The related
[#28514 profiling note](https://github.com/ggml-org/llama.cpp/discussions/28514)
reports a 45% combined increase. That experiment also included chunked GDN and
activation preparation changes.

Eider's cuda-oxide grouped W4A4 kernel issues direct global loads inside its K
loop. The production native path uses the CUTLASS SM120 array TMA mainloop and
the automatic warp-specialized schedule.

The cuda-oxide difference is an alternative-backend opportunity. It does not
identify a missing pipeline in the production native path. The RTX 5090 tile
configuration is not evidence for the best GB10 configuration.

## New checkpoint work

### Official NVIDIA mixed checkpoint

NVIDIA published
[`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4).
It has this documented precision layout:

| Component | Format |
| --- | --- |
| Main routed experts | W4A4 NVFP4 |
| Attention and shared experts | BF16 |
| Other main-model layers | BF16 |
| MTP routed experts | Block FP8, 128 by 128 |
| PLE n-gram table | Per-tensor FP8 |

This checkpoint has better provenance than the current Inferact checkpoint.
It also reduces the PLE payload by approximately half.

Eider supports both special formats. The PLE pager accepts the
per-tensor FP8 table, and the MTP loader accepts BF16 block scales.

The metadata review found 128 numbered PLE shards. The n-gram tensor uses
E4M3 rows with one BF16 scale. Each MTP expert projection uses E4M3 weights and
BF16 inverse scales with 128 by 128 blocks.

Selective tests used the official PLE and MTP shard. The FP8 PLE reader and
the complete MTP MoE block ran successfully with the checkpoint's real
tensors. The full main-model shards were not downloaded for this test.

The local snapshot contains three of the eleven files named by its weight
index. Eight main-model shards are absent, so a full loader gate cannot run
from the current cache.

Official checkpoint support is useful for correctness and storage. It does not
remove the BF16 side-weight traffic from target decode.

### Hybrid-Sharp checkpoint

The
[`Hybrid-Sharp`](https://huggingface.co/travelinlance/Qwen3.8-Flash-Next-RadixArk-NVFP4-Hybrid-Sharp)
checkpoint keeps the routed experts in NVFP4. It converts approximately 15 GiB
of always-read side weights from BF16 to block FP8.

These side weights include attention projections and shared experts. Every
decoded token reads these tensors, independent of the selected routed experts.

The model card reports approximately 20% faster generation and no benchmark
quality-score change. This claim comes from its patched vLLM serving stack.

This format directly targets a likely Eider decode limit. Eider already has
block-FP8 matrix support in other model paths.

Eider now loads the Hybrid-Sharp model type and runs a complete target layer.
Only the files required for layer zero were downloaded for this check.

The local snapshot contains ten of the 206 files named by its weight index.
It cannot run the full target or the production MTP-depth comparison.

The improved block-FP8 path wins the focused complete-layer benchmark. The
partial checkpoint cannot supply an end-to-end serving result.

The checkpoint is a community build from a private candidate base. Loader work
must use tensor metadata, numeric validation, and immutable revisions.

### Local Inference Lab checkpoint

The
[`local-inference-lab` checkpoint](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4)
uses a more aggressive mixture of NVFP4 and FP8 formats. Its repository payload
is approximately 99 GiB at the review date.

The model card does not document calibration, evaluation, or the complete
precision policy. The tensor metadata indicates non-current PLE and side-weight
formats.

Treat this checkpoint as a format study. Do not make it the production
reference without quality evidence.

## Single-Spark evidence

The
[`single-spark-ai` recipe](https://huggingface.co/single-spark-ai/Qwen3.8-Flash-Next-DGX-Spark-Optimized-Recipe)
combines these components on one GB10:

- NVIDIA NVFP4 routed experts.
- Block-FP8 dense side weights.
- An FP8 MTP draft model.
- Four-token speculative cycles.
- Bounded `io_uring` with `O_DIRECT` for the PLE table.
- CUDA graphs for the target and draft paths.

The recipe reports 63.1 output tokens/sec for a 512-token coding prompt. It
also reports a mean accepted length of 3.41 tokens per four-token cycle.

This result comes from one machine and one serving stack. It is not an
apples-to-apples comparison with the current Eider Pi measurement.

The result still provides strong evidence against a 13-token/sec hardware
ceiling. The large result depends on both a faster target and deeper MTP.

## Performance hypotheses

### Decode side-weight traffic

The routed experts use NVFP4, but many target weights remain in BF16. The target
reads attention and shared-expert weights for every token.

This traffic is consistent with a memory-bandwidth limit during decode. This
statement is an inference, not a measured Eider counter result.

Block-FP8 side weights halve the payload for affected tensors. They also leave
more unified memory for KV pages and workspaces.

### MTP depth

Eider allocates Flash Next speculative resources for every configured positive
draft depth. The current upper limit is seven drafts.

Eider validation now proves exact target execution for one, two, and four
drafts. The transaction probe also validates the committed recurrent and
hyperconnection state at these depths.

Accepted tokens must repay the extra draft and target work. Otherwise, deeper
MTP does not increase output throughput. Measure acceptance and cycle time
together.

### Native NVFP4 prefill pipeline

The grouped W4A4 prefill path uses the SM121 `m16n8k64` NVFP4 instruction. Its
CUTLASS mainloop uses TMA and a warp-specialized pipeline.

The next kernel work is tile, stage-count, and schedule tuning. Each candidate
must improve the complete grouped pipeline on GB10.

SM121 uses synchronous warp-collective `mma.sync`. It does not use the SM100
TMEM programming model.

### FP8 PLE storage

The official FP8 PLE table decreases checkpoint size and row payload size. The
current Eider direct reader still transfers aligned 4 KiB regions.

Page-level coalescing can reduce duplicate aligned reads for selected rows on
one page. Random row distribution can limit this benefit.

A real prompt is necessary for this measurement. Repeated-token inputs do not
represent PLE row diversity.

## Recommended work

### 1. Fuse the block-FP8 attention projections

1. Add one kernel call for the QKV and Z projections.
2. Keep the native CUDA and cuda-oxide results numerically aligned.
3. Compare the fused result with the current serial block-FP8 calls.
4. Run the complete target-layer benchmark again.
5. Keep the path only if the complete layer cost decreases.

This work is complete. The fused path and packed E4M3 conversion reduced the
complete Hybrid-Sharp layer time from 1.46 ms to 1.21 ms.

### 2. Measure MTP depth after the target improvement

1. Keep the exact verifier as the correctness gate.
2. Compare depths zero, one, two, and four on the same real prompt.
3. Use enough cycles to report stable timing and acceptance.

Record target rows/sec, output tokens/sec, accepted tokens per cycle, and cycle
latency. Do not select a depth from acceptance alone.

The complete Inferact checkpoint has this comparison and favours depth zero on
the measured README prompt. Repeat the same gate with the improved Hybrid
target when its complete checkpoint is available.

### 3. Tune the grouped W4A4 prefill kernel

1. Keep the current production-shape benchmark as the reference.
2. Compare CUTLASS tile shapes, stage counts, and schedules.
3. Measure register use, shared-memory use, active warps, and tensor-pipe
   activity.
4. Compare several CTA-wave counts on GB10.

The first tile sweep found no improvement. Eider retains the original 128 by
64 by 256 tile. A new candidate needs a different scheduling or data-layout
change, not a shorter K tile.

### PLE measurement with a representative prompt

The PLE benchmark now renders and tokenizes real project text. It also accepts
`QWEN38_PLE_BENCH_PROMPT_FILE` for a fixed external prompt.

The 512-token BF16 measurement requested 8,192 logical rows. It selected about
7,200 unique rows and read 30 MiB. Median storage time was 41 ms, and complete
batch time was 42 ms.

The repeated-token case selected only 16 unique rows and finished in 0.5 ms.
It does not represent normal prompt traffic.

A same-page read-coalescing prototype did not help a preliminary 512-token
case. Its 1,208 unique rows required the same 1,208 direct reads. The prototype
was removed before the final representative measurement.

The official FP8 table selected the same 7,208 unique rows. It read 29 MiB,
with a median storage time of 42 ms and a complete batch time of 43 ms. The
BF16 table read 30 MiB in approximately the same time.

The 4 KiB aligned reads remove most of the raw row-size benefit. The FP8 table
did not improve PLE latency for this prompt. The representative read can
overlap the first prefill layer. It was not the main cost of the pre-change
200-token/sec prefill path.

### 4. Validate the official NVIDIA checkpoint

1. Keep PLE payloads on NVMe during model loading.
2. Run a loader smoke test with the complete checkpoint.
3. Compare outputs with the checkpoint's documented serving implementation.

This support gives Eider a documented production base. It is not, by itself, a
complete decode optimization.

The focused PLE and MTP loader checks are complete. A full loader smoke test
and output comparison still require the complete main-model checkpoint.

## Measurement plan

Use focused benchmarks before a full-model run:

1. Measure BF16 and block-FP8 side projections at exact Flash Next shapes.
2. Measure one complete target layer with identical routing inputs.
3. Measure one exact speculative cycle at each supported draft depth.
4. Measure W4A4 tile and schedule candidates at production group sizes.
5. Measure BF16 and FP8 PLE reads with row IDs from a real prompt.

Validate correctness before each timing phase. Record medians and variation for
all performance results.

Run the complete model only after one isolated path shows a stable gain. Use one
fixed coding prompt and one fixed long-context prompt.

For the final comparison, record these values:

- Target-only decode rate.
- Output decode rate for each MTP depth.
- Accepted drafts per cycle.
- Cold and warm prefill rates.
- PLE logical rows, unique rows, physical bytes, and elapsed time.
- Resident unified memory after graph capture.
- First-token latency and complete request latency.

Do not run another model server or a high-memory benchmark at the same time.
GB10 device allocations consume the same 128 GB unified-memory pool.

## Decision

Keep the fused block-FP8 path and the packed E4M3 conversion. The complete
Hybrid-Sharp target layer is now faster than the BF16 baseline.

The current Inferact depth comparison favours target-only decode. Repeat it
with the complete Hybrid checkpoint before changing the production MTP
default. The local Hybrid snapshot is partial and cannot provide this result.

Keep batched QSA projections and compact sparse traversal. Process QSA
attention one row at a time in production. Keep 48-head chunked GDN out of the
Flash Next path. Its isolated test passed, but the complete model accumulated
large residual-stream drift.

Keep the four-warp cuda-oxide grouped MoE kernel. It reduced the complete
released-checkpoint GDN layer from 14.28 ms to 13.47 ms.

The corrected full-model probe used 512-token chunks. Its 5,840-token README
prompt reached approximately 300 tokens/sec and matched the serial-reference
token. Target-only decode remained between 13 and 14 tokens/sec.
