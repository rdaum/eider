# Nemotron Super 3.5 VL and Bolt text support

Status: released NVFP4 checkpoint loads and passes text-serving and prefix-reuse
checks, 2026-10-15. Batched prefill uses tensor cores. Independent reference parity
and vision support remain pending.

[llama.cpp PR #30520](https://github.com/ggml-org/llama.cpp/pull/30520)
adds `NemotronH_Omni_Reasoning_V3` for Super 3.5 VL and Bolt 74B-A9B.
The inspected head is `e64207bab6c2989a4ae5aa8b9e0cbe700e135ee3`.
The GitHub API reports that the PR merged on 2026-10-15.

[NVIDIA released Bolt 74B-A9B](https://research.nvidia.com/labs/nemotron/Nemotron-3.5-Bolt-24GB/)
on October 15. Its official 24 GB checkpoint uses IQ2_XXS experts, NVFP4 Mamba
projections, and Q8_0 dense weights. Native packed execution supports text generation
and the compact decision head. The community NVFP4 conversion also remains supported.

## Bolt

The launcher defaults to the official 24 GB checkpoint and its LiLiCorr companion.
The earlier measurements used
[cloudnathan5's conversion](https://huggingface.co/cloudnathan5/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4)
at revision `fcf26d4c868500a8e1bacd28b7ec2d1dcb7b2646`.
Its index reports 47,772,692,520 bytes across 63 shards, including unused vision weights.
This is a community conversion from NVIDIA's BF16 release.

The text model has 88 layers and a 262,144-token context limit.
Each MoE layer selects nine experts. Expert counts range from 240 to 496,
and intermediate widths range from 1,536 to 2,560.
The parser accepts the released expert-count fields and retains each layer's dimensions.
Conflicting global counts and invalid per-layer bounds fail before device allocation.

Expert weights use W4A16 NVFP4 without calibrated activation scales.
Mamba projections use calibrated FP8. Attention, latent projections, embeddings,
and the output head remain BF16.
Sampled shard headers confirm these formats.
The loader preserves these precisions and uses the existing text and decision runtime.
This checkpoint has no MTP block. Generation can use the official Bolt LiLiCorr companion.
The companion uses six target features, two KV heads, a 1,024-token window,
and attention sinks. It shares the target output head: Q8_0 for the official
checkpoint, BF16 for the conversion.
Super uses a different companion with five target features and its own output head.

Download the pinned checkpoint into the Hugging Face cache:

```sh
hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB \
  --revision c6cc2e8ffa353403213d6b9b18aaf8eb684d705a
hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr \
  --revision 4c6a5613b6d887194a3a5493e3a18a5b3930bb86
```

Start the server, then use Pi or the decision script:

```sh
scripts/run-eider-nemotron35-bolt.sh
# In another terminal:
scripts/run-pi-eider-nemotron35-bolt.sh
scripts/questions.sh 8 eider-nemotron3.5-bolt
```

Set `NEMOTRON35_BOLT_MODEL` to use another local snapshot.
Explicit server arguments override launcher defaults.
The launcher enables three LiLiCorr drafts by default. Set `LILICORR_DIR` for
another companion path, or `EIDER_LILICORR=0` for generation without the companion.
The decision path shares prompt prefill and recurrent state, then reads selected
label logits through the compact output head. Image inputs are not supported.

On 2026-10-16, the community NVFP4 checkpoint passed an isolated HTTP smoke check.
All three decision types returned the expected factual answers.
Chat returned `4` for a non-thinking `2+2` request.
Repeated requests were deterministic. Grouped and isolated selected logits agreed
within `1e-3`. Concurrent chat and decisions completed with one admission slot.
The check also covered 64 questions, 64 labels, and rejection of 65 questions.
Probabilities remain uncalibrated.

The same parcel/book workload as Super used 384 logical input tokens and a
239-token shared prefix. With a 1,024-token prefill capacity and no prefix retention:

| Workload | HTTP latency |
| --- | ---: |
| Three questions together, median of three warm runs | 1,023 ms |
| Three separate requests, combined | 2,224 ms |
| 64 repeated questions, one run | 9,019 ms |

This run establishes basic serving and decision behaviour, not general model
quality or parity with NVIDIA's 2-bit release.
The focused CUDA fixture covers variable expert widths, W4A16 weights, absent MTP,
recurrent-state forks, and decision logits against isolated full-head inference.
The Nemotron tests passed: 27 checks, with three external-checkpoint tests skipped.
Deployment tests, launcher checks, release build, formatting, and Clippy passed.
Clippy reported existing warnings elsewhere in the inference crate.

### Generation performance

The 2026-10-16 community NVFP4 replay used greedy, non-thinking chat requests and
256 output tokens on one GB10. The prompts contained 1,795 and 7,051 tokens.
A third request reused 7,040 of its 7,061 prompt tokens. Prefill capacity was
2,048, with one active sequence and a 1 GiB retained-prefix budget.
The benchmark context limit was 32,768; the launcher still permits 262,144.

The companion raised decode throughput from about 20 to 28 tokens/s in these
requests. Three drafts performed better than one or seven. Acceptance averaged
1.43–1.57 drafts per verification cycle with three proposals. This narrow replay
does not establish performance across other prompts or parity with NVIDIA's
2-bit target, which trained the companion.

Wider W4A16 expert tiles reuse compressed weights across 32 rows. On the long
request, they raised prefill throughput from 606 to 750 tokens/s and reduced
complete HTTP latency from 21.9 to 19.7 seconds. Short-prompt latency changed
little. Both tile sizes produced identical text for all three requests.

Small FP8 verification projections use six scaled residual components rather
than a single quantized activation. This retains low input bits while sharing
weight reads across the verification batch. The real-checkpoint check matched
64 serial target tokens, including shortened verification blocks and rollback.
The drafter projects only the retained context window; its complete ring and
anchor matched incremental appends exactly. CUDA checks cover Bolt's attention
sinks against a CPU reference and retain the existing Super attention comparison.

For the community conversion, major costs include the 40 Mamba layers and
uncalibrated W4A16 expert execution. Fewer active experts alone do not remove
those costs. This conversion remained close to Super in the replay.

The [official 24GB checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB)
uses different formats. Its revision `c6cc2e8ffa353403213d6b9b18aaf8eb684d705a`
stores GGML-packed IQ2_XXS experts, NVFP4 Mamba projections, and Q8_0 dense
projections and output head. The inspected embedding tensor is BF16.
Native loaders and kernels retain the packed expert and dense weights, including
selected output rows for decisions. LiLiCorr can share the Q8_0 target head.
Focused checks compare packed projections against CPU decoding and exercise
mixed-format prefill followed by decode. The official checkpoint also passed
a 64-token comparison of speculative and serial decoding. The community conversion does not
preserve the official checkpoint's memory footprint or arithmetic.

The packed-kernel benchmark uses the 496-expert, 2,560-wide layer shape.
On GB10, IQ2_XXS up/down projections took 0.29/0.28 ms for 36 verification
routes and 14.0/15.2 ms for 18,432 prefill routes. Prefill timings include
route sorting, tile preparation, and activation preparation. These are kernel
measurements, not complete-request throughput. The prefill kernels keep weights
packed and overlap input copies with tensor-core execution.

The verification benchmark also covers four rows selecting the same nine
experts. That case took about 0.28 ms per IQ2_XXS projection.

The official target's NVFP4 Mamba projections use packed matvec for short batches
and prepared BF16 weights for prefill. At four rows, input/output projections
took about 0.25/0.18 ms. At 512 rows, the prepared path took 3.6/1.5 ms.
Preparation preserves the NVFP4 weight values but adds about 8.2 GiB of resident
BF16 copies across the 40 Mamba layers. That memory cost is additional to the
checkpoint's packed weights.

### Official 24 GB generation replay

The same three-request replay measured the native IQ2/Q8 target and its released companion.
The table uses the warm 7,051-token prompt and its 256-token response.
The cached request has 7,061 prompt tokens, with 7,040 reused.

| Runtime | Prefill tokens/s | Decode tokens/s | Complete request | Cached request |
| --- | ---: | ---: | ---: | ---: |
| Super NVFP4 | 832 | 26.3 | 18.9 s | 9.40 s |
| Bolt official, initial kernels | 516 | 30.1 | 22.6 s | 8.78 s |
| Bolt official, revised kernels | 560 | 37.0 | 19.9 s | 6.85 s |

Revised Bolt decoded at 34.1 tokens/s on the short prompt and 38.9 with prefix reuse.
Its retained text weights occupied about 29.0 GiB, including the prepared Mamba copies.
This excludes drafter weights and runtime workspaces.

Shared IQ2 verification now splits the reduction across quantization blocks.
At four rows, the 5,376-by-4,096 shared projections fell from about 0.20 ms to 0.065–0.070 ms.
Shared prefill uses 32-row tiles; those projections fell from about 8.7 ms to 6.6 ms.
Vector activation reads reduced routed verification projections from 0.28 ms to 0.18–0.21 ms.
Unaligned routed inputs retain scalar reads. Neither change expands the resident expert table.

The long response remained identical across the initial and revised kernels.
Short and cached responses changed after the reduction change; vector reads preserved all three revised responses.
The revised runtime matched 64 serial target tokens in the full-model speculative check.
CUDA checks compare against CPU decoding at the existing tolerance, including unaligned inputs.

Decode is now faster than Super on these requests. Uncached request latency still trails Super because IQ2 prefill remains slower.
Both models retain 40 Mamba layers and the same 5,376-wide shared experts.
Their common dense work limits the benefit from reducing the routed parameter count.

The Bolt launcher selects this official snapshot by default. Set
`NEMOTRON35_BOLT_MODEL` or `--model-dir` to use the community conversion.

### BF16 model execution

Both launchers select BF16 computation and recurrent state by default:

| Launcher | Model | Recurrent-state dtype |
| --- | --- | --- |
| `scripts/run-eider-nemotron35.sh` | Super | BF16, overriding the checkpoint's F32 setting |
| `scripts/run-eider-nemotron35-bolt.sh` | Bolt | BF16, as measured below |

The compute setting covers prefill, single-token decode, and speculative verification.
The generic server default remains F32. Recurrent-state storage can be selected independently.
Explicit `--nemotron-compute-precision` and `--nemotron-mamba-ssm-dtype` arguments override the launcher defaults.
Super's BF16 generation replay is recorded below.

IQ2/Q8 projections keep their packed checkpoint weights. Dense projections no longer
retain the additional 4.44 GiB of expanded BF16 copies. Mamba's existing prepared
NVFP4 weights remain resident. Projections use BF16 inputs and dequantized weights,
with F32 accumulation and BF16 output rounding. The router retains F32 arithmetic.
Small BF16 cuBLAS projections share an eight-row reduction shape across decode
and verification. Paged attention also shares one reduction shape for these batches.
These choices prevent batch-dependent rounding from changing acceptance.
NVFP4 decode and verification read packed weights; large prompt batches use the prepared Mamba copies.

Residual operands use BF16. Their F32 sum feeds the next normalization before
rounding the stored residual. Normalization rounds the normalized value before
multiplication by its BF16 weight. Mamba rounds its SSM output before F32 gating
and grouped normalization. These boundaries follow vLLM's compiled BF16 path.

For prefill batches of at least 64 tokens, Mamba uses 64-token SSD chunks.
Tensor-core products compute chunk states, C-B products, and outputs. Inter-chunk
state passing retains F32 arithmetic and writes the configured state dtype.
One bounded workspace is shared by all Mamba layers. Partial chunks and ragged
sequences retain separate initial and final states. Decode and speculative
verification retain the recurrent kernels and per-token rollback snapshots.

### BF16 generation replay

The complete BF16 path uses the same three-request replay described above.
These runs used three drafts, 2,048-token prefill batches, and 256 greedy output tokens.
The F32 row is the previous reference run. Generated text differs between precision modes.

| Mode | Long prefill tokens/s | Long decode tokens/s | Long request | Cached request |
| --- | ---: | ---: | ---: | ---: |
| Previous F32 reference | 574 | 36.8 | 19.62 s | 6.86 s |
| BF16 throughout, packed IQ2/Q8, chunked SSD | 1,006 | 33.7 | 14.91 s | 7.47 s |

The short request fell from 11.15 s to 8.71 s. Its BF16 decode rate was 39.7 tokens/s.
Long-prompt prefill improved by 1.75 times; it has not reached the two-times target.
The cached request was 9% slower. The Bolt launcher now selects BF16 despite this measured tradeoff.
The earlier prefill-only experiment reached 1,155 tokens/s but retained 4.44 GiB of extra dense-weight copies.
The current implementation removes those copies and covers decode and verification with the same BF16 boundaries.

Seven drafts reduced decode to 16–18 tokens/s across these requests.
Its higher acceptance did not offset verification cost, so the launcher retains three drafts.
Both draft settings passed the five arithmetic, deduplication, ordering, crash-recovery, and retrieval checks.
These checks are smoke coverage, not an accuracy comparison against NVIDIA's runtime.

Focused GB10 measurements explain the dispatch choices:

| Operation | Previous path | Current BF16 path |
| --- | ---: | ---: |
| Mamba recurrence, 1,024 rows | 4.98 ms | 2.75 ms, chunked SSD |
| Packed IQ2 shared up, 2,048 rows | 3.85 ms, 32-token BF16 tiles | 2.65 ms, 128-token BF16 tiles |
| Q8 output head, four rows | 5.16 ms, BF16 tensor cores | 3.22 ms, packed BF16 matvec |

Large dense tiles reuse packed weights across more tokens. Small Q8 and routed
projections use SIMT with BF16 operands and F32 accumulation. Shared IQ2 verification
retains tensor cores with a split reduction. CPU comparisons retain the existing numerical tolerances.

### Super BF16 packed FP8 decode

Super's small BF16 batches previously expanded every FP8 matrix before projection.
The packed kernel reads each weight once and reuses it across up to eight rows.
Inputs round to BF16; accumulation remains F32. Channel scales retain their existing meaning.
Prefill batches above eight rows retain the cuBLAS path and reserve expansion scratch.

The October 16 replay used BF16 computation and recurrent state, three drafts,
2,048-token prefill batches, and 256 greedy output tokens on one GB10.
Both runs used the same prompts and checkpoint. No other model ran concurrently.

| Request | Previous decode tok/s | Packed decode tok/s | Previous HTTP seconds | Packed HTTP seconds |
| --- | ---: | ---: | ---: | ---: |
| 1,795-token prompt | 16.65 | 30.88 | 17.88 | 10.59 |
| 7,051-token prompt | 15.31 | 30.35 | 24.38 | 15.94 |
| 7,061 tokens, 7,040 cached | 14.38 | 26.82 | 18.28 | 9.92 |

Long-prompt prefill changed from 948 to 963 tok/s. Removing expansion scratch
saved 1 GiB from the decode workspace; first-request state fell from 1.533 to 0.533 GiB.
Generated responses changed with the reduction order.

The `nemotron35_fp8_bf16` micromeasure checks numerical agreement before timing.
Four-row Mamba input takes 0.320 ms versus 0.667 ms for prepared BF16 cuBLAS.
The four-row LM head takes 2.251 ms versus 4.730 ms. These cuBLAS figures exclude expansion.
CPU-reference tests cover scalar and channel scales. Serial, batched, and segmented
projections match exactly for one through eight rows.

Replay artifacts are `/tmp/bolt-perf-super-bf16-fp8-before` and
`/tmp/bolt-perf-super-bf16-fp8-after`. The semantic probes passed four of five cases;
the deduplication prompt returned a total of 34 instead of 24.
The previous BF16 projection path returned the same incorrect result for that prompt.
That comparison is recorded in `/tmp/super-fp8-dedup-reference/response.json`.

### NVIDIA implementation comparison

The reference is NVIDIA's Bolt serving recipe and its pinned vLLM revision
`77154ca6116ae34cbfe314beef1016bde9b63989`.
The [b12x backend](https://github.com/vllm-project/vllm/blob/77154ca6116ae34cbfe314beef1016bde9b63989/docs/features/quantization/b12x.md)
keeps IQ2/Q8 matrices packed and uses BF16 activations. The
[Nemotron router](https://github.com/vllm-project/vllm/blob/77154ca6116ae34cbfe314beef1016bde9b63989/vllm/model_executor/models/nemotron_h.py)
explicitly requests FP32 computation and output.

| Component | NVIDIA recipe | Eider BF16 mode |
| --- | --- | --- |
| IQ2/Q8 projections | Packed weights, BF16 activations | Packed weights, BF16 activations throughout generation |
| Residual stream | BF16 model dtype | BF16 operands, F32 sum before normalization |
| Recurrent state | BF16 override | BF16 override available |
| Mamba prefill | Chunked SSD scan | 64-token SSD chunks for 64-channel, 128-state heads |
| Attention cache | FP8 | F32 by default; optional NVFP4 |

vLLM's [compiled normalization](https://github.com/vllm-project/vllm/blob/77154ca6116ae34cbfe314beef1016bde9b63989/vllm/ir/ops/layernorm.py)
normalizes the F32 residual sum. Its eager CUDA kernel rounds that sum first.
Eider follows the compiled path. The
[SSD implementation](https://github.com/vllm-project/vllm/blob/77154ca6116ae34cbfe314beef1016bde9b63989/vllm/model_executor/layers/mamba/ops/ssd_combined.py)
also separates chunk matrix products from state passing. This alignment does
not imply bitwise output parity with vLLM or matching attention-cache precision.

Focused CPU comparisons cover packed projections, residual normalization, Mamba
output rounding, ragged SSD chunks, both state dtypes, and rollback snapshots.
The real Bolt checkpoint matches 64 serial target tokens under LiLiCorr
verification with BF16 execution. Separate cases cover SSD prefill, F32 recurrent
state, and a trailing-space prompt that exposed batch-dependent attention rounding.

The focused Nemotron CUDA checks, router check, and four full-model comparisons pass.
Formatting and Clippy complete; Clippy reports existing warnings in other code.
Strict warning-as-error Clippy stops in the unchanged safetensors reader.
A broader non-GEMM run has one unresolved Qwen long-recurrence test failure:
`actual=0`, `expected=-0.00003282092`, `tolerance=0.00003`.
It also fails in isolation; no clean-HEAD comparison was made. No tolerance was changed.

## Released checkpoint

[NVIDIA's NVFP4 checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4)
is pinned for this inspection at `7d4bc747b6bdc2cff64c58d65da0d08dd9caa25c`.
Its 88 safetensors shards total 75,010,188,240 bytes. This includes vision weights,
which Eider does not load.

The text config has 88 layers, hidden width 4,096, 512 routed experts, and top-22
routing. Attention uses 32 query heads and two KV heads, each 128 elements wide.
The ModelOpt recipe mixes NVFP4 and FP8 linear weights. Downloaded shard headers
confirm BF16 router weights and FP32 correction biases. The model card's FP32
router wording does not describe the stored router-weight dtype.

The release stores tensor-wide quantization scales with both `[]` and `[1]`
shapes. The scalar F32 reader accepts both encodings and still requires exactly
four payload bytes. Its previous scalar-only check blocked the first FP8 Mamba
projection. Tests cover both encodings, malformed tensors, and mixed FP8/NVFP4
loading through the wrapped model fixture.

The config explicitly requests `mamba_ssm_cache_dtype: float32`. Eider now retains
FP32 selective state between tokens, including prefill, prefix checkpoints, and
speculative rollback. Previously, each update rounded that state to BF16.
Convolution state remains BF16. Configurations without the precision field retain
BF16 selective state; unsupported explicit values fail during config parsing.

Each Mamba layer needs 4 MiB of FP32 selective state per sequence. Speculative
verification also stores one state snapshot per verification row. That cost
must enter the memory budget for longer draft blocks.

## Prepared text path

Eider recognises the exact Omni architecture and validates its nested
`llm_config` or `text_config`. The language model uses the existing Nemotron-H
runtime. Tensor names retain the `language_model.` prefix through backbone,
LM head, and MTP loading. The loader does not rewrite snapshot files.

Each MoE layer reads its expert count from the BF16 router shape. The count must
fit the configured maximum, top-k selection, and expert groups. Router buffers,
NVFP4 expert tables, and MTP allocations use that count. Existing Super and Puzzle
configurations retain their layer widths and routing counts.

The change adds no catalogue entry. Local development checkpoints use
`--model-dir`. Recognition of the wrapper does not establish that every inner
configuration can run.

## Validation

Synthetic safetensors fixtures cover both nested configuration keys and the
original unwrapped format. A small GPU fixture loads Mamba, attention, NVFP4
experts, and MTP under the wrapped namespace. Its MoE layers contain four, three,
and two experts against a configured maximum of eight.

Single-token and batched GPU routing match a CPU reference. Tests reject invalid
expert counts, incompatible groups, conflicting nested configurations, and
workspaces from layers with different expert counts. API tests verify that local
deployment and inference select the same runtime.

These tests establish loader and routing behaviour. They do not establish model
quality, throughput, or parity with NVIDIA inference.

A fixture from the released text config covers its dimensions and FP32 state
setting. CUDA tests compare repeated decode and ragged prefill against a CPU
recurrence. They also check zero, partial, and full-acceptance state restoration
in both supported precisions. The inspection command reports actual state bytes,
including BF16 convolution storage.

The initial full-checkpoint smoke test loaded all 88 text layers plus MTP.
At that point, the loader reported 68.17 GiB of device weights. A local
server used a 4,096-token context limit and one active sequence. It requested
`--speculative-drafts 0`, but the scheduler ignored that setting for Nemotron.
Storage flags preserved the checkpoint's dense-weight precision:
`--nemotron-bf16-storage bf16 --nemotron-fp8-storage fp8`. Routed experts remained
NVFP4. The default dense-weight requantization path was not tested in this run.

Both Chat Completions and Responses returned `4` for a deterministic `2 + 2`
prompt. A tool request returned `multiply` with arguments `{"a":6,"b":7}`.
Repeating that 308-token request restored 256 cached prompt tokens and returned
the same arguments. The 30-token arithmetic prompt was below the 128-token cache
page size and retained no prefix.

The uncached tool request processed 307 prefill tokens in 13.01 seconds
(23.6 tok/s). Its 31-token output decoded at 20.5 tok/s. These short smoke
measurements describe the native mixed-precision path; they do not establish
representative throughput or validate the default storage settings.

## Prefill and decode performance

Prefill uses the checkpoint's calibrated activation scales for native FP8 dense
GEMMs and grouped NVFP4 expert GEMMs. Dense BF16 weights remain BF16. Their
projections use three BF16 components per FP32 activation, including the router.
Uncalibrated or unsupported projections retain the FP32-activation path.
Small expert batches retain the W4A16 path.

The expert up projection applies each expert's combined input and weight scale
before ReLU squared. The down projection uses its own calibrated input scale.
Packed weights remain shared with decode. Prepared weight-scale layouts add
about 6.73 GiB, bringing resident text weights and MTP weights to about 74.9 GiB.

Temporary activation buffers are shared between layers with matching shapes.
The service retains full-chunk workspaces and the current tail workspace. It no
longer retains a workspace for every prompt remainder. Grouped activation scales
reserve tiles for actual routes plus padding per expert. At 1,024 tokens, the two
expert scale buffers need about 62 MiB instead of 2.5 GiB.

Attention with 128-element heads splits the context across eight warps and merges
their online softmax results. Mamba keeps each lane's recurrent state in registers
through a chunk. It preserves the configured FP32 or BF16 state rounding. Prefix
checkpoints and speculative rollback still include convolution and recurrent state.

The LM head computes only each sequence's final prefill row. Speculative
verification retains all row logits. Decode captures consecutive Mamba and MoE
layers in CUDA graphs. Attention runs outside those graphs because its page and
position metadata change each token. `--speculative-drafts 0` disables MTP;
`3` enables the existing path.

### Focused measurements

CUDA-event medians below include quantization, conversion, and routing preparation
where the measured path requires them. Each benchmark checks correctness before
timing. Expert shapes use 512 experts, top-22 routing, latent width 1,024, and
intermediate width 2,688.

| Operation | Previous path | Current path |
| --- | ---: | ---: |
| FP8 Mamba input, 1,024 rows | 7.80 ms, expanded BF16 GEMM | 1.17 ms, calibrated FP8 |
| FP8 Mamba output, 1,024 rows | 3.62 ms, expanded BF16 GEMM | 0.61 ms, calibrated FP8 |
| BF16 router, 1,024 rows | 2.51 ms, SIMT | 0.35 ms, cuBLAS |
| Routed up, 1,024 tokens | 14.78 ms, W4A16 | 5.11 ms, W4A4 |
| Routed down, 1,024 tokens | 13.86 ms, W4A16 | 6.18 ms, W4A4 |
| Mamba recurrence, 1,024 tokens | 11.70 ms, generic snapshot kernel | 4.94 ms, register state |

The Mamba reference includes one initial state snapshot. Its optimized outputs
and final state match exactly in both supported state precisions. Attention is
about six times faster in the focused 128-row, 4K/16K-context cases. The attention
benchmark checks all outputs against the previous path and selected outputs
against an FP64 CPU reference.

Run these checks with:

```sh
cargo bench -p eider-cuda --bench nemotron35_dense
cargo bench -p eider-cuda --bench nemotron35_experts
cargo bench -p eider-cuda --bench nemotron35_nvfp4
cargo bench -p eider-cuda --bench nemotron35_attention
cargo bench -p eider-cuda --bench nemotron35_mamba
```

### Full-model replay

A controlled replay compared commit `4111ddd` with the new prefill path on GB10.
Both used the pinned checkpoint, native dense-weight storage, temperature zero,
MTP disabled, one active sequence, an 8,192-token context limit, and no prefix
retention. Nsight Systems traced CUDA execution in both runs. The previous path
used 128-token chunks; the new path used 1,024-token chunks.

| Request | Previous prefill | New prefill | Previous whole request | New whole request |
| --- | ---: | ---: | ---: | ---: |
| Colour retrieval, 5,069 prompt tokens | 45.235 s / 112.0 tok/s | 6.850 s / 739.8 tok/s | 45.727 s | 7.160 s |
| Repository notes and tool call, 3,419 prompt tokens | Not measured | 4.707 s / 726.1 tok/s | Not measured | 6.516 s |

The retrieval prompt repeats blue-key/small-door and red-key/large-door notes
315 times, then asks which key opens the small door. Both paths returned `blue`.
The tool request includes the repository's `AGENTS.md` and asks for
`multiply(a=6, b=7)`. The new path returned those arguments and 31 output tokens.
Its decode rate was 17.1 tok/s.

These single-request measurements establish a 6.6-fold prefill improvement for
the matched retrieval prompt. They are separate from the user's Pi trace, which
reported 93.1 tok/s on a different 5,105-token prompt. Independent reference
logit parity and broader model-quality evaluation remain pending.

The final launcher comparison included the router GEMM change, a 1,048,576-token
context limit, and 1 GiB of prefix retention. Actual prompts remained 3.4K–5.1K
tokens; this does not test generation at a million-token context length.

| Request | 1,024-token chunks: prefill / whole request | 4,096-token chunks: prefill / whole request |
| --- | ---: | ---: |
| Cold colour retrieval, 5,069 tokens | 741 tok/s / 7.035 s | 763 tok/s / 8.840 s |
| Cold tool call, 3,419 tokens | 706 tok/s / 6.632 s | 778 tok/s / 7.416 s |
| Tool call, 3,328 cached tokens | 0.670 s TTFT / 2.416 s | 0.673 s TTFT / 2.440 s |
| Uncached prose, 5,077 tokens | 778 tok/s / 7.165 s | 850 tok/s / 6.620 s |

Both chunk sizes returned the expected colour, explanation, and tool arguments.
The 4,096-token buffers added about two seconds before the first prefill started.
The launcher therefore defaults to 1,024 tokens. Its context limit remains
1,048,576 tokens. `EIDER_PREFILL_TOKEN_CAPACITY` or `--prefill-token-capacity`
can override the chunk size. Decode on the tool requests was about 17 tok/s.
Very short uncached suffixes still use W4A16 experts and have lower throughput:
the cached tool request processed its remaining 90 prefill tokens at 141 tok/s.

Validation covers Nemotron unit and GPU tests, serial versus batched state,
following-token decode, variable expert routers, packed activation bytes,
ragged paged attention, and the five focused benchmarks. Compute Sanitizer
reports zero errors for quantization, shared prefill scratch, and attention.
The shared grouped-scale layout also passes the existing RMS/gather quantization
regression. Formatting, the release server build, and Clippy pass; Clippy reports
existing `chunks_exact` suggestions. The full workspace test suite was not run.

## Remaining checkpoint work

- Compare logits and routing against an independent reference. Cover longer
  prompts, prefix restoration, graph replay, and MTP execution. Local tests cover
  ragged prefill and graph replay across a KV page boundary.
- Validate default dense-weight requantization and measure prefill and decode
  on representative workloads.
- Extend template coverage to reasoning, tool results, and streaming through
  both HTTP APIs.
- Assess native Bolt 2-bit execution separately from the NVFP4 conversion.

The released LiLiCorr companion is available through `EIDER_LILICORR=1` in the
launcher. See [LiLiCorr configuration, validation, and measurements](nemotron-lilicorr.md).

Vision and video execution remain unimplemented. The upstream work includes a
RADIO encoder, a projector, position-table interpolation, and a separate temporal
patch projection. Eider currently serves only text.

## Decisions

The text runtime serves `/v1/decisions` with the same question types and limits as
Gemma and LFM. Each request processes its shared prefix once. Question branches
share complete KV pages and copy the partial tail. Each branch owns its Mamba
convolution and selective state, with the checkpoint's original precision.

Suffix batches share expert execution. Dense projections retain each question's
matrix shape because different reduction orders can cross FP8/FP4 quantization
boundaries. A packed dense trial changed selected logits by up to 1.21 on the
three-question check. That trial is excluded. Short suffix chunks run separately
to preserve their kernel choices. Reusable CUDA graphs cover the non-attention
layers, and the service retains at most four workspace shapes.

A compact head reads the 64 answer labels from the final hidden state. Decisions
allocate no MTP or LiLiCorr sequence state and generate no reasoning or answer
tokens. They share admission, cancellation, and prefill limits with chat requests.
A group holds one admission slot; `--decision-branch-capacity` bounds its internal
branches. Probabilities remain uncalibrated unless a decision calibration profile is loaded.

Start the server and run the question sweep:

```sh
EIDER_LILICORR=0 scripts/run-eider-nemotron35.sh
scripts/questions.sh 8 eider-nemotron3.5-super
```

The normal launcher also supports decisions with LiLiCorr loaded for chat.
Decision requests do not execute the drafter.

On GB10, the three-question HTTP check took 1,043 ms median versus 2,316 ms for
three separate requests. The median covers five runs after one warm-up. The
request had 384 logical input tokens, including a 239-token shared prefix.
A separate 64-question request took 8,945 ms with 2,927 logical input tokens.

These runs used the launcher's BF16/FP8 dense storage, NVFP4 experts, and a loaded
LiLiCorr companion. Prefill capacity was 1,024 tokens, branch capacity was eight,
and admission capacity was one. Prompt retention was disabled. The complete-model
check reported zero selected-logit error against isolated decisions and full-head
readout. These checks cover one factual prompt, not general decision quality.

The focused checkpoint test compares grouped, isolated, and full-head logits.
It also covers reversed question order and cancellation with one admission slot.

```sh
checkpoint="$HOME/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/7d4bc747b6bdc2cff64c58d65da0d08dd9caa25c"
NEMOTRON_MODEL_DIR="$checkpoint" cargo test --release -p eider-inference --lib \
  nemotron_decisions_match_isolated_full_heads_and_cancel_cleanly -- \
  --ignored --test-threads=1 --nocapture
```
