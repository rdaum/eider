# Nemotron Super 3.5 VL and Bolt text support

Status: released NVFP4 checkpoint loads and passes text-serving and prefix-reuse
checks, 2026-10-15. Batched prefill uses tensor cores. Independent reference parity
and vision support remain pending.

[llama.cpp PR #30520](https://github.com/ggml-org/llama.cpp/pull/30520)
adds `NemotronH_Omni_Reasoning_V3` for Super 3.5 VL and Bolt 74B-A9B.
The inspected head is `e64207bab6c2989a4ae5aa8b9e0cbe700e135ee3`.
The GitHub API reports that the PR merged on 2026-10-15.

[NVIDIA announced](https://blogs.nvidia.com/blog/local-ai-rtx-spark-laptops-software-updates/)
Bolt 74B-A9B for October 15. The advertised 24 GB checkpoint uses 2-bit
quantization and quantization-aware training. That announcement does not establish
an NVFP4 checkpoint format. Super 3.5 VL now has an official ModelOpt NVFP4
checkpoint, described below.

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
- Assess Bolt separately if its weights become available. Native 2-bit support
  requires its own format and kernel assessment.

The released LiLiCorr drafter requires a separate integration. See
[the format and implementation assessment](nemotron-lilicorr.md).

Vision and video execution remain unimplemented. The upstream work includes a
RADIO encoder, a projector, position-table interpolation, and a separate temporal
patch projection. Eider currently serves only text.
