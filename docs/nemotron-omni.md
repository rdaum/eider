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

The complete released checkpoint passed topology validation and loaded all 88
text layers plus MTP. The loader reported 68.17 GiB of device weights. A local
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

The performance path preserves checkpoint weight precision and FP32 selective
state. Dense FP8 and NVFP4 projections decode weights into BF16 tensor operands.
Three BF16 components represent each FP32 activation. This changes accumulation
order without applying another low-bit quantizer to the activations.

Prefill groups routes by expert and reuses the pipelined expert GEMM. The LM head
computes only the final row for each sequence. Speculative verification still
retains all row logits. Final hidden states remain available for MTP.

Decode captures consecutive Mamba and MoE layers in CUDA graphs. Attention runs
outside those graphs because its page and position metadata change each token.
The scheduler now honours `--speculative-drafts 0`; `3` enables the existing MTP
path. Other nonzero draft counts fail with a configuration error.

Repeated prefill exposed a metadata-upload race on the non-blocking stream.
Pinned batch metadata and page tables now wait for their previous uploads before
host reuse. The failing third HTTP request now completes. Compute Sanitizer
reports zero errors for the focused workspace-reuse regressions.

The following CUDA-event medians include route sorting for expert GEMMs.
Each benchmark compares every output against the existing SIMT reference before
timing. Expert cases use 512 experts and 128 tokens with 22 routes each.

| Projection | Output × input | SIMT | Tensor cores |
| --- | ---: | ---: | ---: |
| FP8 Mamba input, 128 rows | 18,560 × 4,096 | 40.24 ms | 8.32 ms |
| NVFP4 Mamba input, 128 rows | 18,560 × 4,096 | 30.33 ms | 5.63 ms |
| FP8 Mamba output, 128 rows | 4,096 × 8,192 | 16.73 ms | 1.51 ms |
| NVFP4 routed up | 2,688 × 1,024 | 19.94 ms | 6.51 ms |
| NVFP4 routed down | 1,024 × 2,688 | 18.74 ms | 7.47 ms |

Run the focused checks with:

```sh
cargo bench -p eider-cuda --bench nemotron35_dense
cargo bench -p eider-cuda --bench nemotron35_experts
```

A full-model HTTP replay compared commit `ae38fa7` with this performance path.
Both used the pinned checkpoint above, native dense precision, a 4,096-token
context limit, 128-token prefill chunks, and one active sequence. Prefix retention
was 1 GiB. Each request used temperature zero and a 128-token output limit.

| Request | Previous prefill | Current prefill | Previous whole request | Current whole request |
| --- | ---: | ---: | ---: | ---: |
| Tool call, 308 prompt tokens | 11.988 s | 2.990 s | 13.906 s | 5.253 s |
| Same tool call, 256 cached tokens | 1.980 s | 0.678 s | 3.466 s | 2.455 s |
| Unrelated prose, 546 prompt tokens | CUDA error 700 | 4.707 s | Failed | 8.415 s |

The tool request asked for `multiply(a=6, b=7)`. Both versions returned the same
arguments and 31 output tokens. The prose request repeated blue-key/small-door
and red-key/large-door notes 32 times, then requested eight explanatory sentences.
The current path returned a coherent 62-token answer.

Cold tool prefill improved from 25.6 to 102.7 tok/s. The longer prose request
reached 115.8 tok/s. These are short, single-run measurements with 128-token
chunks; larger batches and longer contexts need separate measurement.

The previous server decoded the tool output at 19.8–20.6 tok/s, but could run
MTP despite the zero-draft setting. A separate build disabled graph capture while
retaining all other changes. Both current builds kept MTP disabled:

| Request | Eager decode | Graph decode |
| --- | ---: | ---: |
| Cold tool call | 16.74 tok/s | 17.31 tok/s |
| Cached tool call | 16.64 tok/s | 17.18 tok/s |
| Prose | 16.07 tok/s | 16.59 tok/s |

Outputs matched exactly between those builds. The measured graph gain was about
3%, with capture cost on the first token. Decode remains the next performance
target. The older MTP-enabled result is not a direct graph comparison.

Validation includes the Nemotron unit and GPU tests, shared page-table tests,
both focused benchmarks, and the HTTP replay above. Formatting, the release
server build, and Clippy pass. Clippy still reports existing `chunks_exact`
suggestions outside this change. The full workspace test suite was not run.

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
