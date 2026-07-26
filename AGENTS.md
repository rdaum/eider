# Eider

CUDA inference and serving for NVFP4 and mixed-precision models on NVIDIA DGX
Spark / GB10 (`sm_121`). The workspace contains the `eider-api` server, the
multi-model `infer` runtime, and the `nvfp4` CUDA kernel crate. Supported model
families currently include Qwen3.5/3.6 MoE, Step-3.7, Laguna-S-2.1, Gemma 4,
Nemotron 3, and DeepSeek V4 Flash.

## Build / run

```sh
cargo build --workspace
cargo build --release -p eider-api --bin eider-serve
cargo test --workspace

scripts/run-eider
cargo run --release -p eider-api --bin eider-serve -- \
    gemma-4-26b-a4b-nvfp4 --offline
```

Catalogue starts resolve pinned Hugging Face revisions and keep snapshots
immutable. Derived model artifacts belong below the Eider XDG cache, not in a
snapshot or the repository. Use `--model-dir` only for local development
checkpoints. Catalogue IDs select deployments; API requests use the served
model names reported by `eider model list` or `/v1/models`.

The server exposes `/v1/responses` and `/v1/chat/completions`; both translate
into the same `ChatRequest`, inference actor, scheduler, prefix cache, and model
runtime. Keep cancellation, tool history, sampling, usage, and finish-reason
semantics aligned across both adapters. Pi launchers default to Responses; set
`PI_EIDER_PROVIDER=eider-chat` to exercise Chat Completions.

The Qwen3.6 fast MoE path is enabled by default: indexed CUTLASS W4A4 gate/up
for single-row decode, grouped W4A4 for larger batches, SM12x down, segmented
decode graph capture, and the shared radix prompt-prefix cache. Compatibility
flags from older experiments are not required for normal runs.

## Microbenchmarks

```sh
cargo bench -p nvfp4 --bench sm121_w4a16_routed_gate_up
cargo bench -p nvfp4 --bench qwen36_routed_moe_decode
cargo bench -p infer --bench qwen36_prefill
cargo bench -p infer --bench step37_prefill
cargo bench -p infer --bench laguna_prefill
cargo bench -p nvfp4 --bench gemma4_prefill_attention
cargo bench -p nvfp4 --bench nemotron3_dense_linear
```

The benches use my [`micromeasure`](https://github.com/rdaum/micromeasure) crate
and should validate correctness before timing.
Keep shape-specific kernel changes backed by a focused benchmark and compare
the result against the existing reference path.

## CUTLASS / CUDA

`scripts/setup-cutlass-sm12x.sh` configures CUTLASS for `sm_121` under the
repo-local, git-ignored `.deps/` directory:

```sh
scripts/setup-cutlass-sm12x.sh
source .deps/cutlass-sm12x.env
```

The build defaults are CUDA 13.0, `.deps/cutlass`, and
`.deps/cutlass-build-sm121`; those paths are used automatically when present.
`scripts/probe-cutlass-sm12x.sh` checks CUTLASS SM12x compile support.

## TLDR SM121

For CUDA and CUTLASS work on GB10, keep these
[SM12x NVFP4 notes](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/)
in view:

- SM121 is Blackwell, but it is not the SM100/B200 programming model. Its tensor
  cores use synchronous, warp-collective `mma.sync` with operand and accumulator
  fragments in registers; it has no `tcgen05` or TMEM. SM100 GEMM kernels are
  incompatible. SM8x mainloop, scheduling, and pipelining ideas are the more
  useful starting point, with SM12x additions such as TMA.
- Build SM121 architecture-specific MMA translation units with
  `compute_121a,code=sm_121a`, as `crates/nvfp4/build.rs` does. Do not silently
  compile them under a conservative generic target. Conversely, do not assume
  an SM120a/f-only feature such as warpgroup register reallocation exists on
  GB10; confirm support with the installed CUDA/CUTLASS toolchain and a device
  probe.
- The hardware NVFP4 atom is fixed:
  `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`.
  It consumes E2M1 A `16x64`, E2M1 B `64x8`, UE4M3 SFA `16x4`, UE4M3 SFB
  `4x8`, and FP32 accumulators. The `m16n8k64` shape, `4X` scale vector, and
  UE4M3 scale type are not tuning choices.
- NVFP4 means packed E2M1 values with one UE4M3 scale per 16 values along K.
  E2M1 has only `0`, `±0.5`, `±1`, `±1.5`, `±2`, `±3`, `±4`, and `±6`; it has
  no NaN or infinity encoding, and UE4M3 scales are nonnegative.
  Logically, A `[M,K]` has SFA `[M,K/16]` and B `[K,N]` has SFB `[K/16,N]`.
  One MMA K atom therefore consumes four scales. An unguarded MMA mainloop still
  needs K in 64-element atoms; handle or pad tails rather than treating
  divisibility by 16 as sufficient.
- Keep logical, checkpoint, cuBLASLt, CUTLASS, shared-memory, and native MMA
  scale layouts distinct. The hardware fixes the register fragment layout, not
  the global- or shared-memory layout. ModelOpt row-major scales are not
  cuBLASLt `VEC16_UE4M3` scales and neither can be reinterpreted as a native MMA
  layout. Convert once while preparing/caching weights, not in the decode hot
  path.
- Scale register ownership is quad-based. With CUTLASS's selector convention,
  lanes 0 and 1 of each four-lane quad supply SFA while lane 0 supplies SFB.
  CUTLASS layouts consequently replicate SFA registers 2x and SFB registers 4x;
  that apparent duplication is part of feeding the instruction correctly.
  Preserve the `byte-id`/`thread-id` selectors and validate lane-to-scale
  mapping when writing inline PTX.
- Pipeline A, B, SFA, and SFB together. A/B commonly use TMA to shared memory
  and `ldmatrix` into registers, while scale factors use the SM12x scale-factor
  TV layouts and a universal copy. In CuTe code, use the SM120 scale-layout
  helpers and remove broadcast coordinates (for example with `filter_zeros`)
  rather than hand-deriving a superficially contiguous copy.
- `mma.sync` is a warp collective: every participating lane must execute it
  consistently. Keep divergent bounds and routing decisions outside the MMA,
  and make register pressure, shared-memory staging, pipeline depth, and the
  number of MMA warps explicit tuning dimensions.
- Do not transfer published RTX Pro 6000 tile choices or throughput numbers to
  DGX Spark. CTA-wave quantization depends on the device's SM count and problem
  shape, and published GEMM figures may exclude quantization and layout
  conversion. Count waves for GB10, include preparation costs when they occur at
  runtime, and gate changes on both a correctness-first micromeasure and the
  real prefill/decode path.

## Layout

- `crates/eider-api/src/` — catalogue deployment, inference actor, HTTP APIs,
  streaming protocol adapters, and server telemetry.
- `crates/infer/src/runtime/` — shared scheduling, prefix/KV caches, sampling,
  chat rendering, output parsing, and serving state.
- `crates/infer/src/{qwen3,step37,laguna,gemma4,nemotron3,deepseek4}/` —
  family-specific model loading and execution.
- `crates/infer/benches/` — model-runtime and prefill micromeasures.
- `crates/nvfp4/src/cublaslt/` — cuBLASLt descriptors and matmul plans.
- `crates/nvfp4/src/kernels/` — Eider SM121 W4A16, non-GEMM, and SM12x operations.
- `crates/nvfp4/src/diagnostics/` — smoke checks and GPU-counter helpers.
- `crates/nvfp4/src/` — storage, checkpoint, CUDA, and FFI support.
- `crates/nvfp4/native/` — CUDA kernels and native FFI implementations.
- `crates/nvfp4/benches/` — focused NVFP4, SM121 W4A16, attention, and GEMV
  micromeasures.
- `pi/agent/models.json` — repository-local Pi providers for Responses and Chat
  Completions.
- `scripts/` — model/Pi launchers, checkpoint preparation, CUTLASS probes, and
  external vLLM comparison helpers.
- `docs/` — benchmark findings, kernel notes, and model-specific proposals.

## Engineering policy

- Keep changes narrow, explicit, and grounded in the current runtime and kernel
  boundaries.
- Prefer one coherent current API over compatibility wrappers or parallel old
  and new paths. Preserve an older path only when it is still an intentional
  benchmark or fallback.
- Do not design for hypothetical downstream users or hardware without a
  concrete requirement.
- Names should describe current behaviour, not history or an implementation
  accident. Keep comments, commit messages, and documentation factual.
- Avoid generic abstractions until they remove real duplication or clarify a
  stable boundary.

## Performance policy

- Treat performance as a design constraint in the runtime and CUDA paths.
- Keep hot-path allocations, synchronizations, host/device transfers, and
  temporary representations visible and intentional.
- Measure optimization claims with a focused micromeasure or end-to-end decode
  benchmark; do not infer a runtime win from an isolated kernel result.
- Prefer compact, cache-friendly layouts and existing workspace plans over new
  layers of indirection.
- Add focused correctness coverage for new kernels and regression cases.
- GB10 device allocations consume the same 128 GB unified memory used by the
  host. Do not start another full model, vLLM instance, or high-memory benchmark
  while a server is running. Never assume a near-1.0 vLLM GPU-memory ratio is
  safe on this architecture.

## Rust style

- Format with `cargo fmt --all`.
- Prefer early returns, `let else`, and guarded matches over deep nesting.
- Keep imports at the top of the module and errors precise.
- Keep test output clean and prefer real logic over mocks.
- Use Canadian English in documentation and comments.

## Review checklist

Before handing work back:

1. Does the change fit the current runtime, kernel, and benchmark boundaries?
2. Did it avoid stale compatibility scaffolding and unnecessary allocations?
3. Are new behaviours covered by focused tests or correctness checks?
4. Were `cargo fmt --all`, relevant tests, and Clippy run, or is a gap stated?
5. Are docs and comments concise, current, and factual?

## Correctness notes

- Qwen3.6 expert gate/up scaling is per expert and must be applied before
  SiLU; the down path applies its per-expert scale during accumulation.
- Step-3.7 routed experts use bounded residency and disk-backed paging; its
  dense and shared weights remain resident. Preserve paging telemetry and do
  not silently turn paging misses into synchronous default-stream work.
- Laguna-S-2.1 alternates global and sliding-window attention, gates attention
  heads, and routes ten experts with a learned correction bias. Preserve the
  checkpoint's layer pattern and routing order.
- Gemma 4 uses heterogeneous local/global attention. Preserve the checkpoint's
  per-layer attention pattern and validate both prefill and decode when changing
  its shared kernels.
- Nemotron 3 combines attention, Mamba recurrent state, latent MoE, and an
  optional MTP block. Do not treat its sequence state as a conventional KV-only
  cache.
- DeepSeek V4 combines sliding, compressed sparse, and hierarchical compressed
  attention with mHC streams, hash and learned routers, and shared experts.
  Preserve the checkpoint's layer schedule and routing order. Its complete
  routed table is resident in blockwise Q3; original NVFP4 hot overlays must
  keep gate/up and down mappings consistent.
- Keep CUDA stream semantics explicit. Non-blocking streams do not synchronize
  with the default stream.
- Do not add permanent debug flags or probe prints. Do not revert unrelated
  user changes.
