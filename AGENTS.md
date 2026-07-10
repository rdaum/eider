# Eider

CUDA inference and kernel research for Qwen models on NVIDIA GB10 / Blackwell
(`sm_121`). The workspace contains the `infer` runtime and the
`nvfp4` CUDA/NVFP4 kernel crate.

## Build / run

```sh
cargo build -p infer
cargo build --release -p infer
cargo test --workspace

cargo run --release -p infer --bin qwen36-generate -- \
    models/qwen3.6-35b-a3-nvfp4 "What is 2+2?" 30
```

The Qwen3.6 fast MoE path is enabled by default: Marlin W4A16 gate/up,
SM12x down, grouped workspace allocation, and segmented decode graph capture.
The environment flags are retained for compatibility with older experiments,
but are not required for normal runs.

## Microbenchmarks

```sh
cargo bench -p nvfp4 --bench marlin_routed_gate_up
cargo bench -p infer --bench qwen36_routed_gate_up
cargo bench -p nvfp4 --bench lm_head_top1
cargo bench -p nvfp4 --bench sm12x_indexed_gemv
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

## Layout

- `crates/infer/src/` — model loading, Qwen runtime, decode, and probes.
- `crates/infer/src/runtime/` — reusable KV-cache, sampling, and generation
  state.
- `crates/infer/src/qwen3/` — Qwen model formats, layers, and decoding.
- `crates/infer/benches/` — runtime and routed-MoE micromeasures.
- `crates/nvfp4/src/cublaslt/` — cuBLASLt descriptors and matmul plans.
- `crates/nvfp4/src/kernels/` — Marlin, non-GEMM, and SM12x operations.
- `crates/nvfp4/src/diagnostics/` — smoke checks and GPU-counter helpers.
- `crates/nvfp4/src/` — storage, checkpoint, CUDA, and FFI support.
- `crates/nvfp4/native/` — CUDA kernels and native FFI implementations.
- `crates/nvfp4/benches/` — focused NVFP4, Marlin, attention, and GEMV
  micromeasures.
- `scripts/` — CUTLASS setup/probes and external vLLM comparison helpers.
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
- Keep CUDA stream semantics explicit. Non-blocking streams do not synchronize
  with the default stream.
- Do not add permanent debug flags or probe prints. Do not revert unrelated
  user changes.
