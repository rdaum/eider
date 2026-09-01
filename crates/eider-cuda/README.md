# eider-cuda

`eider-cuda` owns CUDA resources, GPU storage, cuBLASLt plans, and SM121 kernels
for Eider. It targets the GB10 GPU in NVIDIA DGX Spark systems.

## Responsibilities

The crate provides:

- CUDA streams, events, graphs, and device buffers
- deferred command ownership through recording, in-flight, and bounded-slot
  state transitions
- ModelOpt-to-device preparation boundaries
- cuBLASLt plans for BF16, FP8, INT8, and NVFP4 matrix products
- SM121 W4A16 and non-GEMM kernels
- device-resident DFlash2 projection, top-k, and coherent path selection
- diagnostic smoke checks and GPU counter collection.

The crate does not select a model family, schedule requests, or parse an HTTP
request.

## NVFP4 rules

GB10 uses the SM121 NVFP4 MMA instruction. Its hardware tile, scale-vector
shape, and scale type are fixed. Keep checkpoint, logical, cuBLASLt, shared
memory, and native-MMA scale layouts separate.

Prepare and cache layout conversions before decode. Do not add conversion work
to a decode hot path. Keep CUDA stream ownership explicit.

## Deferred execution

`Recording<B, R>` owns one backend encoder and all resources referenced by its
work. Submission moves both into `InFlight<B, R>`. Callers can reclaim `R` only
after `poll` or `wait` observes completion. `BoundedExecutionSlots` applies the
same rule to a fixed set of reusable workspaces.

`CudaBackend` records through `CudaPass`. CUDA launches are eager, so a failed
or discarded recording remains in flight until the stream completes. Its
fence queries the existing stream and does not allocate an event. Pinned host
readback uses this lifecycle directly.

## Build and test

Configure the local CUTLASS build when a kernel needs it:

```sh
scripts/setup-cutlass-sm12x.sh
source .deps/cutlass-sm12x.env
cargo test -p eider-cuda --lib
```

Run a focused benchmark for each shape-specific kernel change. Benchmarks use
`micromeasure` and must establish correctness before timing.

```sh
cargo bench -p eider-cuda --bench sm121_w4a16_routed_gate_up
cargo bench -p eider-cuda --bench dflash2_selector
```

The default build compiles Eider kernels with NVCC. The optional `cuda-oxide`
feature selects cuda-oxide kernels at compile time.

The cuda-oxide path contains the Qwen3.8 27B target-prefill, target-decode, and
DFlash2 kernels. It includes W4A16, compact FP4 KV, GDN, IMRoPE, sampling,
fused LM-head top-1, and DFlash2 operations. Both builds use the same safe Rust
API and device layouts.

The Qwen3.8 27B path does not dispatch custom Eider kernels built with NVCC.
It still uses the CUDA driver, CUDA runtime, and cuBLASLt. The server build can
include NVCC and CUTLASS kernels for other model paths.

The separate Qwen3.8 Flash Next runtime has cuda-oxide hyperconnection, PLE,
and QSA primitives. Its complete execution path has not yet been audited.

The `eider-api/cuda-oxide` feature enables this backend in a production server
build.

```sh
scripts/setup-cuda-oxide.sh
scripts/run-eider-qwen38.sh --cuda-oxide --offline
```

See [`../../backends/cuda-oxide/README.md`](../../backends/cuda-oxide/README.md)
for the cuda-oxide requirements and build commands.

CAUTION: GB10 device allocations use the same 128 GB unified memory as the
host. Do not start another full model while a server is active.
