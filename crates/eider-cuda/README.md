# eider-cuda

`eider-cuda` owns CUDA resources, GPU storage, cuBLASLt plans, and native
SM121 kernels for Eider. It targets the GB10 GPU in NVIDIA DGX Spark systems.

## Responsibilities

The crate provides:

- CUDA streams, events, graphs, and device buffers
- deferred command ownership through recording, in-flight, and bounded-slot
  state transitions
- ModelOpt-to-device preparation boundaries
- cuBLASLt plans for BF16, FP8, INT8, and NVFP4 matrix products
- native SM121 W4A16 and non-GEMM kernels
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
```

CAUTION: GB10 device allocations use the same 128 GB unified memory as the
host. Do not start another full model while a server is active.
