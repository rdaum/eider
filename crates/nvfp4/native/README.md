# Native CUDA Layer

This directory contains the native implementation behind the `nvfp4` Rust
crate. The Rust API and `ffi.rs` own shape validation, buffer ownership, and
stream arguments; these files own CUDA kernels and the small `extern "C"`
entrypoints called by those wrappers.

## Translation Units

| File | Responsibility |
| --- | --- |
| `non_gemm.cu` | Elementwise, quantization, RoPE, attention, MoE routing, lm-head, FP8, GDN, and W4A16 kernels. |
| `ngram.cu` | Rowwise BF16, FP8, and NVFP4 n-gram gathers plus fused input projection. |
| `sm12x_mma.cu` | Experimental and production SM12x NVFP4 MMA/GEMV kernels. |
| `sm121_w4a16.cu` | Eider-owned SM121 W4A16 routed and dense tensor-core kernels. |
| `cutlass_gemv.cu` | Optional CUTLASS SM12x GEMV implementation. |
| `cutlass_gemv_stub.cpp` | Fallback symbols when CUTLASS is unavailable. |
| `fp4_oracle.cpp` | Host/CUDA conversion oracle used by format checks. |
| `gpu_counters.cpp` | CUPTI/NVPerf diagnostic counter collection. |

`build.rs` compiles the CUDA translation units for GB10 (`sm_121`, with the
SM12x MMA unit using `compute_121a`) and archives them with the host helpers.
Keep the FFI symbol names stable when reorganizing implementation files; the
Rust wrappers are the API boundary.

## `non_gemm.cu` Sections

The large general-purpose translation unit is organized in this order:

1. shared device helpers and NVFP4/UE4M3 quantization;
2. elementwise, normalization, residual, and layout kernels;
3. RoPE and Qwen3.6 attention preparation;
4. MoE routing, grouped pointer gathering, activation, and accumulation;
5. RoPE, KV-cache, and attention kernels;
6. lm-head reductions and BF16 conversion;
7. FP8 and Gated Delta Net kernels;
8. Qwen3.6-specific GDN helpers and W4A16 matvec/top-1 kernels.

New kernel families should get a dedicated translation unit once they have
multiple helpers or a distinct build/architecture requirement. Keep launch
validation at the Rust boundary and keep native entrypoints thin.

## Stream and Memory Rules

- Every asynchronous entrypoint accepts an explicit CUDA stream.
- Do not assume a non-blocking stream synchronizes with the default stream.
- Device pointers are borrowed for the duration of enqueued work; wrappers
  must keep their owning buffers alive until the stream has completed.
- Host readbacks and synchronization belong in explicit diagnostic or fallback
  paths, not hidden inside the steady-state decode kernels.

## Qwen MRoPE Invariant

Interleaved MRoPE selects whether each rotary pair uses the temporal, height,
or width position. It does not renumber the rotary frequency: pair `i` always
uses `theta^(-2*i/rotary_dim)`. Text decode supplies equal temporal, height,
and width positions, so it must reduce exactly to ordinary partial Neox RoPE.
