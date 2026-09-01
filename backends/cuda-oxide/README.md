# cuda-oxide backend

This crate contains cuda-oxide device kernels for Eider. It is a separate
workspace because cuda-oxide needs nightly Rust and custom compiler components.

The stable Eider workspace does not link the nightly Rust runtime. The
`eider-cuda` build script uses cuda-oxide to produce PTX. It then uses `ptxas`
to produce an `sm_121a` CUBIN and embeds that CUBIN in `eider-cuda`.

The `cuda-oxide` feature selects these kernels at compile time. The stable host
API, CUDA streams, device buffers, and weight layouts do not change.

The production paths support Qwen3.8 27B and Qwen3.8 Flash Next. The dense path
supports target prefill, target decode, and DFlash2 speculation. It includes
these Eider kernel groups:

- dense SM121 W4A16
- compact FP4 KV cache and attention
- dense Qwen attention and recurrent or chunked GDN operations
- interleaved MRoPE for dense Qwen decode and prefill
- fused NVFP4 LM-head top-1 selection
- FP8 and NVFP4 activation preparation
- token sampling and DFlash2 path selection
- DFlash2 convolution, attention, projection, and state capture.

Flash Next adds these kernel groups:

- Qwen Sparse Attention and its index cache
- paged BF16 PLE row gathering and PLE transforms
- hyperconnection mixing and finalization
- normalized 512-expert top-10 routing
- W4A16 routed MoE decode
- grouped W4A4 routed MoE prefill
- routed and shared FFN finalization.

Both Qwen3.8 paths avoid custom Eider kernels built with NVCC. They still use
the CUDA driver, CUDA runtime, and cuBLASLt.

The server build can include NVCC and CUTLASS kernels for other model paths.
The operations in the list do not use silent native-kernel fallbacks.

Use the matching launcher for each model:

```sh
scripts/run-eider-qwen38.sh --cuda-oxide --offline
scripts/run-eider-qwen38-flash-next.sh --cuda-oxide --offline
```

## Requirements

- CUDA 13.0 with `nvcc` and `ptxas`
- Rust nightly `nightly-2026-08-28`
- `cargo-oxide` from cuda-oxide revision
  `97f8b2b7882f0c15ad9ce9b53abed5553920caa8`
- an NVIDIA GB10 GPU with compute capability 12.1

The toolchain file pins the required nightly release. The crate lock file pins
the cuda-oxide dependencies.

## Build

Install the pinned `cargo-oxide` release in the repository:

```sh
scripts/setup-cuda-oxide.sh
```

The script installs the tool in `.deps/cuda-oxide`. The Qwen3.8 launcher finds
this repository-local tool automatically:

```sh
scripts/run-eider-qwen38.sh --cuda-oxide --offline
```

You can also enable the Eider feature directly:

```sh
CARGO_OXIDE="$PWD/.deps/cuda-oxide/bin/cargo-oxide" \
  cargo build --release -p eider-api --features cuda-oxide
```

If `cargo-oxide` is not on `PATH`, set its full path:

```sh
CARGO_OXIDE=/path/to/cargo-oxide \
  cargo build --release -p eider-api --features cuda-oxide
```

The build fails if cuda-oxide, the pinned nightly release, `ptxas`, or an
`sm_121a` target is not available. It does not use the native CUDA kernel as a
silent fallback.

## Test

Run the focused W4A16 benchmark. It validates the result before it records
timing data.

```sh
CARGO_OXIDE=/path/to/cargo-oxide \
  cargo bench -p eider-cuda --features cuda-oxide \
  --bench sm121_w4a16_routed_gate_up
```

Run the chunked GDN prefill benchmark when you change the dense Qwen linear
attention path:

```sh
CARGO_OXIDE=/path/to/cargo-oxide \
  cargo bench -p eider-cuda --features cuda-oxide \
  --bench qwen36_chunked_gdn
```

Run the Flash Next MoE benchmark after a routed-kernel change:

```sh
CARGO_OXIDE=/path/to/cargo-oxide \
  cargo bench -p eider-cuda --features cuda-oxide \
  --bench qwen38_oxide_moe
```

The benchmark checks grouped W4A4 output against the W4A16 reference before
it records timing data.

The CUBIN loads once into the CUDA primary context. Eider launches its kernels
on the stream that the caller supplies.

Run the production server with Qwen3.8 27B and two DFlash2 drafts:

```sh
target/release/eider-serve qwen3.8-27b --offline --speculative-drafts 2
```
