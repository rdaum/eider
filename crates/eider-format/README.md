# eider-format

`eider-format` reads Eider checkpoint formats and derived model artifacts. It
only owns host-side file representations.

## Responsibilities

The crate provides readers and records for:

- safetensors indexes and tensor shards
- ModelOpt NVFP4 and block-FP8 tensors
- GGUF metadata and quantized payloads
- Eider NVFP4 artifact directories.

The crate does not allocate device memory, create CUDA streams, select a
kernel, or manage request state.

## Main types

`ModelOptCheckpoint` locates tensors in a ModelOpt checkpoint. The
`ModelOptNvfp4Linear`, `ModelOptFp8Linear`, and
`ModelOptBlockScaledFp8Linear` records preserve the checkpoint layout.
`Nvfp4Artifact` identifies Eider-generated files below the artifact cache.

Callers must convert a format record at an explicit preparation boundary. Do
not reinterpret ModelOpt scales as cuBLASLt or native-MMA scales.

## Development

Run the focused checks from the workspace root:

```sh
cargo test -p eider-format --lib
cargo clippy -p eider-format --lib -- -D warnings
cargo doc -p eider-format --lib --no-deps
```

Keep parsing errors precise. A missing tensor, an invalid shape, and an
unsupported format must remain distinguishable to the caller.
