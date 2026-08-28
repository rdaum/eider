# eider-inference

`eider-inference` loads supported checkpoints and executes their text models.
It owns model-family code, model-specific scheduling state, prefix restoration,
and the deployment selection path.

## Responsibilities

The crate provides these capabilities:

- checkpoint architecture detection and model loading
- model-family execution for Qwen, Step, Laguna, Gemma, Muse Glimmer,
  Nemotron, Ling, BitNet, Bonsai, and DeepSeek
- shared execution support for scheduling, prefix caches, and expert residency
- model-specific prefill and decode state
- focused runtime microbenchmarks.

`with_loaded_engine` loads one concrete model service and lends it to the API
as `dyn EngineService`. Model construction and layer execution remain
statically typed within this crate.

## Deployment

`InferenceEngineConfig` contains checkpoint paths, storage choices, scheduler
limits, and prefix-cache settings. A catalogue deployment resolves an immutable
checkpoint revision before this crate loads it.

Derived artifacts belong in the Eider cache, not in a Hugging Face snapshot.
Use `--model-dir` only for local checkpoint development.

## Development

Run focused checks from the workspace root:

```sh
cargo test -p eider-inference --lib
cargo clippy -p eider-inference --lib -- -D warnings
cargo doc -p eider-inference --lib --no-deps
```

Use a focused benchmark before you claim a performance result:

```sh
cargo bench -p eider-inference --bench qwen36_prefill
cargo bench -p eider-inference --bench qwen38_flash_next_prefill
```

Avoid full-model runs unless a focused check cannot answer the question. A
full model competes with the host for the same 128 GB unified-memory pool.

## Boundaries

Keep checkpoint records in `eider-format`, CUDA resources and kernels in
`eider-cuda`, and model-independent request policy in `eider-runtime`.
Do not add compatibility services or duplicate serving APIs for model code.
