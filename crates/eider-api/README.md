# eider-api

`eider-api` is the HTTP and server-control crate for Eider. It exposes
OpenAI-compatible Responses and Chat Completions endpoints for one loaded
Eider model.

## Responsibilities

The crate owns these concerns:

- catalogue resolution and immutable checkpoint deployment
- HTTP request parsing, streaming, and protocol errors
- the inference actor and request cancellation
- API telemetry and Prometheus output
- server startup and shutdown.

The crate does not load tensors, select CUDA kernels, or parse checkpoint
weights. It passes rendered `ChatRequest` values to the model-neutral
`EngineService` contract from `eider-runtime`.

## API surface

`InferenceActor` accepts requests through a bounded actor channel. Each actor
owns one loaded model service. `ApiConfig`, `serve`, and `serve_with_shutdown`
create the Axum server around that actor.

`InferenceActorConfig` owns API settings. Its `engine` field contains the
inference loading and execution configuration. Its `event_capacity` field
limits the event queue for each API request.

Both HTTP endpoints produce the same `ChatRequest` and actor events. Keep
tool calls, sampling, cancellation, token usage, and finish reasons aligned
between the two protocol modules.

## Run a server

Build the serving binary from the workspace root:

```sh
cargo build --release -p eider-api --bin eider-serve
target/release/eider-serve qwen3.8-27b --offline
```

Use a catalogue ID to select a pinned deployment. Use `/v1/models` to obtain
the served model name for requests.

## Development

Run the focused library checks from the workspace root:

```sh
cargo test -p eider-api --lib
cargo clippy -p eider-api --lib -- -D warnings
cargo doc -p eider-api --lib --no-deps
```

The actor contract has a fake-engine test. It covers actor lifecycle handling
without loading a model or allocating GPU memory.

## Boundaries

Keep model selection in `eider-inference`, request policy in `eider-runtime`,
and CUDA work in `eider-cuda`. Do not import model-specific state or CUDA
types into this crate.
