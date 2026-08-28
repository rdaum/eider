# eider-runtime

`eider-runtime` contains model-independent state and policy for Eider serving.
It has no CUDA dependency and does not load a checkpoint.

## Responsibilities

The crate defines these shared parts of a request:

- `ChatRequest`, generation settings, and token usage
- chat-template rendering and structured output parsing
- stop sequences and tool grammars
- prefix-cache identities and cache policy
- sampling policy
- the `EngineService` contract used by the API actor.

## Engine contract

`EngineService` is the boundary between a serving actor and a concrete model
service. It accepts rendered requests, advances one scheduler tick, reports
lifecycle events, supports cancellation, and releases resources at shutdown.

The contract uses `EngineRequestId`, `EngineTick`, and `EngineError`. It does
not expose model-specific request IDs, scheduler records, CUDA types, or
checkpoint errors.

Calls occur once per actor tick. Model layer execution remains statically
typed inside `eider-inference`.

## Development

Run the focused checks from the workspace root:

```sh
cargo test -p eider-runtime --lib
cargo clippy -p eider-runtime --lib -- -D warnings
cargo doc -p eider-runtime --lib --no-deps
```

Keep this crate independent of CUDA and model families. Add a type here only
when more than one engine needs the same stable policy or contract.
