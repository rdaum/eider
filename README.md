# Eider

> ... the eider duck: a small northern creature with an unreasonable amount of
> insulation (ahem)

Eider is an inference and serving runtime for NVIDIA DGX Spark (GB10,
Grace Blackwell) and variants thereof.

It is built from scratch (in Rust and CUDA) specifically to take
advantage of the NVFP4 capabilities of the SM121 GPU.

It is not built on top of any existing tensor or inference library, and does
not use llama.cpp or vLLM.

It is capable of running Qwen3.6, the Qwen3.5-MoE fine-tune Agents-A1,
StepFun's Step-3.7-Flash, Poolside's Laguna-S-2.1, Gemma 4 26B-A4B, and
NVIDIA's Nemotron 3 Puzzle hybrid model, DeepSeek V4 Flash, and Meta's
Muse Glimmer 30B text model.

It includes an OpenAI-compatible Responses and Chat Completions server
with continuous multi-session scheduling and a compact FP4 KV cache
for the Qwen, Step, Laguna, and Gemma attention paths. Nemotron combines
backbone attention with Mamba recurrent state.

This started as a personal research project and is crawling towards more of a
production engine -- most parts of the kernel layer are agent-written; see the
authorship note below. It also contains a self-contained Rust crate with NVFP4 /
GB10 specific kernels that others may potentially reuse in their projects.

## Quick start

Eider targets the NVIDIA DGX Spark and its GB10 (`sm_121`) GPU. Building it
requires a current stable Rust toolchain and CUDA 13.x, including `nvcc` and
cuBLASLt. Checkpoint and derived-artifact storage varies substantially by
model; Step-3.7 alone needs roughly 110 GiB for its prepared expert records in
addition to the Hugging Face snapshot.

The convenience launcher builds the release server, downloads the pinned
NVIDIA Gemma 4 NVFP4 checkpoint on first use, and listens on
`127.0.0.1:8080`:

```sh
scripts/run-eider
```

In another shell, send a request using the served model name:

```sh
curl -fsS http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"eider-gemma4-26b","input":"What is 2+2?","max_output_tokens":64}'
```

The same runtime is also available through Chat Completions:

```sh
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"eider-gemma4-26b","messages":[{"role":"user","content":"What is 2+2?"}],"max_completion_tokens":64}'
```

Select another catalogue model with `EIDER_MODEL`, or list the catalogue
without downloading anything:

```sh
EIDER_MODEL=agents-a1 scripts/run-eider
cargo run --release -p eider-api --bin eider -- model list
```

If `EIDER_API_KEY` is set for the server, clients must also send it as a
Bearer token.

## Why... ?

The Spark is not a datacentre GPU (despite the marketing.) The SM12x Blackwell
in it is not the same as the heavy duty "grown up" version. So I have a personal
suspicion that it will likely be better served in the medium term by a bespoke
runtime than by work being done on the vLLM mainline.

But mainly, I wanted to learn more about how an inference runtime is structured, and
I wanted to understand more about the Spark's architecture, and I got annoyed by
how heavy weight vLLM is, and I got frustrated by the state (or lack of it) of
NVFP4 in `llama.cpp`.

## Performance

Representative single-session results on one GB10 are below. Decode rates are
through the OpenAI-compatible API; a dash means that a comparable vLLM
throughput run has not been completed.

| Model | Eider decode tok/s | vLLM decode tok/s | Configuration |
| --- | ---: | ---: | --- |
| [Qwen3.6-35B-A3B](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) | 72.7 | 77.0 | NVFP4 weights; compact FP4 KV in Eider, BF16 KV in vLLM |
| [Agents-A1](https://internscience.github.io/Agents-A1/) | 63.6 | 37.2 | Eider FP8-converted attention and LM head; checkpoint-native vLLM |
| [Step-3.7-Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash-NVFP4) | 20.4 | — | 240 of 288 routed experts resident per layer |
| [Laguna-S-2.1](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4) | 16.2 | — | Resident NVFP4 experts; compact FP4 KV cache |
| [Gemma 4 26B-A4B](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4) | 30.1 | 29.6 | Same ModelOpt NVFP4 weights; compact FP4 KV in Eider, FP8 E4M3 KV in vLLM |
| [Muse Glimmer 30B](https://huggingface.co/Inferact/Muse-Glimmer-30B-NVFP4-W4A4) | 5.5 | — | ModelOpt NVFP4 weights through the initial W4A16 decode path; compact FP4 KV |
| [Nemotron Labs 3 Puzzle 75B-A9B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4) | — | — | Throughput comparison pending |

Gemma prefills a fresh roughly 2,700-token Pi/API prompt at about 6,740 prompt
tokens/sec, compared with about 7,060 in vLLM. Prefix reuse brought a typical
follow-up to 235 ms TTFT. An Agents-A1 Pi session sustained 58-60 decode
tokens/sec through 4,200-token turns and 44.5 at 17,748 tokens.
Laguna prefills a fresh roughly 3,300-token API prompt at about 135 prompt
tokens/sec. Muse Glimmer's initial sequential path measured 5.9 prompt
tokens/sec and 5.5 decode tokens/sec on a 50-token API prompt; batched prefill
and a tensor-core decode path remain future optimization work.

Step-3.7 is a 198B checkpoint served with disk-backed expert paging. Converting
its remaining BF16 weights to NVFP4 reduces resident device weights from 95.5
to 87.3 GiB. Puzzle occupies 41.9 GiB of Eider device weights.

The largest differences from vLLM are operational: Eider starts substantially
faster and has a smaller idle footprint because sequence state is allocated on
demand rather than as a large up-front KV pool. It also supports a compact
NVFP4 KV cache directly on SM121. The local vLLM Puzzle control used FP8 E4M3
KV rather than NVFP4.

Both advantages leave more of the Spark's 128 GB unified memory available for
longer contexts and concurrent requests. In particular, the smaller NVFP4 KV
cache leaves more room for model weights and could raise the practical
model-size ceiling relative to a runtime using BF16 KV. Finding the largest
useful checkpoint that fits on one Spark is a planned next test.

## Supported models

Catalogue IDs select pinned checkpoints at startup; API requests use the
corresponding served model name.

| Catalogue ID | API model | Notes |
| --- | --- | --- |
| [`qwen3.6-35b-a3b`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) | `eider-qwen3.6` | 35B-A3B MoE; compact FP4 KV cache |
| [`agents-a1`](https://internscience.github.io/Agents-A1/) | `eider-agents-a1` | Qwen3.5-MoE agentic fine-tune; 262K-token limit |
| [`step-3.7-flash`](https://huggingface.co/stepfun-ai/Step-3.7-Flash-NVFP4) | `eider-step3.7` | 198B MoE with disk-backed expert paging |
| [`laguna-s-2.1`](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4) | `eider-laguna-s-2.1` | 256-expert MoE; compact FP4 KV cache |
| [`muse-glimmer-30b-nvfp4`](https://huggingface.co/Inferact/Muse-Glimmer-30B-NVFP4-W4A4) | `eider-muse-glimmer-30b` | Dense agentic text model; ATEM tools and compact FP4 KV cache |
| [`gemma-4-26b-a4b-nvfp4`](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4) | `eider-gemma4-26b` | Native NVIDIA NVFP4 checkpoint |
| [`gemma-4-26b-a4b-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) | `eider-gemma4-26b` | Upstream BF16 source served by the same text runtime |
| [`nemotron-3-puzzle-75b-a9b`](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4) | `eider-nemotron3-puzzle` | Mamba-2, latent-MoE, and attention hybrid |
| [`nemotron-3-super-120b-a12b`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) | `eider-nemotron3-super` | 120B-A12B Nemotron hybrid |

Supported catalogue checkpoints are fetched from Hugging Face into its local
snapshot cache; the repository does not retain model weights under `models/`.
Use `--model-dir PATH` only for local conversion or development checkpoints.

[DeepSeek V4 Flash](https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4)
uses a separate prepared-local workflow because its complete source checkpoint
is larger than the Spark's unified memory. Its launcher serves
`eider-deepseek-v4` from a thin checkpoint and bounded, disk-backed exact
NVFP4 expert slots.

Agents-A1 uses the same Qwen3.5-MoE runtime as Qwen3.6. Its checkpoint has BF16
attention projections and a BF16 LM head alongside its NVFP4 experts; Eider can
retain those types or convert the BF16 weights to FP8 or NVFP4. The checkpoint
also contains a vision tower, but Eider currently serves its text path only.

[Gemma 4 26B-A4B](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4)
uses native NVFP4 experts alongside BF16 attention, routing, and dense MLP
weights. Eider serves its heterogeneous local/global attention with a compact
FP4 KV cache and the checkpoint's thinking and tool-call protocol.

[NVIDIA Nemotron Labs 3 Puzzle 75B-A9B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4)
uses an 88-layer Mamba-2, sparse latent-MoE, and grouped-query attention
backbone with layer-specific expert widths and top-k routing. Eider loads those
checkpoint settings directly and retains 41.9 GiB including its MTP block.

## Running and deployment

The convenience launcher builds and runs the release server through Cargo. It
defaults to the NVIDIA Gemma 4 NVFP4 catalogue entry and forwards all server
arguments:

```sh
scripts/run-eider
scripts/run-eider --offline
EIDER_MODEL=gemma-4-26b-a4b-it scripts/run-eider
```

To select any supported model directly, pass its stable catalogue ID to
`eider-serve`:

```sh
eider-serve qwen3.6-35b-a3b
eider-serve agents-a1
eider-serve step-3.7-flash
eider-serve laguna-s-2.1
eider-serve gemma-4-26b-a4b-nvfp4
eider-serve gemma-4-26b-a4b-it
eider-serve nemotron-3-puzzle-75b-a9b
eider-serve nemotron-3-super-120b-a12b
```

List the current catalogue without downloading anything:

```sh
cargo run --release -p eider-api --bin eider -- model list
```

### Checkpoints and prepared artifacts

The first catalogue start resolves the pinned Hugging Face revision, downloads
the required checkpoint files, prepares any Eider artifacts, and then serves.
Later starts reuse both caches; pass `--offline` to require a complete local
snapshot. Use `--model-dir PATH` only for local conversion or development
checkpoints. The NVIDIA NVFP4 and upstream BF16 Gemma 4 entries both serve the
Gemma text path; multimodal inputs are not yet exposed through Eider's API.

Catalogue deployments keep Hugging Face snapshots immutable. The first Qwen3.6
or Agents-A1 start builds its SM12x down-weight cache below
`$XDG_CACHE_HOME/eider/models/`; Laguna's native down weights and Step-3.7
expert records are stored there too.
This is a one-time, down-only repack of roughly 5 GiB for the 35B-A3B
checkpoint. Mixed-precision checkpoints build it only for layers whose down
weights are NVFP4. Cache files are written atomically and incomplete layers are
resumed on the next startup; later runs validate and reuse the completed cache
automatically.

DeepSeek V4 requires a bounded streaming conversion before serving because its
complete source checkpoint is larger than the Spark's unified memory. See
[DeepSeek V4 expert storage](docs/deepseek4-experts.md) for its paged NVFP4
layout, memory budget, and preparation commands.

### Model-specific controls

The Nemotron launcher stores dense weights in NVFP4 and uses an FP32
backbone-attention cache by default. Pass `--nemotron-kv-cache nvfp4` when
long-context headroom matters more than serving speed. It also uses the shared
prompt-prefix cache by default; pass `--prefix-cache-gib 0` to disable it.

Step-3.7 prepares or validates the disk-backed expert cache before starting the
server and defaults to 240 resident experts per routed layer; use
`--step-expert-capacity` to change that tradeoff. Its BF16 attention, dense
MLP, shared-expert, and LM-head weights default to NVFP4; select their storage
with the corresponding `--step-bf16-*` flags. Agents-A1 accepts the checkpoint's
full 262,144-token context; use `--max-context-tokens` to override it. Its BF16
attention projections and LM head default to NVFP4; use
`--qwen-bf16-attention` and `--qwen-bf16-lm-head` to select `bf16`, `fp8`, or
`nvfp4`.

DeepSeek V4 defaults to eight resident exact-NVFP4 expert slots per layer and
a 32,768-token context. Use `--deepseek-expert-capacity` and
`--max-context-tokens` to change those limits.

### API and agent clients

The server listens on `127.0.0.1:8080` by default and reads an optional bearer
token from `EIDER_API_KEY`. Both `/v1/responses` and `/v1/chat/completions`
support streaming, sampling controls, tool definitions and tool-call history;
they share the same scheduler and generation path. Use
`EIDER_MODEL` to change the `scripts/run-eider` model, or pass `--listen`,
`--served-model-name`, and `--api-key-env` to the server.
The server exposes Prometheus text at `/metrics` and health at `/healthz`; set
`EIDER_DOGSTATSD_ENDPOINT` (with optional `EIDER_DOGSTATSD_INTERVAL_SECS`) to
additionally push metrics over UDP. The `eider-serve` binary also takes
`--decode-capacity`, `--prefill-sequence-capacity`, `--prefill-token-capacity`,
`--max-active-sequences`, and `--max-context-tokens` flags that map directly to
the scheduler admission limits. All supported serving paths retain up to 2
GiB of device-resident prompt checkpoints by default; pass
`--prefix-cache-gib 0` to disable it or another whole-GiB value to change the
budget. Responses report reused input tokens as
`usage.input_tokens_details.cached_tokens`, and the checkpoint cache exports
hit, miss, eviction, retained-byte, and copy-latency metrics alongside the other
scheduler telemetry.

The Agents-A1 Pi entry advertises a 131,072-token working window so compaction
starts well before the checkpoint's hard limit.

Run Pi against the matching server with:

```sh
scripts/run-pi-eider-qwen.sh
scripts/run-pi-eider-agents-a1.sh
scripts/run-pi-eider-stepfun.sh
scripts/run-pi-eider-laguna.sh
scripts/run-pi-eider-gemma4.sh
scripts/run-pi-eider-nemotron3-super.sh
scripts/run-pi-eider-deepseek4.sh
```

Arguments are forwarded to `pi`, so a non-interactive smoke request looks like:

```sh
scripts/run-pi-eider-stepfun.sh --print "Reply with one short greeting."
```

The Pi launcher uses `pi/agent/models.json` through `PI_CODING_AGENT_DIR` and
does not modify the user's global Pi configuration. It uses Responses by
default; select the side-by-side Chat Completions provider with:

```sh
PI_EIDER_PROVIDER=eider-chat scripts/run-pi-eider-gemma4.sh
```

Point Codex at it with a custom provider in `~/.codex/config.toml`:

```toml
model = "eider-qwen3.6"
model_provider = "eider"

[model_providers.eider]
name = "Eider"
base_url = "http://127.0.0.1:8080/v1"
env_key = "EIDER_API_KEY"
wire_api = "responses"
```

The adapter accepts Codex message and function-call history, renders function
tools through the checkpoint chat template, streams Responses lifecycle and
function-argument events, and cancels scheduler work when a client disconnects.
Run the full local Codex integration test explicitly with:

```sh
QWEN36_MODEL=models/qwen3.6-35b-a3-nvfp4 \
cargo test --release -p eider-api --test codex -- --ignored --nocapture
```

## Development

The workspace has three crates:

- `crates/nvfp4` owns device buffers, ModelOpt NVFP4/FP8 loading, cuBLASLt
  plans, CUDA FFI, custom kernels, and low-level micromeasures. Its source is
  grouped into `cublaslt/`, `kernels/`, and `diagnostics/` by topic.
- `crates/infer` owns model loading, model execution, KV-cache state,
  request-scoped sampling and generation, model runtimes, prefill/decode
  orchestration, CLI binaries, and runtime benchmarks. Its reusable execution
  state lives under `runtime/`.
- `crates/eider-api` owns the inference actor, catalogue deployment, and
  OpenAI-compatible HTTP/SSE adapter used by agent clients. CUDA state remains
  on the actor's OS thread while async handlers submit, stream, and cancel
  requests over bounded channels. It also exposes Prometheus and optional
  DogStatsD metrics.

CUDA kernels live in `crates/nvfp4/native/` and are linked into the Rust crate
by its build script. Normal Qwen3.6 serving uses CUTLASS W4A4 routed gate/up,
SM12x routed down, and cuBLASLt.

### Build and test

Requirements are Rust, CUDA 13.x, cuBLASLt, and an `nvcc` capable of compiling
`sm_121` code. Build and test with:

```sh
cargo build -p infer
cargo build --release -p infer
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

### Smoke tests and throughput

Qwen3.6 smoke test:

```sh
cargo run --release -p infer --bin qwen36-generate -- \
    models/qwen3.6-35b-a3-nvfp4 "What is 2+2?" 30
```

Nemotron Labs 3 Puzzle smoke test:

```sh
cargo run --release -p infer --bin nemotron3-generate -- \
    models/nemotron-labs-3-puzzle-75b-a9b-nvfp4 "What is 2+2?" 30 nvfp4 nvfp4
```

The generator applies the checkpoint's text chat prefix and reads its sampling
defaults from `generation_config.json` (`temperature`, `top_k`, `top_p`, and
EOS token IDs). Positional overrides follow the token count in this order:
temperature, top-k, top-p, seed, presence penalty, and frequency penalty. Pass
`0` as the temperature to use the faster deterministic GPU top-1 path:

```sh
cargo run --release -p infer --bin qwen36-generate -- \
    models/qwen3.6-35b-a3-nvfp4 "What is 2+2?" 30 0
```

Throughput benchmark:

```sh
cargo run --release -p infer --bin qwen-bench -- \
    --model models/qwen3.6-35b-a3-nvfp4 \
    --prompt "What is the meaning of life?" \
    --decode-tokens 512 \
    --warmup-repeats 1 \
    --repeats 3 \
    --temperature 0
```

On 2026-07-14 at revision `1f445ce`, that run used a seven-token prompt and
reported a median 6,602.8 ms for 512 decode tokens, or 77.54 tokens/sec.

Use `--profile-decode` for stage timings. It synchronizes between stage groups,
so its throughput is diagnostic and is not directly comparable with the normal
CUDA-graph benchmark.

For decode correctness isolation, set `EIDER_DISABLE_DECODE_GRAPHS=1` to run
the same model path without segmented CUDA graph replay. This is a diagnostic
comparison, not the fast default.

### Runtime shape

The steady-state Qwen3.6 path is a device-resident decode loop. Rust owns the
layer orchestration and state transitions; CUDA, cuBLASLt, and SM12x
implement the measured kernels underneath it.

```mermaid
flowchart TD
    A[Prompt chunks] --> B[Ragged batched prefill]
    B --> C[Per-sequence NVFP4 KV and GDN state]
    C --> D[Select changing decode batch]
    D --> E[Embedding lookup]
    E --> F{Layer attention type}
    F -->|Linear: captured through layer tail| G[QKV/Z + GDN update + gated norm + output projection]
    F -->|Full: captured pre-attention| H[Q/K/V + Q/K norm + iMROPE]
    H -->|Eager for active rows| I[NVFP4 KV append + cached attention]
    I -->|Captured post-attention through layer tail| J[Sigmoid gate + output projection]
    G --> K[Attention residual + post-attention RMSNorm]
    J --> K
    K -->|Main stream| L[BF16 router + top-k]
    K -->|Shared stream| M[NVFP4 shared expert + BF16 gate]
    L --> N[Indexed CUTLASS W4A4 routed gate/up]
    N --> O[SiLU + indexed SM12x routed down]
    O --> P[Fused routed/shared combine + residual]
    M --> P
    P --> Q{Last layer?}
    Q -->|No| F
    Q -->|Yes| R[Final RMSNorm + LM-head logits]
    R --> S{Sampling policy}
    S -->|Greedy| T[GPU top-1]
    S -->|Bounded top-k/top-p| U[GPU penalties + hierarchical sampling]
    S -->|Unbounded/large top-k| V[Logit readback + CPU sampling]
    T --> W[Selected tokens]
    U --> W
    V --> W
    W --> D

    subgraph Host[Host orchestration]
        A
        D
        V
        W
    end
    subgraph Device[Device-resident work]
        B
        C
        E
        F
        G
        H
        I
        J
        K
        L
        M
        N
        O
        P
        Q
        R
        S
        T
        U
    end
```

Sequence-owned KV and recurrent states remain device-resident as requests move
between decode batches. Linear-attention layers replay one captured layer
graph. Full-attention layers replay captured pre- and post-attention graphs
around the active-row KV-cache operation. The shared expert overlaps the routed
experts on a second CUDA stream. `EIDER_DISABLE_DECODE_GRAPHS=1` retains the
same execution path but submits its layer work eagerly.

### CUTLASS setup

CUTLASS is needed for the normal Qwen3.6 routed gate/up path, the dense Qwen
GEMV backend, and CUTLASS-specific low-level tests and benchmarks.

The build looks for it under `.deps/cutlass` and uses
`.deps/cutlass-build-sm121` by default. Configure it with:

```sh
scripts/setup-cutlass-sm12x.sh
source .deps/cutlass-sm12x.env
```

The compile probe is available as:

```sh
scripts/probe-cutlass-sm12x.sh
```

### Microbenchmarks

Benchmarks use my [`micromeasure`](https://github.com/rdaum/micromeasure) crate.

Useful current targets include:

```sh
cargo bench -p nvfp4 --bench sm121_w4a16_routed_gate_up
cargo bench -p nvfp4 --bench sm121_w4a16_shared_expert
cargo bench -p nvfp4 --bench lm_head_top1
cargo bench -p nvfp4 --bench sm12x_indexed_gemv
cargo bench -p nvfp4 --bench nvfp4_kv_attention
cargo bench -p nvfp4 --bench fp8_routed_moe
cargo bench -p nvfp4 --bench fp8_linear
cargo bench -p nvfp4 --bench fp4_cublaslt
cargo bench -p nvfp4 --bench fp4_quantization
cargo bench -p nvfp4 --bench moe_topk
cargo bench -p nvfp4 --bench gated_delta_net
cargo bench -p nvfp4 --bench qwen36_routed_moe_decode
cargo bench -p infer --bench qwen36_routed_gate_up
cargo bench -p infer --bench qwen36_decode_batch
cargo bench -p infer --bench qwen36_prefill
cargo bench -p infer --bench step37_prefill
cargo bench -p infer --bench qwen36_cpu_shared_expert
cargo bench -p infer --bench sampling
```

Keep kernel claims tied to a shape-appropriate micromeasure and an end-to-end
decode run. An isolated faster kernel is not evidence of a faster model.

### Comparison helpers

For an already-running OpenAI-compatible vLLM server:

```sh
VLLM_URL=http://127.0.0.1:8000/v1/completions \
EIDER_MODEL_DIR=models/qwen3.6-35b-a3-nvfp4 \
scripts/compare-vllm.sh
```

`scripts/bench-eider-vllm-docker.sh` starts the configured vLLM Docker image
and runs the same comparison. The scripts accept environment overrides for the
model, prompt, token count, repeat count, and endpoint.

## Troubleshooting

### The CUDA build cannot target `sm_121`

Use CUDA 13.x and confirm that `nvcc`, the CUDA headers, and cuBLASLt come from
the same installation. `scripts/probe-cutlass-sm12x.sh` checks the CUTLASS
toolchain used by normal Qwen3.6 serving.

### A catalogue model will not start offline

An offline start requires the complete pinned Hugging Face snapshot and any
derived Eider artifacts. Start once without `--offline`, or fetch explicitly
with `cargo run --release -p eider-api --bin eider -- model fetch MODEL_ID`.
Snapshots use the Hugging Face cache; prepared weights use
`$XDG_CACHE_HOME/eider/models/` (or `~/.cache/eider/models/` when
`XDG_CACHE_HOME` is unset).

### CUDA allocation fails or the process is OOM-killed

GB10 device allocations consume the Spark's shared 128 GB memory. Stop other
GPU or memory-heavy processes before serving large checkpoints. Reducing
`--prefill-token-capacity`, concurrent sequences, or requested output length
lowers workspace or live sequence state. Reducing `--step-expert-capacity`
trades Step-3.7 throughput for model headroom. Nemotron can use
`--nemotron-kv-cache nvfp4` when KV capacity matters more than its faster
default cache.

### An API request is rejected

Send requests to `/v1/responses` or `/v1/chat/completions` using the API model
name from the supported models table, not the catalogue ID. When
`EIDER_API_KEY` is set on the server, include
`Authorization: Bearer $EIDER_API_KEY` in the request.

## A note about authorship

The opening shape of this project was substantially written by hand because
the point was to understand the hardware rather than treat it as a black box.
As the work moved into FFI boilerplate, format conversions, CUDA wrappers, and
benchmark harnesses, AI assistance became a larger part of the implementation,
and the performance tuning is heavily agent driven.

That boundary is deliberate and visible rather than hidden. Not every detail
of every kernel is something I can yet explain from scratch, and turning those
pieces back into things I can explain and trust better is part of the ongoing
work here.

## Further notes

- `docs/qwen36-batch-decode-plan.md` for the batch contract, correctness
  evidence, and remaining scheduler-admission measurements.
- `docs/cutlass-sm12x-nvfp4.md` for the SM12x/CUTLASS kernel investigation.
