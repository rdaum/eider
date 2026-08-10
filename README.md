# Eider

> ... the eider duck: a small northern creature with an unreasonable amount of
> insulation (ahem)

Eider is a native inference server for the NVIDIA DGX Spark. It runs NVFP4 and
mixed-precision models directly on the GB10's `sm_121` GPU and serves them over
the OpenAI Responses and Chat Completions APIs.

The runtime is Rust and CUDA. It is not built on PyTorch, llama.cpp, vLLM, or
another tensor library. The point is to make one Spark run useful models well,
while keeping the machine and its odd little Blackwell GPU understandable.

## Quick start

You need a DGX Spark, stable Rust, and CUDA 13.x with `nvcc` and cuBLASLt.

```sh
scripts/run-eider
```

That builds the release server, downloads the pinned Gemma 4 NVFP4 checkpoint
on first use, and listens on `127.0.0.1:8080`.

```sh
curl -fsS http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"eider-gemma4-26b","input":"What is 2+2?","max_output_tokens":64}'
```

Choose another model with `EIDER_MODEL`:

```sh
EIDER_MODEL=muse-glimmer-30b-nvfp4 scripts/run-eider
```

The repository also contains Pi launchers. With a matching server running:

```sh
scripts/run-pi-eider-muse.sh
```

Set `EIDER_API_KEY` to require bearer authentication. Use `--offline` after the
checkpoint and any derived artifacts have been cached.

## Models

Catalogue IDs are used to start the server. API clients use the served model
name in the second column.

| Catalogue ID | API model | Model path |
| --- | --- | --- |
| [`bitnet-b1.58-2b-4t`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) | `eider-bitnet-b1.58-2b` | Native BitNet b1.58 weights |
| [`muse-glimmer-30b-nvfp4`](https://huggingface.co/Inferact/Muse-Glimmer-30B-NVFP4-W4A4) | `eider-muse-glimmer-30b` | Dense W4A4 target, official DFlash drafter, ATEM tools, compact FP4 KV |
| [`qwen3.6-35b-a3b`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) | `eider-qwen3.6` | 35B-A3B MoE, compact FP4 KV |
| [`agents-a1`](https://internscience.github.io/Agents-A1/) | `eider-agents-a1` | Qwen3.5-MoE agentic fine-tune, 262K context |
| [`step-3.7-flash`](https://huggingface.co/stepfun-ai/Step-3.7-Flash-NVFP4) | `eider-step3.7` | 198B MoE with disk-backed expert paging |
| [`laguna-s-2.1`](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4) | `eider-laguna-s-2.1` | 256-expert MoE, compact FP4 KV |
| [`gemma-4-26b-a4b-nvfp4`](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4) | `eider-gemma4-26b` | Native NVIDIA NVFP4 checkpoint |
| [`gemma-4-26b-a4b-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) | `eider-gemma4-26b` | Upstream BF16 checkpoint served by the same runtime |
| [`nemotron-3-puzzle-75b-a9b`](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4) | `eider-nemotron3-puzzle` | Mamba-2, latent MoE, and attention hybrid |
| [`nemotron-3-super-120b-a12b`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) | `eider-nemotron3-super` | 120B-A12B Nemotron hybrid |

[DeepSeek V4 Flash](https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4)
uses a separate local preparation path because the complete source checkpoint
is larger than the Spark's unified memory. See
[the expert-storage notes](docs/deepseek4-experts.md).

### Muse Glimmer and DFlash

Muse Glimmer is the first Eider model to use speculative decoding. Eider pairs
the NVFP4 target with
[Meta's official DFlash GGUF](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF),
imports the drafter into resident NVFP4, and prefix-caches both models' sequence
state. The catalogue entry fetches both pinned checkpoints.

```sh
EIDER_MODEL=muse-glimmer-30b-nvfp4 scripts/run-eider
scripts/run-pi-eider-muse.sh
```

For an already-downloaded development checkpoint, `scripts/run-eider-muse.sh`
uses `MUSE_GLIMMER_MODEL` and `MUSE_GLIMMER_DFLASH` as path overrides.

DFlash throughput varies with acceptance; current end-to-end results range from
15.8 tokens/sec in a long Pi session to 28.5 tokens/sec on a short prompt. The
server exposes acceptance and cycle telemetry when the details matter.

## Performance

These are representative single-session API results on one GB10, not peak
kernel numbers. Context length, sampling, expert residency, and speculative
acceptance all matter.

| Model | Eider decode tok/s | vLLM decode tok/s | Configuration |
| --- | ---: | ---: | --- |
| Qwen3.6-35B-A3B | 72.7 | 77.0 | NVFP4 weights; compact FP4 KV in Eider, BF16 KV in vLLM |
| Agents-A1 | 63.6 | 37.2 | Eider FP8-converted attention and LM head |
| Gemma 4 26B-A4B | 30.1 | 29.6 | Same ModelOpt NVFP4 weights |
| Muse Glimmer 30B | 15.8–28.5 | — | DFlash; long Pi session to short prompt |
| Step-3.7-Flash | 20.4 | — | 240 of 288 routed experts resident per layer |
| Laguna-S-2.1 | 16.2 | — | Resident NVFP4 experts; compact FP4 KV |

## Why Eider exists

The Spark is not a datacentre GPU, despite the marketing. Its SM12x Blackwell
is not the B200 programming model, and choices made for large server GPUs do
not always fit a single 128 GB unified-memory machine.

I wanted to understand how an inference runtime actually works, got annoyed by
the weight of vLLM, and got frustrated by the state of NVFP4 in llama.cpp. A
small native runtime for this specific machine seemed both more interesting
and, increasingly, more useful.

Eider treats the Spark's 128 GB as one budget shared by weights, KV state,
workspaces, the host, and everything else. It allocates sequence state on
demand instead of reserving a giant KV pool at startup. Attention models use a
compact FP4 KV cache; repeated prompts use a shared radix cache; large MoEs can
page experts from disk. Model-specific state stays model-specific: Nemotron's
Mamba state is not pretending to be KV, and Muse's target and drafter advance
and roll back together.

## Serving

List the pinned catalogue without downloading a model:

```sh
cargo run --release -p eider-api --bin eider -- model list
```

Start a catalogue model directly:

```sh
eider-serve qwen3.6-35b-a3b
eider-serve muse-glimmer-30b-nvfp4
eider-serve step-3.7-flash --offline
```

The first online start resolves an immutable Hugging Face revision and prepares
any derived weights below `$XDG_CACHE_HOME/eider/models/`. `--model-dir` is for
local development checkpoints, not catalogue deployments.

Both API adapters reach the same scheduler and model runtime:

- `POST /v1/responses`
- `POST /v1/chat/completions`
- `GET /healthz`
- `GET /metrics`

The server supports streaming, tool-call history, sampling controls,
cancellation, concurrent requests, and prompt-prefix caching. Run
`eider-serve --help` for memory, scheduler, context, and model-specific knobs.

### Pi

The launchers use the repository's `pi/agent/models.json` without changing the
user's global Pi configuration:

```sh
scripts/run-pi-eider-qwen.sh
scripts/run-pi-eider-agents-a1.sh
scripts/run-pi-eider-stepfun.sh
scripts/run-pi-eider-laguna.sh
scripts/run-pi-eider-muse.sh
scripts/run-pi-eider-gemma4.sh
scripts/run-pi-eider-nemotron3-super.sh
scripts/run-pi-eider-deepseek4.sh
```

They use the Responses API by default. Set `PI_EIDER_PROVIDER=eider-chat` to
exercise Chat Completions instead.

### Codex

```toml
model = "eider-qwen3.6"
model_provider = "eider"

[model_providers.eider]
name = "Eider"
base_url = "http://127.0.0.1:8080/v1"
env_key = "EIDER_API_KEY"
wire_api = "responses"
```

## Development

The workspace has three main crates:

- `nvfp4` owns CUDA, cuBLASLt, device storage, checkpoint formats, and focused
  GPU benchmarks.
- `infer` owns model execution, sequence state, sampling, scheduling, prefix
  caching, and model-runtime benchmarks.
- `eider-api` owns catalogue deployment, the inference actor, HTTP/SSE, and
  telemetry.

Build and test with:

```sh
cargo build --workspace
cargo build --release -p eider-api --bin eider-serve
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

SM121 CUTLASS support lives in the ignored `.deps/` directory:

```sh
scripts/setup-cutlass-sm12x.sh
source .deps/cutlass-sm12x.env
scripts/probe-cutlass-sm12x.sh
```

Focused benchmarks use
[`micromeasure`](https://github.com/rdaum/micromeasure) and check correctness
before timing. The common entry points are:

```sh
cargo bench -p nvfp4 --bench qwen36_routed_moe_decode
cargo bench -p infer --bench qwen36_prefill
cargo bench -p infer --bench step37_prefill
cargo bench -p infer --bench laguna_prefill
```

An isolated kernel win is not enough; changes are kept only when the full model
path also improves.

## Troubleshooting

If CUDA cannot target `sm_121`, make sure `nvcc`, the CUDA headers, and
cuBLASLt all come from the same CUDA 13.x installation, then run
`scripts/probe-cutlass-sm12x.sh`.

If an offline catalogue start fails, fetch it once online or run:

```sh
cargo run --release -p eider-api --bin eider -- model fetch MODEL_ID
```

If CUDA allocation fails or the process is OOM-killed, stop other GPU and
memory-heavy processes first. GPU allocations share the Spark's 128 GB with
the host. Smaller prefill capacity, fewer concurrent sequences, or a lower
Step expert capacity can reduce the live footprint.

## Authorship

The opening shape of this project was substantially written by hand because
the point was to understand the hardware rather than treat it as a black box.
As the work moved into FFI boilerplate, format conversions, CUDA wrappers, and
benchmark harnesses, AI assistance became a larger part of the implementation,
and the performance tuning is heavily agent-driven.

That boundary is visible rather than hidden. Not every detail of every kernel
is something I can yet explain from scratch, and turning those pieces back into
things I can explain and trust is part of the work.

## Further reading

- [Model deployment](docs/model-deployment.md)
- [Qwen3.6 batching and scheduling](docs/qwen36-batch-decode-plan.md)
- [Step-3.7 expert paging](docs/step37-paging.md)
- [DeepSeek V4 expert storage](docs/deepseek4-experts.md)
- [SM12x CUTLASS and NVFP4](docs/cutlass-sm12x-nvfp4.md)
