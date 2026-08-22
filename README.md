# Eider

Eider is a native inference server for the NVIDIA DGX Spark. It runs NVFP4 and
mixed-precision models directly on the GB10 `sm_121` GPU.

The runtime uses Rust, CUDA, and cuBLASLt. It does not depend on PyTorch,
llama.cpp, vLLM, or another tensor runtime.

Eider has three priorities:

- It starts models quickly without the large software stack that vLLM uses.
- It supports fast iteration on GB10-specific kernels, especially for NVFP4
  paths that llama.cpp does not support well.
- It provides a practical platform for experiments and for learning how
  inference systems work.

## Qwen3.8 quick start

Qwen3.8 27B is the primary dense model in Eider. The standard launcher uses
the [Inferact NVFP4 checkpoint](https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4)
and its
[`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
companion.

You need a DGX Spark, stable Rust, and CUDA 13.x. The CUDA installation must
include `nvcc` and cuBLASLt.

```sh
scripts/run-eider-qwen38.sh
```

The target checkpoint uses about 24.6 GiB of disk space. The first start also
resolves the DFlash2 checkpoint. Eider listens on `127.0.0.1:8080`.

In a second terminal, start Pi:

```sh
scripts/run-pi-eider-qwen38.sh
```

The launcher enables two DFlash2 drafts and the native 262,144-token context
window. It uses one active sequence to keep the full-attention cache within the
Spark memory budget.

### Current performance

These Qwen3.8 results come from a live Pi tool-use session on one DGX Spark.
The session used a release build on August 22, 2026.

| Measurement | Result | Workload |
| --- | ---: | --- |
| Cold prefill | 661.2 tokens/sec | 5,802 prompt tokens, no cached prefix |
| Cold time to first token | 8.81 sec | Same first turn |
| Cached time to first token | 2.35 sec | 6,623 prompt tokens, 5,760 cached tokens |
| Decode | 18.4–23.2 tokens/sec | First three completed Pi tool turns |
| Accepted drafts per cycle | 1.35–1.94 | Two DFlash2 drafts per cycle |

The three completed turns had 5.8K to 10.3K prompt tokens. The shared prefix
cache restored state from each previous turn.

A separate correctness-gated 4K benchmark measures 12.6 tokens/sec for the
target alone. DFlash2 reaches 25.2 effective tokens/sec on that synthetic
sequence.

These numbers are API results, not isolated kernel rates. Context length,
sampling, tool grammar, prefix reuse, and draft acceptance change the result.

### API request

Send a Responses API request:

```sh
curl -fsS http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"eider-qwen3.8","input":"What is 2+2?","max_output_tokens":64}'
```

Use `--offline` after Eider caches the checkpoint and its derived artifacts:

```sh
scripts/run-eider-qwen38.sh --offline
```

### Runtime details

The Qwen3.8 target has 64 dense layers. Its layer schedule repeats three
linear-attention layers and one full-attention layer.

The DFlash2 companion proposes tokens for target verification. Eider commits
only the proposal prefix that matches the target output.

The shared radix cache retains full-attention prompt pages. A separate cache
retains exact DFlash2 state for the same prompt prefix.

Set `--speculative-drafts 0` to measure target-only decoding:

```sh
scripts/run-eider-qwen38.sh --speculative-drafts 0
```

For local checkpoints, supply both paths:

```sh
eider-serve --model-dir ./models/qwen3.8-27b-nvfp4 \
  --dflash2-dir ./models/qwen3.8-27b-dflash2 \
  --speculative-drafts 2
```

## Supported models

Catalogue IDs start the server. API clients use the served model name in the
second column.

| Catalogue ID | API model | Runtime path |
| --- | --- | --- |
| [`qwen3.8-27b`](https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4) | `eider-qwen3.8` | Dense hybrid, ModelOpt NVFP4, DFlash2, compact FP4 KV, 262K model context |
| [`qwen3.6-35b-a3b`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) | `eider-qwen3.6` | 35B-A3B MoE and compact FP4 KV |
| [`agents-a1`](https://internscience.github.io/Agents-A1/) | `eider-agents-a1` | Qwen3.5-MoE agent model with 262K context |
| [`gemma-4-26b-a4b-nvfp4`](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4) | `eider-gemma4-26b` | Native NVIDIA NVFP4 checkpoint |
| [`gemma-4-26b-a4b-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) | `eider-gemma4-26b` | Upstream BF16 checkpoint on the same runtime |
| [`muse-glimmer-30b-nvfp4`](https://huggingface.co/Inferact/Muse-Glimmer-30B-NVFP4-W4A4) | `eider-muse-glimmer-30b` | Text path, DFlash, ATEM tools, and compact FP4 KV |
| [`step-3.7-flash`](https://huggingface.co/stepfun-ai/Step-3.7-Flash-NVFP4) | `eider-step3.7` | 198B MoE with disk-backed expert paging |
| [`laguna-s-2.1`](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4) | `eider-laguna-s-2.1` | 256-expert MoE and compact FP4 KV |
| [`nemotron-3-puzzle-75b-a9b`](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4) | `eider-nemotron3-puzzle` | Mamba-2, latent MoE, and attention hybrid |
| [`nemotron-3-super-120b-a12b`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) | `eider-nemotron3-super` | 120B-A12B Nemotron hybrid |
| [`ling-3.0-tiny-nvfp4`](https://huggingface.co/inclusionAI/Ling-3.0-tiny-fp8) | `eider-ling-3.0-tiny` | Small hybrid model for runtime development |
| [`bitnet-b1.58-2b-4t`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) | `eider-bitnet-b1.58-2b` | Native BitNet b1.58 weights |

[DeepSeek V4 Flash](https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4)
uses a local preparation path. The complete source checkpoint is larger than
the Spark unified memory. Read the
[expert-storage notes](docs/deepseek4-experts.md) for the storage layout.

The generic launcher starts Gemma 4 by default. Select another catalogue model
with `EIDER_MODEL`:

```sh
scripts/run-eider
EIDER_MODEL=muse-glimmer-30b-nvfp4 scripts/run-eider
```

## Other model performance

The table contains representative single-session API results on one GB10.
These values are not peak kernel rates.

| Model | Eider decode tok/s | vLLM decode tok/s | Configuration |
| --- | ---: | ---: | --- |
| Qwen3.6 35B-A3B | 72.7 | 77.0 | NVFP4 weights, compact FP4 KV in Eider |
| Agents-A1 | 63.6 | 37.2 | FP8-converted attention and LM head in Eider |
| Gemma 4 26B-A4B | 30.1 | 29.6 | Same ModelOpt NVFP4 weights |
| Muse Glimmer 30B | 15.8–28.5 | — | DFlash, from a long Pi session to a short prompt |
| Step-3.7-Flash | 20.4 | — | 240 of 288 routed experts resident per layer |
| Laguna-S-2.1 | 16.2 | — | Resident NVFP4 experts and compact FP4 KV |

Use the telemetry from `eider-serve` for comparisons. It reports prompt cache
hits, prefill rate, time to first token, decode rate, and speculative
acceptance.

## Tomatoes smoke test

`scripts/tomatoes.sh` sends one fixed Chat Completions request to Eider. The
prompt asks for a detailed twenty-point guide to growing tomatoes.

The script needs `curl`, `jq`, and a running Eider server.

Start the server first. Then run the script in a second terminal:

```sh
scripts/tomatoes.sh
```

With no argument, the script reads `/v1/models` and uses the served model. An
explicit argument requires the server to report the same model:

```sh
scripts/tomatoes.sh eider-qwen3.8
scripts/tomatoes.sh qwen3.8-27b
```

The script uses a 4,000-token output limit by default. If you need a shorter
smoke test, set `EIDER_TOMATO_MAX_TOKENS`:

```sh
EIDER_TOMATO_MAX_TOKENS=256 scripts/tomatoes.sh
```

Use these environment variables for other server configurations:

| Variable | Purpose | Default |
| --- | --- | --- |
| `EIDER_BASE_URL` | Eider server URL | `http://127.0.0.1:8080` |
| `EIDER_API_KEY` | Bearer token | Not set |
| `EIDER_TOMATO_MAX_TOKENS` | Maximum output tokens | `4000` |
| `NO_COLOR` | Disable terminal colours when set | Not set |

The final report includes prompt tokens, cached tokens, prefill rate, decode
rate, time to first token, and wall time. The answer goes to standard output.
Progress and performance data go to standard error.

## Serving

List the pinned catalogue without a model download:

```sh
cargo run --release -p eider-api --bin eider -- model list
```

Start a catalogue model directly:

```sh
eider-serve qwen3.8-27b
eider-serve qwen3.6-35b-a3b
eider-serve muse-glimmer-30b-nvfp4
eider-serve step-3.7-flash --offline
```

The direct Qwen3.8 command uses conservative server capacities. The dedicated
launcher selects the 262K context profile for a single Pi session.

The first online start resolves an immutable Hugging Face revision. Eider puts
derived artifacts below `$XDG_CACHE_HOME/eider/models/`.

Use `--model-dir` only for local development checkpoints. Catalogue IDs select
pinned deployments.

Both API adapters use the same scheduler and model runtime:

- `POST /v1/responses`
- `POST /v1/chat/completions`
- `GET /healthz`
- `GET /metrics`

The server supports streaming, tool history, sampling, cancellation, concurrent
requests, and prompt-prefix caching. Run `eider-serve --help` for the complete
command reference.

Set `EIDER_API_KEY` to require bearer authentication.

### Pi

The Pi launchers use `pi/agent/models.json` from this repository. They do not
change the global Pi configuration.

```sh
scripts/run-pi-eider-qwen38.sh
scripts/run-pi-eider-qwen.sh
scripts/run-pi-eider-agents-a1.sh
scripts/run-pi-eider-stepfun.sh
scripts/run-pi-eider-laguna.sh
scripts/run-pi-eider-muse.sh
scripts/run-pi-eider-gemma4.sh
scripts/run-pi-eider-nemotron3-super.sh
scripts/run-pi-eider-deepseek4.sh
```

The Qwen3.8 Pi launcher uses `medium` reasoning by default. Set
`PI_EIDER_THINKING=xhigh` for tasks that need more reasoning tokens.

The launchers use the Responses API by default. Set
`PI_EIDER_PROVIDER=eider-chat` to use Chat Completions.

### Codex

```toml
model = "eider-qwen3.8"
model_provider = "eider"

[model_providers.eider]
name = "Eider"
base_url = "http://127.0.0.1:8080/v1"
env_key = "EIDER_API_KEY"
wire_api = "responses"
```

## Why Eider exists

The DGX Spark is not a data-centre GPU. Its SM12x Blackwell GPU does not use
the B200 programming model.

Eider treats the 128 GB unified memory as one budget. Weights, active sequence
state, retained prefixes, CUDA workspaces, and host processes share this
budget.

The runtime uses bounded page pools and explicit cache budgets. Large MoE
models can keep selected experts resident and page the remaining experts from
disk.

Each model keeps its native sequence state. Nemotron Mamba state does not use a
KV-cache abstraction. Qwen3.8 target state and DFlash2 state advance together.

This narrow hardware target keeps performance decisions visible. It also makes
the runtime useful as a place to study modern inference systems.

## Development

The workspace has three main crates:

- `nvfp4` contains CUDA kernels, cuBLASLt plans, device storage, checkpoint
  formats, and GPU benchmarks.
- `infer` contains model execution, sampling, scheduling, sequence caches, and
  model benchmarks.
- `eider-api` contains catalogue deployment, the inference actor, HTTP APIs,
  streaming, and telemetry.

Build and test the workspace:

```sh
cargo build --workspace
cargo build --release -p eider-api --bin eider-serve
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Configure the repository-local CUTLASS tree for SM121:

```sh
scripts/setup-cutlass-sm12x.sh
source .deps/cutlass-sm12x.env
scripts/probe-cutlass-sm12x.sh
```

Focused benchmarks use
[`micromeasure`](https://github.com/rdaum/micromeasure). Each benchmark has a
correctness gate before it records timing data.

```sh
cargo bench -p nvfp4 --bench qwen36_routed_moe_decode
cargo bench -p infer --bench qwen36_prefill
cargo bench -p infer --bench step37_prefill
cargo bench -p infer --bench laguna_prefill
```

A kernel result is not sufficient evidence for a server improvement. Measure
the complete model path before you keep an optimization.

## Troubleshooting

If CUDA cannot target `sm_121`, use one CUDA 13.x installation for all CUDA
components. Then run:

```sh
scripts/probe-cutlass-sm12x.sh
```

If an offline catalogue start fails, fetch the model once online:

```sh
cargo run --release -p eider-api --bin eider -- model fetch MODEL_ID
```

If a CUDA allocation fails, stop other GPU and memory-heavy processes. GPU
allocations and host processes share the Spark unified memory.

Reduce the context, the prefill capacity, or the number of active sequences to
reduce the memory footprint.

## Authorship

The project began as a hand-written study of the GB10 GPU and its software
stack. AI assistance now contributes to implementation and performance work,
especially for boilerplate, CUDA wrappers, and benchmark infrastructure.

This boundary stays visible. The project treats measurable behaviour and
correctness checks as more important than claims about how code was produced.

## Further reading

- [Model deployment](docs/model-deployment.md)
- [Qwen3.6 batching and scheduling](docs/qwen36-batch-decode-plan.md)
- [Step-3.7 expert paging](docs/step37-paging.md)
- [DeepSeek V4 expert storage](docs/deepseek4-experts.md)
- [SM12x CUTLASS and NVFP4](docs/cutlass-sm12x-nvfp4.md)
