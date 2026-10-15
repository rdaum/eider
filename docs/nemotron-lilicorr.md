# Nemotron 3.5 VL LiLiCorr

Eider loads the released companion and supports one to seven proposals per
cycle. The Nemotron 3.5 launcher enables LiLiCorr by default. The target and
companion use the same prefix cache and preserve Mamba state through partial
acceptance.

## Checkpoint contract

[NVIDIA's NVFP4 LiLiCorr checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr)
uses revision `7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2`. Its single safetensors
file is 1,317,952,596 bytes, about 1.23 GiB.

| Component | Released configuration |
| --- | --- |
| Architecture | `LiLiCorrDraftModel`; the `qwen3` model type alone is insufficient |
| Draft block | Eight positions: one anchor and seven proposals |
| Target features | Layers `[1, 22, 43, 64, 85]` from the 88-layer target |
| Feature projection | 20,480 inputs to 4,096 outputs |
| Backbone | Five layers; hidden 4,096; FFN 12,288 |
| Attention | 32 query heads, eight KV heads, head dimension 128 |
| Context | 8,192-position sliding window; noncausal draft attention |
| Position encoding | RoPE theta 1,000,000 |
| Dynamic convolution | Two taps, group size 16, around attention and MLP |
| Candidate head | Top eight candidates per slot; 56 candidate nodes |
| Correlator | Two layers, hidden/factor width 1,024, eight attention heads |
| Vocabulary | 131,072; mask token 990; its own NVFP4 LM head |

Tensor names start with `layers.`, `fc.`, `lilicorr.`, and `lm_head.`; they have
no `model.` prefix. The header contains 178 tensors. Attention weights and several
correlator projections remain BF16. FFNs, the feature projection, the LM head, and
some correlator linears use ModelOpt `W4A16_NVFP4`, with K16 scales. Loading this
recipe must preserve BF16 activations; it does not request W4A4 quantization.

## Correlator behaviour

The [vLLM implementation](https://github.com/vllm-project/vllm/blob/73a78e6f1f38e280986b81e0f2a9aa5e1ee6fe47/vllm/model_executor/models/lilicorr.py)
uses a DFlash backbone with grouped dynamic convolution. Its correlation head
consumes token embeddings, draft hidden states, and candidate log probabilities.

Each candidate gets five features: log probability, probability, distance from
the best log probability, fractional rank, and a top-rank indicator. A LayerNorm
and MLP transform those features. The head adds projected token and draft states,
plus learned slot and rank embeddings.

Two transformer layers attend over all 56 candidates. Learned per-head biases
encode relative slot distance and same-slot membership. These layers have no
persistent KV cache. The output produces normalised incoming and outgoing factor
vectors, conditioned on a projected target anchor. Dot products produce anchor
scores and adjacent-slot transition scores, scaled by eight.

Preserve full-vocabulary log-softmax values before top-eight selection. Ranking,
LayerNorm biases, the anchor term, and the slot-bias offset are trained semantics.
Shorter proposals must retain the original slot-bias offset and use the trained
prefix of the lattice.

## Eider integration

Download the companion into the normal Hugging Face cache:

```sh
hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr \
  --revision 7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2
scripts/run-eider-nemotron35.sh
```

The launcher resolves that pinned snapshot and enables three proposals by default.
Set `EIDER_LILICORR=0` to run without the companion.
Set `LILICORR_DIR` to select a different local snapshot path.
Explicit server arguments still take precedence.
`EIDER_SPECULATIVE_DRAFTS=0` disables proposals. Values one through seven select
the proposal length. The direct server option is
`--lilicorr-dir DIR`. It defaults to three proposals unless overridden.

`crates/infer/src/nemotron3/lilicorr/` validates all 178 consumed tensors and
implements the backbone, correlator, and sequential candidate walk. Target
features come from committed prefill, decode, or verification rows. Rejected
rows never enter the draft context. Prefix snapshots retain the draft ring
cache and anchor alongside target Mamba and attention state.

Greedy verification downloads only row argmax IDs. Other sampling policies use
full target logits and the realised sparse proposal probabilities. Acceptance
uses `min(1, p/q)`; rejection samples from `max(p-q, 0)`. Target probabilities
include temperature, top-k, top-p, presence penalties, and frequency penalties.
The proposal walk uses temperature over the learned conditional scores. It does
not add the unary draft logits to those scores.

Native W4A16 dense batches compute eight activation rows per MMA tile. Sliding
draft attention splits the context into 256-key partitions and merges their
softmax results. Both paths retain BF16 activation semantics.

Proposal projections, dynamic convolutions, and the correlation head use CUDA
graphs after their first execution. Target verification captures segments between
attention layers. Positions and attention remain explicit launches so graph
replay cannot retain a previous sequence position. All device work uses the
model stream. The scheduler retains the full verification shape and one tail
shape, which bounds recurrent snapshot storage.

Each live draft context uses about 320 MiB for its F32-backed, BF16-valued KV
window. Eight FP32 Mamba snapshot slots add 1.25 GiB per full verification
workspace. Prefix snapshots retain only the populated draft window. Companion
weights, projection scratch, and a shortened verification workspace add further
memory; these costs are separate from the target weights.

Target verification groups routes by expert and shares each decoded weight across
up to four tokens. The kernel stages 1,024 input values at a time and preserves
the previous FP32 dot-product order. Sorting and projection run on the model
stream and participate in verification graphs.

FP8 dense projections now use checkpoint activation scales for batches of two
through seven rows, matching the existing path for larger batches. Small NVFP4
shared projections retain their weight-reusing matvec path. Calibrated FP8 can
change token choices relative to the previous small-batch activation path.

Nemotron prepares the checkpoint's NVFP4 dense weights once for batched
projections. The shared experts add about 3.3 GiB of prepared BF16 weights.
Single-token decode retains packed weights. Explicit BF16/FP8-to-NVFP4 storage
policies retain their compact representation and expand batch weights in scratch.

## Validation

`scripts/nemotron-lilicorr-reference.py` produces a CPU PyTorch reference from
local checkpoints. It can consume real target features saved by the ignored
`capture_lilicorr_reference_context` test. It does not download or alter weights.

The checkpoint comparison checks context projections, individual backbone layers,
the independent correlation head, full draft logits, candidate overlap, and exact
prefix replay. The full-target check compares 64 generated tokens with ordinary
decoding and exercises both seven-token and shortened proposals. The serving
check covers prefix reuse, seeded stochastic sampling with penalties, and queued
and active cancellation.

```sh
# Set these to local immutable HF snapshots.
export NEMOTRON_MODEL_DIR=/path/to/target
export LILICORR_DIR=/path/to/companion
cargo test --release -p eider-inference --lib \
  lilicorr_commits_match_serial_target -- --ignored
cargo test --release -p eider-inference --lib \
  lilicorr_serving_restores_prefix_and_preserves_request_policies -- --ignored
```

Each test loads the full target. Run them separately with the server stopped.
For the serving check, `LILICORR_PROMPT` selects a text prompt file and
`LILICORR_SERVING_REPORT` writes request timings and acceptance counts to JSON.
`LILICORR_TEST_DRAFTS` selects the proposal count for that check; it defaults to seven.

## GB10 measurements

The 2026-10-15 serving-engine replay used one code-review request with 11,298
prompt tokens and 128 output tokens. A repeated request restored 11,264 prompt
tokens. These are single-run measurements with the released NVFP4 target and
companion, without an HTTP adapter.

| Mode | Cold TTFT | Cached TTFT | Cached decode | Accepted drafts/cycle |
| --- | ---: | ---: | ---: | ---: |
| Initial target only, across both runs | 19.75–20.72 s | 547–549 ms | 15.12–15.25 tok/s | — |
| Initial three LiLiCorr proposals | 22.17 s | 602 ms | 14.93 tok/s | 2.10 |
| Initial seven LiLiCorr proposals | 22.57 s | 605 ms | 14.34 tok/s | 2.71 |
| Updated target only | 19.01 s | 530 ms | 15.12 tok/s | — |
| Updated seven LiLiCorr proposals | 19.23 s | 554 ms | 18.10 tok/s | 2.71 |
| Expert reuse, seven proposals | 19.76 s | 550 ms | 20.97 tok/s | 2.71 |
| Expert reuse and calibrated short batches, three proposals | 19.85 s | 549 ms | 21.52 tok/s | 1.74 |

The first update uses batched W4A16, split-context attention, and prepared target
dense weights. Cached request time fell from 9.46 seconds to 7.57 seconds.
Decode improved 26% over the initial seven-proposal path and 20% over the paired
target-only run. Both versions accepted 92 drafts across 34 cycles.
With expert reuse and calibrated short batches, three proposals completed the
cached request in 6.45 seconds. They accepted 80 drafts across 46 cycles.
Seven proposals took 6.61 seconds and retained 92 accepted drafts across 34 cycles.
Three proposals improved decode by 19% over the previous default and reduced
cached request time by 15%. These single-workload results select the current default.

Prefix replay, seeded sampling, and cancellation passed. These throughput figures
use greedy decoding; they do not establish the same gain for stochastic requests.
The launcher enables LiLiCorr with three proposals by default.
The direct server selects three proposals when supplied with `--lilicorr-dir`.
Set `EIDER_SPECULATIVE_DRAFTS=7` to use the full proposal length.

The focused `lilicorr_kernels` benchmark checks dense outputs against the prior
per-row kernel and attention against the serial implementation. It covers partial
MMA tiles, empty and short contexts, and a wrapped 8,192-token ring.
The candidate LM head fell from 11.74 ms to 1.46 ms; attention fell from 3.57 ms
to 1.24 ms per layer. Prepared shared projections preserve exact outputs and
measured 2.1–2.5 times faster than projection with repeated expansion.

```sh
cargo bench -p eider-cuda --bench lilicorr_kernels
cargo bench -p eider-cuda --bench nemotron35_dense -- 'prepared 8 rows'
```

The `nemotron35_experts` micromeasure covers verification sizes of two, four,
and eight rows. It compares every output against the previous SIMT kernel,
including repeated experts and partial groups. The new kernel matches bit for bit.

At eight rows with four routes sharing each expert, up/down times fell from
1.067/0.884 ms to 0.731/0.554 ms, including sorting. With distinct experts, up
was unchanged and down cost about 4% more. Routing overlap therefore determines
the benefit.

The four-row Mamba input projection fell from 1.345 ms to 0.415 ms with calibrated
FP8. Its benchmark checks against a reference with the same activation quantization.
The shared NVFP4 projection remained faster on its small-batch matvec path:
0.120 ms versus 0.164 ms with prepared dense GEMM weights.

```sh
cargo bench -p eider-cuda --bench nemotron35_experts -- verification
cargo bench -p eider-cuda --bench nemotron35_dense -- Verification
```

The model card reports coding acceptance lengths of 5.65 at temperature zero and
5.29 at temperature one with seven speculative tokens. These are reference
acceptance results, not Eider throughput measurements. They justify investigating
this path; whole-request measurements determine its net benefit on GB10.
The table above excludes the guaranteed next token from accepted drafts per cycle.
Adding that token gives 3.71 under vLLM's acceptance-length convention.

The algorithm is described in the [LiLiCorr paper](https://arxiv.org/abs/2608.20530).
The [llama.cpp conversion](https://github.com/ggml-org/llama.cpp/blob/e64207bab6c2989a4ae5aa8b9e0cbe700e135ee3/conversion/nemotron.py)
and [draft graph](https://github.com/ggml-org/llama.cpp/blob/e64207bab6c2989a4ae5aa8b9e0cbe700e135ee3/src/models/dflash.cpp)
provide another implementation to compare against.
