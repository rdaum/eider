# Nemotron 3.5 VL LiLiCorr assessment

Status: checkpoint metadata and reference code inspected, 2026-10-15.
Eider does not yet load or execute this drafter.

## Checkpoint contract

[NVIDIA's NVFP4 LiLiCorr checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr)
uses revision `7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2`. Its single safetensors
file is 1,317,952,596 bytes, about 1.23 GiB. This inspection fetched metadata and
the safetensors header only.

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

Existing code provides useful components:

- `crates/infer/src/qwen3/qwen36/dflash2.rs` implements draft context storage,
  noncausal attention, grouped dynamic convolution, and captured target features.
- `crates/eider-cuda/src/kernels/non_gemm.rs` provides the associated CUDA kernels
  and candidate selection primitives.
- `crates/infer/src/nemotron3/model.rs` provides block verification and transactional
  Mamba snapshots. The Rust and CUDA argmax verifier currently caps batches at
  four tokens; the eight-position LiLiCorr block needs that bound extended.

The existing DFlash2 loader is tied to Qwen target geometry. Its selector uses a
hidden projection and predecessor/successor codebooks. Those tensors do not match
LiLiCorr's transformer head. The native Nemotron MTP flag cannot load this drafter.

A concrete implementation needs these steps:

1. Load the exact draft architecture and validate every tensor against the pinned
   metadata. Handle mixed BF16/NVFP4 linears, biases, and the separate LM head.
2. Capture the five target layers during prefill, decode, and verification. Match
   the reference's layer-output and token-position conventions. Eider currently
   exposes only final hidden rows for Nemotron MTP.
3. Implement the correlation head and proposal selection. Compare candidate IDs,
   log probabilities, anchor scores, transitions, and selected paths against an
   independent reference before enabling serving.
4. Extend and test the verification size, then connect draft context updates.
   Commit only accepted features and truncate draft KV state after rejection. Preserve both states
   through prefix restoration and graph replay.
5. Establish greedy parity first. Stochastic serving requires its own acceptance
   and resampling proof; the current Nemotron argmax verifier does not supply it.
6. Measure complete requests on GB10. Include draft execution, target verification,
   state snapshots, prefix restoration, and additional memory in the comparison.

Keep proposal execution and state commit on explicit CUDA streams. Allocate
scratch and graph buffers once per supported shape. With Eider's current anchor
plus draft verification convention, eight FP32 snapshot slots require 32 MiB per
Mamba layer. Across 40 Mamba layers, that is 1.25 GiB per sequence, before live
state and other workspaces.

The model card reports coding acceptance lengths of 5.65 at temperature zero and
5.29 at temperature one with seven speculative tokens. These are reference
acceptance results, not Eider throughput measurements. They justify investigating
this path; its net benefit on GB10 remains unmeasured.

The algorithm is described in the [LiLiCorr paper](https://arxiv.org/abs/2608.20530).
The [llama.cpp conversion](https://github.com/ggml-org/llama.cpp/blob/e64207bab6c2989a4ae5aa8b9e0cbe700e135ee3/conversion/nemotron.py)
and [draft graph](https://github.com/ggml-org/llama.cpp/blob/e64207bab6c2989a4ae5aa8b9e0cbe700e135ee3/src/models/dflash.cpp)
provide another implementation to compare against.
