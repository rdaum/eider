# Nemotron Super 3.5 VL and Bolt text support

Status: preliminary loader support, 2026-10-15. End-to-end validation with released
weights remains pending.

[llama.cpp PR #30520](https://github.com/ggml-org/llama.cpp/pull/30520)
adds `NemotronH_Omni_Reasoning_V3` for Super 3.5 VL and Bolt 74B-A9B.
The inspected head is `e64207bab6c2989a4ae5aa8b9e0cbe700e135ee3`.
The GitHub API reports that the PR merged on 2026-10-15.

[NVIDIA announced](https://blogs.nvidia.com/blog/local-ai-rtx-spark-laptops-software-updates/)
Bolt 74B-A9B for October 15. The advertised 24 GB checkpoint uses 2-bit
quantization and quantization-aware training. That announcement does not establish
an NVFP4 checkpoint format. The NVIDIA Hugging Face listings did not expose Bolt
or Super 3.5 weights during this investigation.

## Prepared text path

Eider recognises the exact Omni architecture and validates its nested
`llm_config` or `text_config`. The language model uses the existing Nemotron-H
runtime. Tensor names retain the `language_model.` prefix through backbone,
LM head, and MTP loading. The loader does not rewrite snapshot files.

Each MoE layer reads its expert count from the BF16 router shape. The count must
fit the configured maximum, top-k selection, and expert groups. Router buffers,
NVFP4 expert tables, and MTP allocations use that count. Existing Super and Puzzle
configurations retain their layer widths and routing counts.

The change adds no catalogue entry or checkpoint revision. Local development
checkpoints use `--model-dir` after their format is verified. Recognition of the
wrapper does not establish that every inner configuration can run.

## Validation

Synthetic safetensors fixtures cover both nested configuration keys and the
original unwrapped format. A small GPU fixture loads Mamba, attention, NVFP4
experts, and MTP under the wrapped namespace. Its MoE layers contain four, three,
and two experts against a configured maximum of eight.

Single-token and batched GPU routing match a CPU reference. Tests reject invalid
expert counts, incompatible groups, conflicting nested configurations, and
workspaces from layers with different expert counts. API tests verify that local
deployment and inference select the same runtime.

These tests establish loader and routing behaviour. They do not establish model
quality, throughput, or parity with NVIDIA inference.

## Remaining checkpoint work

- Inspect the released configuration, tokenizer, template, tensor index, and
  quantization metadata. Pin the revision after validation.
- Verify the expert layout, latent projections, attention dimensions, and Mamba
  state geometry. The current Eider MoE implementation requires latent projections.
- Determine whether a usable ModelOpt NVFP4 checkpoint exists. Native 2-bit
  support requires its own format and kernel assessment.
- Compare logits and routing against an independent reference. Exercise prefill,
  decode, prefix restoration, graphs, MTP, and tool-call framing.

Vision and video execution remain unimplemented. The upstream work includes a
RADIO encoder, a projector, position-table interpolation, and a separate temporal
patch projection. Eider currently prepares only the language-model loader.
