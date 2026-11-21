# Inkling 2 Mini text execution

Eider serves the pinned
[`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
checkpoint through the `inkling-2-mini-text` catalogue ID and the
`eider-inkling2-mini-text` API model name. This is a fixed-three-pass, text-only
execution path for GB10 (`sm_121`). It is not the multimodal or variable-loop
runtime.

```sh
scripts/run-eider-inkling2.sh
```

The pinned revision is `ed40e7da625f594fc9c4990f5f66be24873c22d7`.
For checkpoint development, `--model-dir` accepts a local Inkling checkout with
its original `config.json`, tokenizer, chat template, and safetensors index.

Start Pi in a separate shell after the server is ready:

```sh
scripts/run-pi-eider-inkling2.sh
# Or use Chat Completions explicitly:
PI_EIDER_PROVIDER=eider-chat scripts/run-pi-eider-inkling2.sh
```

The launchers default to offline operation. The server launcher restricts the
initial deployment to one active sequence and retains up to 8 GiB of prompt
prefixes (`--retained-prefix-gib 8`). Pi defaults
to `eider-inkling2-mini-text` with low reasoning effort, and selects Responses
unless `PI_EIDER_PROVIDER=eider-chat` is set.

## Execution

The 24 stored layers execute as 48 visits: the six-layer prelude, three passes
over 12 shared middle layers, and the six-layer coda. Every visit has its own
attention KV history and short-convolution state. Later middle-layer passes
restart from the prelude's residual blocks and read the previous pass only
through the learned message controllers. A later pass never attends to earlier
passes' K/V. Global K/V convolution weights use current-first taps while their
raw state keeps chronological history; the output is normalized by
`(2 * width + 1)^-0.5`, compared with `(width + 1)^-0.5` for static convolution.

Eider implements normalized Attention Residuals, separate Q/K/V/MLP message
controllers, relative-distance attention bias, projected output gates and
LSE-based attention sinks, local and global attention, and the checkpoint's
capped SwiGLU router/selector semantics. Local KV rings retain 1024 positions;
only the eight global visits keep sequence-length history. The default
context limit is 262144 tokens.

A forward call evaluates a chunk of consecutive tokens visit by visit. A visit
depends only on earlier visits of the same token and on its own KV and
convolution history, so each visit processes every chunk row before the next
visit starts. Dense projections are BF16 GEMMs over the chunk; attention reads
earlier chunk rows directly and appends the chunk to its cache afterwards. Rows
are evaluated in groups whose softmax partials fit the attention workspace, so
long contexts do not need chunk-sized partial storage. Routed experts are sorted
by expert on the device; each work item of up to 64 routes decodes an expert
tile once. Prefill-sized batches stage route activations in shared memory and
tile 128 output rows per CTA, while decode-sized batches split K across warps.
The LM head runs only on the chunk's last row. Prompt chunks default to 1024
tokens. One active sequence is the catalogue default. There is no CUDA graph
replay for Inkling yet.

## Sequence cache

Inkling uses the shared `seqcache` manager. The eight global visits append
BF16 K/V to 128-token page bundles laid out as `[global visit][K/V][page
token][KV head][dimension]`, about 4 MiB per page; relative attention reads
them through each sequence's device page table. The cache reserves a
request's worst-case pages at admission and keeps it waiting while the budget
is exhausted.

The rest of the history is request-private: forty 1024-position local KV rings
and the K, V, attention-output, and MLP-output convolution inputs of all 48
visits, about 172 MB per sequence. A retained prefix shares the global pages
and copies that private history at the prompt's largest page-aligned prefix.
A later request with the same prefix restores the copy and prefills only the
remainder. Because the private history can only be resumed where it was
copied, a retained entry serves prompts that extend that exact prefix, such as
the next turn of an agent session; a new session that shares only the system
prompt does not hit it.

## Checkpoint precision

- Physical layers 0–19 retain packed NVFP4 expert weights and execute through
  Eider's grouped SM121 W4A16 expert kernels. N-major gate/up storage and
  per-expert outer scales are converted to the native weight tiles at load.
- Physical layers 20–23 use an explicit MXFP8-to-BF16 preparation fallback.
  This uses more resident memory than the source MXFP8 storage and is not the
  reference runtime's native W8A8 path.
- The W4A16 path does not reproduce the reference deployment's Four Over Six
  activation quantization or FP4 activation arithmetic. Numerical comparisons
  must therefore check model behaviour and tolerances rather than assume
  bitwise equivalence.
- The language model keeps the padded-vocabulary mask and final logit softcap.

## Prompts and tools

Prompts are rendered natively to token IDs, as the reference frontend does,
rather than through the checkpoint's Jinja template. Tool declarations form a
leading `tool_declare` system block of compact, key-sorted JSON; the thinking
effort level follows the initial system messages, and disabled thinking renders
effort zero. Assistant history renders thinking, text, and one
`<|content_invoke_tool_json|>` block per call with the function as block
author; tool results take their function name from the matching call. Payload
text is encoded with special tokens split, so message content cannot produce
framing tokens. A fixture generated by vLLM's `render_inkling2_messages`
checks the rendering token for token.

Inkling samples its own assistant header. Real control-token IDs select
Thinking, Text, or JSON tool blocks; ordinary control-marker text remains
content. Each parallel call is a separate block that may be followed by text,
so a turn ends at `<|content_model_end_sampling|>` and reports `tool_calls`
when it produced any. Calls to undeclared functions are errors. Only `auto` and
`none` tool choice are accepted by the API adapters.

## Current boundary

- Vision, image inputs, DSpark/MTP, variable loop counts, promoted MXFP8 KV,
  and native MXFP8 activations are not implemented.

The SGLang implementation at commit
`92e894ef904bfdb5c0cf2f2a541cbd4eec023573` is the executable specification for
the loop and attention semantics. Thinking Machines' native NVFP4 BBBK training
kernel is not part of the released checkpoint and is not linked into Eider.

## Validation

```sh
cargo test -p eider-cuda --lib kernels::inkling2::tests -- --test-threads=1
cargo test -p eider-runtime -p eider-inference -p eider-api inkling2 -- --test-threads=1

cargo run --release -p eider-inference --example inkling2_logits -- \
    /path/to/Inkling-2-Mini "Explain looped inference in one sentence."
```

The focused tests use CPU references for Attention Residual math, L2 clipping,
short-convolution history updates, route normalization, relative-attention
sink gating, asymmetric dynamic tap orientation, and grouped BF16 experts.
Chunked convolution and relative attention, including chunks longer than the
local ring and row-grouped attention, are checked against single-row steps;
grouped W4A16 experts are checked against the routed GEMV kernel. The example loads the full
checkpoint and performs text generation through the model-owned execution path
and output codec.

### Reference comparison

Eider was compared with an independent PyTorch W4A16 implementation of the
transformers Inkling2 layers and vLLM's `Inkling2LoopModel` loop. That
implementation uses BF16 activations into dequantized NVFP4/MXFP8 experts, as
vLLM's Hopper expert path does. For a 26-token chat prompt, all 48 visits
selected the same expert sets with residual partials within BF16 drift, and
the first 12 greedy tokens were identical. The PyTorch and vLLM references
began with the same reasoning text: `The user asks a simple arithmetic
question`.

Through the server, a 3527-token prompt that wraps every local KV ring
answered correctly. Both Chat Completions and streaming Responses returned
reasoning followed by text. Chunked prefill produced the same greedy tokens as
single-token prefill for the 26-token prompt. With 1024-token chunks, the
3527-token prompt prefilled at about 965 tokens/s (3.7 s to the first token);
decode runs at about 19 tokens/s on GB10.

Pi completed read, write, and execute tool tasks through both the Responses and
Chat Completions providers. A session's first turn prefilled its 2055-token
prompt in about 2.1 s; later turns restored 2048–2304 cached tokens and reached
the first token in 0.27–0.58 s. With the paged global KV, the 26-token greedy
probe reproduces its tokens, and a 1923-token prompt that restores 1920 cached
tokens generates the same tokens as the uncached run.

The reference deployment's SM12x expert path quantizes activations with the
Four Over Six NVFP4 encoding. Eider's W4A16 path, like vLLM's Hopper path, keeps
BF16 activations, so token-level parity with SM12x vLLM is not expected over
long generations. Cancellation, multi-sequence batching, and decode graph
replay have not been validated for this family.
