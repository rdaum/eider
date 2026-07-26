# Step-3.7 expert paging

Eider can load the complete Step-3.7-Flash NVFP4 checkpoint on one GB10 while
keeping only a bounded set of routed experts resident. Dense layers and shared
experts remain resident and are converted to NVFP4 by default. Routed gate/up
weights use Eider's SM121 W4A16 kernel; routed down weights use the SM12x FP4
tensor-core path.

The official checkpoint stores each routed projection as one stacked
all-expert tensor. Preparation reads one expert slice at a time and writes
fixed-size records containing SM121 W4A16 gate/up weights and native SM12x down
tiles. The checkpoint's BF16 attention projections, first three dense FFNs,
shared experts, and LM head are quantized to native NVFP4 storage at load time.
The compact SM12x KV cache supports both the 64-head full attention and the
96-head, 512-token sliding-attention layers.

Against the Python reference, the production path keeps the same top-eight
routes across representative dense, full-attention, sliding-attention, and MoE
layers. The complete layer outputs remain within the focused probe tolerances.

## Preparing and running

The prepared cache occupies about 100 GiB on disk. With 240 routed-expert slots
per layer, the default NVFP4 configuration reports 87.3 GiB of resident device
weights, leaving unified-memory headroom for compact KV state and runtime
workspaces.

## Resident-weight conversion

The deterministic probe used 64 generated tokens, two passes, and 240 expert
slots per routed layer. The second pass had no paging misses.

| NVFP4 conversion | Device weights | Decode tokens/sec |
| --- | ---: | ---: |
| None | 95.542 GiB | 12.801 |
| LM head | 94.835 GiB | 13.534 |
| Attention | 89.490 GiB | 18.275 |
| Shared experts | 94.657 GiB | 13.522 |
| Dense MLPs | 94.986 GiB | 13.392 |
| All | 87.342 GiB | 22.362 |

LM-head-only and dense-MLP-only conversion selected the same 64 greedy tokens
as the BF16 run. Attention and shared-expert conversion changed the synthetic
greedy sequence, so token identity is not an accuracy claim. A warmed
1,024-output-token API request with all conversions enabled decoded at 20.376
tokens/sec.

Prepare all routed layers once with:

```sh
cargo run --release -p infer --bin step37-experts -- \
    prepare models/step-3.7-flash-nvfp4
```

Run the deterministic decode probe with:

```sh
target/release/step37-generate \
    models/step-3.7-flash-nvfp4 240 0 64 2
```

The positional argument order is model directory, slots per MoE layer, initial
token, generated token count, and pass count. Set any of
`STEP37_BF16_ATTENTION`, `STEP37_BF16_DENSE_MLP`,
`STEP37_BF16_SHARED_EXPERT`, or `STEP37_BF16_LM_HEAD` to `bf16` to isolate the
checkpoint-native path.
