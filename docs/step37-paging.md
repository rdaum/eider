# Step-3.7 expert paging

Eider can load the complete Step-3.7-Flash NVFP4 checkpoint on one GB10 while
keeping only a bounded set of routed experts resident. Dense layers and shared
experts remain resident. Routed gate/up weights use Marlin; routed down weights
use the SM12x FP4 tensor-core path.

The official checkpoint stores each routed projection as one stacked
all-expert tensor. Preparation reads one expert slice at a time and writes
fixed-size records containing Marlin gate/up weights and native SM12x down
tiles. Attention, dense FFNs, and shared experts remain in their checkpoint
BF16 representation. The compact SM12x KV cache supports both the 64-head full
attention and the 96-head, 512-token sliding-attention layers.

Against the Python reference, the production path keeps the same top-eight
routes across representative dense, full-attention, sliding-attention, and MoE
layers. The complete layer outputs remain within the focused probe tolerances.

## Preparing and running

The prepared cache occupies about 100 GiB on disk. With 240 routed-expert slots
per layer, the complete text model reports 95.5 GiB of resident device weights,
leaving unified-memory headroom for compact KV state and runtime workspaces.

Prepare all routed layers once with:

```sh
cargo run --release -p infer --bin step37-experts -- \
    prepare models/step-3.7-flash-nvfp4
```

Run the deterministic decode probe with:

```sh
target/release/step37-generate \
    models/step-3.7-flash-nvfp4 240 0 512
```

The positional argument order is model directory, slots per MoE layer, initial
token, and generated token count.
