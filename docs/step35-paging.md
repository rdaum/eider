# Step-3.5 expert paging

Eider can load the complete Step-3.5-Flash NVFP4 checkpoint on one GB10 while
keeping only a bounded set of routed experts resident. Dense layers and shared
experts remain resident. Routed gate/up weights use Marlin; routed down weights
use the SM12x FP4 tensor-core path.

Prepared expert records contain the Marlin gate/up weights and native SM12x down
tiles. Resident attention and shared-expert linears also use native SM12x tiles.
Their activations are represented by a primary FP4 vector plus two residual FP4
vectors; a shared power-of-two amplification keeps the residual scales in the
useful E4M3 range. The compact SM12x KV cache supports both the 64-head full
attention and 96-head sliding-attention layers.

Against the Python reference, the production path keeps the same top-eight
routes in the focused layer probe. Router-logit NRMSE is at most 0.001832 and
complete-layer output NRMSE is at most 0.003972 across layers 0, 1, 3, and 4.

## Resident capacity

The following deterministic measurement starts from token 0, feeds each greedy
output token back into the model, uses the compact FP4 KV cache, and generates
512 tokens. Paging counters exclude no work; the first token starts with empty
expert caches. The model retains 240 routed-expert slots per MoE layer and
reports 88.049 GiB of device weights after loading.

| Token window | Decode rate | Paging misses |
| ---: | ---: | ---: |
| 0–64 | 4.803 tok/s | 4,683 |
| 64–128 | 10.446 tok/s | 1,348 |
| 128–192 | 11.388 tok/s | 1,120 |
| 192–256 | 12.888 tok/s | 815 |
| 256–320 | 15.119 tok/s | 490 |
| 320–384 | 17.826 tok/s | 194 |
| 384–448 | 18.044 tok/s | 170 |
| 448–512 | 16.949 tok/s | 270 |

The complete run reaches 11.332 tok/s including the cold-cache ramp, with 9,090
misses and a 94.716% hit rate. The last four 64-token windows sustain at least
15 tok/s while continuing to fault experts, so this is a paged result rather
than a fully resident replay. Workloads with a broader expert working set should
still be measured rather than inferred from this trajectory.

Reproduce either run with:

```sh
target/release/step35-generate \
    models/step-3.5-flash-nvfp4 240 0 512
```

The positional argument order is model directory, slots per MoE layer, initial
token, and generated token count.
