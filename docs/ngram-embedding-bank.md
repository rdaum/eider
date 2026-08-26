# Quantized n-gram embedding bank

Eider's n-gram input primitive is independent of a model-family adapter. It
implements the public LongCat-Flash-Lite contract: every token selects one row
from each order/split table, applies one table-specific projection, adds the
word embedding, and averages all sources.

The device bank supports three row-major formats:

| Format | Values | Scale storage |
| --- | --- | --- |
| BF16 | 16 bits per element | None |
| FP8 | 8 bits per element | One F32 value per row |
| NVFP4 | Two E2M1 values per byte | One UE4M3 byte per 16 columns |

The NVFP4 layout costs exactly 4.5 bits per parameter when the row width is a
multiple of 16. It is a compact embedding-row layout. It is not a cuBLASLt or
native `mma.sync` scale layout and must not be reinterpreted as one.

For the provisional 125-billion-parameter model plus 51-billion-parameter
n-gram memory estimate, the raw NVFP4 payloads are approximately 65.5 GiB and
26.7 GiB. Their combined 92.2 GiB leaves approximately 35.8 GiB of a 128 GiB
GB10 pool for other weights, CUDA workspaces, recurrent state, KV pages, and
staging. These figures exclude alignment and metadata.

The fused kernel stages only the selected embedding rows in shared memory. It
does not materialize `[tokens, tables, embedding_dim]` in device memory. It
then streams BF16 projection weights in `[table, embedding_dim, hidden]` order,
adds the F32 word embedding, and writes one averaged F32 output row. Row IDs,
word embeddings, projection weights, and output buffers remain at stable
addresses and are suitable for CUDA Graph capture.

`cargo bench -p nvfp4 --bench ngram_embedding` validates the fused NVFP4 path
against the scalar reference before timing BF16, FP8, and NVFP4 decode plus a
128-token NVFP4 prefill shape.
