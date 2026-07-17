# Agents-A1 BF16 weight storage

Agents-A1 stores its attention projections and LM head in BF16. Eider can keep
those weights in BF16 or convert each component independently to FP8 or NVFP4
at load time.

## Short-context decode trial

These results were measured on GB10 with a 14-token prompt, 200 greedy decode
steps, one warmup, and two measured repetitions. The reported rate is the
slower of the two measured repetitions.

| Attention | LM head | Decode tokens/sec | Change from FP8/FP8 |
| --- | --- | ---: | ---: |
| FP8 | FP8 | 64.2 | -- |
| NVFP4 | FP8 | 72.9 | +13.6% |
| FP8 | NVFP4 | 72.8 | +13.4% |
| NVFP4 | NVFP4 | 79.0 | +23.1% |

Both conversions independently reduce decode time, and their gains compose.
The all-NVFP4 configuration also produced a coherent two-paragraph response
and a correctly structured calculator tool call through the API. This is a
smoke check rather than an accuracy evaluation, so the Agents-A1 launcher
continues to default both components to FP8.

Reproduce a configuration with:

```sh
target/release/qwen-bench \
    --model models/agents-a1-nvfp4 \
    --prompt 'Explain why a lock-free queue needs careful memory ordering in Rust.' \
    --decode-tokens 200 \
    --warmup-repeats 1 \
    --repeats 2 \
    --temperature 0 \
    --qwen-bf16-attention nvfp4 \
    --qwen-bf16-lm-head nvfp4
```

The result covers short-context decode. It does not imply the same percentage
gain at long context, where full attention over the KV cache consumes a larger
share of each step.
