# Marlin source attribution

The files below this directory are derived from vLLM's Apache-2.0 licensed
Marlin implementation, which in turn is adapted from IST-DASLab/Marlin:

```text
https://github.com/vllm-project/vllm
csrc/libtorch_stable/quantization/marlin/
csrc/libtorch_stable/moe/marlin_moe_wna16/marlin_template.h
```

Only the low-level CUDA headers are included. Torch dispatch, tensor ownership,
and runtime configuration remain implemented by nvfp4.
