# FlashAttention 2 forward headers

Source: [vllm-project/flash-attention](https://github.com/vllm-project/flash-attention/tree/b273a02621d3c94bc747df67204f591b67f38fae/csrc/flash_attn/src).
Revision: `b273a02621d3c94bc747df67204f591b67f38fae`.
Licence: BSD-3-Clause; see `LICENSE` and `AUTHORS`.

These files are an unchanged subset of `csrc/flash_attn/src`.
Eider compiles the BF16 forward kernel for 256-wide heads in `dory_flash_attention.cu`.
The wrapper supports causal and sliding-window attention on an explicit CUDA stream.
Decode uses the split-KV forward and combine kernels, with grouped query heads represented as rows.
Dropout is disabled at compile time.
The build uses the existing CUTLASS headers and does not require PyTorch.

If CUTLASS headers are absent, Dory uses its scalar attention path.
