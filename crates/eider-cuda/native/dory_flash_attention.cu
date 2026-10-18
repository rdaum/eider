// Native FA2 forward only: BF16, head dimension 256, causal/global or sliding.
#define FLASHATTENTION_DISABLE_DROPOUT
#include "vendor/flash-attention/flash.h"
#include "vendor/flash-attention/flash_fwd_kernel.h"
using DoryFlashTraits = Flash_fwd_kernel_traits<256, 64, 64, 4, false, false, cutlass::bfloat16_t>;
namespace eider_dory_flash {
template <bool Local>
__global__ void dory_flash_kernel(const __grid_constant__ flash::Flash_fwd_params params) {
    flash::compute_attn<DoryFlashTraits, false, !Local, Local, false, false, true, false, false>(params);
}
template <bool Local>
cudaError_t launch(flash::Flash_fwd_params& p, cudaStream_t stream) {
    static const cudaError_t status = cudaFuncSetAttribute(dory_flash_kernel<Local>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, DoryFlashTraits::kSmemSize);
    if (status != cudaSuccess) return status;
    dory_flash_kernel<Local><<<dim3((p.seqlen_q + 63) / 64, 1, p.h),
        DoryFlashTraits::kNThreads, DoryFlashTraits::kSmemSize, stream>>>(p);
    return cudaGetLastError();
}
}
extern "C" int infer_dory_flash_attention(void* query, void* key, void* value, void* output,
    float* lse, int* key_length, int rows, int capacity, int qheads, int kvheads,
    int window, cudaStream_t stream) {
    flash::Flash_fwd_params p{};
    p.q_ptr=query; p.k_ptr=key; p.v_ptr=value; p.o_ptr=output;
    p.softmax_lse_ptr=lse; p.seqused_k=key_length;
    p.b=1; p.h=qheads; p.h_k=kvheads; p.h_h_k_ratio=qheads/kvheads;
    p.d=p.d_rounded=256;
    p.seqlen_q=p.total_q=rows; p.seqlen_k=capacity;
    p.seqlen_q_rounded=(rows+127)/128*128; p.seqlen_k_rounded=(capacity+127)/128*128;
    p.q_head_stride=p.k_head_stride=p.v_head_stride=p.o_head_stride=256;
    p.q_row_stride=p.o_row_stride=qheads*256; p.k_row_stride=p.v_row_stride=kvheads*256;
    p.q_batch_stride=p.o_batch_stride=int64_t(rows)*qheads*256;
    p.k_batch_stride=p.v_batch_stride=int64_t(capacity)*kvheads*256;
    p.scale_softmax=.0625f; p.scale_softmax_log2=.09016844005556021f;
    p.p_dropout=1.f; p.rp_dropout=1.f; p.scale_softmax_rp_dropout=.0625f;
    p.is_bf16=true; p.is_causal=window==0;
    p.window_size_left=window ? window-1 : -1; p.window_size_right=0;
    return window ? eider_dory_flash::launch<true>(p,stream) : eider_dory_flash::launch<false>(p,stream);
}

namespace eider_dory_flash {
__global__ void dory_split_kernel(const __grid_constant__ flash::Flash_fwd_params p) {
    flash::compute_attn_splitkv<DoryFlashTraits,false,false,false,false,true,false,true,false>(p);
}
__global__ void dory_combine_kernel(const __grid_constant__ flash::Flash_fwd_params p) {
    flash::combine_attn_seqk_parallel<DoryFlashTraits,4,5,true>(p);
}
}
extern "C" int infer_dory_flash_decode(void* query,void* key,void* value,void* output,float* scratch,
    int* key_length,int* leftpad,int capacity,int qheads,int kvheads,int splits,cudaStream_t stream) {
    flash::Flash_fwd_params p{};
    p.q_ptr=query;p.k_ptr=key;p.v_ptr=value;p.o_ptr=output;
    p.oaccum_ptr=scratch;p.softmax_lseaccum_ptr=scratch+splits*qheads*256;
    p.softmax_lse_ptr=scratch+splits*qheads*257;
    p.seqused_k=key_length;p.leftpad_k=leftpad;
    // Reshape GQA's query heads into rows of one KV head. They all attend to
    // the same decode position, so this view is noncausal. leftpad selects
    // the sliding window without assigning different positions to those rows.
    p.b=1;p.h=p.h_k=kvheads;p.h_h_k_ratio=1;
    p.d=p.d_rounded=256;
    p.seqlen_q=p.total_q=qheads/kvheads;p.seqlen_k=capacity;
    p.seqlen_q_rounded=128;p.seqlen_k_rounded=(capacity+127)/128*128;
    p.q_row_stride=p.o_row_stride=256;p.q_head_stride=p.o_head_stride=p.seqlen_q*256;
    p.k_row_stride=p.v_row_stride=kvheads*256;p.k_head_stride=p.v_head_stride=256;
    p.q_batch_stride=p.o_batch_stride=qheads*256;
    p.k_batch_stride=p.v_batch_stride=int64_t(capacity)*kvheads*256;
    p.scale_softmax=.0625f;p.scale_softmax_log2=.09016844005556021f;
    p.p_dropout=p.rp_dropout=1.f;p.scale_softmax_rp_dropout=.0625f;
    p.is_bf16=true;p.window_size_left=p.window_size_right=-1;p.num_splits=splits;
    static const cudaError_t status=cudaFuncSetAttribute(eider_dory_flash::dory_split_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,DoryFlashTraits::kSmemSize);
    if(status!=cudaSuccess)return status;
    eider_dory_flash::dory_split_kernel<<<dim3(1,splits,kvheads),128,DoryFlashTraits::kSmemSize,stream>>>(p);
    eider_dory_flash::dory_combine_kernel<<<(qheads+3)/4,128,0,stream>>>(p);
    return cudaGetLastError();
}

namespace eider_dory_flash {
template<bool Local>
__global__ void dory_branches_kernel(const __grid_constant__ flash::Flash_fwd_params p) {
    // The unsplit cache kernel supports a page table and variable query lengths.
    flash::compute_attn_splitkv<DoryFlashTraits,!Local,Local,false,false,true,false,false,false>(p);
}
template<bool Local>
cudaError_t launch_branches(flash::Flash_fwd_params& p,cudaStream_t stream) {
    static const cudaError_t status=cudaFuncSetAttribute(dory_branches_kernel<Local>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,DoryFlashTraits::kSmemSize);
    if(status!=cudaSuccess)return status;
    dory_branches_kernel<Local><<<dim3((p.seqlen_q+63)/64,p.b,p.h),128,DoryFlashTraits::kSmemSize,stream>>>(p);
    return cudaGetLastError();
}
}
extern "C" int infer_dory_flash_branches(void* query,void* key,void* value,void* output,float* lse,
    int* offsets,int* lengths,int* pages,int branches,int rows,int max_rows,int page_stride,
    int qheads,int kvheads,int window,cudaStream_t stream) {
    flash::Flash_fwd_params p{};
    p.q_ptr=query;p.k_ptr=key;p.v_ptr=value;p.o_ptr=output;p.softmax_lse_ptr=lse;
    p.cu_seqlens_q=offsets;p.seqused_k=lengths;p.block_table=pages;
    p.block_table_batch_stride=page_stride;p.page_block_size=64;
    p.b=branches;p.h=qheads;p.h_k=kvheads;p.h_h_k_ratio=qheads/kvheads;
    p.d=p.d_rounded=256;p.seqlen_q=max_rows;p.total_q=rows;p.seqlen_k=page_stride*64;
    p.seqlen_q_rounded=(max_rows+127)/128*128;p.seqlen_k_rounded=(p.seqlen_k+127)/128*128;
    p.q_head_stride=p.k_head_stride=p.v_head_stride=p.o_head_stride=256;
    p.q_row_stride=p.o_row_stride=qheads*256;p.k_row_stride=p.v_row_stride=kvheads*256;
    p.k_batch_stride=p.v_batch_stride=int64_t(64)*kvheads*256;
    p.scale_softmax=.0625f;p.scale_softmax_log2=.09016844005556021f;
    p.p_dropout=p.rp_dropout=1.f;p.scale_softmax_rp_dropout=.0625f;
    p.is_bf16=true;p.is_causal=window==0;p.unpadded_lse=true;
    p.window_size_left=window?window-1:-1;p.window_size_right=0;
    return window?eider_dory_flash::launch_branches<true>(p,stream):eider_dory_flash::launch_branches<false>(p,stream);
}
