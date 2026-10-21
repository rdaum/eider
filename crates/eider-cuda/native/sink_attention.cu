// Causal GQA with a learned zero-value sink and BF16 ring storage.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

namespace {
__device__ float warp_sum(float x) {
    for (int d=16; d; d>>=1) x+=__shfl_xor_sync(0xffffffff,x,d);
    return x;
}

template <bool Paged, bool DecodeBatch = false>
__global__ void partial_attention(
    const float* q, const __nv_bfloat16* ck, const __nv_bfloat16* cv,
    const float* k, const float* v, float* partial, uint32_t end,
    uint32_t rows, uint32_t qheads, uint32_t kvheads, uint32_t capacity, uint32_t splits, const __nv_bfloat16* const* pages,
    uint32_t layer_offset, uint32_t page_tokens, const uint32_t* position,
    const __nv_bfloat16* const* batch_keys=nullptr,
    const __nv_bfloat16* const* batch_values=nullptr,
    const __nv_bfloat16* const* const* batch_pages=nullptr,
    uint32_t layers=0, uint32_t layer=0) {
    const uint32_t head=blockIdx.x, query=blockIdx.y, part=blockIdx.z, lane=threadIdx.x;
    if (position) end=position[DecodeBatch ? query : 0];
    if constexpr (DecodeBatch) {
        if constexpr (Paged) pages=batch_pages[query];
        else { ck=batch_keys[query*layers+layer]; cv=batch_values[query*layers+layer]; }
    }
    const uint32_t kvhead=head/(qheads/kvheads);
    const uint64_t last=uint64_t(end)+(DecodeBatch ? 0 : query)+1, first=last>capacity ? last-capacity : 0;
    float query_value[4], acc[4]={};
    #pragma unroll
    for (int c=0;c<4;++c) query_value[c]=q[(query*qheads+head)*128+lane+c*32];
    float mx=-INFINITY, total=0.f;
    for (uint64_t i=first+part*256;i<min(last,first+(part+1)*256);++i) {
        const bool cached=i<end;
        const size_t base=(size_t(cached ? (Paged ? (i & 127) : i%capacity) : (DecodeBatch ? query : i-end))*kvheads+kvhead)*128+lane;
        const __nv_bfloat16* pk = cached && Paged ? pages[i >> 7]+layer_offset : ck;
        const __nv_bfloat16* pv = cached && Paged ? pk+size_t(page_tokens)*kvheads*128 : cv;
        float dot=0.f, value[4];
        #pragma unroll
        for (int c=0;c<4;++c) {
            const float key=cached ? __bfloat162float(pk[base+c*32]) : k[base+c*32];
            value[c]=cached ? __bfloat162float(pv[base+c*32]) : v[base+c*32];
            dot=fmaf(query_value[c],key,dot);
        }
        dot=warp_sum(dot)*0.08838834764831845f;
        const float next=fmaxf(mx,dot), keep=expf(mx-next), p=expf(dot-next);
        total=total*keep+p;
        #pragma unroll
        for (int c=0;c<4;++c) acc[c]=acc[c]*keep+p*value[c];
        mx=next;
    }
    float* dst=partial+((size_t(query)*qheads+head)*splits+part)*130;
    #pragma unroll
    for (int c=0;c<4;++c) dst[lane+c*32]=acc[c];
    if(lane==0) { dst[128]=mx; dst[129]=total; }
}

__global__ void merge_attention(const float* partial, const float* sinks, float* out,
                                 uint32_t splits, uint32_t qheads) {
    const uint32_t lane=threadIdx.x,row=blockIdx.x;
    const float* src=partial+size_t(row)*splits*130;
    const float sink=sinks[row%qheads];
    float mx=sink,total=0.f,acc[4]={};
    for(uint32_t s=0;s<splits;++s) mx=fmaxf(mx,src[s*130+128]);
    // Include the sink once, after merging the real-key partitions.
    total=expf(sink-mx);
    for(uint32_t s=0;s<splits;++s) {
        const float p=expf(src[s*130+128]-mx);
        total+=src[s*130+129]*p;
        #pragma unroll
        for(int c=0;c<4;++c) acc[c]+=src[s*130+lane+c*32]*p;
    }
    #pragma unroll
    for(int c=0;c<4;++c) out[row*128+lane+c*32]=acc[c]/total;
}

template <bool Paged, bool DecodeBatch = false>
__global__ void append_kv(const float* k,const float* v,__nv_bfloat16* ck,__nv_bfloat16* cv,
                          uint32_t end,uint32_t rows,uint32_t width,uint32_t capacity,
                          __nv_bfloat16* const* pages, uint32_t layer_offset,
                          uint32_t page_tokens, const uint32_t* position,
                          __nv_bfloat16* const* batch_keys=nullptr,
                          __nv_bfloat16* const* batch_values=nullptr,
                          __nv_bfloat16* const* const* batch_pages=nullptr,
                          uint32_t layers=0, uint32_t layer=0) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(rows)*width) return;
    const uint32_t row=i/width;
    if(position) end=position[DecodeBatch ? row : 0];
    if constexpr (DecodeBatch) {
        if constexpr (Paged) pages=batch_pages[row];
        else { ck=batch_keys[row*layers+layer]; cv=batch_values[row*layers+layer]; }
    } else {
        if(rows>capacity && row<rows-capacity) return;
        end+=row;
    }
    const size_t target=size_t(Paged ? (end&127) : end%capacity)*width+i%width;
    if constexpr(Paged) { ck=pages[end>>7]+layer_offset; cv=ck+size_t(page_tokens)*width; }
    ck[target]=__float2bfloat16_rn(k[i]); cv[target]=__float2bfloat16_rn(v[i]);
}
}

extern "C" cudaError_t infer_causal_sink_attention(
    const float* q, __nv_bfloat16* ck, __nv_bfloat16* cv, const float* k, const float* v,
    const float* sinks, float* partial, float* out, uint32_t end, uint32_t rows,
    uint32_t qheads, uint32_t kvheads, uint32_t capacity, cudaStream_t stream) {
    const uint32_t splits=(min(uint64_t(end)+rows,uint64_t(capacity))+255)/256;
    partial_attention<false><<<dim3(qheads,rows,splits),32,0,stream>>>(q,ck,cv,k,v,partial,end,rows,qheads,kvheads,capacity,splits,nullptr,0,0,nullptr);
    merge_attention<<<qheads*rows,32,0,stream>>>(partial,sinks,out,splits,qheads);
    append_kv<false><<<(size_t(rows)*kvheads*128+255)/256,256,0,stream>>>(k,v,ck,cv,end,rows,kvheads*128,capacity,nullptr,0,0,nullptr);
    return cudaGetLastError();
}

// Page bundles hold [global layer][K/V][page token][KV head][dimension].
// A device position allows the same launch to be replayed in a decode graph.
extern "C" cudaError_t infer_paged_sink_attention(
    const float* q, __nv_bfloat16* ck, __nv_bfloat16* cv, __nv_bfloat16* const* pages,
    const float* k, const float* v, const float* sinks, float* partial, float* out,
    uint32_t end, const uint32_t* position, uint32_t rows, uint32_t qheads,
    uint32_t kvheads, uint32_t capacity, uint32_t splits, uint32_t layer_offset,
    uint32_t page_tokens, cudaStream_t stream) {
    if(page_tokens != 128) return cudaErrorInvalidValue;
    if(pages) {
        partial_attention<true><<<dim3(qheads,rows,splits),32,0,stream>>>(q,ck,cv,k,v,partial,end,rows,qheads,kvheads,capacity,splits,pages,layer_offset,page_tokens,position);
        merge_attention<<<qheads*rows,32,0,stream>>>(partial,sinks,out,splits,qheads);
        append_kv<true><<<(size_t(rows)*kvheads*128+255)/256,256,0,stream>>>(k,v,ck,cv,end,rows,kvheads*128,capacity,pages,layer_offset,page_tokens,position);
    } else {
        partial_attention<false><<<dim3(qheads,rows,splits),32,0,stream>>>(q,ck,cv,k,v,partial,end,rows,qheads,kvheads,capacity,splits,nullptr,0,0,position);
        merge_attention<<<qheads*rows,32,0,stream>>>(partial,sinks,out,splits,qheads);
        append_kv<false><<<(size_t(rows)*kvheads*128+255)/256,256,0,stream>>>(k,v,ck,cv,end,rows,kvheads*128,capacity,nullptr,0,0,position);
    }
    return cudaGetLastError();
}

// One token per independent sequence. All pointer tables belong to the workspace,
// so graph replay can change the active requests without capturing their addresses.
extern "C" cudaError_t infer_paged_sink_decode_batch(
    const float* q, __nv_bfloat16* const* keys, __nv_bfloat16* const* values,
    __nv_bfloat16* const* const* pages, const uint32_t* positions,
    const float* k, const float* v, const float* sinks, float* partial, float* out,
    uint32_t rows, uint32_t qheads, uint32_t kvheads, uint32_t capacity,
    uint32_t splits, uint32_t layer_offset, uint32_t layers, uint32_t layer,
    bool local, cudaStream_t stream) {
    if (!q || !keys || !values || !pages || !positions || !k || !v || !sinks
        || !partial || !out || !rows || !qheads || !kvheads || !capacity || !splits
        || layer >= layers) return cudaErrorInvalidValue;
    const dim3 grid(qheads, rows, splits);
    const uint32_t blocks=(size_t(rows)*kvheads*128+255)/256;
    if (local) {
        partial_attention<false,true><<<grid,32,0,stream>>>(q,nullptr,nullptr,k,v,partial,0,
            rows,qheads,kvheads,capacity,splits,nullptr,layer_offset,128,positions,
            keys,values,pages,layers,layer);
        merge_attention<<<qheads*rows,32,0,stream>>>(partial,sinks,out,splits,qheads);
        append_kv<false,true><<<blocks,256,0,stream>>>(k,v,nullptr,nullptr,0,rows,kvheads*128,
            capacity,nullptr,layer_offset,128,positions,keys,values,pages,layers,layer);
    } else {
        partial_attention<true,true><<<grid,32,0,stream>>>(q,nullptr,nullptr,k,v,partial,0,
            rows,qheads,kvheads,capacity,splits,nullptr,layer_offset,128,positions,
            keys,values,pages,layers,layer);
        merge_attention<<<qheads*rows,32,0,stream>>>(partial,sinks,out,splits,qheads);
        append_kv<true,true><<<blocks,256,0,stream>>>(k,v,nullptr,nullptr,0,rows,kvheads*128,
            capacity,nullptr,layer_offset,128,positions,keys,values,pages,layers,layer);
    }
    return cudaGetLastError();
}
