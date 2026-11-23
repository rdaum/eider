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
    uint32_t layers=0, uint32_t layer=0,
    const float* relative=nullptr, uint32_t rel_extent=0, uint32_t query_offset=0) {
    const uint32_t head=blockIdx.x, query=blockIdx.y, part=blockIdx.z, lane=threadIdx.x;
    if (position) end=position[DecodeBatch ? query : 0];
    if constexpr (DecodeBatch) {
        if constexpr (Paged) pages=batch_pages[query];
        else { ck=batch_keys[query*layers+layer]; cv=batch_values[query*layers+layer]; }
    }
    const uint32_t kvhead=head/(qheads/kvheads);
    // Chunk-local query rows may start after earlier rows of the same chunk.
    const uint64_t last=uint64_t(end)+(DecodeBatch ? 0 : query_offset+query)+1, first=last>capacity ? last-capacity : 0;
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
        if(relative) {
            const uint64_t distance=uint64_t(end)+query_offset+query-i;
            if(distance<rel_extent)
                dot+=relative[(size_t(query)*qheads+head)*rel_extent+distance];
        }
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

// Reuse each KV tile across 16 prompt queries. Every query retains the decode
// kernel's 256-key partitions, lane dot products, and sequential online softmax.
template <bool Paged>
__global__ void prefill_attention(
    const float* q, const __nv_bfloat16* ck, const __nv_bfloat16* cv,
    const float* k, const float* v, float* partial, uint32_t end,
    uint32_t rows, uint32_t qheads, uint32_t kvheads, uint32_t capacity,
    uint32_t splits, const __nv_bfloat16* const* pages, uint32_t layer_offset,
    uint32_t page_tokens, const uint32_t* position,
    const float* relative=nullptr, uint32_t rel_extent=0, uint32_t query_offset=0) {
    constexpr unsigned Warps=8, Queries=2, QueryRows=Warps*Queries, Keys=32;
    __shared__ float keys[Keys][128], values[Keys][128];
    const unsigned lane=threadIdx.x%32, warp=threadIdx.x/32;
    const unsigned head=blockIdx.x, first_query=blockIdx.y*QueryRows, part=blockIdx.z;
    const unsigned kvhead=head/(qheads/kvheads);
    if (position) end=position[0];
    float query_value[Queries][4], acc[Queries][4]={};
    float mx[Queries], total[Queries]={};
    uint64_t begin[Queries], limit[Queries];
    #pragma unroll
    for(unsigned n=0;n<Queries;++n) {
        const unsigned query=first_query+warp+n*Warps;
        const uint64_t last=uint64_t(end)+query_offset+query+1;
        const uint64_t first=last>capacity ? last-capacity : 0;
        begin[n]=first+part*256;
        limit[n]=min(last,first+(part+1)*256);
        mx[n]=-INFINITY;
        #pragma unroll
        for(unsigned c=0;c<4;++c)
            query_value[n][c]=query<rows ? q[(size_t(query)*qheads+head)*128+lane+c*32] : 0.f;
    }
    const uint64_t first_last=uint64_t(end)+query_offset+first_query+1;
    const uint64_t first=first_last>capacity ? first_last-capacity : 0;
    const uint64_t last=uint64_t(end)+query_offset+min(first_query+QueryRows,rows);
    const uint64_t last_first=last>capacity ? last-capacity : 0;
    const uint64_t stop=min(last,last_first+(part+1)*256);
    for(uint64_t tile=first+part*256;tile<stop;tile+=Keys) {
        for(unsigned j=threadIdx.x;j<Keys*128;j+=blockDim.x) {
            const uint64_t token=tile+j/128;
            const unsigned col=j%128;
            float key=0.f, value=0.f;
            if(token<stop) {
                const bool cached=token<end;
                const size_t row=cached ? (Paged ? (token&127) : token%capacity) : token-end;
                const size_t base=(row*kvheads+kvhead)*128+col;
                if(cached) {
                    const __nv_bfloat16* pk=Paged ? pages[token>>7]+layer_offset : ck;
                    const __nv_bfloat16* pv=Paged ? pk+size_t(page_tokens)*kvheads*128 : cv;
                    key=__bfloat162float(pk[base]); value=__bfloat162float(pv[base]);
                } else { key=k[base]; value=v[base]; }
            }
            keys[j/128][col]=key; values[j/128][col]=value;
        }
        __syncthreads();
        for(unsigned t=0;t<Keys && tile+t<stop;++t) {
            float key[4], value[4];
            #pragma unroll
            for(unsigned c=0;c<4;++c) {
                key[c]=keys[t][lane+c*32]; value[c]=values[t][lane+c*32];
            }
            #pragma unroll
            for(unsigned n=0;n<Queries;++n) {
                const unsigned query=first_query+warp+n*Warps;
                if(query<rows && tile+t>=begin[n] && tile+t<limit[n]) {
                    float dot=0.f;
                    #pragma unroll
                    for(unsigned c=0;c<4;++c) dot=fmaf(query_value[n][c],key[c],dot);
                    dot=warp_sum(dot)*0.08838834764831845f;
                    if(relative) {
                        const uint64_t distance=uint64_t(end)+query_offset+query-(tile+t);
                        if(distance<rel_extent)
                            dot+=relative[(size_t(query)*qheads+head)*rel_extent+distance];
                    }
                    const float next=fmaxf(mx[n],dot), keep=expf(mx[n]-next), p=expf(dot-next);
                    total[n]=total[n]*keep+p;
                    #pragma unroll
                    for(unsigned c=0;c<4;++c) acc[n][c]=acc[n][c]*keep+p*value[c];
                    mx[n]=next;
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(unsigned n=0;n<Queries;++n) {
        const unsigned query=first_query+warp+n*Warps;
        if(query<rows) {
            float* dst=partial+((size_t(query)*qheads+head)*splits+part)*130;
            #pragma unroll
            for(unsigned c=0;c<4;++c) dst[lane+c*32]=acc[n][c];
            if(lane==0) { dst[128]=mx[n]; dst[129]=total[n]; }
        }
    }
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

// Inkling's sink does not add a zero-value KV entry. It gates ordinary
// attention using the real-key log-sum-exp and a projected per-head sink.
__global__ void merge_relative_attention(const float* partial, const float* gate,
    const float* sinks, float* out, uint32_t splits, uint32_t qheads) {
    const uint32_t lane=threadIdx.x,row=blockIdx.x,head=row%qheads;
    const float* src=partial+size_t(row)*splits*130;
    float mx=-INFINITY,total=0.f,acc[4]={};
    for(uint32_t s=0;s<splits;++s) mx=fmaxf(mx,src[s*130+128]);
    for(uint32_t s=0;s<splits;++s) {
        const float p=expf(src[s*130+128]-mx);
        total+=src[s*130+129]*p;
        #pragma unroll
        for(int c=0;c<4;++c) acc[c]+=src[s*130+lane+c*32]*p;
    }
    const float lse=mx+logf(total);
    const float sink=sinks[row];
    #pragma unroll
    for(int c=0;c<4;++c) {
        const uint32_t column=lane+c*32;
        const float gate_value=gate[row*128+column];
        const float factor=1.f/(1.f+expf(-gate_value))/(1.f+expf(-(lse-sink)));
        out[row*128+column]=__bfloat162float(__float2bfloat16_rn(acc[c]/total*factor));
    }
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

namespace flash {
constexpr unsigned Rows=64, Keys=64, Dim=128, KeyStride=Dim+8, ValueStride=Keys+8;
constexpr float Scale=0.08838834764831845f, Log2e=1.4426950408889634f, Ln2=0.6931471805599453f;

__device__ __forceinline__ uint32_t pack(float low, float high) {
    const __nv_bfloat162 pair=__floats2bfloat162_rn(low,high);
    return *reinterpret_cast<const uint32_t*>(&pair);
}

__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float quad_max(float x) {
    x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,1));
    return fmaxf(x,__shfl_xor_sync(0xffffffff,x,2));
}

__device__ __forceinline__ float quad_sum(float x) {
    x+=__shfl_xor_sync(0xffffffff,x,1);
    return x+__shfl_xor_sync(0xffffffff,x,2);
}
}

// Tensor-core prompt attention for Inkling's relative-bias, LSE-sink-gated
// heads. Each block owns 64 queries of one head and streams 64-key BF16
// tiles from the cache (`token < end`) or the chunk's own K/V. Scores and
// softmax statistics stay in F32; probabilities enter the PV product as BF16.
// The sink gate is applied in the epilogue, so no partials are written.
template <bool Paged>
__global__ void __launch_bounds__(128) relative_flash_attention(
    const float* q, const __nv_bfloat16* ck, const __nv_bfloat16* cv,
    const __nv_bfloat16* const* pages, uint32_t layer_offset, const float* k, const float* v,
    const float* relative, const float* gate, const float* sinks, float* out, uint32_t end,
    const uint32_t* position, uint32_t rows, uint32_t qheads, uint32_t kvheads,
    uint32_t capacity, uint32_t rel_extent) {
    using namespace flash;
    __shared__ __align__(16) __nv_bfloat16 keys[Keys*KeyStride];
    // Values are stored transposed, [dimension][key], for the PV B operand.
    __shared__ __align__(16) __nv_bfloat16 values[Dim*ValueStride];
    const unsigned lane=threadIdx.x%32, warp=threadIdx.x/32, g=lane/4, t=lane%4;
    const unsigned head=blockIdx.x, kvhead=head/(qheads/kvheads);
    const unsigned block_first=blockIdx.y*Rows;
    if(position) end=position[0];
    const unsigned row_of[2]={block_first+warp*16+g, block_first+warp*16+g+8};
    const size_t width=size_t(kvheads)*Dim;

    uint32_t query[8][4];
    #pragma unroll
    for(unsigned kk=0;kk<8;++kk) {
        #pragma unroll
        for(unsigned half=0;half<2;++half) {
            #pragma unroll
            for(unsigned r=0;r<2;++r) {
                const unsigned row=row_of[r], col=kk*16+half*8+2*t;
                float2 x=make_float2(0.f,0.f);
                if(row<rows) x=*reinterpret_cast<const float2*>(q+(size_t(row)*qheads+head)*Dim+col);
                query[kk][half*2+r]=pack(x.x,x.y);
            }
        }
    }

    const unsigned last_row=min(block_first+Rows,rows)-1;
    const uint64_t first_position=uint64_t(end)+block_first;
    const uint64_t stop=uint64_t(end)+last_row+1;
    const uint64_t start=first_position+1>capacity ? first_position+1-capacity : 0;
    float acc[16][4]={}, mx[2]={-INFINITY,-INFINITY}, total[2]={};
    for(uint64_t tile=start;tile<stop;tile+=Keys) {
        for(unsigned item=threadIdx.x;item<Keys*Dim/8;item+=blockDim.x) {
            const unsigned key=item/(Dim/8), col=item%(Dim/8)*8;
            const uint64_t token=tile+key;
            uint4 kx=make_uint4(0,0,0,0), vx=kx;
            if(token<stop) {
                if(token<end) {
                    const size_t row=Paged ? (token&127) : token%capacity;
                    const __nv_bfloat16* pk=Paged ? pages[token>>7]+layer_offset : ck;
                    const __nv_bfloat16* pv=Paged ? pk+128*width : cv;
                    kx=*reinterpret_cast<const uint4*>(pk+row*width+kvhead*Dim+col);
                    vx=*reinterpret_cast<const uint4*>(pv+row*width+kvhead*Dim+col);
                } else {
                    const size_t base=(token-end)*width+kvhead*Dim+col;
                    const float4 k0=*reinterpret_cast<const float4*>(k+base);
                    const float4 k1=*reinterpret_cast<const float4*>(k+base+4);
                    const float4 v0=*reinterpret_cast<const float4*>(v+base);
                    const float4 v1=*reinterpret_cast<const float4*>(v+base+4);
                    kx=make_uint4(pack(k0.x,k0.y),pack(k0.z,k0.w),pack(k1.x,k1.y),pack(k1.z,k1.w));
                    vx=make_uint4(pack(v0.x,v0.y),pack(v0.z,v0.w),pack(v1.x,v1.y),pack(v1.z,v1.w));
                }
            }
            *reinterpret_cast<uint4*>(keys+key*KeyStride+col)=kx;
            const __nv_bfloat16* vs=reinterpret_cast<const __nv_bfloat16*>(&vx);
            #pragma unroll
            for(unsigned i=0;i<8;++i) values[(col+i)*ValueStride+key]=vs[i];
        }
        __syncthreads();

        float score[8][4]={};
        #pragma unroll
        for(unsigned kk=0;kk<8;++kk) {
            #pragma unroll
            for(unsigned n=0;n<8;++n) {
                const __nv_bfloat16* base=keys+(n*8+g)*KeyStride+kk*16+2*t;
                mma(score[n],query[kk],*reinterpret_cast<const uint32_t*>(base),
                    *reinterpret_cast<const uint32_t*>(base+8));
            }
        }
        float tile_max[2]={-INFINITY,-INFINITY};
        #pragma unroll
        for(unsigned n=0;n<8;++n) {
            #pragma unroll
            for(unsigned e=0;e<4;++e) {
                const unsigned row=row_of[e/2];
                const uint64_t token=tile+n*8+2*t+e%2, query_position=uint64_t(end)+row;
                float x=-INFINITY;
                if(row<rows && token<=query_position && query_position-token<capacity) {
                    x=score[n][e]*Scale;
                    const uint64_t distance=query_position-token;
                    if(distance<rel_extent)
                        x+=relative[(size_t(row)*qheads+head)*rel_extent+distance];
                    x*=Log2e;
                }
                score[n][e]=x;
                tile_max[e/2]=fmaxf(tile_max[e/2],x);
            }
        }
        float keep[2], base_max[2];
        #pragma unroll
        for(unsigned r=0;r<2;++r) {
            const float next=fmaxf(mx[r],quad_max(tile_max[r]));
            base_max[r]=next==-INFINITY ? 0.f : next;
            keep[r]=exp2f(mx[r]-base_max[r]);
            mx[r]=next;
            total[r]*=keep[r];
        }
        #pragma unroll
        for(unsigned n=0;n<8;++n) {
            #pragma unroll
            for(unsigned e=0;e<4;++e) {
                score[n][e]=exp2f(score[n][e]-base_max[e/2]);
                total[e/2]+=score[n][e];
            }
        }
        #pragma unroll
        for(unsigned n=0;n<16;++n) {
            #pragma unroll
            for(unsigned e=0;e<4;++e) acc[n][e]*=keep[e/2];
        }
        #pragma unroll
        for(unsigned kk=0;kk<4;++kk) {
            const uint32_t p[4]={pack(score[2*kk][0],score[2*kk][1]),
                pack(score[2*kk][2],score[2*kk][3]),pack(score[2*kk+1][0],score[2*kk+1][1]),
                pack(score[2*kk+1][2],score[2*kk+1][3])};
            #pragma unroll
            for(unsigned n=0;n<16;++n) {
                const __nv_bfloat16* base=values+(n*8+g)*ValueStride+kk*16+2*t;
                mma(acc[n],p,*reinterpret_cast<const uint32_t*>(base),
                    *reinterpret_cast<const uint32_t*>(base+8));
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(unsigned r=0;r<2;++r) {
        const unsigned row=row_of[r];
        const float sum=quad_sum(total[r]);
        if(row>=rows) continue;
        const size_t index=size_t(row)*qheads+head;
        const float lse=(mx[r]+log2f(sum))*Ln2;
        const float sink=1.f/(1.f+expf(-(lse-sinks[index])));
        #pragma unroll
        for(unsigned n=0;n<16;++n) {
            #pragma unroll
            for(unsigned e=0;e<2;++e) {
                const unsigned col=n*8+2*t+e;
                const float gated=acc[n][r*2+e]/sum*sink/(1.f+expf(-gate[index*Dim+col]));
                out[index*Dim+col]=__bfloat162float(__float2bfloat16_rn(gated));
            }
        }
    }
}

template <bool Paged>
cudaError_t relative_attention(
    const float* q, __nv_bfloat16* ck, __nv_bfloat16* cv, __nv_bfloat16* const* pages,
    uint32_t layer_offset, const float* k, const float* v, const float* relative,
    const float* gate, const float* sinks, float* partial, float* out, uint32_t end,
    const uint32_t* position, uint32_t end_bound, uint32_t rows, uint32_t qheads,
    uint32_t kvheads, uint32_t capacity, uint32_t rel_extent, uint64_t partial_capacity,
    cudaStream_t stream) {
    constexpr uint32_t PageTokens=128;
    // A device position replaces `end` at execution time; `end_bound` then
    // bounds the partitions so a captured launch stays valid as it grows.
    const uint64_t extent=position ? uint64_t(end_bound) : uint64_t(end)+rows;
    if(rows>=flash::Rows) {
        relative_flash_attention<Paged><<<dim3(qheads,(rows+flash::Rows-1)/flash::Rows),128,0,stream>>>(
            q,ck,cv,pages,layer_offset,k,v,relative,gate,sinks,out,end,position,rows,qheads,
            kvheads,capacity,rel_extent);
        append_kv<Paged><<<(size_t(rows)*kvheads*128+255)/256,256,0,stream>>>(
            k,v,ck,cv,end,rows,kvheads*128,capacity,pages,layer_offset,PageTokens,position);
        return cudaGetLastError();
    }
    for(uint32_t first=0;first<rows;) {
        const uint32_t splits=(min(extent,uint64_t(capacity))+255)/256;
        const uint64_t per_row=uint64_t(qheads)*splits*130;
        const uint32_t count=uint32_t(min(uint64_t(rows-first),partial_capacity/per_row));
        if(count==0) return cudaErrorInvalidValue;
        const size_t qoff=size_t(first)*qheads;
        if(count>=64)
            prefill_attention<Paged><<<dim3(qheads,(count+15)/16,splits),256,0,stream>>>(
                q+qoff*128,ck,cv,k,v,partial,end,count,qheads,kvheads,capacity,splits,pages,
                layer_offset,PageTokens,position,relative+qoff*rel_extent,rel_extent,first);
        else
            partial_attention<Paged><<<dim3(qheads,count,splits),32,0,stream>>>(
                q+qoff*128,ck,cv,k,v,partial,end,count,qheads,kvheads,capacity,splits,pages,
                layer_offset,PageTokens,position,nullptr,nullptr,nullptr,0,0,
                relative+qoff*rel_extent,rel_extent,first);
        merge_relative_attention<<<qheads*count,32,0,stream>>>(
            partial,gate+qoff*128,sinks+qoff,out+qoff*128,splits,qheads);
        first+=count;
    }
    append_kv<Paged><<<(size_t(rows)*kvheads*128+255)/256,256,0,stream>>>(
        k,v,ck,cv,end,rows,kvheads*128,capacity,pages,layer_offset,PageTokens,position);
    return cudaGetLastError();
}

// Rows are evaluated in sub-chunks whose softmax partials fit `partial_capacity`.
// Every sub-chunk reads earlier rows of the same chunk from `k`/`v`; the
// chunk is appended to the cache once all rows have attended. With `pages`,
// K/V live in 128-token page bundles at `layer_offset` and the window is
// unbounded; otherwise they live in a `capacity`-position local ring.
extern "C" cudaError_t infer_causal_relative_attention(
    const float* q, __nv_bfloat16* ck, __nv_bfloat16* cv, __nv_bfloat16* const* pages,
    uint32_t layer_offset, const float* k, const float* v, const float* relative,
    const float* gate, const float* sinks, float* partial, float* out, uint32_t end,
    const uint32_t* position, uint32_t end_bound, uint32_t rows, uint32_t qheads,
    uint32_t kvheads, uint32_t capacity, uint32_t rel_extent, uint64_t partial_capacity,
    cudaStream_t stream) {
    if(!q || !k || !v || !relative || !gate || !sinks || !partial || !out || !rel_extent || !rows
        || (pages ? (ck || cv) : (!ck || !cv))) return cudaErrorInvalidValue;
    if(pages)
        return relative_attention<true>(q,nullptr,nullptr,pages,layer_offset,k,v,relative,gate,
            sinks,partial,out,end,position,end_bound,rows,qheads,kvheads,UINT32_MAX,rel_extent,
            partial_capacity,stream);
    return relative_attention<false>(q,ck,cv,nullptr,0,k,v,relative,gate,sinks,partial,out,end,
        position,end_bound,rows,qheads,kvheads,capacity,rel_extent,partial_capacity,stream);
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
        if(rows>=64)
            prefill_attention<true><<<dim3(qheads,(rows+15)/16,splits),256,0,stream>>>(q,ck,cv,k,v,partial,end,rows,qheads,kvheads,capacity,splits,pages,layer_offset,page_tokens,position);
        else
            partial_attention<true><<<dim3(qheads,rows,splits),32,0,stream>>>(q,ck,cv,k,v,partial,end,rows,qheads,kvheads,capacity,splits,pages,layer_offset,page_tokens,position);
        merge_attention<<<qheads*rows,32,0,stream>>>(partial,sinks,out,splits,qheads);
        append_kv<true><<<(size_t(rows)*kvheads*128+255)/256,256,0,stream>>>(k,v,ck,cv,end,rows,kvheads*128,capacity,pages,layer_offset,page_tokens,position);
    } else {
        if(rows>=64)
            prefill_attention<false><<<dim3(qheads,(rows+15)/16,splits),256,0,stream>>>(q,ck,cv,k,v,partial,end,rows,qheads,kvheads,capacity,splits,nullptr,0,0,position);
        else
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
