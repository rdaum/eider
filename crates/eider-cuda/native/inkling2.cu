// Inkling 2 text operators. F32 storage carries explicit BF16 rounding points.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace {
__device__ __forceinline__ float bf(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}
__device__ __forceinline__ float bf16(uint16_t x) {
    return __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&x));
}
__device__ float block_sum(float x, float* partial) {
    const int lane=threadIdx.x;
    partial[lane]=x;
    __syncthreads();
    for(int stride=blockDim.x/2;stride;stride>>=1) {
        if(lane<stride) partial[lane]+=partial[lane+stride];
        __syncthreads();
    }
    const float sum=partial[0];
    // Back-to-back reductions reuse `partial`; every lane must read first.
    __syncthreads();
    return sum;
}

__global__ void embed_kernel(const uint16_t* embedding, const uint32_t* tokens,
                             const float* gamma, float* out, uint32_t first,
                             uint32_t width, float eps) {
    __shared__ float partial[256];
    const int lane=threadIdx.x;
    const uint32_t token=tokens[first+blockIdx.x];
    const size_t offset=size_t(token)*width;
    const size_t base=size_t(blockIdx.x)*width;
    float sum=0;
    for(int i=lane;i<width;i+=blockDim.x) {
        const float x=bf16(embedding[offset+i]);
        sum+=x*x;
    }
    const float inv=rsqrtf(block_sum(sum,partial)/width+eps);
    for(int i=lane;i<width;i+=blockDim.x)
        out[base+i]=bf(gamma[i]*bf(bf16(embedding[offset+i])*inv));
}

__global__ void rms_kernel(const float* x, const float* gamma, float* out,
                           uint32_t width, float eps) {
    __shared__ float partial[256];
    const int lane=threadIdx.x;
    const size_t offset=size_t(blockIdx.x)*width;
    float sum=0;
    for(int i=lane;i<width;i+=blockDim.x) sum+=x[offset+i]*x[offset+i];
    const float inv=rsqrtf(block_sum(sum,partial)/width+eps);
    for(int i=lane;i<width;i+=blockDim.x)
        out[offset+i]=gamma ? bf(gamma[i]*bf(x[offset+i]*inv)) : bf(x[offset+i]*inv);
}

// One normalized pseudo-query over at most five previous block outputs.
// A non-null addend is added to the last (partial) source independently.
__global__ void attn_res_kernel(const float* const* unused,
    const float* s0, const float* s1, const float* s2, const float* s3, const float* s4,
    const float* addend, const float* query, float* out, uint32_t sources,
    uint32_t width, float eps) {
    __shared__ float partial[256], dots[5], norms[5], weights[5];
    const int lane=threadIdx.x;
    const size_t row=size_t(blockIdx.x)*width;
    const float* src[5]={s0,s1,s2,s3,s4};
    const float scale=rsqrtf(float(width));
    for(uint32_t n=0;n<sources;++n) {
        float dot=0, sum=0;
        for(int i=lane;i<width;i+=blockDim.x) {
            float x=src[n][row+i];
            if(addend && n+1==sources) x=__fadd_rn(x,addend[row+i]);
            dot=__fmaf_rn(x,query[i]*scale,dot);
            sum=__fmaf_rn(x,x,sum);
        }
        dot=block_sum(dot,partial);
        if(lane==0) dots[n]=dot;
        sum=block_sum(sum,partial);
        if(lane==0) norms[n]=rsqrtf(sum/width+eps);
        __syncthreads();
    }
    if(lane==0) {
        float maximum=-INFINITY;
        for(uint32_t n=0;n<sources;++n) maximum=fmaxf(maximum,dots[n]*norms[n]);
        float total=0;
        for(uint32_t n=0;n<sources;++n) {
            weights[n]=expf(dots[n]*norms[n]-maximum);
            total+=weights[n];
        }
        for(uint32_t n=0;n<sources;++n) weights[n]/=total;
    }
    __syncthreads();
    for(int i=lane;i<width;i+=blockDim.x) {
        float value=0;
        for(uint32_t n=0;n<sources;++n) {
            float x=src[n][row+i];
            if(addend && n+1==sources) x=__fadd_rn(x,addend[row+i]);
            value=__fmaf_rn(weights[n]*norms[n],x,value);
        }
        out[row+i]=value;
    }
}

__global__ void controller_norm_kernel(const float* destination, const float* message,
                                       float* normed, uint32_t width, float eps) {
    __shared__ float partial[256];
    const int lane=threadIdx.x;
    const size_t row=size_t(blockIdx.x)*width;
    float a=0,b=0;
    for(int i=lane;i<width;i+=blockDim.x) {
        a+=destination[row+i]*destination[row+i];
        b+=message[row+i]*message[row+i];
    }
    const float na=rsqrtf(block_sum(a,partial)/width+eps);
    const float nb=rsqrtf(block_sum(b,partial)/width+eps);
    for(int i=lane;i<width;i+=blockDim.x) {
        normed[row*2+i]=bf(destination[row+i]*na);
        normed[row*2+width+i]=bf(message[row+i]*nb);
    }
}

__global__ void controller_mix_kernel(const float* destination, const float* message,
    const float* gain, const float* forget, float* out, uint32_t count) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) {
        const float g=tanhf(gain[i]), f=tanhf(forget[i]);
        out[i]=__fadd_rn(__fmul_rn(1.f-f,destination[i]),__fmul_rn(g,message[i]));
    }
}

// Each state contains the last width-1 raw BF16 inputs, not convolution outputs.
// Static weights use [channel][tap]; global layers add a dynamic per-head kernel.
__global__ void sconv_kernel(const float* x, const float* weight, const float* dynamic,
                             const float* state, float* out, uint32_t hidden,
                             uint32_t width, uint32_t nheads) {
    const uint32_t c=blockIdx.x*blockDim.x+threadIdx.x, row=blockIdx.y;
    if(c>=hidden) return;
    float sum=0;
    for(uint32_t tap=0;tap<width;++tap) {
        const int64_t source=int64_t(row)+tap-(width-1);
        const float xval=source<0 ? state[size_t(c)*(width-1)+row+tap]
                                  : bf(x[size_t(source)*hidden+c]);
        const uint32_t head=c/(hidden/nheads);
        // Dynamic taps put the current input at index zero; static taps use history order.
        const uint32_t weight_tap=dynamic ? width-1-tap : tap;
        const float w=weight[size_t(c)*width+weight_tap]
            +(dynamic ? dynamic[size_t(row)*nheads*width+head*width+weight_tap] : 0.f);
        sum=__fmaf_rn(xval,w,sum);
    }
    const float scale=rsqrtf(float(width)*(dynamic ? 2 : 1)+1);
    out[size_t(row)*hidden+c]=bf((x[size_t(row)*hidden+c]+sum)*scale);
}
__global__ void sconv_commit_kernel(const float* x, float* state, uint32_t hidden,
                                    uint32_t rows, uint32_t width) {
    const uint32_t c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=hidden) return;
    for(uint32_t tap=0;tap+1<width;++tap) {
        const int64_t source=int64_t(rows)+tap-(width-1);
        state[size_t(c)*(width-1)+tap]=source<0
            ? state[size_t(c)*(width-1)+rows+tap]
            : bf(x[size_t(source)*hidden+c]);
    }
}

__global__ void l2_clip_kernel(const float* x, const float* scale, float* out,
    uint32_t heads, uint32_t dim, float threshold, float eps,
    float position, uint32_t tau_floor, float tau_alpha) {
    __shared__ float partial[128];
    const int lane=threadIdx.x;
    const uint32_t head=blockIdx.x, row=blockIdx.y;
    const size_t offset=(size_t(row)*heads+head)*dim+lane;
    const size_t scale_offset=size_t(head)*dim+lane;
    const float value=lane<dim ? x[offset]*(scale ? scale[scale_offset] : 1.f) : 0.f;
    partial[lane]=value*value;
    __syncthreads();
    for(int stride=64;stride;stride>>=1) {
        if(lane<stride) partial[lane]+=partial[lane+stride];
        __syncthreads();
    }
    const float sum=partial[0];
    const float norm=sum>0 ? sum*rsqrtf(sum) : 0.f;
    const float clipped=bf(value*fminf(1.f,threshold/(norm+eps)));
    float tau=1.f;
    if(tau_floor) tau=1.f+tau_alpha*logf(fmaxf((position+row+1.f)/tau_floor,1.f));
    if(lane<dim) out[offset]=bf(clipped*tau);
}

__global__ void rel_projection_kernel(const float* r, const uint16_t* projection,
                                      float* out, uint32_t heads, uint32_t extent) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    const uint32_t head=i/extent, distance=i%extent;
    if(head>=heads) return;
    float sum=0;
    // `head` spans rows-by-heads; R rows are contiguous in the same order.
    for(uint32_t d=0;d<16;++d)
        sum=__fmaf_rn(r[size_t(head)*16+d],bf16(projection[d*extent+distance]),sum);
    out[i]=bf(sum);
}

// One warp per row. Selection keeps the highest biased sigmoid score and,
// for equal scores, the lowest expert id.
__global__ void router_kernel(const float* logits, const float* bias, uint32_t* ids,
                              float* weights, uint32_t experts, float route_scale) {
    constexpr unsigned MaxPerLane=4;
    const unsigned lane=threadIdx.x;
    logits+=size_t(blockIdx.x)*experts;
    ids+=size_t(blockIdx.x)*8;
    weights+=size_t(blockIdx.x)*8;
    float score[MaxPerLane];
    #pragma unroll
    for(unsigned j=0;j<MaxPerLane;++j) {
        const unsigned e=lane+32*j;
        score[j]=e<experts ? 1.f/(1.f+expf(-logits[e]))+(bias ? bias[e] : 0.f) : -INFINITY;
    }
    float selected_logits[8];
    for(unsigned k=0;k<8;++k) {
        float best=-INFINITY;
        unsigned id=0xffffffffu;
        #pragma unroll
        for(unsigned j=0;j<MaxPerLane;++j)
            if(score[j]>best) { best=score[j]; id=lane+32*j; }
        for(int offset=16;offset;offset>>=1) {
            const float other=__shfl_xor_sync(0xffffffffu,best,offset);
            const unsigned other_id=__shfl_xor_sync(0xffffffffu,id,offset);
            if(other>best || (other==best && other_id<id)) { best=other; id=other_id; }
        }
        if(id%32==lane) score[id/32]=-INFINITY;
        if(lane==0) ids[k]=id;
        selected_logits[k]=logits[id];
    }
    if(lane!=0) return;
    float maximum=-INFINITY;
    for(int k=0;k<8;++k) {
        const float z=selected_logits[k];
        selected_logits[k]=fminf(z,0.f)-log1pf(expf(-fabsf(z)));
        maximum=fmaxf(maximum,selected_logits[k]);
    }
    float total=0;
    for(int k=0;k<8;++k) total+=expf(selected_logits[k]-maximum);
    for(int k=0;k<8;++k)
        weights[k]=route_scale*expf(selected_logits[k]-maximum)/total;
}

__global__ void capped_swiglu_kernel(const float* gate_up, const float* gammas,
    float* out, uint32_t routes, uint32_t intermediate,
    float gate_cap, float up_cap) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=routes*intermediate) return;
    const uint32_t route=i/intermediate;
    const float raw_gate=gate_up[i*2], raw_up=gate_up[i*2+1];
    const float gate=gate_cap*tanhf(raw_gate/gate_cap);
    const float up=up_cap*tanhf(raw_up/up_cap);
    out[i]=bf(gate*(1.f/(1.f+expf(-raw_gate)))*up*gammas[route]);
}

__device__ __forceinline__ uint32_t pack_bf16(float low, float high) {
    return uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(low)))
        | (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(high))) << 16);
}

// BF16 weights in [expert][out][in] order. Decode-sized batches: one CTA
// computes a 16-row output tile for one work item, splitting K across warps.
// Items with more than `Groups * 8` routes repeat the K loop.
template<unsigned Warps, unsigned Groups>
__global__ __launch_bounds__(Warps*32) void grouped_bf16_expert_kernel(
    const uint32_t* sorted, const uint32_t* item_expert, const uint32_t* item_start,
    const uint32_t* item_count, const uint32_t* items, const float* input,
    const uint16_t* weights, float* out, uint32_t top_k, bool per_route_input,
    uint32_t outputs, uint32_t inputs) {
    const unsigned item=blockIdx.y;
    if(item>=items[0]) return;
    const unsigned lane=threadIdx.x&31, warp=threadIdx.x>>5;
    const unsigned row0=lane>>2, row1=row0+8, pair=(lane&3)*2;
    const unsigned start=item_start[item], count=item_count[item];
    const uint16_t* w=weights+(size_t(item_expert[item])*outputs+blockIdx.x*16)*inputs;
    __shared__ float partial[Warps][Groups][128];
    for(unsigned first=0;first<count;first+=Groups*8) {
        const float* rows[Groups];
        #pragma unroll
        for(unsigned g=0;g<Groups;++g) {
            const unsigned column=first+g*8+row0;
            rows[g]=nullptr;
            if(column<count) {
                const unsigned route=sorted[start+column];
                rows[g]=input+size_t(per_route_input ? route : route/top_k)*inputs;
            }
        }
        float d[Groups][4]={};
        for(unsigned k=warp*16;k<inputs;k+=Warps*16) {
            const uint32_t a0=*reinterpret_cast<const uint32_t*>(w+size_t(row0)*inputs+k+pair);
            const uint32_t a1=*reinterpret_cast<const uint32_t*>(w+size_t(row1)*inputs+k+pair);
            const uint32_t a2=*reinterpret_cast<const uint32_t*>(w+size_t(row0)*inputs+k+pair+8);
            const uint32_t a3=*reinterpret_cast<const uint32_t*>(w+size_t(row1)*inputs+k+pair+8);
            #pragma unroll
            for(unsigned g=0;g<Groups;++g) {
                if(first+g*8>=count) break;
                uint32_t b0=0,b1=0;
                if(rows[g]) {
                    const float* x=rows[g]+k+pair;
                    b0=pack_bf16(x[0],x[1]);
                    b1=pack_bf16(x[8],x[9]);
                }
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                    :"+f"(d[g][0]),"+f"(d[g][1]),"+f"(d[g][2]),"+f"(d[g][3])
                    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
        }
        #pragma unroll
        for(unsigned g=0;g<Groups;++g) {
            partial[warp][g][row0*8+pair]=d[g][0];
            partial[warp][g][row0*8+pair+1]=d[g][1];
            partial[warp][g][row1*8+pair]=d[g][2];
            partial[warp][g][row1*8+pair+1]=d[g][3];
        }
        __syncthreads();
        for(unsigned index=threadIdx.x;index<Groups*128;index+=blockDim.x) {
            const unsigned g=index/128, cell=index%128, column=first+g*8+cell%8;
            if(column>=count) continue;
            float sum=0.f;
            #pragma unroll
            for(unsigned i=0;i<Warps;++i) sum+=partial[i][g][cell];
            out[size_t(sorted[start+column])*outputs+blockIdx.x*16+cell/8]=bf(sum);
        }
        __syncthreads();
    }
}

// Prefill-sized batches: one CTA computes 128 output rows for one work item,
// each warp owning a 16-row tile over the full K. Route activations are staged
// once per 64-wide K chunk, so each activation is read once per CTA.
__global__ __launch_bounds__(256) void grouped_bf16_expert_tiled_kernel(
    const uint32_t* sorted, const uint32_t* item_expert, const uint32_t* item_start,
    const uint32_t* item_count, const uint32_t* items, const float* input,
    const uint16_t* weights, float* out, uint32_t top_k, bool per_route_input,
    uint32_t outputs, uint32_t inputs) {
    constexpr unsigned Warps=8, Columns=64, Groups=Columns/8, Chunk=64, Words=Chunk/2, Stride=Words+4;
    __shared__ uint32_t staged[Columns][Stride];
    __shared__ const float* rows[Columns];
    const unsigned item=blockIdx.y;
    if(item>=items[0]) return;
    const unsigned lane=threadIdx.x&31, warp=threadIdx.x>>5;
    const unsigned row0=lane>>2, row1=row0+8, quad=lane&3;
    const unsigned start=item_start[item], count=item_count[item];
    for(unsigned column=threadIdx.x;column<Columns;column+=blockDim.x) {
        rows[column]=nullptr;
        if(column<count) {
            const unsigned route=sorted[start+column];
            rows[column]=input+size_t(per_route_input ? route : route/top_k)*inputs;
        }
    }
    const unsigned tile=blockIdx.x*Warps+warp;
    const bool active=tile*16<outputs;
    const uint16_t* w=weights+(size_t(item_expert[item])*outputs+tile*16)*inputs;
    float d[Groups][4]={};
    __syncthreads();
    for(unsigned k0=0;k0<inputs;k0+=Chunk) {
        for(unsigned index=threadIdx.x;index<Columns*(Words/4);index+=blockDim.x) {
            const unsigned column=index/(Words/4), word=index%(Words/4)*4;
            uint4 packed=make_uint4(0,0,0,0);
            if(rows[column] && k0+word*2<inputs) {
                const float4 lo=*reinterpret_cast<const float4*>(rows[column]+k0+word*2);
                const float4 hi=*reinterpret_cast<const float4*>(rows[column]+k0+word*2+4);
                packed=make_uint4(pack_bf16(lo.x,lo.y),pack_bf16(lo.z,lo.w),
                                  pack_bf16(hi.x,hi.y),pack_bf16(hi.z,hi.w));
            }
            *reinterpret_cast<uint4*>(&staged[column][word])=packed;
        }
        __syncthreads();
        if(active) {
            #pragma unroll
            for(unsigned kt=0;kt<Chunk/16;++kt) {
                const unsigned k=k0+kt*16;
                if(k>=inputs) break;
                const uint32_t a0=*reinterpret_cast<const uint32_t*>(w+size_t(row0)*inputs+k+quad*2);
                const uint32_t a1=*reinterpret_cast<const uint32_t*>(w+size_t(row1)*inputs+k+quad*2);
                const uint32_t a2=*reinterpret_cast<const uint32_t*>(w+size_t(row0)*inputs+k+quad*2+8);
                const uint32_t a3=*reinterpret_cast<const uint32_t*>(w+size_t(row1)*inputs+k+quad*2+8);
                #pragma unroll
                for(unsigned g=0;g<Groups;++g) {
                    if(g*8>=count) break;
                    const uint32_t* b=staged[g*8+row0]+kt*8+quad;
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        :"+f"(d[g][0]),"+f"(d[g][1]),"+f"(d[g][2]),"+f"(d[g][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b[0]),"r"(b[4]));
                }
            }
        }
        __syncthreads();
    }
    if(!active) return;
    #pragma unroll
    for(unsigned g=0;g<Groups;++g) {
        #pragma unroll
        for(unsigned half=0;half<2;++half) {
            const unsigned column=g*8+quad*2+half;
            if(column>=count) continue;
            float* o=out+size_t(sorted[start+column])*outputs+tile*16;
            o[row0]=bf(d[g][half]);
            o[row1]=bf(d[g][2+half]);
        }
    }
}

// Decode-sized BF16 projections: each warp streams one weight row with 16-byte
// loads and accumulates it against up to MaxRows F32 input rows.
template<unsigned MaxRows>
__global__ __launch_bounds__(256) void bf16_matvec_kernel(const float* __restrict__ x,
    const uint16_t* __restrict__ weight, float* __restrict__ y, uint32_t rows,
    uint32_t outputs, uint32_t inputs) {
    const unsigned lane=threadIdx.x&31;
    const unsigned output=blockIdx.x*(blockDim.x/32)+(threadIdx.x>>5);
    if(output>=outputs) return;
    const uint16_t* w=weight+size_t(output)*inputs;
    float acc[MaxRows]={};
    for(unsigned k=lane*8;k<inputs;k+=32*8) {
        const uint4 packed=*reinterpret_cast<const uint4*>(w+k);
        const uint32_t words[4]={packed.x,packed.y,packed.z,packed.w};
        float wv[8];
        #pragma unroll
        for(unsigned i=0;i<4;++i) {
            wv[2*i]=__uint_as_float(words[i]<<16);
            wv[2*i+1]=__uint_as_float(words[i]&0xffff0000u);
        }
        #pragma unroll
        for(unsigned r=0;r<MaxRows;++r) {
            if(r>=rows) break;
            const float4 lo=*reinterpret_cast<const float4*>(x+size_t(r)*inputs+k);
            const float4 hi=*reinterpret_cast<const float4*>(x+size_t(r)*inputs+k+4);
            float sum=acc[r];
            sum=__fmaf_rn(wv[0],lo.x,sum); sum=__fmaf_rn(wv[1],lo.y,sum);
            sum=__fmaf_rn(wv[2],lo.z,sum); sum=__fmaf_rn(wv[3],lo.w,sum);
            sum=__fmaf_rn(wv[4],hi.x,sum); sum=__fmaf_rn(wv[5],hi.y,sum);
            sum=__fmaf_rn(wv[6],hi.z,sum); sum=__fmaf_rn(wv[7],hi.w,sum);
            acc[r]=sum;
        }
    }
    #pragma unroll
    for(unsigned r=0;r<MaxRows;++r) {
        if(r>=rows) break;
        float sum=acc[r];
        for(int offset=16;offset;offset>>=1) sum+=__shfl_xor_sync(0xffffffffu,sum,offset);
        if(lane==0) y[size_t(r)*outputs+output]=sum;
    }
}

__global__ void routed_sum_kernel(const float* routes, float* out, uint32_t rows,
                                  uint32_t top_k, uint32_t hidden) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=rows*hidden) return;
    const uint32_t row=i/hidden, c=i%hidden;
    float sum=0;
    for(uint32_t k=0;k<top_k;++k) sum+=routes[(size_t(row)*top_k+k)*hidden+c];
    out[i]=sum;
}

__global__ void finish_logits_kernel(float* logits, uint32_t count,
                                     uint32_t real_vocab, float softcap) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) logits[i]=i>=real_vocab ? -INFINITY : softcap*tanhf(logits[i]/softcap);
}
}

extern "C" int infer_inkling2_embed(const uint16_t* embedding, const uint32_t* tokens,
    const float* gamma, float* out, uint32_t first, uint32_t rows, uint32_t width, float eps,
    cudaStream_t stream) {
    embed_kernel<<<rows,256,0,stream>>>(embedding,tokens,gamma,out,first,width,eps);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_rms(const float* x, const float* gamma, float* out,
    uint32_t rows, uint32_t width, float eps, cudaStream_t stream) {
    rms_kernel<<<rows,256,0,stream>>>(x,gamma,out,width,eps);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_attn_res(const float* const* unused,
    const float* s0, const float* s1, const float* s2, const float* s3, const float* s4,
    const float* addend, const float* query, float* out, uint32_t rows,
    uint32_t sources, uint32_t width, float eps, cudaStream_t stream) {
    if(sources==0 || sources>5) return cudaErrorInvalidValue;
    attn_res_kernel<<<rows,256,0,stream>>>(unused,s0,s1,s2,s3,s4,addend,query,out,sources,width,eps);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_controller_norm(const float* destination,
    const float* message, float* normed, uint32_t rows, uint32_t width, float eps,
    cudaStream_t stream) {
    controller_norm_kernel<<<rows,256,0,stream>>>(destination,message,normed,width,eps);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_controller_mix(const float* destination,
    const float* message, const float* gain, const float* forget, float* out,
    uint32_t count, cudaStream_t stream) {
    controller_mix_kernel<<<(count+255)/256,256,0,stream>>>(destination,message,gain,forget,out,count);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_sconv(const float* x, const float* weight,
    const float* dynamic, const float* state, float* out, uint32_t rows, uint32_t hidden,
    uint32_t width, uint32_t nheads, cudaStream_t stream) {
    sconv_kernel<<<dim3((hidden+255)/256,rows),256,0,stream>>>(x,weight,dynamic,state,out,hidden,width,nheads);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_sconv_commit(const float* x, float* state,
    uint32_t rows, uint32_t hidden, uint32_t width, cudaStream_t stream) {
    sconv_commit_kernel<<<(hidden+255)/256,256,0,stream>>>(x,state,hidden,rows,width);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_l2_clip(const float* x, const float* scale, float* out,
    uint32_t rows, uint32_t heads, uint32_t dim, float threshold, float eps, float position,
    uint32_t tau_floor, float tau_alpha, cudaStream_t stream) {
    l2_clip_kernel<<<dim3(heads,rows),128,0,stream>>>(x,scale,out,heads,dim,threshold,eps,position,tau_floor,tau_alpha);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_rel_projection(const float* r, const uint16_t* projection,
    float* out, uint32_t rows, uint32_t heads, uint32_t extent, cudaStream_t stream) {
    const uint32_t count=rows*heads*extent;
    rel_projection_kernel<<<(count+255)/256,256,0,stream>>>(r,projection,out,rows*heads,extent);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_router(const float* logits, const float* bias,
    uint32_t* ids, float* weights, uint32_t rows, uint32_t experts,
    float route_scale, cudaStream_t stream) {
    if(experts>128) return cudaErrorInvalidValue;
    router_kernel<<<rows,32,0,stream>>>(logits,bias,ids,weights,experts,route_scale);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_capped_swiglu(const float* gate_up,
    const float* gammas, float* out, uint32_t routes, uint32_t intermediate,
    float gate_cap, float up_cap, cudaStream_t stream) {
    const uint32_t count=routes*intermediate;
    capped_swiglu_kernel<<<(count+255)/256,256,0,stream>>>(gate_up,gammas,out,routes,intermediate,gate_cap,up_cap);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_grouped_bf16_expert(const uint32_t* sorted,
    const uint32_t* item_expert, const uint32_t* item_start, const uint32_t* item_count,
    const uint32_t* items, uint32_t max_items, const float* input, const uint16_t* weights,
    float* out, uint32_t top_k, bool per_route_input, uint32_t outputs, uint32_t inputs,
    bool tiled, uint32_t max_columns, cudaStream_t stream) {
    if(outputs%16 || inputs%16 || !max_items) return cudaErrorInvalidValue;
    if(tiled) {
        if(inputs%8) return cudaErrorInvalidValue;
        grouped_bf16_expert_tiled_kernel<<<dim3((outputs/16+7)/8,max_items),256,0,stream>>>(sorted,
            item_expert,item_start,item_count,items,input,weights,out,top_k,per_route_input,outputs,inputs);
    } else if(max_columns<=8) {
        // A token never selects an expert twice, so decode items hold at most
        // one route per row.
        grouped_bf16_expert_kernel<16,1><<<dim3(outputs/16,max_items),512,0,stream>>>(sorted,
            item_expert,item_start,item_count,items,input,weights,out,top_k,per_route_input,outputs,inputs);
    } else {
        grouped_bf16_expert_kernel<8,8><<<dim3(outputs/16,max_items),256,0,stream>>>(sorted,
            item_expert,item_start,item_count,items,input,weights,out,top_k,per_route_input,outputs,inputs);
    }
    return cudaGetLastError();
}
extern "C" int infer_inkling2_bf16_matvec(const float* x, const uint16_t* weight, float* y,
    uint32_t rows, uint32_t outputs, uint32_t inputs, cudaStream_t stream) {
    if(!rows || rows>8 || !outputs || !inputs || inputs%8) return cudaErrorInvalidValue;
    bf16_matvec_kernel<8><<<(outputs+7)/8,256,0,stream>>>(x,weight,y,rows,outputs,inputs);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_routed_sum(const float* routes, float* out,
    uint32_t rows, uint32_t top_k, uint32_t hidden, cudaStream_t stream) {
    routed_sum_kernel<<<(rows*hidden+255)/256,256,0,stream>>>(routes,out,rows,top_k,hidden);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_finish_logits(float* logits, uint32_t count,
    uint32_t real_vocab, float softcap, cudaStream_t stream) {
    finish_logits_kernel<<<(count+255)/256,256,0,stream>>>(logits,count,real_vocab,softcap);
    return cudaGetLastError();
}
