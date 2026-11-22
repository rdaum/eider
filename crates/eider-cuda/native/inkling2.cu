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
// Block-wide sum for blocks of whole warps; every thread receives the result.
// `partial` must hold one value per warp.
__device__ float block_sum(float x, float* partial) {
    const unsigned lane=threadIdx.x&31, warp=threadIdx.x>>5, warps=blockDim.x>>5;
    for(int offset=16;offset;offset>>=1) x+=__shfl_xor_sync(0xffffffffu,x,offset);
    if(lane==0) partial[warp]=x;
    __syncthreads();
    float sum=0.f;
    for(unsigned w=0;w<warps;++w) sum+=partial[w];
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
// One pass accumulates every source's query dot and squared norm; a single
// warp-shuffle and shared-memory reduction combines them.
__global__ __launch_bounds__(256) void attn_res_kernel(
    const float* s0, const float* s1, const float* s2, const float* s3, const float* s4,
    const float* addend, const float* query, float* out, uint32_t sources,
    uint32_t width, float eps) {
    constexpr unsigned MaxSources=5, Warps=8;
    __shared__ float reduced[Warps][2*MaxSources];
    __shared__ float coefficient[MaxSources];
    const unsigned lane=threadIdx.x&31, warp=threadIdx.x>>5;
    const size_t row=size_t(blockIdx.x)*width;
    const float* src[MaxSources]={s0,s1,s2,s3,s4};
    const float scale=rsqrtf(float(width));
    float dot[MaxSources]={}, sum[MaxSources]={};
    for(unsigned i=threadIdx.x;i<width;i+=blockDim.x) {
        const float q=query[i]*scale;
        #pragma unroll
        for(unsigned n=0;n<MaxSources;++n) {
            if(n>=sources) break;
            float x=src[n][row+i];
            if(addend && n+1==sources) x=__fadd_rn(x,addend[row+i]);
            dot[n]=__fmaf_rn(x,q,dot[n]);
            sum[n]=__fmaf_rn(x,x,sum[n]);
        }
    }
    #pragma unroll
    for(unsigned n=0;n<MaxSources;++n) {
        for(int offset=16;offset;offset>>=1) {
            dot[n]+=__shfl_xor_sync(0xffffffffu,dot[n],offset);
            sum[n]+=__shfl_xor_sync(0xffffffffu,sum[n],offset);
        }
    }
    if(lane==0) {
        #pragma unroll
        for(unsigned n=0;n<MaxSources;++n) {
            reduced[warp][2*n]=dot[n];
            reduced[warp][2*n+1]=sum[n];
        }
    }
    __syncthreads();
    if(threadIdx.x==0) {
        float logits[MaxSources], norms[MaxSources], maximum=-INFINITY;
        for(unsigned n=0;n<sources;++n) {
            float d=0.f, q=0.f;
            for(unsigned w=0;w<Warps;++w) { d+=reduced[w][2*n]; q+=reduced[w][2*n+1]; }
            norms[n]=rsqrtf(q/width+eps);
            logits[n]=d*norms[n];
            maximum=fmaxf(maximum,logits[n]);
        }
        float total=0.f;
        for(unsigned n=0;n<sources;++n) {
            logits[n]=expf(logits[n]-maximum);
            total+=logits[n];
        }
        for(unsigned n=0;n<sources;++n) coefficient[n]=logits[n]/total*norms[n];
    }
    __syncthreads();
    for(unsigned i=threadIdx.x;i<width;i+=blockDim.x) {
        float value=0;
        #pragma unroll
        for(unsigned n=0;n<MaxSources;++n) {
            if(n>=sources) break;
            float x=src[n][row+i];
            if(addend && n+1==sources) x=__fadd_rn(x,addend[row+i]);
            value=__fmaf_rn(coefficient[n],x,value);
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
    float position, const uint32_t* device_position, uint32_t tau_floor, float tau_alpha) {
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
    if(device_position) position=float(device_position[0]);
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

__device__ __forceinline__ float e4m3(uint8_t code) {
    const uint32_t sign=uint32_t(code&0x80u)<<24, exponent=(code>>3)&0x0fu, mantissa=code&7u;
    if(exponent==0) {
        const float value=float(mantissa)*0x1p-9f;
        return sign ? -value : value;
    }
    return __uint_as_float(sign|((exponent+120u)<<23)|(mantissa<<20));
}

__device__ __forceinline__ float e8m0(uint8_t code) {
    return code ? __uint_as_float(uint32_t(code)<<23) : 0x1p-127f;
}

// BF16 MMA operand for MXFP8 weights `w[row][k]` and `w[row][k + 1]`. Both lie
// in one 32-value scale group. E4M3 times a power of two is exact in BF16.
__device__ __forceinline__ uint32_t mxfp8_pair(const uint8_t* w, const uint8_t* scales,
                                               uint32_t row, uint32_t inputs, uint32_t k) {
    const uint16_t bytes=*reinterpret_cast<const uint16_t*>(w+size_t(row)*inputs+k);
    const float scale=e8m0(scales[size_t(row)*(inputs/32)+k/32]);
    return pack_bf16(e4m3(bytes&0xffu)*scale,e4m3(bytes>>8)*scale);
}

// MXFP8 weights in [expert][out][in] order with one E8M0 scale per 32 inputs.
// Decode-sized batches: one CTA
// computes a 16-row output tile for one work item, splitting K across warps.
// Items with more than `Groups * 8` routes repeat the K loop.
template<unsigned Warps, unsigned Groups>
__global__ __launch_bounds__(Warps*32) void grouped_mxfp8_expert_kernel(
    const uint32_t* sorted, const uint32_t* item_expert, const uint32_t* item_start,
    const uint32_t* item_count, const uint32_t* items, const float* input,
    const uint8_t* weights, const uint8_t* weight_scales, float* out, uint32_t top_k,
    bool per_route_input, uint32_t outputs, uint32_t inputs) {
    const unsigned item=blockIdx.y;
    if(item>=items[0]) return;
    const unsigned lane=threadIdx.x&31, warp=threadIdx.x>>5;
    const unsigned row0=lane>>2, row1=row0+8, pair=(lane&3)*2;
    const unsigned start=item_start[item], count=item_count[item];
    const size_t base=size_t(item_expert[item])*outputs+blockIdx.x*16;
    const uint8_t* w=weights+base*inputs;
    const uint8_t* sc=weight_scales+base*(inputs/32);
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
            const uint32_t a0=mxfp8_pair(w,sc,row0,inputs,k+pair);
            const uint32_t a1=mxfp8_pair(w,sc,row1,inputs,k+pair);
            const uint32_t a2=mxfp8_pair(w,sc,row0,inputs,k+pair+8);
            const uint32_t a3=mxfp8_pair(w,sc,row1,inputs,k+pair+8);
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
__global__ __launch_bounds__(256) void grouped_mxfp8_expert_tiled_kernel(
    const uint32_t* sorted, const uint32_t* item_expert, const uint32_t* item_start,
    const uint32_t* item_count, const uint32_t* items, const float* input,
    const uint8_t* weights, const uint8_t* weight_scales, float* out, uint32_t top_k,
    bool per_route_input, uint32_t outputs, uint32_t inputs) {
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
    const size_t base=size_t(item_expert[item])*outputs+tile*16;
    const uint8_t* w=weights+base*inputs;
    const uint8_t* sc=weight_scales+base*(inputs/32);
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
                const uint32_t a0=mxfp8_pair(w,sc,row0,inputs,k+quad*2);
                const uint32_t a1=mxfp8_pair(w,sc,row1,inputs,k+quad*2);
                const uint32_t a2=mxfp8_pair(w,sc,row0,inputs,k+quad*2+8);
                const uint32_t a3=mxfp8_pair(w,sc,row1,inputs,k+quad*2+8);
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
    attn_res_kernel<<<rows,256,0,stream>>>(s0,s1,s2,s3,s4,addend,query,out,sources,width,eps);
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
    const uint32_t* device_position, uint32_t tau_floor, float tau_alpha, cudaStream_t stream) {
    l2_clip_kernel<<<dim3(heads,rows),128,0,stream>>>(x,scale,out,heads,dim,threshold,eps,position,device_position,tau_floor,tau_alpha);
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
extern "C" int infer_inkling2_grouped_mxfp8_expert(const uint32_t* sorted,
    const uint32_t* item_expert, const uint32_t* item_start, const uint32_t* item_count,
    const uint32_t* items, uint32_t max_items, const float* input, const uint8_t* weights,
    const uint8_t* scales, float* out, uint32_t top_k, bool per_route_input, uint32_t outputs,
    uint32_t inputs, bool tiled, uint32_t max_columns, cudaStream_t stream) {
    if(outputs%16 || inputs%32 || !max_items) return cudaErrorInvalidValue;
    if(tiled) {
        if(inputs%8) return cudaErrorInvalidValue;
        grouped_mxfp8_expert_tiled_kernel<<<dim3((outputs/16+7)/8,max_items),256,0,stream>>>(sorted,
            item_expert,item_start,item_count,items,input,weights,scales,out,top_k,per_route_input,outputs,inputs);
    } else if(max_columns<=8) {
        // A token never selects an expert twice, so decode items hold at most
        // one route per row.
        grouped_mxfp8_expert_kernel<16,1><<<dim3(outputs/16,max_items),512,0,stream>>>(sorted,
            item_expert,item_start,item_count,items,input,weights,scales,out,top_k,per_route_input,outputs,inputs);
    } else {
        grouped_mxfp8_expert_kernel<8,8><<<dim3(outputs/16,max_items),256,0,stream>>>(sorted,
            item_expert,item_start,item_count,items,input,weights,scales,out,top_k,per_route_input,outputs,inputs);
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

// DSpark draft operators. Draft activations are BF16 values in F32 storage;
// draft K/V share the target's BF16 ring and page-bundle layouts.
namespace {
constexpr uint32_t DRAFT_TILE=32;
constexpr uint32_t DRAFT_MAX_QUERIES=24;
constexpr uint32_t TOPK_CHUNK=2048;
constexpr uint32_t TOPK_MERGE=4096;

__global__ void finish_logit_rows_kernel(float* logits, uint32_t count, uint32_t vocab,
                                         uint32_t real_vocab, float softcap) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) logits[i]=i%vocab>=real_vocab ? -INFINITY : softcap*tanhf(logits[i]/softcap);
}

// RMS normalization of the alpha-weighted sum of feature taps laid out as
// `[tap][tap_rows][width]`, reading rows `first..first+rows`.
__global__ void draft_mix_kernel(const float* features, const float* alpha, const float* weight,
                                 float* out, uint32_t first, uint32_t tap_rows, uint32_t taps,
                                 uint32_t width, float eps) {
    __shared__ float partial[32];
    const size_t row=blockIdx.x;
    float sum=0.f;
    for(uint32_t i=threadIdx.x;i<width;i+=blockDim.x) {
        float mixed=0.f;
        for(uint32_t t=0;t<taps;++t)
            mixed+=features[(size_t(t)*tap_rows+first+row)*width+i]*alpha[t];
        out[row*width+i]=mixed;
        sum+=mixed*mixed;
    }
    const float inv=rsqrtf(block_sum(sum,partial)/width+eps);
    for(uint32_t i=threadIdx.x;i<width;i+=blockDim.x)
        out[row*width+i]=bf(out[row*width+i]*inv*weight[i]);
}

// In-place optional per-head RMS normalization (rounded to BF16) followed by
// NeoX RoPE at position `first + row` for 128-dimensional heads.
__global__ void draft_norm_rope_kernel(float* x, const float* weight, uint32_t row_stride,
                                       uint32_t first, float theta, float eps) {
    __shared__ float partial[2];
    const uint32_t lane=threadIdx.x, head=blockIdx.x, row=blockIdx.y;
    float* p=x+size_t(row)*row_stride+head*128;
    float a=p[lane], b=p[lane+64];
    if(weight) {
        float ss=a*a+b*b;
        for(int offset=16;offset;offset>>=1) ss+=__shfl_xor_sync(0xffffffffu,ss,offset);
        if((lane&31)==0) partial[lane>>5]=ss;
        __syncthreads();
        const float inv=1.f/sqrtf((partial[0]+partial[1])/128.f+eps);
        a=bf(a*inv*weight[lane]);
        b=bf(b*inv*weight[lane+64]);
    }
    const float inv_freq=1.f/powf(theta,float(2*lane)/128.f);
    const float angle=float(first+row)*inv_freq;
    const float c=cosf(angle), s=sinf(angle);
    p[lane]=bf(a*c-b*s);
    p[lane+64]=bf(b*c+a*s);
}

template <bool Paged>
__device__ __forceinline__ void draft_context_kv(const __nv_bfloat16* ck, const __nv_bfloat16* cv,
        const __nv_bfloat16* const* pages, uint32_t layer_offset, uint32_t capacity,
        uint32_t width, uint64_t position, uint32_t column, float& key, float& value) {
    if constexpr(Paged) {
        const __nv_bfloat16* page=pages[position>>7]+layer_offset;
        const size_t offset=size_t(position&127)*width+column;
        key=__bfloat162float(page[offset]);
        value=__bfloat162float(page[size_t(128)*width+offset]);
    } else {
        const size_t offset=size_t(position%capacity)*width+column;
        key=__bfloat162float(ck[offset]);
        value=__bfloat162float(cv[offset]);
    }
}

// Bidirectional attention of `rows` block queries over context keys
// `[first, end)` followed by the block's own keys. One CTA serves every query
// head of one KV head over one key split; warps own block rows.
template <bool Paged>
__global__ void draft_attention_kernel(const float* q, const float* bk, const float* bv,
        const __nv_bfloat16* ck, const __nv_bfloat16* cv, const __nv_bfloat16* const* pages,
        uint32_t layer_offset, uint32_t first, uint32_t end, uint32_t capacity, uint32_t rows,
        uint32_t qheads, uint32_t kvheads, uint32_t chunk, float* partial) {
    __shared__ float keys[DRAFT_TILE][129];
    __shared__ float values[DRAFT_TILE][129];
    __shared__ float query[DRAFT_MAX_QUERIES][128];
    const uint32_t kvhead=blockIdx.x, split=blockIdx.y, warp=threadIdx.x>>5, lane=threadIdx.x&31;
    const uint32_t group=qheads/kvheads, width=kvheads*128;
    const uint64_t context=end-first, total=context+rows;
    const uint64_t begin=uint64_t(split)*chunk, stop=min(total,begin+chunk);
    for(uint32_t i=threadIdx.x;i<rows*group*128;i+=blockDim.x) {
        const uint32_t index=i/128, d=i%128;
        query[index][d]=q[(size_t(index/group)*qheads+kvhead*group+index%group)*128+d];
    }
    float mx[3]={-INFINITY,-INFINITY,-INFINITY}, total_weight[3]={}, acc[3][4]={};
    const bool active=warp<rows;
    for(uint64_t base=begin;base<stop;base+=DRAFT_TILE) {
        __syncthreads();
        for(uint32_t i=threadIdx.x;i<DRAFT_TILE*128;i+=blockDim.x) {
            const uint32_t j=i/128, d=i%128;
            const uint64_t key=base+j;
            float kx=0.f, vx=0.f;
            if(key<stop) {
                if(key<context) {
                    draft_context_kv<Paged>(ck,cv,pages,layer_offset,capacity,width,first+key,
                                            kvhead*128+d,kx,vx);
                } else {
                    const size_t offset=size_t(key-context)*width+kvhead*128+d;
                    kx=bk[offset];
                    vx=bv[offset];
                }
            }
            keys[j][d]=kx;
            values[j][d]=vx;
        }
        __syncthreads();
        if(!active) continue;
        const bool valid=base+lane<stop;
        for(uint32_t g=0;g<group;++g) {
            const float* qv=query[warp*group+g];
            float score=0.f;
            for(uint32_t d=0;d<128;++d) score=fmaf(qv[d],keys[lane][d],score);
            score=valid ? score*0.08838834764831845f : -INFINITY;
            float tile=score;
            for(int offset=16;offset;offset>>=1) tile=fmaxf(tile,__shfl_xor_sync(0xffffffffu,tile,offset));
            const float next=fmaxf(mx[g],tile);
            const float keep=expf(mx[g]-next), p=valid ? expf(score-next) : 0.f;
            float sum=p;
            for(int offset=16;offset;offset>>=1) sum+=__shfl_xor_sync(0xffffffffu,sum,offset);
            total_weight[g]=total_weight[g]*keep+sum;
            #pragma unroll
            for(int c=0;c<4;++c) acc[g][c]*=keep;
            for(uint32_t j=0;j<DRAFT_TILE;++j) {
                const float pj=__shfl_sync(0xffffffffu,p,j);
                #pragma unroll
                for(int c=0;c<4;++c) acc[g][c]=fmaf(pj,values[j][lane+c*32],acc[g][c]);
            }
            mx[g]=next;
        }
    }
    if(!active) return;
    for(uint32_t g=0;g<group;++g) {
        float* dst=partial+((size_t(split)*rows+warp)*qheads+kvhead*group+g)*130;
        #pragma unroll
        for(int c=0;c<4;++c) dst[lane+c*32]=acc[g][c];
        if(lane==0) { dst[128]=mx[g]; dst[129]=total_weight[g]; }
    }
}

__global__ void draft_attention_combine_kernel(const float* partial, float* out, uint32_t queries,
                                               uint32_t splits) {
    const uint32_t index=blockIdx.x, d=threadIdx.x;
    float mx=-INFINITY;
    for(uint32_t s=0;s<splits;++s) mx=fmaxf(mx,partial[(size_t(s)*queries+index)*130+128]);
    float total=0.f, acc=0.f;
    for(uint32_t s=0;s<splits;++s) {
        const float* p=partial+(size_t(s)*queries+index)*130;
        const float w=p[128]==-INFINITY ? 0.f : expf(p[128]-mx);
        total+=p[129]*w;
        acc+=p[d]*w;
    }
    out[size_t(index)*128+d]=bf(acc/total);
}

template <bool Paged>
__global__ void draft_store_kernel(const float* k, const float* v, __nv_bfloat16* ck,
        __nv_bfloat16* cv, __nv_bfloat16* const* pages, uint32_t layer_offset, uint32_t first,
        uint32_t rows, uint32_t width, uint32_t capacity) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(rows)*width) return;
    const uint32_t row=i/width, column=i%width;
    const uint64_t position=uint64_t(first)+row;
    if constexpr(Paged) {
        __nv_bfloat16* page=pages[position>>7]+layer_offset;
        const size_t offset=size_t(position&127)*width+column;
        page[offset]=__float2bfloat16_rn(k[i]);
        page[size_t(128)*width+offset]=__float2bfloat16_rn(v[i]);
    } else {
        // Only the last `capacity` rows survive in a ring.
        if(rows>capacity && row<rows-capacity) return;
        const size_t offset=size_t(position%capacity)*width+column;
        ck[offset]=__float2bfloat16_rn(k[i]);
        cv[offset]=__float2bfloat16_rn(v[i]);
    }
}

// Rows after the anchor hold the learned mask embedding.
__global__ void draft_fill_kernel(const float* mask, float* out, uint32_t rows, uint32_t width) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<size_t(rows)*width) out[i]=mask[i%width];
}

// x[row] += a + b for broadcast rows `a` and optional `b`.
__global__ void draft_add_broadcast_kernel(float* x, const float* a, const float* b,
                                           uint32_t rows, uint32_t width) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<size_t(rows)*width) x[i]+=a[i%width]+(b ? b[i%width] : 0.f);
}

__global__ void draft_markov_input_kernel(const float* state, const uint16_t* embedding,
                                          const uint32_t* token, float* out, uint32_t rank) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=rank) return;
    out[i]=state[i];
    out[rank+i]=bf16(embedding[size_t(token[0])*rank+i]);
}

// The recurrent head's gate: each reference torch op rounds to BF16.
__global__ void draft_markov_gate_kernel(const float* raw, float* state, float* latent,
                                         uint32_t rank) {
    const uint32_t c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=rank) return;
    float gate=bf(raw[c]);
    const float candidate=bf(tanhf(bf(raw[rank+c])));
    const float out=bf(raw[2*rank+c]);
    gate=bf(1.f/(1.f+expf(-gate)));
    const float keep=bf(gate*state[c]);
    const float update=bf(bf(1.f-gate)*candidate);
    state[c]=bf(keep+update);
    latent[c]=bf(tanhf(out));
}

// Softcapped draft logits over the vocabulary: the base logit plus the
// folded Markov bias, each rounded to BF16 as in the reference.
__global__ void draft_markov_logits_kernel(const float* base, const float* bias,
        const float* fold_bias, float* out, uint32_t vocab, uint32_t real_vocab, float softcap) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=vocab) return;
    const float x=bf(base[i]+bf(bias[i]+fold_bias[i]));
    out[i]=i>=real_vocab ? -INFINITY : bf(softcap*tanhf(x/softcap));
}

// Descending bitonic sort of `n` (a power of two) value/ID pairs; ties order by ID.
__device__ void bitonic_descending(float* value, uint32_t* id, uint32_t n) {
    for(uint32_t k=2;k<=n;k<<=1) {
        for(uint32_t j=k>>1;j>0;j>>=1) {
            for(uint32_t i=threadIdx.x;i<n;i+=blockDim.x) {
                const uint32_t l=i^j;
                if(l<=i) continue;
                const bool before=value[i]>value[l] || (value[i]==value[l] && id[i]<id[l]);
                if(((i&k)==0) != before) {
                    const float tv=value[i]; value[i]=value[l]; value[l]=tv;
                    const uint32_t ti=id[i]; id[i]=id[l]; id[l]=ti;
                }
            }
            __syncthreads();
        }
    }
}

__global__ void draft_topk_chunks_kernel(const float* logits, uint32_t stride, uint32_t real_vocab,
        uint32_t k, float* chunk_values, uint32_t* chunk_ids) {
    __shared__ float value[TOPK_CHUNK];
    __shared__ uint32_t id[TOPK_CHUNK];
    const uint32_t chunk=blockIdx.x, row=blockIdx.y, chunks=gridDim.x;
    for(uint32_t i=threadIdx.x;i<TOPK_CHUNK;i+=blockDim.x) {
        const uint32_t token=chunk*TOPK_CHUNK+i;
        value[i]=token<real_vocab ? logits[size_t(row)*stride+token] : -INFINITY;
        id[i]=token;
    }
    __syncthreads();
    bitonic_descending(value,id,TOPK_CHUNK);
    for(uint32_t i=threadIdx.x;i<k;i+=blockDim.x) {
        chunk_values[(size_t(row)*chunks+chunk)*k+i]=value[i];
        chunk_ids[(size_t(row)*chunks+chunk)*k+i]=id[i];
    }
}

__global__ void draft_topk_merge_kernel(const float* chunk_values, const uint32_t* chunk_ids,
        uint32_t count, uint32_t k, float* values, uint32_t* ids) {
    __shared__ float value[TOPK_MERGE];
    __shared__ uint32_t id[TOPK_MERGE];
    const uint32_t row=blockIdx.x;
    for(uint32_t i=threadIdx.x;i<TOPK_MERGE;i+=blockDim.x) {
        value[i]=i<count ? chunk_values[size_t(row)*count+i] : -INFINITY;
        id[i]=i<count ? chunk_ids[size_t(row)*count+i] : UINT32_MAX;
    }
    __syncthreads();
    bitonic_descending(value,id,TOPK_MERGE);
    for(uint32_t i=threadIdx.x;i<k;i+=blockDim.x) {
        values[size_t(row)*k+i]=value[i];
        ids[size_t(row)*k+i]=id[i];
    }
}

// Copies the ring slots that appending `rows` tokens at `position` overwrites,
// or restores rows `keep..rows` of them.
__global__ void ring_slots_kernel(uint16_t* key, uint16_t* value, uint16_t* saved_key,
        uint16_t* saved_value, uint32_t position, const uint32_t* device_position, uint32_t keep,
        uint32_t rows, uint32_t width, uint32_t capacity, bool restore) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(rows)*width) return;
    if(device_position) position=device_position[0];
    const uint32_t row=i/width;
    if(row<keep) return;
    const size_t slot=size_t((uint64_t(position)+row)%capacity)*width+i%width;
    if(restore) { key[slot]=saved_key[i]; value[slot]=saved_value[i]; }
    else { saved_key[i]=key[slot]; saved_value[i]=value[slot]; }
}
}

extern "C" int infer_inkling2_finish_logit_rows(float* logits, uint32_t rows, uint32_t vocab,
    uint32_t real_vocab, float softcap, cudaStream_t stream) {
    const uint32_t count=rows*vocab;
    finish_logit_rows_kernel<<<(count+255)/256,256,0,stream>>>(logits,count,vocab,real_vocab,softcap);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_draft_mix(const float* features, const float* alpha,
    const float* weight, float* out, uint32_t first, uint32_t rows, uint32_t tap_rows,
    uint32_t taps, uint32_t width, float eps, cudaStream_t stream) {
    draft_mix_kernel<<<rows,256,0,stream>>>(features,alpha,weight,out,first,tap_rows,taps,width,eps);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_draft_norm_rope(float* x, const float* weight, uint32_t rows,
    uint32_t heads, uint32_t row_stride, uint32_t first, float theta, float eps,
    cudaStream_t stream) {
    draft_norm_rope_kernel<<<dim3(heads,rows),64,0,stream>>>(x,weight,row_stride,first,theta,eps);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_draft_attention(const float* q, const float* bk, const float* bv,
    const __nv_bfloat16* ck, const __nv_bfloat16* cv, const __nv_bfloat16* const* pages,
    uint32_t layer_offset, uint32_t first, uint32_t end, uint32_t capacity, uint32_t rows,
    uint32_t qheads, uint32_t kvheads, uint32_t splits, float* partial, float* out,
    cudaStream_t stream) {
    if(!rows || rows>8 || !kvheads || qheads%kvheads || qheads/kvheads>3 || !splits)
        return cudaErrorInvalidValue;
    const uint64_t total=uint64_t(end-first)+rows;
    const uint32_t chunk=uint32_t((total+splits-1)/splits+DRAFT_TILE-1)/DRAFT_TILE*DRAFT_TILE;
    if(pages) draft_attention_kernel<true><<<dim3(kvheads,splits),256,0,stream>>>(q,bk,bv,ck,cv,
        pages,layer_offset,first,end,capacity,rows,qheads,kvheads,chunk,partial);
    else draft_attention_kernel<false><<<dim3(kvheads,splits),256,0,stream>>>(q,bk,bv,ck,cv,
        pages,layer_offset,first,end,capacity,rows,qheads,kvheads,chunk,partial);
    draft_attention_combine_kernel<<<rows*qheads,128,0,stream>>>(partial,out,rows*qheads,splits);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_draft_store(const float* k, const float* v, __nv_bfloat16* ck,
    __nv_bfloat16* cv, __nv_bfloat16* const* pages, uint32_t layer_offset, uint32_t first,
    uint32_t rows, uint32_t width, uint32_t capacity, cudaStream_t stream) {
    const size_t count=size_t(rows)*width;
    if(pages) draft_store_kernel<true><<<(count+255)/256,256,0,stream>>>(k,v,ck,cv,pages,
        layer_offset,first,rows,width,capacity);
    else draft_store_kernel<false><<<(count+255)/256,256,0,stream>>>(k,v,ck,cv,pages,
        layer_offset,first,rows,width,capacity);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_draft_fill(const float* mask, float* out, uint32_t rows,
    uint32_t width, cudaStream_t stream) {
    const size_t count=size_t(rows)*width;
    draft_fill_kernel<<<(count+255)/256,256,0,stream>>>(mask,out,rows,width);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_draft_add_broadcast(float* x, const float* a, const float* b,
    uint32_t rows, uint32_t width, cudaStream_t stream) {
    const size_t count=size_t(rows)*width;
    draft_add_broadcast_kernel<<<(count+255)/256,256,0,stream>>>(x,a,b,rows,width);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_draft_markov_input(const float* state, const uint16_t* embedding,
    const uint32_t* token, float* out, uint32_t rank, cudaStream_t stream) {
    draft_markov_input_kernel<<<(rank+255)/256,256,0,stream>>>(state,embedding,token,out,rank);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_draft_markov_gate(const float* raw, float* state, float* latent,
    uint32_t rank, cudaStream_t stream) {
    draft_markov_gate_kernel<<<(rank+255)/256,256,0,stream>>>(raw,state,latent,rank);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_draft_markov_logits(const float* base, const float* bias,
    const float* fold_bias, float* out, uint32_t vocab, uint32_t real_vocab, float softcap,
    cudaStream_t stream) {
    draft_markov_logits_kernel<<<(vocab+255)/256,256,0,stream>>>(base,bias,fold_bias,out,vocab,
        real_vocab,softcap);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_draft_topk(const float* logits, uint32_t rows, uint32_t stride,
    uint32_t real_vocab, uint32_t k, float* chunk_values, uint32_t* chunk_ids, float* values,
    uint32_t* ids, cudaStream_t stream) {
    const uint32_t chunks=(real_vocab+TOPK_CHUNK-1)/TOPK_CHUNK;
    if(!rows || !k || k>TOPK_CHUNK || chunks*k>TOPK_MERGE) return cudaErrorInvalidValue;
    draft_topk_chunks_kernel<<<dim3(chunks,rows),1024,0,stream>>>(logits,stride,real_vocab,k,
        chunk_values,chunk_ids);
    draft_topk_merge_kernel<<<rows,1024,0,stream>>>(chunk_values,chunk_ids,chunks*k,k,values,ids);
    return cudaGetLastError();
}
extern "C" int infer_inkling2_ring_slots(uint16_t* key, uint16_t* value, uint16_t* saved_key,
    uint16_t* saved_value, uint32_t position, const uint32_t* device_position, uint32_t keep,
    uint32_t rows, uint32_t width, uint32_t capacity, int restore, cudaStream_t stream) {
    const size_t count=size_t(rows)*width;
    ring_slots_kernel<<<(count+255)/256,256,0,stream>>>(key,value,saved_key,saved_value,position,
        device_position,keep,rows,width,capacity,restore!=0);
    return cudaGetLastError();
}
