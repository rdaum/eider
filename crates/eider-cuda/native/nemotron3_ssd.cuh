// BF16 Mamba SSD prefill for Nemotron's 64-channel, 128-state heads.
// Chunk products use warp tensor cores; only inter-chunk state passing is serial.
#include <mma.h>
extern "C" __device__ float __nv_expf(float);
extern "C" __device__ float __nv_log1pf(float);
namespace nemotron_ssd {
using namespace nvcuda;
constexpr unsigned T = 64, D = 64, S = 128;
using A = wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major>;
using B = wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major>;
using Acc = wmma::fragment<wmma::accumulator,16,16,16,float>;
struct Chunk { unsigned sequence, begin, length; };
__device__ Chunk chunk(unsigned index, const unsigned* offsets, const unsigned* lengths, unsigned sequences) {
    for (unsigned s=0; s<sequences; ++s) {
        const unsigned count=(lengths[s]+T-1)/T;
        if (index<count) return {s, offsets[s]+index*T, min(T,lengths[s]-index*T)};
        index-=count;
    }
    return {0,0,0};
}
__global__ void coefficients(const float* projected, const uint16_t* a_log, const uint16_t* bias,
    const unsigned* offsets, const unsigned* lengths, unsigned sequences, unsigned heads,
    unsigned projection, float2* coeff) {
    const auto ch=chunk(blockIdx.x,offsets,lengths,sequences);
    if (!ch.length) return;
    const unsigned head=blockIdx.y;
    // A short prefix per head avoids cancellation from subtracting long sequence sums.
    if (threadIdx.x) return;
    const float a=-__nv_expf(__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(a_log)[head]));
    const float b=__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(bias)[head]);
    float sum=0.f;
    for (unsigned t=0;t<T;++t) {
        float dt=0.f;
        if (t<ch.length) {
            const float v=projected[size_t(ch.begin+t)*projection+projection-heads+head]+b;
            dt=__nv_log1pf(__nv_expf(-fabsf(v)))+fmaxf(v,0.f);
            sum=__fadd_rn(sum,__fmul_rn(a,dt));
        }
        coeff[(size_t(blockIdx.x)*heads+head)*T+t]=make_float2(dt,sum);
    }
    coeff[size_t(gridDim.x)*heads*T+size_t(blockIdx.x)*heads+head]=make_float2(0.f,__nv_expf(sum));
}
__global__ void chunk_state(const float* conv, const float2* coeff, const unsigned* offsets,
    const unsigned* lengths, unsigned sequences, unsigned heads, unsigned groups, float* states) {
    const auto ch=chunk(blockIdx.x,offsets,lengths,sequences);
    if (!ch.length) return;
    const unsigned head=blockIdx.y, group=head/(heads/groups), width=heads*D+2*groups*S;
    __shared__ __align__(32) __nv_bfloat16 x[D*T], b[T*S];
    __shared__ float scales[T];
    const auto* cs=coeff+(size_t(blockIdx.x)*heads+head)*T;
    if (threadIdx.x<T) scales[threadIdx.x]=__nv_expf(fminf(cs[T-1].y-cs[threadIdx.x].y,0.f))*cs[threadIdx.x].x;
    __syncthreads();
    for (unsigned i=threadIdx.x;i<D*T;i+=blockDim.x) {
        const unsigned d=i/T,t=i%T;
        x[i]=__float2bfloat16_rn(t<ch.length ? conv[size_t(ch.begin+t)*width+head*D+d] : 0.f);
    }
    for (unsigned i=threadIdx.x;i<T*S;i+=blockDim.x) {
        const unsigned t=i/S,n=i%S;
        const float value=t<ch.length ? conv[size_t(ch.begin+t)*width+heads*D+group*S+n] : 0.f;
        b[i]=__float2bfloat16_rn(value*scales[t]);
    }
    __syncthreads();
    const unsigned warp=threadIdx.x/32;
    float* dest=states+(size_t(head)*gridDim.x+blockIdx.x)*D*S;
    for (unsigned tile=warp;tile<(D/16)*(S/16);tile+=8) {
        const unsigned m=tile/(S/16)*16,n=tile%(S/16)*16;
        Acc acc; wmma::fill_fragment(acc,0.f);
        for (unsigned k=0;k<T;k+=16) {
            A a; B bb;
            wmma::load_matrix_sync(a,x+m*T+k,T);
            wmma::load_matrix_sync(bb,b+k*S+n,S);
            wmma::mma_sync(acc,a,bb,acc);
        }
        wmma::store_matrix_sync(dest+m*S+n,acc,S,wmma::mem_row_major);
    }
}
// Overwrite chunk contributions with the BF16 incoming state used by its output scan.
// Keep the carried state in F32 registers, matching vLLM's state-passing kernel.
template<typename State>
__global__ void pass_states(float* __restrict__ states, const float2* __restrict__ coeff,
    uint8_t* const* __restrict__ table, const unsigned* lengths, unsigned heads, unsigned chunks) {
    constexpr unsigned V=4;
    const unsigned index=(blockIdx.x*blockDim.x+threadIdx.x)*V, sequence=blockIdx.y;
    if (index>=heads*D*S) return;
    unsigned start=0;
    for (unsigned s=0;s<sequence;++s) start+=(lengths[s]+T-1)/T;
    const unsigned count=(lengths[sequence]+T-1)/T, head=index/(D*S);
    State* live=reinterpret_cast<State*>(table[sequence]);
    float value[V];
    #pragma unroll
    for(unsigned i=0;i<V;++i) value[i]=float(live[index+i]);
    for (unsigned c=start;c<start+count;++c) {
        float* entry=states+(size_t(head)*chunks+c)*D*S+index%(D*S);
        const float4 contribution=*reinterpret_cast<const float4*>(entry);
        *reinterpret_cast<float4*>(entry)=make_float4(
            __bfloat162float(__float2bfloat16_rn(value[0])),__bfloat162float(__float2bfloat16_rn(value[1])),
            __bfloat162float(__float2bfloat16_rn(value[2])),__bfloat162float(__float2bfloat16_rn(value[3])));
        const float decay=coeff[size_t(chunks)*heads*T+size_t(c)*heads+head].y;
        value[0]=decay*value[0]+contribution.x;
        value[1]=decay*value[1]+contribution.y;
        value[2]=decay*value[2]+contribution.z;
        value[3]=decay*value[3]+contribution.w;
    }
    #pragma unroll
    for(unsigned i=0;i<V;++i) live[index+i]=State(value[i]);
}
__global__ void cb_product(const float* conv, const unsigned* offsets, const unsigned* lengths,
    unsigned sequences, unsigned heads, unsigned groups, float* cb) {
    const auto ch=chunk(blockIdx.x,offsets,lengths,sequences);
    if (!ch.length) return;
    const unsigned group=blockIdx.y,width=heads*D+2*groups*S;
    __shared__ __align__(32) __nv_bfloat16 c[T*S], b[S*T];
    for (unsigned i=threadIdx.x;i<T*S;i+=blockDim.x) {
        const unsigned t=i/S,n=i%S;
        c[i]=__float2bfloat16_rn(t<ch.length ? conv[size_t(ch.begin+t)*width+heads*D+groups*S+group*S+n] : 0.f);
        b[n*T+t]=__float2bfloat16_rn(t<ch.length ? conv[size_t(ch.begin+t)*width+heads*D+group*S+n] : 0.f);
    }
    __syncthreads();
    const unsigned warp=threadIdx.x/32;
    float* dest=cb+(size_t(blockIdx.x)*groups+group)*T*T;
    for (unsigned tile=warp;tile<16;tile+=8) {
        const unsigned m=tile/4*16,n=tile%4*16;
        Acc acc;wmma::fill_fragment(acc,0.f);
        for (unsigned k=0;k<S;k+=16) {
            A a;B bb;wmma::load_matrix_sync(a,c+m*S+k,S);
            wmma::load_matrix_sync(bb,b+k*T+n,T);wmma::mma_sync(acc,a,bb,acc);
        }
        wmma::store_matrix_sync(dest+m*T+n,acc,T,wmma::mem_row_major);
    }
}
__global__ void scan(const float* projected, const float* conv, const float2* coeff,
    const float* states, const float* cb, const uint16_t* d_weight, const unsigned* offsets,
    const unsigned* lengths, unsigned sequences, unsigned heads, unsigned groups, float* output) {
    const auto ch=chunk(blockIdx.x,offsets,lengths,sequences);
    if (!ch.length) return;
    const unsigned head=blockIdx.y, group=head/(heads/groups), inner=heads*D;
    const unsigned width=inner+2*groups*S, projection=inner+width+heads;
    __shared__ __align__(32) __nv_bfloat16 left[T*S], right[S*D];
    __shared__ __align__(32) float result[T*D];
    const auto* cs=coeff+(size_t(blockIdx.x)*heads+head)*T;
    const auto* initial=states+(size_t(head)*gridDim.x+blockIdx.x)*D*S;
    for (unsigned i=threadIdx.x;i<T*S;i+=blockDim.x) {
        const unsigned t=i/S,n=i%S;
        left[i]=__float2bfloat16_rn(t<ch.length ? conv[size_t(ch.begin+t)*width+inner+groups*S+group*S+n] : 0.f);
    }
    for (unsigned i=threadIdx.x;i<S*D;i+=blockDim.x)
        right[i]=__float2bfloat16_rn(initial[(i%D)*S+i/D]);
    __syncthreads();
    const unsigned warp=threadIdx.x/32;
    Acc acc[2];
    for (unsigned q=0;q<2;++q) {
        const unsigned tile=warp+q*8,m=tile/4*16,n=tile%4*16;
        wmma::fill_fragment(acc[q],0.f);
        for (unsigned k=0;k<S;k+=16) {
            A a;B b;wmma::load_matrix_sync(a,left+m*S+k,S);
            wmma::load_matrix_sync(b,right+k*D+n,D);wmma::mma_sync(acc[q],a,b,acc[q]);
        }
        wmma::store_matrix_sync(result+m*D+n,acc[q],D,wmma::mem_row_major);
    }
    __syncthreads();
    // Operand storage is dead here; reuse it for one scale per output row.
    float* row_scale=reinterpret_cast<float*>(right);
    if (threadIdx.x<T) row_scale[threadIdx.x]=__nv_expf(cs[threadIdx.x].y);
    __syncthreads();
    for (unsigned i=threadIdx.x;i<T*D;i+=blockDim.x) result[i]*=row_scale[i/D];
    __syncthreads();
    const float* products=cb+(size_t(blockIdx.x)*groups+group)*T*T;
    for (unsigned i=threadIdx.x;i<T*T;i+=blockDim.x) {
        const unsigned t=i/T,k=i%T;
        left[i]=__float2bfloat16_rn(k<=t && t<ch.length ? (products[i]*__nv_expf(fminf(cs[t].y-cs[k].y,0.f)))*cs[k].x : 0.f);
    }
    for (unsigned i=threadIdx.x;i<T*D;i+=blockDim.x) {
        const unsigned t=i/D,d=i%D;
        right[i]=__float2bfloat16_rn(t<ch.length ? conv[size_t(ch.begin+t)*width+head*D+d] : 0.f);
    }
    __syncthreads();
    for (unsigned q=0;q<2;++q) {
        const unsigned tile=warp+q*8,m=tile/4*16,n=tile%4*16;
        wmma::load_matrix_sync(acc[q],result+m*D+n,D,wmma::mem_row_major);
        for (unsigned k=0;k<T;k+=16) {
            A a;B b;wmma::load_matrix_sync(a,left+m*T+k,T);
            wmma::load_matrix_sync(b,right+k*D+n,D);wmma::mma_sync(acc[q],a,b,acc[q]);
        }
        wmma::store_matrix_sync(result+m*D+n,acc[q],D,wmma::mem_row_major);
    }
    __syncthreads();
    const float d=__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(d_weight)[head]);
    for (unsigned i=threadIdx.x;i<ch.length*D;i+=blockDim.x) {
        const unsigned t=i/D,flat=head*D+i%D;
        const float y=__bfloat162float(__float2bfloat16_rn(result[i]+d*conv[size_t(ch.begin+t)*width+flat]));
        const float z=projected[size_t(ch.begin+t)*projection+flat];
        output[size_t(ch.begin+t)*inner+flat]=y*(z/(1.f+__nv_expf(-z)));
    }
}
} // namespace nemotron_ssd
extern "C" cudaError_t infer_nemotron3_ssd_prefill(
    const float* projected, const float* conv, const uint16_t* a_log, const uint16_t* d,
    const uint16_t* bias, const uint16_t* norm, uint8_t* const* table,
    const unsigned* offsets, const unsigned* lengths, unsigned sequences, unsigned tokens,
    unsigned heads, unsigned groups, int ssm_fp32, float eps,
    float* coeff, float* states, float* cb, float* output, cudaStream_t stream) {
    using namespace nemotron_ssd;
    const unsigned chunks=(tokens+T-1)/T+sequences-1;
    coefficients<<<dim3(chunks,heads),32,0,stream>>>(projected,a_log,bias,offsets,lengths,sequences,heads,heads*D*2+2*groups*S+heads,reinterpret_cast<float2*>(coeff));
    chunk_state<<<dim3(chunks,heads),256,0,stream>>>(conv,reinterpret_cast<float2*>(coeff),offsets,lengths,sequences,heads,groups,states);
    if (ssm_fp32) pass_states<float><<<dim3((heads*D*S+1023)/1024,sequences),256,0,stream>>>(states,reinterpret_cast<float2*>(coeff),table,lengths,heads,chunks);
    else pass_states<__nv_bfloat16><<<dim3((heads*D*S+1023)/1024,sequences),256,0,stream>>>(states,reinterpret_cast<float2*>(coeff),table,lengths,heads,chunks);
    cb_product<<<dim3(chunks,groups),256,0,stream>>>(conv,offsets,lengths,sequences,heads,groups,cb);
    scan<<<dim3(chunks,heads),256,0,stream>>>(projected,conv,reinterpret_cast<float2*>(coeff),states,cb,d,offsets,lengths,sequences,heads,groups,output);
    infer_nemotron3_group_rms_norm_f32_chunks_kernel<<<tokens*groups,256,0,stream>>>(output,norm,groups,heads*D/groups,eps,groups>1 ? 2 : 1);
    return cudaGetLastError();
}
