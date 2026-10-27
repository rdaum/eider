// LFM3 BF16 operator boundaries in F32 scratch and recurrent storage.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>

__device__ __forceinline__ float bf(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

__global__ void norm(const float* x, const float* weight, float* out, uint32_t width, float eps) {
    __shared__ float partial[256];
    const uint32_t lane = threadIdx.x;
    const size_t offset = size_t(blockIdx.x) * width;
    float sum = 0;
    for (uint32_t i=lane; i<width; i+=256) sum += x[offset+i] * x[offset+i];
    partial[lane] = sum;
    __syncthreads();
    for (uint32_t stride=128; stride; stride>>=1) {
        if (lane < stride) partial[lane] += partial[lane+stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / width + eps);
    for (uint32_t i=lane; i<width; i+=256)
        out[offset+i] = bf(weight[i] * bf(x[offset+i] * inv));
}

extern "C" int infer_lfm3_norm(const float* x, const float* weight, float* out,
                              uint32_t rows, uint32_t width, float eps, cudaStream_t stream) {
    norm<<<rows,256,0,stream>>>(x,weight,out,width,eps);
    return cudaGetLastError();
}

__global__ void conv(const float* bcx, const float* weight, const float* state,
                     float* out, uint32_t hidden, uint32_t width) {
    const uint32_t c=blockIdx.x*256+threadIdx.x, row=blockIdx.y;
    if (c>=hidden) return;
    float sum=0;
    for (uint32_t j=0; j<width; ++j) {
        const int64_t source=int64_t(row)+j-(width-1);
        float bx;
        if (source<0) bx=state[size_t(c)*(width-1)+row+j];
        else {
            const float* previous=bcx+size_t(source)*3*hidden;
            bx=bf(previous[c]*previous[2*hidden+c]);
        }
        sum=fmaf(bx,weight[size_t(c)*width+j],sum);
    }
    out[size_t(row)*hidden+c]=bf(bcx[size_t(row)*3*hidden+hidden+c]*bf(sum));
}

__global__ void commit(const float* bcx, float* state, uint32_t hidden, uint32_t width, uint32_t rows) {
    const uint32_t c=blockIdx.x*256+threadIdx.x;
    if (c>=hidden) return;
    for (uint32_t j=0; j+1<width; ++j) {
        const int64_t source=int64_t(rows)+j-(width-1);
        if (source<0) state[size_t(c)*(width-1)+j]=state[size_t(c)*(width-1)+rows+j];
        else {
            const float* previous=bcx+size_t(source)*3*hidden;
            state[size_t(c)*(width-1)+j]=bf(previous[c]*previous[2*hidden+c]);
        }
    }
}

extern "C" int infer_lfm3_conv(const float* bcx, const float* weight, float* state, float* out,
                              uint32_t rows, uint32_t hidden, uint32_t width, cudaStream_t stream) {
    conv<<<dim3((hidden+255)/256,rows),256,0,stream>>>(bcx,weight,state,out,hidden,width);
    auto status=cudaGetLastError();
    if (status!=cudaSuccess) return status;
    commit<<<(hidden+255)/256,256,0,stream>>>(bcx,state,hidden,width,rows);
    return cudaGetLastError();
}

__global__ void rope(const float* x, const float* freq, float* out,
                     uint32_t count, uint32_t heads, uint32_t dim, uint32_t start) {
    const uint32_t index=blockIdx.x*256+threadIdx.x;
    if (index>=count) return;
    const uint32_t half=dim/2, pair=index%half, head=index/half;
    const size_t offset=size_t(head)*dim+pair;
    const float angle=float(start+head/heads)*freq[pair];
    const float cosine=bf(cosf(angle)), sine=bf(sinf(angle));
    const float a=x[offset], b=x[offset+half];
    out[offset]=bf(__fadd_rn(__fmul_rn(a,cosine),-__fmul_rn(b,sine)));
    out[offset+half]=bf(__fadd_rn(__fmul_rn(b,cosine),__fmul_rn(a,sine)));
}
extern "C" int infer_lfm3_rope(const float* x, const float* freq, float* out,
                              uint32_t rows, uint32_t heads, uint32_t dim, uint32_t start, cudaStream_t stream) {
    const uint32_t count=rows*heads*(dim/2);
    rope<<<(count+255)/256,256,0,stream>>>(x,freq,out,count,heads,dim,start);
    return cudaGetLastError();
}

__global__ void silu(const float* gate, const float* up, float* out, uint32_t count) {
    const uint32_t i=blockIdx.x*256+threadIdx.x;
    if (i<count) out[i]=bf(bf(gate[i]/(1.f+expf(-gate[i])))*up[i]);
}
extern "C" int infer_lfm3_silu(const float* gate, const float* up, float* out, uint32_t count, cudaStream_t stream) {
    silu<<<(count+255)/256,256,0,stream>>>(gate,up,out,count);
    return cudaGetLastError();
}

__global__ void head_gate(const float* gate, const float* x, float* out, uint32_t count, uint32_t dim) {
    const uint32_t i=blockIdx.x*256+threadIdx.x;
    if (i<count) out[i]=bf(bf(1.f/(1.f+expf(-gate[i/dim])))*bf(x[i]));
}
extern "C" int infer_lfm3_head_gate(const float* gate, const float* x, float* out, uint32_t heads, uint32_t dim, cudaStream_t stream) {
    const uint32_t count=heads*dim;
    head_gate<<<(count+255)/256,256,0,stream>>>(gate,x,out,count,dim);
    return cudaGetLastError();
}
