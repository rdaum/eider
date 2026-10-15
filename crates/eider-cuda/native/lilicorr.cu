// LiLiCorr candidate-lattice operations. Activations use BF16 values in F32 storage.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

namespace {
__device__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ float warp_sum(float x) {
    for (int d = 16; d; d >>= 1) x += __shfl_down_sync(0xffffffff, x, d);
    return __shfl_sync(0xffffffff, x, 0);
}
__device__ float block_sum(float x) {
    __shared__ float sums[8];
    x = warp_sum(x);
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x / 32] = x;
    __syncthreads();
    x = threadIdx.x < 8 ? sums[threadIdx.x] : 0.f;
    x = warp_sum(x);
    if (threadIdx.x == 0) sums[0] = x;
    __syncthreads();
    return sums[0];
}
__device__ float block_max(float x) {
    __shared__ float maxima[8];
    for (int d = 16; d; d >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffff, x, d));
    if ((threadIdx.x & 31) == 0) maxima[threadIdx.x / 32] = x;
    __syncthreads();
    x = threadIdx.x < 8 ? maxima[threadIdx.x] : -INFINITY;
    for (int d = 16; d; d >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffff, x, d));
    if (threadIdx.x == 0) maxima[0] = x;
    __syncthreads();
    return maxima[0];
}

__global__ void candidates(const float* logits, uint32_t* ids, float* logp,
                           uint32_t vocab, uint32_t k) {
    const uint32_t row = blockIdx.x;
    const float* x = logits + size_t(row) * vocab;
    float mx = -INFINITY;
    for (uint32_t i = threadIdx.x; i < vocab; i += 256) mx = fmaxf(mx, x[i]);
    mx = block_max(mx);
    float sum = 0.f;
    for (uint32_t i = threadIdx.x; i < vocab; i += 256) sum += expf(x[i] - mx);
    const float logz = logf(block_sum(sum));
    __shared__ uint32_t selected[16];
    __shared__ uint32_t minima[256];
    for (uint32_t rank = 0; rank < k; ++rank) {
        float best = -INFINITY;
        uint32_t id = UINT32_MAX;
        for (uint32_t i = threadIdx.x; i < vocab; i += 256) {
            bool used = false;
            for (uint32_t r = 0; r < rank; ++r) used |= selected[r] == i;
            if (!used && (x[i] > best || (x[i] == best && i < id))) { best = x[i]; id = i; }
        }
        const float top = block_max(best);
        minima[threadIdx.x] = best == top ? id : UINT32_MAX;
        __syncthreads();
        for (int d = 128; d; d >>= 1) {
            if (threadIdx.x < d) minima[threadIdx.x] = min(minima[threadIdx.x], minima[threadIdx.x+d]);
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            selected[rank] = minima[0];
            ids[row*k+rank] = minima[0];
            logp[row*k+rank] = (top - mx) - logz;
        }
        __syncthreads();
    }
}

__global__ void bias_activation(float* x, const float* bias, uint32_t count,
                                uint32_t width, bool silu) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float v = bf(x[i] + (bias ? bias[i % width] : 0.f));
    x[i] = silu ? bf(v / (1.f + expf(-v))) : v;
}

__global__ void features(const float* logp, const float* gamma, const float* beta,
                         float* out, uint32_t count, uint32_t k) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint32_t rank = i % k;
    float p = logp[i];
    float f[5] = {bf(p), bf(expf(p)), bf(p - logp[i-rank]),
                  bf(k > 1 ? float(rank) / float(k-1) : 0.f), rank == 0 ? 1.f : 0.f};
    float mean = 0.f, variance = 0.f;
    for (int c = 0; c < 5; ++c) mean += f[c] * .2f;
    for (int c = 0; c < 5; ++c) variance += (f[c]-mean) * (f[c]-mean) * .2f;
    float inv = rsqrtf(variance + 1.e-5f);
    for (int c = 0; c < 5; ++c) out[i*5+c] = bf((f[c]-mean) * inv * gamma[c] + beta[c]);
}

__global__ void hidden(const float* token, const float* pass, const float* feature,
                       const float* slot, const float* rank, float* out,
                       uint32_t count, uint32_t width, uint32_t k) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint32_t node = i / width, col = i % width;
    float v = bf(token[i] + pass[(node/k)*width+col]);
    v = bf(v + feature[i]);
    v = bf(v + slot[(node/k)*width+col]);
    out[i] = bf(v + rank[(node%k)*width+col]);
}

// One warp owns a query/head; 128-wide dot products and online softmax stay F32.
__global__ void lattice_attention(const float* qkv, const float* relative,
                                  const float* same, float* out, uint32_t nodes,
                                  uint32_t width, uint32_t heads, uint32_t k,
                                  uint32_t trained_slots) {
    uint32_t query = blockIdx.x / heads, head = blockIdx.x % heads, lane = threadIdx.x;
    uint32_t dim = width / heads;
    float acc[4] = {0.f,0.f,0.f,0.f};
    float mx = -INFINITY, sum = 0.f;
    for (uint32_t key = 0; key < nodes; ++key) {
        float dot = 0.f;
        for (uint32_t c = 0; c < 4; ++c) {
            uint32_t col = head*dim + lane + 32*c;
            dot += qkv[query*3*width+col] * qkv[key*3*width+width+col];
        }
        dot = warp_sum(dot) * rsqrtf(float(dim));
        int distance = int(query/k) - int(key/k);
        float bias = relative[head*(2*trained_slots+1)+distance+trained_slots];
        bias = bf(bias + (distance == 0 ? same[head] : 0.f));
        dot += bias;
        float next = fmaxf(mx, dot), keep = expf(mx-next), p = expf(dot-next);
        sum = sum*keep + p;
        for (uint32_t c = 0; c < 4; ++c) {
            uint32_t col = head*dim + lane + 32*c;
            acc[c] = acc[c]*keep + p*qkv[key*3*width+2*width+col];
        }
        mx = next;
    }
    for (uint32_t c = 0; c < 4; ++c) out[query*width+head*dim+lane+32*c] = bf(acc[c]/sum);
}

__global__ void anchor_cross(const float* x, const float* anchor, float* cross,
                             uint32_t count, uint32_t width) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) cross[i] = bf(x[i]*anchor[i%width]);
}

__global__ void factors(const float* self, const float* anchor, const float* cross,
                        float* out, uint32_t count, uint32_t width) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float v = bf(bf(self[i] + anchor[i%width]) + cross[i]);
    out[i] = bf(v / (1.f + expf(-v)));
}

__global__ void normalize(float* x, uint32_t width, float eps) {
    float* row = x + size_t(blockIdx.x)*width;
    float sum = 0.f;
    for (uint32_t c = threadIdx.x; c < width; c += 256) sum += row[c]*row[c];
    float norm = fmaxf(bf(sqrtf(block_sum(sum))), bf(eps));
    for (uint32_t c = threadIdx.x; c < width; c += 256) row[c] = bf(row[c]/norm);
}

__global__ void edges(const float* outgoing, const float* incoming, const float* anchor,
                      float* scores, uint32_t width, uint32_t k, float scale) {
    uint32_t edge = blockIdx.x, step = edge/(k*k), pred = (edge/k)%k, next = edge%k;
    const float* left = step ? outgoing + ((step-1)*k+pred)*width : anchor;
    const float* right = incoming + (step*k+next)*width;
    float sum = 0.f;
    for (uint32_t c = threadIdx.x; c < width; c += 256) {
        float v = left[c]*right[c];
        // The anchor uses an elementwise product followed by sum in the reference.
        sum += step ? v : bf(v);
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0) scores[edge] = scale*bf(sum);
}

__global__ void capture(const float* hidden, float* out, uint32_t count, uint32_t width,
                        uint32_t taps, uint32_t tap) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[(i/width*taps+tap)*width+i%width] = bf(hidden[i]);
}

// Split the sliding context into 256-key partitions. A warp owns one
// query/head/partition and keeps its 128-dimensional query and output in registers.
__global__ void context_attention_partial(
    const float* q, const float* ck, const float* cv, const float* bk, const float* bv,
    float* partial, uint32_t end, uint32_t retained, uint32_t rows,
    uint32_t qheads, uint32_t kvheads, uint32_t window, uint32_t splits) {
    const uint32_t head = blockIdx.x, query = blockIdx.y, part = blockIdx.z;
    const uint32_t lane = threadIdx.x;
    const uint32_t kvhead = head / (qheads / kvheads);
    const uint32_t first = max(end - retained, end + query + 1 > window ? end + query + 1 - window : 0);
    const uint32_t context = end - first;
    const uint32_t count = context + rows;
    float query_value[4], acc[4] = {};
    #pragma unroll
    for (int c=0;c<4;++c) query_value[c] = q[(query*qheads+head)*128+lane+c*32];
    float mx = -INFINITY, total = 0.f;
    for (uint32_t i=part*256;i<min(count,(part+1)*256);++i) {
        const bool cached = i<context;
        const uint32_t pos = cached ? (first+i)%window : i-context;
        const size_t base = (size_t(pos)*kvheads+kvhead)*128+lane;
        const float* key = (cached ? ck : bk) + base;
        const float* val = (cached ? cv : bv) + base;
        float dot = 0.f;
        #pragma unroll
        for (int c=0;c<4;++c) dot = fmaf(query_value[c], key[c*32], dot);
        dot = warp_sum(dot) * 0.08838834764831845f;
        const float next = fmaxf(mx,dot), keep = expf(mx-next), p = expf(dot-next);
        total = total*keep+p;
        #pragma unroll
        for (int c=0;c<4;++c) acc[c] = acc[c]*keep+p*val[c*32];
        mx=next;
    }
    float* dst = partial + ((query*qheads+head)*splits+part)*130;
    #pragma unroll
    for (int c=0;c<4;++c) dst[lane+c*32]=acc[c];
    if (lane==0) { dst[128]=mx; dst[129]=total; }
}

__global__ void context_attention_merge(const float* partial, float* out, uint32_t splits) {
    const uint32_t lane=threadIdx.x, row=blockIdx.x;
    const float* src=partial+row*splits*130;
    float mx=-INFINITY, total=0.f, acc[4]={};
    for (uint32_t s=0;s<splits;++s) mx=fmaxf(mx,src[s*130+128]);
    for (uint32_t s=0;s<splits;++s) {
        const float p=expf(src[s*130+128]-mx);
        total+=src[s*130+129]*p;
        #pragma unroll
        for (int c=0;c<4;++c) acc[c]+=src[s*130+lane+c*32]*p;
    }
    #pragma unroll
    for (int c=0;c<4;++c) out[row*128+lane+c*32]=acc[c]/total;
}
}

extern "C" cudaError_t infer_lilicorr_context_attention(
    const float* q, const float* ck, const float* cv, const float* bk, const float* bv,
    float* partial, float* out, uint32_t end, uint32_t retained, uint32_t rows,
    uint32_t qheads, uint32_t kvheads, uint32_t window, cudaStream_t stream) {
    const uint32_t splits=(retained+rows+255)/256;
    context_attention_partial<<<dim3(qheads,rows,splits),32,0,stream>>>(
        q,ck,cv,bk,bv,partial,end,retained,rows,qheads,kvheads,window,splits);
    context_attention_merge<<<qheads*rows,32,0,stream>>>(partial,out,splits);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_lilicorr_candidates(const float* x, uint32_t* ids, float* logp,
    uint32_t rows, uint32_t vocab, uint32_t k, cudaStream_t s) {
    candidates<<<rows,256,0,s>>>(x,ids,logp,vocab,k); return cudaGetLastError();
}
extern "C" cudaError_t infer_lilicorr_bias_activation(float* x, const float* b, uint32_t n,
    uint32_t w, bool silu, cudaStream_t s) {
    bias_activation<<<(n+255)/256,256,0,s>>>(x,b,n,w,silu); return cudaGetLastError();
}
extern "C" cudaError_t infer_lilicorr_features(const float* p, const float* g, const float* b,
    float* out, uint32_t n, uint32_t k, cudaStream_t s) {
    features<<<(n+255)/256,256,0,s>>>(p,g,b,out,n,k); return cudaGetLastError();
}
extern "C" cudaError_t infer_lilicorr_hidden(const float* t, const float* p, const float* f,
    const float* slot, const float* rank, float* out, uint32_t n, uint32_t w, uint32_t k, cudaStream_t s) {
    hidden<<<(n+255)/256,256,0,s>>>(t,p,f,slot,rank,out,n,w,k); return cudaGetLastError();
}
extern "C" cudaError_t infer_lilicorr_attention(const float* qkv, const float* rel, const float* same,
    float* out, uint32_t n, uint32_t w, uint32_t heads, uint32_t k, uint32_t slots, cudaStream_t s) {
    lattice_attention<<<n*heads,32,0,s>>>(qkv,rel,same,out,n,w,heads,k,slots); return cudaGetLastError();
}
extern "C" cudaError_t infer_lilicorr_cross(const float* x, const float* a, float* out,
    uint32_t n, uint32_t w, cudaStream_t s) {
    anchor_cross<<<(n+255)/256,256,0,s>>>(x,a,out,n,w); return cudaGetLastError();
}
extern "C" cudaError_t infer_lilicorr_factors(const float* x, const float* a, const float* cross,
    float* out, uint32_t n, uint32_t w, cudaStream_t s) {
    factors<<<(n+255)/256,256,0,s>>>(x,a,cross,out,n,w); return cudaGetLastError();
}
extern "C" cudaError_t infer_lilicorr_normalize(float* x, uint32_t rows, uint32_t w, float eps, cudaStream_t s) {
    normalize<<<rows,256,0,s>>>(x,w,eps); return cudaGetLastError();
}
extern "C" cudaError_t infer_lilicorr_edges(const float* out, const float* in, const float* a,
    float* scores, uint32_t slots, uint32_t w, uint32_t k, float scale, cudaStream_t s) {
    edges<<<slots*k*k,256,0,s>>>(out,in,a,scores,w,k,scale); return cudaGetLastError();
}
extern "C" cudaError_t infer_lilicorr_capture(const float* x, float* out, uint32_t n,
    uint32_t w, uint32_t taps, uint32_t tap, cudaStream_t s) {
    capture<<<(n+255)/256,256,0,s>>>(x,out,n,w,taps,tap); return cudaGetLastError();
}
