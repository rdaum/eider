// Dory nGPT operations. BF16 activations; reductions accumulate in F32.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#if EIDER_FLASH_ATTENTION
extern "C" int infer_dory_flash_decode(void*,void*,void*,void*,float*,int*,int*,int,int,int,int,cudaStream_t);
extern "C" int infer_dory_flash_attention(void*,void*,void*,void*,float*,int*,int,int,int,int,int,cudaStream_t);
#endif
namespace {
__device__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ float warp_sum(float x) {
    for (int d=16; d; d>>=1) x += __shfl_down_sync(0xffffffff, x, d);
    return __shfl_sync(0xffffffff, x, 0);
}
__device__ float sum(float x) {
    __shared__ float s[8];
    x=warp_sum(x);
    if (!(threadIdx.x&31)) s[threadIdx.x/32]=x;
    __syncthreads();
    x=warp_sum(threadIdx.x<8 ? s[threadIdx.x] : 0.f);
    if (!threadIdx.x) s[0]=x;
    __syncthreads();
    const float result=s[0];
    __syncthreads();
    return result;
}
// mode: plain L2, BF16 add then L2, normalized interpolation, gated attention,
// scaled SwiGLU. Intermediates are rounded at the reference model's boundaries.
template<int Width,int Mode>
__global__ void norm(const __nv_bfloat16* a,const __nv_bfloat16* b,const float* p,const float* q,
                     __nv_bfloat16* o,uint32_t width) {
    const uint32_t w=Width ? Width : width;
    constexpr uint32_t mode=Mode;
    const size_t base=size_t(blockIdx.x)*w;
    float na=0, nb=0;
    if (mode==2) {
        #pragma unroll
        for(uint32_t j=threadIdx.x;j<w;j+=256) { const float av=a[base+j],bv=b[base+j]; na+=av*av; nb+=bv*bv; }
        na=rsqrtf(sum(na)+1e-6f); nb=rsqrtf(sum(nb)+1e-6f);
    }
    // Maximum supported width is 8192; keep each element live across reduction.
    float v[32]; float total=0;
    #pragma unroll
    for(uint32_t j=threadIdx.x,i=0;j<w;j+=256,++i) {
        float x=a[base+j];
        if(mode==1) x=bf(x+float(b[base+j]));
        if(mode==2) { const float ya=bf(x*na),yb=bf(float(b[base+j])*nb); x=ya+p[j]*(yb-ya); }
        if(mode==3) x=bf(bf(x)/(1.f+__expf(-float(b[base+j])*p[j])));
        if(mode==4) { const float g=bf(x*p[j]),u=bf(float(b[base+j])*q[j]); x=bf((g/(1.f+__expf(-g)))*u); }
        v[i]=x; total+=x*x;
    }
    const float inv=rsqrtf(sum(total)+1e-6f);
    #pragma unroll
    for(uint32_t j=threadIdx.x,i=0;j<w;j+=256,++i) o[base+j]=bf(v[i]*inv);
}
__global__ void rope_norm(const __nv_bfloat16* x,const float* scale,__nv_bfloat16* out,const uint32_t* pos,
                          uint32_t heads,float theta) {
    const uint32_t head=blockIdx.x%heads, row=blockIdx.x/heads, d=threadIdx.x;
    const size_t base=size_t(blockIdx.x)*256;
    const uint32_t h=d%128;
    const float angle=float(*pos+row)*powf(theta,-float(h)/128.f);
    float sn,cs; sincosf(angle,&sn,&cs);
    const float other=x[base+(d<128?d+128:d-128)];
    const float rotated=bf(float(x[base+d])*cs+(d<128?-other:other)*sn);
    const float inv=rsqrtf(sum(rotated*rotated)+1e-6f);
    float y=bf(rotated*inv);
    if(scale) y=bf(y*scale[head*256+d]);
    out[base+d]=y;
}
__global__ void append(const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* ck,__nv_bfloat16* cv,const uint32_t* pos,uint32_t n,uint32_t width,uint32_t capacity,int* length,int* leftpad,uint32_t window) {
    const uint32_t i=blockIdx.x*256+threadIdx.x;
    if(i==0 && length) { *length=*pos+n/width; if(leftpad) *leftpad=window && *length>window?*length-window:0; }
    if(i<n) {const size_t dst=size_t(*pos)*width+i;if(dst<size_t(capacity)*width){ck[dst]=k[i];cv[dst]=v[i];}}
}
// One CTA per query head and key tile. Warps cooperate on QK and columns
// cooperate on PV, exposing parallel work even at single-token decode.
__global__ void attention(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,float* partial,
                          const uint32_t* pos,uint32_t qh,uint32_t kh,uint32_t window,uint32_t splits,uint32_t capacity) {
    const uint32_t lane=threadIdx.x&31,warp=threadIdx.x/32,head=blockIdx.x,row=blockIdx.y,part=blockIdx.z;
    const uint32_t end=uint32_t(min(size_t(*pos)+row+1,size_t(capacity))),begin=window&&end>window?end-window:0;
    const uint32_t first=begin+part*256,last=min(end,first+256);
    const size_t dst=((size_t(row)*qh+head)*splits+part)*258;
    if(first>=end) {partial[dst+threadIdx.x]=0;if(!threadIdx.x){partial[dst+256]=-INFINITY;partial[dst+257]=0;}return;}
    __shared__ float scores[256];
    float query[8];
    for(int i=0;i<8;++i) query[i]=q[(size_t(row)*qh+head)*256+lane+32*i];
    for(uint32_t t=first+warp;t<last;t+=8) {
        const size_t base=(size_t(t)*kh+head/(qh/kh))*256;
        float dot=0;
        for(int i=0;i<8;++i) dot+=query[i]*__bfloat162float(k[base+lane+32*i]);
        dot=warp_sum(dot)*(1.f/16.f);
        if(!lane)scores[t-first]=dot;
    }
    __syncthreads();
    float mx=threadIdx.x<last-first?scores[threadIdx.x]:-INFINITY;
    // Eight warp maxima, followed by a warp reduction.
    __shared__ float maxima[8];
    for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,d));
    if(!lane)maxima[warp]=mx;
    __syncthreads();
    mx=threadIdx.x<8?maxima[threadIdx.x]:-INFINITY;
    for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,d));
    if(!threadIdx.x)maxima[0]=mx;
    __syncthreads();mx=maxima[0];
    const float weight=threadIdx.x<last-first?expf(scores[threadIdx.x]-mx):0;
    scores[threadIdx.x]=weight;
    const float den=sum(weight);
    float acc=0;
    for(uint32_t t=first;t<last;++t)acc+=scores[t-first]*__bfloat162float(v[(size_t(t)*kh+head/(qh/kh))*256+threadIdx.x]);
    partial[dst+threadIdx.x]=acc;
    if(!threadIdx.x){partial[dst+256]=mx;partial[dst+257]=den;}
}
__global__ void merge(const float* p,__nv_bfloat16* o,uint32_t splits) {
    const size_t base=size_t(blockIdx.x)*splits*258;
    float mx=-INFINITY;
    for(uint32_t s=0;s<splits;++s) mx=fmaxf(mx,p[base+s*258+256]);
    float den=0,value=0;
    for(uint32_t s=0;s<splits;++s){const float weight=expf(p[base+s*258+256]-mx);den+=weight*p[base+s*258+257];value+=weight*p[base+s*258+threadIdx.x];}
    o[size_t(blockIdx.x)*256+threadIdx.x]=bf(value/den);
}
// The packed tiles are Sm121W4A16HostWeight's [output16, K16] layout.
// Tensor-wide ModelOpt scaling happens after F32 accumulation, before BF16 rounding.
__device__ uint16_t scale_bf16(uint8_t code) {
    return code < 8 ? __bfloat16_as_ushort(__float2bfloat16_rn(code * 0x1p-9f))
        : uint16_t((unsigned(code) << 4) + 0x3c00u);
}
__device__ uint32_t weight_pair(uint8_t packed, uint16_t scale) {
    const unsigned selector = (packed & 7) | ((packed >> 4 & 7) << 8);
    const unsigned lo = __byte_perm(0xc0800000u, 0xc0804000u, selector) & 0x00ff00ffu;
    const unsigned hi = (__byte_perm(0x3f3f3f00u, 0x40404040u, selector) & 0x00ff00ffu) << 8;
    const unsigned sign = (unsigned(packed & 8) << 12) | (unsigned(packed & 128) << 24);
    const unsigned scales = unsigned(scale) | (unsigned(scale) << 16);
    unsigned result;
    asm("mul.bf16x2 %0, %1, %2;" : "=r"(result) : "r"(lo | hi | sign), "r"(scales));
    return result;
}
__device__ void linear_tile(const __nv_bfloat16* input, const uint8_t* weights, const uint8_t* scales,
                       __nv_bfloat16* output, uint32_t out, uint32_t cols, float global, uint32_t output_tile) {
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const unsigned row0 = lane / 4, row1 = row0 + 8, pair = lane % 4;
    const unsigned tiles = cols / 16;
    input += size_t(blockIdx.y) * cols;
    float acc[4] = {};
    for (unsigned k = warp; k < tiles; k += 16) {
        const size_t tile = size_t(output_tile) * tiles + k;
        const uint8_t* weight = weights + tile * 128;
        const uint16_t s0 = scale_bf16(scales[tile * 16 + row0]);
        const uint16_t s1 = scale_bf16(scales[tile * 16 + row1]);
        uint32_t a[4] = {weight_pair(weight[row0 * 8 + pair], s0),
                      weight_pair(weight[row1 * 8 + pair], s1),
                      weight_pair(weight[row0 * 8 + pair + 4], s0),
                      weight_pair(weight[row1 * 8 + pair + 4], s1)};
        uint32_t b0 = 0, b1 = 0;
        if (lane < 4) {
            const uint32_t* x = reinterpret_cast<const uint32_t*>(input + k * 16 + pair * 2);
            b0 = x[0];
            b1 = x[4];
        }
        b0 = __shfl_sync(0xffffffff, b0, pair);
        b1 = __shfl_sync(0xffffffff, b1, pair);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
    __shared__ float partial[16][16];
    if (pair == 0) { partial[warp][row0] = acc[0]; partial[warp][row1] = acc[2]; }
    __syncthreads();
    if (threadIdx.x < 16) {
        float value = 0;
#pragma unroll
        for (unsigned w = 0; w < 16; ++w) value += partial[w][threadIdx.x];
        output[size_t(blockIdx.y) * out + output_tile * 16 + threadIdx.x] = bf(value * global);
    }
}
struct Projection {
    const uint8_t* weight;
    const uint8_t* scale;
    __nv_bfloat16* output;
    uint32_t width;
    float global;
};
struct Projections { Projection items[4]; };
template<int Count>
__global__ void linear_group(const __nv_bfloat16* input, const __grid_constant__ Projections projections,
                            uint32_t cols) {
    uint32_t tile=blockIdx.x;
    #pragma unroll
    for(uint32_t i=0;i<Count;++i) {
        const Projection& p=projections.items[i];
        if(tile<p.width/16) {
            linear_tile(input,p.weight,p.scale,p.output,p.width,cols,p.global,tile);
            return;
        }
        tile-=p.width/16;
    }
}
__global__ void embedding(const __nv_bfloat16* table,const uint32_t* ids,__nv_bfloat16* out,
                           uint32_t vocab,uint32_t width,uint32_t count) {
    const uint32_t i=blockIdx.x*256+threadIdx.x;
    if(i<count) { const uint32_t id=ids[i/width]; out[i]=id<vocab?table[size_t(id)*width+i%width]:__float2bfloat16_rn(0.f); }
}
__global__ void logits(float* x,const float* scales,uint32_t n) {
    const uint32_t i=blockIdx.x*256+threadIdx.x;
    if(i<n)x[i]=bf(x[i])*scales[i];
}
}
// Fixed model widths keep the temporary vector in registers instead of thread-local memory.
template<int Width>
int launch_norm(const __nv_bfloat16* a,const __nv_bfloat16* b,const float* p,const float* q,
    __nv_bfloat16* o,uint32_t rows,uint32_t w,uint32_t mode,cudaStream_t s) {
    switch(mode) {
        case 0: norm<Width,0><<<rows,256,0,s>>>(a,b,p,q,o,w); break;
        case 1: norm<Width,1><<<rows,256,0,s>>>(a,b,p,q,o,w); break;
        case 2: norm<Width,2><<<rows,256,0,s>>>(a,b,p,q,o,w); break;
        case 3: norm<Width,3><<<rows,256,0,s>>>(a,b,p,q,o,w); break;
        case 4: norm<Width,4><<<rows,256,0,s>>>(a,b,p,q,o,w); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
extern "C" int infer_dory_norm(const __nv_bfloat16* a,const __nv_bfloat16* b,const float* p,const float* q,
    __nv_bfloat16* o,uint32_t rows,uint32_t w,uint32_t mode,cudaStream_t s) {
    switch(w) {
        case 2560: return launch_norm<2560>(a,b,p,q,o,rows,w,mode,s);
        case 4096: return launch_norm<4096>(a,b,p,q,o,rows,w,mode,s);
        case 7168: return launch_norm<7168>(a,b,p,q,o,rows,w,mode,s);
        default: return launch_norm<0>(a,b,p,q,o,rows,w,mode,s);
    }
}
extern "C" int infer_dory_rope_norm(const __nv_bfloat16* x,const float* scale,__nv_bfloat16* o,const uint32_t* pos,uint32_t rows,uint32_t heads,float theta,cudaStream_t s){rope_norm<<<rows*heads,256,0,s>>>(x,scale,o,pos,heads,theta);return cudaGetLastError();}
extern "C" int infer_dory_attention(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* ck,__nv_bfloat16* cv,float* p,__nv_bfloat16* o,const uint32_t* pos,uint32_t rows,uint32_t qh,uint32_t kh,uint32_t window,uint32_t splits,uint32_t capacity,cudaStream_t s){
    int* length = nullptr;
    int* leftpad = nullptr;
#if EIDER_FLASH_ATTENTION
    if(rows >= 16) length = reinterpret_cast<int*>(p + rows * qh);
    if(rows == 1 && qh<=64) {length=reinterpret_cast<int*>(p+8*qh*257+qh);leftpad=length+1;}
#endif
    append<<<(rows*kh*256+255)/256,256,0,s>>>(k,v,ck,cv,pos,rows*kh*256,kh*256,capacity,length,leftpad,window);
#if EIDER_FLASH_ATTENTION
    if (rows == 1 && qh<=64) return infer_dory_flash_decode(const_cast<__nv_bfloat16*>(q),ck,cv,o,p,length,leftpad,capacity,qh,kh,8,s);
    if (rows >= 16) {
        return infer_dory_flash_attention(const_cast<__nv_bfloat16*>(q),ck,cv,o,p,length,
            rows,capacity,qh,kh,window,s);
    }
#endif
    attention<<<dim3(qh,rows,splits),256,0,s>>>(q,ck,cv,p,pos,qh,kh,window,splits,capacity);
    merge<<<rows*qh,256,0,s>>>(p,o,splits);return cudaGetLastError();
}
extern "C" int infer_dory_logits(float* x,const float* s,uint32_t n,cudaStream_t stream){logits<<<(n+255)/256,256,0,stream>>>(x,s,n);return cudaGetLastError();}

extern "C" int infer_dory_linear(const __nv_bfloat16* x,const Projection* projections,
    uint32_t count,uint32_t rows,uint32_t cols,cudaStream_t stream) {
    Projections params{};
    uint32_t tiles=0;
    for(uint32_t i=0;i<count;++i) { params.items[i]=projections[i]; tiles+=projections[i].width/16; }
    switch(count) {
        case 1: linear_group<1><<<dim3(tiles,rows),512,0,stream>>>(x,params,cols);break;
        case 2: linear_group<2><<<dim3(tiles,rows),512,0,stream>>>(x,params,cols);break;
        case 3: linear_group<3><<<dim3(tiles,rows),512,0,stream>>>(x,params,cols);break;
        case 4: linear_group<4><<<dim3(tiles,rows),512,0,stream>>>(x,params,cols);break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

extern "C" int infer_dory_embedding(const __nv_bfloat16* table,const uint32_t* ids,__nv_bfloat16* out,
    uint32_t vocab,uint32_t width,uint32_t rows,cudaStream_t s) {
    embedding<<<(rows*width+255)/256,256,0,s>>>(table,ids,out,vocab,width,rows*width);
    return cudaGetLastError();
}
