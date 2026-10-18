// Dory nGPT operations. BF16 values use F32 storage; reductions accumulate in F32.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
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
__global__ void norm(const float* a,const float* b,const float* p,const float* q,
                     float* o,uint32_t w,uint32_t mode) {
    const size_t base=size_t(blockIdx.x)*w;
    float na=0, nb=0;
    if (mode==2) {
        for(uint32_t j=threadIdx.x;j<w;j+=256) { na+=a[base+j]*a[base+j]; nb+=b[base+j]*b[base+j]; }
        na=rsqrtf(sum(na)+1e-6f); nb=rsqrtf(sum(nb)+1e-6f);
    }
    // Maximum supported width is 8192; keep each element live across reduction.
    float v[32]; float total=0;
    for(uint32_t j=threadIdx.x,i=0;j<w;j+=256,++i) {
        float x=a[base+j];
        if(mode==1) x=bf(x+b[base+j]);
        if(mode==2) { const float ya=bf(x*na),yb=bf(b[base+j]*nb); x=ya+p[j]*(yb-ya); }
        if(mode==3) x=bf(bf(x)/(1.f+expf(-b[base+j]*p[j])));
        if(mode==4) { const float g=bf(x*p[j]),u=bf(b[base+j]*q[j]); x=bf((g/(1.f+expf(-g)))*u); }
        v[i]=x; total+=x*x;
    }
    const float inv=rsqrtf(sum(total)+1e-6f);
    for(uint32_t j=threadIdx.x,i=0;j<w;j+=256,++i) o[base+j]=bf(v[i]*inv);
}
__global__ void rope_norm(const float* x,const float* scale,float* out,const uint32_t* pos,
                          uint32_t heads,float theta) {
    const uint32_t head=blockIdx.x%heads, row=blockIdx.x/heads, d=threadIdx.x;
    const size_t base=size_t(blockIdx.x)*256;
    const uint32_t h=d%128;
    const float angle=float(*pos+row)*powf(theta,-float(h)/128.f);
    float sn,cs; sincosf(angle,&sn,&cs);
    const float other=x[base+(d<128?d+128:d-128)];
    const float rotated=bf(x[base+d]*cs+(d<128?-other:other)*sn);
    const float inv=rsqrtf(sum(rotated*rotated)+1e-6f);
    float y=bf(rotated*inv);
    if(scale) y=bf(y*scale[head*256+d]);
    out[base+d]=y;
}
__global__ void append(const float* k,const float* v,__nv_bfloat16* ck,__nv_bfloat16* cv,const uint32_t* pos,uint32_t n,uint32_t width,uint32_t capacity) {
    const uint32_t i=blockIdx.x*256+threadIdx.x;
    if(i<n) {const size_t dst=size_t(*pos)*width+i;if(dst<size_t(capacity)*width){ck[dst]=__float2bfloat16_rn(k[i]);cv[dst]=__float2bfloat16_rn(v[i]);}}
}
// One CTA per query head and key tile. Warps cooperate on QK and columns
// cooperate on PV, exposing parallel work even at single-token decode.
__global__ void attention(const float* q,const __nv_bfloat16* k,const __nv_bfloat16* v,float* partial,
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
__global__ void merge(const float* p,float* o,uint32_t splits) {
    const size_t base=size_t(blockIdx.x)*splits*258;
    float mx=-INFINITY;
    for(uint32_t s=0;s<splits;++s) mx=fmaxf(mx,p[base+s*258+256]);
    float den=0,value=0;
    for(uint32_t s=0;s<splits;++s){const float weight=expf(p[base+s*258+256]-mx);den+=weight*p[base+s*258+257];value+=weight*p[base+s*258+threadIdx.x];}
    o[size_t(blockIdx.x)*256+threadIdx.x]=bf(value/den);
}
__global__ void logits(float* x,const float* scales,uint32_t n) {
    const uint32_t i=blockIdx.x*256+threadIdx.x;
    if(i<n)x[i]=bf(x[i])*scales[i];
}
}
extern "C" int infer_dory_norm(const float* a,const float* b,const float* p,const float* q,float* o,uint32_t rows,uint32_t w,uint32_t mode,cudaStream_t s){norm<<<rows,256,0,s>>>(a,b,p,q,o,w,mode);return cudaGetLastError();}
extern "C" int infer_dory_rope_norm(const float* x,const float* scale,float* o,const uint32_t* pos,uint32_t rows,uint32_t heads,float theta,cudaStream_t s){rope_norm<<<rows*heads,256,0,s>>>(x,scale,o,pos,heads,theta);return cudaGetLastError();}
extern "C" int infer_dory_attention(const float* q,const float* k,const float* v,__nv_bfloat16* ck,__nv_bfloat16* cv,float* p,float* o,const uint32_t* pos,uint32_t rows,uint32_t qh,uint32_t kh,uint32_t window,uint32_t splits,uint32_t capacity,cudaStream_t s){
    append<<<(rows*kh*256+255)/256,256,0,s>>>(k,v,ck,cv,pos,rows*kh*256,kh*256,capacity);
    attention<<<dim3(qh,rows,splits),256,0,s>>>(q,ck,cv,p,pos,qh,kh,window,splits,capacity);
    merge<<<rows*qh,256,0,s>>>(p,o,splits);return cudaGetLastError();
}
extern "C" int infer_dory_logits(float* x,const float* s,uint32_t n,cudaStream_t stream){logits<<<(n+255)/256,256,0,stream>>>(x,s,n);return cudaGetLastError();}
