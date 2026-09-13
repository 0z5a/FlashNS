// Standalone activation oracle kernels, NOT fused GEMM training kernels.
// This CUDA translation unit has NOT been compiled or GPU-tested in this session.
// Layout: [sample, jet, channel], channel contiguous. Non-aliasing buffers.
#include <cuda_runtime.h>
#include <cstddef>
#include <climits>
#include <cstdint>
#include "jet_primitives.cuh"

namespace {
template<int Dim> struct Ops;
template<> struct Ops<2> {
  static constexpr int Q=10;
  __device__ static void fwd(const double*z,double*h) {flashns::tanh_fwd_2d3(z,h);}
  __device__ static void vjp(const double*h,const double*b,double*z) {flashns::tanh_vjp_2d3(h,b,z);}
};
template<> struct Ops<3> {
  static constexpr int Q=20;
  __device__ static void fwd(const double*z,double*h) {flashns::tanh_fwd_3d3(z,h);}
  __device__ static void vjp(const double*h,const double*b,double*z) {flashns::tanh_vjp_3d3(h,b,z);}
};

template<int Dim, bool Backward>
__global__ void activation(const double* __restrict__ input,
                           const double* __restrict__ adjoint,
                           double* __restrict__ output,
                           std::size_t B, std::size_t C) {
  constexpr int Q=Ops<Dim>::Q;
  for (std::size_t n=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
       n<B*C; n+=std::size_t(blockDim.x)*gridDim.x) {
    const std::size_t b=n/C,c=n%C;
    double a[Q], out[Q];
    #pragma unroll
    for(int q=0;q<Q;++q) a[q]=input[(b*Q+q)*C+c];
    if constexpr(Backward) {
      double ba[Q];
      #pragma unroll
      for(int q=0;q<Q;++q) ba[q]=adjoint[(b*Q+q)*C+c];
      Ops<Dim>::vjp(a,ba,out);
    } else {
      Ops<Dim>::fwd(a,out);
    }
    #pragma unroll
    for(int q=0;q<Q;++q) output[(b*Q+q)*C+c]=out[q];
  }
}
}

// Runtime compiles device code for Dim=2 and Dim=3, p=3 only.
// backwards=false: input=z, adjoint=null, output=h.
// backwards=true: input=h, adjoint=bar_h, output=bar_z.
extern "C" cudaError_t flashns_launch_jet_activation(
    int dim, bool backwards, const double* input, const double* adjoint,
    double* output, std::size_t B, std::size_t C, cudaStream_t stream) {
  if(dim!=2 && dim!=3) return cudaErrorInvalidValue;
  if(B==0 || C==0) return cudaSuccess;
  if(!input || !output || (backwards && !adjoint)) return cudaErrorInvalidValue;
  if(B>SIZE_MAX/C || B*C>SIZE_MAX/(20*sizeof(double))) return cudaErrorInvalidValue;
  constexpr unsigned threads=128;
  const std::size_t nblocks=(B*C+threads-1)/threads;
  const unsigned blocks=unsigned(nblocks<65535?nblocks:65535);
  if(dim==2) {
    if(backwards) activation<2,true><<<blocks,threads,0,stream>>>(input,adjoint,output,B,C);
    else activation<2,false><<<blocks,threads,0,stream>>>(input,adjoint,output,B,C);
  } else {
    if(backwards) activation<3,true><<<blocks,threads,0,stream>>>(input,adjoint,output,B,C);
    else activation<3,false><<<blocks,threads,0,stream>>>(input,adjoint,output,B,C);
  }
  // Caller must synchronize/check asynchronous errors on its stream.
  return cudaGetLastError();
}
