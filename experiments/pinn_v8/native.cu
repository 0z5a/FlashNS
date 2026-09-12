#include <cuda_runtime.h>
#include <cstdint>
#include "../cuda_jet_h100/stable_jet.cuh"
#include "residual_generated.cuh"
#include "coordinate_jet.cuh"
#include "coordinate_wgrad.cuh"
#include "coordinate_activation_wgrad.cuh"

template<int D, bool BACKWARD>
__global__ void packed_activation(const double* input, const double* auxiliary,
    const double* seed, double* output, double* output_auxiliary,
    int64_t full_points, int64_t value_points, int channels) {
  constexpr int Q = D == 2 ? 10 : 20;
  const int64_t index = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (index >= (full_points+value_points)*channels) return;
  const int64_t point = index/channels;
  const int channel = int(index%channels);
  if (point < full_points) {
    const int64_t offset = point*Q*channels+channel;
    double h[Q], result[Q], a1;
    #pragma unroll
    for (int j=0;j<Q;++j) h[j]=input[offset+int64_t(j)*channels];
    if constexpr (BACKWARD) {
      double bar[Q];
      #pragma unroll
      for (int j=0;j<Q;++j) bar[j]=seed[offset+int64_t(j)*channels];
      if constexpr (D == 2) flashns_stable::tanh_vjp_2d3(h,bar,auxiliary[index],result);
      else flashns_stable::tanh_vjp_3d3(h,bar,auxiliary[index],result);
    } else {
      if constexpr (D == 2) flashns_stable::tanh_fwd_2d3(h,result,&a1);
      else flashns_stable::tanh_fwd_3d3(h,result,&a1);
      output_auxiliary[index]=a1;
    }
    #pragma unroll
    for (int j=0;j<Q;++j) output[offset+int64_t(j)*channels]=result[j];
  } else {
    const int64_t offset=(full_points*Q+point-full_points)*channels+channel;
    if constexpr (BACKWARD) output[offset]=seed[offset]*auxiliary[index];
    else {
      const double z=input[offset], r=::exp(-::fabs(z)), v=(2.0*r)/(1.0+r*r);
      output[offset]=::tanh(z);
      output_auxiliary[index]=v*v;
    }
  }
}

extern "C" int flashns_v8_activation(int dimension, int backward,
    const double* input, const double* auxiliary, const double* seed,
    double* output, double* output_auxiliary, int64_t full_points,
    int64_t value_points, int channels, cudaStream_t stream) {
  if ((dimension!=2 && dimension!=3) || full_points<0 || value_points<0 || channels<1)
    return int(cudaErrorInvalidValue);
  const int64_t count=(full_points+value_points)*channels;
  if (count==0) return int(cudaSuccess);
  if ((count+127)/128 > 2147483647LL) return int(cudaErrorInvalidConfiguration);
  const unsigned blocks=unsigned((count+127)/128);
  if (dimension==2 && backward) packed_activation<2,true><<<blocks,128,0,stream>>>(input,auxiliary,seed,output,output_auxiliary,full_points,value_points,channels);
  else if (dimension==2) packed_activation<2,false><<<blocks,128,0,stream>>>(input,auxiliary,seed,output,output_auxiliary,full_points,value_points,channels);
  else if (backward) packed_activation<3,true><<<blocks,128,0,stream>>>(input,auxiliary,seed,output,output_auxiliary,full_points,value_points,channels);
  else packed_activation<3,false><<<blocks,128,0,stream>>>(input,auxiliary,seed,output,output_auxiliary,full_points,value_points,channels);
  return int(cudaGetLastError());
}

// Physical Cartesian coordinates: zero order is x, first derivatives are the
// identity, and all higher input coefficients vanish. Coefficient order is the
// full Taylor basis revision 1 (unit indices appear in reverse axis order).
template<int D, bool ACTIVATION>
__global__ void packed_coordinate_affine(const double* coordinates,
    const double* weight, const double* bias, double* output, double* output_auxiliary,
    int64_t full_points, int64_t value_points, int channels) {
  constexpr int Q = D == 2 ? 10 : 20;
  const int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (index >= (full_points+value_points)*channels) return;
  const int64_t point=index/channels;
  const int channel=int(index%channels);
  const double* w=weight+int64_t(channel)*D;
  const double* x=coordinates+point*D;
  const int64_t row=point<full_points ? point*Q : full_points*Q+point-full_points;
  double result[Q];
  if constexpr (ACTIVATION) {
    double a1;
    flashns_coordinate::affine_activation<D>(x,w,bias[channel],point<full_points,result,&a1);
    output_auxiliary[index]=a1;
  } else {
    flashns_coordinate::affine_jet<D>(x,w,bias[channel],point<full_points,result);
  }
  const int count=point<full_points ? Q : 1;
  #pragma unroll
  for (int j=0;j<count;++j) output[(row+j)*channels+channel]=result[j];
}

template<bool ACTIVATION>
int launch_coordinate_affine(int dimension,
    const double* coordinates, const double* weight, const double* bias,
    double* output, double* output_auxiliary, int64_t full_points, int64_t value_points,
    int channels, cudaStream_t stream) {
  if ((dimension!=2 && dimension!=3) || full_points<0 || value_points<0 || channels<1)
    return int(cudaErrorInvalidValue);
  const int64_t q=dimension==2 ? 10 : 20;
  if (full_points>(INT64_MAX-value_points)/q) return int(cudaErrorInvalidValue);
  const int64_t rows=full_points*q+value_points;
  const int64_t widest=channels>dimension ? channels : dimension;
  if (rows>INT64_MAX/int64_t(sizeof(double))/widest) return int(cudaErrorInvalidValue);
  const int64_t count=(full_points+value_points)*channels;
  if (count==0) return int(cudaSuccess);
  if (!coordinates || !weight || !bias || !output || (ACTIVATION && !output_auxiliary))
    return int(cudaErrorInvalidValue);
  const int64_t blocks=count/128+(count%128!=0);
  if (blocks>2147483647LL) return int(cudaErrorInvalidConfiguration);
  if (dimension==2) packed_coordinate_affine<2,ACTIVATION><<<unsigned(blocks),128,0,stream>>>(coordinates,weight,bias,output,output_auxiliary,full_points,value_points,channels);
  else packed_coordinate_affine<3,ACTIVATION><<<unsigned(blocks),128,0,stream>>>(coordinates,weight,bias,output,output_auxiliary,full_points,value_points,channels);
  return int(cudaGetLastError());
}

extern "C" int flashns_v8_coordinate_affine(int dimension,
    const double* coordinates, const double* weight, const double* bias,
    double* output, int64_t full_points, int64_t value_points,
    int channels, cudaStream_t stream) {
  return launch_coordinate_affine<false>(dimension,coordinates,weight,bias,output,nullptr,
                                         full_points,value_points,channels,stream);
}

extern "C" int flashns_v8_coordinate_affine_activation(int dimension,
    const double* coordinates, const double* weight, const double* bias,
    double* hidden, double* auxiliary, int64_t full_points, int64_t value_points,
    int channels, cudaStream_t stream) {
  return launch_coordinate_affine<true>(dimension,coordinates,weight,bias,hidden,auxiliary,
                                        full_points,value_points,channels,stream);
}

// Both candidates use this exact scalar arithmetic. The target build retains
// --fmad=false; U3 differs from F3 only at the barH materialization boundary.
__device__ __forceinline__ double packed_dot3(const double* d,
    double w0, double w1, double w2) {
  return (d[0]*w0+d[1]*w1)+d[2]*w2;
}

template<int D, bool FUSED>
__global__ void packed_tail3(const double* d, const double* weight,
    const double* hidden, const double* auxiliary, double* output,
    int64_t full_points, int64_t value_points, int channels) {
  constexpr int Q = D == 2 ? 10 : 20;
  const int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (index >= (full_points+value_points)*channels) return;
  const int64_t point=index/channels;
  const int channel=int(index%channels);
  const double w0=weight[channel], w1=weight[int64_t(channels)+channel],
               w2=weight[int64_t(2)*channels+channel];
  if (point < full_points) {
    const int64_t row=point*Q;
    double bar[Q];
    #pragma unroll
    for (int j=0;j<Q;++j) bar[j]=packed_dot3(d+(row+j)*3,w0,w1,w2);
    if constexpr (FUSED) {
      double h[Q], result[Q];
      #pragma unroll
      for (int j=0;j<Q;++j) h[j]=hidden[(row+j)*channels+channel];
      if constexpr (D == 2) flashns_stable::tanh_vjp_2d3(h,bar,auxiliary[index],result);
      else flashns_stable::tanh_vjp_3d3(h,bar,auxiliary[index],result);
      #pragma unroll
      for (int j=0;j<Q;++j) output[(row+j)*channels+channel]=result[j];
    } else {
      #pragma unroll
      for (int j=0;j<Q;++j) output[(row+j)*channels+channel]=bar[j];
    }
  } else {
    const int64_t row=full_points*Q+point-full_points;
    const double bar=packed_dot3(d+row*3,w0,w1,w2);
    if constexpr (FUSED) output[row*channels+channel]=bar*auxiliary[index];
    else output[row*channels+channel]=bar;
  }
}

extern "C" int flashns_v8_tail3(int dimension, int fused,
    const double* d, const double* weight, const double* hidden,
    const double* auxiliary, double* output, int64_t full_points,
    int64_t value_points, int channels, cudaStream_t stream) {
  if ((dimension!=2 && dimension!=3) || (fused!=0 && fused!=1) ||
      full_points<0 || value_points<0 || channels<1)
    return int(cudaErrorInvalidValue);
  const int64_t q=dimension == 2 ? 10 : 20;
  if (full_points > (INT64_MAX-value_points)/q)
    return int(cudaErrorInvalidValue);
  const int64_t rows=full_points*q+value_points;
  const int64_t widest=channels > 3 ? channels : 3;
  if (rows > INT64_MAX/int64_t(sizeof(double))/widest)
    return int(cudaErrorInvalidValue);
  const int64_t count=(full_points+value_points)*channels;
  if (count==0) return int(cudaSuccess);
  if (!d || !weight || !output || (fused && (!hidden || !auxiliary)))
    return int(cudaErrorInvalidValue);
  const int64_t block_count=count/128+(count%128!=0);
  if (block_count > 2147483647LL) return int(cudaErrorInvalidConfiguration);
  const unsigned blocks=unsigned(block_count);
  if (dimension==2 && fused) packed_tail3<2,true><<<blocks,128,0,stream>>>(d,weight,hidden,auxiliary,output,full_points,value_points,channels);
  else if (dimension==2) packed_tail3<2,false><<<blocks,128,0,stream>>>(d,weight,hidden,auxiliary,output,full_points,value_points,channels);
  else if (fused) packed_tail3<3,true><<<blocks,128,0,stream>>>(d,weight,hidden,auxiliary,output,full_points,value_points,channels);
  else packed_tail3<3,false><<<blocks,128,0,stream>>>(d,weight,hidden,auxiliary,output,full_points,value_points,channels);
  return int(cudaGetLastError());
}

template<int D>
__global__ void coordinate_wgrad_partials(const double* derivative, const double* coordinates,
    double* partials, int64_t full_points, int64_t value_points, int channels, int64_t tiles) {
  const int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (index>=tiles*channels) return;
  double result[D+1];
  flashns_coordinate_wgrad::tile<D>(derivative,coordinates,full_points,value_points,
                                   channels,index/channels,int(index%channels),result);
  for (int axis=0;axis<=D;++axis) partials[index*(D+1)+axis]=result[axis];
}

template<int D>
__global__ void coordinate_wgrad_finish(const double* partials, double* weight_gradient,
    double* bias_gradient, int channels, int64_t tiles) {
  const int64_t channel=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (channel>=channels) return;
  flashns_coordinate_wgrad::finish<D>(partials,tiles,channels,int(channel),weight_gradient,bias_gradient);
}

template<int D>
__global__ void coordinate_activation_wgrad_partials(const double* hidden, const double* auxiliary,
    const double* seed, const double* coordinates, double* partials,
    int64_t full_points, int64_t value_points, int channels, int64_t tiles) {
  const int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (index>=tiles*channels) return;
  double result[D+1];
  flashns_coordinate_wgrad::tile_vjp<D>(hidden,auxiliary,seed,coordinates,full_points,value_points,
                                       channels,index/channels,int(index%channels),result);
  for (int axis=0;axis<=D;++axis) partials[index*(D+1)+axis]=result[axis];
}

template<bool FUSED>
int launch_coordinate_wgrad(int dimension,
    const double* derivative_or_hidden, const double* auxiliary, const double* seed,
    const double* coordinates, double* partials,
    double* weight_gradient, double* bias_gradient, int64_t full_points,
    int64_t value_points, int channels, cudaStream_t stream) {
  if ((dimension!=2 && dimension!=3) || full_points<0 || value_points<0 || channels<1)
    return int(cudaErrorInvalidValue);
  const int64_t q=dimension==2 ? 10 : 20;
  if (full_points>(INT64_MAX-value_points)/q) return int(cudaErrorInvalidValue);
  const int64_t rows=full_points*q+value_points;
  const int64_t widest=channels>dimension ? channels : dimension;
  if (rows>INT64_MAX/int64_t(sizeof(double))/widest) return int(cudaErrorInvalidValue);
  const int64_t points=full_points+value_points;
  const int64_t tiles=points/flashns_coordinate_wgrad::point_tile+
                      (points%flashns_coordinate_wgrad::point_tile!=0);
  if (tiles>INT64_MAX/int64_t(sizeof(double))/(dimension+1)/channels)
    return int(cudaErrorInvalidValue);
  if (!weight_gradient || !bias_gradient || (points && (!derivative_or_hidden || !coordinates || !partials
      || (FUSED && (!auxiliary || !seed)))))
    return int(cudaErrorInvalidValue);
  const int64_t count=tiles*channels;
  const int64_t blocks=count/128+(count%128!=0);
  if (blocks>2147483647LL) return int(cudaErrorInvalidConfiguration);
  if (blocks) {
    if constexpr (FUSED) {
      if (dimension==2) coordinate_activation_wgrad_partials<2><<<unsigned(blocks),128,0,stream>>>(derivative_or_hidden,auxiliary,seed,coordinates,partials,full_points,value_points,channels,tiles);
      else coordinate_activation_wgrad_partials<3><<<unsigned(blocks),128,0,stream>>>(derivative_or_hidden,auxiliary,seed,coordinates,partials,full_points,value_points,channels,tiles);
    } else {
      if (dimension==2) coordinate_wgrad_partials<2><<<unsigned(blocks),128,0,stream>>>(derivative_or_hidden,coordinates,partials,full_points,value_points,channels,tiles);
      else coordinate_wgrad_partials<3><<<unsigned(blocks),128,0,stream>>>(derivative_or_hidden,coordinates,partials,full_points,value_points,channels,tiles);
    }
    const cudaError_t status=cudaGetLastError();
    if (status!=cudaSuccess) return int(status);
  }
  // Even an empty segment owns nonempty parameter gradients and writes zeros.
  const unsigned finish_blocks=unsigned((int64_t(channels)+127)/128);
  if (dimension==2) coordinate_wgrad_finish<2><<<finish_blocks,128,0,stream>>>(partials,weight_gradient,bias_gradient,channels,tiles);
  else coordinate_wgrad_finish<3><<<finish_blocks,128,0,stream>>>(partials,weight_gradient,bias_gradient,channels,tiles);
  return int(cudaGetLastError());
}

extern "C" int flashns_v8_coordinate_wgrad(int dimension,
    const double* derivative, const double* coordinates, double* partials,
    double* weight_gradient, double* bias_gradient, int64_t full_points,
    int64_t value_points, int channels, cudaStream_t stream) {
  return launch_coordinate_wgrad<false>(dimension,derivative,nullptr,nullptr,coordinates,partials,
                                        weight_gradient,bias_gradient,full_points,value_points,channels,stream);
}

extern "C" int flashns_v8_coordinate_activation_wgrad(int dimension,
    const double* hidden, const double* auxiliary, const double* seed,
    const double* coordinates, double* partials, double* weight_gradient,
    double* bias_gradient, int64_t full_points, int64_t value_points,
    int channels, cudaStream_t stream) {
  return launch_coordinate_wgrad<true>(dimension,hidden,auxiliary,seed,coordinates,partials,
                                       weight_gradient,bias_gradient,full_points,value_points,channels,stream);
}

__global__ void loss_seed_kernel(const double* jets, const double* boundary,
    const double* pde_weights, const double* boundary_weights, const double* target,
    double* interior_seed, double* boundary_seed, double* partial_loss,
    int64_t ni, int64_t nb, int64_t boundary_stride, double nu, double gamma, double bc_scale) {
  __shared__ double reduction[128];
  const int64_t point=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  double loss=0.0;
  if (point<ni) {
    double z[30], dz[30];
    #pragma unroll
    for (int k=0;k<30;++k) z[k]=jets[point*30+k];
    loss=flashns_v8::residual_seed(z,pde_weights[point],nu,gamma,dz);
    #pragma unroll
    for (int k=0;k<30;++k) interior_seed[point*30+k]=dz[k];
  } else if (point<ni+nb) {
    const int64_t b=point-ni;
    #pragma unroll
    for (int k=0;k<3;++k) {
      const double error=boundary[b*boundary_stride+k]-target[b*3+k];
      const double weight=boundary_weights[b*3+k];
      loss+=bc_scale*weight*error*error;
      boundary_seed[b*3+k]=2.0*bc_scale*weight*error;
    }
  }
  reduction[threadIdx.x]=loss;
  __syncthreads();
  for (int step=64;step>0;step>>=1) {
    if (threadIdx.x<step) reduction[threadIdx.x]+=reduction[threadIdx.x+step];
    __syncthreads();
  }
  if (threadIdx.x==0) partial_loss[blockIdx.x]=reduction[0];
}

extern "C" int flashns_v8_loss_seed(const double* jets, const double* boundary,
    const double* pde_weights, const double* boundary_weights, const double* target,
    double* interior_seed, double* boundary_seed, double* partial_loss,
    int64_t ni, int64_t nb, int64_t boundary_stride,
    double nu, double gamma, double bc_scale, cudaStream_t stream) {
  if (ni<0 || nb<0 || boundary_stride<3) return int(cudaErrorInvalidValue);
  if (ni+nb==0) return int(cudaSuccess);
  if ((ni+nb+127)/128 > 2147483647LL) return int(cudaErrorInvalidConfiguration);
  const unsigned blocks=unsigned((ni+nb+127)/128);
  loss_seed_kernel<<<blocks,128,0,stream>>>(jets,boundary,pde_weights,boundary_weights,target,
      interior_seed,boundary_seed,partial_loss,ni,nb,boundary_stride,nu,gamma,bc_scale);
  return int(cudaGetLastError());
}
