#pragma once
#include "../cuda_jet_h100/stable_jet.cuh"
#include "coordinate_wgrad.cuh"

#if defined(__CUDACC__)
#define FLASHNS_FIRST_VJP_INLINE __host__ __device__ __forceinline__
#else
#define FLASHNS_FIRST_VJP_INLINE inline
#endif

namespace flashns_coordinate_wgrad {
// The complete existing VJP is evaluated, including all high-order bar_Z
// entries and their nonfinite propagation through structural zero products.
template<int D>
FLASHNS_FIRST_VJP_INLINE void point_vjp(const double* hidden, const double* seed,
    int64_t stride, double auxiliary, bool full, double* derivative) {
  constexpr int Q = D == 2 ? 10 : 20;
  if (full) {
    double h[Q], bar[Q];
    for (int j=0;j<Q;++j) {
      h[j]=hidden[int64_t(j)*stride];
      bar[j]=seed[int64_t(j)*stride];
    }
    if constexpr (D==2) flashns_stable::tanh_vjp_2d3(h,bar,auxiliary,derivative);
    else flashns_stable::tanh_vjp_3d3(h,bar,auxiliary,derivative);
  } else {
    derivative[0]=seed[0]*auxiliary;
  }
}

template<int D>
FLASHNS_FIRST_VJP_INLINE void tile_vjp(const double* hidden, const double* auxiliary,
    const double* seed, const double* coordinates, int64_t full_points,
    int64_t value_points, int channels, int64_t tile_index, int channel,
    double* result) {
  constexpr int Q = D == 2 ? 10 : 20;
  for (int axis=0;axis<=D;++axis) result[axis]=0.0;
  const int64_t begin=tile_index*point_tile;
  const int64_t remaining=full_points+value_points-begin;
  const int count=int(remaining<point_tile ? remaining : point_tile);
  for (int k=0;k<count;++k) {
    const int64_t p=begin+k;
    const int64_t row=p<full_points ? p*Q : full_points*Q+p-full_points;
    const int64_t offset=row*channels+channel;
    double derivative[Q], value[D+1];
    point_vjp<D>(hidden+offset,seed+offset,channels,auxiliary[p*channels+channel],p<full_points,derivative);
    point<D>(derivative,1,coordinates+p*D,p<full_points,value);
    for (int axis=0;axis<=D;++axis) result[axis]+=value[axis];
  }
}
}  // namespace flashns_coordinate_wgrad

#undef FLASHNS_FIRST_VJP_INLINE
