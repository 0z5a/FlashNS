#pragma once
#include <cstdint>

#if defined(__CUDACC__)
#define FLASHNS_WGRAD_INLINE __host__ __device__ __forceinline__
#else
#define FLASHNS_WGRAD_INLINE inline
#endif

// Terminal contraction for physical Cartesian coordinate jets. This helper
// consumes the COMPLETE bar_Z produced by the unchanged activation VJP.
// Every structural zero multiplication is retained for NaN/Inf propagation.
// Fixed sequential point tiles and a fixed tile merge avoid floating atomics;
// their rounding order is not the order of the baseline library GEMM.
namespace flashns_coordinate_wgrad {
constexpr int point_tile = 8;

template<int D>
FLASHNS_WGRAD_INLINE void point(const double* derivative, int64_t stride,
    const double* coordinates, bool full, double* contribution) {
  constexpr int Q = D == 2 ? 10 : 20;
  for (int axis=0;axis<D;++axis) {
    double value=derivative[0]*coordinates[axis];
    if (full) {
      for (int j=1;j<Q;++j)
        value+=derivative[int64_t(j)*stride]*(j==D-axis ? 1.0 : 0.0);
    }
    contribution[axis]=value;
  }
  contribution[D]=derivative[0];
}

template<int D>
FLASHNS_WGRAD_INLINE void tile(const double* derivative, const double* coordinates,
    int64_t full_points, int64_t value_points, int channels,
    int64_t tile_index, int channel, double* result) {
  constexpr int Q = D == 2 ? 10 : 20;
  for (int axis=0;axis<=D;++axis) result[axis]=0.0;
  const int64_t begin=tile_index*point_tile;
  const int64_t remaining=full_points+value_points-begin;
  const int count=int(remaining<point_tile ? remaining : point_tile);
  for (int k=0;k<count;++k) {
    const int64_t p=begin+k;
    const int64_t row=p<full_points ? p*Q : full_points*Q+p-full_points;
    double value[D+1];
    point<D>(derivative+row*channels+channel,channels,coordinates+p*D,p<full_points,value);
    for (int axis=0;axis<=D;++axis) result[axis]+=value[axis];
  }
}

template<int D>
FLASHNS_WGRAD_INLINE void finish(const double* partials, int64_t tiles,
    int channels, int channel, double* weight_gradient, double* bias_gradient) {
  double result[D+1];
  for (int axis=0;axis<=D;++axis) result[axis]=0.0;
  for (int64_t t=0;t<tiles;++t)
    for (int axis=0;axis<=D;++axis)
      result[axis]+=partials[(t*channels+channel)*(D+1)+axis];
  for (int axis=0;axis<D;++axis) weight_gradient[int64_t(channel)*D+axis]=result[axis];
  bias_gradient[channel]=result[D];
}
}  // namespace flashns_coordinate_wgrad

#undef FLASHNS_WGRAD_INLINE
