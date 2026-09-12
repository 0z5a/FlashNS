#pragma once
#include "../cuda_jet_h100/stable_jet.cuh"

#ifdef __CUDACC__
#define FLASHNS_COORD_HD __host__ __device__ __forceinline__
#else
#define FLASHNS_COORD_HD inline
#endif

namespace flashns_coordinate {

// Shared by the materialized control and fused candidate. Retain the original
// axis accumulation order, including IEEE 0*nonfinite propagation. No fast math
// or FMA contraction is allowed for either arm.
template<int D>
FLASHNS_COORD_HD void affine_jet(const double* x, const double* w, double bias,
                                bool full, double* z) {
  static_assert(D == 2 || D == 3, "full Taylor Q10/Q20 only");
  constexpr int Q = D == 2 ? 10 : 20;
  double value=x[0]*w[0];
  for (int axis=1;axis<D;++axis) value+=x[axis]*w[axis];
  z[0]=value+bias;
  if (!full) return;
  double zero=0.0*w[0];
  for (int axis=1;axis<D;++axis) zero+=0.0*w[axis];
  for (int j=1;j<Q;++j) {
    double coefficient=zero;
    if (j<=D) {
      const int selected=D-j;
      coefficient=(selected==0 ? 1.0 : 0.0)*w[0];
      for (int axis=1;axis<D;++axis)
        coefficient+=(selected==axis ? 1.0 : 0.0)*w[axis];
    }
    z[j]=coefficient;
  }
}

// The same stable activation and a1 checkpoint as the separate activation ABI.
// The first affine Z remains thread-local; H and a1 remain materialized for VJP.
template<int D>
FLASHNS_COORD_HD void affine_activation(const double* x, const double* w, double bias,
                                       bool full, double* h, double* auxiliary) {
  constexpr int Q = D == 2 ? 10 : 20;
  double z[Q];
  affine_jet<D>(x,w,bias,full,z);
  if (full) {
    if constexpr (D == 2) flashns_stable::tanh_fwd_2d3(z,h,auxiliary);
    else flashns_stable::tanh_fwd_3d3(z,h,auxiliary);
  } else {
    const double r=::exp(-::fabs(z[0])), v=(2.0*r)/(1.0+r*r);
    h[0]=::tanh(z[0]);
    *auxiliary=v*v;
  }
}

}  // namespace flashns_coordinate
#undef FLASHNS_COORD_HD
