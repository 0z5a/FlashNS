#pragma once
#include <cmath>
// Generated factorial-normalized 2-D/3-D jets, total degree <=3.
// Stable auxiliary-state revision 2; validation scope lives in target reports.
// Inputs and outputs must not alias. FP64 ordinary arithmetic;
// Explicit a1 checkpoint preserves derivatives lost by rounded H[0].
#ifdef __CUDACC__
#define FLASHNS_JET_HD __host__ __device__ __forceinline__
#else
#define FLASHNS_JET_HD inline
#endif
namespace flashns_stable {
// Coefficient order 2d3: [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0), (0, 3), (1, 2), (2, 1), (3, 0)]
FLASHNS_JET_HD void tanh_fwd_2d3(const double* z, double* h, double* aux) {
  const double t = ::tanh(z[0]);
  const double r = ::exp(-::fabs(z[0]));
  const double v = (2.0*r)/(1.0+r*r);
  const double a1 = v*v;
  *aux = a1;
  const double a2 = -t*a1;
  const double a3 = a1*(t*t - 1.0/3.0);
  h[0] = t;
  const double p2_3 = z[1] * z[1];
  const double p2_4 = 2.0 * z[1] * z[2];
  const double p2_5 = z[2] * z[2];
  const double p2_6 = 2.0 * z[1] * z[3];
  const double p2_7 = 2.0 * z[1] * z[4] + 2.0 * z[2] * z[3];
  const double p2_8 = 2.0 * z[1] * z[5] + 2.0 * z[2] * z[4];
  const double p2_9 = 2.0 * z[2] * z[5];
  h[1] = a1*z[1];
  h[2] = a1*z[2];
  h[3] = a1*z[3] + a2*p2_3;
  h[4] = a1*z[4] + a2*p2_4;
  h[5] = a1*z[5] + a2*p2_5;
  h[6] = a1*z[6] + a2*p2_6 + a3*(z[1]*p2_3);
  h[7] = a1*z[7] + a2*p2_7 + a3*(z[1]*p2_4 + z[2]*p2_3);
  h[8] = a1*z[8] + a2*p2_8 + a3*(z[1]*p2_5 + z[2]*p2_4);
  h[9] = a1*z[9] + a2*p2_9 + a3*(z[2]*p2_5);
}
FLASHNS_JET_HD void tanh_vjp_2d3(const double* h, const double* bh, double a1, double* bz) {
  const double g0 = a1;
  const double g1 = -(2.0 * h[0] * h[1]);
  const double g2 = -(2.0 * h[0] * h[2]);
  const double g3 = -(2.0 * h[0] * h[3] + h[1] * h[1]);
  const double g4 = -(2.0 * h[0] * h[4] + 2.0 * h[1] * h[2]);
  const double g5 = -(2.0 * h[0] * h[5] + h[2] * h[2]);
  const double g6 = -(2.0 * h[0] * h[6] + 2.0 * h[1] * h[3]);
  const double g7 = -(2.0 * h[0] * h[7] + 2.0 * h[1] * h[4] + 2.0 * h[2] * h[3]);
  const double g8 = -(2.0 * h[0] * h[8] + 2.0 * h[1] * h[5] + 2.0 * h[2] * h[4]);
  const double g9 = -(2.0 * h[0] * h[9] + 2.0 * h[2] * h[5]);
  bz[0] = bh[0]*g0 + bh[1]*g1 + bh[2]*g2 + bh[3]*g3 + bh[4]*g4 + bh[5]*g5 + bh[6]*g6 + bh[7]*g7 + bh[8]*g8 + bh[9]*g9;
  bz[1] = bh[1]*g0 + bh[3]*g1 + bh[4]*g2 + bh[6]*g3 + bh[7]*g4 + bh[8]*g5;
  bz[2] = bh[2]*g0 + bh[4]*g1 + bh[5]*g2 + bh[7]*g3 + bh[8]*g4 + bh[9]*g5;
  bz[3] = bh[3]*g0 + bh[6]*g1 + bh[7]*g2;
  bz[4] = bh[4]*g0 + bh[7]*g1 + bh[8]*g2;
  bz[5] = bh[5]*g0 + bh[8]*g1 + bh[9]*g2;
  bz[6] = bh[6]*g0;
  bz[7] = bh[7]*g0;
  bz[8] = bh[8]*g0;
  bz[9] = bh[9]*g0;
}
// Coefficient order 3d3: [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0), (0, 0, 3), (0, 1, 2), (0, 2, 1), (0, 3, 0), (1, 0, 2), (1, 1, 1), (1, 2, 0), (2, 0, 1), (2, 1, 0), (3, 0, 0)]
FLASHNS_JET_HD void tanh_fwd_3d3(const double* z, double* h, double* aux) {
  const double t = ::tanh(z[0]);
  const double r = ::exp(-::fabs(z[0]));
  const double v = (2.0*r)/(1.0+r*r);
  const double a1 = v*v;
  *aux = a1;
  const double a2 = -t*a1;
  const double a3 = a1*(t*t - 1.0/3.0);
  h[0] = t;
  const double p2_4 = z[1] * z[1];
  const double p2_5 = 2.0 * z[1] * z[2];
  const double p2_6 = z[2] * z[2];
  const double p2_7 = 2.0 * z[1] * z[3];
  const double p2_8 = 2.0 * z[2] * z[3];
  const double p2_9 = z[3] * z[3];
  const double p2_10 = 2.0 * z[1] * z[4];
  const double p2_11 = 2.0 * z[1] * z[5] + 2.0 * z[2] * z[4];
  const double p2_12 = 2.0 * z[1] * z[6] + 2.0 * z[2] * z[5];
  const double p2_13 = 2.0 * z[2] * z[6];
  const double p2_14 = 2.0 * z[1] * z[7] + 2.0 * z[3] * z[4];
  const double p2_15 = 2.0 * z[1] * z[8] + 2.0 * z[2] * z[7] + 2.0 * z[3] * z[5];
  const double p2_16 = 2.0 * z[2] * z[8] + 2.0 * z[3] * z[6];
  const double p2_17 = 2.0 * z[1] * z[9] + 2.0 * z[3] * z[7];
  const double p2_18 = 2.0 * z[2] * z[9] + 2.0 * z[3] * z[8];
  const double p2_19 = 2.0 * z[3] * z[9];
  h[1] = a1*z[1];
  h[2] = a1*z[2];
  h[3] = a1*z[3];
  h[4] = a1*z[4] + a2*p2_4;
  h[5] = a1*z[5] + a2*p2_5;
  h[6] = a1*z[6] + a2*p2_6;
  h[7] = a1*z[7] + a2*p2_7;
  h[8] = a1*z[8] + a2*p2_8;
  h[9] = a1*z[9] + a2*p2_9;
  h[10] = a1*z[10] + a2*p2_10 + a3*(z[1]*p2_4);
  h[11] = a1*z[11] + a2*p2_11 + a3*(z[1]*p2_5 + z[2]*p2_4);
  h[12] = a1*z[12] + a2*p2_12 + a3*(z[1]*p2_6 + z[2]*p2_5);
  h[13] = a1*z[13] + a2*p2_13 + a3*(z[2]*p2_6);
  h[14] = a1*z[14] + a2*p2_14 + a3*(z[1]*p2_7 + z[3]*p2_4);
  h[15] = a1*z[15] + a2*p2_15 + a3*(z[1]*p2_8 + z[2]*p2_7 + z[3]*p2_5);
  h[16] = a1*z[16] + a2*p2_16 + a3*(z[2]*p2_8 + z[3]*p2_6);
  h[17] = a1*z[17] + a2*p2_17 + a3*(z[1]*p2_9 + z[3]*p2_7);
  h[18] = a1*z[18] + a2*p2_18 + a3*(z[2]*p2_9 + z[3]*p2_8);
  h[19] = a1*z[19] + a2*p2_19 + a3*(z[3]*p2_9);
}
FLASHNS_JET_HD void tanh_vjp_3d3(const double* h, const double* bh, double a1, double* bz) {
  const double g0 = a1;
  const double g1 = -(2.0 * h[0] * h[1]);
  const double g2 = -(2.0 * h[0] * h[2]);
  const double g3 = -(2.0 * h[0] * h[3]);
  const double g4 = -(2.0 * h[0] * h[4] + h[1] * h[1]);
  const double g5 = -(2.0 * h[0] * h[5] + 2.0 * h[1] * h[2]);
  const double g6 = -(2.0 * h[0] * h[6] + h[2] * h[2]);
  const double g7 = -(2.0 * h[0] * h[7] + 2.0 * h[1] * h[3]);
  const double g8 = -(2.0 * h[0] * h[8] + 2.0 * h[2] * h[3]);
  const double g9 = -(2.0 * h[0] * h[9] + h[3] * h[3]);
  const double g10 = -(2.0 * h[0] * h[10] + 2.0 * h[1] * h[4]);
  const double g11 = -(2.0 * h[0] * h[11] + 2.0 * h[1] * h[5] + 2.0 * h[2] * h[4]);
  const double g12 = -(2.0 * h[0] * h[12] + 2.0 * h[1] * h[6] + 2.0 * h[2] * h[5]);
  const double g13 = -(2.0 * h[0] * h[13] + 2.0 * h[2] * h[6]);
  const double g14 = -(2.0 * h[0] * h[14] + 2.0 * h[1] * h[7] + 2.0 * h[3] * h[4]);
  const double g15 = -(2.0 * h[0] * h[15] + 2.0 * h[1] * h[8] + 2.0 * h[2] * h[7] + 2.0 * h[3] * h[5]);
  const double g16 = -(2.0 * h[0] * h[16] + 2.0 * h[2] * h[8] + 2.0 * h[3] * h[6]);
  const double g17 = -(2.0 * h[0] * h[17] + 2.0 * h[1] * h[9] + 2.0 * h[3] * h[7]);
  const double g18 = -(2.0 * h[0] * h[18] + 2.0 * h[2] * h[9] + 2.0 * h[3] * h[8]);
  const double g19 = -(2.0 * h[0] * h[19] + 2.0 * h[3] * h[9]);
  bz[0] = bh[0]*g0 + bh[1]*g1 + bh[2]*g2 + bh[3]*g3 + bh[4]*g4 + bh[5]*g5 + bh[6]*g6 + bh[7]*g7 + bh[8]*g8 + bh[9]*g9 + bh[10]*g10 + bh[11]*g11 + bh[12]*g12 + bh[13]*g13 + bh[14]*g14 + bh[15]*g15 + bh[16]*g16 + bh[17]*g17 + bh[18]*g18 + bh[19]*g19;
  bz[1] = bh[1]*g0 + bh[4]*g1 + bh[5]*g2 + bh[7]*g3 + bh[10]*g4 + bh[11]*g5 + bh[12]*g6 + bh[14]*g7 + bh[15]*g8 + bh[17]*g9;
  bz[2] = bh[2]*g0 + bh[5]*g1 + bh[6]*g2 + bh[8]*g3 + bh[11]*g4 + bh[12]*g5 + bh[13]*g6 + bh[15]*g7 + bh[16]*g8 + bh[18]*g9;
  bz[3] = bh[3]*g0 + bh[7]*g1 + bh[8]*g2 + bh[9]*g3 + bh[14]*g4 + bh[15]*g5 + bh[16]*g6 + bh[17]*g7 + bh[18]*g8 + bh[19]*g9;
  bz[4] = bh[4]*g0 + bh[10]*g1 + bh[11]*g2 + bh[14]*g3;
  bz[5] = bh[5]*g0 + bh[11]*g1 + bh[12]*g2 + bh[15]*g3;
  bz[6] = bh[6]*g0 + bh[12]*g1 + bh[13]*g2 + bh[16]*g3;
  bz[7] = bh[7]*g0 + bh[14]*g1 + bh[15]*g2 + bh[17]*g3;
  bz[8] = bh[8]*g0 + bh[15]*g1 + bh[16]*g2 + bh[18]*g3;
  bz[9] = bh[9]*g0 + bh[17]*g1 + bh[18]*g2 + bh[19]*g3;
  bz[10] = bh[10]*g0;
  bz[11] = bh[11]*g0;
  bz[12] = bh[12]*g0;
  bz[13] = bh[13]*g0;
  bz[14] = bh[14]*g0;
  bz[15] = bh[15]*g0;
  bz[16] = bh[16]*g0;
  bz[17] = bh[17]*g0;
  bz[18] = bh[18]*g0;
  bz[19] = bh[19]*g0;
}
} // namespace flashns_stable
#undef FLASHNS_JET_HD
