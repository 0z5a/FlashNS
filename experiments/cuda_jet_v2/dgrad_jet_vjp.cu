// Experimental FP64 dgrad + full-jet VJP using a pinned CUTLASS mainloop.
// Inputs retain [B,Q,C]. Every CTA owns complete jets for its samples/channels.
// Dprev remains materialized for independent wgrad and the next dgrad.
#include <cuda_runtime.h>
#include <climits>
#include <cstddef>
#include <cstdint>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/layout/matrix.h"
#include "../cuda_jet/input/jet_primitives.cuh"

namespace flashns_v2 {

using Layout = cutlass::layout::RowMajor;
using BlockShape = cutlass::gemm::GemmShape<64, 64, 16>;
using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
using Kernel = typename cutlass::gemm::kernel::DefaultGemm<
    double, Layout, 1, double, Layout, 1, double, Layout, double,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    BlockShape, WarpShape, InstructionShape,
    cutlass::epilogue::thread::LinearCombination<double, 1, double, double>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3, false, cutlass::arch::OpMultiplyAdd>::GemmKernel;
using Mma = typename Kernel::Mma;
using WarpMma = typename Mma::Operator;
constexpr int kThreads = Mma::WarpCount::kCount * 32;
static_assert(Mma::WarpCount::kK == 1, "No split-K is allowed before jet VJP");

union SharedStorage {
  typename Mma::SharedStorage mainloop;
  double full_jets[BlockShape::kM * BlockShape::kN];
};

template<int Dim> struct Jet;
template<> struct Jet<2> {
  static constexpr int Q = 10;
  __device__ static void vjp(const double* h, const double* b, double* d) {
    flashns::tanh_vjp_2d3(h, b, d);
  }
};
template<> struct Jet<3> {
  static constexpr int Q = 20;
  __device__ static void vjp(const double* h, const double* b, double* d) {
    flashns::tanh_vjp_3d3(h, b, d);
  }
};

template<int Dim, bool Fused>
__global__ __launch_bounds__(kThreads) void dgrad(
    const double* __restrict__ input_d,
    const double* __restrict__ weight,
    const double* __restrict__ hidden,
    double* __restrict__ output,
    int batch, int cout, int cin,
    Mma::IteratorA::Params params_a, Mma::IteratorB::Params params_b) {
  constexpr int Q = Jet<Dim>::Q;
  // Both Q=10 and Q=20 use 60 of 64 physical GEMM rows. No Q->32 padding.
  constexpr int samples = BlockShape::kM / Q;
  const int first_sample = int(blockIdx.x) * samples;
  const int first_row = first_sample * Q;
  const int last_row = min(batch, first_sample + samples) * Q;
  const int first_channel = int(blockIdx.y) * BlockShape::kN;
  const int lane = int(threadIdx.x) % 32;
  const int warp = cutlass::canonical_warp_idx_sync();
  __shared__ SharedStorage shared;

  Mma::IteratorA a(params_a, const_cast<double*>(input_d), {last_row, cout}, int(threadIdx.x), {first_row, 0});
  Mma::IteratorB b(params_b, const_cast<double*>(weight), {cout, cin}, int(threadIdx.x), {0, first_channel});
  Mma mma(shared.mainloop, int(threadIdx.x), warp, lane);
  Mma::FragmentC accumulators;
  accumulators.clear();
  mma((cout + BlockShape::kK - 1) / BlockShape::kK,
      accumulators, a, b, accumulators);

  // Drain the CUTLASS async-copy pipeline after the final synchronous MMA.
  // All CTA threads stop using A/B shared storage before the union is reused.
  cutlass::arch::cp_async_wait<0>();
  __syncthreads();
  typename WarpMma::IteratorC gather(
      {shared.full_jets, Layout(BlockShape::kN)}, lane);
  gather.add_tile_offset({warp % Mma::WarpCount::kM,
                         warp / Mma::WarpCount::kM});
  gather.store(accumulators);
  __syncthreads();

  for (int owner = int(threadIdx.x); owner < samples * BlockShape::kN;
       owner += kThreads) {
    const int sample = owner / BlockShape::kN;
    const int channel = owner % BlockShape::kN;
    const int global_sample = first_sample + sample;
    const int global_channel = first_channel + channel;
    if (global_sample >= batch || global_channel >= cin) continue;
    double bar_h[Q];
    #pragma unroll
    for (int q = 0; q < Q; ++q) {
      bar_h[q] = shared.full_jets[(sample * Q + q) * BlockShape::kN + channel];
    }
    if constexpr (Fused) {
      double h[Q], d[Q];
      #pragma unroll
      for (int q = 0; q < Q; ++q) {
        h[q] = hidden[(std::size_t(global_sample) * Q + q) * cin + global_channel];
      }
      Jet<Dim>::vjp(h, bar_h, d);
      #pragma unroll
      for (int q = 0; q < Q; ++q) {
        output[(std::size_t(global_sample) * Q + q) * cin + global_channel] = d[q];
      }
    } else {
      #pragma unroll
      for (int q = 0; q < Q; ++q) {
        output[(std::size_t(global_sample) * Q + q) * cin + global_channel] = bar_h[q];
      }
    }
  }
}

static bool overlaps(const double* a, std::size_t na,
                     const double* b, std::size_t nb) {
  auto pa = reinterpret_cast<std::uintptr_t>(a);
  auto pb = reinterpret_cast<std::uintptr_t>(b);
  // Subtractions avoid overflow when a malformed pointer is near UINTPTR_MAX.
  return pa <= pb ? pb - pa < na * sizeof(double)
                  : pa - pb < nb * sizeof(double);
}

template<int Dim, bool Fused>
cudaError_t launch(const double* d, const double* w, const double* h, double* out,
                   int batch, int cout, int cin, cudaStream_t stream) {
  constexpr int samples = BlockShape::kM / Jet<Dim>::Q;
  dim3 grid((batch + samples - 1) / samples,
            (cin + BlockShape::kN - 1) / BlockShape::kN);
  dgrad<Dim, Fused><<<grid, kThreads, 0, stream>>>(
      d, w, h, out, batch, cout, cin,
      Mma::IteratorA::Params{Layout(cout)}, Mma::IteratorB::Params{Layout(cin)});
  return cudaGetLastError();
}

}  // namespace flashns_v2

extern "C" cudaError_t flashns_dgrad_jet_vjp_fp64(
    int dim, bool fused, const double* d, const double* w, const double* h,
    double* out, int batch, int cout, int cin, cudaStream_t stream) {
  using namespace flashns_v2;
  if ((dim != 2 && dim != 3) || batch < 0 || cin < 0 || cout <= 0 ||
      batch > INT_MAX / 20 || cin > 64 * 65535 || cout > INT_MAX - 16) {
    return cudaErrorInvalidValue;
  }
  if (batch == 0 || cin == 0) return cudaSuccess;
  if (!d || !w || !out || (fused && !h)) return cudaErrorInvalidValue;
  const std::size_t q = dim == 2 ? 10 : 20;
  const std::size_t no = std::size_t(batch) * q * cin;
  if (no > SIZE_MAX / sizeof(double) || std::size_t(batch) * q * cout > SIZE_MAX / sizeof(double) || std::size_t(cout) * cin > SIZE_MAX / sizeof(double)) {
    return cudaErrorInvalidValue;
  }
  if (overlaps(out, no, d, std::size_t(batch) * q * cout) ||
      overlaps(out, no, w, std::size_t(cout) * cin) ||
      (h && overlaps(out, no, h, no))) return cudaErrorInvalidValue;
  if (dim == 2) {
    return fused ? launch<2, true>(d, w, h, out, batch, cout, cin, stream)
                 : launch<2, false>(d, w, h, out, batch, cout, cin, stream);
  }
  return fused ? launch<3, true>(d, w, h, out, batch, cout, cin, stream)
               : launch<3, false>(d, w, h, out, batch, cout, cin, stream);
}

extern "C" int flashns_dgrad_shared_bytes() {
  return sizeof(flashns_v2::SharedStorage);
}
