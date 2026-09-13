// Controlled FP64 Hopper dgrad + complete-jet VJP experiment.
// All copy modes use identical tensor-core instructions, tiling and epilogue.
// PTX 8.7: sections 5.5.7, 9.7.9.25 and 9.7.14.5.2 define the layouts here.
#include <cuda.h>
#include <cuda_runtime.h>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <new>
#include "../cuda_jet_h100/stable_jet.cuh"

#ifndef HOPPER_M
#define HOPPER_M 64
#endif
#ifndef HOPPER_N
#define HOPPER_N 64
#endif
#ifndef HOPPER_STAGES
#define HOPPER_STAGES 2
#endif
#ifndef HOPPER_COPY
#define HOPPER_COPY 2
#endif
#ifndef HOPPER_SEPARATE_EPILOGUE
#define HOPPER_SEPARATE_EPILOGUE 0
#endif
#ifndef HOPPER_SWIZZLE
#define HOPPER_SWIZZLE 1
#endif

namespace flashns_hopper {
constexpr int M = HOPPER_M, N = HOPPER_N, TK = 16, Stages = HOPPER_STAGES;
constexpr int Threads = (M / 32) * (N / 32) * 32;
constexpr int Copy = HOPPER_COPY;  // 0: scalar vector copy, 1: cp.async, 2: TMA
static_assert((M == 32 || M == 64) && (N == 32 || N == 64));
static_assert(Stages >= 1 && Stages <= 3 && Copy >= 0 && Copy <= 2);

struct alignas(1024) Storage {
  union {
    struct {
      double a[Stages][M * TK];
      double b[Stages][TK * N];
    } pipeline;
    double full_jets[M * N];
  } data;
  alignas(8) std::uint64_t barrier[Stages];
};

struct alignas(64) Plan {
  CUtensorMap a, b;
  const double *d, *w;
  int batch, cout, cin, q;
};

template<int Dim> struct Jet;
template<> struct Jet<2> {
  static constexpr int Q = 10;
  __device__ static __forceinline__ void vjp(const double* h, const double* b, double a, double* d) {
    flashns_stable::tanh_vjp_2d3(h, b, a, d);
  }
};
template<> struct Jet<3> {
  static constexpr int Q = 20;
  __device__ static __forceinline__ void vjp(const double* h, const double* b, double a, double* d) {
    flashns_stable::tanh_vjp_3d3(h, b, a, d);
  }
};

__device__ __forceinline__ int swizzle(int element) {
  // A/B starts and every stage are 1024-byte aligned. A 16-byte pair stays
  // intact; row-within-1024B XORs its index into the 16-byte pair index.
  return HOPPER_SWIZZLE ? (element ^ ((element >> 3) & 14)) : element;
}

__device__ __forceinline__ void async_pair(double* dst, const double* src, bool valid) {
  auto shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  int bytes = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               :: "r"(shared), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void wait_copy(int groups) {
  if (groups == 2) asm volatile("cp.async.wait_group 2;" ::: "memory");
  else if (groups == 1) asm volatile("cp.async.wait_group 1;" ::: "memory");
  else asm volatile("cp.async.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ void wait_tma(std::uint64_t* barrier, unsigned phase) {
  auto address = static_cast<unsigned>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "{ .reg .pred done; AGAIN: "
      "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1; "
      "@!done bra AGAIN; }"
      :: "r"(address), "r"(phase) : "memory");
}

__device__ __forceinline__ void prefetch(
    Storage& shared, int slot, int ktile, int first_row, int first_col,
    const double* d, const double* w, int rows, int cout, int cin,
    const CUtensorMap* map_a, const CUtensorMap* map_b) {
  double* a = shared.data.pipeline.a[slot];
  double* b = shared.data.pipeline.b[slot];
  if constexpr (Copy == 2) {
    if (threadIdx.x == 0) {
      auto barrier = static_cast<unsigned>(__cvta_generic_to_shared(&shared.barrier[slot]));
      auto sa = static_cast<unsigned>(__cvta_generic_to_shared(a));
      auto sb = static_cast<unsigned>(__cvta_generic_to_shared(b));
      constexpr unsigned bytes = (M * TK + TK * N) * sizeof(double);
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                   :: "r"(barrier), "r"(bytes) : "memory");
      asm volatile(
          "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
          "[%0], [%1, {%3, %4}], [%2];"
          :: "r"(sa), "l"(map_a), "r"(barrier), "r"(ktile * TK), "r"(first_row) : "memory");
      // Splitting the contiguous Cin dimension into [Cin/16,16] permits
      // 128B swizzling without packing or transposing the global matrix.
      asm volatile(
          "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
          "[%0], [%1, {%3, %4, %5}], [%2];"
          :: "r"(sb), "l"(map_b), "r"(barrier), "r"(0),
             "r"(first_col / 16), "r"(ktile * TK) : "memory");
    }
  } else {
    for (int pair = int(threadIdx.x); pair < M * TK / 2; pair += Threads) {
      const int offset = pair * 2, row = offset / TK, k = offset % TK;
      bool valid = first_row + row < rows && ktile * TK + k + 1 < cout;
      const double* source = valid ? d + std::size_t(first_row + row) * cout + ktile * TK + k : d;
      if constexpr (Copy == 1) async_pair(a + swizzle(offset), source, valid);
      else *reinterpret_cast<double2*>(a + swizzle(offset)) = valid ? *reinterpret_cast<const double2*>(source) : make_double2(0, 0);
    }
    for (int pair = int(threadIdx.x); pair < TK * N / 2; pair += Threads) {
      const int offset = pair * 2, k = offset / N, col = offset % N;
      bool valid = ktile * TK + k < cout && first_col + col + 1 < cin;
      const double* source = valid ? w + std::size_t(ktile * TK + k) * cin + first_col + col : w;
      if constexpr (Copy == 1) async_pair(b + swizzle(offset), source, valid);
      else *reinterpret_cast<double2*>(b + swizzle(offset)) = valid ? *reinterpret_cast<const double2*>(source) : make_double2(0, 0);
    }
    if constexpr (Copy == 1) asm volatile("cp.async.commit_group;" ::: "memory");
  }
}

#if HOPPER_SEPARATE_EPILOGUE
#define EPILOGUE_INLINE __noinline__
#else
#define EPILOGUE_INLINE __forceinline__
#endif
template<int Dim, bool Fused>
__device__ EPILOGUE_INLINE void epilogue(const double* full, const double* hidden,
    const double* aux, double* output, int first_sample, int first_col, int batch, int cin) {
  constexpr int Q = Jet<Dim>::Q, Samples = M / Q;
  for (int owner = int(threadIdx.x); owner < Samples * N; owner += Threads) {
    int sample = owner / N, channel = owner % N;
    int global_sample = first_sample + sample, global_col = first_col + channel;
    if (global_sample >= batch || global_col >= cin) continue;
    double bar[Q];
    #pragma unroll
    for (int q = 0; q < Q; ++q) bar[q] = full[(sample * Q + q) * N + channel];
    if constexpr (Fused) {
      double h[Q], result[Q];
      #pragma unroll
      for (int q = 0; q < Q; ++q) h[q] = hidden[(std::size_t(global_sample) * Q + q) * cin + global_col];
      Jet<Dim>::vjp(h, bar, aux[std::size_t(global_sample) * cin + global_col], result);
      #pragma unroll
      for (int q = 0; q < Q; ++q) output[(std::size_t(global_sample) * Q + q) * cin + global_col] = result[q];
    } else {
      #pragma unroll
      for (int q = 0; q < Q; ++q) output[(std::size_t(global_sample) * Q + q) * cin + global_col] = bar[q];
    }
  }
}

template<int Dim, bool Fused>
__global__ __launch_bounds__(Threads) void dgrad(
    const double* __restrict__ d, const double* __restrict__ w,
    const double* __restrict__ hidden, const double* __restrict__ aux,
    double* __restrict__ output, int batch, int cout, int cin,
    const __grid_constant__ CUtensorMap map_a, const __grid_constant__ CUtensorMap map_b) {
  constexpr int Q = Jet<Dim>::Q, Samples = M / Q;
  const int first_sample = int(blockIdx.x) * Samples;
  const int first_row = first_sample * Q, first_col = int(blockIdx.y) * N;
  const int warp = int(threadIdx.x) / 32, lane = int(threadIdx.x) % 32;
  const int warp_row = (warp / (N / 32)) * 32, warp_col = (warp % (N / 32)) * 32;
  extern __shared__ __align__(1024) unsigned char storage[];
  Storage& shared = *reinterpret_cast<Storage*>(storage);
  if constexpr (Copy == 2) {
    if (threadIdx.x == 0) {
      #pragma unroll
      for (int slot = 0; slot < Stages; ++slot) {
        auto address = static_cast<unsigned>(__cvta_generic_to_shared(&shared.barrier[slot]));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(address) : "memory");
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
  }
  const int tiles = cout / TK;
  for (int tile = 0; tile < min(Stages, tiles); ++tile)
    prefetch(shared, tile, tile, first_row, first_col, d, w, batch * Q, cout, cin, &map_a, &map_b);
  {
    // Every lane owns two FP64 accumulators in each of 16 8x8 subtiles.
    // This scope ends before the complete-jet arrays in the epilogue.
    double c[4][4][2] = {};
    for (int tile = 0; tile < tiles; ++tile) {
      int slot = tile % Stages;
      if constexpr (Copy == 1) wait_copy(min(Stages - 1, tiles - tile - 1));
      if constexpr (Copy == 2) wait_tma(&shared.barrier[slot], (tile / Stages) & 1);
      __syncthreads();
      const double* a = shared.data.pipeline.a[slot];
      const double* b = shared.data.pipeline.b[slot];
      #pragma unroll
      for (int kk = 0; kk < TK; kk += 4) {
        double ar[4], br[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          ar[i] = a[swizzle((warp_row + i * 8 + lane / 4) * TK + kk + lane % 4)];
          br[i] = b[swizzle((kk + lane % 4) * N + warp_col + i * 8 + lane / 4)];
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          #pragma unroll
          for (int j = 0; j < 4; ++j) {
            asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                         "{%0, %1}, {%2}, {%3}, {%0, %1};"
                         : "+d"(c[i][j][0]), "+d"(c[i][j][1]) : "d"(ar[i]), "d"(br[j]));
          }
        }
      }
      // mma.sync has completed; all warps finish reading before slot reuse.
      __syncthreads();
      if (tile + Stages < tiles)
        prefetch(shared, slot, tile + Stages, first_row, first_col, d, w, batch * Q, cout, cin, &map_a, &map_b);
    }
    if constexpr (Copy == 1) wait_copy(0);
    __syncthreads();
    // All copies and synchronous MMA operations have completed. Reuse the
    // mainloop union as a canonical full-jet tile for the nonlinear VJP.
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        int row = warp_row + i * 8 + lane / 4;
        int col = warp_col + j * 8 + 2 * (lane % 4);
        shared.data.full_jets[row * N + col] = c[i][j][0];
        shared.data.full_jets[row * N + col + 1] = c[i][j][1];
      }
    }
  }
  __syncthreads();
  epilogue<Dim, Fused>(shared.data.full_jets, hidden, aux, output, first_sample, first_col, batch, cin);
}

template<int Dim, bool Fused>
cudaError_t configure() {
  return cudaFuncSetAttribute(dgrad<Dim, Fused>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Storage));
}
template<int Dim, bool Fused>
cudaError_t resources(long long* values) {
  cudaFuncAttributes attributes;
  cudaError_t code = cudaFuncGetAttributes(&attributes, dgrad<Dim, Fused>);
  if (code != cudaSuccess) return code;
  int blocks;
  code = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, dgrad<Dim, Fused>, Threads, sizeof(Storage));
  if (code != cudaSuccess) return code;
  values[0] = attributes.numRegs; values[1] = Threads;
  values[2] = sizeof(Storage); values[3] = attributes.sharedSizeBytes;
  values[4] = attributes.localSizeBytes; values[5] = blocks;
  values[6] = attributes.maxDynamicSharedSizeBytes;
  values[7] = attributes.binaryVersion; values[8] = attributes.ptxVersion;
  return cudaSuccess;
}

bool overlaps(const void* a, std::size_t na, const void* b, std::size_t nb) {
  auto aa = reinterpret_cast<std::uintptr_t>(a), bb = reinterpret_cast<std::uintptr_t>(b);
  return na && nb && (aa <= bb ? bb - aa < na : aa - bb < nb);
}

template<int Dim, bool Fused>
cudaError_t launch(const Plan& plan, const double* hidden, const double* aux, double* output, cudaStream_t stream) {
  constexpr int Samples = M / Jet<Dim>::Q;
  dim3 grid((plan.batch + Samples - 1) / Samples, (plan.cin + N - 1) / N);
  dgrad<Dim, Fused><<<grid, Threads, sizeof(Storage), stream>>>(plan.d, plan.w, hidden, aux, output,
      plan.batch, plan.cout, plan.cin, plan.a, plan.b);
  return cudaGetLastError();
}
}  // namespace flashns_hopper

extern "C" int flashns_hopper_initialize() {
  using namespace flashns_hopper;
  cudaError_t code;
  if ((code = configure<2, false>()) != cudaSuccess) return code;
  if ((code = configure<2, true>()) != cudaSuccess) return code;
  if ((code = configure<3, false>()) != cudaSuccess) return code;
  return configure<3, true>();
}

extern "C" int flashns_hopper_resources(int dim, bool fused, long long* values) {
  using namespace flashns_hopper;
  if (!values || (dim != 2 && dim != 3)) return cudaErrorInvalidValue;
  if (dim == 2) return fused ? resources<2, true>(values) : resources<2, false>(values);
  return fused ? resources<3, true>(values) : resources<3, false>(values);
}

extern "C" int flashns_hopper_create(int dim, const double* d, const double* w,
    int batch, int cout, int cin, void** result) {
  using namespace flashns_hopper;
  if (!result) return cudaErrorInvalidValue;
  *result = nullptr;
  if ((dim != 2 && dim != 3) || !d || !w || batch <= 0 || batch > INT_MAX / 20 ||
      (cout != 32 && cout != 64) || (cin != 32 && cin != 64) ||
      (reinterpret_cast<std::uintptr_t>(d) & 15) || (reinterpret_cast<std::uintptr_t>(w) & 15)) return cudaErrorInvalidValue;
  Plan* plan = new (std::nothrow) Plan{};
  if (!plan) return cudaErrorMemoryAllocation;
  plan->d = d; plan->w = w; plan->batch = batch; plan->cout = cout; plan->cin = cin; plan->q = dim == 2 ? 10 : 20;
  if constexpr (Copy == 2) {
    const auto swizzle = HOPPER_SWIZZLE ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE;
    const cuuint64_t sizes_a[2] = {cuuint64_t(cout), cuuint64_t(batch) * plan->q};
    const cuuint64_t strides_a[1] = {cuuint64_t(cout) * sizeof(double)};
    const cuuint32_t box_a[2] = {TK, M}, element_strides[3] = {1, 1, 1};
    CUresult code = cuTensorMapEncodeTiled(&plan->a, CU_TENSOR_MAP_DATA_TYPE_FLOAT64, 2,
        const_cast<double*>(d), sizes_a, strides_a, box_a, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (code != CUDA_SUCCESS) { delete plan; return 10000 + int(code); }
    const cuuint64_t sizes_b[3] = {16, cuuint64_t(cin / 16), cuuint64_t(cout)};
    const cuuint64_t strides_b[2] = {16 * sizeof(double), cuuint64_t(cin) * sizeof(double)};
    const cuuint32_t box_b[3] = {16, N / 16, TK};
    code = cuTensorMapEncodeTiled(&plan->b, CU_TENSOR_MAP_DATA_TYPE_FLOAT64, 3,
        const_cast<double*>(w), sizes_b, strides_b, box_b, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (code != CUDA_SUCCESS) { delete plan; return 10000 + int(code); }
  }
  *result = plan;
  return cudaSuccess;
}

extern "C" void flashns_hopper_destroy(void* plan) {
  delete static_cast<flashns_hopper::Plan*>(plan);
}

extern "C" int flashns_hopper_launch(void* handle, int dim, bool fused,
    const double* hidden, const double* aux, double* output, cudaStream_t stream) {
  using namespace flashns_hopper;
  if (!handle || !output || (dim != 2 && dim != 3) || (fused && (!hidden || !aux))) return cudaErrorInvalidValue;
  const Plan& plan = *static_cast<Plan*>(handle);
  const int q = dim == 2 ? 10 : 20;
  if (plan.q != q) return cudaErrorInvalidValue;
  std::size_t bytes = std::size_t(plan.batch) * q * plan.cin * sizeof(double);
  if (overlaps(output, bytes, plan.d, std::size_t(plan.batch) * q * plan.cout * sizeof(double)) ||
      overlaps(output, bytes, plan.w, std::size_t(plan.cout) * plan.cin * sizeof(double)) ||
      (hidden && overlaps(output, bytes, hidden, bytes)) ||
      (aux && overlaps(output, bytes, aux, std::size_t(plan.batch) * plan.cin * sizeof(double)))) return cudaErrorInvalidValue;
  if (dim == 2) return fused ? launch<2, true>(plan, hidden, aux, output, stream) : launch<2, false>(plan, hidden, aux, output, stream);
  return fused ? launch<3, true>(plan, hidden, aux, output, stream) : launch<3, false>(plan, hidden, aux, output, stream);
}
