// FP64 block GEMM A/B adapter, using public cuBLASDx shared-memory execution.
// CUDA 13 compiles a cubin; the Python harness submits it through the driver API.
#include <cublasdx.hpp>
#include <cstdint>

template<int M, int N, int K>
using Description = decltype(cublasdx::Size<M, N, K>() + cublasdx::Precision<double>() +
    cublasdx::Type<cublasdx::type::real>() + cublasdx::Function<cublasdx::function::MM>() +
    cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major, cublasdx::row_major>() +
    cublasdx::Block() + cublasdx::SM<900>());

template<int M, int N, int K>
__device__ __forceinline__ void execute(const double* a, const double* b,
    double* output, int64_t rows, int transpose_b) {
  using BLAS = Description<M, N, K>;
  extern __shared__ __align__(16) cublasdx::byte shared[];
  auto [sa, sb, sc] = cublasdx::slice_shared_memory<BLAS>(shared);
  auto ta = cublasdx::make_tensor(sa, BLAS::get_layout_smem_a());
  auto tb = cublasdx::make_tensor(sb, BLAS::get_layout_smem_b());
  auto tc = cublasdx::make_tensor(sc, BLAS::get_layout_smem_c());
  const int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int threads = blockDim.x * blockDim.y * blockDim.z;
  const int64_t base = int64_t(blockIdx.x) * M;
  for (int index = tid; index < M * K; index += threads) {
    const int i = index / K, k = index % K;
    ta(i, k) = base + i < rows ? a[(base + i) * K + k] : 0.0;
  }
  // Both arrangements read global B in storage order; remap in shared memory.
  for (int index = tid; index < K * N; index += threads) {
    const int k = transpose_b ? index % K : index / N;
    const int j = transpose_b ? index / K : index % N;
    tb(k, j) = b[index];
  }
  // Define C even when beta=0: do not depend on an implementation skipping reads.
  for (int index = tid; index < M * N; index += threads)
    tc(index / N, index % N) = 0.0;
  __syncthreads();
  BLAS().execute(1.0, ta, tb, 0.0, tc);
  __syncthreads();
  for (int index = tid; index < M * N; index += threads) {
    const int i = index / N, j = index % N;
    if (base + i < rows) output[(base + i) * N + j] = tc(i, j);
  }
}

#define INSTANCE(M, N, K) \
extern "C" __global__ void dx_##M##_##N##_##K(const double* a, const double* b, \
    double* output, int64_t rows, int transpose_b) { \
  execute<M, N, K>(a, b, output, rows, transpose_b); \
} \
extern "C" __device__ __constant__ int cfg_##M##_##N##_##K[] = { \
  int(Description<M, N, K>::block_dim.x), int(Description<M, N, K>::block_dim.y), \
  int(Description<M, N, K>::block_dim.z), int(cublasdx::get_shared_storage_size<Description<M, N, K>>()) \
};

INSTANCE(32, 32, 2)
INSTANCE(32, 32, 3)
INSTANCE(32, 32, 32)
INSTANCE(32, 3, 32)
INSTANCE(32, 64, 2)
INSTANCE(32, 64, 3)
INSTANCE(32, 64, 64)
INSTANCE(32, 3, 64)
INSTANCE(64, 32, 2)
INSTANCE(64, 32, 3)
INSTANCE(64, 32, 32)
INSTANCE(64, 3, 32)
INSTANCE(64, 64, 2)
INSTANCE(64, 64, 3)
INSTANCE(64, 64, 64)
INSTANCE(64, 3, 64)
