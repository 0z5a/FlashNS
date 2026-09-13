#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include "stable_jet.cuh"

template<int Dim> struct StableJet;
template<> struct StableJet<2> {
  static constexpr int Q = 10;
  __device__ static void forward(const double* z, double* h, double* a) {
    flashns_stable::tanh_fwd_2d3(z, h, a);
  }
  __device__ static void vjp(const double* h, const double* b, double a, double* d) {
    flashns_stable::tanh_vjp_2d3(h, b, a, d);
  }
};
template<> struct StableJet<3> {
  static constexpr int Q = 20;
  __device__ static void forward(const double* z, double* h, double* a) {
    flashns_stable::tanh_fwd_3d3(z, h, a);
  }
  __device__ static void vjp(const double* h, const double* b, double a, double* d) {
    flashns_stable::tanh_vjp_3d3(h, b, a, d);
  }
};

template<int Dim, bool Backward>
__global__ void activation(const double* value, const double* adjoint,
                           const double* aux_in, double* output, double* aux_out,
                           std::size_t batch, std::size_t channels) {
  constexpr int Q = StableJet<Dim>::Q;
  for (std::size_t owner = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       owner < batch * channels; owner += std::size_t(gridDim.x) * blockDim.x) {
    const std::size_t b = owner / channels, c = owner % channels;
    double x[Q], y[Q];
    #pragma unroll
    for (int j = 0; j < Q; ++j) x[j] = value[(b * Q + j) * channels + c];
    if constexpr (Backward) {
      double bar[Q];
      #pragma unroll
      for (int j = 0; j < Q; ++j) bar[j] = adjoint[(b * Q + j) * channels + c];
      StableJet<Dim>::vjp(x, bar, aux_in[owner], y);
    } else {
      double aux;
      StableJet<Dim>::forward(x, y, &aux);
      aux_out[owner] = aux;
    }
    #pragma unroll
    for (int j = 0; j < Q; ++j) output[(b * Q + j) * channels + c] = y[j];
  }
}

template<int Dim, bool Fused>
__global__ void tail3(const double* d, const double* weight, const double* hidden,
                      const double* aux, double* output,
                      std::size_t batch, std::size_t channels) {
  constexpr int Q = StableJet<Dim>::Q;
  for (std::size_t owner = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       owner < batch * channels; owner += std::size_t(gridDim.x) * blockDim.x) {
    const std::size_t b = owner / channels, c = owner % channels;
    const double w0 = weight[c], w1 = weight[channels + c], w2 = weight[2 * channels + c];
    double bar[Q];
    #pragma unroll
    for (int j = 0; j < Q; ++j) {
      const std::size_t index = (b * Q + j) * 3;
      // Same exact three-term dot in U3 and F3, without K=16 padding.
      bar[j] = d[index] * w0 + d[index + 1] * w1 + d[index + 2] * w2;
    }
    if constexpr (Fused) {
      double h[Q], result[Q];
      #pragma unroll
      for (int j = 0; j < Q; ++j) h[j] = hidden[(b * Q + j) * channels + c];
      StableJet<Dim>::vjp(h, bar, aux[owner], result);
      #pragma unroll
      for (int j = 0; j < Q; ++j) output[(b * Q + j) * channels + c] = result[j];
    } else {
      #pragma unroll
      for (int j = 0; j < Q; ++j) output[(b * Q + j) * channels + c] = bar[j];
    }
  }
}

static bool valid_size(int dim, std::size_t batch, std::size_t channels) {
  return (dim == 2 || dim == 3) && (channels == 0 || batch <= SIZE_MAX / 20 / sizeof(double) / channels);
}
static bool overlaps(const void* a, std::size_t na, const void* b, std::size_t nb) {
  auto pa = reinterpret_cast<std::uintptr_t>(a), pb = reinterpret_cast<std::uintptr_t>(b);
  return na && nb && (pa <= pb ? pb - pa < na : pa - pb < nb);
}
static unsigned blocks(std::size_t n) {
  return unsigned(n / 128 + (n % 128 != 0) > 65535 ? 65535 : n / 128 + (n % 128 != 0));
}

extern "C" cudaError_t flashns_stable_activation(
    int dim, bool backward, const double* value, const double* adjoint,
    const double* aux_in, double* output, double* aux_out,
    std::size_t batch, std::size_t channels, cudaStream_t stream) {
  if (!valid_size(dim, batch, channels)) return cudaErrorInvalidValue;
  if (batch == 0 || channels == 0) return cudaSuccess;
  if (!value || !output || (backward ? (!adjoint || !aux_in) : !aux_out)) return cudaErrorInvalidValue;
  const std::size_t n = batch * channels;
  const std::size_t bytes = n * (dim == 2 ? 10 : 20) * sizeof(double);
  if (overlaps(output, bytes, value, bytes) || (backward && (overlaps(output, bytes, adjoint, bytes) || overlaps(output, bytes, aux_in, n * sizeof(double)))) || (!backward && (overlaps(aux_out, n * sizeof(double), output, bytes) || overlaps(aux_out, n * sizeof(double), value, bytes)))) return cudaErrorInvalidValue;
  if (dim == 2) {
    if (backward) activation<2, true><<<blocks(n), 128, 0, stream>>>(value, adjoint, aux_in, output, aux_out, batch, channels);
    else activation<2, false><<<blocks(n), 128, 0, stream>>>(value, adjoint, aux_in, output, aux_out, batch, channels);
  } else {
    if (backward) activation<3, true><<<blocks(n), 128, 0, stream>>>(value, adjoint, aux_in, output, aux_out, batch, channels);
    else activation<3, false><<<blocks(n), 128, 0, stream>>>(value, adjoint, aux_in, output, aux_out, batch, channels);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t flashns_stable_tail3(
    int dim, bool fused, const double* d, const double* weight,
    const double* hidden, const double* aux, double* output,
    std::size_t batch, std::size_t channels, cudaStream_t stream) {
  if (!valid_size(dim, batch, channels) || batch > SIZE_MAX / 20 / 3 / sizeof(double)) return cudaErrorInvalidValue;
  if (batch == 0 || channels == 0) return cudaSuccess;
  if (!d || !weight || !output || (fused && (!hidden || !aux))) return cudaErrorInvalidValue;
  const std::size_t n = batch * channels, q = dim == 2 ? 10 : 20;
  const std::size_t bytes = n * q * sizeof(double);
  if (overlaps(output, bytes, d, batch * q * 3 * sizeof(double)) || overlaps(output, bytes, weight, 3 * channels * sizeof(double)) || (fused && (overlaps(output, bytes, hidden, bytes) || overlaps(output, bytes, aux, n * sizeof(double))))) return cudaErrorInvalidValue;
  if (dim == 2) {
    if (fused) tail3<2, true><<<blocks(n), 128, 0, stream>>>(d, weight, hidden, aux, output, batch, channels);
    else tail3<2, false><<<blocks(n), 128, 0, stream>>>(d, weight, hidden, aux, output, batch, channels);
  } else {
    if (fused) tail3<3, true><<<blocks(n), 128, 0, stream>>>(d, weight, hidden, aux, output, batch, channels);
    else tail3<3, false><<<blocks(n), 128, 0, stream>>>(d, weight, hidden, aux, output, batch, channels);
  }
  return cudaGetLastError();
}
