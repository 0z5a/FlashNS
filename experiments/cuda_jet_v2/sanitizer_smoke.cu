// Standalone CUDA workload for all four Compute Sanitizer tools, without Torch.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "../cuda_jet/input/jet_primitives.cuh"

extern "C" cudaError_t flashns_dgrad_jet_vjp_fp64(
    int, bool, const double*, const double*, const double*, double*,
    int, int, int, cudaStream_t);

void check(cudaError_t value) {
  if (value != cudaSuccess) {
    std::cerr << cudaGetErrorString(value) << '\n';
    std::exit(2);
  }
}

int main(int argc, char**) {
  const bool short_run = argc > 1;
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  int checked = 0;
  for (int dim : {2, 3}) {
    const int q = dim == 2 ? 10 : 20;
    for (int batch : {1, 7, 13}) {
      if (short_run && batch != 7) continue;
      for (int cout : {3, 65, 128}) {
        const int cin = cout == 3 ? 33 : 65;
        if (short_run && cout != 65) continue;
        const std::size_t nd = std::size_t(batch) * q * cout;
        const std::size_t nh = std::size_t(batch) * q * cin;
        const std::size_t nw = std::size_t(cout) * cin;
        constexpr int guard = 16;
        constexpr double sentinel = -9173.125;
        std::vector<double> d(nd), w(nw), h(nh), expected(nh);
        std::vector<double> initial(nh + guard * 2, sentinel), result(initial.size());
        for (std::size_t i = 0; i < nd; ++i) d[i] = std::sin(i * 0.731) * 0.2;
        for (std::size_t i = 0; i < nw; ++i) w[i] = std::cos(i * 0.317) * 0.2;
        for (int b = 0; b < batch; ++b) {
          for (int c = 0; c < cin; ++c) {
            double z[20], local_h[20];
            for (int j = 0; j < q; ++j) z[j] = std::sin(b + c * 0.3 + j) * 0.2;
            if (dim == 2) flashns::tanh_fwd_2d3(z, local_h);
            else flashns::tanh_fwd_3d3(z, local_h);
            for (int j = 0; j < q; ++j) h[(std::size_t(b) * q + j) * cin + c] = local_h[j];
          }
        }
        double *gd, *gw, *gh, *go;
        check(cudaMalloc(&gd, nd * sizeof(double)));
        check(cudaMalloc(&gw, nw * sizeof(double)));
        check(cudaMalloc(&gh, nh * sizeof(double)));
        check(cudaMalloc(&go, initial.size() * sizeof(double)));
        check(cudaMemcpyAsync(gd, d.data(), nd * sizeof(double), cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(gw, w.data(), nw * sizeof(double), cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(gh, h.data(), nh * sizeof(double), cudaMemcpyHostToDevice, stream));
        for (bool fused : {false, true}) {
          for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < cin; ++c) {
              double bar[20], local_h[20], local_d[20];
              for (int j = 0; j < q; ++j) {
                double value = 0;
                for (int k = 0; k < cout; ++k) value += d[(std::size_t(b) * q + j) * cout + k] * w[std::size_t(k) * cin + c];
                bar[j] = value;
                local_h[j] = h[(std::size_t(b) * q + j) * cin + c];
              }
              if (fused) {
                if (dim == 2) flashns::tanh_vjp_2d3(local_h, bar, local_d);
                else flashns::tanh_vjp_3d3(local_h, bar, local_d);
              }
              for (int j = 0; j < q; ++j) expected[(std::size_t(b) * q + j) * cin + c] = fused ? local_d[j] : bar[j];
            }
          }
          check(cudaMemcpyAsync(go, initial.data(), initial.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
          for (int repeat = 0; repeat < 3; ++repeat) check(flashns_dgrad_jet_vjp_fp64(dim, fused, gd, gw, gh, go + guard, batch, cout, cin, stream));
          check(cudaMemcpyAsync(result.data(), go, result.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
          check(cudaStreamSynchronize(stream));
          for (std::size_t i = 0; i < result.size(); ++i) {
            if (i < guard || i >= nh + guard) {
              if (result[i] != sentinel) return 3;
            } else {
              const double oracle = expected[i - guard];
              if (!std::isfinite(result[i]) || std::abs(result[i] - oracle) > 2e-12 + 2e-11 * std::abs(oracle)) return 4;
            }
          }
          ++checked;
        }
        if (flashns_dgrad_jet_vjp_fp64(dim, true, gd, gw, gh, gh, batch, cout, cin, stream) != cudaErrorInvalidValue) return 5;
        check(cudaFree(gd)); check(cudaFree(gw)); check(cudaFree(gh)); check(cudaFree(go));
      }
    }
  }
  check(flashns_dgrad_jet_vjp_fp64(2, true, nullptr, nullptr, nullptr, nullptr, 0, 1, 1, stream));
  if (flashns_dgrad_jet_vjp_fp64(4, true, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, stream) != cudaErrorInvalidValue) return 6;
  if (flashns_dgrad_jet_vjp_fp64(2, true, nullptr, nullptr, nullptr, nullptr, 1, 1, 1, stream) != cudaErrorInvalidValue) return 7;
  if (flashns_dgrad_jet_vjp_fp64(2, true, nullptr, nullptr, nullptr, nullptr, 1, 1, 64 * 65535 + 1, stream) != cudaErrorInvalidValue) return 8;
  check(cudaStreamDestroy(stream));
  std::cout << "{\"passed\":true,\"numeric_cases\":" << checked << ",\"repeated_launches_per_case\":3,\"guards_and_invalid_inputs\":true}" << std::endl;
}
