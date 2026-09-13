// Exercise the exact measured shared libraries on a non-default stream.
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "stable_jet.cuh"

void check(cudaError_t code) {
  if (code != cudaSuccess) {
    std::cerr << cudaGetErrorString(code) << '\n';
    std::exit(2);
  }
}
using Activation = cudaError_t (*)(int, bool, const double*, const double*,
    const double*, double*, double*, std::size_t, std::size_t, cudaStream_t);
using Tail = cudaError_t (*)(int, bool, const double*, const double*,
    const double*, const double*, double*, std::size_t, std::size_t, cudaStream_t);
using Dgrad = cudaError_t (*)(int, bool, const double*, const double*,
    const double*, const double*, double*, int, int, int, cudaStream_t);

void* library(const std::string& path) {
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) { std::cerr << dlerror() << '\n'; std::exit(3); }
  return handle;
}
template<class T> T symbol(void* handle, const char* name) {
  void* result = dlsym(handle, name);
  if (!result) { std::cerr << dlerror() << '\n'; std::exit(3); }
  return reinterpret_cast<T>(result);
}

struct Buffer {
  static constexpr int guard = 16;
  static constexpr double sentinel = -9173.125;
  std::size_t count;
  double* base;
  explicit Buffer(std::size_t n): count(n) {
    check(cudaMalloc(&base, (n + guard * 2) * sizeof(double)));
  }
  ~Buffer() { check(cudaFree(base)); }
  double* data() { return base + guard; }
  void upload(const std::vector<double>& values, cudaStream_t stream) {
    std::vector<double> padded(count + guard * 2, sentinel);
    for (std::size_t i = 0; i < values.size(); ++i) padded[i + guard] = values[i];
    check(cudaMemcpyAsync(base, padded.data(), padded.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
    check(cudaStreamSynchronize(stream));
  }
  void verify(const std::vector<double>& expected, cudaStream_t stream) {
    std::vector<double> result(count + guard * 2);
    check(cudaMemcpyAsync(result.data(), base, result.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    for (std::size_t i = 0; i < result.size(); ++i) {
      if (i < guard || i >= count + guard) {
        if (result[i] != sentinel) { std::cerr << "guard changed\n"; std::exit(4); }
      } else {
        double reference = expected[i - guard];
        if (!std::isfinite(result[i]) || std::abs(result[i] - reference) > 2e-12 + 2e-11 * std::abs(reference)) {
          std::cerr << "numeric mismatch " << i << ' ' << result[i] << ' ' << reference << '\n';
          std::exit(5);
        }
      }
    }
  }
};

int main(int argc, char** argv) {
  if (argc != 3) return 1;
  std::string directory = argv[1];
  check(cudaSetDevice(std::stoi(argv[2])));
  void* stable = library(directory + "/libstable.so");
  void* n32 = library(directory + "/libdgrad_n32.so");
  void* n64 = library(directory + "/libdgrad_n64.so");
  auto activation = symbol<Activation>(stable, "flashns_stable_activation");
  auto tail = symbol<Tail>(stable, "flashns_stable_tail3");
  std::vector<Dgrad> dgrads = {symbol<Dgrad>(n32, "flashns_stable_dgrad_vjp"), symbol<Dgrad>(n64, "flashns_stable_dgrad_vjp")};
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  int checked = 0;
  for (int dim : {2, 3}) for (int batch : {1, 7, 13}) for (int cout : {3, 65, 128}) {
    const int q = dim == 2 ? 10 : 20, cin = cout == 3 ? 33 : 65;
    const std::size_t nd = std::size_t(batch) * q * cout, nh = std::size_t(batch) * q * cin;
    const std::size_t nw = std::size_t(cout) * cin, na = std::size_t(batch) * cin;
    std::vector<double> d(nd), w(nw), z(nh), h(nh), aux(na), bar(nh), vjp(nh);
    for (std::size_t i = 0; i < nd; ++i) d[i] = std::sin(i * 0.731) * 0.2;
    for (std::size_t i = 0; i < nw; ++i) w[i] = std::cos(i * 0.317) * 0.2;
    for (int b = 0; b < batch; ++b) for (int c = 0; c < cin; ++c) {
      double local_z[20], local_h[20], local_bar[20], local_vjp[20], a;
      for (int j = 0; j < q; ++j) {
        local_z[j] = std::sin(b + c * 0.3 + j) * 0.2;
        if (j == 0 && c % 3 == 0) local_z[j] = c % 2 ? 20. : -20.;
        z[(std::size_t(b) * q + j) * cin + c] = local_z[j];
        local_bar[j] = 0;
        for (int k = 0; k < cout; ++k) local_bar[j] += d[(std::size_t(b) * q + j) * cout + k] * w[std::size_t(k) * cin + c];
      }
      if (dim == 2) flashns_stable::tanh_fwd_2d3(local_z, local_h, &a);
      else flashns_stable::tanh_fwd_3d3(local_z, local_h, &a);
      if (dim == 2) flashns_stable::tanh_vjp_2d3(local_h, local_bar, a, local_vjp);
      else flashns_stable::tanh_vjp_3d3(local_h, local_bar, a, local_vjp);
      aux[std::size_t(b) * cin + c] = a;
      for (int j = 0; j < q; ++j) {
        const std::size_t i = (std::size_t(b) * q + j) * cin + c;
        h[i] = local_h[j]; bar[i] = local_bar[j]; vjp[i] = local_vjp[j];
      }
    }
    Buffer gd(nd), gw(nw), gz(nh), gh(nh), ga(na), gb(nh), go(nh);
    gd.upload(d, stream); gw.upload(w, stream); gz.upload(z, stream);
    gh.upload({}, stream); ga.upload({}, stream); gb.upload(bar, stream); go.upload({}, stream);
    check(activation(dim, false, gz.data(), nullptr, nullptr, gh.data(), ga.data(), batch, cin, stream));
    gh.verify(h, stream); ga.verify(aux, stream); checked += 2;
    check(activation(dim, true, gh.data(), gb.data(), ga.data(), go.data(), nullptr, batch, cin, stream));
    go.verify(vjp, stream); ++checked;
    for (bool fused : {false, true}) {
      for (auto dgrad : dgrads) {
        go.upload({}, stream);
        for (int repeat = 0; repeat < 3; ++repeat) check(dgrad(dim, fused, gd.data(), gw.data(), gh.data(), ga.data(), go.data(), batch, cout, cin, stream));
        go.verify(fused ? vjp : bar, stream); ++checked;
        if (dgrad(dim, true, gd.data(), gw.data(), gh.data(), ga.data(), gh.data(), batch, cout, cin, stream) != cudaErrorInvalidValue) return 6;
      }
      if (cout == 3) {
        go.upload({}, stream);
        for (int repeat = 0; repeat < 3; ++repeat) check(tail(dim, fused, gd.data(), gw.data(), gh.data(), ga.data(), go.data(), batch, cin, stream));
        go.verify(fused ? vjp : bar, stream); ++checked;
      }
    }
    if (activation(dim, true, gh.data(), gb.data(), ga.data(), gh.data(), nullptr, batch, cin, stream) != cudaErrorInvalidValue) return 7;
    if (tail(dim, true, gd.data(), gw.data(), gh.data(), ga.data(), gh.data(), batch, cin, stream) != cudaErrorInvalidValue) return 8;
  }
  check(activation(2, false, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 1, stream));
  check(tail(2, true, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 1, stream));
  for (auto dgrad : dgrads) check(dgrad(2, true, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 1, 1, stream));
  check(cudaStreamDestroy(stream));
  dlclose(n64); dlclose(n32); dlclose(stable);
  std::cout << "{\"passed\":true,\"numeric_buffer_checks\":" << checked << ",\"nondefault_stream\":true,\"guard_and_alias_checks\":true}" << std::endl;
}
