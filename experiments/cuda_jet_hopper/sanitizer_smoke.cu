// Exercise the exact shared libraries, including all stage/tile specializations.
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "../cuda_jet_h100/stable_jet.cuh"

static void check(int code) {
  if (code) { std::cerr << "CUDA/API error " << code << '\n'; std::exit(2); }
}
template<class T> T symbol(void* handle, const char* name) {
  void* pointer = dlsym(handle, name);
  if (!pointer) { std::cerr << dlerror() << '\n'; std::exit(3); }
  return reinterpret_cast<T>(pointer);
}
struct Buffer {
  static constexpr int Guard = 16;
  static constexpr double Sentinel = 93571.125;
  std::size_t count;
  double* base;
  explicit Buffer(std::size_t n): count(n) { check(cudaMalloc(&base, (n + Guard * 2) * sizeof(double))); }
  ~Buffer() { check(cudaFree(base)); }
  double* data() { return base + Guard; }
  void upload(const std::vector<double>& values, cudaStream_t stream) {
    std::vector<double> guarded(count + Guard * 2, Sentinel);
    for (std::size_t i = 0; i < values.size(); ++i) guarded[Guard + i] = values[i];
    check(cudaMemcpyAsync(base, guarded.data(), guarded.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
    check(cudaStreamSynchronize(stream));
  }
  void verify(const std::vector<double>& expected, cudaStream_t stream) {
    std::vector<double> output(count + Guard * 2);
    check(cudaMemcpyAsync(output.data(), base, output.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    for (std::size_t i = 0; i < output.size(); ++i) {
      if (i < Guard || i >= Guard + count) {
        if (output[i] != Sentinel) { std::cerr << "canary changed\n"; std::exit(4); }
      } else {
        double reference = expected[i - Guard];
        if (!std::isfinite(output[i]) || std::abs(output[i] - reference) > 2e-12 + 2e-11 * std::abs(reference)) {
          std::cerr << "mismatch " << i << ' ' << output[i] << ' ' << reference << '\n'; std::exit(5);
        }
      }
    }
  }
};
using Initialize = int (*)();
using Create = int (*)(int, const double*, const double*, int, int, int, void**);
using Destroy = void (*)(void*);
using Launch = int (*)(void*, int, bool, const double*, const double*, double*, cudaStream_t);

int main(int argc, char** argv) {
  if (argc < 2) return 1;
  check(cudaSetDevice(0));
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  int checks = 0;
  for (int argument = 1; argument < argc; ++argument) {
    void* library = dlopen(argv[argument], RTLD_NOW | RTLD_LOCAL);
    if (!library) { std::cerr << dlerror() << '\n'; return 3; }
    auto initialize = symbol<Initialize>(library, "flashns_hopper_initialize");
    auto create = symbol<Create>(library, "flashns_hopper_create");
    auto destroy = symbol<Destroy>(library, "flashns_hopper_destroy");
    auto launch = symbol<Launch>(library, "flashns_hopper_launch");
    check(initialize());
    for (int dim : {2, 3}) for (int batch : {1, 7, 13}) for (int cin : {32, 64}) for (int cout : {32, 64}) {
      int q = dim == 2 ? 10 : 20;
      std::size_t nd = std::size_t(batch) * q * cout, nh = std::size_t(batch) * q * cin;
      std::vector<double> d(nd), w(std::size_t(cout) * cin), h(nh), a(std::size_t(batch) * cin), bar(nh), expected(nh);
      for (std::size_t i = 0; i < d.size(); ++i) d[i] = std::sin(i * .071) * .2;
      for (std::size_t i = 0; i < w.size(); ++i) w[i] = std::cos(i * .317) * .2;
      for (int b = 0; b < batch; ++b) for (int c = 0; c < cin; ++c) {
        double z[20], local_h[20], local_bar[20], local_out[20], aux;
        for (int j = 0; j < q; ++j) {
          z[j] = std::sin(b + c * .3 + j) * .2;
          if (!j && c % 3 == 0) z[j] = c % 2 ? 20 : -20;
          local_bar[j] = 0;
          for (int k = 0; k < cout; ++k) local_bar[j] += d[(std::size_t(b) * q + j) * cout + k] * w[std::size_t(k) * cin + c];
        }
        if (dim == 2) {
          flashns_stable::tanh_fwd_2d3(z, local_h, &aux);
          flashns_stable::tanh_vjp_2d3(local_h, local_bar, aux, local_out);
        } else {
          flashns_stable::tanh_fwd_3d3(z, local_h, &aux);
          flashns_stable::tanh_vjp_3d3(local_h, local_bar, aux, local_out);
        }
        a[std::size_t(b) * cin + c] = aux;
        for (int j = 0; j < q; ++j) {
          auto index = (std::size_t(b) * q + j) * cin + c;
          h[index] = local_h[j]; bar[index] = local_bar[j]; expected[index] = local_out[j];
        }
      }
      Buffer gd(d.size()), gw(w.size()), gh(h.size()), ga(a.size()), output(h.size());
      gd.upload(d, stream); gw.upload(w, stream); gh.upload(h, stream); ga.upload(a, stream); output.upload({}, stream);
      void* plan = nullptr;
      check(create(dim, gd.data(), gw.data(), batch, cout, cin, &plan));
      for (bool fused : {false, true}) {
        for (int repeat = 0; repeat < 3; ++repeat) check(launch(plan, dim, fused, gh.data(), ga.data(), output.data(), stream));
        output.verify(fused ? expected : bar, stream);
        ++checks;
      }
      if (launch(plan, dim, true, gh.data(), ga.data(), gh.data(), stream) != cudaErrorInvalidValue) return 6;
      destroy(plan);
    }
    check(cudaStreamSynchronize(stream));
    dlclose(library);
    std::cout << "checked " << argv[argument] << '\n';
  }
  check(cudaStreamDestroy(stream));
  std::cout << "{\"passed\":true,\"numeric_buffer_checks\":" << checks << ",\"nondefault_stream\":true,\"guard_and_alias_checks\":true}" << std::endl;
}
