// Standalone driver-API validation of the exact MathDx cubin, without PyTorch.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) { std::fprintf(stderr, "CUDA %d: %s\n", int(status), cudaGetErrorString(status)); std::exit(2); }
}
static void driver_check(CUresult status) {
  if (status != CUDA_SUCCESS) { std::fprintf(stderr, "Driver %d\n", int(status)); std::exit(3); }
}

int main(int argc, char** argv) {
  if (argc != 2) return 4;
  cuda_check(cudaSetDevice(0));
  cuda_check(cudaFree(nullptr));
  CUmodule module;
  driver_check(cuModuleLoad(&module, argv[1]));
  cudaStream_t stream;
  cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  constexpr double guard = 0x1.abcdef12345p+100;
  const int shapes[][2] = {{32,2},{32,3},{32,32},{3,32},{64,2},{64,3},{64,64},{3,64}};
  int cases = 0;
  for (int tile : {32, 64}) for (const auto& shape : shapes) {
    const int n = shape[0], k = shape[1];
    char name[64];
    std::snprintf(name, sizeof(name), "dx_%d_%d_%d", tile, n, k);
    CUfunction function;
    driver_check(cuModuleGetFunction(&function, module, name));
    std::snprintf(name, sizeof(name), "cfg_%d_%d_%d", tile, n, k);
    CUdeviceptr config_address;
    size_t config_size;
    driver_check(cuModuleGetGlobal(&config_address, &config_size, module, name));
    if (config_size != sizeof(int) * 4) return 5;
    int config[4];
    driver_check(cuMemcpyDtoH(config, config_address, config_size));
    driver_check(cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, config[3]));
    for (int transpose : {0, 1}) for (int64_t rows : {0, 1, 7, 31, 33, 63, 65, 129}) {
      std::vector<double> a(rows * k), b(k * n), host(rows * n + 16, guard);
      for (size_t i = 0; i < a.size(); ++i) a[i] = (int(i % 17) - 8) / 32.0;
      for (size_t i = 0; i < b.size(); ++i) b[i] = (int(i % 13) - 6) / 64.0;
      for (int64_t i = 0; i < rows * n; ++i) host[8 + i] = NAN;
      double *da, *db, *storage;
      cuda_check(cudaMalloc(&da, a.empty() ? 8 : a.size() * sizeof(double)));
      cuda_check(cudaMalloc(&db, b.size() * sizeof(double)));
      cuda_check(cudaMalloc(&storage, host.size() * sizeof(double)));
      double* output = storage + 8;
      if (!a.empty()) cuda_check(cudaMemcpyAsync(da, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
      cuda_check(cudaMemcpyAsync(db, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
      cuda_check(cudaMemcpyAsync(storage, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
      if (rows) {
        void* parameters[] = {&da, &db, &output, &rows, &transpose};
        driver_check(cuLaunchKernel(function, (rows + tile - 1) / tile, 1, 1,
          config[0], config[1], config[2], config[3], reinterpret_cast<CUstream>(stream), parameters, nullptr));
      }
      cuda_check(cudaMemcpyAsync(host.data(), storage, host.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
      cuda_check(cudaStreamSynchronize(stream));
      for (int i = 0; i < 8; ++i) if (host[i] != guard || host[rows * n + 8 + i] != guard) return 6;
      for (int64_t i = 0; i < rows; ++i) for (int j = 0; j < n; ++j) {
        double expected = 0;
        for (int p = 0; p < k; ++p) expected += a[i * k + p] * b[transpose ? j * k + p : p * n + j];
        double actual = host[8 + i * n + j];
        if (!std::isfinite(actual) || std::fabs(actual - expected) > 2e-12 + 2e-11 * std::fabs(expected)) {
          std::fprintf(stderr, "Mismatch tile=%d n=%d k=%d tb=%d rows=%lld at %lld,%d: %.17g / %.17g\n", tile,n,k,transpose,(long long)rows,(long long)i,j,actual,expected);
          return 7;
        }
      }
      cuda_check(cudaFree(da)); cuda_check(cudaFree(db)); cuda_check(cudaFree(storage));
      ++cases;
    }
  }
  cuda_check(cudaStreamDestroy(stream));
  driver_check(cuModuleUnload(module));
  std::printf("PASS MathDx: %d cases; 16 kernels, both B layouts, empty/boundary rows, output guards, nondefault stream\n", cases);
  return 0;
}
