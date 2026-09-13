// Bounded FP64 cuBLASLt plans. Planning is separate from stream submission.
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <vector>

struct Plan {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  std::vector<cublasLtMatmulHeuristicResult_t> candidates;
  ~Plan() {
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (operation) cublasLtMatmulDescDestroy(operation);
    if (handle) cublasLtDestroy(handle);
  }
};

static void check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) throw int(status);
}

static void layout(cublasLtMatrixLayout_t* out, int64_t rows, int64_t columns) {
  check(cublasLtMatrixLayoutCreate(out, CUDA_R_64F, rows, columns, columns));
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  check(cublasLtMatrixLayoutSetAttribute(*out, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
}

extern "C" int flashns_lt_create(int64_t m, int64_t n, int64_t k, bool transpose_a,
    bool transpose_b, uint32_t alignment_a, uint32_t alignment_b,
    std::size_t workspace_limit, int requested_candidates, void** result) {
  if (!result) return CUBLAS_STATUS_INVALID_VALUE;
  *result = nullptr;
  if (m <= 0 || n <= 0 || k <= 0 || requested_candidates < 1 || requested_candidates > 64)
    return CUBLAS_STATUS_INVALID_VALUE;
  try {
    auto p = std::make_unique<Plan>();
    check(cublasLtCreate(&p->handle));
    check(cublasLtMatmulDescCreate(&p->operation, CUBLAS_COMPUTE_64F, CUDA_R_64F));
    cublasOperation_t ta = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t tb = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    check(cublasLtMatmulDescSetAttribute(p->operation, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    check(cublasLtMatmulDescSetAttribute(p->operation, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    layout(&p->a, transpose_a ? k : m, transpose_a ? m : k);
    layout(&p->b, transpose_b ? n : k, transpose_b ? k : n);
    layout(&p->c, m, n);
    check(cublasLtMatmulPreferenceCreate(&p->preference));
    check(cublasLtMatmulPreferenceSetAttribute(p->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit, sizeof(workspace_limit)));
    check(cublasLtMatmulPreferenceSetAttribute(p->preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &alignment_a, sizeof(alignment_a)));
    check(cublasLtMatmulPreferenceSetAttribute(p->preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &alignment_b, sizeof(alignment_b)));
    uint32_t output_alignment = 256;
    check(cublasLtMatmulPreferenceSetAttribute(p->preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &output_alignment, sizeof(output_alignment)));
    check(cublasLtMatmulPreferenceSetAttribute(p->preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &output_alignment, sizeof(output_alignment)));
    p->candidates.resize(requested_candidates);
    int returned = 0;
    check(cublasLtMatmulAlgoGetHeuristic(p->handle, p->operation, p->a, p->b, p->c,
        p->c, p->preference, requested_candidates, p->candidates.data(), &returned));
    p->candidates.resize(returned);
    *result = p.release();
    return CUBLAS_STATUS_SUCCESS;
  } catch (int status) {
    return status;
  } catch (...) {
    return -1;
  }
}

extern "C" int flashns_lt_count(void* plan) {
  return plan ? static_cast<Plan*>(plan)->candidates.size() : -1;
}

extern "C" int flashns_lt_info(void* plan, int index, int* attributes,
    std::size_t* workspace_bytes, float* waves) {
  if (!plan || !attributes || !workspace_bytes || !waves) return CUBLAS_STATUS_INVALID_VALUE;
  auto* p = static_cast<Plan*>(plan);
  if (index < 0 || index >= int(p->candidates.size())) return CUBLAS_STATUS_INVALID_VALUE;
  const auto& item = p->candidates[index];
  *workspace_bytes = item.workspaceSize;
  *waves = item.wavesCount;
  const cublasLtMatmulAlgoConfigAttributes_t names[] = {
    CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID,
    CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
    CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, CUBLASLT_ALGO_CONFIG_STAGES_ID
  };
  for (int i = 0; i < 6; ++i) {
    attributes[i] = -1;
    std::size_t bytes = 0;
    if (cublasLtMatmulAlgoConfigGetAttribute(&item.algo, names[i], &attributes[i],
        sizeof(attributes[i]), &bytes) != CUBLAS_STATUS_SUCCESS) attributes[i] = -1;
  }
  return int(item.state);
}

extern "C" int flashns_lt_run(void* plan, int index, const double* a, const double* b,
    double* output, void* workspace, std::size_t workspace_bytes, cudaStream_t stream) {
  if (!plan || !a || !b || !output) return CUBLAS_STATUS_INVALID_VALUE;
  auto* p = static_cast<Plan*>(plan);
  if (index < 0 || index >= int(p->candidates.size())) return CUBLAS_STATUS_INVALID_VALUE;
  const auto& item = p->candidates[index];
  if (item.state != CUBLAS_STATUS_SUCCESS || workspace_bytes < item.workspaceSize ||
      (item.workspaceSize && !workspace)) return CUBLAS_STATUS_INVALID_VALUE;
  const double alpha = 1, beta = 0;
  return int(cublasLtMatmul(p->handle, p->operation, &alpha, a, p->a, b, p->b,
      &beta, output, p->c, output, p->c, &item.algo, workspace, workspace_bytes, stream));
}

extern "C" void flashns_lt_destroy(void* plan) {
  delete static_cast<Plan*>(plan);
}
