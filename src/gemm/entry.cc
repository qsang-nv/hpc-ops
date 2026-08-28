// Copyright (C) 2026 Tencent.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>
#include <optional>

#include "src/gemm/gemm.h"
#include "src/utils/utils.h"

namespace hpc {
namespace gemm {
namespace {

struct KernelConfig {
  int split_k;
  int k_warpgroup_n;
  int kTileM;  // 16 for small-m MMA, 64 for large-m MMA
};

inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }

inline int normalized_m(int m, int n, int k) {
  constexpr int kRefN = 192;
  constexpr int kRefK = 4096;
  return static_cast<int>(
      (static_cast<int64_t>(m) * n * kRefK + static_cast<int64_t>(kRefN) * k - 1) /
      (static_cast<int64_t>(kRefN) * k));
}

inline int select_split_k_by_work(int norm_m) {
  if (norm_m <= 64) {
    return 8;
  }
  if (norm_m <= 144) {
    return 4;
  }
  if (norm_m <= 304) {
    return 2;
  }
  return 1;
}

inline int select_tile16_wgn(int m, int n, int split_k) {
  constexpr int kTileM = 16;
  constexpr int kTileN = 64;
  constexpr int kWgn2 = 2;
  constexpr int kTargetTiles = 64;
  const int tiles_with_wgn2 = ceil_div_int(m, kTileM) * ceil_div_int(n, kTileN * kWgn2) * split_k;
  return tiles_with_wgn2 < kTargetTiles ? 1 : 2;
}

inline KernelConfig select_config(int m, int n, int k, bool use_splitk) {
  const int norm_m = normalized_m(m, n, k);

  if (norm_m > 624 && norm_m <= 832) {
    return {2, 1, 64};
  }
  if (norm_m > 832 && norm_m <= 896) {
    return {2, 2, 16};
  }
  if (norm_m > 1024 && norm_m <= 1088) {
    return {1, 2, 16};
  }
  if (norm_m > 1088 && norm_m <= 1152) {
    return {4, 1, 64};
  }
  if (norm_m > 1152 && norm_m <= 1536) {
    return {1, 1, 64};
  }
  if (norm_m > 1536 && norm_m <= 2048) {
    return {4, 1, 64};
  }
  if (norm_m > 2048) {
    return {1, 1, 64};
  }

  // kTileM=16 path: select split_k by workload, then wgn by occupancy.
  const int split_k = select_split_k_by_work(norm_m);
  return {split_k, select_tile16_wgn(m, n, split_k), 16};
}

}  // namespace

torch::Tensor gemm_bf16xfp32_entry(const torch::Tensor &x, const torch::Tensor &w_high,
                                   const torch::Tensor &w_low, double scale, bool use_fp32_output,
                                   bool use_splitk, std::optional<torch::Tensor> split_flag) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.is_contiguous(), "x tensor must be contiguous");
  TORCH_CHECK(w_high.is_contiguous(), "w_high tensor must be contiguous");
  TORCH_CHECK(w_low.is_contiguous(), "w_low tensor must be contiguous");

  TORCH_CHECK(x.dtype() == torch::kBFloat16, "x dtype must be bfloat16");
  TORCH_CHECK(w_high.dtype() == torch::kBFloat16, "w_high dtype must be bfloat16");
  TORCH_CHECK(w_low.dtype() == torch::kBFloat16, "w_low dtype must be bfloat16");

  int m = x.size(0);
  int k = x.size(1);
  int n = w_high.size(0);

  TORCH_CHECK(n % 64 == 0, "n must to be divided by 64.");

  auto options = x.options();

  auto out_dtype = torch::kBFloat16;
  if (use_fp32_output) {
    out_dtype = torch::kFloat32;
  }

  KernelConfig cfg = select_config(m, n, k, use_splitk);

  torch::Tensor split_y;
  torch::Tensor split_flag_tensor;
  void *split_y_ptr = nullptr;
  void *split_flag_ptr = nullptr;

  if (cfg.split_k != 1) {
    split_y = torch::empty({cfg.split_k, m, n}, options.dtype(torch::kFloat32));
    if (split_flag.has_value()) {
      split_flag_tensor = split_flag.value();
    } else {
      const int tile_m = cfg.kTileM;
      const int tile_n = 64 * cfg.k_warpgroup_n;
      split_flag_tensor = torch::zeros({(m + tile_m - 1) / tile_m, (n + tile_n - 1) / tile_n},
                                       options.dtype(torch::kInt32));
    }
    split_y_ptr = split_y.mutable_data_ptr();
    split_flag_ptr = split_flag_tensor.mutable_data_ptr();
  }

  torch::Tensor y = torch::empty({m, n}, options.dtype(out_dtype));

  const auto *x_ptr = x.const_data_ptr();
  const auto *w_high_ptr = w_high.const_data_ptr();
  const auto *w_low_ptr = w_low.const_data_ptr();
  auto *y_ptr = y.mutable_data_ptr();

  bool running = false;
  HPC_ARCH_DISPATCH(
      "gemm_bf16xfp32", 90,
      running = gemm_bf16xfp32_async(y_ptr, split_y_ptr, split_flag_ptr, x_ptr, w_high_ptr,
                                     w_low_ptr, m, n, k, scale, use_fp32_output, cfg.split_k,
                                     cfg.kTileM, cfg.k_warpgroup_n, stream));

  TORCH_CHECK(running, "gemm_bf16xfp32 launch failed!");

  return y;
}

torch::Tensor gated_mla_gemm_entry(const torch::Tensor &x, const torch::Tensor &weight,
                                   const torch::Tensor &atten_out) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.is_contiguous(), "x tensor must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor must be contiguous");
  TORCH_CHECK(atten_out.is_contiguous(), "atten_out tensor must be contiguous");
  TORCH_CHECK(x.dtype() == torch::kBFloat16, "x dtype must be bfloat16");
  TORCH_CHECK(weight.dtype() == torch::kBFloat16, "weight dtype must be bfloat16");
  TORCH_CHECK(atten_out.dtype() == torch::kBFloat16, "atten_out dtype must be bfloat16");
  TORCH_CHECK(x.size(1) == weight.size(1), "x and weight must share the same k");
  TORCH_CHECK(x.size(0) == atten_out.size(0), "x and atten_out must share the same m");
  TORCH_CHECK(weight.size(0) == atten_out.size(1), "weight and atten_out must shared the same n");
  TORCH_CHECK(weight.size(0) % 256 == 0, "weight.size(0) (n) must be a multiple of 256");

  int m = x.size(0);
  int k = x.size(1);
  int n = weight.size(0);

  auto options = x.options();
  torch::Tensor y = torch::empty({m, n}, options.dtype(torch::kBFloat16));

  const auto *x_ptr = x.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();
  const auto *atten_out_ptr = atten_out.const_data_ptr();
  auto *y_ptr = y.mutable_data_ptr();

  bool running = false;
  HPC_ARCH_DISPATCH(
      "gated_mla_gemm", 100,
      running = gated_mla_gemm_async(y_ptr, x_ptr, weight_ptr, atten_out_ptr, m, n, k, stream), 103,
      running = gated_mla_gemm_async(y_ptr, x_ptr, weight_ptr, atten_out_ptr, m, n, k, stream));
  TORCH_CHECK(running, "gated_mla_gemm_async launch failed!");

  return y;
}

}  // namespace gemm
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "gemm_bf16xfp32(Tensor x, Tensor w_high, Tensor w_low, "
      "float scale, bool use_fp32_output, bool use_splitk, Tensor? split_flag) -> (Tensor)");
  m.impl("gemm_bf16xfp32", torch::kCUDA, &hpc::gemm::gemm_bf16xfp32_entry);

  m.def("gated_mla_gemm(Tensor x, Tensor w, Tensor x2) -> Tensor");
  m.impl("gated_mla_gemm", torch::kCUDA, &hpc::gemm::gated_mla_gemm_entry);
}
