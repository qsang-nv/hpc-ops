// Copyright (C) 2026 Tencent.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>

#include "src/group_gemm/build_task_map.h"
#include "src/group_gemm/group_gemm.h"
#include "src/group_gemm/group_gemm_cp_async.h"
#include "src/utils/utils.h"

namespace hpc {
namespace group_gemm {

torch::Tensor group_gemm_fp8_entry(const torch::Tensor &x, const torch::Tensor &weight,
                                   const torch::Tensor &seqlens, const torch::Tensor &cu_seqlens,
                                   const torch::Tensor &y_scale,
                                   const int64_t num_seq_per_group_avg,
                                   std::optional<torch::Tensor> output,
                                   std::optional<torch::Tensor> tma_desc,
                                   std::optional<torch::Tensor> task_map_workspace) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.device().is_cuda(), "x tensor must be cuda");
  TORCH_CHECK(weight.device().is_cuda(), "weight tensor must be cuda");
  TORCH_CHECK(seqlens.device().is_cuda(), "seqlens tensor must be cuda");
  TORCH_CHECK(cu_seqlens.device().is_cuda(), "cu_seqlens tensor must be cuda");
  TORCH_CHECK(x.dtype() == torch::kFloat8_e4m3fn && weight.dtype() == torch::kFloat8_e4m3fn,
              "x and weight dtype must be fp8_e4m3");
  TORCH_CHECK(seqlens.dtype() == torch::kInt32 && cu_seqlens.dtype() == torch::kInt32,
              "seqlens and cu_seqlens dtype must be int32");
  TORCH_CHECK(y_scale.dtype() == torch::kFloat32, "y_scale dtype must be float32");
  TORCH_CHECK(x.is_contiguous(), "x tensor a must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor a must be contiguous");
  TORCH_CHECK(seqlens.size(0) == weight.size(0),
              "seqlens and weight must share the same num_group");
  TORCH_CHECK(x.size(1) == weight.size(2), "x and weight must share the same k");

  int m = x.size(0);
  int k = x.size(1);
  int n = weight.size(1);
  int num_group = seqlens.size(0);

  auto options = x.options();
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({m, n}, options.dtype(torch::kBFloat16));
  }

  torch::Tensor tmas;
  bool update_tma = true;
  if (tma_desc.has_value()) {
    tmas = tma_desc.value();
    update_tma = false;
  } else {
    tmas = torch::empty({num_group * 2, 128}, options);
  }

  int num_waves = 0;
  torch::Tensor task_map;
  void *task_map_ptr = nullptr;

  if (num_seq_per_group_avg <= 8 && update_tma && task_map_workspace.has_value()) {
    num_waves = task_map_workspace.value().size(0);
    task_map_ptr = task_map_workspace.value().mutable_data_ptr();
  }

  torch::Tensor tiles = torch::empty({num_group}, options.dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty({num_group + 1}, options.dtype(torch::kInt32));

  const auto *x_ptr = x.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();
  const auto *seqlens_ptr = seqlens.const_data_ptr();
  const auto *cu_seqlens_ptr = cu_seqlens.const_data_ptr();
  const auto *yscale_ptr = y_scale.const_data_ptr();
  auto *tmas_ptr = tmas.mutable_data_ptr();
  auto *y_ptr = y.mutable_data_ptr();

  auto *tiles_ptr = tiles.mutable_data_ptr();
  auto *cu_tiles_ptr = cu_tiles.mutable_data_ptr();

  HPC_ARCH_DISPATCH(
      "group_gemm_fp8", 90,
      group_gemm_fp8_async(y_ptr, x_ptr, weight_ptr, seqlens_ptr, cu_seqlens_ptr, yscale_ptr,
                           tmas_ptr, tiles_ptr, cu_tiles_ptr, task_map_ptr, num_waves, num_group, m,
                           n, k, num_seq_per_group_avg, update_tma, false, stream));

  return y;
}

torch::Tensor group_gemm_blockwise_fp8_entry(
    const torch::Tensor &x, const torch::Tensor &weight, const torch::Tensor &seqlens,
    const torch::Tensor &cu_seqlens, const torch::Tensor &x_scale, const torch::Tensor &w_scale,
    const int64_t num_seq_per_group_avg, std::optional<torch::Tensor> output,
    std::optional<torch::Tensor> tma_desc, std::optional<torch::Tensor> task_map_workspace) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.device().is_cuda(), "x tensor must be cuda");
  TORCH_CHECK(weight.device().is_cuda(), "weight tensor must be cuda");
  TORCH_CHECK(seqlens.device().is_cuda(), "seqlens tensor must be cuda");
  TORCH_CHECK(cu_seqlens.device().is_cuda(), "cu_seqlens tensor must be cuda");
  TORCH_CHECK(x.is_contiguous(), "x tensor a must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor a must be contiguous");
  TORCH_CHECK(x.dtype() == torch::kFloat8_e4m3fn && weight.dtype() == torch::kFloat8_e4m3fn,
              "x and weight dtype must be fp8_e4m3");
  TORCH_CHECK(seqlens.dtype() == torch::kInt32 && cu_seqlens.dtype() == torch::kInt32,
              "seqlens and cu_seqlens dtype must be int32");
  TORCH_CHECK(x_scale.dtype() == torch::kFloat32 && w_scale.dtype() == torch::kFloat32,
              "x_scale and w_scale dtype must be float32");
  TORCH_CHECK(seqlens.size(0) == weight.size(0),
              "seqlens and weight must share the same num_group");
  TORCH_CHECK(x.size(1) == weight.size(2), "x and weight must share the same k");
  TORCH_CHECK(w_scale.size(2) % 4 == 0, "w_scale must be multiple of 4");

  int m = x.size(0);
  int k = x.size(1);
  int n = weight.size(1);
  int m_pad = x_scale.size(1);
  int num_block_k_pad4 = w_scale.size(2);
  int num_group = seqlens.size(0);

  auto options = x.options();
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({m, n}, options.dtype(torch::kBFloat16));
  }

  torch::Tensor tmas;
  bool update_tma = true;
  if (tma_desc.has_value()) {
    tmas = tma_desc.value();
    update_tma = false;
  } else {
    tmas = torch::empty({num_group * 2, 128}, options);
  }

  int num_waves = 0;
  torch::Tensor task_map;
  void *task_map_ptr = nullptr;

  if (num_seq_per_group_avg <= 8 && update_tma && task_map_workspace.has_value()) {
    num_waves = task_map_workspace.value().size(0);
    task_map_ptr = task_map_workspace.value().mutable_data_ptr();
  }

  torch::Tensor tiles = torch::empty({num_group}, options.dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty({num_group + 1}, options.dtype(torch::kInt32));

  const auto *x_ptr = x.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();
  const auto *seqlens_ptr = seqlens.const_data_ptr();
  const auto *cu_seqlens_ptr = cu_seqlens.const_data_ptr();
  const auto *xscale_ptr = x_scale.const_data_ptr();
  const auto *wscale_ptr = w_scale.const_data_ptr();
  auto *tmas_ptr = tmas.mutable_data_ptr();
  auto *y_ptr = y.mutable_data_ptr();

  auto *tiles_ptr = tiles.mutable_data_ptr();
  auto *cu_tiles_ptr = cu_tiles.mutable_data_ptr();

  HPC_ARCH_DISPATCH(
      "group_gemm_blockwise_fp8", 90,
      group_gemm_blockwise_fp8_async(
          y_ptr, x_ptr, weight_ptr, seqlens_ptr, cu_seqlens_ptr, xscale_ptr, wscale_ptr, tmas_ptr,
          tiles_ptr, cu_tiles_ptr, task_map_ptr, num_waves, num_group, m, n, k, m_pad,
          num_block_k_pad4, num_seq_per_group_avg, update_tma, false, stream));

  return y;
}

torch::Tensor reformat_x_scale_entry(const torch::Tensor &x_scale, const torch::Tensor &seqlens,
                                     const torch::Tensor &cu_seqlens,
                                     std::optional<torch::Tensor> out_x_scale,
                                     const int64_t num_seq_per_group_avg) {
  auto stream = at::cuda::getCurrentCUDAStream(x_scale.get_device());
  TORCH_CHECK(x_scale.device().is_cuda(), "x_scale tensor must be cuda");
  TORCH_CHECK(seqlens.device().is_cuda(), "seqlens tensor must be cuda");
  TORCH_CHECK(cu_seqlens.device().is_cuda(), "cu_seqlens tensor must be cuda");
  TORCH_CHECK(x_scale.is_contiguous(), "x_scale tensor a must be contiguous");
  TORCH_CHECK(seqlens.is_contiguous(), "seqlens tensor a must be contiguous");
  TORCH_CHECK(cu_seqlens.is_contiguous(), "cu_seqlens tensor a must be contiguous");

  int m = x_scale.size(0);
  int n = x_scale.size(1);
  TORCH_CHECK(n == 16 || n == 32 || n == 56,
              "n must be 16, 32 or 56(for group gemm k=2048, k=4096 or k=7168)");

  int num_group = seqlens.size(0);
  int tilem = 0;
  if (num_seq_per_group_avg <= 8) {
    tilem = 8;
  } else if (num_seq_per_group_avg <= 16) {
    tilem = 16;
  } else if (num_seq_per_group_avg <= 32) {
    tilem = 32;
  } else if (num_seq_per_group_avg <= 48) {
    tilem = 48;
  } else {
    tilem = 64;
  }
  int num_seq_pad_per_group = m / num_group;
  TORCH_CHECK(num_seq_pad_per_group % tilem == 0,
              "The sparse pad length of x_scale for each group must be aligned to multiple of "
              "8/16/32/48/64 according to num_seq_per_group_avg");

  torch::Tensor output;
  if (out_x_scale.has_value()) {
    output = out_x_scale.value();
  } else {
    output = torch::empty({n, m}, x_scale.options());
  }

  const auto *xscale_ptr = x_scale.const_data_ptr();
  const auto *seqlens_ptr = seqlens.const_data_ptr();
  const auto *cu_seqlens_ptr = cu_seqlens.const_data_ptr();
  auto *output_ptr = output.mutable_data_ptr();

  HPC_ARCH_DISPATCH("reformat_x_scale", 90,
                    reformat_x_scale_async(output_ptr, xscale_ptr, seqlens_ptr, cu_seqlens_ptr,
                                           num_group, m, n, tilem, stream));

  return output;
}

}  // namespace group_gemm

namespace group_gemm_cp_async {

static int pick_tile_m(int num_seq_per_group_avg) {
  if (num_seq_per_group_avg <= 8) {
    return 8;
  }
  if (num_seq_per_group_avg <= 16) {
    return 16;
  }
  if (num_seq_per_group_avg <= 32) {
    return 32;
  }
  if (num_seq_per_group_avg <= 48) {
    return 48;
  }
  return 64;
}

constexpr int kTileN = 64;

static int compute_task_map_len(int total_tokens, int num_group, int n, int tile_m) {
  int num_tile_n = (n + kTileN - 1) / kTileN;
  int max_tile_m = total_tokens / tile_m + num_group;
  return max_tile_m * num_tile_n;
}

torch::Tensor group_gemm_fp8_entry(const torch::Tensor &x, const torch::Tensor &weight,
                                   const torch::Tensor &y_scale, const torch::Tensor &seqlens,
                                   const torch::Tensor &cu_seqlens, const torch::Tensor &tiles,
                                   const torch::Tensor &cu_tiles, bool use_task_map) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kFloat8_e4m3fn, "x must be float8_e4m3fn");
  TORCH_CHECK(weight.scalar_type() == at::kFloat8_e4m3fn, "weight must be float8_e4m3fn");
  TORCH_CHECK(y_scale.scalar_type() == at::kFloat, "y_scale must be float32");
  TORCH_CHECK(seqlens.size(0) <= 512, "num_group must be <= 512");
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  int total_tokens = x.size(0);
  int k = x.size(1);
  int n = weight.size(1);
  int num_group = seqlens.size(0);

  auto options = x.options();
  auto y = torch::empty({total_tokens, n}, options.dtype(torch::kBFloat16));

  auto *y_ptr = y.mutable_data_ptr();
  const auto *x_ptr = x.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();
  const auto *y_scale_ptr = y_scale.const_data_ptr();
  const auto *seqlens_ptr = seqlens.const_data_ptr();
  const auto *cu_seqlens_ptr = cu_seqlens.const_data_ptr();
  auto *tiles_ptr = tiles.mutable_data_ptr();
  auto *cu_tiles_ptr = cu_tiles.mutable_data_ptr();

  torch::Tensor task_map;
  void *task_map_ptr = nullptr;
  int task_map_len = 0;
  int num_seq_per_group_avg = (num_group > 0) ? (total_tokens / num_group) : 0;
  int tile_m = pick_tile_m(num_seq_per_group_avg);
  if (use_task_map) {
    task_map_len = compute_task_map_len(total_tokens, num_group, n, tile_m);
    task_map = torch::empty({task_map_len, 4}, options.dtype(torch::kInt32));
    task_map_ptr = task_map.mutable_data_ptr();
    cudaMemsetAsync(task_map_ptr, 0xFF, task_map_len * 4 * sizeof(int), stream);
    int num_tile_n = (n + kTileN - 1) / kTileN;
    HPC_ARCH_DISPATCH("build_task_map", 90,
                      launch_build_task_map(task_map_ptr, cu_tiles_ptr, tiles_ptr, num_group,
                                            num_tile_n, /*use_pdl=*/false, stream));
  }

  HPC_ARCH_DISPATCH("group_gemm_fp8", 90,
                    group_gemm_fp8_multistage_async(
                        y_ptr, x_ptr, weight_ptr, y_scale_ptr, seqlens_ptr, cu_seqlens_ptr,
                        tiles_ptr, cu_tiles_ptr, task_map_ptr, task_map_len, total_tokens, n, k,
                        num_group, num_seq_per_group_avg, /*use_pdl=*/false, stream));

  return y;
}

torch::Tensor group_gemm_fp8_scatter_entry(
    const torch::Tensor &x, const torch::Tensor &weight, const torch::Tensor &y_scale,
    const torch::Tensor &row_indices, const torch::Tensor &seqlens, const torch::Tensor &cu_seqlens,
    const torch::Tensor &tiles, const torch::Tensor &cu_tiles, bool use_task_map) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kFloat8_e4m3fn, "x must be float8_e4m3fn");
  TORCH_CHECK(weight.scalar_type() == at::kFloat8_e4m3fn, "weight must be float8_e4m3fn");
  TORCH_CHECK(y_scale.scalar_type() == at::kFloat, "y_scale must be float32");
  TORCH_CHECK(row_indices.scalar_type() == at::kInt, "row_indices must be int32");
  TORCH_CHECK(seqlens.size(0) <= 512, "group_gemm_fp8_scatter_cp_async: num_group must be <= 512");
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  int total_tokens = x.size(0);
  int k = x.size(1);
  int n = weight.size(1);
  int num_group = seqlens.size(0);

  auto options = x.options();
  auto y = torch::empty({total_tokens, n}, options.dtype(torch::kBFloat16));

  auto *y_ptr = y.mutable_data_ptr();
  const auto *x_ptr = x.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();
  const auto *y_scale_ptr = y_scale.const_data_ptr();
  const auto *row_indices_ptr = row_indices.const_data_ptr();
  const auto *seqlens_ptr = seqlens.const_data_ptr();
  const auto *cu_seqlens_ptr = cu_seqlens.const_data_ptr();
  auto *tiles_ptr = tiles.mutable_data_ptr();
  auto *cu_tiles_ptr = cu_tiles.mutable_data_ptr();

  torch::Tensor task_map;
  void *task_map_ptr = nullptr;
  int task_map_len = 0;
  int num_seq_per_group_avg = (num_group > 0) ? (total_tokens / num_group) : 0;
  int tile_m = pick_tile_m(num_seq_per_group_avg);
  if (use_task_map) {
    task_map_len = compute_task_map_len(total_tokens, num_group, n, tile_m);
    task_map = torch::empty({task_map_len, 4}, options.dtype(torch::kInt32));
    task_map_ptr = task_map.mutable_data_ptr();
    cudaMemsetAsync(task_map_ptr, 0xFF, task_map_len * 4 * sizeof(int), stream);
    int num_tile_n = (n + kTileN - 1) / kTileN;
    HPC_ARCH_DISPATCH("build_task_map", 90,
                      launch_build_task_map(task_map_ptr, cu_tiles_ptr, tiles_ptr, num_group,
                                            num_tile_n, /*use_pdl=*/false, stream));
  }

  HPC_ARCH_DISPATCH(
      "group_gemm_fp8_scatter", 90,
      group_gemm_fp8_scatter_async(y_ptr, x_ptr, weight_ptr, y_scale_ptr, row_indices_ptr,
                                   seqlens_ptr, cu_seqlens_ptr, tiles_ptr, cu_tiles_ptr,
                                   task_map_ptr, task_map_len, total_tokens, n, k, num_group,
                                   num_seq_per_group_avg, /*use_pdl=*/false, stream));

  return y;
}

}  // namespace group_gemm_cp_async
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "group_gemm_fp8(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, Tensor y_scale, "
      "int num_seq_per_group_avg, Tensor? output, Tensor? tma_desc, Tensor? task_map_workspace) -> "
      "(Tensor)");
  m.impl("group_gemm_fp8", torch::kCUDA, &hpc::group_gemm::group_gemm_fp8_entry);

  m.def(
      "group_gemm_pertensor_fp8(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, Tensor "
      "y_scale, int num_seq_per_group_avg, Tensor? output, Tensor? tma_desc, Tensor? "
      "task_map_workspace) -> (Tensor)");
  m.impl("group_gemm_pertensor_fp8", torch::kCUDA, &hpc::group_gemm::group_gemm_fp8_entry);

  m.def(
      "group_gemm_blockwise_fp8(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, Tensor "
      "xscale, Tensor wscale,"
      "int num_seq_per_group_avg, Tensor? output, Tensor? tma_desc, Tensor? task_map_workspace) -> "
      "(Tensor)");
  m.impl("group_gemm_blockwise_fp8", torch::kCUDA,
         &hpc::group_gemm::group_gemm_blockwise_fp8_entry);

  m.def(
      "reformat_x_scale(Tensor x_scale, Tensor seqlens, Tensor cu_seqlens, "
      "Tensor? out_x_scale, int num_seq_per_group_avg) -> (Tensor)");
  m.impl("reformat_x_scale", torch::kCUDA, &hpc::group_gemm::reformat_x_scale_entry);

  m.def(
      "group_gemm_fp8_cp_async(Tensor x, Tensor weight, Tensor y_scale, Tensor seqlens, Tensor "
      "cu_seqlens, Tensor tiles, "
      "Tensor cu_tiles, bool use_task_map=False) -> "
      "(Tensor)");
  m.impl("group_gemm_fp8_cp_async", torch::kCUDA, &hpc::group_gemm_cp_async::group_gemm_fp8_entry);

  m.def(
      "group_gemm_fp8_scatter_cp_async(Tensor x, Tensor weight, Tensor y_scale, Tensor "
      "row_indices, Tensor seqlens, Tensor cu_seqlens, Tensor tiles, "
      "Tensor cu_tiles, bool use_task_map=False) -> "
      "(Tensor)");
  m.impl("group_gemm_fp8_scatter_cp_async", torch::kCUDA,
         &hpc::group_gemm_cp_async::group_gemm_fp8_scatter_entry);
}
