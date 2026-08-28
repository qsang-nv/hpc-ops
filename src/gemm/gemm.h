// Copyright 2025 hpc-ops authors

#ifndef SRC_GEMM_GEMM_H_
#define SRC_GEMM_GEMM_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

#include "src/utils/utils.h"

namespace hpc {
namespace gemm {

bool gemm_bf16xfp32_async(void *y_ptr, void *splitk_y_ptr, void *split_flag_ptr, const void *x_ptr,
                          const void *w_high_ptr, const void *w_low_ptr, int m, int n, int k,
                          float scale, bool use_fp32_output, int splitk, int kTileM, int wgn,
                          cudaStream_t stream);

bool gated_mla_gemm_async(void *y_ptr, const void *x_ptr, const void *w_ptr, const void *x2_ptr,
                          int m, int n, int k, cudaStream_t stream);

}  // namespace gemm
}  // namespace hpc

#endif  // SRC_GEMM_GEMM_H_
