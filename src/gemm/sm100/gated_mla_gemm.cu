// Copyright (C) 2026 Tencent.

#include <cuda.h>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "src/gemm/gemm.h"
#include "src/gemm/sm100/gated_mla_gemm.cuh"

namespace hpc {
namespace gemm {

using namespace cute;  // NOLINT

template <typename Tin_, typename Tout_, int kTileM_, int kClusterX_ = 2, int kStageAB_ = 6,
          int kStageEpi_ = 2, int kEpiTM_ = 32, int kBlockSwizzle_ = 12>
struct GatedMlaGemmConfig {
  using TA = Tin_;
  using TB = Tin_;
  using TD = Tout_;
  using TC = TD;

  static constexpr int kTileM = kTileM_;
  static constexpr int kTileK = 64;
  static constexpr int kStageAB = kStageAB_;
  static constexpr int kStageAcc = 2;
  static constexpr int kStageEpi = kStageEpi_;
  static constexpr int kEpiTM = kEpiTM_;
  static constexpr int kBlockSwizzle = kBlockSwizzle_;
  static constexpr int kClusterX = kClusterX_;
  static constexpr int kTileN = (kClusterX == 2) ? 256 : 128;
  static constexpr int kStageCLC = 2;

  static_assert(kTileN / kClusterX == 128,
                "Epilogue uses 4 warp x 32 tmem, kTileN / kClusterX must be 128");
  static_assert(kTileM % kEpiTM == 0 && (kTileM / kEpiTM) % kStageEpi == 0, "Epilogue uses Stages");
  static_assert(kTileM * kStageAcc <= 512 && (kTileM & (kTileM - 1)) == 0 && kTileM >= 16,
                "kTileM must be power of 2");
  static_assert(kBlockSwizzle >= 2, "kBlockSwizzle must >= 2");

  static constexpr int kTileMCta = kTileM / kClusterX;
  static constexpr int kTileNCta = kTileN / kClusterX;

  using SLayoutA =
      decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<TA>{},
                             make_shape(Int<kTileMCta>{}, Int<kTileK>{}, Int<kStageAB>{})));
  using SLayoutB =
      decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<TB>{},
                             make_shape(Int<kTileNCta>{}, Int<kTileK>{}, Int<kStageAB>{})));
  using SLayoutC = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<TA>{},
                                          make_shape(Int<kEpiTM>{}, Int<kTileNCta>{})));
  using SLayoutD =
      decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<TD>{},
                             make_shape(Int<kEpiTM>{}, Int<kTileNCta>{}, Int<kStageEpi>{})));

  using CopyBoxA = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<TA>{},
                                          make_shape(Int<kTileM>{}, Int<kTileK>{})));
  using CopyBoxB = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<TB>{},
                                          make_shape(Int<kTileN>{}, Int<kTileK>{})));
  using CopyBoxD = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<TD>{},
                                          make_shape(Int<kEpiTM>{}, Int<kTileNCta / 2>{})));

  using TmaLoadT = conditional_t<kClusterX == 2, SM100_TMA_2SM_LOAD, SM90_TMA_LOAD>;
  using TmaStoreT = SM90_TMA_STORE;

  using MmaAtom = conditional_t<
      kClusterX == 2,
      SM100_MMA_F16BF16_2x1SM_SS<TB, TA, float, kTileN, kTileM, UMMA::Major::K, UMMA::Major::K>,
      SM100_MMA_F16BF16_SS<TB, TA, float, kTileN, kTileM, UMMA::Major::K, UMMA::Major::K>>;
  using TiledMma = decltype(make_tiled_mma(MmaAtom{}));

  static constexpr int kShmABytes = sizeof_bits_v<TA> * cosize(SLayoutA{}) / 8;
  static constexpr int kShmBBytes = sizeof_bits_v<TB> * cosize(SLayoutB{}) / 8;
  static constexpr int kShmDBytes = sizeof_bits_v<TD> * cosize(SLayoutD{}) / 8;
  static constexpr int kShmCBytes = sizeof_bits_v<TA> * cosize(SLayoutC{}) / 8;
  static constexpr int kShmSize = kShmABytes + kShmBBytes + kShmDBytes + kShmCBytes;
  static_assert(kShmSize <= 226 * 1024, "shm overflow");

  template <typename TX, typename TW, typename TY>
  auto get_tma(TX x, TW w, TY y) {
    auto tma_a = make_tma_copy(TmaLoadT{}, x, CopyBoxA{}, Int<kClusterX>{});
    auto tma_b = make_tma_copy(TmaLoadT{}, w, CopyBoxB{}, Int<kClusterX>{});
    auto tma_d = make_tma_copy(TmaStoreT{}, y, CopyBoxD{});
    return std::make_tuple(tma_a, tma_b, tma_d);
  }
};

template <typename Config>
bool launch_gated_mla_gemm_kernel(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                  const void *x2_ptr, int m, int n, int k, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using TA = typename Config::TA;
  using TB = typename Config::TB;
  using TD = typename Config::TD;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kClusterX = Config::kClusterX;
  constexpr int kBlockSwizzle = Config::kBlockSwizzle;

  auto A = make_tensor(make_gmem_ptr(reinterpret_cast<const TA *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));
  auto B = make_tensor(make_gmem_ptr(reinterpret_cast<const TB *>(w_ptr)), make_shape(n, k),
                       make_stride(k, Int<1>{}));
  auto D = make_tensor(make_gmem_ptr(reinterpret_cast<TD *>(y_ptr)), make_shape(m, n),
                       make_stride(n, Int<1>{}));

  Config config;
  auto [tma_a, tma_b, tma_d] = config.get_tma(A, B, D);

  auto kernel =
      kernels::gated_mla_gemm_kernel<Config, decltype(tma_a), decltype(tma_b), decltype(tma_d)>;

  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Config::kShmSize);

  int num_tile_m = (m + kTileM - 1) / kTileM;
  int num_tile_n = (n + kTileN - 1) / kTileN;
  int num_tile = num_tile_m * num_tile_n;
  int num_tile_mz = num_tile_m / kBlockSwizzle;

  int mz_tail = num_tile_m % kBlockSwizzle;
  mz_tail = mz_tail == 0 ? 1 : mz_tail;
  cutlass::FastDivmod mz_tiler_divider(kBlockSwizzle * num_tile_n);
  cutlass::FastDivmod mz_inner_divider(kBlockSwizzle);
  cutlass::FastDivmod tail_divider(mz_tail);

  dim3 block(256);
  dim3 grid(num_tile * kClusterX);

  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = kClusterX;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;

  cudaLaunchConfig_t launch_config{};
  launch_config.gridDim = grid;
  launch_config.blockDim = block;
  launch_config.dynamicSmemBytes = Config::kShmSize;
  launch_config.stream = stream;
  launch_config.attrs = attr;
  launch_config.numAttrs = 1;

  cudaLaunchKernelEx(&launch_config, kernel, tma_a, tma_b, tma_d,
                     reinterpret_cast<const TA *>(x2_ptr), m, n, k, num_tile_mz, mz_tiler_divider,
                     mz_inner_divider, tail_divider);

  return true;
}

bool gated_mla_gemm_async(void *y_ptr, const void *x_ptr, const void *w_ptr, const void *x2_ptr,
                          int m, int n, int k, cudaStream_t stream) {
  using Tin = cute::bfloat16_t;
  using Tout = cute::bfloat16_t;

  if (m <= 16) {
    // 1sm, kTileM/kEpiTM == 1
    constexpr int kTileM = 16;
    constexpr int kClusterX = 1;
    constexpr int kStageAB = 10;
    constexpr int kStageEpi = 1;
    constexpr int kEpiTM = 16;
    constexpr int kBlockSwizzle = 2;
    using Config = GatedMlaGemmConfig<Tin, Tout, kTileM, kClusterX, kStageAB, kStageEpi, kEpiTM,
                                      kBlockSwizzle>;
    return launch_gated_mla_gemm_kernel<Config>(y_ptr, x_ptr, w_ptr, x2_ptr, m, n, k, stream);
  } else if (m <= 32) {
    // 1sm, kTileM/32 == 1
    constexpr int kTileM = 32;
    constexpr int kClusterX = 1;
    constexpr int kStageAB = 10;
    constexpr int kStageEpi = 1;
    constexpr int kEpiTM = 32;
    constexpr int kBlockSwizzle = 2;
    using Config = GatedMlaGemmConfig<Tin, Tout, kTileM, kClusterX, kStageAB, kStageEpi, kEpiTM,
                                      kBlockSwizzle>;
    return launch_gated_mla_gemm_kernel<Config>(y_ptr, x_ptr, w_ptr, x2_ptr, m, n, k, stream);
  } else if (m <= 64) {
    // 1sm
    constexpr int kTileM = 64;
    constexpr int kClusterX = 1;
    constexpr int kStageAB = 8;
    constexpr int kStageEpi = 2;
    constexpr int kEpiTM = 32;
    constexpr int kBlockSwizzle = 2;
    using Config = GatedMlaGemmConfig<Tin, Tout, kTileM, kClusterX, kStageAB, kStageEpi, kEpiTM,
                                      kBlockSwizzle>;
    return launch_gated_mla_gemm_kernel<Config>(y_ptr, x_ptr, w_ptr, x2_ptr, m, n, k, stream);
  } else if (m <= 128) {
    // 1sm
    constexpr int kTileM = 128;
    constexpr int kClusterX = 1;
    constexpr int kStageAB = 6;
    constexpr int kStageEpi = 2;
    constexpr int kEpiTM = 32;
    constexpr int kBlockSwizzle = 2;
    using Config = GatedMlaGemmConfig<Tin, Tout, kTileM, kClusterX, kStageAB, kStageEpi, kEpiTM,
                                      kBlockSwizzle>;
    return launch_gated_mla_gemm_kernel<Config>(y_ptr, x_ptr, w_ptr, x2_ptr, m, n, k, stream);
  } else {
    // 2sm
    constexpr int kTileM = 256;
    constexpr int kClusterX = 2;
    constexpr int kStageAB = 6;
    constexpr int kStageEpi = 2;
    constexpr int kEpiTM = 32;
    constexpr int kBlockSwizzle = 8;
    using Config = GatedMlaGemmConfig<Tin, Tout, kTileM, kClusterX, kStageAB, kStageEpi, kEpiTM,
                                      kBlockSwizzle>;
    return launch_gated_mla_gemm_kernel<Config>(y_ptr, x_ptr, w_ptr, x2_ptr, m, n, k, stream);
  }
}

}  // namespace gemm
}  // namespace hpc
