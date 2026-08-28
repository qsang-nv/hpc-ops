// Copyright (C) 2026 Tencent.

#ifndef SRC_GEMM_SM100_GATED_MLA_GEMM_CUH_
#define SRC_GEMM_SM100_GATED_MLA_GEMM_CUH_

#include <cuda.h>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace gemm {
namespace kernels {

__device__ __forceinline__ auto get_next_tile(int iblock, int num_tile_m, int num_tile_n,
                                              int num_tile_mz, cutlass::FastDivmod mz_tiler_divider,
                                              cutlass::FastDivmod mz_inner_divider,
                                              cutlass::FastDivmod tail_divider) {
  int itile_m;
  int itile_n;

  int imz;
  int imxn;
  mz_tiler_divider(imz, imxn, iblock);

  int im, in;
  if (imz < num_tile_mz) {
    // kBlockSwizzle >= 2 by construction, so FastDivmod's divisor == 1 fallback is dead
    // code here; inlining its fast path drops a predicate and a select from
    // the chain. mz_inner_divider(in, im, imxn);
    in = __umulhi(imxn, mz_inner_divider.multiplier) >> mz_inner_divider.shift_right;
    im = imxn - in * mz_inner_divider.divisor;
  } else {
    tail_divider(in, im, imxn);
  }

  itile_m = imz * mz_inner_divider.divisor + im;
  itile_n = (imz & 0x1) ? num_tile_n - 1 - in : in;

  if (itile_m >= num_tile_m) {
    itile_m = -1;
  }

  return cute::make_tuple(itile_m, itile_n);
}

template <typename Config, typename TmaA, typename TmaB, typename TmaD>
__global__ void __launch_bounds__(256, 1)
    gated_mla_gemm_kernel(const __grid_constant__ TmaA tma_a, const __grid_constant__ TmaB tma_b,
                          const __grid_constant__ TmaD tma_d, const typename Config::TA* x2_ptr,
                          int m, int n, int k, int num_tile_mz,
                          cutlass::FastDivmod mz_tiler_divider,
                          cutlass::FastDivmod mz_inner_divider, cutlass::FastDivmod tail_divider) {
  using namespace cute;  // NOLINT

  using TA = typename Config::TA;
  using TB = typename Config::TB;
  using TC = typename Config::TC;
  using TD = typename Config::TD;
  using TiledMma = typename Config::TiledMma;
  using SLayoutA = typename Config::SLayoutA;
  using SLayoutB = typename Config::SLayoutB;
  using SLayoutC = typename Config::SLayoutC;
  using SLayoutD = typename Config::SLayoutD;

  constexpr int kClusterX = Config::kClusterX;
  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kStageAB = Config::kStageAB;
  constexpr int kStageAcc = Config::kStageAcc;
  constexpr int kStageEpi = Config::kStageEpi;
  constexpr int kEpiTM = Config::kEpiTM;
  constexpr int kStageCLC = Config::kStageCLC;

  // idx
  int idx = threadIdx.x;
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  int elected = cute::elect_one_sync();
  bool is_leader_in_block = (iwarp == 0) && elected;

  int iblock = blockIdx.x / kClusterX;
  int icta_in_cluster = cute::block_rank_in_cluster();
  bool is_leader_cta = (icta_in_cluster == 0);

  // barrier
  __shared__ uint64_t shmab_readable[kStageAB];
  __shared__ uint64_t shmab_writable[kStageAB];

  __shared__ uint64_t tmemd_readable[kStageAcc];
  __shared__ uint64_t tmemd_writable[kStageAcc];

  // CLC
  __shared__ uint64_t clc_readable[kStageCLC];
  __shared__ uint64_t clc_writable[kStageCLC];
  __shared__ uint4 clc_info[kStageCLC];

  __shared__ uint32_t shm_tmem_addr;

  // shm
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto* shm_a = reinterpret_cast<TA*>(shm_data);
  auto* shm_b = reinterpret_cast<TB*>(shm_a + cosize(SLayoutA{}));
  auto* shm_c = reinterpret_cast<TA*>(shm_b + cosize(SLayoutB{}));
  auto* shm_d = reinterpret_cast<TD*>(shm_c + cosize(SLayoutC{}));

  auto sA = make_tensor(make_smem_ptr(shm_a), SLayoutA{});
  auto sB = make_tensor(make_smem_ptr(shm_b), SLayoutB{});
  auto sC = make_tensor(make_smem_ptr(shm_c), SLayoutC{});
  auto sD = make_tensor(make_smem_ptr(shm_d), SLayoutD{});

  // gmem
  auto gA = tma_a.get_tma_tensor(make_shape(m, k));
  auto gB = tma_b.get_tma_tensor(make_shape(n, k));
  auto gAcc = make_tensor(make_gmem_ptr(static_cast<float*>(nullptr)),
                          make_shape(Int<kTileN>{}, Int<kTileM>{}, Int<kStageAcc>{}),
                          make_stride(Int<kTileM>{}, Int<1>{}, Int<kTileM * kTileN>{}));
  // tma
  auto btma_a01 = tma_a.get_slice(icta_in_cluster);
  auto btma_b01 = tma_b.get_slice(icta_in_cluster);

  auto btma_a00 = tma_a.get_slice(0);
  auto btma_b00 = tma_b.get_slice(0);

  auto tAg = btma_a01.partition_S(gA);  // (V, M, K)
  auto tAs = btma_a00.partition_D(sA);  // (V, M, K, kStageAB)

  auto tBg = btma_b01.partition_S(gB);  // (V, N, K)
  auto tBs = btma_b00.partition_D(sB);  // (V, N, K, kStageAB)

  int num_tile_m = size<1>(tAg);
  int num_tile_n = size<1>(tBg);

  constexpr int kTransactionBytes = (sizeof_bits_v<TA> * cosize(SLayoutA{}(_, _, 0)) +   // NOLINT
                                     sizeof_bits_v<TB> * cosize(SLayoutB{}(_, _, 0))) *  // NOLINT
                                    kClusterX /
                                    8;  // bits -> byte

  // mma
  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(icta_in_cluster);
  auto tAs4r = thr_mma.partition_B(sA);
  auto tBs4r = thr_mma.partition_A(sB);

  auto tAr = thr_mma.make_fragment_B(tAs4r);
  auto tBr = thr_mma.make_fragment_A(tBs4r);

  auto tAccr_layout = thr_mma.partition_fragment_C(gAcc).layout();

  if (is_leader_in_block) {
#pragma unroll
    for (int i = 0; i < kStageAB; ++i) {
      initialize_barrier(shmab_readable[i]);
      initialize_barrier(shmab_writable[i]);
    }

#pragma unroll
    for (int i = 0; i < kStageAcc; ++i) {
      initialize_barrier(tmemd_readable[i]);
      initialize_barrier(tmemd_writable[i], kClusterX);
    }

    // leader-cta: tma-1, mma-32, epi-128
    // peer-cta:   tma-1, mma-0,  epi-128
    constexpr int kWritableArrive = kClusterX == 2 ? (1 + 32 + 128) + (1 + 128) : 1 + 32 + 128;
#pragma unroll
    for (int i = 0; i < kStageCLC; ++i) {
      initialize_barrier(clc_readable[i]);
      initialize_barrier(clc_writable[i], kWritableArrive);
    }

    cutlass::arch::fence_barrier_init();
  }

  int ntile_k = size<2>(tAg);

  if constexpr (kClusterX == 2) {
    cluster_relaxed_sync();
  } else {
    __syncthreads();
  }

  using TmemAllocator =
      cute::conditional_t<kClusterX == 2, cute::TMEM::Allocator2Sm, cute::TMEM::Allocator1Sm>;

  int istage_clc = 0;
  int phase_clc = 0;

  // main
  if (iwarp == 0) {
    // ---------- CLC schedule warp --------
    // only leader cta
    if (is_leader_cta) {
      int phase_clc_w = 1;
      int phase_clc_r = 0;

      while (true) {
        wait_barrier(clc_writable[istage_clc], phase_clc_w);

        if (elected) {
#pragma unroll
          for (int i = 0; i < kClusterX; i++) {
            hpc::set_barrier_transaction_bytes_cluster(clc_readable[istage_clc], sizeof(uint4), i);
          }

          hpc::find_next_block(&clc_info[istage_clc], &clc_readable[istage_clc]);
        }

        wait_barrier(clc_readable[istage_clc], phase_clc_r);
        auto [ctaid_x, ok] = hpc::get_next_block(&clc_info[istage_clc]);
        if (!ok) {
          wait_barrier(clc_writable[istage_clc], phase_clc_w ^ 1);
          break;
        }

        ++istage_clc;
        if (istage_clc == kStageCLC) {
          istage_clc = 0;
          phase_clc_w ^= 1;
          phase_clc_r ^= 1;
        }
      }  // while
    }
  } else if (iwarp == 1) {
    // ---------- TMA load warp ----------
    if (elected) {
      int ismem_write = 0;
      int phase = 1;

      while (true) {
        auto [itile_m, itile_n] = get_next_tile(iblock, num_tile_m, num_tile_n, num_tile_mz,
                                                mz_tiler_divider, mz_inner_divider, tail_divider);
        if (itile_m < 0) {
          break;
        }

#pragma unroll 1
        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(shmab_writable[ismem_write], phase);

          cute::copy(tma_a.with(shmab_readable[ismem_write]), tAg(_, itile_m, itile_k),
                     tAs(_, 0, 0, ismem_write));
          cute::copy(tma_b.with(shmab_readable[ismem_write]), tBg(_, itile_n, itile_k),
                     tBs(_, 0, 0, ismem_write));

          if (is_leader_cta) {
            set_barrier_transaction_bytes(shmab_readable[ismem_write], kTransactionBytes);
          }

          ++ismem_write;
          if (ismem_write == kStageAB) {
            ismem_write = 0;
            phase ^= 1;
          }
        }

        wait_barrier(clc_readable[istage_clc], phase_clc);
        auto [ctaid_x, ok] = hpc::get_next_block(&clc_info[istage_clc]);
        hpc::arrive_cluster_barrier(clc_writable[istage_clc]);
        if (!ok) {
          break;
        }
        iblock = ctaid_x / kClusterX;
        ++istage_clc;
        if (istage_clc == kStageCLC) {
          istage_clc = 0;
          phase_clc ^= 1;
        }
      }  // while
    }  // if elected
  } else if (iwarp == 2) {
    // ---------- MMA warp ----------
    auto umma_arrive = [](uint64_t& mbar) {
      if constexpr (kClusterX == 1) {
        cutlass::arch::umma_arrive(&mbar);
      } else if constexpr (kClusterX == 2) {
        cutlass::arch::umma_arrive_multicast_2x1SM(&mbar, 0x3);
      }
    };

    TmemAllocator tmem_allocator;
    tmem_allocator.allocate(kTileM * kStageAcc, &shm_tmem_addr);

    uint32_t tmem_addr = 0;  // hard code, no need to read it from shm
    auto tAccr = make_tensor(make_tmem_ptr<float>(tmem_addr), tAccr_layout);

    int ismem_read = 0;
    int phase_shmab = 0;

    int itmem_write = 0;
    int phase_tmem = 1;

    if (is_leader_cta) {
      while (true) {
        auto [itile_m, itile_n] = get_next_tile(iblock, num_tile_m, num_tile_n, num_tile_mz,
                                                mz_tiler_divider, mz_inner_divider, tail_divider);
        if (itile_m < 0) {
          break;
        }

        wait_barrier(tmemd_writable[itmem_write], phase_tmem);

        bool ab_ready = test_barrier(shmab_readable[ismem_read], phase_shmab);
        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll 1
        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          if (!ab_ready) {
            wait_barrier(shmab_readable[ismem_read], phase_shmab);
          }

#pragma unroll
          for (int ik = 0; ik < size<2>(tAr); ++ik) {
            cute::gemm(tiled_mma, tBr(_, _, ik, ismem_read), tAr(_, _, ik, ismem_read),
                       tAccr(_, _, _, itmem_write));
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
          }
          umma_arrive(shmab_writable[ismem_read]);

          ++ismem_read;
          if (ismem_read == kStageAB) {
            ismem_read = 0;
            phase_shmab ^= 1;
          }

          ab_ready = test_barrier(shmab_readable[ismem_read], phase_shmab);
        }  // for itile_k

        umma_arrive(tmemd_readable[itmem_write]);
        ++itmem_write;
        if (itmem_write == kStageAcc) {
          itmem_write = 0;
          phase_tmem ^= 1;
        }

        wait_barrier(clc_readable[istage_clc], phase_clc);
        auto [ctaid_x, ok] = hpc::get_next_block(&clc_info[istage_clc]);
        hpc::arrive_cluster_barrier(clc_writable[istage_clc]);
        if (!ok) {
          break;
        }
        iblock = ctaid_x / kClusterX;
        ++istage_clc;
        if (istage_clc == kStageCLC) {
          istage_clc = 0;
          phase_clc ^= 1;
        }
      }  // while
    }  // is_leader_cta
  } else if (iwarp >= 4) {
    // ----------Epilogue warp 4/5/6/7 ----------
    idx -= 128;
    int phase = 0;
    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int itmem_read = 0;

    bool is_store_warp = (iwarp % 2) == 0;

    uint32_t tmem_addr = 0;
    auto tAccr = make_tensor(make_tmem_ptr<float>(tmem_addr), tAccr_layout);

    auto tAccnm = tAccr(make_coord(_, _), Int<0>{}, Int<0>{}, _);               // (n, m, stage)
    auto tDrmn = make_tensor(tAccnm.data(), select<1, 0, 2>(tAccnm.layout()));  // (m, n, stage)

    // tmem -> reg
    using TmemLoadOp = cute::conditional_t<kEpiTM == 16, cute::SM100_TMEM_LOAD_16dp256b2x,
                                           cute::SM100_TMEM_LOAD_16dp256b4x>;
    auto tiled_copy_t2r = make_tmem_copy(TmemLoadOp{}, tDrmn(_, _, 0));
    auto thr_copy_t2r = tiled_copy_t2r.get_slice(idx);
    auto tDt4rx = thr_copy_t2r.partition_S(tDrmn);  // (V, m, n)
    auto tDr4t = make_fragment_like<float>(thr_copy_t2r.partition_D(tDrmn(_, _, 0)));
    auto tDr4t_TD = make_fragment_like<TD>(tDr4t);

    // reg -> smem
    auto tiled_copy_r2s = make_tiled_copy_D(Copy_Atom<SM90_U16x8_STSM_T, TD>{}, tiled_copy_t2r);
    auto thr_copy_r2s = tiled_copy_r2s.get_slice(idx);
    auto tDr4s = thr_copy_r2s.retile_S(tDr4t_TD);
    auto tDs4r = thr_copy_r2s.partition_D(sD);

    // tma store
    auto gD = tma_d.get_tma_tensor(make_shape(m, n));
    auto btma_d = tma_d.get_slice(0);
    auto tDs = btma_d.partition_S(sD);  // (V, 1, 2, kStageEpi)
    auto tDg = btma_d.partition_D(gD);

    // preload(cpasync) subtile gC -> sC
    auto gC = make_tensor(make_gmem_ptr(x2_ptr), make_shape(m, n), make_stride(n, 1));
    constexpr int kThreadsPerRow = 128 / 8;  // 16byte load
    constexpr int kRowsPerIter = 128 / kThreadsPerRow;
    constexpr int kIters = kEpiTM / kRowsPerIter;
    int row = idx / kThreadsPerRow;
    int col = idx % kThreadsPerRow;

    // smem(C) -> reg
    auto tiled_copy_s2r = make_tiled_copy_D(Copy_Atom<SM75_U16x8_LDSM_T, TA>{}, tiled_copy_t2r);
    auto thr_copy_s2r = tiled_copy_s2r.get_slice(idx);
    auto tCs4r = thr_copy_s2r.partition_S(sC);
    auto tCr4t = make_fragment_like<TC>(tDr4t);
    auto tCr4t_view = thr_copy_s2r.retile_D(tCr4t);

    while (true) {
      auto [itile_m, itile_n] = get_next_tile(iblock, num_tile_m, num_tile_n, num_tile_mz,
                                              mz_tiler_divider, mz_inner_divider, tail_divider);
      if (itile_m < 0) {
        break;
      }

      int itile_n64 = itile_n * 2 * kClusterX + icta_in_cluster * 2 + iwarp / 2;
      int itile_m_epi = itile_m * (kTileM / kEpiTM);

      auto tDt4r = tDt4rx(_, _, _, itmem_read);

      // cp_async gC to smem_C (x2 gate operand)
      auto itile_addr = &gC(itile_m * kTileM, itile_n * kTileN + icta_in_cluster * 128);
#pragma unroll
      for (int i = 0; i < kIters; i++) {
        int r = i * kRowsPerIter + row;
        bool valid = itile_m * kTileM + r < m;
        int src_size = valid ? 16 : 0;
        cp_async_16b(&sC(r, col * 8), itile_addr + r * n + col * 8, src_size);
      }
      cp_async_fence();

      wait_barrier(tmemd_readable[itmem_read], phase);

#pragma unroll
      for (int im = 0; im < size<1>(tDr4t); ++im) {
        int istage = im % kStageEpi;

        cute::copy(tiled_copy_t2r, tDt4r(_, im, _), tDr4t(_, 0, _));

        cp_async_wait<0>();
        cutlass::arch::NamedBarrier::sync(128, 0);
        // ldmatrix sC -> reg
        cute::copy(tiled_copy_s2r, tCs4r(_, 0, _), tCr4t_view(_, 0, _));

#pragma unroll
        for (int in = 0; in < size<2>(tDr4t); ++in) {
#pragma unroll
          for (int i = 0; i < size<0>(tDr4t) / 2; ++i) {
            float2 tmp = __fmul2_rn(
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&tCr4t(2 * i, 0, in))),
                {sigmoid(tDr4t(2 * i, 0, in)), sigmoid(tDr4t(2 * i + 1, 0, in))});
            __nv_bfloat162 out = __float22bfloat162_rn(tmp);
            *reinterpret_cast<__nv_bfloat162*>(&tDr4t_TD(2 * i, 0, in)) = out;
          }
        }

        // cp_async next subtile; gC -> sC
        cutlass::arch::NamedBarrier::sync(128, 0);
        if (im < size<1>(tDr4t) - 1) {
          itile_addr += kEpiTM * n;
#pragma unroll
          for (int i = 0; i < kIters; i++) {
            int r = i * kRowsPerIter + row;
            bool valid = itile_m * kTileM + (im + 1) * kEpiTM + r < m;
            int src_size = valid ? 16 : 0;
            cp_async_16b(&sC(r, col * 8), itile_addr + r * n + col * 8, src_size);
          }
          cp_async_fence();
        }

        tma_store_wait<kStageEpi - 1>();
        cutlass::arch::NamedBarrier::sync(128, 0);

        // reg -> shm
        cute::copy(tiled_copy_r2s, tDr4s(_, 0, _), tDs4r(_, 0, _, istage));

        tma_store_fence();
        cutlass::arch::NamedBarrier::sync(128, 0);

        if (is_store_warp) {
          cute::copy(tma_d, tDs(_, 0, iwarp / 2, istage), tDg(_, itile_m_epi + im, itile_n64));
        }
        tma_store_arrive();
      }  // for im

      if (iwarp == 0 && cute::elect_one_sync()) {
        if constexpr (kClusterX == 1) {
          arrive_barrier(tmemd_writable[itmem_read]);
        } else if constexpr (kClusterX == 2) {
          cutlass::arch::umma_arrive_2x1SM_sm0(&tmemd_writable[itmem_read]);
        }
      }

      ++itmem_read;
      if (itmem_read == kStageAcc) {
        itmem_read = 0;
        phase ^= 1;
      }

      wait_barrier(clc_readable[istage_clc], phase_clc);
      auto [ctaid_x, ok] = hpc::get_next_block(&clc_info[istage_clc]);
      hpc::arrive_cluster_barrier(clc_writable[istage_clc]);
      if (!ok) {
        break;
      }
      iblock = ctaid_x / kClusterX;
      ++istage_clc;
      if (istage_clc == kStageCLC) {
        istage_clc = 0;
        phase_clc ^= 1;
      }
    }  // while

    if (iwarp == 1) {
      TmemAllocator tmem_allocator;
      tmem_allocator.free(tmem_addr, kTileM * kStageAcc);
    }
  } else {
    return;  // warp 3
  }
}

}  // namespace kernels
}  // namespace gemm
}  // namespace hpc

#endif  // SRC_GEMM_SM100_GATED_MLA_GEMM_CUH_
