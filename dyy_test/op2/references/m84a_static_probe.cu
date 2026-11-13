#include <cstdio>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#ifndef SM_MODE
#define SM_MODE 2
#endif
#ifndef TILE_M
#define TILE_M 256
#endif
#ifndef TILE_N
#define TILE_N 256
#endif
#ifndef TILE_K
#define TILE_K 256
#endif
#ifndef CLUSTER_M
#define CLUSTER_M 2
#endif
#ifndef CLUSTER_N
#define CLUSTER_N 1
#endif
#ifndef CLUSTER_K
#define CLUSTER_K 1
#endif
#ifndef EPI_AUTO
#define EPI_AUTO 1
#endif
#ifndef EPI_M
#define EPI_M 128
#endif
#ifndef EPI_N
#define EPI_N 128
#endif

using namespace cute;
using ElementA = cutlass::float_e2m1_t;
using ElementAPair = cutlass::nv_float4_t<ElementA>;
using ElementB = cutlass::float_e2m1_t;
using ElementBPair = cutlass::nv_float4_t<ElementB>;
using ElementC = void;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;

using TileShape = Shape<Int<TILE_M>, Int<TILE_N>, Int<TILE_K>>;
using ClusterShape = Shape<Int<CLUSTER_M>, Int<CLUSTER_N>, Int<CLUSTER_K>>;

#if EPI_AUTO
using RequestedEpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
#else
using RequestedEpilogueTile = Shape<Int<EPI_M>, Int<EPI_N>>;
#endif

#if SM_MODE == 1
using RequestedEpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1SmNvf4;
using RequestedKernelSchedule = cutlass::gemm::KernelSparseTmaWarpSpecialized1SmNvf4Sm100;
#else
using RequestedEpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2SmNvf4;
using RequestedKernelSchedule = cutlass::gemm::KernelSparseTmaWarpSpecialized2SmNvf4Sm100;
#endif

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassBlockScaledSparseTensorOp,
    TileShape,
    ClusterShape,
    RequestedEpilogueTile,
    ElementAccumulator,
    ElementAccumulator,
    ElementC,
    cutlass::layout::RowMajor,
    1,
    ElementD,
    cutlass::layout::RowMajor,
    8,
    RequestedEpilogueSchedule>::CollectiveOp;

using MainloopBuilder = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassBlockScaledSparseTensorOp,
    ElementAPair,
    cutlass::layout::RowMajor,
    64,
    ElementBPair,
    cutlass::layout::ColumnMajor,
    32,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
    RequestedKernelSchedule>;

using CollectiveMainloop = typename MainloopBuilder::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;

__global__ void probe_layout() {
  using EpiTile = typename CollectiveEpilogue::EpilogueTile;
  constexpr bool kOverlap = CollectiveMainloop::DispatchPolicy::IsOverlappingAccum;
  auto storage = CollectiveMainloop::template init_tmem_tensors<EpiTile, kOverlap>(EpiTile{});

  if (threadIdx.x == 0) {
    int const acc = int(cutlass::detail::find_tmem_tensor_col_offset(storage.accumulators));
    int const sfa = int(cutlass::detail::find_tmem_tensor_col_offset(storage.tCtSFA));
    int const sfb = int(cutlass::detail::find_tmem_tensor_col_offset(storage.tCtSFB));
    printf("RESULT,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu\n",
           SM_MODE, TILE_M, TILE_N, TILE_K, CLUSTER_M, CLUSTER_N, CLUSTER_K,
           int(size<0>(typename CollectiveMainloop::CtaShape_MNK{})),
           int(size<1>(typename CollectiveMainloop::CtaShape_MNK{})),
           int(size<2>(typename CollectiveMainloop::CtaShape_MNK{})),
           int(size<0>(EpiTile{})), int(size<1>(EpiTile{})),
           CollectiveMainloop::MainloopPipeline::Stages,
           int(kOverlap), acc, sfa, sfb,
           static_cast<unsigned long long>(sizeof(typename CollectiveEpilogue::SharedStorage)),
           static_cast<unsigned long long>(sizeof(typename GemmKernel::SharedStorage)),
           static_cast<unsigned long long>(sizeof(typename CollectiveMainloop::TensorStorage)),
           static_cast<unsigned long long>(sizeof(typename CollectiveMainloop::SharedStorage)));
  }
}

int main() {
  probe_layout<<<1, 32>>>();
  auto status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::cerr << cudaGetErrorString(status) << "\n";
    return 2;
  }
  return 0;
}
