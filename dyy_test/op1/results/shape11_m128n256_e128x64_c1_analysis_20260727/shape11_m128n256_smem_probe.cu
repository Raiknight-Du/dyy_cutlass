#include <iostream>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

using namespace cute;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementD = cutlass::bfloat16_t;
using ElementC = void;
using ElementAccumulator = float;
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 1;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

template <int TileM, int TileN, int TileK,
          int ClusterM, int ClusterN, int ClusterK,
          int EpilogueM, int EpilogueN>
struct KernelConfig {
  using MmaTileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
  using ClusterShape = Shape<Int<ClusterM>, Int<ClusterN>, Int<ClusterK>>;
  using EpilogueTile = Shape<Int<EpilogueM>, Int<EpilogueN>>;
  using EpilogueSchedule = cute::conditional_t<
      TileM == 256,
      cutlass::epilogue::TmaWarpSpecialized2Sm,
      cutlass::epilogue::TmaWarpSpecialized1Sm>;
  using FusionOperation = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<
          cutlass::epilogue::thread::Identity, ElementD, ElementAccumulator,
          cutlass::FloatRoundStyle::round_to_nearest>,
      cutlass::epilogue::fusion::Sm90AccFetch>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass,
          MmaTileShape, ClusterShape,
          EpilogueTile,
          ElementAccumulator, ElementAccumulator,
          ElementC, LayoutCTag, AlignmentC,
          ElementD, LayoutDTag, AlignmentD,
          EpilogueSchedule,
          FusionOperation>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass,
          ElementA, LayoutATag, AlignmentA,
          ElementB, LayoutBTag, AlignmentB,
          ElementAccumulator,
          MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
};

template <class Config>
void print_config(char const* name) {
  using Mainloop = typename Config::CollectiveMainloop;
  using Epilogue = typename Config::CollectiveEpilogue;
  using Kernel = typename Config::GemmKernel;
  std::cout
      << name
      << ",stage=" << Mainloop::DispatchPolicy::Stages
      << ",mainloop_tensor=" << sizeof(typename Mainloop::TensorStorage)
      << ",mainloop_pipeline=" << sizeof(typename Mainloop::PipelineStorage)
      << ",mainloop_shared=" << sizeof(typename Mainloop::SharedStorage)
      << ",epilogue_tensor=" << sizeof(typename Epilogue::TensorStorage)
      << ",epilogue_pipeline=" << sizeof(typename Epilogue::PipelineStorage)
      << ",epilogue_shared=" << sizeof(typename Epilogue::SharedStorage)
      << ",kernel_shared=" << sizeof(typename Kernel::SharedStorage)
      << '\n';
}

int main() {
  using Baseline = KernelConfig<128, 192, 128, 1, 1, 1, 128, 64>;
  using M256N192 = KernelConfig<256, 192, 128, 2, 1, 1, 128, 64>;
  using M128N256E64 = KernelConfig<128, 256, 128, 1, 1, 1, 128, 64>;
  using M128N256E128 = KernelConfig<128, 256, 128, 1, 1, 1, 128, 128>;
  using M256N256 = KernelConfig<256, 256, 128, 2, 1, 1, 128, 64>;
  print_config<Baseline>("M128_N192_E128x64_C1x1");
  print_config<M256N192>("M256_N192_E128x64_C2x1");
  print_config<M128N256E64>("M128_N256_E128x64_C1x1");
  print_config<M128N256E128>("M128_N256_E128x128_C1x1");
  print_config<M256N256>("M256_N256_E128x64_C2x1");
  return 0;
}
