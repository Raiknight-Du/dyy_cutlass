#include <iostream>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

using namespace cute;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC = void;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

template <int TileM, int TileN, int TileK,
          int ClusterM, int ClusterN, int ClusterK,
          bool EpilogueAuto, int EpilogueM, int EpilogueN>
struct Config {
  using MmaTileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
  using ClusterShape = Shape<Int<ClusterM>, Int<ClusterN>, Int<ClusterK>>;
  using ExplicitEpilogueTile = Shape<Int<EpilogueM>, Int<EpilogueN>>;
  using RequestedEpilogueTile = cute::conditional_t<
      EpilogueAuto,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ExplicitEpilogueTile>;
  using ExplicitEpilogueSchedule = cute::conditional_t<
      TileM == 256,
      cutlass::epilogue::TmaWarpSpecialized2Sm,
      cutlass::epilogue::TmaWarpSpecialized1Sm>;
  using RequestedEpilogueSchedule = cute::conditional_t<
      EpilogueAuto,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      ExplicitEpilogueSchedule>;
  using FusionOperation = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<
          cutlass::epilogue::thread::Identity, ElementD, ElementAccumulator,
          cutlass::FloatRoundStyle::round_to_nearest>,
      cutlass::epilogue::fusion::Sm90AccFetch>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      MmaTileShape, ClusterShape,
      RequestedEpilogueTile,
      ElementAccumulator, ElementAccumulator,
      ElementC, cutlass::layout::RowMajor, 1,
      ElementD, cutlass::layout::RowMajor, 8,
      RequestedEpilogueSchedule,
      FusionOperation>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, cutlass::layout::RowMajor, 32,
      ElementB, cutlass::layout::ColumnMajor, 32,
      ElementAccumulator,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
};

template <class C>
void print_config(char const* name) {
  using Main = typename C::CollectiveMainloop;
  using Epi = typename C::CollectiveEpilogue;
  using Kernel = typename C::GemmKernel;
  std::cout
      << name << ','
      << Main::DispatchPolicy::Stages << ','
      << sizeof(typename Main::TensorStorage) << ','
      << sizeof(typename Main::PipelineStorage) << ','
      << sizeof(typename Epi::TensorStorage) << ','
      << sizeof(typename Epi::PipelineStorage) << ','
      << sizeof(typename Kernel::SharedStorage) << ','
      << Main::ABTmaTransactionBytes << ','
      << Main::SFTransactionBytes
      << '\n';
}

int main() {
  std::cout
      << "variant,stages,mainloop_tensor_B,mainloop_pipeline_B,"
      << "epilogue_tensor_B,epilogue_pipeline_B,total_shared_B,"
      << "ab_tma_transaction_B,sf_tma_transaction_B\n";
  print_config<Config<128, 64, 128, 1, 1, 1, true, 128, 64>>(
      "M128_N64_K128_Eauto_C1x1x1");
  print_config<Config<128, 128, 128, 1, 1, 1, true, 128, 64>>(
      "M128_N128_K128_Eauto_C1x1x1");
  print_config<Config<128, 192, 128, 1, 1, 1, false, 128, 64>>(
      "M128_N192_K128_E128x64_C1x1x1");
  print_config<Config<128, 256, 128, 1, 1, 1, false, 128, 128>>(
      "M128_N256_K128_E128x128_C1x1x1");
  print_config<Config<128, 64, 256, 1, 1, 1, true, 128, 64>>(
      "M128_N64_K256_Eauto_C1x1x1");
  print_config<Config<128, 128, 256, 1, 1, 1, true, 128, 64>>(
      "M128_N128_K256_Eauto_C1x1x1");
  print_config<Config<256, 128, 256, 2, 2, 1, false, 128, 16>>(
      "M256_N128_K256_E128x16_C2x2x1");
  print_config<Config<256, 128, 256, 2, 1, 1, false, 64, 64>>(
      "M256_N128_K256_E64x64_C2x1x1");
  print_config<Config<256, 192, 256, 2, 2, 1, true, 128, 64>>(
      "M256_N192_K256_Eauto_C2x2x1");
  print_config<Config<256, 256, 128, 2, 1, 1, false, 128, 64>>(
      "M256_N256_K128_E128x64_C2x1x1");
  print_config<Config<256, 256, 256, 2, 1, 1, false, 64, 64>>(
      "M256_N256_K256_E64x64_C2x1x1");
  print_config<Config<256, 256, 256, 2, 1, 1, true, 128, 64>>(
      "M256_N256_K256_Eauto_C2x1x1");
  print_config<Config<256, 256, 512, 2, 1, 1, true, 128, 64>>(
      "M256_N256_K512_Eauto_C2x1x1");
}
