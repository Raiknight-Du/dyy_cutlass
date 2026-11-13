// Shape5 multicast probe derived from the op1 pure-matmul harness.
// ElementC=void and the epilogue is Identity(convert(accumulator)); there is
// no C load or alpha/beta visitor. NO-TC builds use the isolated header overlay.
/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief A GEMM example using CUTLASS for the NVIDIA Blackwell SM100 architecture.

    This example demonstrates a simple way to instantiate and run a blockscaled NVFP4 GEMM on the NVIDIA Blackwell SM100 architecture.

    The Blackwell SM100 CUTLASS kernel uses the new Block Scaled Tensor Core MMA Instructions (tcgen05.mma.blockscaled) introduced
    on the Blackwell architecture (sm100a) which have 2x throughput compared to fp8 Tensor Core MMA instructions (tcgen05.mma)
    and 4x throughput compared to fp8 Hopper Tensor Core MMA Instructions (WGMMA) (See https://docs.nvidia.com/cuda/parallel-thread-execution).

    Similar to 70_blackwell_gemm, this kernel leverages:
    1. Per-SM memory called Tensor Memory (TMEM)  (Please refer to CUDA 12.8 docs on https://docs.nvidia.com/cuda/).

    2. The extended warp-specialized kernel design introduced in Hopper enabled by use of TMEM
    which allows us to decouple the execution of MMA and epilogue into separate warps.

    3. A new SW controlled dynamic scheduler based on cluster launch control (See https://docs.nvidia.com/cuda/parallel-thread-execution).

    Usage:

      $ ./examples/72_blackwell_narrow_precision_gemm/M_72a_blackwell_nvfp4_bf16_gemm --m=2048 --n=2048 --k=2048
*/

#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"


#include <iostream>

#include "helper.h"
#include "m72a_random_dataset.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)


/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 32;                                             // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 32;                                             // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementD    = cutlass::bfloat16_t;                            // Element type for D matrix operand
using         ElementC    = void;                                           // Pure matmul: no source C matrix
using         LayoutCTag  = cutlass::layout::RowMajor;                      // Layout type for C matrix operand
using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for D matrix operand
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
constexpr int AlignmentC  = 1;                                              // Void C has no memory access alignment requirement
// Kernel functional config
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm100;                           // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      // Operator class tag

template <int TileM, int TileN, int TileK,
          int ClusterM, int ClusterN, int ClusterK,
          bool EpilogueAuto, int EpilogueM, int EpilogueN>
struct KernelConfig {
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

  // Pure matmul: D = convert(accumulator), with no alpha/beta visitor path.
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
      ElementC, LayoutCTag, AlignmentC,
      ElementD, LayoutDTag, AlignmentD,
      RequestedEpilogueSchedule,
      FusionOperation>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB,
      ElementAccumulator,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using Config_00 = KernelConfig<256, 256, 256, 2, 1, 1, false, 64, 64>;
using Config_01 = KernelConfig<256, 192, 256, 2, 1, 1, false, 64, 32>;
using Config_02 = KernelConfig<256, 192, 256, 2, 1, 1, false, 128, 16>;
using Config_03 = KernelConfig<256, 256, 256, 2, 1, 1, false, 128, 64>;
using Config_04 = KernelConfig<256, 192, 256, 2, 1, 1, true, 128, 64>;
using Config_05 = KernelConfig<256, 192, 256, 2, 1, 1, false, 64, 16>;
using Config_06 = KernelConfig<256, 192, 256, 2, 1, 1, false, 64, 48>;
using Config_07 = KernelConfig<256, 256, 256, 2, 1, 1, false, 64, 128>;
using Config_08 = KernelConfig<256, 192, 256, 2, 1, 1, false, 128, 32>;
using Config_09 = KernelConfig<256, 192, 256, 2, 1, 1, false, 128, 192>;
using Config_10 = KernelConfig<256, 256, 256, 2, 1, 1, false, 128, 256>;
using Config_11 = KernelConfig<256, 256, 128, 2, 1, 1, false, 128, 64>;
using Config_12 = KernelConfig<256, 256, 128, 2, 2, 1, false, 128, 32>;
using Config_13 = KernelConfig<256, 256, 128, 2, 1, 1, true, 128, 64>;
using Config_14 = KernelConfig<256, 256, 128, 2, 1, 1, false, 128, 128>;
using Config_15 = KernelConfig<256, 192, 256, 2, 1, 1, false, 64, 64>;
using Config_16 = KernelConfig<256, 256, 128, 2, 1, 1, false, 128, 256>;
using Config_17 = KernelConfig<256, 256, 256, 2, 1, 1, true, 128, 64>;
using Config_18 = KernelConfig<256, 256, 256, 2, 1, 1, false, 128, 128>;
using Config_19 = KernelConfig<256, 256, 128, 2, 1, 1, false, 64, 128>;
using Config_20 = KernelConfig<256, 256, 256, 4, 1, 1, false, 64, 64>;
using Config_21 = KernelConfig<256, 256, 256, 4, 1, 1, false, 128, 64>;
using Config_22 = KernelConfig<256, 256, 256, 2, 2, 1, false, 64, 64>;
using Config_23 = KernelConfig<256, 192, 256, 2, 2, 1, false, 64, 32>;
using Config_24 = KernelConfig<256, 256, 128, 4, 1, 1, false, 128, 32>;
using Config_25 = KernelConfig<256, 256, 128, 2, 2, 1, false, 128, 64>;
using Config_26 = KernelConfig<256, 192, 256, 2, 1, 1, false, 128, 64>;
using Config_27 = KernelConfig<256, 256, 128, 2, 1, 1, false, 128, 32>;
using Config_28 = KernelConfig<256, 192, 256, 4, 1, 1, false, 128, 64>;
using Config_29 = KernelConfig<256, 128, 256, 2, 2, 1, false, 128, 32>;
using Config_30 = KernelConfig<256, 128, 256, 2, 2, 1, false, 128, 16>;
using Config_31 = KernelConfig<256, 192, 256, 4, 1, 1, false, 128, 48>;
using Config_32 = KernelConfig<256, 192, 256, 2, 2, 1, false, 128, 32>;
using Config_33 = KernelConfig<256, 192, 256, 4, 1, 1, true, 128, 64>;
using Config_34 = KernelConfig<256, 192, 256, 2, 1, 1, false, 128, 48>;
using Config_35 = KernelConfig<256, 192, 256, 4, 1, 1, false, 128, 32>;
using Config_36 = KernelConfig<256, 192, 256, 2, 2, 1, true, 128, 64>;
using Config_37 = KernelConfig<256, 192, 256, 4, 1, 1, false, 128, 96>;
using Config_38 = KernelConfig<256, 192, 256, 2, 2, 1, false, 128, 48>;
using Config_39 = KernelConfig<256, 192, 256, 2, 2, 1, false, 128, 96>;
using Config_40 = KernelConfig<256, 192, 256, 2, 1, 1, false, 128, 96>;
using Config_41 = KernelConfig<256, 192, 256, 2, 2, 1, false, 128, 64>;
using Config_42 = KernelConfig<256, 128, 256, 4, 1, 1, false, 128, 32>;
using Config_43 = KernelConfig<256, 128, 256, 2, 1, 1, false, 128, 32>;
using Config_44 = KernelConfig<128, 192, 256, 1, 1, 1, false, 128, 64>;
using Config_45 = KernelConfig<256, 192, 256, 2, 2, 1, false, 128, 16>;
using Config_46 = KernelConfig<256, 192, 256, 4, 1, 1, false, 128, 16>;
using Config_47 = KernelConfig<256, 192, 256, 4, 1, 1, false, 64, 32>;
using Config_48 = KernelConfig<256, 256, 128, 4, 1, 1, false, 128, 64>;
using Config_49 = KernelConfig<128, 256, 128, 4, 1, 1, false, 128, 64>;
using Config_50 = KernelConfig<256, 256, 128, 4, 1, 1, false, 128, 128>;
using Config_51 = KernelConfig<256, 192, 256, 4, 1, 1, false, 64, 64>;
using Config_52 = KernelConfig<256, 192, 256, 2, 2, 1, false, 64, 64>;
using Config_53 = KernelConfig<256, 256, 128, 4, 1, 1, false, 64, 32>;
using Config_54 = KernelConfig<256, 128, 256, 2, 1, 1, true, 128, 64>;
using Config_55 = KernelConfig<256, 128, 256, 2, 1, 1, false, 128, 16>;
using Config_56 = KernelConfig<256, 192, 128, 2, 1, 1, false, 128, 32>;
using Config_57 = KernelConfig<256, 192, 128, 2, 1, 1, true, 128, 64>;
using Config_58 = KernelConfig<256, 256, 256, 2, 2, 1, false, 128, 64>;
using Config_59 = KernelConfig<256, 128, 256, 2, 1, 1, false, 64, 64>;
using Config_60 = KernelConfig<256, 256, 256, 4, 2, 1, false, 64, 64>;
using Config_61 = KernelConfig<128, 256, 128, 1, 2, 1, true, 128, 64>;
using Config_62 = KernelConfig<128, 256, 128, 1, 2, 1, false, 128, 128>;
using Config_63 = KernelConfig<128, 256, 128, 1, 1, 1, false, 128, 128>;
using Config_64 = KernelConfig<128, 256, 128, 2, 1, 1, true, 128, 64>;
using Config_65 = KernelConfig<128, 256, 128, 1, 1, 1, true, 128, 64>;
using Config_66 = KernelConfig<128, 256, 128, 2, 1, 1, false, 128, 128>;
using Config_67 = KernelConfig<256, 256, 64, 2, 1, 1, false, 128, 128>;
using Config_68 = KernelConfig<256, 256, 64, 2, 1, 1, true, 128, 64>;
using Config_69 = KernelConfig<128, 256, 128, 2, 2, 1, false, 128, 128>;
using Config_70 = KernelConfig<128, 256, 64, 2, 1, 1, true, 128, 64>;
using Config_71 = KernelConfig<256, 256, 128, 2, 2, 1, false, 128, 128>;
using Config_72 = KernelConfig<128, 256, 128, 4, 1, 1, true, 128, 64>;
using Config_73 = KernelConfig<128, 256, 64, 1, 2, 1, false, 128, 128>;
using Config_74 = KernelConfig<128, 256, 128, 2, 2, 1, true, 128, 64>;
using Config_75 = KernelConfig<256, 256, 128, 2, 2, 1, true, 128, 64>;
using Config_76 = KernelConfig<128, 256, 128, 4, 1, 1, false, 128, 128>;
using Config_77 = KernelConfig<256, 192, 128, 2, 1, 1, false, 128, 64>;
using Config_78 = KernelConfig<128, 192, 128, 1, 1, 1, false, 128, 64>;
using Config_79 = KernelConfig<128, 192, 128, 1, 2, 1, false, 128, 64>;
using Config_80 = KernelConfig<128, 192, 128, 2, 1, 1, false, 128, 64>;
using Config_81 = KernelConfig<256, 256, 128, 4, 1, 1, true, 128, 64>;
using Config_82 = KernelConfig<128, 192, 128, 1, 2, 1, false, 128, 96>;
using Config_83 = KernelConfig<128, 192, 128, 2, 1, 1, false, 128, 96>;
using Config_84 = KernelConfig<128, 192, 128, 4, 1, 1, false, 128, 64>;
using Config_85 = KernelConfig<128, 192, 128, 1, 1, 1, false, 128, 96>;
using Config_86 = KernelConfig<128, 256, 64, 4, 1, 1, false, 128, 128>;
using Config_87 = KernelConfig<128, 192, 128, 4, 1, 1, false, 128, 96>;
using Config_88 = KernelConfig<256, 192, 128, 2, 1, 1, false, 128, 96>;
using Config_89 = KernelConfig<256, 192, 64, 2, 1, 1, false, 128, 96>;
using Config_90 = KernelConfig<256, 192, 64, 2, 1, 1, false, 128, 64>;
using Config_91 = KernelConfig<128, 256, 128, 2, 1, 1, false, 128, 64>;
using Config_92 = KernelConfig<128, 256, 64, 1, 2, 1, true, 128, 64>;
using Config_93 = KernelConfig<128, 256, 64, 1, 1, 1, false, 128, 128>;
// Focused bottleneck probes. These extend the pure-matmul search binary without
// changing op1_best, and expose smaller N tiles plus a larger K tile.
using Config_94 = KernelConfig<256, 256, 512, 2, 1, 1, true, 128, 64>;
using Config_95 = KernelConfig<128, 128, 128, 1, 1, 1, true, 128, 64>;
using Config_96 = KernelConfig<128, 64, 128, 1, 1, 1, true, 128, 64>;
using Config_97 = KernelConfig<128, 128, 256, 1, 1, 1, true, 128, 64>;
using Config_98 = KernelConfig<128, 64, 256, 1, 1, 1, true, 128, 64>;

using ReferenceConfig = Config_00;
using Gemm = typename ReferenceConfig::Gemm;

// Reference device GEMM implementation type
using StrideA   = typename Gemm::GemmKernel::StrideA;
using LayoutA   = decltype(cute::make_layout(make_shape(0,0,0), StrideA{}));
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
using StrideB   = typename Gemm::GemmKernel::StrideB;
using LayoutB   = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
using StrideC   = typename Gemm::GemmKernel::StrideC;
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutD   = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));

//
// Data members
//

/// Initialization
StrideA stride_A;
LayoutA layout_A;
LayoutSFA layout_SFA;
StrideB stride_B;
LayoutB layout_B;
LayoutSFB layout_SFB;
StrideC stride_C;
StrideD stride_D;
LayoutD layout_D;
uint64_t seed;

// The HostTensors are only used for allocating memory on host and device, and transferring data between host and device
// Use cute::Tensor and cute::Layout for iterating thru the matrix elements
cutlass::HostTensor<ElementA::DataType, cutlass::layout::RowMajor> block_A;
cutlass::HostTensor<ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
cutlass::HostTensor<ElementB::DataType, cutlass::layout::ColumnMajor> block_B;
cutlass::HostTensor<ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
// Output Tensor
cutlass::HostTensor<ElementD, cutlass::layout::RowMajor> block_D;
// Reference Output Tensor
cutlass::HostTensor<ElementD, cutlass::layout::RowMajor> block_reference_D;

// Experiment-only device buffers avoid multi-GB CPU HostTensor initialization.
cutlass::DeviceAllocation<uint8_t> fast_A;
cutlass::DeviceAllocation<uint8_t> fast_B;
cutlass::DeviceAllocation<uint8_t> fast_D;
cutlass::DeviceAllocation<uint8_t> fast_SFA;
cutlass::DeviceAllocation<uint8_t> fast_SFB;
#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <typename T>
auto make_iterator(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  int iterations;
  int m, n, k;
  int shape_id = 0;
  int config_index = -1;
  int config_order_start = 0;
  int config_order_step = 1;
  int swizzle = 0;
  std::string dataset_root;
  bool generate_dataset = false;
  bool dataset_only = false;

  Options():
    help(false),
    m(1024), n(1024), k(1024),
    iterations(10),
    swizzle(0)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("shape-id", shape_id);
    cmd.get_cmd_line_argument("config-index", config_index);
    cmd.get_cmd_line_argument("config-order-start", config_order_start);
    cmd.get_cmd_line_argument("config-order-step", config_order_step);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("swizzle", swizzle);
    cmd.get_cmd_line_argument("dataset-root", dataset_root);
    generate_dataset = cmd.check_cmd_line_flag("generate-dataset");
    dataset_only = cmd.check_cmd_line_flag("dataset-only");
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "72a_blackwell_nvfp4_bf16_gemm\n\n"
      << "  Blackwell NVFP4 matmul (ElementC=void) using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --config-order-start=<int>  First generated config index when scanning all configs\n"
      << "  --config-order-step=<int>   Config traversal direction: positive or negative\n"
      << "  --swizzle=<int>             Cluster rasterization swizzle\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";
    out << "  --dataset-root=<path>      Persistent random A/B/SFA/SFB dataset root\n"
        << "  --generate-dataset         Generate missing deterministic dataset files\n"
        << "  --dataset-only             Initialize/load data and exit before GEMM\n\n";

    out << "\n\nExamples:\n\n"
      << "$ " << "./examples/72_blackwell_narrow_precision_gemm/M_72a_blackwell_nvfp4_bf16_gemm" << " --m=1024 --n=512 --k=1024 --alpha=1 --beta=0 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <typename Element, typename Layout>
bool initialize_block(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed) {

  double scope_max, scope_min;
  constexpr int bits_input = cutlass::sizeof_bits<Element>::value;

  if constexpr (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  }
  else if constexpr (bits_input <= 6) {
    scope_max = 2;
    scope_min = -2;
  }
  else if constexpr (bits_input <= 8) {
    if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t>) {
      scope_max = 4;
      scope_min = 1;
    }
    else {
      scope_max = 1;
      scope_min = -1;
    }
  }
  else{
    scope_max = 4;
    scope_min = -4;
  }
  cutlass::reference::host::TensorFillRandomUniform(
    view, seed, scope_max, scope_min, 0);

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {
  using namespace cute;
  // For SFA and SFB tensors layouts
  using Sm1xxBlkScaledConfig =  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

  layout_A = make_layout(make_shape(options.m, options.k, 1), stride_A);
  layout_B = make_layout(make_shape(options.n, options.k, 1), stride_B);
  layout_D = make_layout(make_shape(options.m, options.n, 1), stride_D);
  layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(options.m, options.n, options.k, 1));
  layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(options.m, options.n, options.k, 1));

  auto packed_bytes = [](uint64_t elements, int bits) {
    return size_t((elements * uint64_t(bits) + 7) / 8);
  };
  fast_A.reset(packed_bytes(uint64_t(options.m) * options.k,
                            cutlass::sizeof_bits<ElementA::DataType>::value));
  fast_B.reset(packed_bytes(uint64_t(options.n) * options.k,
                            cutlass::sizeof_bits<ElementB::DataType>::value));
  fast_D.reset(packed_bytes(uint64_t(options.m) * options.n,
                            cutlass::sizeof_bits<ElementD>::value));
  fast_SFA.reset(packed_bytes(uint64_t(size(filter_zeros(layout_SFA))),
                              cutlass::sizeof_bits<ElementA::ScaleFactorType>::value));
  fast_SFB.reset(packed_bytes(uint64_t(size(filter_zeros(layout_SFB))),
                              cutlass::sizeof_bits<ElementB::ScaleFactorType>::value));
  if (options.dataset_root.empty()) {
    throw std::runtime_error("--dataset-root is required by the random-data harness");
  }

  if (options.generate_dataset) {
    block_A.reset(cutlass::make_Coord(options.m, options.k));
    block_B.reset(cutlass::make_Coord(options.n, options.k));
    block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
    block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));
    initialize_block(block_A.host_view(), seed + 2021);
    initialize_block(block_B.host_view(), seed + 2022);
    initialize_block(block_SFA.host_view(), seed + 2024);
    initialize_block(block_SFB.host_view(), seed + 2025);

    auto directory = m72a_dataset::shape_directory(
        options.dataset_root, options.m, options.n, options.k);
    m72a_dataset::write_cutlass_tensor_file(
        directory / "A.bin", block_A.host_data(), fast_A.bytes(), seed + 2021);
    m72a_dataset::write_cutlass_tensor_file(
        directory / "B.bin", block_B.host_data(), fast_B.bytes(), seed + 2022);
    m72a_dataset::write_cutlass_tensor_file(
        directory / "SFA.bin", block_SFA.host_data(), fast_SFA.bytes(), seed + 2024);
    m72a_dataset::write_cutlass_tensor_file(
        directory / "SFB.bin", block_SFB.host_data(), fast_SFB.bytes(), seed + 2025);
  }

  uint64_t shape_seed = 0x72a0000000000000ULL ^
      (uint64_t(options.m) << 32) ^ (uint64_t(options.n) << 12) ^ uint64_t(options.k);
  m72a_dataset::load_operand(options.dataset_root, options.m, options.n, options.k,
      "A", fast_A.get(), fast_A.bytes(), shape_seed ^ 0xa11ce001ULL,
      m72a_dataset::DataKind::kPackedFp4, false);
  m72a_dataset::load_operand(options.dataset_root, options.m, options.n, options.k,
      "B", fast_B.get(), fast_B.bytes(), shape_seed ^ 0xb11ce002ULL,
      m72a_dataset::DataKind::kPackedFp4, false);
  m72a_dataset::load_operand(options.dataset_root, options.m, options.n, options.k,
      "SFA", fast_SFA.get(), fast_SFA.bytes(), shape_seed ^ 0x5fa00003ULL,
      m72a_dataset::DataKind::kUe4m3Scale, false);
  m72a_dataset::load_operand(options.dataset_root, options.m, options.n, options.k,
      "SFB", fast_SFB.get(), fast_SFB.bytes(), shape_seed ^ 0x5fb00004ULL,
      m72a_dataset::DataKind::kUe4m3Scale, false);
  m72a_dataset::fill_device_random(fast_D.get(), fast_D.bytes(), shape_seed ^ 0xd0000005ULL);
}

// Populates a Gemm::Arguments structure from the given commandline options
template <typename GemmType>
typename GemmType::Arguments args_from_options(const Options &options)
{
  typename GemmType::Arguments arguments {
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, 1},
    { // Mainloop arguments
      reinterpret_cast<ElementA::DataType*>(fast_A.get()), stride_A,
      reinterpret_cast<ElementB::DataType*>(fast_B.get()), stride_B,
      reinterpret_cast<ElementA::ScaleFactorType*>(fast_SFA.get()), layout_SFA,
      reinterpret_cast<ElementB::ScaleFactorType*>(fast_SFB.get()), layout_SFB
    },
    { // Epilogue arguments
      {},
      nullptr, stride_C,
      reinterpret_cast<ElementD*>(fast_D.get()), stride_D
    }
  };

  arguments.scheduler.max_swizzle_size = options.swizzle;
  return arguments;
}

bool verify(const Options &options) {
  return true;
  // using namespace cute;
  // // Create the arguments for host reference implementation
  // Tensor tensor_A = make_tensor(make_iterator(block_A.host_data()), layout_A);
  // Tensor tensor_SFA = make_tensor(block_SFA.host_data(), layout_SFA);
  // Tensor tensor_B = make_tensor(make_iterator(block_B.host_data()), layout_B);
  // Tensor tensor_SFB = make_tensor(block_SFB.host_data(), layout_SFB);

  // cutlass::reference::host::GettBlockScalingMainloopParams<
  //     ElementAccumulator,                 // ElementAccumulator
  //     decltype(tensor_A),                 // TensorA
  //     decltype(tensor_SFA),               // TensorSfA
  //     decltype(tensor_B),                 // TensorB
  //     decltype(tensor_SFB)                // TensorSfB
  //   > mainloop_params{tensor_A, tensor_SFA, tensor_B, tensor_SFB};

  // auto tensor_C = cute::make_tensor(make_iterator(block_C.host_data()), layout_C);
  // auto tensor_D = cute::make_tensor(make_iterator(block_reference_D.host_data()), layout_D);

  // cutlass::reference::host::GettBlockScalingEpilogueParams<
  //     ElementAccumulator,                   // ElementScalar
  //     ElementAccumulator,                   // ElementAccumulator
  //     ElementAccumulator,                   // ElementCompute
  //     decltype(tensor_C),                   // TensorC
  //     decltype(tensor_D)                    // TensorD

  // cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  // // Comparison
  // block_D.sync_host();
  // bool passed = cutlass::reference::host::TensorEquals(block_reference_D.host_view(), block_D.host_view());
  // passed &= (cutlass::reference::host::TensorNorm(block_reference_D.host_view()) > 0);
  // passed &= (cutlass::reference::host::TensorNorm(block_D.host_view()) > 0);

  // return passed;
}

/// Execute a given example GEMM computation
template <typename Config>
int run_config(char const* variant, int shape_id, Options &options)
{
  using GemmType = typename Config::Gemm;

  // Instantiate CUTLASS kernel depending on templates
  GemmType gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options<GemmType>(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = GemmType::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  auto implementable = gemm.can_implement(arguments);
  if (implementable != cutlass::Status::kSuccess) {
    std::cout << "RESULT," << shape_id << ',' << variant
              << ",can_implement_failed,0,0" << std::endl;
    return 1;
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  cudaDeviceSynchronize();

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    float elapsed_ms = timer.elapsed_millis();
    double avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    std::cout << "RESULT," << shape_id << ',' << variant << ",ok,"
              << options.iterations << ',' << avg_runtime_ms << std::endl;
  }

  return 0;
}

int run_one_generated_config(int shape_id, Options &options) {
  int failures = 0;
#if defined(OVERHEAD_TARGETS_ONLY)
  if (shape_id == 8) {
    failures += run_config<Config_00>(
        "M256_N256_K256_E64x64_C2x1x1", shape_id, options);
  } else if (shape_id == 11) {
    failures += run_config<Config_78>(
        "M128_N192_K128_E128x64_C1x1x1", shape_id, options);
  } else if (shape_id == 13) {
    failures += run_config<Config_63>(
        "M128_N256_K128_E128x128_C1x1x1", shape_id, options);
  }
#else
  if ((options.config_index < 0 || options.config_index == 0) && (shape_id == 1 || shape_id == 2 || shape_id == 3 || shape_id == 4 || shape_id == 7 || shape_id == 8 || shape_id == 9)) failures += run_config<Config_00>("M256_N256_K256_E64x64_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 1) && (shape_id == 1 || shape_id == 2 || shape_id == 3 || shape_id == 4 || shape_id == 6 || shape_id == 7)) failures += run_config<Config_01>("M256_N192_K256_E64x32_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 2) && (shape_id == 1 || shape_id == 2 || shape_id == 3 || shape_id == 4 || shape_id == 6 || shape_id == 7 || shape_id == 8 || shape_id == 9)) failures += run_config<Config_02>("M256_N192_K256_E128x16_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 3) && (shape_id == 1 || shape_id == 2 || shape_id == 3 || shape_id == 4 || shape_id == 6 || shape_id == 7 || shape_id == 8 || shape_id == 9)) failures += run_config<Config_03>("M256_N256_K256_E128x64_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 4) && (shape_id == 1 || shape_id == 2 || shape_id == 3 || shape_id == 4 || shape_id == 5 || shape_id == 6 || shape_id == 7 || shape_id == 8 || shape_id == 9)) failures += run_config<Config_04>("M256_N192_K256_Eauto_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 5) && (shape_id == 1 || shape_id == 2 || shape_id == 3 || shape_id == 4)) failures += run_config<Config_05>("M256_N192_K256_E64x16_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 6) && (shape_id == 1 || shape_id == 2 || shape_id == 3 || shape_id == 4 || shape_id == 7)) failures += run_config<Config_06>("M256_N192_K256_E64x48_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 7) && (shape_id == 1 || shape_id == 2 || shape_id == 3 || shape_id == 4 || shape_id == 7 || shape_id == 8 || shape_id == 9)) failures += run_config<Config_07>("M256_N256_K256_E64x128_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 8) && (shape_id == 1 || shape_id == 2 || shape_id == 3 || shape_id == 4 || shape_id == 5 || shape_id == 6 || shape_id == 7 || shape_id == 8 || shape_id == 9)) failures += run_config<Config_08>("M256_N192_K256_E128x32_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 9) && (shape_id == 1)) failures += run_config<Config_09>("M256_N192_K256_E128x192_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 10) && (shape_id == 1)) failures += run_config<Config_10>("M256_N256_K256_E128x256_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 11) && (shape_id == 1 || shape_id == 2 || shape_id == 3 || shape_id == 4 || shape_id == 8 || shape_id == 9 || shape_id == 11 || shape_id == 13)) failures += run_config<Config_11>("M256_N256_K128_E128x64_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 12) && (shape_id == 1 || shape_id == 2 || shape_id == 3 || shape_id == 4)) failures += run_config<Config_12>("M256_N256_K128_E128x32_C2x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 13) && (shape_id == 1 || shape_id == 2 || shape_id == 3 || shape_id == 4 || shape_id == 10 || shape_id == 11 || shape_id == 12 || shape_id == 13)) failures += run_config<Config_13>("M256_N256_K128_Eauto_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 14) && (shape_id == 1 || shape_id == 2 || shape_id == 3 || shape_id == 4 || shape_id == 10 || shape_id == 11 || shape_id == 12 || shape_id == 13)) failures += run_config<Config_14>("M256_N256_K128_E128x128_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 15) && (shape_id == 1 || shape_id == 2 || shape_id == 3 || shape_id == 4 || shape_id == 7 || shape_id == 8 || shape_id == 9)) failures += run_config<Config_15>("M256_N192_K256_E64x64_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 16) && (shape_id == 1 || shape_id == 2 || shape_id == 3 || shape_id == 4)) failures += run_config<Config_16>("M256_N256_K128_E128x256_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 17) && (shape_id == 1 || shape_id == 2 || shape_id == 5 || shape_id == 8 || shape_id == 10 || shape_id == 13)) failures += run_config<Config_17>("M256_N256_K256_Eauto_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 18) && (shape_id == 1 || shape_id == 2 || shape_id == 6 || shape_id == 8 || shape_id == 10 || shape_id == 13)) failures += run_config<Config_18>("M256_N256_K256_E128x128_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 19) && (shape_id == 1 || shape_id == 4)) failures += run_config<Config_19>("M256_N256_K128_E64x128_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 20) && (shape_id == 2 || shape_id == 4 || shape_id == 8 || shape_id == 9)) failures += run_config<Config_20>("M256_N256_K256_E64x64_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 21) && (shape_id == 2 || shape_id == 4 || shape_id == 6 || shape_id == 7 || shape_id == 8 || shape_id == 9)) failures += run_config<Config_21>("M256_N256_K256_E128x64_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 22) && (shape_id == 2 || shape_id == 3 || shape_id == 8 || shape_id == 9)) failures += run_config<Config_22>("M256_N256_K256_E64x64_C2x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 23) && (shape_id == 3 || shape_id == 6)) failures += run_config<Config_23>("M256_N192_K256_E64x32_C2x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 24) && (shape_id == 3 || shape_id == 7)) failures += run_config<Config_24>("M256_N256_K128_E128x32_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 25) && (shape_id == 3)) failures += run_config<Config_25>("M256_N256_K128_E128x64_C2x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 26) && (shape_id == 3 || shape_id == 4 || shape_id == 5)) failures += run_config<Config_26>("M256_N192_K256_E128x64_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 27) && (shape_id == 4 || shape_id == 9)) failures += run_config<Config_27>("M256_N256_K128_E128x32_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 28) && (shape_id == 5)) failures += run_config<Config_28>("M256_N192_K256_E128x64_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 29) && (shape_id == 5)) failures += run_config<Config_29>("M256_N128_K256_E128x32_C2x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 30) && (shape_id == 5)) failures += run_config<Config_30>("M256_N128_K256_E128x16_C2x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 31) && (shape_id == 5)) failures += run_config<Config_31>("M256_N192_K256_E128x48_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 32) && (shape_id == 5 || shape_id == 6)) failures += run_config<Config_32>("M256_N192_K256_E128x32_C2x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 33) && (shape_id == 5 || shape_id == 6 || shape_id == 7)) failures += run_config<Config_33>("M256_N192_K256_Eauto_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 34) && (shape_id == 5)) failures += run_config<Config_34>("M256_N192_K256_E128x48_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 35) && (shape_id == 5 || shape_id == 6 || shape_id == 7 || shape_id == 8)) failures += run_config<Config_35>("M256_N192_K256_E128x32_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 36) && (shape_id == 5 || shape_id == 6)) failures += run_config<Config_36>("M256_N192_K256_Eauto_C2x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 37) && (shape_id == 5)) failures += run_config<Config_37>("M256_N192_K256_E128x96_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 38) && (shape_id == 5)) failures += run_config<Config_38>("M256_N192_K256_E128x48_C2x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 39) && (shape_id == 5)) failures += run_config<Config_39>("M256_N192_K256_E128x96_C2x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 40) && (shape_id == 5)) failures += run_config<Config_40>("M256_N192_K256_E128x96_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 41) && (shape_id == 5)) failures += run_config<Config_41>("M256_N192_K256_E128x64_C2x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 42) && (shape_id == 5)) failures += run_config<Config_42>("M256_N128_K256_E128x32_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 43) && (shape_id == 5 || shape_id == 8 || shape_id == 9)) failures += run_config<Config_43>("M256_N128_K256_E128x32_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 44) && (shape_id == 5)) failures += run_config<Config_44>("M128_N192_K256_E128x64_C1x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 45) && (shape_id == 6 || shape_id == 8 || shape_id == 9)) failures += run_config<Config_45>("M256_N192_K256_E128x16_C2x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 46) && (shape_id == 6 || shape_id == 7 || shape_id == 9)) failures += run_config<Config_46>("M256_N192_K256_E128x16_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 47) && (shape_id == 6)) failures += run_config<Config_47>("M256_N192_K256_E64x32_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 48) && (shape_id == 6 || shape_id == 7)) failures += run_config<Config_48>("M256_N256_K128_E128x64_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 49) && (shape_id == 6)) failures += run_config<Config_49>("M128_N256_K128_E128x64_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 50) && (shape_id == 6 || shape_id == 10 || shape_id == 12)) failures += run_config<Config_50>("M256_N256_K128_E128x128_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 51) && (shape_id == 6)) failures += run_config<Config_51>("M256_N192_K256_E64x64_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 52) && (shape_id == 6)) failures += run_config<Config_52>("M256_N192_K256_E64x64_C2x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 53) && (shape_id == 7)) failures += run_config<Config_53>("M256_N256_K128_E64x32_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 54) && (shape_id == 7 || shape_id == 8 || shape_id == 9)) failures += run_config<Config_54>("M256_N128_K256_Eauto_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 55) && (shape_id == 7 || shape_id == 8 || shape_id == 9)) failures += run_config<Config_55>("M256_N128_K256_E128x16_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 56) && (shape_id == 7)) failures += run_config<Config_56>("M256_N192_K128_E128x32_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 57) && (shape_id == 7)) failures += run_config<Config_57>("M256_N192_K128_Eauto_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 58) && (shape_id == 8 || shape_id == 9)) failures += run_config<Config_58>("M256_N256_K256_E128x64_C2x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 59) && (shape_id == 8 || shape_id == 9)) failures += run_config<Config_59>("M256_N128_K256_E64x64_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 60) && (shape_id == 9)) failures += run_config<Config_60>("M256_N256_K256_E64x64_C4x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 61) && (shape_id == 10 || shape_id == 13)) failures += run_config<Config_61>("M128_N256_K128_Eauto_C1x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 62) && (shape_id == 10 || shape_id == 13)) failures += run_config<Config_62>("M128_N256_K128_E128x128_C1x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 63) && (shape_id == 10 || shape_id == 11 || shape_id == 12 || shape_id == 13)) failures += run_config<Config_63>("M128_N256_K128_E128x128_C1x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 64) && (shape_id == 10 || shape_id == 11 || shape_id == 12 || shape_id == 13)) failures += run_config<Config_64>("M128_N256_K128_Eauto_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 65) && (shape_id == 10 || shape_id == 11 || shape_id == 12 || shape_id == 13)) failures += run_config<Config_65>("M128_N256_K128_Eauto_C1x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 66) && (shape_id == 10 || shape_id == 11 || shape_id == 12 || shape_id == 13)) failures += run_config<Config_66>("M128_N256_K128_E128x128_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 67) && (shape_id == 10 || shape_id == 13)) failures += run_config<Config_67>("M256_N256_K64_E128x128_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 68) && (shape_id == 10 || shape_id == 13)) failures += run_config<Config_68>("M256_N256_K64_Eauto_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 69) && (shape_id == 10)) failures += run_config<Config_69>("M128_N256_K128_E128x128_C2x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 70) && (shape_id == 10)) failures += run_config<Config_70>("M128_N256_K64_Eauto_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 71) && (shape_id == 10 || shape_id == 13)) failures += run_config<Config_71>("M256_N256_K128_E128x128_C2x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 72) && (shape_id == 10 || shape_id == 11 || shape_id == 12 || shape_id == 13)) failures += run_config<Config_72>("M128_N256_K128_Eauto_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 73) && (shape_id == 10)) failures += run_config<Config_73>("M128_N256_K64_E128x128_C1x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 74) && (shape_id == 10)) failures += run_config<Config_74>("M128_N256_K128_Eauto_C2x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 75) && (shape_id == 10)) failures += run_config<Config_75>("M256_N256_K128_Eauto_C2x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 76) && (shape_id == 11 || shape_id == 12 || shape_id == 13)) failures += run_config<Config_76>("M128_N256_K128_E128x128_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 77) && (shape_id == 11 || shape_id == 12)) failures += run_config<Config_77>("M256_N192_K128_E128x64_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 78) && (shape_id == 11 || shape_id == 12 || shape_id == 13)) failures += run_config<Config_78>("M128_N192_K128_E128x64_C1x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 79) && (shape_id == 11 || shape_id == 12)) failures += run_config<Config_79>("M128_N192_K128_E128x64_C1x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 80) && (shape_id == 11 || shape_id == 12)) failures += run_config<Config_80>("M128_N192_K128_E128x64_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 81) && (shape_id == 11 || shape_id == 12)) failures += run_config<Config_81>("M256_N256_K128_Eauto_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 82) && (shape_id == 11 || shape_id == 12)) failures += run_config<Config_82>("M128_N192_K128_E128x96_C1x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 83) && (shape_id == 11 || shape_id == 12)) failures += run_config<Config_83>("M128_N192_K128_E128x96_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 84) && (shape_id == 11)) failures += run_config<Config_84>("M128_N192_K128_E128x64_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 85) && (shape_id == 11 || shape_id == 12)) failures += run_config<Config_85>("M128_N192_K128_E128x96_C1x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 86) && (shape_id == 11)) failures += run_config<Config_86>("M128_N256_K64_E128x128_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 87) && (shape_id == 11)) failures += run_config<Config_87>("M128_N192_K128_E128x96_C4x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 88) && (shape_id == 12)) failures += run_config<Config_88>("M256_N192_K128_E128x96_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 89) && (shape_id == 12)) failures += run_config<Config_89>("M256_N192_K64_E128x96_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 90) && (shape_id == 12)) failures += run_config<Config_90>("M256_N192_K64_E128x64_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 91) && (shape_id == 13)) failures += run_config<Config_91>("M128_N256_K128_E128x64_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 92) && (shape_id == 13)) failures += run_config<Config_92>("M128_N256_K64_Eauto_C1x2x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 93) && (shape_id == 13)) failures += run_config<Config_93>("M128_N256_K64_E128x128_C1x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 94) && (shape_id == 5 || shape_id == 8)) failures += run_config<Config_94>("M256_N256_K512_Eauto_C2x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 95) && (shape_id == 11 || shape_id == 13)) failures += run_config<Config_95>("M128_N128_K128_Eauto_C1x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 96) && (shape_id == 11 || shape_id == 13)) failures += run_config<Config_96>("M128_N64_K128_Eauto_C1x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 97) && (shape_id == 5 || shape_id == 8)) failures += run_config<Config_97>("M128_N128_K256_Eauto_C1x1x1", shape_id, options);
  if ((options.config_index < 0 || options.config_index == 98) && (shape_id == 5 || shape_id == 8)) failures += run_config<Config_98>("M128_N64_K256_Eauto_C1x1x1", shape_id, options);
#endif
  return failures;
}

int run_generated_configs(int shape_id, Options &options) {
  if (options.config_index >= 0) {
    return run_one_generated_config(shape_id, options);
  }

  constexpr int ConfigCount = 99;
  int saved_config_index = options.config_index;
  int index = ((options.config_order_start % ConfigCount) + ConfigCount) % ConfigCount;
  int step = options.config_order_step < 0 ? -1 : 1;
  int failures = 0;
  for (int visited = 0; visited < ConfigCount; ++visited) {
    options.config_index = index;
    failures += run_one_generated_config(shape_id, options);
    index = (index + step + ConfigCount) % ConfigCount;
  }
  options.config_index = saved_config_index;
  return failures;
}

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.8 or higher Toolkit to run this example
  // and must have compute capability at least 100.
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    std::cerr << "This example requires CUDA 12.8 or newer." << std::endl;
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));

  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));

  if (props.major != 10 || (props.minor != 0 && props.minor != 1 && props.minor != 3)) {
    std::cerr << "This example requires a GPU with compute capability 100a|f, 101a|f, or 103a|f)." << std::endl;
    return 0;
  } 

  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  struct Problem { int id, m, n, k; };
  Problem const shapes[] = {
    {1, 170100, 13824, 5120}, {2, 510300, 13824, 5120},
    {3, 170100, 5120, 5120}, {4, 510300, 5120, 5120},
    {5, 8192, 768, 4096}, {6, 16384, 768, 4096},
    {7, 32768, 768, 4096}, {8, 8192, 4096, 2048},
    {9, 16384, 4096, 2048}, {10, 32768, 4096, 128},
    {11, 16384, 768, 128}, {12, 32768, 768, 128},
    {13, 8192, 4096, 128}
  };
  int failures = 0;
  for (auto const& problem : shapes) {
    if (options.shape_id != 0 && options.shape_id != problem.id) {
      continue;
    }
    options.m = problem.m;
    options.n = problem.n;
    options.k = problem.k;
    initialize(options);
    if (options.dataset_only) {
      continue;
    }
    failures += run_generated_configs(problem.id, options);
  }
  if (options.dataset_only) {
    std::cout << "All 13 datasets are ready\n";
  }
  return failures == 0 ? 0 : 2;
#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
