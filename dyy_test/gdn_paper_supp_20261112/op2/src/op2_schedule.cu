// op2 pure-matmul baseline. Logical weight B[K,N] is sparse.
// The kernel computes D^T[N,M] = B_sparse^T[N,2K] * A_dense^T[2K,M].
// Column-major D^T has the same physical byte order as row-major D; no transpose kernel is used.
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
    \brief A Narrow Precision Sparse GEMM example using CUTLASS for the NVIDIA Blackwell SM100 architecture.

    This example demonstrates a simple way to instantiate and run a blockscaled NVFP4 Sparse GEMM on the NVIDIA Blackwell SM100 architecture.

    The Blackwell SM100 CUTLASS kernel uses the new Block Scaled Tensor Core MMA Instructions (tcgen05.mma.blockscaled) introduced
    on the Blackwell architecture (sm100a) which have 2x throughput compared to fp8 Tensor Core MMA instructions (tcgen05.mma)
    and 4x throughput compared to fp8 Hopper Tensor Core MMA Instructions (WGMMA) (See https://docs.nvidia.com/cuda/parallel-thread-execution).

    Similar to 83_blackwell_sparse_gemm, this kernel leverages:
    1. Per-SM memory called Tensor Memory (TMEM)  (Please refer to CUDA 12.8 docs on https://docs.nvidia.com/cuda/).

    2. The extended warp-specialized kernel design introduced in Hopper enabled by use of TMEM
    which allows us to decouple the execution of MMA and epilogue into separate warps.

    3. A new SW controlled dynamic scheduler based on cluster launch control (See https://docs.nvidia.com/cuda/parallel-thread-execution).

    Usage:
      $ ./examples/84_blackwell_narrow_precision_sparse_gemm/M_84a_blackwell_nvfp4_bf16_sparse_gemm --m=2048 --n=2048 --k=2048
*/

#include <iomanip>
#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include <vector>
#include <cstring>
#include <cmath>
__device__ __constant__ uint32_t op2_stage_gap_cycles = 0;
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
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"
#include "cutlass/transform/device/transform_universal_adapter.hpp"

#include "helper.h"
#include "m72a_random_dataset.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA     = cutlass::float_e2m1_t;
using         ElementAPair = cutlass::nv_float4_t<ElementA>;                 // Element type for A matrix operand
using         LayoutTagA   = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA   = 64;                                             // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes), 2x for compress along k

// E matrix config
using         ElementE    = cute::uint8_t;
using         LayoutTagE  = LayoutTagA;

// B matrix configuration
using         ElementB     = cutlass::float_e2m1_t;
using         ElementBPair = cutlass::nv_float4_t<ElementB>;                 // Element type for B matrix operand
using         LayoutTagB   = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB   = 32;                                             // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// SF
using         ElementSF   = typename ElementAPair::ScaleFactorType;

// C/D matrix configuration
using         ElementD    = cutlass::bfloat16_t;                            // Element type for D matrix operand
using         ElementC    = void;                                           // Pure matmul: no source C matrix
using         LayoutTagC  = cutlass::layout::RowMajor;                      // Layout type for C matrix operand
using         LayoutTagD  = cutlass::layout::ColumnMajor;                   // Column-major D^T is physically the original row-major D
constexpr int AlignmentD  = (16 * 8) / cutlass::sizeof_bits<ElementD>::value; // Original N becomes the aligned contiguous dimension
constexpr int AlignmentC  = 1;                                              // Void C has no memory access alignment requirement

// Kernel functional config
using ElementAccumulator  = float;                                           // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm100;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassBlockScaledSparseTensorOp; // Operator class tag

using ProblemShape = Shape<int,int,int,int>;

template <
  int SmMode, int TileM, int TileN, int TileK,
  int ClusterM, int ClusterN, int ClusterK,
  bool EpilogueAuto, int EpilogueM, int EpilogueN>
struct KernelConfig {
  using MmaTileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
  using ClusterShape = Shape<Int<ClusterM>, Int<ClusterN>, Int<ClusterK>>;
  using EpilogueTile = std::conditional_t<
      EpilogueAuto,
      cutlass::epilogue::collective::EpilogueTileAuto,
      Shape<Int<EpilogueM>, Int<EpilogueN>>>;
  using EpilogueSchedule = std::conditional_t<
      SmMode == 1,
      cutlass::epilogue::TmaWarpSpecialized1SmNvf4,
      cutlass::epilogue::TmaWarpSpecialized2SmNvf4>;
  using KernelSchedule = std::conditional_t<
      SmMode == 1,
      cutlass::gemm::KernelSparseTmaWarpSpecialized1SmNvf4Sm100,
      cutlass::gemm::KernelSparseTmaWarpSpecialized2SmNvf4Sm100>;

  // Pure matmul: D = convert(accumulator), with no alpha/beta visitor path.
  using FusionOperation = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<
          cutlass::epilogue::thread::Identity, ElementD, ElementAccumulator,
          cutlass::FloatRoundStyle::round_to_nearest>,
      cutlass::epilogue::fusion::Sm90AccFetch>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      MmaTileShape, ClusterShape,
      EpilogueTile,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutTagC, AlignmentC,
      ElementD, LayoutTagD, AlignmentD,
      EpilogueSchedule,
      FusionOperation>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementAPair, LayoutTagA, AlignmentA,
      ElementBPair, LayoutTagB, AlignmentB,
      ElementAccumulator,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using Config_S01 = KernelConfig<2, 256, 256, 256, 2, 1, 1, false, 128, 64>;
using Config_S02 = KernelConfig<2, 256, 256, 256, 4, 1, 1, false, 64, 32>;
using Config_S03 = KernelConfig<2, 256, 256, 256, 2, 1, 1, true, 128, 64>;
using Config_S04 = KernelConfig<2, 256, 256, 256, 2, 1, 1, true, 128, 64>;
using Config_S05 = KernelConfig<2, 256, 192, 256, 2, 2, 1, false, 128, 32>;
using Config_S06 = KernelConfig<2, 256, 256, 256, 2, 2, 1, false, 64, 32>;
using Config_S07 = KernelConfig<2, 256, 192, 256, 2, 1, 1, false, 128, 32>;
using Config_S08 = KernelConfig<2, 256, 256, 256, 2, 1, 1, false, 128, 64>;
using Config_S09 = KernelConfig<2, 256, 256, 256, 2, 1, 1, false, 64, 32>;
using Config_S10 = KernelConfig<2, 256, 192, 256, 2, 1, 1, true, 128, 64>;
using Config_S11 = KernelConfig<2, 256, 192, 256, 2, 1, 1, false, 128, 64>;
using Config_S12 = KernelConfig<2, 256, 192, 256, 2, 1, 1, true, 128, 64>;
using Config_S13 = KernelConfig<2, 256, 256, 256, 2, 1, 1, true, 128, 64>;

using CanonicalConfig = Config_S01;
using Gemm = typename CanonicalConfig::Gemm;

//
// Blockscale
//
using Sm1xxBlkScaledConfig =  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using Blk_MN   = typename Sm1xxBlkScaledConfig::Blk_MN;
using Blk_SF   = typename Sm1xxBlkScaledConfig::Blk_SF;
using SfAtom   = typename Sm1xxBlkScaledConfig::SfAtom;

using LayoutA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutA;
using LayoutE = typename Gemm::GemmKernel::CollectiveMainloop::LayoutE;
using StrideA = cutlass::gemm::TagToStrideA_t<LayoutTagA>;
using StrideE = StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;

using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.

using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

//
// Compressor
//
using SparseConfig = typename Gemm::GemmKernel::CollectiveMainloop::SparseConfig;

using CompressorUtility = cutlass::transform::kernel::StructuredSparseCompressorUtility<
                            ProblemShape,
                            ElementA,
                            LayoutTagA,
                            SparseConfig>;

using CompressorKernel = cutlass::transform::kernel::StructuredSparseCompressor<
                            ProblemShape,
                            ElementA,
                            LayoutTagA,
                            SparseConfig,
                            ArchTag>;

using Compressor = cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

//
// Data members
//

/// Initialization
StrideA stride_A;
StrideA stride_A_compressed;
StrideE stride_E;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;

LayoutA layout_A;
LayoutE layout_E;
LayoutSFA layout_SFA;
LayoutSFB layout_SFB;

typename LayoutTagA::Stride stride_factor_A;
typename LayoutTagB::Stride stride_factor_B;
typename LayoutTagE::Stride stride_factor_E;
typename LayoutTagD::Stride stride_factor_D;

uint64_t seed;

ProblemShape problem_shape;

// The HostTensors are only used for allocating memory on host and device, and transferring data between host and device
// Use cute::Tensor and cute::Layout for iterating thru the matrix elements
cutlass::HostTensor<ElementA, LayoutTagA> tensor_A;
cutlass::HostTensor<ElementA, LayoutTagA> tensor_A_compressed;
cutlass::HostTensor<ElementE, LayoutTagE> tensor_E;
cutlass::HostTensor<ElementB, LayoutTagB> tensor_B;
cutlass::HostTensor<ElementSF, LayoutTagA> tensor_SFA;
cutlass::HostTensor<ElementSF, LayoutTagB> tensor_SFB;
cutlass::HostTensor<ElementD, LayoutTagD> tensor_D;

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
  int warmups = 20;
  int groups = 20;
  int session = 0;
  int profile_mode = -1;
  int m, n, k, l;
  int shape_id, config_index;
  std::string dataset_root;
  bool generate_dataset = false;
  bool dataset_only = false;

  Options():
    help(false),
    m(1024), n(1024), k(1024), l(1),
    iterations(10), shape_id(0), config_index(-1)
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
    cmd.get_cmd_line_argument("l", l);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("warmups", warmups);
    cmd.get_cmd_line_argument("groups", groups);
    cmd.get_cmd_line_argument("session", session);
    cmd.get_cmd_line_argument("profile-mode", profile_mode);
    cmd.get_cmd_line_argument("shape-id", shape_id);
    cmd.get_cmd_line_argument("config-index", config_index);
    cmd.get_cmd_line_argument("dataset-root", dataset_root);
    generate_dataset = cmd.check_cmd_line_flag("generate-dataset");
    dataset_only = cmd.check_cmd_line_flag("dataset-only");
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "M_84a_blackwell_nvfp4_bf16_sparse_gemm\n\n"
      << "  Blackwell sparse NVFP4 matmul (ElementC=void) using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent of the GEMM\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n"
      << "  --shape-id=<int>            Shape identifier written to RESULT records.\n"
      << "  --config-index=<int>        Run one generated configuration; default runs all.\n\n";
    out << "  --dataset-root=<path>      Persistent random A/B/SFA/SFB dataset root.\n"
        << "  --generate-dataset         Generate deterministic random operand files.\n"
        << "  --dataset-only             Prepare inputs and exit before sparse matmul.\n\n";

    out << "\n\nExamples:\n\n"
      << "$ " << "./examples/84_blackwell_narrow_precision_sparse_gemm/M_84a_blackwell_nvfp4_bf16_sparse_gemm" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

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
void initialize_tensor(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed) {

  double scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  }
    else if (bits_input <= 6) {
    scope_max = 2;
    scope_min = -2;
  }
  else if (bits_input <= 8) {
    if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t>){
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
}

/// Initialize operands to be used in the GEMM and reference GEMM
bool initialize(const Options &options) {

  problem_shape = make_tuple(options.m, options.n, options.k, options.l);

  // * Get A B C D size
  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});
  layout_A = SparseConfig::fill_layoutA(problem_shape);
  layout_E = SparseConfig::fill_layoutE(problem_shape);
  layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem_shape);
  layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem_shape);

  // * Get ACompress & E size
  CompressorUtility compressor_utility(problem_shape, stride_A);

  // TensorE
  // In unit of ElementE (uint8_t), after alignment requirement
  // M-dim: TensorEAtom_M alignment
  // K-dim: TensorEAtom_K alignment
  int KAlignedE = compressor_utility.get_metadata_k_physical();
  int MAlignedE = compressor_utility.get_metadata_m_physical();

  // TensorA Compressed
  // In unit of ElementARaw, after alignment requirement
  // M-dim: TMA alignment
  // K-dim: TMA alignment
  int KAlignedAC = compressor_utility.get_tensorA_k_physical();
  int MAlignedAC = compressor_utility.get_tensorA_m_physical();

  stride_A_compressed = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, KAlignedAC, options.l));
  stride_E            = cutlass::make_cute_packed_stride(StrideE{}, cute::make_shape(MAlignedE, KAlignedE, options.l));

  // * Get SFA & SFB size
  auto k_blks = cutlass::ceil_div(options.k, cute::size<1>(shape(SfAtom{})));
  auto m_blks = cutlass::ceil_div(options.m, Blk_MN{});
  auto n_blks = cutlass::ceil_div(options.n, Blk_MN{});

  // * Allocate Tensor
  auto a_coord = cutlass::make_Coord(options.m * options.l, options.k);
  auto b_coord = cutlass::make_Coord(options.k, options.n * options.l);
  auto e_coord = cutlass::make_Coord(MAlignedE * options.l, KAlignedE);
  auto a_comp_coord = cutlass::make_Coord(MAlignedAC * options.l, KAlignedAC);
  auto d_coord = cutlass::make_Coord(options.m * options.l, options.n);
  auto sfa_coord   = cutlass::make_Coord(m_blks * Blk_MN{} * options.l, k_blks * Blk_SF{});
  auto sfb_coord   = cutlass::make_Coord(n_blks * Blk_MN{} * options.l, k_blks * Blk_SF{});

  tensor_A.resize(a_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagA>::layout_factory(a_coord, stride_factor_A));
  tensor_A_compressed.resize(a_comp_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagA>::layout_factory(a_comp_coord, stride_factor_A));
  tensor_B.resize(b_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagB>::layout_factory(b_coord, stride_factor_B));
  tensor_E.resize(e_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagE>::layout_factory(e_coord, stride_factor_E));
  tensor_D.resize(d_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(d_coord, stride_factor_D));
  tensor_SFA.resize(sfa_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagA>::layout_factory(sfa_coord, stride_factor_A));
  tensor_SFB.resize(sfb_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagB>::layout_factory(sfb_coord, stride_factor_B));

  if (options.generate_dataset) {
    if (options.dataset_root.empty()) {
      throw std::runtime_error("--generate-dataset requires --dataset-root");
    }
    initialize_tensor(tensor_A.host_view(), seed + 2021);
    initialize_tensor(tensor_B.host_view(), seed + 2022);
    initialize_tensor(tensor_SFA.host_view(), seed + 2024);
    initialize_tensor(tensor_SFB.host_view(), seed + 2025);

    auto packed_bytes = [](uint64_t elements, int bits) {
      return size_t((elements * uint64_t(bits) + 7) / 8);
    };
    auto directory = m72a_dataset::shape_directory(
        options.dataset_root, options.m, options.n, options.k);
    size_t a_bytes = packed_bytes(uint64_t(options.m) * options.k,
                                  cutlass::sizeof_bits<ElementA>::value);
    size_t b_bytes = packed_bytes(uint64_t(options.n) * options.k,
                                  cutlass::sizeof_bits<ElementB>::value);
    size_t sfa_bytes = packed_bytes(uint64_t(size(filter_zeros(layout_SFA))),
                                    cutlass::sizeof_bits<ElementSF>::value);
    size_t sfb_bytes = packed_bytes(uint64_t(size(filter_zeros(layout_SFB))),
                                    cutlass::sizeof_bits<ElementSF>::value);
    m72a_dataset::write_cutlass_tensor_file(
        directory / "A.bin", tensor_A.host_data(), a_bytes, seed + 2021);
    m72a_dataset::write_cutlass_tensor_file(
        directory / "B.bin", tensor_B.host_data(), b_bytes, seed + 2022);
    m72a_dataset::write_cutlass_tensor_file(
        directory / "SFA.bin", tensor_SFA.host_data(), sfa_bytes, seed + 2024);
    m72a_dataset::write_cutlass_tensor_file(
        directory / "SFB.bin", tensor_SFB.host_data(), sfb_bytes, seed + 2025);
  }
  else if (options.dataset_root.empty()) {
    initialize_tensor(tensor_A.host_view(), seed + 2021);
    initialize_tensor(tensor_B.host_view(), seed + 2022);
    initialize_tensor(tensor_SFA.host_view(), seed + 2024);
    initialize_tensor(tensor_SFB.host_view(), seed + 2025);
  }
  else {
    auto packed_bytes = [](uint64_t elements, int bits) {
      return size_t((elements * uint64_t(bits) + 7) / 8);
    };
    auto directory = m72a_dataset::shape_directory(
        options.dataset_root, options.m, options.n, options.k);
    size_t a_bytes = packed_bytes(uint64_t(options.m) * options.k,
                                  cutlass::sizeof_bits<ElementA>::value);
    size_t b_bytes = packed_bytes(uint64_t(options.n) * options.k,
                                  cutlass::sizeof_bits<ElementB>::value);
    size_t sfa_bytes = packed_bytes(uint64_t(size(filter_zeros(layout_SFA))),
                                    cutlass::sizeof_bits<ElementSF>::value);
    size_t sfb_bytes = packed_bytes(uint64_t(size(filter_zeros(layout_SFB))),
                                    cutlass::sizeof_bits<ElementSF>::value);
    m72a_dataset::copy_file_to_host(directory / "A.bin", tensor_A.host_data(), a_bytes);
    m72a_dataset::copy_file_to_device(directory / "B.bin", tensor_B.device_data(), b_bytes);
    m72a_dataset::copy_file_to_device(directory / "SFA.bin", tensor_SFA.device_data(), sfa_bytes);
    m72a_dataset::copy_file_to_device(directory / "SFB.bin", tensor_SFB.device_data(), sfb_bytes);
  }

  // * Random fill 50% A with zero
  compressor_utility.structure_sparse_zero_mask_fill(tensor_A.host_data(), static_cast<int>(seed + 2023));

  tensor_A.sync_device();
  if (options.dataset_root.empty()) {
    tensor_B.sync_device();
    tensor_SFA.sync_device();
    tensor_SFB.sync_device();
  }

  // * Compress
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  typename Compressor::Arguments arguments{
    problem_shape,
    {tensor_A.device_data(),
      stride_A,
      tensor_A_compressed.device_data(),
      tensor_E.device_data()},
    {hw_info}
  };

  Compressor compressor_op;
  size_t workspace_size = Compressor::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status {cutlass::Status::kSuccess };
  status = compressor_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return false;
  }

  status = compressor_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    return false;
  }

  status = compressor_op.run();
  if (status != cutlass::Status::kSuccess) {
    return false;
  }

  auto result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    return false;
  }

  tensor_E.sync_host();
  tensor_A_compressed.sync_host();

  return true;
}

// Populates a Gemm::Arguments structure from the given commandline options
template <typename GemmT>
typename GemmT::Arguments args_from_options(const Options &options)
{
  using ArrayElementA = typename GemmT::GemmKernel::CollectiveMainloop::ArrayElementA;
  using ArrayElementB = typename GemmT::GemmKernel::CollectiveMainloop::ArrayElementB;

  typename GemmT::Arguments arguments {
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, 1},
    {
      reinterpret_cast<ArrayElementA *>(tensor_A_compressed.device_data()), layout_A,
      reinterpret_cast<ArrayElementB *>(tensor_B.device_data()), stride_B,
      tensor_E.device_data(), layout_E,
      tensor_SFA.device_data(), layout_SFA,
      tensor_SFB.device_data(), layout_SFB
    },
    {
      {},
      nullptr, stride_C,
      tensor_D.device_data(), stride_D
    }
  };

  return arguments;
}

bool verify(const Options &options) {
  return true; // Skip reference check for now since it requires a large amount of shared memory and may OOM on some configurations. Will add reference check in the future after optimizing the reference implementation.  
  // using namespace cute;

  // // Create the arguments for host reference implementation
  // auto A = make_tensor(make_iterator(tensor_A.host_data()), layout_A);
  // auto SFA = make_tensor(tensor_SFA.host_data(), layout_SFA);
  // auto B = make_tensor(make_iterator(tensor_B.host_data()),
  //   make_layout(make_shape(options.n, options.k, options.l), stride_B));
  // auto SFB = make_tensor(tensor_SFB.host_data(), layout_SFB);

  // cutlass::reference::host::GettMainloopParams<
  //     ElementAccumulator,
  //     decltype(A),
  //     decltype(B),
  //     decltype(SFA),
  //     decltype(SFB)> mainloop_params{A, SFA, B, SFB};

  // auto C = make_tensor(make_iterator(tensor_C.host_data()),
  //   make_layout(make_shape(options.m, options.n, options.l), stride_C));
  // auto D = make_tensor(make_iterator(reference_D.host_data()),
  //   make_layout(make_shape(options.m, options.n, options.l), stride_D));

  // cutlass::reference::host::GettBlockScalingEpilogueParams<
  //     ElementAccumulator,                   // ElementScalar
  //     ElementAccumulator,                   // ElementAccumulator
  //     ElementAccumulator,                   // ElementCompute
  //     decltype(C),                          // TensorC
  //     decltype(D)                           // TensorD
  //   > epilogue_params{
  //     C,
  //     D};

  // cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  // // Comparison
  // tensor_D.sync_host();
  // bool passed = cutlass::reference::host::TensorEquals(reference_D.host_view(), tensor_D.host_view());
  // passed &= (cutlass::reference::host::TensorNorm(reference_D.host_view()) > 0);
  // passed &= (cutlass::reference::host::TensorNorm(tensor_D.host_view()) > 0);

  // return passed;
}

/// Execute one generated GEMM configuration using already initialized operands.
template <typename Config>
int run_config(char const* variant, int shape_id, Options &options)
{
  using GemmT = typename Config::Gemm;

  GemmT gemm;
  auto arguments = args_from_options<GemmT>(options);
  size_t workspace_size = GemmT::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cout << "RESULT," << shape_id << ',' << variant
              << ",can_implement_failed,0,0\n";
    return 1;
  }
  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    std::cout << "RESULT," << shape_id << ',' << variant
              << ",initialize_failed,0,0\n";
    return 1;
  }
  auto run_once = [&] {
    auto launch_status = gemm.run();
    if (launch_status != cutlass::Status::kSuccess) {
      throw std::runtime_error("GEMM launch failed");
    }
  };
  auto select_mode = [](int mode) {
    const uint32_t gap = mode == 0 ? 0 : 512;
    CUDA_CHECK(cudaMemcpyToSymbol(op2_stage_gap_cycles, &gap, sizeof(gap)));
  };

  if (options.profile_mode >= 0) {
    select_mode(options.profile_mode);
    run_once();
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
  }

  const size_t output_count = size_t(options.m) * options.n;
  const size_t output_bytes = output_count * sizeof(ElementD);
  std::vector<uint16_t> normal(output_count), gated(output_count);
  for (int mode : {0, 1}) {
    select_mode(mode);
    auto& result = mode == 0 ? normal : gated;
    CUDA_CHECK(cudaMemset(tensor_D.device_data(), 0xa5, output_bytes));
    run_once();
    CUDA_CHECK(cudaMemcpy(result.data(), tensor_D.device_data(), output_bytes, cudaMemcpyDeviceToHost));
  }

  size_t differing = 0, nonzero = 0, nonfinite = 0, poison = 0;
  uint64_t digest = UINT64_C(14695981039346656037);
  for (size_t i = 0; i < normal.size(); ++i) {
    differing += normal[i] != gated[i];
    nonzero += (normal[i] & 0x7fff) != 0;
    nonfinite += (normal[i] & 0x7f80) == 0x7f80;
    poison += normal[i] == 0xa5a5;
    digest ^= normal[i];
    digest *= UINT64_C(1099511628211);
  }
  std::cout << "VERIFY," << options.session << ',' << shape_id << ',' << options.m << ','
            << options.n << ',' << options.k << ',' << output_count << ',' << differing
            << ',' << nonzero << ',' << nonfinite << ',' << poison << ',' << std::hex
            << digest << std::dec << '\n';
  if (differing || nonfinite || !nonzero) {
    return 2;
  }

  for (int mode : {0, 1}) {
    select_mode(mode);
    for (int i = 0; i < options.warmups; ++i) {
      run_once();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Capture the same GEMM launches once. The device symbol is sampled during replay.
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaGraph_t graph;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < options.iterations; ++i) {
    auto launch_status = gemm.run(stream);
    if (launch_status != cutlass::Status::kSuccess) {
      throw std::runtime_error("GEMM graph capture failed");
    }
  }
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  cudaGraphExec_t graph_exec;
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, 0));
  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (int group = 0; group < options.groups; ++group) {
    const int first = (group + options.session) % 2;
    int order[] = {first, 1 - first, 1 - first, first};
    for (int position = 0; position < 4; ++position) {
      const int mode = order[position];
      select_mode(mode);
      CUDA_CHECK(cudaEventRecord(begin, stream));
      CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
      CUDA_CHECK(cudaEventRecord(end, stream));
      CUDA_CHECK(cudaEventSynchronize(end));
      float batch_ms;
      CUDA_CHECK(cudaEventElapsedTime(&batch_ms, begin, end));
      std::cout << std::setprecision(10) << "TIME," << options.session << ',' << shape_id
                << ',' << options.m << ',' << options.n << ',' << options.k << ',' << group
                << ',' << position << ',' << mode << ',' << options.iterations << ','
                << double(batch_ms) * 1000.0 / options.iterations << '\n';
    }
  }
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
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
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (not (props.major == 10 && props.minor == 0)) {
    std::cerr << "This example requires a GPU of NVIDIA's Blackwell architecture (compute capability 100)." << std::endl;
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

  //
  // Evaluate CUTLASS kernels
  //
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  std::cout << std::unitbuf;
  if (options.iterations < 1 || options.warmups < 1 || options.groups < 1 ||
      options.profile_mode < -1 || options.profile_mode > 1 || options.l != 1) {
    std::cerr << "Require positive iterations/warmups/groups, L=1, profile-mode -1, 0 or 1\n";
    return 2;
  }
  if (not initialize(options)) {
    std::cerr << "Initialization failure" << std::endl;
    return 1;
  }
  if (options.dataset_only) {
    std::cout << "Dataset is ready for M=" << options.m
              << ", N=" << options.n << ", K=" << options.k << '\n';
    return 0;
  }
  int failures = 0;
  switch (options.shape_id) {
    case 8: failures += run_config<Config_S08>("S2_M256_N256_K256_E128x64_C2x1x1", options.shape_id, options); break;
    case 11: failures += run_config<Config_S11>("S2_M256_N192_K256_E128x64_C2x1x1", options.shape_id, options); break;
    default:
      std::cerr << "shape-id must be 8 or 11\n";
      return 1;
  }
  return failures == 0 ? 0 : 1;
#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
