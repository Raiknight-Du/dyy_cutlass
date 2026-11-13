// op1 per-shape best pure matmul selected from each shape's old top-20.
// ElementC=void; epilogue is Identity(convert(accumulator)); no alpha/beta visitor.
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

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

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
#include <array>
#include <iomanip>
#include <stdexcept>

using namespace cute;

#if defined(PACKED_INPUT)
extern "C" int op1_packed_sizes(
    int config, int M, int N, int K,
    uint64_t* bytes_a, uint64_t* bytes_b);
extern "C" int op1_pack_inputs(
    int config, int M, int N, int K,
    void const* A, void const* B, void const* SFA, void const* SFB,
    void* packed_a, void* packed_b);
#endif

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
using Config_01 = Config_00;
using Config_02 = KernelConfig<256, 192, 256, 2, 2, 1, true, 128, 64>;
using Config_03 = KernelConfig<256, 192, 256, 4, 1, 1, false, 128, 16>;
using Config_04 = KernelConfig<256, 192, 256, 2, 1, 1, false, 128, 16>;
using Config_05 = KernelConfig<128, 256, 128, 1, 2, 1, true, 128, 64>;
using Config_06 = KernelConfig<128, 192, 128, 1, 1, 1, false, 128, 64>;
using Config_07 = KernelConfig<128, 256, 128, 1, 1, 1, false, 128, 128>;
// Shape-7 controls: isolate packed-mainloop correctness from epilogue tiling.
using Config_08 = KernelConfig<256, 192, 256, 2, 1, 1, true, 128, 64>;
using Config_09 = KernelConfig<256, 192, 256, 2, 1, 1, false, 64, 64>;
using Config_10 = KernelConfig<256, 192, 256, 2, 1, 1, false, 128, 64>;

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
extern "C" int op1_pack_reuse(
    int config, int m, int n, int k, void const* a, void const* b,
    void const* sfa, void const* sfb, void* pa, void* pb,
    cudaStream_t stream, bool configure);
#if defined(PACKED_INPUT)
cutlass::DeviceAllocation<uint8_t> fast_packed_A;
cutlass::DeviceAllocation<uint8_t> fast_packed_B;
#endif
#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

__global__ void output_digest_kernel(
    uint16_t const* data, uint64_t elements,
    unsigned long long* sum, unsigned long long* value_xor) {
  uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  unsigned long long local_sum = 0;
  unsigned long long local_xor = 0;
  for (uint64_t i = index; i < elements; i += stride) {
    unsigned long long value =
        (i * 0x9e3779b97f4a7c15ULL) ^ uint64_t(data[i]);
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    local_sum += value;
    local_xor ^= value;
  }
  atomicAdd(sum, local_sum);
  atomicXor(value_xor, local_xor);
}

std::pair<unsigned long long, unsigned long long>
output_digest(uint64_t elements) {
  unsigned long long* device_digest = nullptr;
  cudaMalloc(&device_digest, 2 * sizeof(unsigned long long));
  cudaMemset(device_digest, 0, 2 * sizeof(unsigned long long));
  output_digest_kernel<<<1024, 256>>>(
      reinterpret_cast<uint16_t const*>(fast_D.get()), elements,
      device_digest, device_digest + 1);
  unsigned long long host_digest[2] = {};
  cudaMemcpy(
      host_digest, device_digest, sizeof(host_digest),
      cudaMemcpyDeviceToHost);
  cudaFree(device_digest);
  return {host_digest[0], host_digest[1]};
}


#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

constexpr int max_reuse = 16;
using Buffer = cutlass::DeviceAllocation<uint8_t>;

struct Problem {
  int id, m, n, k, packing;
};

__global__ void fill_bytes(
    uint8_t* bytes, uint64_t size, uint64_t seed, bool scales) {
  for (uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < size; index += uint64_t(gridDim.x) * blockDim.x) {
    uint64_t value = index + seed;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    value ^= value >> 31;
    bytes[index] = scales ? uint8_t(0x28 + (value & 15)) : uint8_t(value);
  }
}

void fill(Buffer& buffer, uint64_t seed, bool scales = false) {
  fill_bytes<<<2048, 256>>>(buffer.get(), buffer.bytes(), seed, scales);
  CUDA_CHECK(cudaGetLastError());
}

struct InputSet {
  Buffer a;
  Buffer sfa;
#if defined(PACKED_INPUT)
  Buffer packed_a;
#endif
};

template<class Config>
int benchmark(Problem problem, int repeats, int seed_index, bool profile) {
  using Operator = typename Config::Gemm;
  using Mainloop = typename Config::CollectiveMainloop;
  using Scale = typename Mainloop::Sm1xxBlkScaledConfig;
  auto [id, m, n, k, packing] = problem;

  cudaDeviceProp device{};
  CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
  auto shape = make_shape(m, n, k, 1);
  auto stride_a = cutlass::make_cute_packed_stride(
      typename Operator::GemmKernel::StrideA{}, {m, k, 1});
  auto stride_b = cutlass::make_cute_packed_stride(
      typename Operator::GemmKernel::StrideB{}, {n, k, 1});
  auto stride_c = cutlass::make_cute_packed_stride(
      typename Operator::GemmKernel::StrideC{}, {m, n, 1});
  auto stride_d = cutlass::make_cute_packed_stride(
      typename Operator::GemmKernel::StrideD{}, {m, n, 1});
  auto layout_sfa = Scale::tile_atom_to_shape_SFA(shape);
  auto layout_sfb = Scale::tile_atom_to_shape_SFB(shape);
  auto seed = 0x72a0000000000000ULL ^ (uint64_t(id) << 24)
      ^ (uint64_t(seed_index) << 32);
  const int input_count = profile ? 1 : max_reuse;
  std::array<InputSet, max_reuse> inputs;
  fast_B.reset(uint64_t(n) * k / 2);
  fast_SFB.reset(size(filter_zeros(layout_sfb)));
  fast_D.reset(uint64_t(m) * n * sizeof(ElementD));
  fill(fast_B, seed ^ 0xb11ce002ULL);
  fill(fast_SFB, seed ^ 0x5fb00004ULL, true);
  for (int index = 0; index < input_count; ++index) {
    inputs[index].a.reset(uint64_t(m) * k / 2);
    inputs[index].sfa.reset(size(filter_zeros(layout_sfa)));
    uint64_t salt = uint64_t(index + 1) * 0x9e3779b97f4a7c15ULL;
    fill(inputs[index].a, seed ^ salt ^ 0xa11ce001ULL);
    fill(inputs[index].sfa, seed ^ salt ^ 0x5fa00003ULL, true);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
#if defined(PACKED_INPUT)
  uint64_t a_bytes = 0, b_bytes = 0;
  if (op1_packed_sizes(packing, m, n, k, &a_bytes, &b_bytes) != 0) {
    throw std::runtime_error("Cannot determine packed buffer sizes");
  }
  fast_packed_A.reset(a_bytes);
  fast_packed_B.reset(b_bytes);
  auto pack = [&](int index, void* a, void* b, bool configure = false) {
    auto& input = inputs[index];
    int error = op1_pack_reuse(
        packing, m, n, k, input.a.get(), fast_B.get(),
        input.sfa.get(), fast_SFB.get(), a, b, stream, configure);
    if (error != 0) throw std::runtime_error("Input packing launch failed");
  };
  pack(0, fast_packed_A.get(), fast_packed_B.get(), true);
  for (int index = 0; index < input_count && !profile; ++index) {
    inputs[index].packed_a.reset(a_bytes);
    pack(index, inputs[index].packed_a.get(), nullptr);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cout << "PACKED_BYTES," << a_bytes << ',' << b_bytes << std::endl;
#endif

  std::array<Operator, max_reuse> gemm;
  std::array<Buffer, max_reuse> workspace;
#if defined(PACKED_INPUT)
  Operator gemm_shared;
  Buffer workspace_shared;
#endif
  auto init = [&](Operator& op, Buffer& work, int index, bool shared_packed) {
    auto& input = inputs[index];
#if defined(PACKED_INPUT)
    auto* ptr_a = shared_packed ? fast_packed_A.get() : input.packed_a.get();
    auto* ptr_b = fast_packed_B.get();
#else
    auto* ptr_a = input.a.get();
    auto* ptr_b = fast_B.get();
#endif
    typename Operator::Arguments args {
      cutlass::gemm::GemmUniversalMode::kGemm, shape,
      {reinterpret_cast<ElementA::DataType*>(ptr_a), stride_a,
       reinterpret_cast<ElementB::DataType*>(ptr_b), stride_b,
       reinterpret_cast<ElementA::ScaleFactorType*>(input.sfa.get()), layout_sfa,
       reinterpret_cast<ElementB::ScaleFactorType*>(fast_SFB.get()), layout_sfb},
      {{}, nullptr, stride_c, reinterpret_cast<ElementD*>(fast_D.get()), stride_d}
    };
    CUTLASS_CHECK(op.can_implement(args));
    work.reset(Operator::get_workspace_size(args));
    CUTLASS_CHECK(op.initialize(args, work.get(), stream));
  };
#if defined(PACKED_INPUT)
  init(gemm_shared, workspace_shared, 0, true);
#endif
  std::cout << "META," << id << ',' << m << ',' << n << ',' << k << ','
            << seed_index << ',' << device.multiProcessorCount << ',' << repeats
            << std::endl;
  for (int index = 0; index < input_count; ++index) {
#if defined(PACKED_INPUT)
    if (!profile) init(gemm[index], workspace[index], index, false);
    pack(index, fast_packed_A.get(), nullptr);
    CUTLASS_CHECK(gemm_shared.run(stream));
#else
    init(gemm[index], workspace[index], index, false);
    CUTLASS_CHECK(gemm[index].run(stream));
#endif
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto digest = output_digest(uint64_t(m) * n);
    CUDA_CHECK(cudaGetLastError());
    std::cout << "DIGEST," << index << ',' << std::hex << digest.first << ','
              << digest.second << std::dec << std::endl;
  }
  if (profile) {
    for (int count = 0; count < 8; ++count) {
#if defined(PACKED_INPUT)
      CUTLASS_CHECK(gemm_shared.run(stream));
#else
      CUTLASS_CHECK(gemm[0].run(stream));
#endif
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0;
  }

  auto capture = [&](auto&& launch) {
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return executable;
  };
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  auto warmup = [&](cudaGraphExec_t graph, int calls) {
    for (int count = 0; count < calls; ++count) {
      CUDA_CHECK(cudaGraphLaunch(graph, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  };
  auto measure = [&](cudaGraphExec_t graph, int reuse, int group, char const* mode) {
    // Individually report group means so launch amortization is reproducible.
    constexpr int launches = 8;
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int count = 0; count < launches; ++count) {
      CUDA_CHECK(cudaGraphLaunch(graph, stream));
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    std::cout << "TIME," << reuse << ',' << group << ',' << mode << ','
              << launches << ',' << std::setprecision(10) << elapsed / launches
              << std::endl;
  };
  for (int reuse : {1, 4, 16}) {
    auto gemm_graph = capture([&] {
      for (int index = 0; index < reuse; ++index) {
        CUTLASS_CHECK(gemm[index].run(stream));
      }
    });
    warmup(gemm_graph, 10);
#if defined(PACKED_INPUT)
    auto chain_graph = capture([&] {
      pack(0, nullptr, fast_packed_B.get());
      for (int index = 0; index < reuse; ++index) {
        pack(index, fast_packed_A.get(), nullptr);
        CUTLASS_CHECK(gemm_shared.run(stream));
      }
    });
    auto pack_graph = capture([&] {
      pack(0, nullptr, fast_packed_B.get());
      for (int index = 0; index < reuse; ++index) {
        pack(index, fast_packed_A.get(), nullptr);
      }
    });
    warmup(chain_graph, 10);
    warmup(pack_graph, 10);
#endif
    for (int group = 0; group < repeats; ++group) {
#if defined(PACKED_INPUT)
      if (group % 2) measure(chain_graph, reuse, group, "online");
      measure(gemm_graph, reuse, group, "gemm_only");
      if (!(group % 2)) measure(chain_graph, reuse, group, "online");
      measure(pack_graph, reuse, group, "pack_only");
#else
      measure(gemm_graph, reuse, group, "baseline");
#endif
    }
    CUDA_CHECK(cudaGraphExecDestroy(gemm_graph));
#if defined(PACKED_INPUT)
    CUDA_CHECK(cudaGraphExecDestroy(chain_graph));
    CUDA_CHECK(cudaGraphExecDestroy(pack_graph));
#endif
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
#endif

int main(int argc, char const** argv) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  cutlass::CommandLine command(argc, argv);
  int id = 0;
  int repeats = 20;
  int seed_index = 0;
  command.get_cmd_line_argument("shape-id", id);
  command.get_cmd_line_argument("repeats", repeats);
  command.get_cmd_line_argument("seed-index", seed_index);
  bool profile = command.check_cmd_line_flag("profile");
  try {
    if (id == 1) return benchmark<Config_00>({1,170100,13824,5120,0}, repeats, seed_index, profile);
    if (id == 7) return benchmark<Config_04>({7,32768,768,4096,1}, repeats, seed_index, profile);
    if (id == 10) return benchmark<Config_05>({10,32768,4096,128,2}, repeats, seed_index, profile);
  } catch (std::exception const& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
#endif
  return 2;
}
