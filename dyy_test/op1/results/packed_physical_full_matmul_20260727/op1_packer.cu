#include <cstdint>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementAccumulator = float;
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;

template <int TileM, int TileN, int TileK>
struct PackConfig {
  static constexpr int AtomCtas = TileM == 256 ? 2 : 1;
  using TileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
  using ClusterShape = Shape<Int<AtomCtas>, _1, _1>;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCount<1>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
};

template <class Mainloop>
struct PackStorage {
  typename Mainloop::TensorStorage tensors;
  alignas(16) uint64_t barrier;
};

template <class Mainloop>
struct PackTraits {
  static constexpr int AtomCtas =
      size(typename Mainloop::AtomThrShapeMNK{});
  static constexpr int AElements =
      cosize(take<0,3>(typename Mainloop::SmemLayoutA{}));
  static constexpr int BElements =
      cosize(take<0,3>(typename Mainloop::SmemLayoutB{}));
  static constexpr int SFAElements =
      cosize(take<0,3>(typename Mainloop::SmemLayoutSFA{}));
  static constexpr int SFBElements =
      cosize(take<0,3>(typename Mainloop::SmemLayoutSFB{}));
  static constexpr int ABytes = cutlass::bits_to_bytes(
      AElements * cute::sizeof_bits_v<typename Mainloop::ElementAMma>);
  static constexpr int BBytes = cutlass::bits_to_bytes(
      BElements * cute::sizeof_bits_v<typename Mainloop::ElementBMma>);
  static constexpr int SFABytes = cutlass::bits_to_bytes(
      SFAElements * cute::sizeof_bits_v<typename Mainloop::ElementSF>);
  static constexpr int SFBBytes = cutlass::bits_to_bytes(
      SFBElements * cute::sizeof_bits_v<typename Mainloop::ElementSF>);
  static constexpr int PackedABytes = ABytes + SFABytes;
  static constexpr int PackedBBytes = BBytes + SFBBytes;
};

template <bool PackA, class Mainloop>
__global__ void pack_operand_kernel(
    CUTE_GRID_CONSTANT typename Mainloop::Params const params,
    int M, int N, int K,
    uint8_t* packed_output) {
  extern __shared__ char smem_raw[];
  auto& storage =
      *reinterpret_cast<PackStorage<Mainloop>*>(smem_raw);
  using Traits = PackTraits<Mainloop>;
  using ClusterShape = typename Mainloop::DispatchPolicy::ClusterShape;

  int peer = int(blockIdx.x) % Traits::AtomCtas;
  int logical_tile = int(blockIdx.x) / Traits::AtomCtas;
  bool leader = peer == 0;
  bool elected = (threadIdx.x / 32 == 0) && elect_one_sync();

  Mainloop mainloop(params, ClusterShape{}, uint32_t(peer));
  auto problem = make_shape(M, N, K, 1);
  auto inputs = mainloop.load_init(problem, storage.tensors);
  auto [
      k_tiles_unused,
      tAgA_mkl, tBgB_nkl, tAsA, tBsB,
      tAgSFA_mkl, tBgSFB_nkl, tAsSFA, tBsSFB,
      mcast_a, mcast_b, mcast_sfa, mcast_sfb] = inputs;

  auto tAgA = tAgA_mkl(_, logical_tile, _, 0);
  auto tAgSFA = tAgSFA_mkl(_, logical_tile, _, 0);
  auto tBgB = tBgB_nkl(_, logical_tile, _, 0);
  auto tBgSFB = tBgSFB_nkl(_, logical_tile, _, 0);

  if (elected && leader) {
    storage.barrier = 0;
    initialize_barrier(storage.barrier, 1);
  }
  cluster_sync();

  int phase = 0;
  int k_tiles = int(ceil_div(K, size<2>(typename Mainloop::TileShape{})));
  for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
    if (elected && leader) {
      constexpr int transaction_bytes = Traits::AtomCtas * (
          PackA ? Traits::PackedABytes : Traits::PackedBBytes);
      set_barrier_transaction_bytes(storage.barrier, transaction_bytes);
    }
    if (elected) {
      if constexpr (PackA) {
        copy(
            params.tma_load_a.with(storage.barrier, mcast_a),
            tAgA(_, k_tile), tAsA(_, 0));
        copy(
            params.tma_load_sfa.with(storage.barrier, mcast_sfa),
            tAgSFA(_, k_tile), tAsSFA(_, 0));
      } else {
        copy(
            params.tma_load_b.with(storage.barrier, mcast_b),
            tBgB(_, k_tile), tBsB(_, 0));
        copy(
            params.tma_load_sfb.with(storage.barrier, mcast_sfb),
            tBgSFB(_, k_tile), tBsSFB(_, 0));
      }
    }
    if (leader && threadIdx.x == 0) {
      wait_barrier(storage.barrier, phase);
      phase ^= 1;
    }
    cluster_sync();

    if constexpr (PackA) {
      auto a = recast_ptr<uint8_t>(storage.tensors.smem_A.begin());
      auto sfa = recast_ptr<uint8_t>(storage.tensors.smem_SFA.begin());
      uint64_t tile = (
          (uint64_t(logical_tile) * k_tiles + k_tile) *
          Traits::AtomCtas + peer);
      uint8_t* destination =
          packed_output + tile * Traits::PackedABytes;
      for (int i = int(threadIdx.x); i < Traits::ABytes;
           i += int(blockDim.x)) {
        destination[i] = a[i];
      }
      for (int i = int(threadIdx.x); i < Traits::SFABytes;
           i += int(blockDim.x)) {
        destination[Traits::ABytes + i] = sfa[i];
      }
    } else {
      auto b = recast_ptr<uint8_t>(storage.tensors.smem_B.begin());
      auto sfb = recast_ptr<uint8_t>(storage.tensors.smem_SFB.begin());
      uint64_t tile = (
          (uint64_t(logical_tile) * k_tiles + k_tile) *
          Traits::AtomCtas + peer);
      uint8_t* destination =
          packed_output + tile * Traits::PackedBBytes;
      for (int i = int(threadIdx.x); i < Traits::BBytes;
           i += int(blockDim.x)) {
        destination[i] = b[i];
      }
      for (int i = int(threadIdx.x); i < Traits::SFBBytes;
           i += int(blockDim.x)) {
        destination[Traits::BBytes + i] = sfb[i];
      }
    }
    cluster_sync();
  }
}

template <class Config>
int packed_sizes_impl(
    int M, int N, int K, uint64_t* bytes_a, uint64_t* bytes_b) {
  using Mainloop = typename Config::Mainloop;
  using Traits = PackTraits<Mainloop>;
  int m_tiles = int(ceil_div(M, size<0>(typename Config::TileShape{})));
  int n_tiles = int(ceil_div(N, size<1>(typename Config::TileShape{})));
  int k_tiles = int(ceil_div(K, size<2>(typename Config::TileShape{})));
  *bytes_a = uint64_t(m_tiles) * k_tiles * Traits::AtomCtas *
      Traits::PackedABytes;
  *bytes_b = uint64_t(n_tiles) * k_tiles * Traits::AtomCtas *
      Traits::PackedBBytes;
  return 0;
}

template <class Config>
int pack_impl(
    int M, int N, int K,
    void const* A, void const* B, void const* SFA, void const* SFB,
    void* packed_a, void* packed_b) {
  using Mainloop = typename Config::Mainloop;
  using Traits = PackTraits<Mainloop>;
  using StrideA = typename Mainloop::StrideA;
  using StrideB = typename Mainloop::StrideB;
  using ScaleConfig = typename Mainloop::Sm1xxBlkScaledConfig;
  auto stride_a = cutlass::make_cute_packed_stride(
      StrideA{}, make_shape(M, K, 1));
  auto stride_b = cutlass::make_cute_packed_stride(
      StrideB{}, make_shape(N, K, 1));
  auto problem = make_shape(M, N, K, 1);
  auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(problem);
  auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(problem);
  typename Mainloop::Arguments arguments{
      reinterpret_cast<typename Mainloop::ArrayElementA const*>(A),
      stride_a,
      reinterpret_cast<typename Mainloop::ArrayElementB const*>(B),
      stride_b,
      reinterpret_cast<typename Mainloop::ElementSF const*>(SFA),
      layout_sfa,
      reinterpret_cast<typename Mainloop::ElementSF const*>(SFB),
      layout_sfb};
  auto params = Mainloop::to_underlying_arguments(
      problem, arguments, nullptr, cutlass::KernelHardwareInfo{});

  int m_tiles = int(ceil_div(M, size<0>(typename Config::TileShape{})));
  int n_tiles = int(ceil_div(N, size<1>(typename Config::TileShape{})));
  dim3 cluster(Traits::AtomCtas, 1, 1);
  int smem_bytes = sizeof(PackStorage<Mainloop>);

  auto* pack_a_kernel = &pack_operand_kernel<true, Mainloop>;
  auto* pack_b_kernel = &pack_operand_kernel<false, Mainloop>;
  cudaFuncSetAttribute(
      pack_a_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_bytes);
  cudaFuncSetAttribute(
      pack_b_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_bytes);

  cutlass::ClusterLaunchParams launch_a{
      dim3(m_tiles * Traits::AtomCtas, 1, 1),
      dim3(128, 1, 1), cluster, smem_bytes};
  cutlass::ClusterLaunchParams launch_b{
      dim3(n_tiles * Traits::AtomCtas, 1, 1),
      dim3(128, 1, 1), cluster, smem_bytes};
  auto status_a = cutlass::launch_kernel_on_cluster(
      launch_a, reinterpret_cast<void const*>(pack_a_kernel),
      params, M, N, K, reinterpret_cast<uint8_t*>(packed_a));
  auto status_b = cutlass::launch_kernel_on_cluster(
      launch_b, reinterpret_cast<void const*>(pack_b_kernel),
      params, M, N, K, reinterpret_cast<uint8_t*>(packed_b));
  if (status_a != cutlass::Status::kSuccess ||
      status_b != cutlass::Status::kSuccess) {
    return 1;
  }
  return int(cudaGetLastError());
}

template <class Function>
int dispatch_config(int config, Function&& function) {
  switch (config) {
    case 0: return function(PackConfig<256, 256, 256>{});
    case 1: return function(PackConfig<256, 192, 256>{});
    case 2: return function(PackConfig<128, 256, 128>{});
    case 3: return function(PackConfig<128, 192, 128>{});
    default: return -1;
  }
}

#endif

extern "C" int op1_packed_sizes(
    int config, int M, int N, int K,
    uint64_t* bytes_a, uint64_t* bytes_b) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  return dispatch_config(config, [&](auto selected) {
    return packed_sizes_impl<decltype(selected)>(
        M, N, K, bytes_a, bytes_b);
  });
#else
  return -1;
#endif
}

extern "C" int op1_pack_inputs(
    int config, int M, int N, int K,
    void const* A, void const* B, void const* SFA, void const* SFB,
    void* packed_a, void* packed_b) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  return dispatch_config(config, [&](auto selected) {
    return pack_impl<decltype(selected)>(
        M, N, K, A, B, SFA, SFB, packed_a, packed_b);
  });
#else
  return -1;
#endif
}
