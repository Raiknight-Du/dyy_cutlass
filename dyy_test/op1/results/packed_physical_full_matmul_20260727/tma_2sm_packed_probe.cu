#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

constexpr int kLocalBytes = 18 * 1024;
constexpr int kRowBytes = 128;
constexpr int kRows = kLocalBytes / kRowBytes;
constexpr int kClusterCtas = 2;
constexpr int kThreads = 128;

using SmemLayout = decltype(make_layout(
    make_shape(Int<kRowBytes>{}, Int<kRows>{}),
    make_stride(_1{}, Int<kRowBytes>{})));
using GlobalLayout = decltype(make_layout(
    make_shape(
        Int<kRowBytes>{}, Int<kRows>{}, Int<kClusterCtas>{}),
    make_stride(
        _1{}, Int<kRowBytes>{}, Int<kLocalBytes>{})));

struct SharedStorage {
  alignas(128) uint8_t packed[kLocalBytes];
  alignas(16) uint64_t tma_barrier;
};

template <class TmaAtom>
__global__ void packed_probe(
    uint8_t const* input,
    uint32_t* digest,
    CUTE_GRID_CONSTANT TmaAtom const tma_atom) {
  extern __shared__ char smem_raw[];
  auto& storage = *reinterpret_cast<SharedStorage*>(smem_raw);

  int peer = int(blockIdx.x) & 1;
  bool leader = peer == 0;
  bool elected = (threadIdx.x / 32 == 0) && elect_one_sync();

  Tensor mPacked = tma_atom.get_tma_tensor(GlobalLayout{}.shape());
  Tensor gPacked = mPacked(_, _, peer);
  Tensor sPacked = make_tensor(
      make_smem_ptr(storage.packed), SmemLayout{});

  auto [tG, tS] = tma_partition(
      tma_atom, Int<0>{}, Layout<_1>{},
      group_modes<0, 2>(sPacked),
      group_modes<0, 2>(gPacked));

  if (elected && leader) {
    storage.tma_barrier = 0;
    initialize_barrier(storage.tma_barrier, 1);
    set_barrier_transaction_bytes(
        storage.tma_barrier, kClusterCtas * kLocalBytes);
  }
  cluster_sync();

  if (elected) {
    copy(tma_atom.with(storage.tma_barrier), tG, tS);
  }
  if (leader && threadIdx.x == 0) {
    wait_barrier(storage.tma_barrier, 0);
  }
  cluster_sync();

  if (threadIdx.x == 0) {
    uint32_t value = 2166136261u;
    for (int i = 0; i < kLocalBytes; i += 257) {
      value = (value ^ storage.packed[i]) * 16777619u;
    }
    digest[peer] = value;
  }
}

#endif

int main() {
#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  return 1;
#else
  std::vector<uint8_t> host(kClusterCtas * kLocalBytes);
  for (int peer = 0; peer < kClusterCtas; ++peer) {
    for (int i = 0; i < kLocalBytes; ++i) {
      host[peer * kLocalBytes + i] =
          uint8_t((peer * 73 + i * 29 + 17) & 0xff);
    }
  }

  uint8_t* input = nullptr;
  uint32_t* digest = nullptr;
  cudaMalloc(&input, host.size());
  cudaMalloc(&digest, kClusterCtas * sizeof(uint32_t));
  cudaMemcpy(input, host.data(), host.size(), cudaMemcpyHostToDevice);

  Tensor gPacked = make_tensor(
      make_gmem_ptr(input), GlobalLayout{});
  auto tma_atom = make_tma_atom<uint8_t>(
      SM100_TMA_2SM_LOAD{},
      gPacked,
      SmemLayout{},
      make_shape(Int<kRowBytes>{}, Int<kRows>{}),
      Int<1>{});

  using Kernel = decltype(packed_probe<decltype(tma_atom)>);
  auto* kernel = &packed_probe<decltype(tma_atom)>;
  cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(SharedStorage));

  cutlass::ClusterLaunchParams launch{
      dim3(kClusterCtas, 1, 1),
      dim3(kThreads, 1, 1),
      dim3(kClusterCtas, 1, 1),
      int(sizeof(SharedStorage))};
  auto status = cutlass::launch_kernel_on_cluster(
      launch, reinterpret_cast<void const*>(kernel),
      input, digest, tma_atom);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "cluster launch failed\n");
    return 2;
  }
  auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
    return 3;
  }

  uint32_t actual[kClusterCtas] = {};
  cudaMemcpy(actual, digest, sizeof(actual), cudaMemcpyDeviceToHost);
  for (int peer = 0; peer < kClusterCtas; ++peer) {
    uint32_t expected = 2166136261u;
    for (int i = 0; i < kLocalBytes; i += 257) {
      expected =
          (expected ^ host[peer * kLocalBytes + i]) * 16777619u;
    }
    std::printf(
        "peer=%d actual=%08x expected=%08x %s\n",
        peer, actual[peer], expected,
        actual[peer] == expected ? "PASS" : "FAIL");
    if (actual[peer] != expected) {
      return 4;
    }
  }
  return 0;
#endif
}
