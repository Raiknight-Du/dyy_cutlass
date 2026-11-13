// Controlled SM100 TMA experiment:
//   separate: A, B, SFA, SFB use four tensor maps and four TMA loads/stage.
//   packed:   A+SFA and B+SFB use two packed tensor maps and two TMA loads/stage.
//
// Both variants move exactly 36 KiB/stage with the same grid, K rounds,
// five-stage pipeline, barrier accounting, working-set size, and random data.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

constexpr int kGridBlocks = 35910;
constexpr int kRounds = 40;
constexpr int kStages = 5;
constexpr int kThreads = 128;
constexpr int kTileCount = 1792;

constexpr int kRowBytes = 128;
constexpr int kMainRows = 128;
constexpr int kSfRows = 16;
constexpr int kPackedRows = kMainRows + kSfRows;

constexpr int kMainBytes = kMainRows * kRowBytes;
constexpr int kSfBytes = kSfRows * kRowBytes;
constexpr int kPackedBytes = kPackedRows * kRowBytes;
constexpr int kStageBytes = 2 * kPackedBytes;

using SmemMainLayout = decltype(
    make_layout(make_shape(Int<kMainRows>{}, Int<kRowBytes>{}), GenRowMajor{}));
using SmemSfLayout = decltype(
    make_layout(make_shape(Int<kSfRows>{}, Int<kRowBytes>{}), GenRowMajor{}));
using SmemPackedLayout = decltype(
    make_layout(make_shape(Int<kPackedRows>{}, Int<kRowBytes>{}), GenRowMajor{}));

struct SeparateStorage {
  alignas(128) uint8_t A[kStages][kMainBytes];
  alignas(128) uint8_t B[kStages][kMainBytes];
  alignas(128) uint8_t SFA[kStages][kSfBytes];
  alignas(128) uint8_t SFB[kStages][kSfBytes];
  alignas(16) uint64_t barrier[kStages];
};

struct PackedStorage {
  alignas(128) uint8_t ASFA[kStages][kPackedBytes];
  alignas(128) uint8_t BSFB[kStages][kPackedBytes];
  alignas(16) uint64_t barrier[kStages];
};

__device__ __forceinline__ uint8_t random_byte(
    uint32_t tile, uint32_t offset, uint32_t stream) {
  uint32_t x = tile * 0x9e3779b9u + offset * 0x85ebca6bu +
               stream * 0xc2b2ae35u + 0x27d4eb2fu;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return static_cast<uint8_t>(x);
}

__global__ void initialize_separate(
    uint8_t* A, uint8_t* B, uint8_t* SFA, uint8_t* SFB) {
  uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  uint64_t main_elements = uint64_t(kTileCount) * kMainBytes;
  uint64_t sf_elements = uint64_t(kTileCount) * kSfBytes;

  for (uint64_t i = index; i < main_elements; i += stride) {
    uint32_t tile = uint32_t(i / kMainBytes);
    uint32_t offset = uint32_t(i % kMainBytes);
    A[i] = random_byte(tile, offset, 0);
    B[i] = random_byte(tile, offset, 2);
  }
  for (uint64_t i = index; i < sf_elements; i += stride) {
    uint32_t tile = uint32_t(i / kSfBytes);
    uint32_t offset = uint32_t(i % kSfBytes);
    SFA[i] = random_byte(tile, offset, 1);
    SFB[i] = random_byte(tile, offset, 3);
  }
}

__global__ void initialize_packed(uint8_t* ASFA, uint8_t* BSFB) {
  uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  uint64_t elements = uint64_t(kTileCount) * kPackedBytes;

  for (uint64_t i = index; i < elements; i += stride) {
    uint32_t tile = uint32_t(i / kPackedBytes);
    uint32_t packed_offset = uint32_t(i % kPackedBytes);
    bool is_sf = packed_offset >= kMainBytes;
    uint32_t offset = is_sf ? packed_offset - kMainBytes : packed_offset;
    ASFA[i] = random_byte(tile, offset, is_sf ? 1 : 0);
    BSFB[i] = random_byte(tile, offset, is_sf ? 3 : 2);
  }
}

__device__ __forceinline__ uint32_t digest_byte(uint32_t digest, uint8_t value) {
  return (digest ^ uint32_t(value)) * 16777619u;
}

template <
    class TmaA, class TmaB, class TmaSFA, class TmaSFB,
    class LayoutA, class LayoutB, class LayoutSFA, class LayoutSFB,
    class TileA, class TileB, class TileSFA, class TileSFB>
__global__ void separate_tma_kernel(
    uint32_t* output,
    CUTE_GRID_CONSTANT TmaA const tma_A,
    CUTE_GRID_CONSTANT TmaB const tma_B,
    CUTE_GRID_CONSTANT TmaSFA const tma_SFA,
    CUTE_GRID_CONSTANT TmaSFB const tma_SFB,
    LayoutA layout_A, LayoutB layout_B,
    LayoutSFA layout_SFA, LayoutSFB layout_SFB,
    TileA tile_A, TileB tile_B, TileSFA tile_SFA, TileSFB tile_SFB) {
  extern __shared__ char smem_raw[];
  auto& storage = *reinterpret_cast<SeparateStorage*>(smem_raw);

  Tensor mA = tma_A.get_tma_tensor(shape(layout_A));
  Tensor mB = tma_B.get_tma_tensor(shape(layout_B));
  Tensor mSFA = tma_SFA.get_tma_tensor(shape(layout_SFA));
  Tensor mSFB = tma_SFB.get_tma_tensor(shape(layout_SFB));
  Tensor gA = flat_divide(mA, tile_A);
  Tensor gB = flat_divide(mB, tile_B);
  Tensor gSFA = flat_divide(mSFA, tile_SFA);
  Tensor gSFB = flat_divide(mSFB, tile_SFB);

  auto cta_A = tma_A.get_slice(Int<0>{});
  auto cta_B = tma_B.get_slice(Int<0>{});
  auto cta_SFA = tma_SFA.get_slice(Int<0>{});
  auto cta_SFB = tma_SFB.get_slice(Int<0>{});
  auto tAgA_x = cta_A.partition_S(gA);
  auto tBgB_x = cta_B.partition_S(gB);
  auto tSFAg_x = cta_SFA.partition_S(gSFA);
  auto tSFBg_x = cta_SFB.partition_S(gSFB);
  auto tAgA = group_modes<1, rank(tAgA_x)>(tAgA_x);
  auto tBgB = group_modes<1, rank(tBgB_x)>(tBgB_x);
  auto tSFAg = group_modes<1, rank(tSFAg_x)>(tSFAg_x);
  auto tSFBg = group_modes<1, rank(tSFBg_x)>(tSFBg_x);

  if (threadIdx.x == 0) {
    int phase[kStages] = {};
    #pragma unroll
    for (int stage = 0; stage < kStages; ++stage) {
      storage.barrier[stage] = 0;
      cute::initialize_barrier(storage.barrier[stage], 1);
    }

    auto issue_stage = [&](int round, int stage) {
      int tile = (int(blockIdx.x) * kRounds + round) % kTileCount;
      cute::set_barrier_transaction_bytes(storage.barrier[stage], kStageBytes);

      auto sA = make_tensor(make_smem_ptr(storage.A[stage]), SmemMainLayout{});
      auto sB = make_tensor(make_smem_ptr(storage.B[stage]), SmemMainLayout{});
      auto sSFA = make_tensor(make_smem_ptr(storage.SFA[stage]), SmemSfLayout{});
      auto sSFB = make_tensor(make_smem_ptr(storage.SFB[stage]), SmemSfLayout{});
      auto tAsA_x = cta_A.partition_D(sA);
      auto tBsB_x = cta_B.partition_D(sB);
      auto tSFAs_x = cta_SFA.partition_D(sSFA);
      auto tSFBs_x = cta_SFB.partition_D(sSFB);
      auto tAsA = group_modes<1, rank(tAsA_x)>(tAsA_x);
      auto tBsB = group_modes<1, rank(tBsB_x)>(tBsB_x);
      auto tSFAs = group_modes<1, rank(tSFAs_x)>(tSFAs_x);
      auto tSFBs = group_modes<1, rank(tSFBs_x)>(tSFBs_x);

      copy(tma_A.with(storage.barrier[stage]), tAgA(_, tile), tAsA(_, 0));
      copy(tma_B.with(storage.barrier[stage]), tBgB(_, tile), tBsB(_, 0));
      copy(tma_SFA.with(storage.barrier[stage]), tSFAg(_, tile), tSFAs(_, 0));
      copy(tma_SFB.with(storage.barrier[stage]), tSFBg(_, tile), tSFBs(_, 0));
    };

    #pragma unroll
    for (int round = 0; round < kStages; ++round) {
      issue_stage(round, round);
    }
    #pragma unroll 1
    for (int round = 0; round < kRounds; ++round) {
      int stage = round % kStages;
      cute::wait_barrier(storage.barrier[stage], phase[stage]);
      phase[stage] ^= 1;
      if (round + kStages < kRounds) {
        issue_stage(round + kStages, stage);
      }
    }

    uint32_t digest = 2166136261u;
    #pragma unroll
    for (int stage = 0; stage < kStages; ++stage) {
      digest = digest_byte(digest, storage.A[stage][0]);
      digest = digest_byte(digest, storage.A[stage][kMainBytes - 1]);
      digest = digest_byte(digest, storage.SFA[stage][0]);
      digest = digest_byte(digest, storage.SFA[stage][kSfBytes - 1]);
      digest = digest_byte(digest, storage.B[stage][0]);
      digest = digest_byte(digest, storage.B[stage][kMainBytes - 1]);
      digest = digest_byte(digest, storage.SFB[stage][0]);
      digest = digest_byte(digest, storage.SFB[stage][kSfBytes - 1]);
    }
    output[blockIdx.x] = digest;
  }
}

template <
    class TmaASFA, class TmaBSFB,
    class LayoutASFA, class LayoutBSFB,
    class TileASFA, class TileBSFB>
__global__ void packed_tma_kernel(
    uint32_t* output,
    CUTE_GRID_CONSTANT TmaASFA const tma_ASFA,
    CUTE_GRID_CONSTANT TmaBSFB const tma_BSFB,
    LayoutASFA layout_ASFA, LayoutBSFB layout_BSFB,
    TileASFA tile_ASFA, TileBSFB tile_BSFB) {
  extern __shared__ char smem_raw[];
  auto& storage = *reinterpret_cast<PackedStorage*>(smem_raw);

  Tensor mASFA = tma_ASFA.get_tma_tensor(shape(layout_ASFA));
  Tensor mBSFB = tma_BSFB.get_tma_tensor(shape(layout_BSFB));
  Tensor gASFA = flat_divide(mASFA, tile_ASFA);
  Tensor gBSFB = flat_divide(mBSFB, tile_BSFB);

  auto cta_ASFA = tma_ASFA.get_slice(Int<0>{});
  auto cta_BSFB = tma_BSFB.get_slice(Int<0>{});
  auto tASFAg_x = cta_ASFA.partition_S(gASFA);
  auto tBSFBg_x = cta_BSFB.partition_S(gBSFB);
  auto tASFAg = group_modes<1, rank(tASFAg_x)>(tASFAg_x);
  auto tBSFBg = group_modes<1, rank(tBSFBg_x)>(tBSFBg_x);

  if (threadIdx.x == 0) {
    int phase[kStages] = {};
    #pragma unroll
    for (int stage = 0; stage < kStages; ++stage) {
      storage.barrier[stage] = 0;
      cute::initialize_barrier(storage.barrier[stage], 1);
    }

    auto issue_stage = [&](int round, int stage) {
      int tile = (int(blockIdx.x) * kRounds + round) % kTileCount;
      cute::set_barrier_transaction_bytes(storage.barrier[stage], kStageBytes);

      auto sASFA = make_tensor(
          make_smem_ptr(storage.ASFA[stage]), SmemPackedLayout{});
      auto sBSFB = make_tensor(
          make_smem_ptr(storage.BSFB[stage]), SmemPackedLayout{});
      auto tASFAs_x = cta_ASFA.partition_D(sASFA);
      auto tBSFBs_x = cta_BSFB.partition_D(sBSFB);
      auto tASFAs = group_modes<1, rank(tASFAs_x)>(tASFAs_x);
      auto tBSFBs = group_modes<1, rank(tBSFBs_x)>(tBSFBs_x);

      copy(tma_ASFA.with(storage.barrier[stage]),
           tASFAg(_, tile), tASFAs(_, 0));
      copy(tma_BSFB.with(storage.barrier[stage]),
           tBSFBg(_, tile), tBSFBs(_, 0));
    };

    #pragma unroll
    for (int round = 0; round < kStages; ++round) {
      issue_stage(round, round);
    }
    #pragma unroll 1
    for (int round = 0; round < kRounds; ++round) {
      int stage = round % kStages;
      cute::wait_barrier(storage.barrier[stage], phase[stage]);
      phase[stage] ^= 1;
      if (round + kStages < kRounds) {
        issue_stage(round + kStages, stage);
      }
    }

    uint32_t digest = 2166136261u;
    #pragma unroll
    for (int stage = 0; stage < kStages; ++stage) {
      digest = digest_byte(digest, storage.ASFA[stage][0]);
      digest = digest_byte(digest, storage.ASFA[stage][kMainBytes - 1]);
      digest = digest_byte(digest, storage.ASFA[stage][kMainBytes]);
      digest = digest_byte(digest, storage.ASFA[stage][kPackedBytes - 1]);
      digest = digest_byte(digest, storage.BSFB[stage][0]);
      digest = digest_byte(digest, storage.BSFB[stage][kMainBytes - 1]);
      digest = digest_byte(digest, storage.BSFB[stage][kMainBytes]);
      digest = digest_byte(digest, storage.BSFB[stage][kPackedBytes - 1]);
    }
    output[blockIdx.x] = digest;
  }
}

template <class Launch>
float time_kernel(Launch const& launch, int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < iterations; ++i) {
    launch();
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsed_ms / iterations;
}

#endif

int main(int argc, char** argv) {
#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  return 1;
#else
  std::string variant = "separate";
  int warmup = 5;
  int iterations = 10;
  bool verify = true;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.rfind("--variant=", 0) == 0) {
      variant = arg.substr(10);
    } else if (arg.rfind("--warmup=", 0) == 0) {
      warmup = std::atoi(arg.c_str() + 9);
    } else if (arg.rfind("--iterations=", 0) == 0) {
      iterations = std::atoi(arg.c_str() + 13);
    } else if (arg == "--no-verify") {
      verify = false;
    }
  }
  if (variant != "separate" && variant != "packed") {
    std::cerr << "variant must be separate or packed\n";
    return 2;
  }

  int max_smem = 0;
  cudaDeviceGetAttribute(
      &max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  if (int(sizeof(SeparateStorage)) > max_smem ||
      int(sizeof(PackedStorage)) > max_smem) {
    std::cerr << "Experiment requires more dynamic SMEM than the device allows\n";
    return 3;
  }

  thrust::device_vector<uint8_t> A(uint64_t(kTileCount) * kMainBytes);
  thrust::device_vector<uint8_t> B(uint64_t(kTileCount) * kMainBytes);
  thrust::device_vector<uint8_t> SFA(uint64_t(kTileCount) * kSfBytes);
  thrust::device_vector<uint8_t> SFB(uint64_t(kTileCount) * kSfBytes);
  thrust::device_vector<uint8_t> ASFA(uint64_t(kTileCount) * kPackedBytes);
  thrust::device_vector<uint8_t> BSFB(uint64_t(kTileCount) * kPackedBytes);
  thrust::device_vector<uint32_t> separate_output(kGridBlocks);
  thrust::device_vector<uint32_t> packed_output(kGridBlocks);

  auto ptr_A = thrust::raw_pointer_cast(A.data());
  auto ptr_B = thrust::raw_pointer_cast(B.data());
  auto ptr_SFA = thrust::raw_pointer_cast(SFA.data());
  auto ptr_SFB = thrust::raw_pointer_cast(SFB.data());
  auto ptr_ASFA = thrust::raw_pointer_cast(ASFA.data());
  auto ptr_BSFB = thrust::raw_pointer_cast(BSFB.data());
  auto ptr_separate_output =
      thrust::raw_pointer_cast(separate_output.data());
  auto ptr_packed_output = thrust::raw_pointer_cast(packed_output.data());

  initialize_separate<<<1024, 256>>>(ptr_A, ptr_B, ptr_SFA, ptr_SFB);
  initialize_packed<<<1024, 256>>>(ptr_ASFA, ptr_BSFB);
  cudaDeviceSynchronize();

  auto layout_A = make_layout(
      make_shape(kTileCount * kMainRows, kRowBytes), GenRowMajor{});
  auto layout_B = layout_A;
  auto layout_SFA = make_layout(
      make_shape(kTileCount * kSfRows, kRowBytes), GenRowMajor{});
  auto layout_SFB = layout_SFA;
  auto layout_ASFA = make_layout(
      make_shape(kTileCount * kPackedRows, kRowBytes), GenRowMajor{});
  auto layout_BSFB = layout_ASFA;

  auto smem_main = SmemMainLayout{};
  auto smem_sf = SmemSfLayout{};
  auto smem_packed = SmemPackedLayout{};
  auto tile_main = shape(smem_main);
  auto tile_sf = shape(smem_sf);
  auto tile_packed = shape(smem_packed);

  auto gA = make_tensor(make_gmem_ptr(ptr_A), layout_A);
  auto gB = make_tensor(make_gmem_ptr(ptr_B), layout_B);
  auto gSFA = make_tensor(make_gmem_ptr(ptr_SFA), layout_SFA);
  auto gSFB = make_tensor(make_gmem_ptr(ptr_SFB), layout_SFB);
  auto gASFA = make_tensor(make_gmem_ptr(ptr_ASFA), layout_ASFA);
  auto gBSFB = make_tensor(make_gmem_ptr(ptr_BSFB), layout_BSFB);

  auto tma_A = make_tma_copy(
      SM90_TMA_LOAD{}, gA, smem_main, tile_main, Int<1>{});
  auto tma_B = make_tma_copy(
      SM90_TMA_LOAD{}, gB, smem_main, tile_main, Int<1>{});
  auto tma_SFA = make_tma_copy(
      SM90_TMA_LOAD{}, gSFA, smem_sf, tile_sf, Int<1>{});
  auto tma_SFB = make_tma_copy(
      SM90_TMA_LOAD{}, gSFB, smem_sf, tile_sf, Int<1>{});
  auto tma_ASFA = make_tma_copy(
      SM90_TMA_LOAD{}, gASFA, smem_packed, tile_packed, Int<1>{});
  auto tma_BSFB = make_tma_copy(
      SM90_TMA_LOAD{}, gBSFB, smem_packed, tile_packed, Int<1>{});

  using SeparateKernel = decltype(separate_tma_kernel<
      decltype(tma_A), decltype(tma_B), decltype(tma_SFA), decltype(tma_SFB),
      decltype(layout_A), decltype(layout_B),
      decltype(layout_SFA), decltype(layout_SFB),
      decltype(tile_main), decltype(tile_main),
      decltype(tile_sf), decltype(tile_sf)>);
  using PackedKernel = decltype(packed_tma_kernel<
      decltype(tma_ASFA), decltype(tma_BSFB),
      decltype(layout_ASFA), decltype(layout_BSFB),
      decltype(tile_packed), decltype(tile_packed)>);
  (void)sizeof(SeparateKernel*);
  (void)sizeof(PackedKernel*);

  auto launch_separate = [&] {
    separate_tma_kernel<<<kGridBlocks, kThreads, sizeof(SeparateStorage)>>>(
        ptr_separate_output,
        tma_A, tma_B, tma_SFA, tma_SFB,
        layout_A, layout_B, layout_SFA, layout_SFB,
        tile_main, tile_main, tile_sf, tile_sf);
  };
  auto launch_packed = [&] {
    packed_tma_kernel<<<kGridBlocks, kThreads, sizeof(PackedStorage)>>>(
        ptr_packed_output,
        tma_ASFA, tma_BSFB,
        layout_ASFA, layout_BSFB,
        tile_packed, tile_packed);
  };

  cudaFuncSetAttribute(
      separate_tma_kernel<
          decltype(tma_A), decltype(tma_B),
          decltype(tma_SFA), decltype(tma_SFB),
          decltype(layout_A), decltype(layout_B),
          decltype(layout_SFA), decltype(layout_SFB),
          decltype(tile_main), decltype(tile_main),
          decltype(tile_sf), decltype(tile_sf)>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(SeparateStorage));
  cudaFuncSetAttribute(
      packed_tma_kernel<
          decltype(tma_ASFA), decltype(tma_BSFB),
          decltype(layout_ASFA), decltype(layout_BSFB),
          decltype(tile_packed), decltype(tile_packed)>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(PackedStorage));

  if (verify) {
    launch_separate();
    launch_packed();
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
      std::cerr << cudaGetErrorString(error) << "\n";
      return 4;
    }
    thrust::host_vector<uint32_t> h_separate = separate_output;
    thrust::host_vector<uint32_t> h_packed = packed_output;
    if (!std::equal(
            h_separate.begin(), h_separate.end(), h_packed.begin())) {
      auto mismatch = std::mismatch(
          h_separate.begin(), h_separate.end(), h_packed.begin());
      std::cerr << "Verification failed at block "
                << std::distance(h_separate.begin(), mismatch.first) << "\n";
      return 5;
    }
    std::cout << "verification,passed\n";
  }

  float mean_ms = variant == "separate"
      ? time_kernel(launch_separate, warmup, iterations)
      : time_kernel(launch_packed, warmup, iterations);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << cudaGetErrorString(error) << "\n";
    return 6;
  }

  uint64_t logical_bytes =
      uint64_t(kGridBlocks) * kRounds * kStageBytes;
  double effective_tib_s =
      (double(logical_bytes) / double(uint64_t(1) << 40)) /
      (double(mean_ms) / 1000.0);
  int tma_per_stage = variant == "separate" ? 4 : 2;

  std::cout << "variant," << variant << "\n";
  std::cout << "grid_blocks," << kGridBlocks << "\n";
  std::cout << "rounds," << kRounds << "\n";
  std::cout << "stages," << kStages << "\n";
  std::cout << "threads_per_cta," << kThreads << "\n";
  std::cout << "payload_per_stage_kib," << kStageBytes / 1024.0 << "\n";
  std::cout << "tma_per_stage," << tma_per_stage << "\n";
  std::cout << "expected_tma_loads,"
            << uint64_t(kGridBlocks) * kRounds * tma_per_stage << "\n";
  std::cout << "working_set_mib,"
            << double(uint64_t(kTileCount) * kStageBytes) / (1 << 20) << "\n";
  std::cout << "smem_per_cta_kib,"
            << (variant == "separate"
                    ? sizeof(SeparateStorage)
                    : sizeof(PackedStorage)) /
                   1024.0
            << "\n";
  std::cout << "logical_bytes," << logical_bytes << "\n";
  std::cout << "mean_ms," << mean_ms << "\n";
  std::cout << "effective_tib_s," << effective_tib_s << "\n";
  return 0;
#endif
}

