#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace {
constexpr int kWidth = 128;
constexpr int kMainRows = 128;
constexpr int kScaleRows = 16;
constexpr int kRows = kMainRows + kScaleRows;
constexpr int kBytes = kWidth * kRows;
constexpr int kMainBytes = kWidth * kMainRows;
constexpr int kStages = 4;
constexpr int kRounds = 32;
constexpr int kBlocks = 4096;
constexpr int kTraceBlocks = 8;
constexpr int kThreads = 128;
constexpr std::array<int, 8> kSampleOffsets{
    0, kMainBytes - 1, kMainBytes, kBytes - 1,
    kBytes, kBytes + kMainBytes - 1, kBytes + kMainBytes, 2 * kBytes - 1};

struct Storage {
  alignas(128) uint8_t bytes[kStages][2 * kBytes];
  alignas(16) uint64_t barriers[kStages];
};

struct Maps {
  CUtensorMap whole;
  CUtensorMap main;
  CUtensorMap scale;
};

struct Trace {
  uint64_t issue_start;
  uint64_t issue_end;
  uint64_t wait_start;
  uint64_t wait_end;
};

void check(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

void check_driver(CUresult status) {
  if (status != CUDA_SUCCESS) {
    char const* message = nullptr;
    cuGetErrorString(status, &message);
    throw std::runtime_error(message ? message : "CUDA driver error");
  }
}

CUtensorMap tensor_map(void* source, int tiles, int box_rows) {
  CUtensorMap map{};
  const cuuint64_t dimensions[]{kWidth, cuuint64_t(tiles) * kRows};
  const cuuint64_t strides[]{kWidth};
  const cuuint32_t box[]{kWidth, cuuint32_t(box_rows)};
  const cuuint32_t increments[]{1, 1};
  check_driver(cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, source, dimensions, strides,
      box, increments, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return map;
}

__host__ __device__ uint8_t payload(uint32_t offset, uint32_t stream) {
  uint32_t x = offset * 0x9e3779b9u + stream * 0x85ebca6bu;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  return uint8_t(x);
}

__host__ __device__ uint32_t digest_step(uint32_t digest, uint8_t value) {
  return (digest ^ uint32_t(value)) * 16777619u;
}

__global__ void initialize(uint8_t* a, uint8_t* b, int size) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < size; i += blockDim.x * gridDim.x) {
    a[i] = payload(i, 1);
    b[i] = payload(i, 2);
  }
}

__device__ __forceinline__ uint64_t timestamp() {
  uint64_t ticks;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(ticks) :: "memory");
  return ticks;
}

template <bool Split, int Lead, bool Observe>
__global__ void probe(const __grid_constant__ Maps a, const __grid_constant__ Maps b,
                      int tiles, uint32_t* output, Trace* trace) {
  extern __shared__ __align__(128) unsigned char smem[];
  auto& storage = *reinterpret_cast<Storage*>(smem);
  if (threadIdx.x != 0) {
    return;
  }
  int phase[kStages]{};
  #pragma unroll
  for (int stage = 0; stage < kStages; ++stage) {
    cute::initialize_barrier(storage.barriers[stage], 1);
  }
  bool record = Observe && blockIdx.x % (kBlocks / kTraceBlocks) == 0;
  auto* rows = Observe ? trace + (blockIdx.x / (kBlocks / kTraceBlocks)) * kRounds : nullptr;
  auto issue = [&](int round) {
    int stage = round % kStages;
    int first_row = ((blockIdx.x * kRounds + round) % tiles) * kRows;
    auto* destination = storage.bytes[stage];
    auto* barrier = &storage.barriers[stage];
    cute::set_barrier_transaction_bytes(*barrier, 2 * kBytes);
    uint64_t start = record ? timestamp() : 0;
    if constexpr (Split) {
      cute::SM90_TMA_LOAD_2D::copy(&a.main, barrier, 0, destination, 0, first_row);
      cute::SM90_TMA_LOAD_2D::copy(&b.main, barrier, 0, destination + kBytes, 0, first_row);
      cute::SM90_TMA_LOAD_2D::copy(&a.scale, barrier, 0, destination + kMainBytes, 0, first_row + kMainRows);
      cute::SM90_TMA_LOAD_2D::copy(&b.scale, barrier, 0, destination + kBytes + kMainBytes, 0, first_row + kMainRows);
    } else {
      cute::SM90_TMA_LOAD_2D::copy(&a.whole, barrier, 0, destination, 0, first_row);
      cute::SM90_TMA_LOAD_2D::copy(&b.whole, barrier, 0, destination + kBytes, 0, first_row);
    }
    if (record) {
      rows[round].issue_end = timestamp();
      rows[round].issue_start = start;
    }
  };

  #pragma unroll
  for (int round = 0; round < Lead; ++round) {
    issue(round);
  }
  uint32_t digest = 2166136261u;
  #pragma unroll 1
  for (int round = 0; round < kRounds; ++round) {
    int stage = round % kStages;
    uint64_t start = record ? timestamp() : 0;
    cute::wait_barrier(storage.barriers[stage], phase[stage]);
    if (record) {
      rows[round].wait_end = timestamp();
      rows[round].wait_start = start;
    }
    phase[stage] ^= 1;
    auto const* value = storage.bytes[stage];
    digest = digest_step(digest, value[0]);
    digest = digest_step(digest, value[kMainBytes - 1]);
    digest = digest_step(digest, value[kMainBytes]);
    digest = digest_step(digest, value[kBytes - 1]);
    digest = digest_step(digest, value[kBytes]);
    digest = digest_step(digest, value[kBytes + kMainBytes - 1]);
    digest = digest_step(digest, value[kBytes + kMainBytes]);
    digest = digest_step(digest, value[2 * kBytes - 1]);
    if (round + Lead < kRounds) {
      issue(round + Lead);
    }
  }
  output[blockIdx.x] = digest;
}

template <bool Split, int Lead, bool Observe>
void launch(Maps a, Maps b, int tiles, uint32_t* output, Trace* trace) {
  auto* kernel = probe<Split, Lead, Observe>;
  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Storage)));
  kernel<<<kBlocks, kThreads, sizeof(Storage)>>>(a, b, tiles, output, trace);
  check(cudaPeekAtLastError());
}

template <bool Observe>
void run(int arm, Maps a, Maps b, int tiles, uint32_t* output, Trace* trace) {
  switch (arm) {
    case 0: launch<true, 1, Observe>(a, b, tiles, output, trace); break;
    case 1: launch<false, 1, Observe>(a, b, tiles, output, trace); break;
    case 2: launch<true, 4, Observe>(a, b, tiles, output, trace); break;
    case 3: launch<false, 4, Observe>(a, b, tiles, output, trace); break;
    default: throw std::runtime_error("Invalid arm");
  }
}

void verify(thrust::host_vector<uint32_t> const& output, int tiles) {
  for (int block = 0; block < kBlocks; ++block) {
    uint32_t expected = 2166136261u;
    for (int round = 0; round < kRounds; ++round) {
      int tile = (block * kRounds + round) % tiles;
      for (int offset : kSampleOffsets) {
        expected = digest_step(expected, payload(tile * kBytes + offset % kBytes, 1 + offset / kBytes));
      }
    }
    if (output[block] != expected) {
      throw std::runtime_error("Payload digest mismatch at CTA " + std::to_string(block));
    }
  }
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 5) {
    throw std::runtime_error("Usage: same_buffer_tma tiles groups iterations timing|trace|0|1|2|3");
  }
  int tiles = std::stoi(argv[1]);
  int groups = std::stoi(argv[2]);
  int iterations = std::stoi(argv[3]);
  std::string mode = argv[4];
  if (tiles <= 0 || tiles > 16384 || groups <= 0 || iterations <= 0) {
    throw std::runtime_error("Invalid probe dimensions");
  }
  check_driver(cuInit(0));
  thrust::device_vector<uint8_t> a(size_t(tiles) * kBytes), b(size_t(tiles) * kBytes);
  thrust::device_vector<uint32_t> output(kBlocks);
  thrust::device_vector<Trace> trace(kTraceBlocks * kRounds);
  auto* a_ptr = thrust::raw_pointer_cast(a.data());
  auto* b_ptr = thrust::raw_pointer_cast(b.data());
  auto* out = thrust::raw_pointer_cast(output.data());
  auto* trace_out = thrust::raw_pointer_cast(trace.data());
  initialize<<<512, 256>>>(a_ptr, b_ptr, tiles * kBytes);
  check(cudaDeviceSynchronize());
  auto encode = [&](void* source) {
    return Maps{tensor_map(source, tiles, kRows), tensor_map(source, tiles, kMainRows),
                tensor_map(source, tiles, kScaleRows)};
  };
  Maps map_a = encode(a_ptr), map_b = encode(b_ptr);
  if (mode == "trace") {
    std::cout << "tiles,arm,cta,round,issue_start,issue_end,wait_start,wait_end\n";
    for (int arm = 0; arm < 4; ++arm) {
      run<true>(arm, map_a, map_b, tiles, out, trace_out);
      check(cudaDeviceSynchronize());
      verify(output, tiles);
      thrust::host_vector<Trace> rows = trace;
      for (int block = 0; block < kTraceBlocks; ++block) {
        for (int round = 0; round < kRounds; ++round) {
          auto const& row = rows[block * kRounds + round];
          std::cout << tiles << ',' << arm << ',' << block * (kBlocks / kTraceBlocks) << ','
                    << round << ',' << row.issue_start << ',' << row.issue_end << ','
                    << row.wait_start << ',' << row.wait_end << '\n';
        }
      }
    }
    return 0;
  }
  if (mode != "timing") {
    run<false>(std::stoi(mode), map_a, map_b, tiles, out, nullptr);
    check(cudaDeviceSynchronize());
    verify(output, tiles);
    return 0;
  }
  cudaEvent_t start, end;
  check(cudaEventCreate(&start));
  check(cudaEventCreate(&end));
  for (int arm = 0; arm < 4; ++arm) {
    run<false>(arm, map_a, map_b, tiles, out, nullptr);
    check(cudaDeviceSynchronize());
    verify(output, tiles);
  }
  constexpr int order[4][4]{{0, 1, 3, 2}, {1, 2, 0, 3}, {2, 3, 1, 0}, {3, 0, 2, 1}};
  std::cout << "tiles,group,position,arm,requests,lead,payload_bytes,smem_bytes,iterations,ms\n";
  std::cout << std::setprecision(9);
  for (int group = 0; group < groups; ++group) {
    for (int pos = 0; pos < 4; ++pos) {
      int arm = order[group % 4][pos];
      for (int i = 0; i < 3; ++i) {
        run<false>(arm, map_a, map_b, tiles, out, nullptr);
      }
      check(cudaEventRecord(start));
      for (int i = 0; i < iterations; ++i) {
        run<false>(arm, map_a, map_b, tiles, out, nullptr);
      }
      check(cudaEventRecord(end));
      check(cudaEventSynchronize(end));
      float ms = 0;
      check(cudaEventElapsedTime(&ms, start, end));
      std::cout << tiles << ',' << group << ',' << pos << ',' << arm << ','
                << (arm % 2 ? 2 : 4) << ',' << (arm < 2 ? 1 : 4) << ','
                << 2 * kBytes << ',' << sizeof(Storage) << ',' << iterations << ','
                << ms / iterations << '\n';
    }
    std::cout.flush();
  }
  check(cudaEventDestroy(start));
  check(cudaEventDestroy(end));
  return 0;
} catch (std::exception const& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
