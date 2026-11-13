#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t status_ = (expr);                                                \
    if (status_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "%s failed: %s\n", #expr, cudaGetErrorString(status_)); \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

constexpr int kBlocks = 256;
constexpr int kThreads = 256;
constexpr int kSmemBytes = 227328;
constexpr int kSamples = 5;
constexpr int kIterations = 50;

__device__ unsigned linear_block_idx() {
  return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
}

__global__ void empty_cluster_kernel(unsigned* sink) {
  if (threadIdx.x == 0) {
    unsigned block = linear_block_idx();
    sink[block] = block;
  }
}

__global__ void smem_cluster_kernel(unsigned* sink) {
  extern __shared__ unsigned smem[];
  if (threadIdx.x == 0) {
    unsigned block = linear_block_idx();
    smem[0] = block;
    sink[block] = smem[0];
  }
}

__global__ void smem_cluster_sync_kernel(unsigned* sink) {
  extern __shared__ unsigned smem[];
  if (threadIdx.x == 0) {
    smem[0] = linear_block_idx();
  }
  cg::this_cluster().sync();
  if (threadIdx.x == 0) {
    sink[linear_block_idx()] = smem[0];
  }
}

template <typename Kernel>
void launch(Kernel kernel, unsigned* sink, int smem_bytes, bool clustered) {
  cudaLaunchConfig_t config{};
  config.gridDim = clustered ? dim3(kBlocks / 2, 2, 1)
                             : dim3(kBlocks, 1, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = smem_bytes;

  cudaLaunchAttribute attribute{};
  if (clustered) {
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = 2;
    attribute.val.clusterDim.y = 2;
    attribute.val.clusterDim.z = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
  }
  CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, sink));
}

template <typename Kernel>
void benchmark(
    char const* name, Kernel kernel, unsigned* sink, int smem_bytes,
    bool clustered) {
  launch(kernel, sink, smem_bytes, clustered);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int sample = 1; sample <= kSamples; ++sample) {
    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      launch(kernel, sink, smem_bytes, clustered);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::printf(
        "RESULT,%s,%d,%d,%.9f\n", name, sample, kIterations,
        elapsed_ms / kIterations);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
  unsigned* sink = nullptr;
  CUDA_CHECK(cudaMalloc(&sink, kBlocks * sizeof(unsigned)));

  CUDA_CHECK(cudaFuncSetAttribute(
      smem_cluster_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSmemBytes));
  CUDA_CHECK(cudaFuncSetAttribute(
      smem_cluster_sync_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSmemBytes));

  benchmark("empty_cluster_smem0", empty_cluster_kernel, sink, 0, true);
  benchmark(
      "empty_cluster_smem227k", smem_cluster_kernel, sink, kSmemBytes, true);
  benchmark(
      "cluster_sync_smem227k", smem_cluster_sync_kernel, sink, kSmemBytes,
      true);

  CUDA_CHECK(cudaFree(sink));
  return 0;
}
