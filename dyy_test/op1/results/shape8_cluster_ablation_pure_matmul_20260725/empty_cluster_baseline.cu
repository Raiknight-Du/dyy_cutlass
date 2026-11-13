#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

#define CUDA_CHECK(expr)                                                          \
  do {                                                                            \
    cudaError_t status_ = (expr);                                                  \
    if (status_ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "%s failed: %s\n", #expr, cudaGetErrorString(status_)); \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

constexpr int kThreads = 256;
constexpr int kSmemBytes = 214016;
constexpr int kSamples = 25;
constexpr int kIterations = 10;

struct Case {
  char const* name;
  int cluster_x;
  int cluster_y;
};

__global__ void empty_kernel(unsigned* sink) {
  unsigned block = blockIdx.x + gridDim.x * blockIdx.y;
  if (threadIdx.x == 0) {
    sink[block] = block;
  }
}

__global__ void smem_kernel(unsigned* sink) {
  extern __shared__ unsigned smem[];
  unsigned block = blockIdx.x + gridDim.x * blockIdx.y;
  if (threadIdx.x == 0) {
    smem[0] = block;
    sink[block] = smem[0];
  }
}

__global__ void sync_kernel(unsigned* sink) {
  extern __shared__ unsigned smem[];
  unsigned block = blockIdx.x + gridDim.x * blockIdx.y;
  if (threadIdx.x == 0) {
    smem[0] = block;
  }
  cg::this_cluster().sync();
  if (threadIdx.x == 0) {
    sink[block] = smem[0];
  }
}

template <typename Kernel>
void launch(Case const& problem, Kernel kernel, unsigned* sink, int smem_bytes) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(32, 32, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = smem_bytes;

  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x = problem.cluster_x;
  attribute.val.clusterDim.y = problem.cluster_y;
  attribute.val.clusterDim.z = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, sink));
}

template <typename Kernel>
void benchmark(
    Case const& problem, char const* suffix, Kernel kernel, unsigned* sink,
    int smem_bytes) {
  launch(problem, kernel, sink, smem_bytes);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int sample = 1; sample <= kSamples; ++sample) {
    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      launch(problem, kernel, sink, smem_bytes);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::printf(
        "RESULT,%s_%s,%d,%d,%.9f\n", problem.name, suffix, sample,
        kIterations, elapsed_ms / kIterations);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
  Case const cases[] = {
      {"C2x1", 2, 1},
      {"C2x2", 2, 2},
      {"C4x1", 4, 1},
  };

  unsigned* sink = nullptr;
  CUDA_CHECK(cudaMalloc(&sink, 1024 * sizeof(unsigned)));
  CUDA_CHECK(cudaFuncSetAttribute(
      smem_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  CUDA_CHECK(cudaFuncSetAttribute(
      sync_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));

  for (Case const& problem : cases) {
    benchmark(problem, "empty", empty_kernel, sink, 0);
    benchmark(problem, "smem", smem_kernel, sink, kSmemBytes);
    benchmark(problem, "sync", sync_kernel, sink, kSmemBytes);
  }

  CUDA_CHECK(cudaFree(sink));
  return 0;
}
