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
constexpr int kSamples = 5;
constexpr int kIterations = 50;

struct Case {
  char const* name;
  int blocks;
  int smem_bytes;
  int cluster_x;
};

__global__ void empty_kernel(unsigned* sink) {
  if (threadIdx.x == 0) {
    sink[blockIdx.x] = blockIdx.x;
  }
}

__global__ void smem_kernel(unsigned* sink) {
  extern __shared__ unsigned smem[];
  if (threadIdx.x == 0) {
    smem[0] = blockIdx.x;
    sink[blockIdx.x] = smem[0];
  }
}

__global__ void smem_sync_kernel(unsigned* sink, bool clustered) {
  extern __shared__ unsigned smem[];
  if (threadIdx.x == 0) {
    smem[0] = blockIdx.x;
  }
  if (clustered) {
    cg::this_cluster().sync();
  } else {
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    sink[blockIdx.x] = smem[0];
  }
}

template <typename Kernel, typename... Args>
void launch(Case const& problem, Kernel kernel, int smem_bytes, Args... args) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(problem.blocks, 1, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = smem_bytes;

  cudaLaunchAttribute attribute{};
  if (problem.cluster_x > 1) {
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = problem.cluster_x;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
  }
  CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
}

template <typename Kernel, typename... Args>
void benchmark(
    Case const& problem, char const* suffix, Kernel kernel, unsigned* sink,
    int smem_bytes, Args... args) {
  launch(problem, kernel, smem_bytes, sink, args...);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int sample = 1; sample <= kSamples; ++sample) {
    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      launch(problem, kernel, smem_bytes, sink, args...);
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
      {"shape8", 1024, 214016, 2},
      {"shape11", 512, 224256, 1},
      {"shape13", 1024, 206848, 1},
  };

  unsigned* sink = nullptr;
  CUDA_CHECK(cudaMalloc(&sink, 1024 * sizeof(unsigned)));
  CUDA_CHECK(cudaFuncSetAttribute(
      smem_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 227328));
  CUDA_CHECK(cudaFuncSetAttribute(
      smem_sync_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 227328));

  for (Case const& problem : cases) {
    benchmark(problem, "empty", empty_kernel, sink, 0);
    benchmark(problem, "smem", smem_kernel, sink, problem.smem_bytes);
    benchmark(
        problem, "sync", smem_sync_kernel, sink, problem.smem_bytes,
        problem.cluster_x > 1);
  }

  CUDA_CHECK(cudaFree(sink));
  return 0;
}
