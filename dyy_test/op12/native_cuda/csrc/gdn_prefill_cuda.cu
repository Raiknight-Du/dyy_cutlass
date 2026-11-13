#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kHeadDim = 128;
constexpr int kThreads = 256;

enum StateDtype : int {
  kFloat32 = 0,
  kFloat16 = 1,
  kBFloat16 = 2,
  kFloat8E4M3 = 3,
  kFloat8E5M2 = 4,
};

__device__ __forceinline__ float load_state_value(const void* ptr, int dtype, int64_t idx) {
  switch (dtype) {
    case kFloat32:
      return static_cast<const float*>(ptr)[idx];
    case kFloat16:
      return __half2float(static_cast<const __half*>(ptr)[idx]);
    case kBFloat16:
      return __bfloat162float(static_cast<const __nv_bfloat16*>(ptr)[idx]);
    case kFloat8E4M3:
      return static_cast<float>(static_cast<const __nv_fp8_e4m3*>(ptr)[idx]);
    case kFloat8E5M2:
      return static_cast<float>(static_cast<const __nv_fp8_e5m2*>(ptr)[idx]);
    default:
      return 0.0f;
  }
}

__device__ __forceinline__ void store_state_value(void* ptr, int dtype, int64_t idx, float x) {
  switch (dtype) {
    case kFloat32:
      static_cast<float*>(ptr)[idx] = x;
      break;
    case kFloat16:
      static_cast<__half*>(ptr)[idx] = __float2half_rn(x);
      break;
    case kBFloat16:
      static_cast<__nv_bfloat16*>(ptr)[idx] = __float2bfloat16_rn(x);
      break;
    case kFloat8E4M3:
      static_cast<__nv_fp8_e4m3*>(ptr)[idx] = __nv_fp8_e4m3(x);
      break;
    case kFloat8E5M2:
      static_cast<__nv_fp8_e5m2*>(ptr)[idx] = __nv_fp8_e5m2(x);
      break;
  }
}

template <typename T>
__device__ __forceinline__ float load_io(const T* ptr, int64_t idx);

template <>
__device__ __forceinline__ float load_io<__half>(const __half* ptr, int64_t idx) {
  return __half2float(ptr[idx]);
}

template <>
__device__ __forceinline__ float load_io<__nv_bfloat16>(
    const __nv_bfloat16* ptr, int64_t idx) {
  return __bfloat162float(ptr[idx]);
}

template <typename T>
__device__ __forceinline__ void store_io(T* ptr, int64_t idx, float x);

template <>
__device__ __forceinline__ void store_io<__half>(__half* ptr, int64_t idx, float x) {
  ptr[idx] = __float2half_rn(x);
}

template <>
__device__ __forceinline__ void store_io<__nv_bfloat16>(
    __nv_bfloat16* ptr, int64_t idx, float x) {
  ptr[idx] = __float2bfloat16_rn(x);
}

template <typename scalar_t>
__global__ __launch_bounds__(kThreads, 1) void gdn_recurrent_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    scalar_t* __restrict__ output,
    const int32_t* __restrict__ cu_seqlens,
    int num_seqs,
    int hq,
    int hk,
    int hv,
    int ho,
    const void* __restrict__ initial_state,
    int initial_dtype,
    void* __restrict__ output_state,
    int output_state_dtype,
    void* __restrict__ checkpoints,
    int checkpoint_dtype,
    const int32_t* __restrict__ checkpoint_cu_starts,
    int checkpoint_every,
    const int32_t* __restrict__ state_indices,
    float scale,
    bool use_qk_l2norm) {
  const int seq = blockIdx.x;
  const int oh = blockIdx.y;
  if (seq >= num_seqs || oh >= ho) {
    return;
  }

  extern __shared__ float shared[];
  float* state = shared;                              // [K,V]
  float* qv = state + kHeadDim * kHeadDim;
  float* kv = qv + kHeadDim;
  float* vv = kv + kHeadDim;
  float* old_v = vv + kHeadDim;
  float* delta = old_v + kHeadDim;
  float* norm = delta + kHeadDim;

  const int slot = state_indices ? state_indices[seq] : seq;
  for (int idx = threadIdx.x; idx < kHeadDim * kHeadDim; idx += blockDim.x) {
    const int kk = idx / kHeadDim;
    const int val = idx % kHeadDim;
    float x = 0.0f;
    if (initial_state) {
      const int64_t public_idx =
          (((static_cast<int64_t>(slot) * ho + oh) * kHeadDim + val) * kHeadDim + kk);
      x = load_state_value(initial_state, initial_dtype, public_idx);
    }
    state[idx] = x;
  }
  __syncthreads();

  const int qh = oh / (ho / hq);
  const int kh = oh / (ho / hk);
  const int vh = oh / (ho / hv);
  const int token_begin = cu_seqlens[seq];
  const int token_end = cu_seqlens[seq + 1];

  for (int token = token_begin; token < token_end; ++token) {
    if (threadIdx.x < kHeadDim) {
      qv[threadIdx.x] = load_io(q, (static_cast<int64_t>(token) * hq + qh) * kHeadDim + threadIdx.x);
      kv[threadIdx.x] = load_io(k, (static_cast<int64_t>(token) * hk + kh) * kHeadDim + threadIdx.x);
      vv[threadIdx.x] = load_io(v, (static_cast<int64_t>(token) * hv + vh) * kHeadDim + threadIdx.x);
    }
    __syncthreads();

    // Kept for the optimized backend contract.  The fixed SM100 upstream
    // prefill call does not currently forward this option.
    if (use_qk_l2norm) {
      if (threadIdx.x == 0) {
        float q2 = 0.0f;
        float k2 = 0.0f;
        for (int d = 0; d < kHeadDim; ++d) {
          q2 += qv[d] * qv[d];
          k2 += kv[d] * kv[d];
        }
        norm[0] = rsqrtf(fmaxf(q2, 1.0e-12f));
        norm[1] = rsqrtf(fmaxf(k2, 1.0e-12f));
      }
      __syncthreads();
      if (threadIdx.x < kHeadDim) {
        qv[threadIdx.x] *= norm[0];
        kv[threadIdx.x] *= norm[1];
      }
      __syncthreads();
    }

    const float a = gate[static_cast<int64_t>(token) * ho + oh];
    const float b = beta[static_cast<int64_t>(token) * ho + oh];

    if (threadIdx.x < kHeadDim) {
      const int val = threadIdx.x;
      float dot = 0.0f;
#pragma unroll 4
      for (int kk = 0; kk < kHeadDim; ++kk) {
        const int idx = kk * kHeadDim + val;
        const float x = a * state[idx];
        state[idx] = x;
        dot = fmaf(kv[kk], x, dot);
      }
      old_v[val] = dot;
      delta[val] = b * (vv[val] - dot);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < kHeadDim * kHeadDim; idx += blockDim.x) {
      const int kk = idx / kHeadDim;
      const int val = idx % kHeadDim;
      state[idx] = fmaf(kv[kk], delta[val], state[idx]);
    }
    __syncthreads();

    if (threadIdx.x < kHeadDim) {
      const int val = threadIdx.x;
      float dot = 0.0f;
#pragma unroll 4
      for (int kk = 0; kk < kHeadDim; ++kk) {
        dot = fmaf(qv[kk], state[kk * kHeadDim + val], dot);
      }
      store_io(output, (static_cast<int64_t>(token) * ho + oh) * kHeadDim + val, scale * dot);
    }
    __syncthreads();

    if (checkpoints && checkpoint_every > 0) {
      const int local_pos = token - token_begin + 1;
      if (local_pos % checkpoint_every == 0) {
        const int checkpoint_idx =
            checkpoint_cu_starts[seq] + local_pos / checkpoint_every - 1;
        for (int idx = threadIdx.x; idx < kHeadDim * kHeadDim; idx += blockDim.x) {
          const int kk = idx / kHeadDim;
          const int val = idx % kHeadDim;
          const int64_t public_idx =
              (((static_cast<int64_t>(checkpoint_idx) * ho + oh) * kHeadDim + val) *
                   kHeadDim +
               kk);
          store_state_value(checkpoints, checkpoint_dtype, public_idx, state[idx]);
        }
      }
    }
    __syncthreads();
  }

  if (output_state) {
    for (int idx = threadIdx.x; idx < kHeadDim * kHeadDim; idx += blockDim.x) {
      const int kk = idx / kHeadDim;
      const int val = idx % kHeadDim;
      const int64_t public_idx =
          (((static_cast<int64_t>(slot) * ho + oh) * kHeadDim + val) * kHeadDim + kk);
      store_state_value(output_state, output_state_dtype, public_idx, state[idx]);
    }
  }
}

template <typename scalar_t>
__global__ __launch_bounds__(kThreads, 1) void gdn_cp_affine_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const int32_t* __restrict__ cu_seqlens,
    int num_seqs,
    int hk,
    int hv,
    int ho,
    int cp_chunk_len,
    int max_partitions,
    float* __restrict__ workspace_m,
    float* __restrict__ workspace_n) {
  const int seq_part = blockIdx.x;
  const int seq = seq_part / max_partitions;
  const int part = seq_part % max_partitions;
  const int oh = blockIdx.y;
  if (seq >= num_seqs || oh >= ho) return;

  const int seq_begin = cu_seqlens[seq];
  const int seq_end = cu_seqlens[seq + 1];
  const int token_begin = seq_begin + part * cp_chunk_len;
  if (token_begin >= seq_end) return;
  const int token_end = min(seq_end, token_begin + cp_chunk_len);

  extern __shared__ float shared[];
  float* mat_m = shared;
  float* mat_n = mat_m + kHeadDim * kHeadDim;
  float* kv = mat_n + kHeadDim * kHeadDim;
  float* vv = kv + kHeadDim;
  float* proj_m = vv + kHeadDim;
  float* proj_n = proj_m + kHeadDim;

  for (int idx = threadIdx.x; idx < kHeadDim * kHeadDim; idx += blockDim.x) {
    const int row = idx / kHeadDim;
    const int col = idx % kHeadDim;
    mat_m[idx] = row == col ? 1.0f : 0.0f;
    mat_n[idx] = 0.0f;
  }
  __syncthreads();

  const int kh = oh / (ho / hk);
  const int vh = oh / (ho / hv);
  for (int token = token_begin; token < token_end; ++token) {
    if (threadIdx.x < kHeadDim) {
      kv[threadIdx.x] =
          load_io(k, (static_cast<int64_t>(token) * hk + kh) * kHeadDim + threadIdx.x);
      vv[threadIdx.x] =
          load_io(v, (static_cast<int64_t>(token) * hv + vh) * kHeadDim + threadIdx.x);
    }
    __syncthreads();

    if (threadIdx.x < kHeadDim) {
      const int col = threadIdx.x;
      float dot_m = 0.0f;
      float dot_n = 0.0f;
#pragma unroll 4
      for (int row = 0; row < kHeadDim; ++row) {
        dot_m = fmaf(kv[row], mat_m[row * kHeadDim + col], dot_m);
        dot_n = fmaf(kv[row], mat_n[row * kHeadDim + col], dot_n);
      }
      proj_m[col] = dot_m;
      proj_n[col] = dot_n;
    }
    __syncthreads();

    const float a = gate[static_cast<int64_t>(token) * ho + oh];
    const float b = beta[static_cast<int64_t>(token) * ho + oh];
    for (int idx = threadIdx.x; idx < kHeadDim * kHeadDim; idx += blockDim.x) {
      const int row = idx / kHeadDim;
      const int col = idx % kHeadDim;
      const float kr = kv[row];
      mat_m[idx] = fmaf(kr, -a * b * proj_m[col], a * mat_m[idx]);
      mat_n[idx] = fmaf(kr, b * (vv[col] - a * proj_n[col]), a * mat_n[idx]);
    }
    __syncthreads();
  }

  const int64_t base =
      (static_cast<int64_t>(seq_part) * ho + oh) * kHeadDim * kHeadDim;
  for (int idx = threadIdx.x; idx < kHeadDim * kHeadDim; idx += blockDim.x) {
    workspace_m[base + idx] = mat_m[idx];
    workspace_n[base + idx] = mat_n[idx];
  }
}

__global__ __launch_bounds__(kThreads, 1) void gdn_cp_fixup_kernel(
    const int32_t* __restrict__ cu_seqlens,
    int num_seqs,
    int ho,
    int cp_chunk_len,
    int max_partitions,
    const void* __restrict__ initial_state,
    int initial_dtype,
    void* __restrict__ output_state,
    int output_state_dtype,
    const float* __restrict__ workspace_m,
    const float* __restrict__ workspace_n,
    float* __restrict__ workspace_boundaries) {
  const int seq = blockIdx.x;
  const int oh = blockIdx.y;
  if (seq >= num_seqs || oh >= ho) return;

  extern __shared__ float shared[];
  float* state = shared;
  float* next = state + kHeadDim * kHeadDim;
  for (int idx = threadIdx.x; idx < kHeadDim * kHeadDim; idx += blockDim.x) {
    const int row = idx / kHeadDim;
    const int val = idx % kHeadDim;
    float x = 0.0f;
    if (initial_state) {
      const int64_t public_idx =
          (((static_cast<int64_t>(seq) * ho + oh) * kHeadDim + val) * kHeadDim + row);
      x = load_state_value(initial_state, initial_dtype, public_idx);
    }
    state[idx] = x;
  }
  __syncthreads();

  const int seq_len = cu_seqlens[seq + 1] - cu_seqlens[seq];
  const int partitions = (seq_len + cp_chunk_len - 1) / cp_chunk_len;
  for (int part = 0; part < partitions; ++part) {
    const int seq_part = seq * max_partitions + part;
    const int64_t base =
        (static_cast<int64_t>(seq_part) * ho + oh) * kHeadDim * kHeadDim;
    for (int idx = threadIdx.x; idx < kHeadDim * kHeadDim; idx += blockDim.x) {
      workspace_boundaries[base + idx] = state[idx];
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < kHeadDim * kHeadDim; idx += blockDim.x) {
      const int row = idx / kHeadDim;
      const int val = idx % kHeadDim;
      float sum = workspace_n[base + idx];
#pragma unroll 4
      for (int kk = 0; kk < kHeadDim; ++kk) {
        sum = fmaf(
            workspace_m[base + row * kHeadDim + kk],
            state[kk * kHeadDim + val],
            sum);
      }
      next[idx] = sum;
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < kHeadDim * kHeadDim; idx += blockDim.x) {
      state[idx] = next[idx];
    }
    __syncthreads();
  }

  if (output_state) {
    for (int idx = threadIdx.x; idx < kHeadDim * kHeadDim; idx += blockDim.x) {
      const int row = idx / kHeadDim;
      const int val = idx % kHeadDim;
      const int64_t public_idx =
          (((static_cast<int64_t>(seq) * ho + oh) * kHeadDim + val) * kHeadDim + row);
      store_state_value(output_state, output_state_dtype, public_idx, state[idx]);
    }
  }
}

template <typename scalar_t>
__global__ __launch_bounds__(kThreads, 1) void gdn_cp_final_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    scalar_t* __restrict__ output,
    const int32_t* __restrict__ cu_seqlens,
    int num_seqs,
    int hq,
    int hk,
    int hv,
    int ho,
    int cp_chunk_len,
    int max_partitions,
    const float* __restrict__ workspace_boundaries,
    float scale) {
  const int seq_part = blockIdx.x;
  const int seq = seq_part / max_partitions;
  const int part = seq_part % max_partitions;
  const int oh = blockIdx.y;
  if (seq >= num_seqs || oh >= ho) return;

  const int seq_begin = cu_seqlens[seq];
  const int seq_end = cu_seqlens[seq + 1];
  const int token_begin = seq_begin + part * cp_chunk_len;
  if (token_begin >= seq_end) return;
  const int token_end = min(seq_end, token_begin + cp_chunk_len);

  extern __shared__ float shared[];
  float* state = shared;
  float* qv = state + kHeadDim * kHeadDim;
  float* kv = qv + kHeadDim;
  float* vv = kv + kHeadDim;
  float* old_v = vv + kHeadDim;
  float* delta = old_v + kHeadDim;
  const int64_t boundary_base =
      (static_cast<int64_t>(seq_part) * ho + oh) * kHeadDim * kHeadDim;
  for (int idx = threadIdx.x; idx < kHeadDim * kHeadDim; idx += blockDim.x) {
    state[idx] = workspace_boundaries[boundary_base + idx];
  }
  __syncthreads();

  const int qh = oh / (ho / hq);
  const int kh = oh / (ho / hk);
  const int vh = oh / (ho / hv);
  for (int token = token_begin; token < token_end; ++token) {
    if (threadIdx.x < kHeadDim) {
      qv[threadIdx.x] =
          load_io(q, (static_cast<int64_t>(token) * hq + qh) * kHeadDim + threadIdx.x);
      kv[threadIdx.x] =
          load_io(k, (static_cast<int64_t>(token) * hk + kh) * kHeadDim + threadIdx.x);
      vv[threadIdx.x] =
          load_io(v, (static_cast<int64_t>(token) * hv + vh) * kHeadDim + threadIdx.x);
    }
    __syncthreads();
    const float a = gate[static_cast<int64_t>(token) * ho + oh];
    const float b = beta[static_cast<int64_t>(token) * ho + oh];
    if (threadIdx.x < kHeadDim) {
      const int val = threadIdx.x;
      float dot = 0.0f;
#pragma unroll 4
      for (int kk = 0; kk < kHeadDim; ++kk) {
        const int idx = kk * kHeadDim + val;
        const float x = a * state[idx];
        state[idx] = x;
        dot = fmaf(kv[kk], x, dot);
      }
      old_v[val] = dot;
      delta[val] = b * (vv[val] - dot);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < kHeadDim * kHeadDim; idx += blockDim.x) {
      const int kk = idx / kHeadDim;
      const int val = idx % kHeadDim;
      state[idx] = fmaf(kv[kk], delta[val], state[idx]);
    }
    __syncthreads();
    if (threadIdx.x < kHeadDim) {
      const int val = threadIdx.x;
      float dot = 0.0f;
#pragma unroll 4
      for (int kk = 0; kk < kHeadDim; ++kk) {
        dot = fmaf(qv[kk], state[kk * kHeadDim + val], dot);
      }
      store_io(
          output,
          (static_cast<int64_t>(token) * ho + oh) * kHeadDim + val,
          scale * dot);
    }
    __syncthreads();
  }
}

int state_dtype_code(const torch::Tensor& t) {
  if (t.numel() == 0 || t.scalar_type() == torch::kFloat32) return kFloat32;
  if (t.scalar_type() == torch::kFloat16) return kFloat16;
  if (t.scalar_type() == torch::kBFloat16) return kBFloat16;
  if (t.scalar_type() == torch::kFloat8_e4m3fn) return kFloat8E4M3;
  if (t.scalar_type() == torch::kFloat8_e5m2) return kFloat8E5M2;
  TORCH_CHECK(false, "unsupported state dtype ", t.scalar_type());
}

}  // namespace

void gdn_prefill_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor gate,
    torch::Tensor beta,
    torch::Tensor output,
    torch::Tensor cu_seqlens,
    torch::Tensor initial_state,
    torch::Tensor output_state,
    torch::Tensor state_checkpoints,
    torch::Tensor checkpoint_cu_starts,
    torch::Tensor state_indices,
    double scale,
    int64_t checkpoint_every_n_tokens,
    bool use_qk_l2norm) {
  at::cuda::CUDAGuard device_guard(q.device());
  const auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  const int num_seqs = static_cast<int>(cu_seqlens.numel() - 1);
  const int hq = static_cast<int>(q.size(1));
  const int hk = static_cast<int>(k.size(1));
  const int hv = static_cast<int>(v.size(1));
  const int ho = std::max(hq, hv);
  const dim3 grid(num_seqs, ho, 1);
  constexpr size_t shared_bytes =
      (kHeadDim * kHeadDim + 5 * kHeadDim + 2) * sizeof(float);

  const void* initial_ptr = initial_state.numel() ? initial_state.data_ptr() : nullptr;
  void* output_state_ptr = output_state.numel() ? output_state.data_ptr() : nullptr;
  void* checkpoint_ptr = state_checkpoints.numel() ? state_checkpoints.data_ptr() : nullptr;
  const int32_t* checkpoint_starts_ptr =
      checkpoint_cu_starts.numel() ? checkpoint_cu_starts.data_ptr<int32_t>() : nullptr;
  const int32_t* state_indices_ptr =
      state_indices.numel() ? state_indices.data_ptr<int32_t>() : nullptr;

  if (q.scalar_type() == torch::kFloat16) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        gdn_recurrent_kernel<__half>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));
    gdn_recurrent_kernel<__half><<<grid, kThreads, shared_bytes, stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        gate.data_ptr<float>(),
        beta.data_ptr<float>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        cu_seqlens.data_ptr<int32_t>(),
        num_seqs,
        hq,
        hk,
        hv,
        ho,
        initial_ptr,
        state_dtype_code(initial_state),
        output_state_ptr,
        state_dtype_code(output_state),
        checkpoint_ptr,
        state_dtype_code(state_checkpoints),
        checkpoint_starts_ptr,
        static_cast<int>(checkpoint_every_n_tokens),
        state_indices_ptr,
        static_cast<float>(scale),
        use_qk_l2norm);
  } else if (q.scalar_type() == torch::kBFloat16) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        gdn_recurrent_kernel<__nv_bfloat16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));
    gdn_recurrent_kernel<__nv_bfloat16><<<grid, kThreads, shared_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        gate.data_ptr<float>(),
        beta.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        cu_seqlens.data_ptr<int32_t>(),
        num_seqs,
        hq,
        hk,
        hv,
        ho,
        initial_ptr,
        state_dtype_code(initial_state),
        output_state_ptr,
        state_dtype_code(output_state),
        checkpoint_ptr,
        state_dtype_code(state_checkpoints),
        checkpoint_starts_ptr,
        static_cast<int>(checkpoint_every_n_tokens),
        state_indices_ptr,
        static_cast<float>(scale),
        use_qk_l2norm);
  } else {
    TORCH_CHECK(false, "q/k/v must be FP16 or BF16");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn_prefill_cp_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor gate,
    torch::Tensor beta,
    torch::Tensor output,
    torch::Tensor cu_seqlens,
    torch::Tensor initial_state,
    torch::Tensor output_state,
    torch::Tensor workspace_m,
    torch::Tensor workspace_n,
    torch::Tensor workspace_boundaries,
    double scale,
    int64_t cp_chunk_len,
    int64_t max_partitions) {
  at::cuda::CUDAGuard device_guard(q.device());
  const auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  const int num_seqs = static_cast<int>(cu_seqlens.numel() - 1);
  const int hq = static_cast<int>(q.size(1));
  const int hk = static_cast<int>(k.size(1));
  const int hv = static_cast<int>(v.size(1));
  const int ho = std::max(hq, hv);
  const int cp_chunk = static_cast<int>(cp_chunk_len);
  const int max_parts = static_cast<int>(max_partitions);
  const dim3 partition_grid(num_seqs * max_parts, ho, 1);
  const dim3 fixup_grid(num_seqs, ho, 1);
  constexpr size_t affine_shared_bytes =
      (2 * kHeadDim * kHeadDim + 4 * kHeadDim) * sizeof(float);
  constexpr size_t fixup_shared_bytes =
      (2 * kHeadDim * kHeadDim) * sizeof(float);
  constexpr size_t final_shared_bytes =
      (kHeadDim * kHeadDim + 5 * kHeadDim) * sizeof(float);

  const void* initial_ptr = initial_state.numel() ? initial_state.data_ptr() : nullptr;
  void* output_state_ptr = output_state.numel() ? output_state.data_ptr() : nullptr;

  C10_CUDA_CHECK(cudaFuncSetAttribute(
      gdn_cp_fixup_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(fixup_shared_bytes)));

  if (q.scalar_type() == torch::kFloat16) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        gdn_cp_affine_kernel<__half>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(affine_shared_bytes)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        gdn_cp_final_kernel<__half>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(final_shared_bytes)));
    gdn_cp_affine_kernel<__half><<<partition_grid, kThreads, affine_shared_bytes, stream>>>(
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        gate.data_ptr<float>(),
        beta.data_ptr<float>(),
        cu_seqlens.data_ptr<int32_t>(),
        num_seqs,
        hk,
        hv,
        ho,
        cp_chunk,
        max_parts,
        workspace_m.data_ptr<float>(),
        workspace_n.data_ptr<float>());
  } else if (q.scalar_type() == torch::kBFloat16) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        gdn_cp_affine_kernel<__nv_bfloat16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(affine_shared_bytes)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        gdn_cp_final_kernel<__nv_bfloat16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(final_shared_bytes)));
    gdn_cp_affine_kernel<__nv_bfloat16>
        <<<partition_grid, kThreads, affine_shared_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
            gate.data_ptr<float>(),
            beta.data_ptr<float>(),
            cu_seqlens.data_ptr<int32_t>(),
            num_seqs,
            hk,
            hv,
            ho,
            cp_chunk,
            max_parts,
            workspace_m.data_ptr<float>(),
            workspace_n.data_ptr<float>());
  } else {
    TORCH_CHECK(false, "q/k/v must be FP16 or BF16");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  gdn_cp_fixup_kernel<<<fixup_grid, kThreads, fixup_shared_bytes, stream>>>(
      cu_seqlens.data_ptr<int32_t>(),
      num_seqs,
      ho,
      cp_chunk,
      max_parts,
      initial_ptr,
      state_dtype_code(initial_state),
      output_state_ptr,
      state_dtype_code(output_state),
      workspace_m.data_ptr<float>(),
      workspace_n.data_ptr<float>(),
      workspace_boundaries.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (q.scalar_type() == torch::kFloat16) {
    gdn_cp_final_kernel<__half><<<partition_grid, kThreads, final_shared_bytes, stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        gate.data_ptr<float>(),
        beta.data_ptr<float>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        cu_seqlens.data_ptr<int32_t>(),
        num_seqs,
        hq,
        hk,
        hv,
        ho,
        cp_chunk,
        max_parts,
        workspace_boundaries.data_ptr<float>(),
        static_cast<float>(scale));
  } else {
    gdn_cp_final_kernel<__nv_bfloat16>
        <<<partition_grid, kThreads, final_shared_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
            gate.data_ptr<float>(),
            beta.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            cu_seqlens.data_ptr<int32_t>(),
            num_seqs,
            hq,
            hk,
            hv,
            ho,
            cp_chunk,
            max_parts,
            workspace_boundaries.data_ptr<float>(),
            static_cast<float>(scale));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
