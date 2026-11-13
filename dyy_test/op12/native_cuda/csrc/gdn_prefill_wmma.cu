#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

#include <algorithm>
#include <type_traits>

namespace {

namespace wmma = nvcuda::wmma;

constexpr int kD = 128;
constexpr int kChunk = 64;
constexpr int kWmmaThreads = 256;

enum StateDtype : int {
  kFloat32 = 0,
  kFloat16 = 1,
  kBFloat16 = 2,
  kFloat8E4M3 = 3,
  kFloat8E5M2 = 4,
};

__device__ __forceinline__ float load_state(const void* ptr, int dtype, int64_t idx) {
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

__device__ __forceinline__ void store_state(void* ptr, int dtype, int64_t idx, float x) {
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
__device__ __forceinline__ float to_float(T x);

template <>
__device__ __forceinline__ float to_float(__half x) {
  return __half2float(x);
}

template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ __half from_float(float x) {
  return __float2half_rn(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float(float x) {
  return __float2bfloat16_rn(x);
}

template <typename Layout>
__device__ __forceinline__ int matrix_offset(int row, int col, int ld) {
  if constexpr (std::is_same_v<Layout, wmma::row_major>) {
    return row * ld + col;
  } else {
    return row + col * ld;
  }
}

// CTA-wide, warp-striped 16x16x16 Tensor Core GEMM.  C is always row-major
// FP32 in shared memory.  The helper deliberately uses the CUDA WMMA API first;
// the subsequent SM100 backend replaces the instruction atom with tcgen05 while
// preserving this blockwise dataflow and the public API.
template <typename T, typename LayoutA, typename LayoutB>
__device__ void wmma_gemm(
    const T* __restrict__ a,
    const T* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    bool accumulate) {
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const int tiles_n = n / 16;
  const int total_tiles = (m / 16) * tiles_n;
  for (int tile = warp; tile < total_tiles; tile += warps) {
    const int tile_m = tile / tiles_n;
    const int tile_n = tile - tile_m * tiles_n;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, LayoutA> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, LayoutB> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c;
    if (accumulate) {
      wmma::load_matrix_sync(frag_c, c + tile_m * 16 * ldc + tile_n * 16, ldc, wmma::mem_row_major);
    } else {
      wmma::fill_fragment(frag_c, 0.0f);
    }
    for (int kk = 0; kk < k; kk += 16) {
      const T* tile_a = a + matrix_offset<LayoutA>(tile_m * 16, kk, lda);
      const T* tile_b = b + matrix_offset<LayoutB>(kk, tile_n * 16, ldb);
      wmma::load_matrix_sync(frag_a, tile_a, lda);
      wmma::load_matrix_sync(frag_b, tile_b, ldb);
      wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    wmma::store_matrix_sync(
        c + tile_m * 16 * ldc + tile_n * 16,
        frag_c,
        ldc,
        wmma::mem_row_major);
  }
  __syncthreads();
}

template <typename T>
__global__ __launch_bounds__(kWmmaThreads, 1) void gdn_blockwise_wmma_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    T* __restrict__ output,
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
    float scale) {
  const int seq = blockIdx.x;
  const int oh = blockIdx.y;
  if (seq >= num_seqs || oh >= ho) return;

  extern __shared__ __align__(128) unsigned char smem_raw[];
  size_t off = 0;
  auto alloc_float = [&](size_t count) {
    float* p = reinterpret_cast<float*>(smem_raw + off);
    off += count * sizeof(float);
    off = (off + 127) & ~size_t(127);
    return p;
  };
  auto alloc_t = [&](size_t count) {
    T* p = reinterpret_cast<T*>(smem_raw + off);
    off += count * sizeof(T);
    off = (off + 127) & ~size_t(127);
    return p;
  };

  float* s_state = alloc_float(kD * kD);       // persistent [K,V]
  T* s_state_in = alloc_t(kD * kD);            // per-chunk Tensor Core input
  T* s_q = alloc_t(kChunk * kD);                // later holds scaled QS
  T* s_k = alloc_t(kChunk * kD);
  T* s_v = alloc_t(kChunk * kD);                // V -> residual -> NV -> decay NV
  float* s_l = alloc_float(kChunk * kChunk);    // Wkk -> unit lower L
  T* s_ainv = alloc_t(kChunk * kChunk);         // inverse(L) * diag(beta)
  T* s_qk = alloc_t(kChunk * kChunk);           // causal transfer-scaled QK
  float* s_work = alloc_float(kChunk * kD);     // GEMM accumulator / inverse scratch
  float* s_log_prefix = alloc_float(kChunk);

  const int slot = state_indices ? state_indices[seq] : seq;
  for (int idx = threadIdx.x; idx < kD * kD; idx += blockDim.x) {
    const int kk = idx / kD;
    const int val = idx - kk * kD;
    float x = 0.0f;
    if (initial_state) {
      const int64_t public_idx =
          (((static_cast<int64_t>(slot) * ho + oh) * kD + val) * kD + kk);
      x = load_state(initial_state, initial_dtype, public_idx);
    }
    s_state[idx] = x;
  }
  __syncthreads();

  const int qh = oh / (ho / hq);
  const int kh = oh / (ho / hk);
  const int vh = oh / (ho / hv);
  const int token_begin = cu_seqlens[seq];
  const int token_end = cu_seqlens[seq + 1];
  const int seq_len = token_end - token_begin;

  for (int chunk_start = 0; chunk_start < seq_len; chunk_start += kChunk) {
    const int valid = min(kChunk, seq_len - chunk_start);
    for (int idx = threadIdx.x; idx < kChunk * kD; idx += blockDim.x) {
      const int row = idx / kD;
      const int d = idx - row * kD;
      if (row < valid) {
        const int token = token_begin + chunk_start + row;
        s_q[idx] = q[(static_cast<int64_t>(token) * hq + qh) * kD + d];
        s_k[idx] = k[(static_cast<int64_t>(token) * hk + kh) * kD + d];
        s_v[idx] = v[(static_cast<int64_t>(token) * hv + vh) * kD + d];
      } else {
        s_q[idx] = from_float<T>(0.0f);
        s_k[idx] = from_float<T>(0.0f);
        s_v[idx] = from_float<T>(0.0f);
      }
    }
    if (threadIdx.x == 0) {
      float running = 0.0f;
      for (int row = 0; row < kChunk; ++row) {
        const float a = row < valid
            ? gate[static_cast<int64_t>(token_begin + chunk_start + row) * ho + oh]
            : 1.0f;
        running += log2f(fmaxf(a, 1.0e-20f));
        s_log_prefix[row] = running;
      }
    }
    for (int idx = threadIdx.x; idx < kD * kD; idx += blockDim.x) {
      s_state_in[idx] = from_float<T>(s_state[idx]);
    }
    __syncthreads();

    // 1. Wkk = K K^T.
    wmma_gemm<T, wmma::row_major, wmma::col_major>(
        s_k, s_k, s_l, kChunk, kChunk, kD, kD, kD, kChunk, false);

    // Build L = I + strict_lower(T * diag(beta_row) * Wkk).
    for (int idx = threadIdx.x; idx < kChunk * kChunk; idx += blockDim.x) {
      const int row = idx / kChunk;
      const int col = idx - row * kChunk;
      if (row == col) {
        s_l[idx] = 1.0f;
      } else if (col < row && row < valid) {
        const int token = token_begin + chunk_start + row;
        const float transfer = exp2f(s_log_prefix[row] - s_log_prefix[col]);
        s_l[idx] *= transfer * beta[static_cast<int64_t>(token) * ho + oh];
      } else {
        s_l[idx] = 0.0f;
      }
    }
    // s_work[0:64,0:64] is the inverse scratch.  Rows are dependency ordered;
    // columns within a row are independent.
    for (int idx = threadIdx.x; idx < kChunk * kChunk; idx += blockDim.x) {
      const int row = idx / kChunk;
      const int col = idx - row * kChunk;
      s_work[row * kD + col] = row == col ? 1.0f : 0.0f;
    }
    __syncthreads();
    for (int row = 1; row < kChunk; ++row) {
      if (threadIdx.x < row) {
        const int col = threadIdx.x;
        float sum = 0.0f;
        for (int inner = col; inner < row; ++inner) {
          sum = fmaf(s_l[row * kChunk + inner], s_work[inner * kD + col], sum);
        }
        s_work[row * kD + col] = -sum;
      }
      __syncthreads();
    }
    for (int idx = threadIdx.x; idx < kChunk * kChunk; idx += blockDim.x) {
      const int row = idx / kChunk;
      const int col = idx - row * kChunk;
      const float b = col < valid
          ? beta[static_cast<int64_t>(token_begin + chunk_start + col) * ho + oh]
          : 0.0f;
      s_ainv[idx] = from_float<T>(s_work[row * kD + col] * b);
    }
    __syncthreads();

    // 2. QK and causal transfer/attention scaling.
    wmma_gemm<T, wmma::row_major, wmma::col_major>(
        s_q, s_k, s_work, kChunk, kChunk, kD, kD, kD, kD, false);
    for (int idx = threadIdx.x; idx < kChunk * kChunk; idx += blockDim.x) {
      const int row = idx / kChunk;
      const int col = idx - row * kChunk;
      float x = 0.0f;
      if (row < valid && col <= row) {
        x = s_work[row * kD + col] *
            exp2f(s_log_prefix[row] - s_log_prefix[col]) * scale;
      }
      s_qk[idx] = from_float<T>(x);
    }
    __syncthreads();

    // 3. KS = K S_prev; residual = V - cumprod * KS.
    wmma_gemm<T, wmma::row_major, wmma::row_major>(
        s_k, s_state_in, s_work, kChunk, kD, kD, kD, kD, kD, false);
    for (int idx = threadIdx.x; idx < kChunk * kD; idx += blockDim.x) {
      const int row = idx / kD;
      const float decay = exp2f(s_log_prefix[row]);
      s_v[idx] = from_float<T>(to_float(s_v[idx]) - decay * s_work[idx]);
    }
    __syncthreads();

    // 4. QS = Q S_prev; retain scaled inter-chunk output in the Q buffer.
    wmma_gemm<T, wmma::row_major, wmma::row_major>(
        s_q, s_state_in, s_work, kChunk, kD, kD, kD, kD, kD, false);
    for (int idx = threadIdx.x; idx < kChunk * kD; idx += blockDim.x) {
      const int row = idx / kD;
      s_q[idx] = from_float<T>(s_work[idx] * exp2f(s_log_prefix[row]) * scale);
    }
    __syncthreads();

    // 5. NV = Ainv*diag(beta) * residual.  V buffer becomes NV.
    wmma_gemm<T, wmma::row_major, wmma::row_major>(
        s_ainv, s_v, s_work, kChunk, kD, kChunk, kChunk, kD, kD, false);
    for (int idx = threadIdx.x; idx < kChunk * kD; idx += blockDim.x) {
      s_v[idx] = from_float<T>(s_work[idx]);
    }
    __syncthreads();

    // 6. O_intra = Wqkv * NV; add the already scaled QS term.
    wmma_gemm<T, wmma::row_major, wmma::row_major>(
        s_qk, s_v, s_work, kChunk, kD, kChunk, kChunk, kD, kD, false);
    for (int idx = threadIdx.x; idx < valid * kD; idx += blockDim.x) {
      const int row = idx / kD;
      const int d = idx - row * kD;
      const int token = token_begin + chunk_start + row;
      output[(static_cast<int64_t>(token) * ho + oh) * kD + d] =
          from_float<T>(s_work[idx] + to_float(s_q[idx]));
    }

    // Prepare the state update: S_next = decay_total*S_prev + K^T*delta,
    // delta_i = NV_i * decay_total/cumprod_i.
    const float total_decay = exp2f(s_log_prefix[kChunk - 1]);
    for (int idx = threadIdx.x; idx < kChunk * kD; idx += blockDim.x) {
      const int row = idx / kD;
      const float tail_decay = exp2f(s_log_prefix[kChunk - 1] - s_log_prefix[row]);
      s_v[idx] = from_float<T>(to_float(s_v[idx]) * tail_decay);
    }
    for (int idx = threadIdx.x; idx < kD * kD; idx += blockDim.x) {
      s_state[idx] *= total_decay;
    }
    __syncthreads();

    // 7. dS = K^T delta, accumulated directly into the FP32 state.
    wmma_gemm<T, wmma::col_major, wmma::row_major>(
        s_k, s_v, s_state, kD, kD, kChunk, kD, kD, kD, true);

    if (checkpoints && checkpoint_every > 0) {
      const int local_end = min(seq_len, chunk_start + kChunk);
      if (local_end % checkpoint_every == 0) {
        const int checkpoint_idx = checkpoint_cu_starts[seq] + local_end / checkpoint_every - 1;
        for (int idx = threadIdx.x; idx < kD * kD; idx += blockDim.x) {
          const int kk = idx / kD;
          const int val = idx - kk * kD;
          const int64_t public_idx =
              (((static_cast<int64_t>(checkpoint_idx) * ho + oh) * kD + val) * kD + kk);
          store_state(checkpoints, checkpoint_dtype, public_idx, s_state[idx]);
        }
      }
    }
    __syncthreads();
  }

  if (output_state) {
    for (int idx = threadIdx.x; idx < kD * kD; idx += blockDim.x) {
      const int kk = idx / kD;
      const int val = idx - kk * kD;
      const int64_t public_idx =
          (((static_cast<int64_t>(slot) * ho + oh) * kD + val) * kD + kk);
      store_state(output_state, output_state_dtype, public_idx, s_state[idx]);
    }
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

template <typename T>
constexpr size_t shared_bytes() {
  // The allocator rounds each segment to 128 B.  All current segment sizes are
  // already multiples of 128 B, so the closed form is exact.
  return (kD * kD + kChunk * kChunk + kChunk * kD) * sizeof(float) +
      (kD * kD + 3 * kChunk * kD + 2 * kChunk * kChunk) * sizeof(T) +
      kChunk * sizeof(float);
}

}  // namespace

void gdn_prefill_wmma_cuda(
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
    int64_t checkpoint_every_n_tokens) {
  at::cuda::CUDAGuard device_guard(q.device());
  const auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  const int num_seqs = static_cast<int>(cu_seqlens.numel() - 1);
  const int hq = static_cast<int>(q.size(1));
  const int hk = static_cast<int>(k.size(1));
  const int hv = static_cast<int>(v.size(1));
  const int ho = std::max(hq, hv);
  const dim3 grid(num_seqs, ho, 1);

  const void* initial_ptr = initial_state.numel() ? initial_state.data_ptr() : nullptr;
  void* output_state_ptr = output_state.numel() ? output_state.data_ptr() : nullptr;
  void* checkpoint_ptr = state_checkpoints.numel() ? state_checkpoints.data_ptr() : nullptr;
  const int32_t* checkpoint_starts_ptr = checkpoint_cu_starts.numel()
      ? checkpoint_cu_starts.data_ptr<int32_t>()
      : nullptr;
  const int32_t* state_indices_ptr = state_indices.numel()
      ? state_indices.data_ptr<int32_t>()
      : nullptr;

  if (q.scalar_type() == torch::kFloat16) {
    constexpr size_t smem = shared_bytes<__half>();
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        gdn_blockwise_wmma_kernel<__half>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem)));
    gdn_blockwise_wmma_kernel<__half><<<grid, kWmmaThreads, smem, stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        gate.data_ptr<float>(), beta.data_ptr<float>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        cu_seqlens.data_ptr<int32_t>(), num_seqs, hq, hk, hv, ho,
        initial_ptr, state_dtype_code(initial_state),
        output_state_ptr, state_dtype_code(output_state),
        checkpoint_ptr, state_dtype_code(state_checkpoints),
        checkpoint_starts_ptr, static_cast<int>(checkpoint_every_n_tokens),
        state_indices_ptr, static_cast<float>(scale));
  } else if (q.scalar_type() == torch::kBFloat16) {
    constexpr size_t smem = shared_bytes<__nv_bfloat16>();
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        gdn_blockwise_wmma_kernel<__nv_bfloat16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem)));
    gdn_blockwise_wmma_kernel<__nv_bfloat16><<<grid, kWmmaThreads, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        gate.data_ptr<float>(), beta.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        cu_seqlens.data_ptr<int32_t>(), num_seqs, hq, hk, hv, ho,
        initial_ptr, state_dtype_code(initial_state),
        output_state_ptr, state_dtype_code(output_state),
        checkpoint_ptr, state_dtype_code(state_checkpoints),
        checkpoint_starts_ptr, static_cast<int>(checkpoint_every_n_tokens),
        state_indices_ptr, static_cast<float>(scale));
  } else {
    TORCH_CHECK(false, "q/k/v must be FP16 or BF16");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
