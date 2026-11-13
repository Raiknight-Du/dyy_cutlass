# op12 原生 CUDA/CUTLASS GDN Prefill 实现架构

## 兼容目标

公开入口与固定提交的 `chunk_gated_delta_rule` 对齐：TND packed-varlen、
FP16/BF16 Q/K/V、D=128、MHA/GQA/GVA、FP32 gate/beta、可选 initial/final
state、五种 state 存储精度、checkpoint、state pool indices，以及
`use_cp=False/True/auto`。

## 后端边界

公开 API 不变，通过 `OP12_NATIVE_BACKEND` 选择三条路径：

- `tcgen05`（默认）：固定到 `e77a4a0...` 的上游派生 CuTe DSL 性能路径；
  直接复用 op12 归档源码，因此能保持相同的 12-warp、TMA/TMEM、七组 GEMM
  和 CP 四 kernel。它不计作独立 CUDA C 重实现。
- `wmma`：独立 CUDA C 64-token blockwise 路径，七组 GEMM 使用 WMMA/HMMA，
  FP32 累加和三角逆，保留相同公开语义。
- `scalar`：独立逐 token 语义基线。

## 已落地的标量语义基线

当前 `cuda_recurrent_v0` 由一个 CTA 持有一个 sequence/output-head。CTA 将
公开 `[V,K]` state 转成片上 `[K,V]` FP32 state，逐 token 执行：

1. `Sold = alpha * Sprev`；
2. `vold = K * Sold`；
3. `delta = beta * (V - vold)`；
4. `Snew = Sold + K^T * delta`；
5. `O = scale * Q * Snew`。

state 始终在约 64 KiB shared memory 中递推，仅首尾访问 GMEM。该路径用于
冻结接口语义、尾块和 state I/O 行为，不作为最终性能路径。

## 独立 64-token WMMA 路径

最终非 CP kernel 以 64-token chunk 为调度单元，复现上游七组逻辑 GEMM：

1. `Wkk = K K^T`；
2. `Wqk = Q K^T`；
3. `KS = K Sprev`；
4. `QS = Q Sprev`；
5. `NV = A^{-1} diag(beta) R`；
6. `Ointra = Wqkv NV`；
7. `dS = K^T Delta`。

门控累计、严格下三角、三角逆、residual 和 state epilogue 保持 FP32。
七组主 GEMM 使用 CUDA WMMA FP16/BF16 输入和 FP32 accumulator；B200 SASS
出现 `HMMA.16816.F32`。该版本通过 39/39 差分测试，S8192/FP16 非 CP 从
标量的 47.53 ms 降至 15.39 ms，但 Tensor pipe 仅 2.34%，仍受逐 GEMM
同步、共享内存往返和串行三角逆限制。

## tcgen05 性能路径

默认路径使用 op12 内固定 FlashInfer 提交的 CuTe DSL kernel，非 CP 恢复
384 threads/12 warps、226.3 KiB SMEM、TMEM state、TMA pipeline 和
`UTCHMMA`；CP 恢复 T precompute、M/N precompute、fixup、final prefill
四 kernel。40 配置、三独立进程、20 warmup、100 CUDA Event 样本的延迟比
相对同一固定 FlashInfer 为 1.002-1.044，中位数 1.017。

## CP 路径

低 `B x Ho` 长序列具备独立标量 CP 执行路径：partition-local M/N 仿射
预计算、跨 partition fixup、独立 final prefill 三个 kernel。host 根据 SM 数
和 head 数自适应选择 CP chunk，使 partition 数覆盖更多 SM；不能把 CP
chunk 固定写成 4096。B1/H16/S8192 的标量 CUDA CP 基线由 47.5 ms 降至
约 13.8 ms。默认 tcgen05 派生后端使用上游完整四 kernel CP 数据流，并在
相同方法下达到固定 FlashInfer 的 1.0-1.04 倍延迟。
