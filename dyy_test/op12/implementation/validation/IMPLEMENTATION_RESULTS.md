# op12 B200 最终验证摘要：仅七组核心 MMA 改 FP8

- 基线：FlashInfer `e77a4a0d276367895c3b50a642fd8f326c03fb72` SM100 non-CP chunked GDN。
- 唯一变量：七组主 GEMM 的 FP16/BF16 operands 在 MMA 边界直接转 E4M3；scale=1、FP32 accumulator。
- 保持不变：公共 TND、Q/K/V/O FP16/BF16、gate/beta FP32、state、三角逆、残差、epilogue、尾块与 head 语义。
- 环境：NVIDIA B200 / CC 10.0 / PyTorch 2.11.0+cu130 / CUDA 13.0。

## 结果

| 检查 | 结果 |
|---|---:|
| FlashInfer 原版严格 allclose | 15/15 |
| MMA-only FP8 op12 门槛 | 13/15 |
| MMA-only FP8 严格 allclose | 3/15 |
| 全部有限 | 15/15 |
| 最大饱和率 | 0 |
| pytest | 23 passed |
| NCU FP8 GEMM | 12/12 tensor-pipe 非零 |
| SASS | 96 条 `UTCQMMA` |

失败项：

- `FP16_TND_S1`：O relative-L2 `0.229802 > 0.20`。
- `FP16_QKV_low_0p25`：O relative-L2 `0.200916 > 0.20`，cosine `0.979779 < 0.98`。

最终判定：**不完全达标。** 失败是无缩放 E4M3 对小操作数的数值损失，不是上游、构建或 GPU 执行故障。完整表格与量化统计见 `MMA_ONLY_FP8_RESULTS.md`。

## 证据边界

当前 backend 是 unfused correctness vehicle。NCU 证明它确实执行 B200 FP8 Tensor Core，但不能替代最终融合 CuTe kernel 的 cubin、资源、spill 和性能验收。
