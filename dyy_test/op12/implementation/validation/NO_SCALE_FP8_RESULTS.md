# 历史实验（已被 MMA-only 口径取代）：全 FP8 输入/中间量无缩放验证

> 本文件改变了公共 Q/K/V dtype，并把多类中间量整体转换为 E4M3，不满足当前“仅七组核心 MMA 改 FP8，其他保持 FlashInfer 原版”的单变量要求。它仅用于历史复现，不再作为 op12 结论。当前结果见 `MMA_ONLY_FP8_RESULTS.md`。

- 上游固定提交：`e77a4a0d276367895c3b50a642fd8f326c03fb72`
- Gold：FlashInfer `tests/gdn/reference_delta_rule.py::blockwise_delta_rule`。
- 输入：FlashInfer `tests/gdn/conftest.py::gen_qkv`，K 按上游测试先做 L2 normalize，再将 Q/K/V 直接转 E4M3。
- 唯一数值变化：七组逻辑 GEMM 的输入使用 E4M3；中间量直接饱和转 E4M3；所有 MMA operand scale 固定为 1。
- 保持不变：gate/beta、累计门控、三角逆、FP32 accumulator/epilogue、算法 QK scale=1/sqrt(128)。
- 执行载体：unfused `torch._scaled_mm`；本结果验证精度，不声称上游 CuTe 融合 kernel 已改造完成。

## 结论

- op12 阈值：12/13 case 通过。
- 上游 FP16 严格 allclose：4/13 case 通过。
- **最终判定：不完全达标。** `BNSD_S1` 的 O relative-L2 为 0.456409，超过 0.20；所以即使其 cosine 和 state 误差合格，也不能通过完整验收矩阵。
- 失败属于数值精度问题，不是构建或运行故障：13 组结果全部有限，B200 全量回归 20/20 通过。

| case | layout | seq | heads Q/K/V | O rel-L2 | O cosine | state rel-L2 | scaled O rel-L2 | scaled state rel-L2 | zeroed nonzero max | op12 | strict |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BNSD_S1 | BNSD | 1 | 1/1/1 | 0.456409 | 0.996829 | 0.086873 | 0.028962 | 0.028856 | 0.004883 | FAIL | PASS |
| BNSD_S63 | BNSD | 63 | 1/1/1 | 0.066699 | 0.997809 | 0.036320 | 0.050360 | 0.046185 | 0.931885 | PASS | FAIL |
| BNSD_S64 | BNSD | 64 | 1/1/1 | 0.060106 | 0.998204 | 0.037007 | 0.052639 | 0.039650 | 0.911377 | PASS | FAIL |
| BNSD_S65 | BNSD | 65 | 1/1/1 | 0.068088 | 0.997722 | 0.037246 | 0.059073 | 0.028549 | 0.896973 | PASS | FAIL |
| BNSD_S127 | BNSD | 127 | 1/1/1 | 0.067519 | 0.997722 | 0.039524 | 0.051496 | 0.043475 | 0.876709 | PASS | PASS |
| BNSD_S257 | BNSD | 257 | 1/1/1 | 0.061449 | 0.998112 | 0.039017 | 0.056560 | 0.026237 | 0.941406 | PASS | FAIL |
| TND_varlen | TND | 1,63,64,65,127,257 | 1/1/1 | 0.065329 | 0.997880 | 0.033636 | 0.052039 | 0.041789 | 0.984131 | PASS | FAIL |
| GQA_8_2 | TND | 64 | 8/2/2 | 0.061957 | 0.998110 | 0.031189 | 0.052190 | 0.048231 | 0.949463 | PASS | FAIL |
| GVA_2_8 | TND | 64 | 2/2/8 | 0.060037 | 0.998197 | 0.035502 | 0.053669 | 0.045269 | 0.957275 | PASS | FAIL |
| TND_initial_state | TND | 65 | 1/1/1 | 0.059978 | 0.998243 | 0.060495 | 0.057882 | 0.040405 | 0.934326 | PASS | PASS |
| QKV_low_0p25 | TND | 64 | 1/1/1 | 0.194952 | 0.981036 | 0.060994 | 0.058826 | 0.040272 | 0.966064 | PASS | PASS |
| QKV_high_4p0 | TND | 64 | 1/1/1 | 0.042500 | 0.999097 | 0.054102 | 0.051540 | 0.047856 | 0.900879 | PASS | FAIL |
| TND_long_4096 | TND | 4096 | 1/1/1 | 0.063938 | 0.997954 | 0.037707 | 0.052209 | 0.039425 | 0.983643 | PASS | FAIL |

## 判定规则

- op12：O relative-L2≤0.20、O cosine≥0.98、final state relative-L2≤0.25、无 NaN/Inf。
- 上游严格基线：O atol=2e-3/rtol=1e-3；state atol=1e-3/rtol=1e-4。
- `scaled` 两列是同一算法、同一输入下的 tile amax/448 对照，只用于量化损失归因。

## 误差归因

- 13 组的最大饱和率都是 0，因此本轮主要问题不是 E4M3 上溢。
- 无缩放时，大量小中间量会直接舍入为 0。典型 64-token case 中，`A⁻¹`/`Wqkv` 的非零元素归零率约 45%，`delta_state` 最高超过 90%；TND 变长与 4096-token case 的 `delta_state` 峰值分别达到 98.41% 和 98.36%。
- `BNSD_S1` 的绝对 O 误差仅 `8.07e-6`，但 gold 输出范数非常小，relative-L2 被放大到 0.456409。门槛以 relative-L2 为准，所以仍判失败。带 tile scale 的同 case relative-L2 为 0.028962，说明缩放能直接消除该失败。
- 低幅输入 `QKV_low_0p25` 已接近边界：O relative-L2=0.194952、cosine=0.981036。这说明无缩放方案对实际输入幅度敏感，不能仅凭常规幅度 case 通过就认定具有普适精度。

## 工程结论

本实现证明了“FlashInfer GDN 算法保持不变、七组逻辑 GEMM 使用 E4M3 Tensor Core、FP32 累加且无数据相关缩放”可以在多数本轮形状上工作；但它没有通过完整 op12 精度矩阵，也没有维持上游 FP16 的严格精度。因此，无缩放 FP8 不能作为 op12 的最终通用实现。若继续满足“完全无缩放”的约束，需要改变验收范围或保留部分敏感中间量为 FP16/BF16；后者已经不再是“七组 MMA 全 FP8”的原始约束。
