# op12：FlashInfer GDN 仅 MMA 改 FP8

当前 op12 是相对 FlashInfer 固定提交 `e77a4a0d276367895c3b50a642fd8f326c03fb72` 的单变量实验：公共 Q/K/V/O 继续使用 FP16/BF16，gate/beta、state、布局、三角逆、残差和 epilogue 保持原版；只把七组核心 GEMM 的临时 MMA 操作数改为 E4M3，scale 恒为 1，FP32 累加。

可执行正确性入口：

```python
from op12_gdn import chunk_gated_delta_rule_mma_only_fp8
```

它保持原版 packed TND `[total_tokens,H,128] + cu_seqlens` 接口。BNSD、2-SM cluster、tile scaling、固定 FP8 公共输入和固定 FP16 state/O 都不属于本轮单变量方案。

## 当前结论

- B200 精度矩阵：13/15 通过 op12 放宽门槛，整体不达标。
- FlashInfer 原版同数据：15/15 通过严格 allclose。
- MMA-only FP8：3/15 通过严格 allclose。
- B200 NCU：12/12 个物理 FP8 GEMM launch 的 tensor-pipe 计数非零，SASS 检出 96 条 `UTCQMMA`。
- 全量测试：23 passed。

详细设计见 `OP12_FP8_GDN_B200_DESIGN_CN.md`，逐 case 结果见 `validation/MMA_ONLY_FP8_RESULTS.md`，原始 JSON 见 `validation/mma_only_fp8_results.json`。

## 历史实验

`gdn_prefill_fp8()`、`backend_scaled_mm.py`、`NO_SCALE_FP8_RESULTS.md` 和 `cute_fused/` 属于此前改变公共 dtype、内部精度或 tile scaling 的多变量实验，现仅为复现保留，不再作为 op12 当前结论或最终实现依据。

## B200 运行

```bash
cd /home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/implementation
../.venv/bin/python -m pytest -q tests
../.venv/bin/python validation/run_mma_only_validation.py
```
