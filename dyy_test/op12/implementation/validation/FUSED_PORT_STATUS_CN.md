# 单 CTA CuTe FP8 融合移植状态

> **历史多变量移植，不符合当前 MMA-only 口径。** 该目录还改变了主 `io_dtype`、inverse scratch 和输出 dtype；当前只作为失败路径和 CuTe lowering 研究材料保留。新的融合实现必须从 clean FlashInfer 固定提交重新开始，只改变七组主 MMA 和必要的临时 E4M3 operand staging。

## 结论

实验移植源码已保存在 `op12_gdn/cute_fused/`，但未接入 `gdn_prefill_fp8()`。它目前能生成并验证 CuTe IR，随后在 SM100a NVVM backend 阶段失败，未产出 cubin，也没有正确性结果。因此融合 kernel 的状态是 **未完成**。

公共 API 继续使用已验过的 `scaled_mm_unfused` backend；这不是静默回退，因为 `return_stats=True` 会明确返回 backend 名称，README 和验证报告也固定声明该边界。

## 已完成的移植工作

1. 从固定 FlashInfer 提交独立复制 1-SM TND kernel/adapter，保留原 BSD-3-Clause 与 Apache-2.0 文件头；上游文件未改。
2. Q/K/V 主 dtype 改为 `Float8E4M3FN`，inverse scratch 改为 Float32。
3. 七组 tcgen05 atom 改为 `MmaF8F6F4Op(E4M3,E4M3,F32)`，MMA-K 从 16 改为 32。
4. FP8 TMEM fragment 的布局从上游 16-bit `(128,16)` 变为 packed `(64,2,16)`；将两处 `St16x128b x8` 改为合法的 `St32x32b x16`，CuTe layout verification 通过。
5. 上游 V 路径使用的 `LdMatrix8x8x16b` 不能直接以 E4M3 为元素类型，实验路径已换成 32-bit universal vector load。
6. A-inverse/Wqkv 从 FP32 rmem 转 E4M3 后发布到 swizzled SMEM 的旧 16-bit tiled-copy 不兼容；实验路径现以 Uint32 打包 4 个 E4M3，并按 32-bit 数据宽度重建 swizzle base 后发布，CuTe IR verifier 通过。

## 当前阻断

编译进入目标 `sm_100a` 的 libNVVM 后失败：`NVVM_ERROR_COMPILATION`，backend 没有给出更细日志。完整日志为 `raw/fused_compile_attempt.log`，用于复现的入口为 `validation/compile_fused_probe.py`。`raw/fused_ir/` 保存了约 2.1 MB raw MLIR 和约 0.74 MB clean MLIR，证明失败点已经在 CuTe layout/legalization 之后。

Uint32 packed store 已排除逐字节 store 是唯一原因，但 NVVM 仍不给出子操作诊断。下一步应把“FP32 fragment→E4M3→packed swizzled SMEM→F8 MMA”抽成最小 CuTe 子 kernel，确认是 4.7.0 lowering 缺陷还是完整 GDN IR 规模触发；若最小例也失败，应携 clean MLIR 向 CUTLASS DSL 报告。修复后再恢复 `opt-level 3`。随后还需：

- 加入 `maxabs/448`、FP32 tile scale 与反标定；当前实验文件只有直接 E4M3 转换。
- 将 output dtype 从上游单一 `io_dtype` 中拆出 FP16；当前 compile probe 临时使用 FP8 O，只为缩小编译问题。
- 把 upstream 512-column TMEM/full-DV 结构改成计划的 448-column、DV=64 half kernel。
- 完成数值、资源、spill、sanitizer 后才能接公共 API；之后才进入 2-SM cluster 与 CP parallel-prefix。

## 复现

```bash
cd /home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/implementation
PYTHONPATH=.:../upstreams/flashinfer ../.venv/bin/python validation/compile_fused_probe.py
```

该失败不是 CUDA 环境缺失：同一环境的固定 FlashInfer SM100 CuTe kernel 可编译运行，unfused op12 的 12/12 E4M3 GEMM 也有 NCU tensor-pipe/`UTCQMMA` 证据。
