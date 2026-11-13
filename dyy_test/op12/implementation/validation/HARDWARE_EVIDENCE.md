# B200 FP8 硬件执行证据

## 当前 MMA-only 证据

公共 Q/K/V 为 FP16，只有七组核心 MMA 的临时 operands 转 E4M3、scale 恒为 1。`raw/op12_mma_only_fp8_probe.ncu-rep` 捕获一个 64-token/单-head GDN 调用的 12 个物理 FP8 GEMM launch；12/12 的 `sm__inst_executed_pipe_tensor.sum` 均非零（32 或 64），SASS 共检出 96 条 `UTCQMMA`。这证明当前单变量 correctness backend 确实执行 B200 FP8 Tensor Core。

以下 `scaled_mm` 与旧 `unit_scale_fp8` 证据属于历史多变量 backend，仅保留用于复现。

## 环境

- GPU：NVIDIA B200，compute capability 10.0。
- PyTorch 2.11.0+cu130，CUDA runtime 13.0，CuTe DSL 4.7.0。
- 输入：E4M3×E4M3；`torch._scaled_mm(..., out_dtype=torch.float32, use_fast_accum=False)`。

## NCU/SASS 观察

独立 64×128 乘 128×64 探针的 SM100 kernel 名为 `nvjet_sm100_qqsss_64x8_128x16_1x4_h_bz_NNT`。NCU 指标 `sm__inst_executed_pipe_tensor.sum` 为 32；导入 SASS 后可见多条 `UTCQMMA`。原始 NCU 报告保存为 `raw/fp8_mm_probe.ncu-rep`。

完整 64-token、单 head 的 op12 probe 发出 12 次物理 FP8 GEMM：Wkk/Wqk 各一次，其余五组因 DV=128 分为两个64列半片而各两次。报告 `raw/op12_seven_gemm_probe.ncu-rep` 只采集 `nvjet_sm100_*` kernel；12/12 个 launch 的 `sm__inst_executed_pipe_tensor.sum` 均大于零（4、8 或 32），导入 SASS 共检出 96 条 `UTCQMMA`。

本次又对 `backend="unit_scale_fp8"` 单独重采证，报告为 `raw/op12_unit_scale_seven_gemm_probe.ncu-rep`。它同样捕获 12 个 `nvjet_sm100_*` launch，12/12 的 `sm__inst_executed_pipe_tensor.sum` 均大于零（4、8 或 32），SASS 共检出 96 条 `UTCQMMA`。因此无缩放精度实验本身确实在 B200 上执行 FP8 Tensor Core；这里的“无缩放”指 `_scaled_mm` ABI 所需的两个 scale 都是字面值 1，而不是绕过 Tensor Core 做 FP32 模拟。

`tools/check_cute_mma_contract.py` 还用 CuTe DSL 4.7.0 为七组契约逐一构造了：

- A/B：`Float8E4M3FN`
- accumulator：`Float32`
- class：`tcgen05.MmaF8F6F4Op`
- MMA-K：32

## 证据边界

`UTCQMMA`、tensor-pipe 计数和 E4M3 参数证明当前可执行 backend 确实走 B200 FP8 Tensor Core；但其 kernel 由 CUDA/PyTorch 的 SM100 GEMM dispatcher 生成，不是 op12 最终融合 CuTe kernel。只有最终融合产物的 PTX/SASS 同时出现七组 `tcgen05`、且中间量不落 GMEM，才能关闭融合内核门禁。
