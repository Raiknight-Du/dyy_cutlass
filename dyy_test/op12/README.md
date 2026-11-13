# op12：FlashInfer GDN 对照、B200 验证与 FP8 实现基线

## 结论

FlashInfer 固定提交 `e77a4a0d276367895c3b50a642fd8f326c03fb72` 与 op12 **有明确的算法和平台重合，但不能直接满足 op12 的完整要求**。

| op12 要求 | 判定 | 结论 |
|---|---|---|
| LLM prefill | 达成 | 公共入口为 `chunk_gated_delta_rule()`，源码明确标注 prefill。 |
| Gated DeltaNet | 达成 | 实现 chunked gated delta rule，并包含 7 组核心 GEMM。 |
| B200 / SM100 | 达成 | 有专用 SM100 CuTe DSL 非 CP 与 CP 路径，B200 实机验证通过。 |
| TND packed-varlen | 达成 | Q/K/V/O 使用 `[total_tokens, heads, 128]`，序列边界由 `cu_seqlens` 表达。 |
| TND 下的 GDN 内部矩阵乘 | 达成，但精度不符 | QK、Q-state、Ainv-V、QKV、KV-update 均存在，但主计算只支持 FP16/BF16。 |
| FP8 主计算输入和 FP8 MMA | 不达成 | FP8 Q/K/V 被拒绝；核心指令构造为 `tcgen05.MmaF16BF16Op`。 |
| FP8 recurrent state | 部分达成 | 非 CP SM100 路径支持 E4M3/E5M2 state 存取，但这不是 FP8 主计算。 |
| BNSD `[B,H,S,D]` 入口 | 不达成 | 公共接口只声明三维 TND；B200 负向测试确认四维输入不能使用。 |

因此，这份实现可作为 op12 的 **GDN 算法、TND 变长组织、SM100 调度与流水设计参考**；FP8 Tensor Core 主计算与 BNSD 计算入口仍需新增开发。

本目录现已新增 `implementation/` 独立实现基线：它不修改 FlashInfer，提供 BNSD/TND `gdn_prefill_fp8()`、FP32 gold、七组 E4M3 Tensor-Core GEMM、在线 tile scale、GQA/GVA、尾块和 initial/final state。当前 backend 是正确性优先的 unfused 实现；单 CTA CuTe 融合、物理 2-SM cluster/DSM/TMA multicast 和 CP parallel-prefix 仍未完成，不能把当前结果解释为最终性能 kernel。

单 CTA 融合并非只留在文档：`implementation/op12_gdn/cute_fused/` 保存了从固定上游独立移植的实验源码，七组 atom 已替换成 `MmaF8F6F4Op`，编译已越过 CuTe layout verification 并进入 SM100a NVVM backend；但仍失败且没有 cubin，所以未接公共 API。具体阻断见 `implementation/validation/FUSED_PORT_STATUS_CN.md`。

## 目录

- `upstreams/flashinfer/`：完整递归克隆的 FlashInfer，上游代码保持干净，固定到上述提交。
- `OPEN_SOURCE_RELATION_ANALYSIS_CN.md`：op12 要求拆解、源码证据和差异分析。
- `SOURCES.json`：来源、提交、子模块、哈希与验证环境清单。
- `validation/SMOKE_RESULTS.md`：B200 冒烟验证汇总。
- `validation/smoke_raw.log`：最终 7/7 通过的 JSONL 原始日志。
- `validation/smoke_raw_attempt1_dependency_mismatch.log`：隔离环境依赖未闭合时的首轮诊断日志。
- `validation/run_smoke.py`：可重复执行的验证脚本。
- `OP12_FP8_GDN_B200_DESIGN_CN.md`：冻结的 GDN 流程、精度落点、B200 资源规划和实现边界。
- `implementation/`：独立 op12 API、FP32 reference、seven-GEMM FP8 backend、资源/指令门禁和测试。
- `implementation/validation/IMPLEMENTATION_RESULTS.md`：11 个正确性 case 的误差与饱和率。
- `implementation/validation/HARDWARE_EVIDENCE.md`：NCU tensor-pipe 与 SASS `UTCQMMA` 证据。
- `implementation/validation/FUSED_PORT_STATUS_CN.md`：单 CTA CuTe 移植进展、NVVM 阻断与复现方式。

## 验证摘要

- GPU：NVIDIA B200，compute capability 10.0。
- CUDA / PyTorch：CUDA 13.0 runtime，PyTorch 2.11.0+cu130。
- 隔离依赖：CuTe DSL 4.7.0、apache-tvm-ffi 0.1.13.post3。
- 上游能力审计：7/7 smoke 通过。
- op12 独立实现：19/19 pytest 通过，11/11 accuracy case 通过；BNSD/TND compute-sanitizer memcheck 为 0 error。
- 硬件证据：完整单 chunk probe 的 12/12 个 FP8 GEMM launch 均有 tensor-pipe 指令，SASS 共检出 96 条 `UTCQMMA`。
- 本轮仍不设性能目标；没有把 unfused backend 当作融合 kernel 做性能结论。
