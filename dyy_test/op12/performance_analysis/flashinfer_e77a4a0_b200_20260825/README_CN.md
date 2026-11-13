# op12：FlashInfer GDN B200 性能分析归档

本目录是固定提交 `e77a4a0d276367895c3b50a642fd8f326c03fb72` 的 FlashInfer 原版 FP16/BF16 GDN 分析交付，不包含 unfused FP8 实验。

- 主报告：`OP12_FLASHINFER_GDN_B200_PERFORMANCE_ANALYSIS_CN.docx`（19 页逐页渲染检查通过）
- PDF 便览：`OP12_FLASHINFER_GDN_B200_PERFORMANCE_ANALYSIS_CN.pdf`
- Draw.io 多页源文件：`OP12_FLASHINFER_GDN_B200_DIAGRAMS.drawio`
- 插图：`01_api_dispatch.png` 至 `06_bottleneck_chain.png`
- 精简证据：`evidence/`（性能汇总、NCU 汇总、SASS 指令族汇总、7/7 smoke 结果）
- B200 全量原始证据：`/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/performance_analysis/flashinfer_e77a4a0_b200_20260825/`
- B200 报告/图源副本：上述目录的 `report/` 与 `figures/`

关键结论：非 CP 长序列的第一瓶颈是 `B×Ho` 决定的全卡并行度；接近满卡后，活跃 SM 内表现为依赖链、long-scoreboard 和数据等待主导的混合瓶颈，而不是纯 Tensor Core 或纯 DRAM 饱和。B1/H16/S32768 上 CP 将中位时延从约 0.942 ms 降至约 0.352 ms（2.68×），代价是四个 kernel、workspace 和额外流量。
