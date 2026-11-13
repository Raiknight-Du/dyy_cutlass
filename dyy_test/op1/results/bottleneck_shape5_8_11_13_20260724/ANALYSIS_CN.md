# M_72a shape5/8/11/13 深入瓶颈实验

日期：2026-07-24

## 1. 目的与口径

本实验专门检验两种看似直接的优化推论：

1. 缩小 N tile、增加 block waves，能否用更多并行 block 掩盖等待；
2. 增大 N/K tile、减少 block 或 K-loop 次数，能否摊薄固定开销。

所有变体均为 pure matmul：`ElementC=void`、Identity epilogue、固定随机
A/B/SFA/SFB。Event 采用 1 次内置 warmup、每样本 10 iterations、
25 个独立样本。NCU 使用热缓存和非强控时钟口径。

## 2. 关键机制

### 2.1 更多 waves 不等于同一 SM 有更多 CTA

Auto stage 会把缩小 tile 省下的 SMEM继续换成更多 mainloop stage：

| Tile | Stage | Mainloop tensor | Epilogue tensor | 总 SMEM |
|---|---:|---:|---:|---:|
| M128 N64 K128 | 14 | 196.0 KiB | 16.5 KiB | 213.0 KiB |
| M128 N128 K128 | 11 | 198.0 KiB | 16.5 KiB | 215.0 KiB |
| M128 N192 K128 | 8 | 184.0 KiB | 33.0 KiB | 218.0 KiB |
| M128 N256 K128 | 5 | 135.0 KiB | 65.0 KiB | 201.0 KiB |
| M256 N128 K256 | 7 | 196.0 KiB | 8.25 KiB | 204.8 KiB |
| M256 N192 K256 | 6 | 204.0 KiB | 16.5 KiB | 221.0 KiB |
| M256 N256 K256, E64x64 | 5 | 190.0 KiB | 17.0 KiB | 208.0 KiB |
| M256 N256 K512 | 2 | 152.0 KiB | 65.0 KiB | 218.0 KiB |

因此 Auto 变体的 `launch__occupancy_limit_shared_mem` 始终为 1。
增加 grid waves 只会让同一 SM 顺序执行更多 CTA，不会让第二个 CTA
与当前 CTA 的 mbarrier/TMEM 等待重叠。

### 2.2 强制低 stage 仍没有第二个 CTA

对 K=128 的 shape11/13，只有一个 K iteration。额外 stage 没有下一块
数据可预取，因此又补测了 Stage=1/2：

| Shape | Stage | Event | SMEM | SMEM block limit | Reg block limit | 实际 active CTA/SM | No Eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | Auto=8 | 8.915 us | 224.3 KiB | 1 | 2 | 1.000 | 82.12% |
| 11 | 1 | 9.668 us | 59.4 KiB | 2 | 2 | 1.000 | 83.30% |
| 11 | 2 | 9.380 us | 82.9 KiB | 2 | 2 | 1.000 | 82.86% |
| 13 | Auto=5 | 15.052 us | 206.8 KiB | 1 | 1 | 1.000 | 85.22% |
| 13 | 1 | 15.759 us | 96.3 KiB | 1 | 1 | 1.000 | 86.27% |
| 13 | 2 | 15.442 us | 123.9 KiB | 1 | 1 | 1.000 | 85.75% |

实际 active CTA/SM 用
`sm__ctas_active.avg / sm__cycles_active.avg` 计算。shape11 即使 SMEM 和
寄存器都允许 2 blocks，实际仍只有 1 CTA/SM。源码原因是
`sm100_gemm_tma_warpspecialized.hpp` 中 TMEM allocator 固定申请全部
512 columns。shape13 还额外受 153 registers/thread 限制。

所以当前实现不能仅靠缩 stage 得到第二个 TMEM-consuming CTA；实测
No Eligible 也没有下降。

## 3. Shape5：8192 x 768 x 4096

| N tile / K tile | Blocks | Waves | Event | 相对 N192 | No Eligible | Long SB | Tensor |
|---|---:|---:|---:|---:|---:|---:|---:|
| N64 / K256 | 768 | 5.19 | 22.886 us | +49.9% | 84.84% | 7.32 | 28.79% |
| N128 / K256 | 384 | 2.59 | 15.580 us | +2.0% | 85.03% | 5.64 | 42.83% |
| N192 / K256 | 256 | 1.73 | 15.271 us | 基准 | 88.41% | 8.31 | 43.79% |
| N256 / K256 | 192 | 1.30 | 19.224 us | +25.9% | 91.21% | 12.94 | 35.41% |
| N256 / K512 | 192 | 1.30 | 27.557 us | +80.4% | 95.46% | 29.56 | 22.02% |

每个 CTA 的本地 M 输出仍为 128。N tile 从 192 缩到 128/64 时，N tile
数从 4 增至 6/12，A/SFA tile 被分别重复 1.5/3 倍，同时 CTA 初始化、
四路 TMA、barrier 和 epilogue 实例数也增加。N128 确实把 No Eligible
和 Long SB 降低了，但省下的等待不足以覆盖新增工作。

向反方向增至 N256 后，block 数只剩 192，第二波只有 44 blocks，
即 29.7% SM 填充；stage 也从 6 降到 4。进一步把 K tile 增至 512，
单 stage TMA transaction 从 76 KiB 增到 152 KiB、stage 从 4 降到 2，
Long SB 升到 29.56。

具体卡点是：输入 stage 的 TMA-to-SMEM barrier，以及 UMMA/TMEM
accumulator 交给 epilogue 前的 `SYNCS...TRYWAIT`。shape5 的 Long SB
最大 PC 是 accumulator wait 的轮询分支，随后进入 BF16 转换和
epilogue。L2 hit 为 97%–99.7%，DRAM SOL 低于 0.3%，不是字节带宽不够，
而是单 CTA 的 readiness latency 无第二 CTA 可填。

## 4. Shape8：8192 x 4096 x 2048

| 配置 | Blocks | Waves | Event | 相对最佳 | No Eligible | Long SB | Tensor |
|---|---:|---:|---:|---:|---:|---:|---:|
| N64 K256 | 4096 | 27.68 | 52.614 us | +66.4% | 83.21% | 6.24 | 32.30% |
| N128 K256 | 2048 | 13.84 | 35.617 us | +12.6% | 81.00% | 3.77 | 50.91% |
| N256 K256 | 1024 | 6.92 | 31.621 us | 最佳 | 84.02% | 4.33 | 56.66% |
| N256 K128 | 1024 | 6.92 | 35.728 us | +13.0% | 88.68% | 9.26 | 48.91% |
| N256 K512 | 1024 | 6.92 | 46.178 us | +46.0% | 94.70% | 25.43 | 35.68% |

最佳配置已有 6.92 waves，最后一波利用率 91.9%，因此 shape8 不能再
主要归因于设备 grid underfill。N128 将 waves 翻倍并把 No Eligible
降低约 3 个百分点，runtime 仍变差，原因是 A/SFA 重复量和 CTA/
epilogue 数量翻倍，而同一 SM 仍只有一个 CTA。

K tile 扫描给出了更直接的流水证据：

- K128：16 次 K-loop，TMA/barrier/UTCCP 轮数过多；
- K256：8 次 K-loop、5 stages，吞吐和预取距离最平衡；
- K512：4 次 K-loop，但单 stage 为 152 KiB，只能开 2 stages，
  Long SB 和 No Eligible 急升。

因此 shape8 的具体瓶颈是 8-iteration mainloop 中
TMA-to-SMEM readiness 与 UMMA/TMEM-to-epilogue readiness 未充分
摊销。SourceCounters 最大热点是 accumulator pipeline 的
`SYNCS.PHASECHK.TRANS64.TRYWAIT`，另外还有 mainloop stage 和
epilogue store barrier。DRAM SOL 25.8%、L2 hit 89.1%，排除 HBM 饱和。

## 5. Shape11：16384 x 768 x 128

| N tile | Blocks | Waves | Event | 相对 N192 | No Eligible | Long SB |
|---|---:|---:|---:|---:|---:|---:|
| 64 | 1536 | 10.38 | 10.961 us | +23.0% | 77.04% | 3.02 |
| 128 | 768 | 5.19 | 10.868 us | +21.9% | 81.98% | 4.17 |
| 192 | 512 | 3.46 | 8.915 us | 最佳 | 82.11% | 3.96 |
| 256 | 384 | 2.59 | 9.965 us | +11.8% | 85.23% | 5.04 |

N64 将 waves 提高到 10.38，并把 No Eligible 降低约 5 个百分点，
仍慢 23%。这是“提高 grid 并行度不能解决”的直接反证：它把 N tile
数从 4 增到 12，A/SFA 和 CTA 固定路径增至 3 倍，但这些 CTA只能按
wave 顺序执行。

K 只有一个 iteration，单 CTA 的关键路径没有稳态 mainloop：

```text
L2-hot GMEM
  -> A/B/SFA/SFB TMA
  -> SMEM mbarrier wait
  -> SFA/SFB UTCCP: SMEM -> TMEM
  -> UMMA: A/B(SMEM) + SF(TMEM) -> accumulator(TMEM)
  -> accumulator TRYWAIT
  -> LDTM -> register -> BF16 -> SMEM D
  -> TMA store D
```

没有前一轮 UMMA 可掩盖输入，也没有下一轮 TMA 可与当前计算重叠。
低-stage 实验又证明第二个 CTA 仍被 512-column TMEM allocation
挡住。L2 hit 96.7%、DRAM SOL 0.84%，所以瓶颈在 SMEM/TMEM/epilogue
串行 readiness 链，不在 HBM。

## 6. Shape13：8192 x 4096 x 128

| N tile | Blocks | Waves | Event | 相对 N256 | No Eligible | DRAM SOL | L2 hit |
|---|---:|---:|---:|---:|---:|---:|
| 64 | 4096 | 27.68 | 21.090 us | +40.1% | 75.99% | 41.85% | 94.29% |
| 128 | 2048 | 13.84 | 19.079 us | +26.8% | 81.55% | 42.78% | 93.46% |
| 192 | 1408 | 9.51 | 17.124 us | +13.8% | 81.02% | 31.29% | 85.94% |
| 256 | 1024 | 6.92 | 15.052 us | 最佳 | 85.23% | 32.00% | 86.05% |

shape13 与 shape11 不同：输出 D 本身约 64 MiB，N tile 变小时 A/SFA
重复和 TMA transaction 数快速增加，因此 DRAM/L1/L2 服务压力也上升。
N64 虽把 No Eligible 降低约 9 个百分点，却增加 4 倍 CTA，最终仍慢
40%。最佳 N256 已有 6.92 waves 和 91.9% 尾波利用率，不是 grid
underfill 主导。

更大的有效 tile 也没有可用空间：

- Blackwell block-scaled atom 的 CTA N 合法上限为 256；
- `M256` 的 2SM atom 仍拆成每 CTA 本地 `M128xN256` 输出，block 数
  仍是 1024；实测 2SM 版为 15.356 us，比 1SM 版慢约 2.0%；
- K=128 已只有一个 iteration。把 K tile 改成 256 仍是一次迭代，
  不能减少循环，实测 16.036 us，慢约 6.5%；
- Stage=1/2 后寄存器上限仍为 1 CTA/SM，TMEM 也固定申请 512 columns。

SourceCounters 最大 Long SB PC 是 mainloop barrier 的
`SYNCS...TRYWAIT`，下一条有效路径开始从 SMEM 取数据；其他热点位于
UTCCP/accumulator wait 和 epilogue fence。故具体瓶颈是单 K tile 的
TMA-to-SMEM 首块就绪、TMEM accumulator 交接和 D epilogue 串行链。
DRAM SOL 约 32% 说明 D/重复输入流量是重要次因，但尚未达到 HBM 饱和。

## 7. 结论

四个 shape 的“不能继续改善”原因并不相同：

| Shape | 具体主限制 | 缩 tile 的反证 | 放大 tile 的反证 |
|---:|---|---|---|
| 5 | 小网格 + 单 CTA 的 acc/TMA readiness | 5.19 waves 仍慢 49.9% | N256 尾波差；K512 仅 2 stages |
| 8 | 8 次 K-loop 的 SMEM/TMEM readiness | waves 翻倍仍慢 12.6% | K512 Long SB 25.43 |
| 11 | 单 K tile 串行链 + TMEM 独占 | 10.38 waves 仍慢 23.0% | N256 waves 更少；低 stage 仍 1 CTA |
| 13 | 单 K tile输入/epilogue链 + D/cache流量 | 27.68 waves 仍慢 40.1% | N 已达上限；M256/K256 均退化 |

因此这些 shape 不是抽象的“固定开销大”，而是当前 CUTLASS
warp-specialized 实现中的几个不可并行区间被暴露：

1. TMA 数据到 SMEM stage 的 mbarrier readiness；
2. SFA/SFB 从 SMEM 到 TMEM 后才能发 UMMA；
3. accumulator 在 TMEM ready 后 epilogue 才能 LDTM；
4. BF16 D 必须经 register -> SMEM -> TMA store；
5. 固定 512-column TMEM allocation 使第二个 CTA 无法驻留填泡。

## 8. 复现文件

- `op1_bottleneck_probe.cu`：N/K tile 对照；
- `op1_stage_probe.cu`：Stage=1/2 对照；
- `resource_probe.cu`、`resource_sizes.csv`：SMEM 分项；
- `event_summary.csv`、`comparison_summary.csv`：Event 与 NCU 汇总；
- `stage_event_summary.csv`、`ncu_stage/`：低-stage 驻留证据；
- 远程 `.ncu-rep` 位于同名实验目录的 `ncu/` 与 `ncu_stage/`。
