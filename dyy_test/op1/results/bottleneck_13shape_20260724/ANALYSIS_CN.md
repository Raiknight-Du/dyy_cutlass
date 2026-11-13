# M_72a 13 个 shape 最优配置性能瓶颈分析

日期：2026-07-24

## 1. 分析对象与口径

分析对象为 `dyy_test/op1/op1_best.cu` 中 13 个 shape 各自搜索得到的
pure matmul 最优配置。性能基线严格采用 CUDA Event：

- 固定随机输入数据；
- warmup = 1；
- 每个样本 iterations = 10；
- 25 个样本；
- GPU 空闲时测试；
- NCU/NSYS 只用于诊断，不替代 Event 性能。

NCU 使用热缓存语义（`--cache-control none`）并关闭 profiler 时钟强控
（`--clock-control none`），与正式基线的运行方式一致。B200 的 L2 容量由
NCU 设备属性实测为 132,644,864 bytes，即约 126.5 MiB。

## 2. 总结论

13 个 shape 不能统一归因于“TMA 瓶颈”，实际分成四类：

1. shape1-4：Tensor Core/TMEM 吞吐已接近上限，主要受 UMMA 结果就绪和
   accumulator 从 TMEM 交给 epilogue 的依赖链限制；HBM 不是瓶颈。
2. shape5-6：小网格放大单 CTA 的 TMA/accumulator readiness。缩小
   tile 虽增加 grid waves，但 Auto stage 会继续吃满 SMEM，固定
   512-column TMEM allocation 也使第二个 CTA 无法驻留，因此只能顺序
   执行更多 CTA；增大 tile 又会恶化尾波或压缩 stage。
3. shape7-9：较短 K 流水和 L2 容量压力共同作用，属于 readiness
   latency 与缓存压力的混合瓶颈。尤其 shape8 已有 6.92 waves 和
   91.9% 尾波利用率，不能再简单归因于 grid underfill。
4. shape10-13：都只有 1 个 K iteration，但不能合并判断。shape10 的
   258.5 MiB 最小工作集超过 L2，DRAM SOL 达 78.36%，是 HBM 带宽型；
   shape11-13 的数据大部分可驻留 L2；shape11/13 的 K=128 只有一个
   iteration，TMA-to-SMEM、SF SMEM-to-TMEM、UMMA/TMEM accumulator
   交接和 D epilogue 形成串行 readiness 链，且第二 CTA 无法填泡。

`No Eligible` 很高是重要现象，但不是独立根因。所有最优配置的 SMEM
都把理论驻留限制在 1 CTA/SM；warp-specialized kernel 中仅部分 warp
在某阶段活跃，因此 achieved active warps/SM 约 6.2-6.8，并且每个
scheduler 平均只有 0.115-0.203 个 eligible warp。真正的根因必须结合
SOL、工作集、wave 数和 PC/SASS 等待位置判断。

## 3. 最优配置与主瓶颈

| Shape | 最优配置 | Event ms | TFLOP/s | 主瓶颈 | 次要限制 |
|---:|---|---:|---:|---|---|
| 1 | M256 N256 K256 E64x64 C2x1x1 | 3.5261 | 6828.8 | Tensor/TMEM 吞吐与 accumulator readiness | 单 CTA/SM、epilogue 等待 |
| 2 | M256 N256 K256 E64x64 C2x1x1 | 11.3787 | 6348.5 | 同 shape1 | 长时运行的频率/功耗敏感性 |
| 3 | M256 N256 K256 E64x64 C2x1x1 | 1.2943 | 6890.1 | Tensor/TMEM 吞吐与 accumulator readiness | L2/TMA readiness |
| 4 | M256 N256 K256 E64x64 C2x1x1 | 3.8672 | 6918.4 | 最接近计算侧上限 | accumulator-to-epilogue 等待 |
| 5 | M256 N192 K256 Eauto C2x2x1 | 0.01537 | 3353.4 | 小网格放大单 CTA 的 acc/TMA readiness | 缩 tile 重复 A/SFA；增大 tile 损失 stage/尾波 |
| 6 | M256 N192 K256 E128x16 C4x1x1 | 0.02360 | 4368.5 | 仅 3.46 waves，尾波利用率低 | accumulator readiness |
| 7 | M256 N192 K256 E128x16 C2x1x1 | 0.04280 | 4816.3 | readiness latency 与 L2 容量压力 | 6.92 waves 仍有限 |
| 8 | M256 N256 K256 E64x64 C2x1x1 | 0.03168 | 4338.2 | 8 次 K-loop 的 SMEM/TMEM readiness | 单 CTA/SM、prologue/epilogue |
| 9 | M256 N256 K256 E64x64 C2x1x1 | 0.05648 | 4866.5 | 短 K 流水与 L2 容量压力 | readiness latency |
| 10 | M128 N256 K128 Eauto C1x2x1 | 0.04503 | 763.0 | HBM 带宽/数据搬运 | 单 K-tile 固定开销 |
| 11 | M128 N192 K128 E128x64 C1x1x1 | 0.00890 | 362.0 | 单 K-tile 的 SMEM->TMEM->epilogue 串行链 | TMEM 独占阻止第二 CTA |
| 12 | M128 N192 K128 E128x64 C1x1x1 | 0.01300 | 495.4 | 单 K-tile 固定开销 | 6.92 waves、epilogue |
| 13 | M128 N256 K128 E128x128 C1x1x1 | 0.01499 | 573.1 | 单 K-tile mainloop/TMEM/epilogue readiness | D/cache 服务、TMEM/寄存器单 CTA |

## 4. 关键 NCU 证据

`Long SB` 单位是“每条已发射指令对应的平均 stalled warps”，不是百分比。
`No Eligible` 取 Scheduler Statistics 中没有指令可发射的周期比例。

| Shape | K iter | Stage | Waves | Tensor % | TMEM % | DRAM SOL % | L2 hit % | No Eligible % | Long SB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 5 | 485.27 | 87.71 | 91.25 | 23.08 | 80.89 | 89.07 | 9.67 |
| 2 | 20 | 5 | 1455.08 | 87.59 | 91.32 | 23.30 | 81.92 | 89.13 | 9.77 |
| 3 | 20 | 5 | 179.73 | 87.00 | 90.57 | 22.41 | 81.56 | 89.09 | 9.67 |
| 4 | 20 | 5 | 538.92 | 88.17 | 91.68 | 22.46 | 83.03 | 89.00 | 9.60 |
| 5 | 16 | 6 | 1.73 | 44.27 | 45.46 | 1.02 | 98.54 | 88.44 | 8.41 |
| 6 | 16 | 6 | 3.46 | 56.01 | 57.43 | 6.10 | 97.13 | 86.72 | 6.33 |
| 7 | 16 | 6 | 6.92 | 62.59 | 65.01 | 39.02 | 73.73 | 88.80 | 9.18 |
| 8 | 8 | 5 | 6.92 | 55.87 | 60.33 | 25.70 | 90.35 | 83.99 | 4.35 |
| 9 | 8 | 5 | 13.84 | 61.46 | 67.19 | 34.33 | 66.15 | 83.95 | 4.52 |
| 10 | 1 | 5 | 27.68 | 8.90 | 13.42 | 78.36 | 18.51 | 85.49 | 4.98 |
| 11 | 1 | 8 | 3.46 | 4.08 | 7.09 | 0.85 | 96.76 | 82.20 | 3.89 |
| 12 | 1 | 8 | 6.92 | 5.68 | 9.97 | 3.63 | 97.13 | 81.35 | 4.02 |
| 13 | 1 | 5 | 6.92 | 6.53 | 10.26 | 31.12 | 93.48 | 85.26 | 5.28 |

所有配置的 SMEM 为 206.8-227.3 KiB/CTA，NCU
`launch__occupancy_limit_shared_mem = 1`，因此都只能驻留 1 CTA/SM。

### 4.1 小/大 tile 与低 stage 反证

为避免把高 `No Eligible` 或较少 waves 直接当成根因，另用 pure matmul
对 shape5/8/11/13 扫描了更小 N tile、更大 N/K tile，并对 shape11/13
强制 Stage=1/2。下表中的 Event 均为与正式基线相同的 25 样本口径：

| Shape | 对照配置 | Waves | Event | 相对该 shape 最佳 | No Eligible | Long SB | 结论 |
|---:|---|---:|---:|---:|---:|---:|---|
| 5 | N64 K256 | 5.19 | 22.886 us | +49.9% | 84.84% | 7.32 | waves 增加，但 A/SFA 与 CTA 路径增至 3 倍 |
| 5 | N192 K256（最佳） | 1.73 | 15.271 us | 0 | 88.41% | 8.31 | 等待较高但总工作最少 |
| 5 | N256 K256 | 1.30 | 19.224 us | +25.9% | 91.21% | 12.94 | 末波仅 29.7% SM 填充，stage 降至 4 |
| 5 | N256 K512 | 1.30 | 27.557 us | +80.4% | 95.46% | 29.56 | 单 stage 152 KiB，只能开 2 stages |
| 8 | N128 K256 | 13.84 | 35.617 us | +12.6% | 81.00% | 3.77 | waves 翻倍、等待下降，仍被重复输入/CTA 成本反超 |
| 8 | N256 K256（最佳） | 6.92 | 31.621 us | 0 | 84.02% | 4.33 | 已有 91.9% 尾波利用率 |
| 8 | N256 K128 | 6.92 | 35.728 us | +13.0% | 88.68% | 9.26 | K-loop/barrier 从 8 轮增至 16 轮 |
| 8 | N256 K512 | 6.92 | 46.178 us | +46.0% | 94.70% | 25.43 | stage 从 5 降至 2，预取距离不足 |
| 11 | N64 K128 | 10.38 | 10.961 us | +23.0% | 77.04% | 3.02 | No Eligible 下降，但 A/SFA 与 CTA 数增至 3 倍 |
| 11 | N192 K128（最佳） | 3.46 | 8.915 us | 0 | 82.11% | 3.96 | 单 K-tile 串行链最短 |
| 11 | N256 K128 | 2.59 | 9.965 us | +11.8% | 85.23% | 5.04 | 更少 CTA 不能弥补 tile/尾波代价 |
| 13 | N64 K128 | 27.68 | 21.090 us | +40.1% | 75.99% | 3.09 | CTA 增至 4 倍，重复输入和缓存流量上升 |
| 13 | N256 K128（最佳） | 6.92 | 15.052 us | 0 | 85.23% | 5.30 | N 已达该 atom 合法上限 |
| 13 | M256 N256 K128 C2x1x1 | 6.92 | 15.356 us | +2.0% | 84.68% | 4.23 | 2SM atom 仍是 CTA-local M128，block 数不降 |
| 13 | N256 K256 | 6.92 | 16.036 us | +6.5% | 85.84% | 4.86 | K iteration 仍为 1，没有摊薄循环 |

Auto stage 会用缩小 tile 省出的 SMEM继续增加 stages，因此上述配置仍为
1 CTA/SM。更关键的是，当前
`sm100_gemm_tma_warpspecialized.hpp` 的 TMEM allocator 为每个 CTA 固定
申请全部 512 columns。shape11 强制 Stage=1/2 后，SMEM 分别降到
59.4/82.9 KiB，SMEM 与寄存器的理论 block limit 都达到 2，但实测
active CTA/SM 仍为 1.000，`No Eligible` 也未下降：

| Shape | Stage | Event | SMEM/CTA | SMEM block limit | Reg block limit | 实测 CTA/SM | No Eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | Auto=8 | 8.915 us | 224.3 KiB | 1 | 2 | 1.000 | 82.12% |
| 11 | 1 | 9.668 us | 59.4 KiB | 2 | 2 | 1.000 | 83.30% |
| 11 | 2 | 9.380 us | 82.9 KiB | 2 | 2 | 1.000 | 82.86% |
| 13 | Auto=5 | 15.052 us | 206.8 KiB | 1 | 1 | 1.000 | 85.22% |
| 13 | 1 | 15.759 us | 96.3 KiB | 1 | 1 | 1.000 | 86.27% |
| 13 | 2 | 15.442 us | 123.9 KiB | 1 | 1 | 1.000 | 85.75% |

shape13 还受 153 registers/thread 限制。由此可以把“无法提高并行度”
收紧为：小 tile 能增加整个网格的顺序 waves，却不能在同一 SM 上增加
可与当前 readiness wait 重叠的 CTA；低 stage 释放 SMEM 也不足以绕过
TMEM/寄存器驻留限制。

## 5. PC/SASS 与源码闭环

对 8 个唯一 kernel 变体分别采集了 SourceCounters；shape1 代表 shape1-4，
shape8 代表 shape8-9，shape11 代表 shape11-12。

各变体的 Long Scoreboard 热点均落在或依赖于：

```text
SYNCS.PHASECHK.TRANS64.TRYWAIT
```

shape1 的最大热点 PC 位于第一条 `UTMASTG` 之前，不能解释为 D-store
completion。它对应 epilogue 第一次消费 accumulator 前的等待：

```cpp
// include/cutlass/epilogue/collective/
// sm100_epilogue_tma_warpspecialized.hpp:864
if (is_first_iteration) {
  acc_pipeline.consumer_wait(acc_pipe_consumer_state, acc_wait_token);
}
```

mainloop 内部另有输入 stage 就绪等待：

```cpp
// include/cutlass/gemm/collective/
// sm100_blockscaled_mma_warpspecialized.hpp:1003
mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state, barrier_token);
```

源码中的完整责任链如下，行号以本次仓库版本为准：

| 层次 | 源码位置 | 操作与依赖 |
|---|---|---|
| Producer 取空 stage | `sm100_blockscaled_mma_warpspecialized.hpp:903` | `producer_acquire` 等待旧 stage 回收 |
| GMEM/L2 -> SMEM | 同文件 `:915-918` | 在 elected lane 发 A/B/SFA/SFB 四路 TMA |
| Consumer 等输入 | 同文件 `:1003`、`:1050` | `consumer_wait` 检查 TMA stage mbarrier |
| SF: SMEM -> TMEM | 同文件 `:1018-1019`、`:1065-1066` | UTCCP `copy` SFA/SFB，完成后 SF 才能参与 UMMA |
| Tensor Core | 同文件 `:1029-1034` 及后续循环 | A/B SMEM descriptor + SF TMEM -> accumulator TMEM |
| Epilogue 等 accumulator | `sm100_epilogue_tma_warpspecialized.hpp:864-866` | `acc_pipeline.consumer_wait` |
| TMEM -> register | 同文件 `:880-883` | LDTM 读取 accumulator |
| register -> SMEM D | 同文件 `:892-922` | 转 BF16 并写 epilogue SMEM |
| SMEM D -> GMEM | 同文件 `:928-930` | TMA store D |

第二 CTA 无法填泡还有直接的 allocator 依据：

```cpp
// include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp:727
tmem_allocator.allocate(
    TmemAllocator::Sm100TmemCapacityColumns,  // 512 columns
    ...);
```

这不是从低 achieved warps 反推“可能只有一个 CTA”，而是由
NCU 的实际 active CTA/SM=1.000、资源 limit 实验和固定 TMEM
allocation 三项共同证明。

PC 样本显示两类等待都存在，但长 K 配置中最大的单点是
accumulator/UMMA-to-epilogue readiness。因而正确结论是：

```text
Tensor/TMEM 计算与结果生产
        |
        v
accumulator pipeline TRYWAIT
        |
        v
TMEM load -> BF16 conversion -> SMEM -> TMA store D
```

不能只根据 Long Scoreboard 把根因写成“四路输入 TMA”。

## 6. 各 shape 独立分析

### Shape 1：170100 x 13824 x 5120

- 20 个 K iteration、485.27 waves，启动和尾波可忽略。
- SM/Tensor/TMEM 分别为 91.70%/87.71%/91.25%，已经是高利用率。
- DRAM SOL 仅 23.08%，L2 hit 80.89%，排除 HBM 饱和。
- 1 CTA/SM、No Eligible 89.07%、Long SB 9.67；PC 最大热点是
  accumulator pipeline 的 `TRYWAIT`。

主瓶颈是 Tensor/TMEM 结果生产吞吐及其到 epilogue 的就绪依赖。输入
TMA readiness 和单 CTA/SM 无法用第二 CTA 填泡是次要限制。

### Shape 2：510300 x 13824 x 5120

- 与 shape1 使用相同 kernel，20 个 K iteration、1455 waves。
- Tensor/TMEM 为 87.59%/91.32%，DRAM SOL 23.30%，结构性瓶颈与
  shape1 相同。
- 逻辑吞吐比 shape1 低约 7.0%。NSYS 单次 kernel 范围为
  10.65-12.30 ms，NCU 诊断时 GPC 时钟约 1.587 GHz。

主瓶颈仍是计算/TMEM readiness；额外损失更像长时运行下的频率、功耗
和运行波动，而不是缓存或 HBM 带宽变化。

### Shape 3：170100 x 5120 x 5120

- 20 个 K iteration、179.73 waves，仍足以形成稳定流水。
- Tensor/TMEM 为 87.00%/90.57%，DRAM SOL 22.41%。
- 最后一波只有 72.97% block 利用率，但总计约 180 waves，摊销后很小。

主瓶颈与 shape1 相同，为 Tensor/TMEM throughput 和 accumulator
readiness；较低 L2 SOL 不是主要限制。

### Shape 4：510300 x 5120 x 5120

- 20 个 K iteration、538.92 waves，尾波利用率 91.89%。
- 13 个 shape 中计算侧利用率最高：SM 92.19%、Tensor 88.17%、
  TMEM 91.68%；逻辑吞吐 6918.4 TFLOP/s。
- DRAM SOL 仅 22.46%。

这是最接近当前实现计算侧上限的 shape。主要剩余时间在 UMMA/TMEM
生产和 accumulator-to-epilogue 依赖链，而非外存。

### Shape 5：8192 x 768 x 4096

- 16 个 K iteration，但最优 N192/C2x2 只有 256 blocks、1.73 waves；
  Tensor/TMEM 为 44.27%/45.46%，L2 hit 98.54%，DRAM SOL 1.02%。
- 专项复测中，N64 把 waves 提高到 5.19、`No Eligible` 从 88.41%
  降到 84.84%，却由 15.271 us 退化到 22.886 us。因为 N tile 数从
  4 增到 12，每个 M tile 对应的 A/SFA 搬运、CTA 初始化、四路 TMA、
  barrier 和 epilogue 都增加到 3 倍；更多 CTA 仍只能逐波顺序执行。
- N256 的 CTA 更少，但只有 1.30 waves，第二波仅 44/148 SM 被占用，
  Event 退化 25.9%；K512 使单 stage 输入 transaction 增至 152 KiB，
  stage 从 4 降至 2，Long SB 从 12.94 升到 29.56，反而更难预取。

卡点不是抽象的 launch 固定成本，而是每个常驻 CTA 内的两类
readiness：输入 TMA 完成后 mainloop 的 SMEM mbarrier wait，以及
UMMA 结果进入 TMEM 后 epilogue 首次消费 accumulator 的 `TRYWAIT`。
最大 Long SB PC 属于后者，成功后才进入 BF16 转换和 D epilogue。
Auto stage 与固定 512-column TMEM allocation 共同保证同一 SM 只有
一个 CTA，因此小 tile 增加的 grid waves不能填充当前 CTA 的等待；
大 tile 则受尾波和 stage 容量反噬。HBM 利用率很低，只能排除外存
带宽，并不意味着这段片上 readiness latency 可以消失。

### Shape 6：16384 x 768 x 4096

- 网格增加到 512 blocks，但仍只有 3.46 waves；最后一波 block
  利用率仅 45.95%。
- Tensor/TMEM 提升到 56.01%/57.43%，证明增加 waves 能明显改善利用率。
- L2 hit 97.13%、DRAM SOL 6.10%，数据路径主要命中缓存。

主瓶颈仍是 wave 数和 cluster4 调度粒度，尾波损失是直接证据；
accumulator/TMEM readiness 为次要限制。

### Shape 7：32768 x 768 x 4096

- 6.92 waves，Tensor/TMEM 提升到 62.59%/65.01%。
- 最小主数据工作集约 121.7 MiB，接近 126.5 MiB L2；实际 tile
  重复和并发会进一步施压，L2 hit 已降至 73.73%。
- DRAM SOL 39.02% 尚未饱和，但 No Eligible 88.80%、Long SB 9.18。

主瓶颈是 TMA/accumulator readiness latency，L2 容量压力使等待更难
掩盖；它是延迟与缓存压力混合型，而不是纯 HBM 带宽型。

### Shape 8：8192 x 4096 x 2048

- 最优 N256/K256 有 1024 blocks、6.92 waves，最后一波利用率 91.9%；
  Tensor/TMEM 为 55.87%/60.33%，L2 hit 90.35%，DRAM SOL 25.70%。
  因此这里不是设备网格未填满。
- N128 将 waves 翻倍到 13.84，并把 `No Eligible` 降约 3 个百分点，
  Event 仍由 31.621 us 退化到 35.617 us；A/SFA 重复量和 CTA/
  epilogue 实例数翻倍，而同一 SM 仍只有一个常驻 CTA。
- K128 把 mainloop 从 8 轮增加到 16 轮，TMA/barrier/UTCCP 控制次数
  增加，慢 13.0%；K512 虽只剩 4 轮，但单 stage 输入为 152 KiB，
  只能开 2 stages，`No Eligible` 94.70%、Long SB 25.43，慢 46.0%。

最优 K256 是控制轮数与预取距离的折中，但 8 轮仍不足以把
prologue、首次输入 stage wait、accumulator-to-epilogue wait 和尾部
store完全摊入稳态。SourceCounters 最大热点是 accumulator pipeline
的 `SYNCS.PHASECHK.TRANS64.TRYWAIT`，同时存在 mainloop SMEM stage 与
epilogue store barrier 热点。因此具体限制是 SMEM/TMEM readiness，
不是有限 waves 或 HBM 饱和；合法 tile 扫描两侧都已观察到明确退化。

### Shape 9：16384 x 4096 x 2048

- 与 shape8 使用相同 kernel，waves 翻倍到 13.84，Tensor/TMEM
  提升到 61.46%/67.19%，说明 shape8 确有并行度不足。
- 最小工作集约 150.5 MiB，超过 L2；L2 hit 从 shape8 的 90.35%
  降到 66.15%，DRAM SOL 升到 34.33%。

主瓶颈是 8-iteration 短流水与 L2 容量不足的混合效应。增加 M 改善
wave 填充，但同时破坏缓存驻留，因此仍未达到长 K shape 的利用率。

### Shape 10：32768 x 4096 x 128

- 只有 1 个 K iteration，但有 27.68 waves，不能归因于网格过小。
- 最小主数据工作集约 258.5 MiB，明显超过 126.5 MiB L2；L2 hit
  仅 18.51%。
- DRAM SOL 78.36%，实测 DRAM 总带宽约 6.01 TB/s；Tensor 仅 8.90%。
- 最小算术强度仅约 126.7 FLOP/byte，远低于长 K shape。

这是 13 个 shape 中唯一证据明确的 HBM 带宽主导场景。单 K-tile 的
prologue/epilogue 固定成本是次要限制，但减少计算等待不能突破主要
的数据搬运下界。

### Shape 11：16384 x 768 x 128

- K=128 只有一个 iteration，512 blocks、3.46 waves；L2 hit 96.76%、
  DRAM SOL 0.85%，输入大部分为 L2-hot。
- N64 将 waves 提高到 10.38，并把 `No Eligible` 从 82.11% 降到
  77.04%，但 Event 慢 23.0%。N tile 数从 4 增至 12，A/SFA 与整条
  CTA 固定路径增加 3 倍；更大的 N256 又因 waves/尾波恶化而慢 11.8%。
- 强制 Stage=1/2 后，SMEM 降至 59.4/82.9 KiB，SMEM 和寄存器理论上
  都允许 2 blocks；NCU 实测 active CTA/SM 仍为 1.000，Event 分别
  慢 8.4%/5.2%，`No Eligible` 没有下降。原因是 kernel 的 TMEM
  allocator 为每 CTA 固定申请完整 512 columns。

这个 shape 没有稳态 mainloop，单 CTA 必须串行经过：

```text
L2-hot GMEM -> A/B/SFA/SFB TMA -> SMEM mbarrier wait
-> SFA/SFB UTCCP: SMEM -> TMEM
-> UMMA: A/B(SMEM) + SF(TMEM) -> accumulator(TMEM)
-> accumulator TRYWAIT -> LDTM -> register/BF16 -> SMEM D -> TMA store
```

没有前一轮 UMMA 可掩盖首块输入，也没有下一轮 TMA 与当前计算重叠；
第二 CTA 又被 TMEM allocation 阻止。故主瓶颈是 SMEM/TMEM/epilogue
串行 readiness 链，而不是泛化的“固定开销”或 HBM 带宽。

### Shape 12：32768 x 768 x 128

- 与 shape11 使用相同 kernel，M 翻倍使 waves 从 3.46 增至 6.92。
- 时间只从 8.90 us 增至 13.00 us，而工作量翻倍，吞吐从 362.0
  提升至 495.4 TFLOP/s，直接证明 shape11 的 underfill。
- 约 50.3 MiB 工作集仍可驻留 L2，L2 hit 97.13%、DRAM SOL 3.63%。

主瓶颈仍是单 K-tile 固定流水成本；比 shape11 多出的 waves 改善了
摊销，但仍不足以形成长 K 的稳定计算流水。

### Shape 13：8192 x 4096 x 128

- K=128 只有一个 iteration，但最优 N256 已有 1024 blocks、
  6.92 waves 和 91.9% 尾波利用率，因此并非 grid underfill。
- N64/N128/N192 分别把 waves 增至 27.68/13.84/9.51，却比 N256 慢
  40.1%/26.8%/13.8%。较小 N tile 重复 A/SFA、增加 TMA transaction
  和 CTA/epilogue 数量；N64 虽把 `No Eligible` 降约 9 个百分点，
  DRAM SOL 却升至 41.85%。
- N=256 已是该 block-scaled atom 的 CTA N 合法上限。M256/C2 的 2SM
  atom 仍拆成 CTA-local M128xN256，block 数不变，实测慢 2.0%；
  K256 仍只有一次 K iteration，不能减少控制轮数，实测慢 6.5%。
- Stage=1/2 后仍只有 1 CTA/SM：除固定 512-column TMEM allocation
  外，该变体 153 registers/thread 本身也把 block limit 限为 1。

最大 Long SB PC 位于 mainloop barrier 的 `SYNCS...TRYWAIT`，成功后
才从 SMEM 进入下一数据路径；其他热点在 SFA/SFB 的 UTCCP 邻近等待、
accumulator wait 和 epilogue fence。故主限制是单 K tile 的首块
TMA-to-SMEM readiness、SF/accumulator 的 TMEM 交接及 D epilogue
串行链。输出 D 约 64 MiB，DRAM SOL 约 32% 和 L1/L2 压力是重要次因，
但尚无 HBM 饱和证据。更小 tile 增加并行数量的收益被重复数据和控制
成本反超，更大 tile 则已碰到 atom、TMEM 和寄存器边界。

## 7. 最终瓶颈判定矩阵

| Shape | Tensor/TMEM 吞吐 | Readiness/No Eligible | Wave/尾波 | L2 容量 | HBM 带宽 |
|---:|---|---|---|---|---|
| 1 | 主 | 次 | 否 | 否 | 否 |
| 2 | 主 | 次 | 否 | 否 | 否 |
| 3 | 主 | 次 | 否 | 否 | 否 |
| 4 | 主 | 次 | 否 | 否 | 否 |
| 5 | 次 | 主 | 主 | 否 | 否 |
| 6 | 次 | 次 | 主 | 否 | 否 |
| 7 | 次 | 主 | 次 | 次 | 否 |
| 8 | 次 | 主 | 否 | 否 | 否 |
| 9 | 次 | 主 | 次 | 次 | 否 |
| 10 | 否 | 次 | 否 | 主 | 主 |
| 11 | 否 | 主 | 次 | 否 | 否 |
| 12 | 否 | 次 | 主 | 否 | 否 |
| 13 | 否 | 主 | 否 | 次 | 否 |

这里的“主/次”只描述当前最优配置的主要暴露成本，不表示某个高 stall
指标可以单独作为根因。

## 8. 数据与可复现文件

远程目录：

```text
/home/ubuntu/workspace/dyy/dyy_cutlass/
  dyy_test/op1/results/bottleneck_13shape_20260724/
  dyy_test/op1/results/bottleneck_shape5_8_11_13_20260724/
```

其中包含：

- `shapeXX_core.ncu-rep`：13 个 shape 的核心 NCU 报告；
- `shapeXX_source.ncu-rep`：8 个唯一 kernel 变体的 PC/SASS 报告；
- `shapeXX_nsys.nsys-rep`：13 个 shape 的 NSYS 报告；
- `ncu_core_summary.csv`：统一指标底表；
- `source_counter_top_pcs.txt`：热点 PC 与指令上下文；
- `collect.py`、`collect_source.py`、`summarize.py`：复测脚本。

第二个目录是 shape5/8/11/13 的 tile/stage 因果实验，包含：

- `op1_bottleneck_probe.cu`：N/K tile 对照源码；
- `op1_stage_probe.cu`：Stage=1/2 对照源码；
- `comparison_summary.csv`：Event 与 NCU 统一汇总；
- `resource_sizes.csv`：mainloop/epilogue SMEM 分项；
- `ncu/`、`ncu_stage/`：完整 NCU 报告。
