# M_72a pure matmul Shape11 性能瓶颈分析

## 1. 结论

Shape11 的主瓶颈不是输入 TMA，也不是 UMMA 算力，而是小问题规模下无法摊薄的
输出路径与资源驻留底座：

1. `TMEM -> register -> BF16 -> SMEM` 及其同步暴露约 `2.079 us`；
2. D TMA store 暴露约 `2.046 us`；
3. 224256B/CTA 的 SMEM 将 kernel 限制为 1 CTA/SM，512 个 CTA 形成
   3.46 waves；同网格、同 SMEM 的空工作基线已经需要 `4.113 us`；
4. 输入 TMA 与 UMMA 在 E128x64 基线中合计只延长 `0.0095 us`，处于极小量级。

因此更准确的表述是：

> Shape11 只有一个 K iteration，不能建立稳态 mainloop。输入 TMA 和 UMMA
> 基本被 kernel shell 与输出路径覆盖；总时延主要由单 CTA/SM 下的多波
> block 分发底座，以及 accumulator 就绪后严格依赖的 epilogue 和 D store
> 决定。

## 2. 配置与理论量

```text
M = 16384
N = 768
K = 128

Tile = M128 x N192 x K128
Epilogue = E128 x 64
Cluster = 1 x 1 x 1
Stage = Auto = 8
```

每 CTA 负责 `128x192` 输出。网格共有：

```text
(16384 / 128) * (768 / 192) = 128 * 4 = 512 CTA
```

B200 有 148 SM，在 1 CTA/SM 下：

```text
waves = 512 / 148 = 3.459
四波平均 SM 利用率 = 512 / (4 * 148) = 86.49%
```

尾波只有 68 个 CTA，尾波填充率为 `68/148=45.95%`。尾波有损失，但四波
平均仍有 86.49%，不是首要瓶颈。

`K=128` 与 `TileK=128` 相同，因此每 CTA 只有一个 K iteration。Auto
分配的 8 stages 中只有一个 stage 有真实 K tile，额外 stage 不会形成
八轮预取距离。

每 CTA/stage 的输入约为 23 KiB：

```text
A    = 128 * 128 * 0.5 byte = 8 KiB
B    = 128 * 192 * 0.5 byte = 12 KiB
SFA/SFB + TMA layout padding ≈ 3 KiB
合计 ≈ 23 KiB
```

SMEM 约为：

```text
Mainloop = 8 * 23 KiB = 184 KiB
Epilogue = 2 * 128 * 64 * 2 byte + barrier/padding ≈ 33 KiB
NCU dynamic SMEM = 224.256 KB/CTA
```

NCU 给出的 SMEM block limit 为 1。基线还申请完整 512-column TMEM，
所以 CTA 内某个 warp group 等待时，没有第二个独立计算 CTA 填补空洞。

理论计算下界：

```text
FLOPs = 2 * 16384 * 768 * 128 = 3.221225472 GFLOP
T_compute @ 9 PFLOP/s = 0.358 us
```

基线 NCU 的 L2 流量为 51.585 MB：

```text
T_L2 @ 20 TiB/s ≈ 2.347 us
```

两者都显著小于约 8-9 us 的实测时延，且 DRAM/计算吞吐远未饱和。

## 3. 与 Shape5/8 同批的基础消融

下表保留原 Shape5/8/11/13 同批、25 样本、每样本 10 次的结果，用于横向比较：

| 变体 | 额外移除内容 | Mean | 相对前一相关变体减少 |
|---|---|---:|---:|
| Full | 无 | 8.997 us | - |
| NO-TC | UMMA | 8.905 us | 0.093 us（相对 Full） |
| NO-TMA | A/B/SFA/SFB TMA | 8.823 us | 0.174 us（相对 Full） |
| Shell+UTCCP | 输入 TMA + UMMA | 8.846 us | - |
| Shell-noUTCCP | 再移除 UTCCP | 8.826 us | 0.020 us |
| Shell-noDStore | 再移除 D TMA store | 6.811 us | 2.015 us |
| Shell-noEpilogue | 再移除 TMEM load、转换、SMEM staging | 4.451 us | 2.360 us |

这组数据已经表明：

- 输入 TMA 和 UMMA 的暴露量很小；
- UTCCP 暴露接近零；
- D store 和其余 epilogue 各暴露约 2 us；
- 大约一半时间仍留在无数据搬运、无计算、无输出转换的 shell 中。

## 4. 高精度四象限控制实验

为排除 0.1 us 量级差异中的计时波动，重新执行 40 组随机交错测试，每组
100 次 kernel。绝对时间受本次 GPU 状态和迭代长度影响，不与上一节直接
混用；因果判断只使用本组内部的配对差值。

### 4.1 E128x64 基线

| 变体 | Mean | 95% CI |
|---|---:|---:|
| Full | 8.263224 us | +/-0.003458 us |
| NO-TC | 8.261224 us | +/-0.003127 us |
| NO-TMA | 8.259312 us | +/-0.002943 us |
| Shell+UTCCP | 8.253720 us | +/-0.002274 us |
| Shell-noUTCCP | 8.253648 us | +/-0.002493 us |
| Shell-noDStore | 6.207544 us | +/-0.003014 us |
| Shell-noEpilogue | 4.128568 us | +/-0.004183 us |

四象限配对差值：

| 对照 | 暴露时间 | 95% CI |
|---|---:|---:|
| Full - Shell | 0.009504 us | +/-0.004461 us |
| 无 MMA 时加入输入 TMA | 0.007504 us | +/-0.003398 us |
| 无输入 TMA 时加入 UMMA | 0.005592 us | +/-0.003596 us |
| 有 UMMA 时加入输入 TMA | 0.003912 us | +/-0.004399 us |
| 有输入 TMA 时加入 UMMA | 0.002000 us | +/-0.005125 us |
| 移除 UTCCP | 0.000072 us | +/-0.002936 us |
| 移除 D store | 2.046104 us | +/-0.003806 us |
| 移除 TMEM load/转换/SMEM staging | 2.078976 us | +/-0.004978 us |

输入 TMA 与 UMMA 的边际时间均接近零，且有/无另一项时都没有形成显著
关键路径。D store 与其余 epilogue 合计暴露 `4.12508 us`，占 Full 的
`49.92%`。

## 5. 控制底座拆分

使用 512 blocks、256 threads/block 的独立微基线：

| 控制 kernel | Mean |
|---|---:|
| 空 kernel，0 SMEM | 2.603520 us |
| 空工作，224256B SMEM | 4.112512 us |
| 224256B SMEM + `__syncthreads()` | 4.124800 us |
| 4B SMEM + 512-column TMEM alloc/free | 4.155776 us |
| 224256B SMEM + 512-column TMEM alloc/free | 6.170496 us |
| CUTLASS Shell-noEpilogue | 4.128568 us |

最关键的匹配对照是：

```text
CUTLASS Shell-noEpilogue - 同网格同SMEM空工作
= 4.128568 - 4.112512
= 0.016056 us
```

因此不能把 Shell-noEpilogue 的约 4.13 us 全部解释成 CUTLASS pipeline、
scheduler 或 barrier 指令开销。它主要来自 224256B SMEM 导致的
1 CTA/SM 和 3.46 waves block 分发。

TMEM 微基线还给出两点：

- 4B SMEM 时，NCU 的理论 waves 为 0.43，但实际 active warps 仅约
  1.95/SMSP，说明完整 512-column TMEM 会限制实际并行推进；
- 同时使用 224256B SMEM 和完整 TMEM 时，朴素 alloc/free 微内核达到
  6.17 us，说明没有其他 resident CTA 时，TMEM allocation handoff
  可能暴露。

该 6.17 us 不能直接代入 CUTLASS Full：CUTLASS 在
`sm100_gemm_tma_warpspecialized.hpp:783` 提前执行
`release_allocation_lock()`，其生命周期与微内核不同。匹配的
Shell-noEpilogue 对照证明，CUTLASS 已把大部分 allocator 成本隐藏在
4-wave 底座中。

## 6. NCU 与 PC/source 证据

E128x64 Full 的三次 NCU 重复：

| 指标 | Mean | 范围 |
|---|---:|---:|
| NCU duration | 9.547 us | 9.536-9.568 us |
| Active warps/SMSP | 1.694 | 1.692-1.696 |
| Eligible warps/SMSP | 0.193 | 0.193-0.193 |
| No Eligible | 82.09% | 82.07-82.10% |
| Long Scoreboard / issue | 3.934 | 3.924-3.940 |
| Barrier / issue | 0.853 | 0.849-0.856 |
| Wait / issue | 1.592 | 1.592-1.592 |

每个 SM 平均只有约 `1.694*4=6.78` 个 active warps，平均 eligible warps
只有 `0.193*4=0.77`。warp-specialized CTA 中只有一部分 warp 在当前阶段
有工作，其余 warp 等待 TMA、accumulator、epilogue 或 scheduler barrier；
1 CTA/SM 下没有第二条独立 CTA 工作链填泡。

SourceCounters 将 Long Scoreboard 映射到以下源码：

| 位置 | 含义 | E64 Full 样本 |
|---|---|---:|
| `cutlass/arch/barrier.h:416-422` | `mbarrier.try_wait` 自旋等待 | 58 |
| `sm100_pipeline.hpp:1046` | `producer_tail` 等 stage 释放 | 14 |
| `sm100_epilogue_tma_warpspecialized.hpp:866` | 等 accumulator/TMEM 就绪 | 6 |
| `sm100_epilogue_tma_warpspecialized.hpp:768` | D TMA 前 async shared fence | 10 |
| `sm100_blockscaled_mma_warpspecialized.hpp:1073` | UMMA 路径 | 7 |

Shell-noEpilogue 已移除输入 TMA、UMMA、UTCCP、TMEM load、转换和 D store，
但 64 个 Long Scoreboard 样本中仍有 49 个落在
`barrier.h:416-422` 的 mbarrier wait loop。这说明：

- 高 Long Scoreboard 不等价于输入 TMA 慢；
- 当数据工作被移除后，warp 仍会在 pipeline 初始化、accumulator
  handshake、TMEM 回收和 scheduler 同步中等待；
- 这些 stall 比例在工作量很小的 kernel 中会被显著放大。

## 7. E128x96 控制：为什么更粗 epilogue 没有变快

E128x96 与 E128x64 只改变 epilogue tile。高精度结果：

```text
E128x64 Full = 8.263224 us
E128x96 Full = 8.392816 us
差值          = +0.129592 us，+1.57%
```

E128x96 并没有减少底层 consumer 工作：

| NCU 指标 | E128x64 | E128x96 | 变化 |
|---|---:|---:|---:|
| SMEM/CTA | 224.256 KB | 216.064 KB | -3.65% |
| Registers/thread | 105 | 148 | +41.0% |
| D TMA store inst | 1536 | 3072 | 2.00x |
| TMEM pipe inst | 6144 | 12288 | 2.00x |
| SM instructions | 1.071 M | 0.981 M | -8.4% |
| Eligible warps/SMSP | 0.193 | 0.183 | -5.3% |
| No Eligible | 82.09% | 83.32% | +1.23 pp |
| Barrier / issue | 0.853 | 1.007 | +18.1% |
| Wait / issue | 1.592 | 1.789 | +12.4% |

E128x64 对 `N=192` 生成三个 64-column subtile，每个 subtile 对应一次
D TMA，故 `512 CTA * 3 = 1536`。E128x96 虽然逻辑上只有两个 subtile，
但 96 列不能由当前 64-column TMEM-load/TMA-store copy atom 作为单次规则
事务完成，底层被进一步拆分，最终 D TMA 和 TMEM pipe 指令反而翻倍。

四象限进一步显示，E128x96 在只有 TMA 或只有 UMMA 时仍只暴露
`0.004-0.007 us`；两者同时存在时额外产生约 `0.128 us` 的交互项。
结合更高的 barrier/wait 和更低的 eligible warps，可判定该差异来自
E96 的 consumer 分解与 readiness 调度，而不是 epilogue 字节减少后
“更早进入一个原本已被隐藏的等待”。

因此 Shape11 保留 E128x64 是合理的。

## 8. Stage 与 tile 控制

已有纯 matmul Stage 控制：

| Stage | Event | Active CTA/SM | No Eligible |
|---:|---:|---:|---:|
| Auto=8 | 8.915 us | 1.000 | 82.12% |
| 1 | 9.668 us | 1.000 | 83.30% |
| 2 | 9.380 us | 1.000 | 82.86% |

减少 stage 虽然释放 SMEM，但完整 512-column TMEM 和 warp-specialized
资源链没有让第二个 CTA 成为可独立推进的计算链，实测 No Eligible
没有下降，时延反而上升。

N tile 控制也符合相同结论：

| N tile | Waves | Event | No Eligible | 说明 |
|---:|---:|---:|---:|---|
| 64 | 10.38 | 10.961 us | 77.04% | 等待下降，但 CTA 与 A/SFA 重复增至 3 倍 |
| 192 | 3.46 | 8.915 us | 82.11% | 当前最佳 |
| 256 | 2.59 | 9.965 us | 85.23% | 更少 CTA 未抵消 tile/尾波与 consumer 代价 |

降低 No Eligible 并不自动提高性能；增加 CTA 数、输入重复和固定开销可能
超过等待改善。

## 9. 最终瓶颈表述

Shape11 的 `M128_N192_K128_E128x64_C1x1x1` 只有一个 K iteration，
无法建立可持续的 TMA-UMMA mainloop。消融四象限证明输入 TMA 和 UMMA
基本不延长关键路径；相反，D store 与
`TMEM load -> register -> BF16 convert -> SMEM staging` 各暴露约 2 us，
合计约占总时延一半。

另一半并非单一 pipeline 指令：224256B/CTA 的 SMEM 与完整 TMEM 使
kernel 只能以一条 CTA 工作链推进，512 blocks 形成 3.46 waves。同网格、
同 SMEM 的空工作 kernel 已需要 4.11 us，几乎等于移除全部数据工作的
CUTLASS shell。SourceCounters 中高 Long Scoreboard 主要落在通用
mbarrier wait loop、stage tail 和 accumulator readiness wait，反映的是
单 CTA/SM、单 K iteration 下可运行 warp 不足，而不是 HBM、输入 TMA
或 Tensor Core 吞吐饱和。

因此 Shape11 的主要瓶颈可定稿为：

> 小 K 导致稳态 mainloop 缺失；单 CTA/SM 的资源驻留使 scheduler
> 无法用其他 CTA 隐藏 barrier/readiness 等待；输出 epilogue 与 D store
> 位于 accumulator 之后并形成约 4.1 us 的暴露尾部。尾波空闲是次要因素，
> 输入 TMA 和 UMMA 不是当前关键路径。

## 10. SMEM、增大 Tiling 与固定 Stage 补充

后续对 shape11 做了同批 pure-matmul 配对复测和静态 SharedStorage
探针。基线每 CTA 的 SMEM 为：

```text
Mainloop TensorStorage  184 KiB
Epilogue TensorStorage   33 KiB
Kernel control            1 KiB
CUDA driver reserved      1 KiB
总计                    219 KiB
```

增大 tiling 均未超过基线：

| 配置 | Event mean | 相对基线 |
|---|---:|---:|
| M128_N192_K128_E128x64_C1 | 8.2592 us | 基线 |
| M256_N192_K128_E128x64_C2 | 8.2882 us | +0.35% |
| M256_N256_K128_E128x64_C2 | 8.3198 us | +0.73% |
| M128_N256_K128_E128x128_C1 | 9.2601 us | +12.12% |

固定 Stage=1/2 虽将总 SMEM 降至 58/81 KiB，也没有提速：

| Stage | Event mean | 相对 Auto=8 | NCU SMEM block limit | 平均 Active CTA/SM |
|---:|---:|---:|---:|---:|
| Auto=8 | 8.2654 us | 基线 | 1 | 0.680 |
| 1 | 8.8900 us | +7.56% | 2 | 0.671 |
| 2 | 8.4741 us | +2.52% | 2 | 0.680 |

CUTLASS 的 MMA participant 对每个 CTA 调用
`tcgen05.alloc.cta_group::1` 申请完整 512 TMEM columns。因而 Stage1/2
释放 SMEM 后，第二 CTA 仍不能建立独立 TMEM accumulator 工作链；
静态 SMEM block limit 提升没有转化成实际 CTA 并发。大 tile 降低的主要
是已被覆盖的输入流量，同时没有减少 D-store transaction，故没有收益。

补充控制实验表明，Stage1/2 各增加 64 KiB 无访问 SMEM padding、强制回
单 block/SM 后，时延仅从 8.8900/8.4741 us 变为
8.8749/8.4622 us。因此 TMEM allocation 解释了“为何释放 SMEM 没有增加
并发”，但不是固定 Stage 退化的主要原因。

shape11 只有一个 K iteration，不应将退化概括为稳态 pipeline 周转。
Stage1 的直接问题是 producer 在发出当前 TMA 前无条件
`++state; producer_try_acquire(next_state)`；单 stage 会立即绕回当前
buffer 的下一 phase，提前检查尚未被 consumer 释放的 empty barrier。
Stage2 不存在该同槽位前视依赖。Stage2 与 Auto 的剩余小差距来自
StageCount 改变 barrier tail、SharedStorage 和编译后短 kernel 控制路径，
当前 PC-SASS 证据不足以归因于某一条 barrier 指令。

进一步扫描 Stage2 到 Stage8：

| Stage | SMEM/CTA | Event mean | 相对Auto8 | No Eligible |
|---:|---:|---:|---:|---:|
| 2 | 81 KiB | 8.4484 us | +2.28% | 82.66% |
| 3 | 104 KiB | 8.3453 us | +1.03% | 82.50% |
| 4 | 127 KiB | 8.3047 us | +0.53% | 82.43% |
| 5 | 150 KiB | 8.2944 us | +0.41% | 82.41% |
| 6 | 173 KiB | 8.2647 us | +0.05% | 82.31% |
| 7 | 196 KiB | 8.2588 us | -0.02% | 82.25% |
| 8 | 219 KiB | 8.2634 us | +0.04% | 82.12% |
| Auto8 | 219 KiB | 8.2605 us | 基线 | 82.11% |

因此 Stage2 到 Stage8 并非全部等价；Stage6 到 Stage8 才进入与 Auto8
统计等价的平台。变化仍不来自额外 K 预取，而与
`producer_tail` 对 `Stages` 个 empty barrier 的遍历、SharedStorage
地址布局及生成后的短 kernel 控制调度有关。显式 Stage8 与 Auto8 的
SMEM、SM指令数完全相同，验证 Auto 确实解析为 Stage8。

完整分解、NCU 指标和实验文件见：
`../shape11_smem_tiling_analysis_20260727/ANALYSIS_CN.md`。
