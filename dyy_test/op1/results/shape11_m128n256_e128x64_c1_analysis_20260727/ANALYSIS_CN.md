# Shape11 `M128_N256_K128_E128x64_C1x1x1` 分析

## 1. 结论

目标配置合法，但不优于当前 Shape11 best：

```text
Problem = M16384 x N768 x K128
Target  = M128 x N256 x K128, E128x64, C1x1x1
Best    = M128 x N192 x K128, E128x64, C1x1x1
```

40 轮随机交错配对测试结果：

| 配置 | Mean | 95% CI | 相对 best |
|---|---:|---:|---:|
| N192/E64 best | 8.264528 us | +/-0.003450 us | 0 |
| N256/E64 target | 9.260624 us | +/-0.004219 us | +12.05% |
| N256/E128 control | 9.259824 us | +/-0.004626 us | +12.04% |

N256 的 CTA 数虽然减少 25%，但每 CTA 的输入、UMMA 和输出工作变宽。
对这个只有一个 K iteration 的短 kernel，减少总 CTA 数没有形成有效收益，
反而暴露了更长的单 CTA readiness 路径和 D-store 尾部。

相对 N192 best 的约 `+0.994 us` 可拆为：

| 来源 | 增量 | 占总退化 |
|---|---:|---:|
| 输入 TMA/UMMA 所在 mainloop 临界区间 | +0.475896 us | 47.9% |
| D TMA store 路径 | +0.508560 us | 51.2% |
| TMEM load、BF16 转换、SMEM staging | +0.042696 us | 4.3% |
| 空 shell/调度底座 | -0.033368 us | -3.4% |
| UTCCP | +0.000072 us | 约 0 |

因此目标配置的退化不是 HBM/L2 总字节增加，而是：

> N256 增大了单 CTA 的 first-tile readiness 与输出 store pipeline
> 临界路径；K iteration 只有 1，Auto Stage7 没有后续 K tile 可预取，
> 1 CTA/SM 又没有第二条独立 CTA 工作链填补等待。

## 2. 分块、网格与尾波

两个配置都完整覆盖 N，不含边界 tile：

| 项目 | N192 best | N256 target |
|---|---:|---:|
| 每 CTA 输出 | 128x192 | 128x256 |
| M tile 数 | 128 | 128 |
| N tile 数 | 4 | 3 |
| CTA 数 | 512 | 384 |
| NCU waves | 3.46 | 2.59 |
| ceil(waves) | 4 | 3 |
| 平均 wave 填充率 | 86.49% | 86.49% |

N256 的 CTA 数下降 25%，但每 CTA 输出元素增加 33.3%。两者理论上正好
抵消，总 FLOPs 与总 D 元素完全不变。两个配置的平均 wave 填充率也相同，
所以 12.05% 退化不能归因于尾波比例。

## 3. SMEM 与 Auto Stage

静态 `sizeof` 探测和 NCU LaunchStats：

| 配置 | Auto Stage | Mainloop SharedStorage | Epilogue SharedStorage | Kernel SharedStorage | NCU SMEM/CTA | Reg/thread |
|---|---:|---:|---:|---:|---:|---:|
| N192/E64 | 8 | 188544 B | 34816 B | 223232 B | 224256 B | 105 |
| N256/E64 | 7 | 193664 B | 34816 B | 228352 B | 229376 B | 92 |
| N256/E128 | 5 | 138368 B | 67584 B | 205824 B | 206848 B | 153 |

NCU 比 CUTLASS `Kernel::SharedStorage` 多出的 1024 B 是 driver/block
保留空间。三个配置均超过双 CTA/SM 的 SMEM 边界，实际限制为 1 CTA/SM。

### 3.1 N256/E64 每 stage 输入

每个真实 K tile 的 SMEM payload 为 27 KiB：

```text
A   = 128 * 128 * 0.5 B = 8 KiB
B   = 128 * 256 * 0.5 B = 16 KiB
SFA = 128 * 128 / 16 * 1 B = 1 KiB
SFB = 128 * 256 / 16 * 1 B = 2 KiB
Total                         = 27 KiB
```

因此：

```text
Mainloop tensor storage = 7 * 27 KiB = 189 KiB
```

E128x64 的 D staging 使用两个 stage：

```text
2 * 128 * 64 * sizeof(BF16) = 32 KiB
```

加 pipeline/barrier/padding 后，epilogue SharedStorage 为 34 KiB。

### 3.2 为什么 Stage7 没有隐藏更多延迟

Shape11 的 `K=TileK=128`，每 CTA 只有一个真实 K iteration。Stage7
表示可复用的七槽 SMEM ring buffer，不表示这里有七个 K tile：

```text
真实使用：stage0
未装入后续数据：stage1 ... stage6
```

所以 N256/E64 从 E128 控制组的 Stage5 增至 Stage7，只增加资源余量，
不能扩大真实预取距离，也不能隐藏 first-tile 的四路 TMA readiness。

## 4. 数据与指令量

同批 NCU：

| 指标 | N192/E64 | N256/E64 | 变化 |
|---|---:|---:|---:|
| Grid CTA | 512 | 384 | -25.0% |
| Input TMA bytes | 12.059 MB | 10.617 MB | -12.0% |
| Input TMA inst | 2048 | 1536 | -25.0% |
| D TMA bytes | 25.166 MB | 25.166 MB | 不变 |
| D TMA inst | 1536 | 1536 | 不变 |
| Tensor pipe inst | 1024 | 768 | -25.0% |
| TMEM pipe inst | 6144 | 6144 | 不变 |
| SM inst | 1.071 M | 0.904 M | -15.5% |
| L2 bytes | 50.308 MB | 48.979 MB | -2.6% |

N256 确实增加了 A/SFA 在 N 方向的复用：A/SFA 不再为第四个 N tile
重复加载，所以总 input TMA bytes 下降约 12%。但这不是带宽饱和问题：

- 每 CTA 仍发 A/B/SFA/SFB 四路 TMA；
- 每 CTA 的 stage payload 从约 23 KiB 增至 27 KiB；
- N256 UMMA 单条指令覆盖的 N 工作增加；
- 只有一个 K iteration，first-tile latency 无稳态流水可覆盖。

因此总字节减少与单 CTA readiness 变长可以同时发生。

### 4.1 输出路径没有随 CTA 数减少

E128x64 下：

```text
N192: 3 个 64-column epilogue subtile/CTA
N256: 4 个 64-column epilogue subtile/CTA
```

对应 NCU：

```text
D TMA:
512 CTA * 3 = 1536
384 CTA * 4 = 1536

TMEM pipe:
512 CTA * 12 = 6144
384 CTA * 16 = 6144
```

所以 N256 减少 CTA 并没有减少总 D TMA 或 TMEM pipe 工作。更重要的是，
每个 resident CTA 必须顺序推进四个 consumer subtile。D staging 只有两个
stage，后两个 subtile 需要等待前面的 TMA store 释放槽位；在 1 CTA/SM
下，这种 stage 回收等待更容易进入关键路径。

## 5. 目标配置消融

所有变体均为 pure matmul、相同随机输入、40 轮随机交错、每轮 100 次：

| 变体 | 移除内容 | Mean |
|---|---|---:|
| Full | 无 | 9.257080 us |
| NO-TC | UMMA | 9.257248 us |
| NO-TMA | A/B/SFA/SFB TMA | 9.257088 us |
| Shell+UTCCP | 输入 TMA + UMMA | 8.771680 us |
| Shell-noUTCCP | 再移除 UTCCP | 8.771536 us |
| Shell-noDStore | 再移除 D TMA store | 6.216872 us |
| Shell-noEpilogue | 再移除 TMEM load/转换/SMEM staging | 4.095200 us |

暴露时间：

| 对照 | 暴露时间 |
|---|---:|
| Full - Shell | 0.485400 us |
| 无 MMA 时加入输入 TMA | 0.485568 us |
| 无输入 TMA 时加入 UMMA | 0.485408 us |
| 有 MMA 时加入输入 TMA | -0.000008 us |
| 有输入 TMA 时加入 UMMA | -0.000168 us |
| UTCCP | 0.000144 us |
| D store | 2.554664 us |
| TMEM load/转换/SMEM staging | 2.121672 us |

不能把 TMA 和 UMMA 的 `0.485 us` 相加。四象限表明它们在 kernel-wide
关键路径上的边际贡献不相加：只保留任意一项时都占据约 0.485 us，
两项同时存在时仍只延长约 0.485 us。该结果说明它们落在同一 readiness/
调度临界区间，不表示同一 CTA 中依赖当前 TMA 数据的 UMMA 可以提前执行。

与 N192 best 的高精度消融对比：

| 组件 | N192/E64 | N256/E64 | 增量 |
|---|---:|---:|---:|
| Mainloop critical interval | 0.009504 us | 0.485400 us | +0.475896 us |
| UTCCP | 0.000072 us | 0.000144 us | +0.000072 us |
| D store | 2.046104 us | 2.554664 us | +0.508560 us |
| TMEM load/convert/SMEM | 2.078976 us | 2.121672 us | +0.042696 us |
| Shell-noEpilogue | 4.128568 us | 4.095200 us | -0.033368 us |

增量之和为 `0.993856 us`，与两批 Full 的差值完全一致。

## 6. Scheduler 与 Long Scoreboard

Full 的 NCU 调度指标：

| 指标 | N192/E64 | N256/E64 |
|---|---:|---:|
| Active warps/SMSP | 1.662 | 1.641 |
| Eligible warps/SMSP | 0.192 | 0.162 |
| No Eligible | 82.21% | 84.84% |
| Long Scoreboard / issue | 3.898 | 5.040 |
| Barrier / issue | 0.873 | 1.144 |
| Wait / issue | 1.592 | 1.721 |

N256 的 eligible warp 下降约 15.7%，No Eligible 增加 2.63 pp，Long
Scoreboard 增加约 29.3%。这与消融中 first-tile/mainloop 暴露和更长
epilogue store pipeline 一致。

SourceCounters 将目标配置的 Long Scoreboard 主要定位到：

| 文件位置 | 长记分牌样本 | 含义 |
|---|---:|---|
| `cutlass/arch/barrier.h:416-422` | 106 | `mbarrier.try_wait` 自旋等待 |
| `cutlass/arch/barrier.h:372/732` | 30 | Cluster/mbarrier wait 包装路径 |
| `sm100_pipeline.hpp:1046` | 15 | `producer_tail` 等 stage empty |
| `sm100_epilogue_tma_warpspecialized.hpp:768` | 3 | D TMA 前 async shared fence |
| `sm100_epilogue_tma_warpspecialized.hpp:866` | 2 | 等 accumulator/TMEM 就绪 |
| `sm100_epilogue_tma_warpspecialized.hpp:903` | 1 | 延迟发起前一 subtile 的 D TMA |

最高样本落在通用 `mbarrier.try_wait`，不能仅凭该行区分输入 TMA、
epilogue store 或 pipeline tail；但消融已经给出组件级归因，SourceCounters
进一步确认实际暴露位置是 readiness/barrier 等待，而不是算术指令吞吐。

## 7. E128x64 与 E128x128

N256 下两者事件时延只差 `0.0008 us`，统计上相同。E128x128 虽减少
consumer 指令，但没有减少最终 D store：

| 指标 | E128x64 | E128x128 |
|---|---:|---:|
| Stage | 7 | 5 |
| SMEM/CTA | 229376 B | 206848 B |
| Registers/thread | 92 | 153 |
| D TMA inst | 1536 | 1536 |
| TMEM pipe inst | 6144 | 3072 |
| SM inst | 0.904 M | 0.817 M |
| Eligible warps/SMSP | 0.162 | 0.158 |
| No Eligible | 84.84% | 85.37% |

E128x128 将四个 64-column consumer subtile 合为两个逻辑 subtile，
TMEM pipe 指令减半；但底层 D store 仍按 64-column copy 事务拆为四次，
同时寄存器增加 66% 且 eligible warp 更少。减少的指令不在关键路径上，
所以整体时延不变。

## 8. 最终判断

`M128_N256_K128_E128x64_C1x1x1` 的问题不是数据复用不足：

- 它已经将 input TMA bytes 减少约 12%；
- 总 L2 bytes 也略有下降；
- 总 SM 指令减少约 15.5%。

真正的性能损失来自更宽单 CTA 在短 K 下的延迟隐藏恶化：

1. 只有一个 K iteration，Stage7 无法形成稳态预取；
2. 每 CTA 输入 stage 增至 27 KiB，N256 UMMA 工作变宽；
3. 每 CTA epilogue 从三个增至四个 E64 subtile；
4. D-store 两级 buffer 需要更多回收周转；
5. 1 CTA/SM 下没有第二 CTA 填补 mbarrier/TMEM/D-store 等待；
6. NCU 直接观察到 eligible warp 下降、No Eligible 和 Long
   Scoreboard 上升。

因此 Shape11 应继续保留：

```text
M128_N192_K128_E128x64_C1x1x1
```

N256 减少 CTA 和输入字节的收益，无法覆盖单 CTA readiness 与输出尾部
增加的成本。
