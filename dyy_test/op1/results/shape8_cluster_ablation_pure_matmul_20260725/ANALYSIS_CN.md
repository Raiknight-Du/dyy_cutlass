# M_72a shape8 cluster 扩大与消融分析

## 1. 结论

shape8 的输入 TMA 与 UMMA 计算都处于可暴露的关键路径上。当前
`ClusterShape=2x1x1` 的完整 pure-matmul 耗时为 `31.669 us`：

- 移除 UMMA 后为 `25.641 us`，计算路径暴露约 `6.028 us`；
- 移除输入 TMA 及其 mainloop barrier 后为 `25.578 us`，输入路径暴露约
  `6.091 us`。

两项差值几乎相同，说明 shape8 不是单独的“计算瓶颈”或“搬运瓶颈”，而是
TMA readiness 和 UMMA/TMEM 计算链都不能完全覆盖对方。

扩大 cluster 没有缩短 shape8：

| Cluster | Full | 相对 `2x1x1` | NO TC | NO TMA |
|---|---:|---:|---:|---:|
| `2x1x1` | 31.669 us | 0 | 25.641 us | 25.578 us |
| `2x2x1` | 33.594 us | +6.08% | 27.598 us | 29.224 us |
| `4x1x1` | 33.638 us | +6.22% | 27.560 us | 29.321 us |

`2x2x1` 和 `4x1x1` 的多播确实减少了源侧 L2 流量，但没有减少 TMA
指令数、目标 SMEM 写入量和 barrier 管理次数；四 CTA cluster 带来的调度与同步
代价超过了流量收益。因此当前 shape8 应保留 `2x1x1`。

## 2. 实验定义

所有变体都直接派生自当前 pure-matmul 源码，固定：

```text
M=8192, N=4096, K=2048
MmaTile=256x256x256
Epilogue=64x64
Stage=Auto，实际为 5
SMEM/CTA=214.016 KiB
Grid=32x32=1024 CTA
K rounds=8
随机持久化输入
```

三种 cluster：

| 配置 | 作用 |
|---|---|
| `2x1x1` | 2SM UMMA 基线，包含 atom 内部需要的 B/SFB 多播 |
| `2x2x1` | 在 N 方向扩展 cluster，增加 A/SFA 的跨 CTA 复用 |
| `4x1x1` | 在 M 方向扩展 cluster，扩大 B/SFB 的跨 CTA 复用域 |

Event 计时使用 25 个独立样本，每个样本 10 次迭代；NCU 在相同随机数据和
L2-hot 条件下采集。NO TC、NO TMA 和 shell 变体均保持相同 grid、tile、
epilogue 与 cluster。

消融差值表示某条路径在完整流水外暴露的时间，不等于被移除指令的孤立硬件延迟。

## 3. 为什么 shape8 的输入搬运会暴露

shape8 每个 CTA、每轮 K tile 的目标侧输入为：

| Operand | 每 CTA 每轮 |
|---|---:|
| A | 16 KiB |
| B | 16 KiB |
| SFA | 2 KiB |
| SFB | 4 KiB |
| 合计 | 38 KiB |

全 kernel 有 `1024 CTA x 8 rounds = 8192 CTA-rounds`，因此 NCU 测得：

```text
目标 SMEM TMA bytes = 8192 x 38 KiB = 304 MiB
TMA load instructions = 8192 x 4 = 32768
```

相比 shape5：

| 项目 | shape5 | shape8 | 对掩盖的影响 |
|---|---:|---:|---|
| Tile | `256x192x256` | `256x256x256` | shape8 每轮 B payload 更大 |
| Stage | 6 | 5 | shape8 预取距离更短 |
| K rounds | 16 | 8 | shape8 稳态流水段更短，首尾占比更高 |
| 每 CTA-round 输入 | 34 KiB | 38 KiB | shape8 增加 11.8% |
| CTA-rounds | 4096 | 8192 | shape8 四路 TMA/barrier 实例为 2 倍 |
| 目标侧 TMA bytes | 136 MiB | 304 MiB | shape8 总输入流量更大 |
| TMA load 指令 | 16384 | 32768 | shape8 发起与 accounting 次数为 2 倍 |

更关键的是两条路径的相对长度：

| Shape | Full | NO TC：搬运壳层 | NO TMA：计算壳层 | `Full-NO TMA` |
|---|---:|---:|---:|---:|
| shape5 | 14.518 us | 11.801 us | 14.357 us | 0.160 us |
| shape8 | 31.669 us | 25.641 us | 25.578 us | 6.091 us |

shape5 有 16 轮、6-stage，约 `14.4 us` 的计算壳层长于约 `11.8 us` 的搬运
壳层，输入 TMA 的绝大部分落在计算覆盖窗口中。shape8 只有 8 轮、5-stage，
搬运和计算壳层都约为 `25.6 us`，二者几乎同长，任一侧轻微延迟都会直接推迟
下一轮 operand-ready 或 accumulator-ready。

shape8 虽有 `6.92 waves/SM`，但 `214.016 KiB/CTA` 使 NCU 实测
`Active CTA/SM=1.000`。后续 wave 不能作为同一 SM 上的第二个驻留 CTA 来填补
当前 CTA 的 TMA/mbarrier 等待，所以“大 grid”没有转化为 CTA 间延迟掩盖。

## 4. cluster 多播实际减少了什么

完整算子的 NCU 数据：

| Cluster | Event | L2 bytes | 相对 L2 | TMA 目标 bytes | TMA load | Tensor inst |
|---|---:|---:|---:|---:|---:|---:|
| `2x1x1` | 31.669 us | 338.727 MB | 0 | 318.767 MB | 32768 | 16384 |
| `2x2x1` | 33.594 us | 328.203 MB | -3.11% | 318.767 MB | 32768 | 16384 |
| `4x1x1` | 33.638 us | 287.494 MB | -15.13% | 318.767 MB | 32768 | 16384 |

NO-TC 中的源侧 L2 流量变化更清楚：

| Cluster | NO TC | L2 bytes | 相对 `2x1x1` |
|---|---:|---:|---:|
| `2x1x1` | 25.641 us | 350.994 MB | 0 |
| `2x2x1` | 27.598 us | 328.127 MB | -6.52% |
| `4x1x1` | 27.560 us | 286.759 MB | -18.30% |

多播减少的是“从 L2 取同一源数据”的请求。同一份数据仍需写入各 CTA 的本地
SMEM，所以三组的目标侧 TMA bytes 都是 `318.767 MB`；CUTLASS 仍按每
CTA-round 发起 A/B/SFA/SFB 四路 TMA，所以 TMA load 都是 `32768`。

因此扩大 cluster 没有减少：

- 四路 TMA issue 和 descriptor/address 路径；
- mbarrier transaction accounting 与 consumer wait/release；
- 每个 CTA 收到的 38 KiB 数据；
- SFA/SFB 的 UTCCP；
- UMMA、TMEM accumulator 和 epilogue。

这解释了为何 L2 流量下降 `3%–18%`，NO-TC 端到端时间却增加约 `7.5%`：
shape8 的限制不是 L2 字节吞吐单项，而是目标数据、barrier 和 cluster
readiness 共同构成的延迟链。

## 5. Scheduler 指标为何改善但总时间变差

| Cluster | No Eligible | Eligible/sched | Active warps/sched | Long SB/issue | Active CTA/SM |
|---|---:|---:|---:|---:|---:|
| `2x1x1` | 84.07% | 0.170 | 1.612 | 4.298 | 1.000 |
| `2x2x1` | 83.02% | 0.182 | 1.556 | 3.692 | 1.000 |
| `4x1x1` | 83.25% | 0.178 | 1.567 | 3.890 | 1.000 |

`2x2x1` 的 Long Scoreboard 下降约 `14.1%`，说明多播确实使部分输入
readiness 更早；但 stall ratio 是“每个可发射周期/指令”的归一化比例，不是
绝对 kernel 时间。扩大 cluster 后仍只有 1 CTA/SM，active warps 还略降，
并新增了四 CTA 共同到齐的同步/调度等待。这些代价不一定记到 Long Scoreboard，
所以局部 stall 指标改善不能推出总时间改善。

NO-TMA 是排除“增加了更多 MMA 计算”的直接证据：

| Cluster | NO TMA Event | 相对 `2x1x1` | Tensor inst |
|---|---:|---:|---:|
| `2x1x1` | 25.578 us | 0 | 16384 |
| `2x2x1` | 29.224 us | +14.25% | 16384 |
| `4x1x1` | 29.321 us | +14.63% | 16384 |

三组 UMMA 指令数完全相同。变慢的不是算术工作量，而是 cluster 规模扩大后
mainloop/accumulator/epilogue 参与者的调度、同步与交接路径。

## 6. cluster 同步的独立证据

保持相同 `1024 CTA`、`256 threads/CTA` 和约 `214 KiB/CTA` 动态 SMEM 的
空 kernel 对照：

| Cluster | Empty | Large-SMEM shell | 显式 cluster sync | `sync-empty` |
|---|---:|---:|---:|---:|
| `2x1` | 4.252 us | 4.357 us | 7.505 us | 3.253 us |
| `2x2` | 4.238 us | 4.436 us | 8.398 us | 4.160 us |
| `4x1` | 4.223 us | 4.468 us | 8.358 us | 4.136 us |

不做同步时三组约为 `4.22–4.25 us`；加入显式 cluster sync 后，四 CTA
cluster 相对 `2x1` 多出 `0.88–0.91 us`。完整 CUTLASS shell 中不只有一次
同步，且同步与 stage/barrier/epilogue 的依赖组合，因此最终总损失达到约
`1.93–1.97 us`。

进一步移除 TMA、UMMA 和 epilogue 数据路径后的 `Shell-noEpilogue` 仍为：

```text
2x1x1: 5.875 us
2x2x1: 6.430 us
4x1x1: 6.334 us
```

这再次说明四 CTA cluster 的额外时间存在于控制与 readiness 底座中，不是
输入 payload 或 UMMA 次数造成的。

## 7. 最终瓶颈判断

shape8 应表述为：

> 当前 `256x256x256, Stage=5, Cluster=2x1x1` 的 shape8 未受 HBM/L2
> 峰值带宽单独限制。8 轮较短流水、每轮四路 38 KiB TMA、单 CTA/SM 和
> UMMA/TMEM 依赖链使输入 TMA/mbarrier readiness 与计算路径同时暴露；
> 二者各有约 6 us 落在对方覆盖窗口之外。

扩大 cluster 能降低源侧 L2 请求，但不能降低每 CTA 的 TMA 数、目标 SMEM
字节或 barrier 次数，并引入更大的 cluster 同步域。实测 `2x2x1` 和 `4x1x1`
分别慢 `6.08%` 和 `6.22%`，所以“扩大 cluster 来减少 shape8 搬运耗时”在
当前实现上不可行。

本实验支持的后续方向不是继续扩大 cluster，而是缩短每 CTA-round 的 readiness
路径，例如减少独立 TMA transaction/barrier 工作，或在不损失计算吞吐的前提下
增加有效预取距离；这些属于下一轮优化问题，本报告只确认瓶颈与因果。

## 8. 文件

- `event_raw.csv`：525 条 Event 原始样本；
- `event_summary.csv`：三 cluster、七变体的均值与 95% 置信区间；
- `ncu/`：21 组 NCU CSV；
- `ncu_key_metrics.csv`：NCU 关键指标统一表；
- `cluster_comparison.csv`：cluster 与消融差值主表；
- `empty_cluster_summary.csv`：空 cluster 原始统计；
- `analyze_results.py`：从原始结果重建以上汇总表。
