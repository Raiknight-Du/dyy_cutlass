# M_72a shape5 TMA 多播收益分析

## 结论

shape5 为：

```text
M=8192, N=768, K=4096
MmaTile=256x192x256
Stage=6
Grid=256 CTA
K rounds=16
ElementC=void
Epilogue=Identity(convert(accumulator))
```

当前 pure-matmul 验证记录中的最优配置是
`M256_N192_K256_Eauto_C4x1x1`。本次在相同 tile、stage、epilogue、随机
输入和计时方法下，重新比较了：

| 配置 | 逻辑 VMNK | 额外多播复用 |
|---|---|---|
| `C2x1` | `(2,1,1,1)` | 无，作为基准 |
| `C2x2` | `(2,1,2,1)` | A/SFA 沿逻辑 N 方向复用 2 次 |
| `C4x1` | `(2,2,1,1)` | B/SFB 沿逻辑 M 方向复用 2 次 |

核心判断：

1. 多播没有减少 TMA 指令数，也没有减少目标 SMEM 收到的数据。三种配置均执行
   `16,384` 条 TMA load，目标侧均收到 `136 MiB`。
2. 多播明确减少了源侧 L2 请求。完整算子中，`C2x2` 比 `C2x1` 少
   `21.29%`，`C4x1` 少 `11.85%`。
3. pure-matmul NO-TC 中，A/SFA 多播的收益暴露得最清楚：`C2x2` 的 Event
   时延下降 `5.80%`，No Eligible 下降 `1.63` 个百分点，Long Scoreboard
   指标下降 `21.06%`。
4. 完整算子中三种 cluster 的 Event 时延差距小于 `0.2%`。因此 shape5 的
   TMA 多播有效降低了 L2/TMA readiness 压力，但这部分在 UMMA 流水中基本
   被覆盖，不能把当前 `C4x1` 的微弱最优解释成显著的 TMA 多播收益。
5. 补充的 NO-TMA 实验中三种计算壳层相差不超过 `0.27%`，UMMA/tensor
   指令和 active cycles 完全一致。因此不存在 `C2x1` 通过增加明显计算
   开销来抵消搬运收益的证据。

## 实验方法

本次 NO-TC 没有复用旧的 GEMM 或非 pure-matmul 结果，而是直接从当前
pure-matmul 源码派生：

- 保留 `ElementC=void`；
- 保留 Identity epilogue，不含 alpha/beta visitor；
- 保留 A/B/SFA/SFB 四路 TMA；
- 保留 SFA/SFB 的 SMEM 到 TMEM UTCCP；
- 保留 mainloop wait/release、accumulator pipeline 和 D store；
- 只移除两处 `cute::gemm(...)`，即不发起 block-scaled UMMA。

Event 使用 5 个独立样本，每个样本 50 次 timed iterations。NCU 在一次
warmup 后采集一次 timed launch，使用 `cache-control=none` 和
`clock-control=none`。

## 每 stage 搬运量

NCU 给出的目标侧 TMA load 总量为 `136 MiB`。shape5 一共有：

```text
256 CTA x 16 K rounds = 4096 CTA-rounds
```

因此每 CTA、每 K round 的目标 SMEM payload 为：

```text
136 MiB / 4096 = 34 KiB
```

该 `34 KiB` 可以拆成：

| Operand | 每 CTA 每 round | 全 kernel 目标侧字节 |
|---|---:|---:|
| A | 16 KiB | 64 MiB |
| B | 12 KiB | 48 MiB |
| SFA | 2 KiB | 8 MiB |
| SFB | 4 KiB | 16 MiB |
| 合计 | 34 KiB | 136 MiB |

这也解释了多播计数：

- `C2x1`：仅有 2SM atom 内部的 SFB 多播，multicast=`16 MiB`；
- `C2x2`：A+SFA+SFB 多播，`64+8+16=88 MiB`；
- `C4x1`：B+SFB 多播，`48+16=64 MiB`。

CUTLASS 源码中，A/SFA 的 `tma_partition` 和 multicast mask 投影到 N
mode，B/SFB 投影到 M mode。多播直接把同一源数据送到多个 CTA 的本地
SMEM，并不是先搬到一个 CTA 再通过 DSM 二次复制。

## 完整 pure matmul

| Cluster | Event mean | 相对 C2x1 | NCU duration | L2 request | 相对 C2x1 | Multicast / target | No Eligible | Eligible/sched | Long SB/issue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C2x1 | 14.555 us | 0 | 15.008 us | 127.680 MiB | 0 | 16 / 136 MiB | 89.09% | 0.115 | 9.318 |
| C2x2 | 14.542 us | -0.09% | 14.976 us | 100.495 MiB | -21.29% | 88 / 136 MiB | 88.41% | 0.123 | 8.335 |
| C4x1 | 14.530 us | -0.17% | 14.688 us | 112.553 MiB | -11.85% | 64 / 136 MiB | 88.48% | 0.121 | 8.323 |

`C2x2` 的 A/SFA payload 比 `C4x1` 的 B payload 更大，因此它降低的 L2
请求更多。尽管 scheduler 指标也改善，Event 总时延几乎不变，说明这些
TMA/L2 等待大部分已经和 16 轮 UMMA 计算重叠。

完整算子的三组置信区间彼此接近，`C4x1` 的 `0.17%` 优势很弱。它可以继续
作为当前最优配置保留，但证据只支持“略快”，不支持“因 B/SFB 多播而明显
更快”。

## Pure-matmul NO-TC

| Cluster | Event mean | 相对 C2x1 | NCU duration | L2 request | 相对 C2x1 | No Eligible | Eligible/sched | Long SB/issue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C2x1 | 12.494 us | 0 | 12.576 us | 124.640 MiB | 0 | 88.41% | 0.123 | 8.385 |
| C2x2 | 11.770 us | -5.80% | 11.904 us | 94.241 MiB | -24.39% | 86.78% | 0.141 | 6.619 |
| C4x1 | 12.524 us | +0.23% | 12.320 us | 112.872 MiB | -9.44% | 87.15% | 0.136 | 6.919 |

NO-TC 去掉了 UMMA 提供的覆盖窗口，因此 `C2x2` 减少 A/SFA 源侧请求后，
搬运 readiness 更早到达，直接形成 `0.725 us` 的 Event 收益。与此同时：

```text
No Eligible:       88.41% -> 86.78%
Eligible/sched:     0.123 -> 0.141
Long SB/issue:      8.385 -> 6.619
```

这是“A/SFA 多播确实缩短 pure-matmul 搬运壳层关键路径”的证据闭环。

`C4x1` 虽然降低了 L2 请求和 stall 指标，但 Event 没有稳定改善。说明 B
方向节省的源侧请求较少，cluster=4 的调度/同步成本足以抵消该收益。NCU
duration 对 `C4x1` 显示小幅改善，而多次 Event 显示 `+0.23%`，两者方向
不一致，所以不应把这部分解释为确定的时延收益。

## Pure-matmul NO-TMA

NO-TMA 从同一份 pure-matmul 源码派生，旁路 producer 的四路 TMA 和
mainloop full/empty barrier，同时保留：

- SFA/SFB SMEM 到 TMEM 的 UTCCP 指令；
- 完整 16 轮 block-scaled UMMA；
- accumulator pipeline；
- pure-matmul Identity epilogue 和 D store。

由于输入 SMEM 未由 TMA 填充，输出数值没有正确性含义；该实验只用于比较
三种 cluster 的计算、TMEM、UTCCP 和 epilogue 指令成本。

| Cluster | Event mean | 相对 C2x1 | NCU duration | TMA load | Tensor inst | Tensor active cycles | No Eligible | Eligible/sched | Long SB/issue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C2x1 | 14.425 us | 0 | 15.680 us | 0 | 8,192 | 6,291,456 | 91.85% | 0.085 | 10.734 |
| C2x2 | 14.387 us | -0.27% | 15.552 us | 0 | 8,192 | 6,291,456 | 91.99% | 0.083 | 10.549 |
| C4x1 | 14.403 us | -0.16% | 15.424 us | 0 | 8,192 | 6,291,456 | 91.97% | 0.083 | 10.528 |

三组 Event 的置信区间重叠：

```text
C2x1: 14.425 +/- 0.057 us
C2x2: 14.387 +/- 0.044 us
C4x1: 14.403 +/- 0.029 us
```

因此不能认为三者存在可重复的计算时延差异。尤其是：

```text
Tensor instructions:    三组都是 8192
Tensor active cycles:   三组都是 6291456
SMEM/CTA:               三组都是 227.328 KB
Registers/thread:       三组都是 69
Waves/SM:               三组都是 1.73
```

这直接否定了“`C2x1` 的搬运更快，同时增加了 UMMA 计算开销，所以完整
时延接近”的主要假设。

还应注意，按本轮严格受控的 NO-TC 数据，搬运壳层明显更快的是 `C2x2`
而不是 `C2x1`：

```text
NO-TC C2x1: 12.494 us
NO-TC C2x2: 11.770 us
NO-TC C4x1: 12.524 us
```

如果“`C2x1` 搬运更快”来自另一组 producer-only microbenchmark，则它与
当前完整 pure-matmul 壳层的 barrier、UTCCP 和 epilogue 定义不同，不能
直接代入本组完整算子。

## 三路差分解释

将三类实验并列：

| Cluster | NO-TC：搬运壳层 | NO-TMA：计算壳层 | Full |
|---|---:|---:|---:|
| C2x1 | 12.494 us | 14.425 us | 14.555 us |
| C2x2 | 11.770 us | 14.387 us | 14.542 us |
| C4x1 | 12.524 us | 14.403 us | 14.530 us |

这里不能把 NO-TC 与 NO-TMA 相加，因为两者都包含 UTCCP、调度和 epilogue
等公共部分。更合适的近似是：

```text
T_full ~= max(T_load_readiness_path, T_compute_path) + exposed interaction
```

三种配置的计算壳层都约为 `14.4 us`，而搬运壳层为 `11.8-12.5 us`。
所以即使 `C2x2` 将搬运壳层再缩短约 `0.72 us`，该部分仍位于计算覆盖
窗口内，不会等比例缩短完整 kernel。

完整时延相对 NO-TMA 仅增加：

```text
C2x1: 0.130 us
C2x2: 0.155 us
C4x1: 0.127 us
```

即只有约 `0.9-1.1%` 的输入搬运、mainloop barrier 与计算交互成本最终
暴露在计算壳层之外。这才是三种 Full 时延接近的主要原因。

NO-TMA 中 No Eligible 和 Long Scoreboard 仍然很高，也证明这两个指标
不只来自 TMA。移除 producer/TMA 后，可被调度的 producer 指令减少，
consumer 侧仍有连续的 UTCCP、UMMA/TMEM 和 epilogue 依赖链，因此
No Eligible 反而上升到约 `92%`。

## 为什么完整算子没有获得同等收益

shape5 有 16 个 K rounds。完整流水中，每一轮的四路 TMA 与前后轮 UMMA
重叠。多播减少的是源侧 L2 请求和 operand-ready latency，而不是：

- TMA 指令发起次数；
- mbarrier 管理次数；
- 目标侧 SMEM 写入量；
- SFA/SFB UTCCP；
- UMMA 次数；
- epilogue/D store。

因此 NO-TC 里暴露的 `5.80%` 不是完整算子可直接相加的收益上限。UMMA
恢复后，TMA 多播节省的大部分时间落入原有计算覆盖窗口，完整 Event 只剩
不到 `0.2%` 的差异。

此外，本实验为标准 L2-hot 测试。shape5 的输入工作集能够较好驻留 L2，
DRAM 读计数非常低且受 NCU replay 噪声影响，不适合用于比较多播收益。
这里应以目标 TMA 字节、multicast 字节和 L2 request 字节的受控差分为准。

## 最终判断

对于当前 shape5：

```text
TMA multicast 有真实的源侧流量和 readiness 收益；
C2x2 的 A/SFA 多播收益明显大于 C4x1 的 B/SFB 多播收益；
但完整算子的关键路径并未被该收益显著缩短。
```

所以当前瓶颈表述应为：

> shape5 不是由 L2/HBM 搬运吞吐单独限制。TMA readiness 仍会造成较高
> No Eligible 和 Long Scoreboard，但 6-stage、16 轮 UMMA 已覆盖了大部分
> 多播可节省的搬运时间。cluster 多播主要降低 L2 请求压力，完整 kernel
> 的性能差异已进入亚百分比区间。NO-TMA 进一步证明三种 cluster 的计算
> 路径成本基本相同，完整时延接近不是由计算开销反向抵消造成的。
