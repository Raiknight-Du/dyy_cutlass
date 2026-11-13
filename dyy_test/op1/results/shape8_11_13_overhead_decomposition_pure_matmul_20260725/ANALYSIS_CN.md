# M_72a shape5/8/11/13 性能瓶颈统一分析

日期：2026-07-25

## 1. 最终结论

四个 shape 都没有达到硬件 roofline，但原因并不相同。

| Shape | Full | 主限制 | 次要限制 | 不应采用的表述 |
|---:|---:|---|---|---|
| 5 | 14.518 us | 小网格下的 CUTLASS cluster/pipeline/TMEM 固定底座与 epilogue 尾部 | TMA/UMMA readiness，二者已有较多重叠 | “由输入 TMA 单独限制” |
| 8 | 31.503 us | 8 轮 mainloop 中输入 TMA、UMMA/TMEM 依赖交互，以及细粒度 `64x64` epilogue staging | 固定 shell 和 D store | “纯计算瓶颈”或“纯带宽瓶颈” |
| 11 | 8.997 us | 单 K tile 的资源受限 kernel shell 与输出 epilogue 串行链 | 输入 TMA/UMMA 仅有很小暴露 | “增加 stage 可以隐藏首块搬运” |
| 13 | 15.018 us | 64 MiB D 输出的 epilogue staging/TMA store 与固定 shell | 输入 TMA readiness | “Long Scoreboard 就等于输入 TMA 等待” |

共同的调度特征是：

- 每个变体实测约 `1 CTA/SM`；
- Full 的 active warp 仅约 `1.61-1.71/SMSP`；
- eligible warp 仅约 `0.16-0.19/SMSP/cycle`；
- No Eligible 为 `82-85%`；
- 移除全部数据搬运和计算后，No Eligible 反而可升至 `88-90%`。

因此高 No Eligible/Long Scoreboard 是“有限执行链被 readiness/synchronization
阻塞”的表现，不能单独证明输入 TMA 是唯一根因。

## 2. 测试口径

所有新增实验均为 pure matmul：

```text
ElementC = void
alpha/beta visitor = 无
epilogue = Identity(accumulator -> BF16 D)
```

统一使用持久化随机 A/B/SFA/SFB 数据。Event 计时采用：

```text
1 次进程内 warmup
25 个独立样本
每样本 10 iterations
GPU 空闲后启动
```

NCU 使用：

```text
--launch-skip 1 --launch-count 1
--cache-control none
--clock-control none
```

这三个新增 Full 基线与先前标准重测相符：

| Shape | 本轮 Full | 先前标准值 | 差异 |
|---:|---:|---:|---:|
| 8 | 31.503 us | 31.621 us | -0.37% |
| 11 | 8.997 us | 8.915 us | +0.92% |
| 13 | 15.018 us | 15.052 us | -0.22% |

## 3. 分析与实验步骤

分析按以下顺序推进：

1. 从 pure-matmul 全搜索结果确定每个 shape 的最优配置。
2. 扫描 N/K tile，检查增加 waves、减少 CTA 数或减少 K-loop 是否有效。
3. 统计 mainloop/epilogue SMEM，确认 Auto stage 是否带来第二个 CTA。
4. 对 shape11/13 强制 Stage=1/2，直接测 active CTA/SM。
5. 用 NCU 检查 SOL、L2 hit、active/eligible warps、No Eligible 和 Long Scoreboard。
6. 用 SourceCounters/PC-SASS 定位 `SYNCS...TRYWAIT`、TMEM allocator 和 cluster barrier。
7. 分别执行 NO TC 和 NO TMA，判断输入链与计算链的暴露时间。
8. 仿照最终 shape5 方法执行累积 shell 消融：

```text
Shell+UTCCP
  -> 去 UTCCP
  -> 去 D TMA store
  -> 去 TMEM load / FP32->BF16 / SMEM staging
```

9. 用空 kernel、大 SMEM kernel 和显式同步 kernel估计资源/调度底座。
10. 用 NCU 指令计数确认每次消融确实使命令归零。

NO TC、NO TMA 和 shell 变体没有数值正确性意义，只用于性能路径消融。
各差值是从当前关键路径上移除步骤后的“暴露时间”，不是单条指令的硬件
latency，各项不能机械相加。

## 4. 配置、stage 与 SMEM

| Shape | 最优配置 | Stage | K rounds | 每 CTA/stage 输入 TMA | Mainloop SMEM | Epilogue SMEM | NCU SMEM/CTA |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | `M256 N192 K256 C2x2 Eauto` | 6 | 16 | 34 KiB | 204 KiB | 约16.5 KiB | 227.328 KB |
| 8 | `M256 N256 K256 C2x1 E64x64` | 5 | 8 | 38 KiB | 190 KiB | 约17 KiB | 214.016 KB |
| 11 | `M128 N192 K128 C1x1 E128x64` | 8 | 1 | 23 KiB | 184 KiB | 约33 KiB | 224.256 KB |
| 13 | `M128 N256 K128 C1x1 E128x128` | 5 | 1 | 27 KiB | 135 KiB | 约65 KiB | 206.848 KB |

Epilogue SMEM 主要是双缓冲 BF16 D：

```text
2 * EpiM * EpiN * sizeof(BF16) + barrier/padding
```

因此约为：

```text
64x64   -> 16 KiB + 控制区
128x64  -> 32 KiB + 控制区
128x128 -> 64 KiB + 控制区
```

shape11/13 虽然 Auto 分别给出 8/5 stages，但 K 只有一个 tile。多余 stage
只占 SMEM，不会形成多轮预取距离。已有 Stage=1/2 实验显示：

| Shape | Stage | Event | SMEM block limit | Reg block limit | Active CTA/SM | No Eligible |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | Auto=8 | 8.915 us | 1 | 2 | 1.000 | 82.12% |
| 11 | 1 | 9.668 us | 2 | 2 | 1.000 | 83.30% |
| 11 | 2 | 9.380 us | 2 | 2 | 1.000 | 82.86% |
| 13 | Auto=5 | 15.052 us | 1 | 1 | 1.000 | 85.22% |
| 13 | 1 | 15.759 us | 1 | 1 | 1.000 | 86.27% |
| 13 | 2 | 15.442 us | 1 | 1 | 1.000 | 85.75% |

shape11 即使 SMEM/寄存器允许两个 block，CUTLASS TMEM allocator 仍申请完整
512 columns，第二个 TMEM-consuming CTA 没有实际驻留。

## 5. 理论下界

计算下界按 `9 PFLOP/s`，L2 下界按 NCU `lts__t_bytes.sum / 20 TiB/s`：

| Shape | Compute lower | L2 lower | Roofline lower | Full | Full/lower |
|---:|---:|---:|---:|---:|---:|
| 8 | 15.271 us | 15.931 us | 15.931 us | 31.503 us | 1.98x |
| 11 | 0.358 us | 2.347 us | 2.347 us | 8.997 us | 3.83x |
| 13 | 0.954 us | 6.013 us | 6.013 us | 15.018 us | 2.50x |

对应 Full 的 L2 hit：

```text
shape8  = 88.68%
shape11 = 94.36%
shape13 = 96.22%
```

先前同配置 NCU 的 DRAM SOL 分别约为 `25.85% / 0.84% / 32.00%`。
因此三者都未受 HBM 饱和限制。Roofline 下界没有计入 CTA/cluster 分发、
pipeline 初始化、TMEM 分配回收、mbarrier、UTCCP、TMEM-to-register、
转换、SMEM staging 和 epilogue tail。

## 6. 逐层消融总表

| Shape | Full | 去 UMMA | 去输入 TMA | Shell+UTCCP | 再去 UTCCP | 再去 D store | 再去其余 epilogue |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 14.518 | 11.801 (-2.716) | 14.357 (-0.160) | 8.371 | 7.334 (-1.038) | 6.215 (-1.119) | 5.053 (-1.161) |
| 8 | 31.503 | 25.512 (-5.992) | 25.587 (-5.916) | 21.419 | 19.597 (-1.822) | 17.420 (-2.177) | 5.901 (-11.519) |
| 11 | 8.997 | 8.905 (-0.093) | 8.823 (-0.174) | 8.846 | 8.826 (-0.020) | 6.811 (-2.015) | 4.451 (-2.360) |
| 13 | 15.018 | 15.001 (-0.017) | 14.295 (-0.723) | 13.072 | 13.105 (+0.034) | 8.982 (-4.124) | 6.606 (-2.376) |

括号为相对对应上一级的暴露时间变化，单位均为 us。shape13 的
`+0.034 us` 小于统计波动，应解释为 UTCCP 暴露约为零。

NCU 指令闭环：

| Shape | Full 输入 TMA | NO TMA | Full tensor | NO TC | Full D store | noDStore | Full TMEM pipe | noEpilogue |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 32768 | 0 | 16384 | 0 | 8192 | 0 | 32768 | 0 |
| 11 | 2048 | 0 | 1024 | 0 | 1536 | 0 | 6144 | 0 |
| 13 | 4096 | 0 | 2048 | 0 | 4096 | 0 | 8192 | 0 |

## 7. Shape5：最终分析回顾

shape5 只有 `1.73 waves/SM`，Full 为 `14.518 us`。NO TMA 只下降
`0.160 us`，不能解释为 TMA 没有成本，而表示六级、16轮流水已经把大部分
输入搬运放在 UMMA/UTCCP/epilogue 覆盖窗口中。

移除输入 TMA 和 UMMA 后仍有 `8.371 us`，占 Full 的 `57.7%`：

```text
固定 CUTLASS shell             5.053 us
UTCCP 暴露                     1.038 us
D TMA store 暴露               1.119 us
TMEM load/convert/SMEM staging 1.161 us
```

空 cluster 实验和 SourceCounters 将固定底座定位到大 SMEM 下的有限 waves、
cluster/pipeline 初始化、persistent scheduler、完整 512-column TMEM
allocate/free、accumulator 握手和 mbarrier `TRYWAIT`。

所以 shape5 的严格表述是：

> 未受 HBM 饱和限制，主要受资源受限的 cluster/pipeline/TMEM 同步底座和
> epilogue 尾部限制；TMA/UMMA readiness 仍参与关键路径，但输入 TMA
> 不是唯一根因。

## 8. Shape8 深入分析

### 8.1 旧 tile/K 扫描

| 配置 | Event | No Eligible | Long SB/issue |
|---|---:|---:|---:|
| N64 K256 | 52.614 us | 83.21% | 6.24 |
| N128 K256 | 35.617 us | 81.00% | 3.77 |
| N256 K128 | 35.728 us | 88.68% | 9.26 |
| N256 K256 | 31.621 us | 84.02% | 4.33 |
| N256 K512 | 46.178 us | 94.70% | 25.43 |

N128 将 waves 翻倍但增加 CTA、A/SFA 重复和 epilogue 实例，仍慢
12.6%。K128 把 mainloop 增至16轮；K512 虽降至4轮，却只剩2 stages，
单 stage payload 和 Long Scoreboard 急升。K256/5-stage/8-round 是当前
资源与预取距离的平衡点。

### 8.2 输入链与计算链

标准消融：

```text
Full 31.503 us
NO TC 25.512 us   -> UMMA/TMEM 路径暴露 5.992 us
NO TMA 25.587 us  -> 输入 TMA/readiness 暴露 5.916 us
```

两条链都不能被称为唯一根因。Full 比去掉任一链都慢约6 us，说明在
1 CTA/SM 下，8轮流水仍有明显的 TMA-to-SMEM、UTCCP、UMMA/TMEM
readiness 与 barrier 交互，缺少第二条独立 CTA 链填泡。

Full NCU：

```text
No Eligible       84.06%
eligible/SMSP     0.170
active/SMSP       1.612
Long SB/issue     4.31
input TMA         32768
tensor inst       16384
L2 hit            88.68%
```

### 8.3 最大单项：细粒度 epilogue staging

shape8 输出为64 MiB。shell 分解得到：

```text
去 D TMA store                       -2.177 us
去 TMEM load + FP32->BF16 + SMEM     -11.519 us
```

该 `11.519 us` 是本轮最大的单项暴露时间。其原因不是64 MiB字节量本身：
shape13 同样输出64 MiB，但对应步骤仅 `2.376 us`。

NCU 对比：

| 指标 | Shape8 | Shape13 | 比值 |
|---|---:|---:|---:|
| TMEM pipe inst | 32768 | 8192 | 4.0x |
| D TMA store inst | 8192 | 4096 | 2.0x |
| Shell-noDStore SM inst | 2.954 M | 1.300 M | 2.27x |

shape8 使用 `EpilogueTile=64x64` 和 2SM `M256xN256` 路径，输出被切成
更多小块，采用更细粒度 TMEM load、转换和 SMEM staging；shape13 使用
`128x128` epilogue，单位逻辑 D 的指令数量明显更低。

完整配置搜索中 `E128x64` 仅比 `E64x64` 慢约1%，而 `E128x128` 慢约12%。
这说明扩大 epilogue tile 会减少局部处理次数，但同时扩大双缓冲 SMEM、
挤压 mainloop stage 或改变同步粒度。当前 best 是总体折中，不表示
`64x64` epilogue 本身没有成本。

### 8.4 Shape8 瓶颈

> shape8 的主瓶颈是单 CTA/SM 下的两部分共同作用：8轮 TMA/UMMA
> readiness 交互，以及 `64x64` epilogue 导致的高 TMEM-load/转换/staging
> 指令成本。最大单项尾部是 epilogue staging，但完整 kernel 仍同时暴露
> 约6 us输入链和约6 us计算链，不能归类为单纯 epilogue-bound。

## 9. Shape11 深入分析

shape11 只有一个 K tile。关键路径是：

```text
首块 TMA
-> SMEM barrier
-> 一次 UTCCP/UMMA
-> accumulator/TMEM handoff
-> TMEM load/convert/SMEM
-> D TMA store
```

没有前一轮 UMMA 可以掩盖首块输入，也没有下一轮 TMA 与当前计算重叠。
但消融进一步表明输入/计算本身并不是主要耗时：

```text
Full            8.997 us
NO TC           8.905 us
NO TMA          8.823 us
Shell+UTCCP     8.846 us
Shell-noUTCCP   8.826 us
```

也就是说，删除 UMMA、输入 TMA、UTCCP 后，时延只变化 `0.02-0.17 us`。
主要时间被以下路径决定：

```text
固定 shell                         4.451 us
D TMA store                        2.015 us
TMEM load/convert/SMEM staging     2.360 us
```

固定 shell 占 Full `49.5%`，输出 epilogue 两段合计占 `48.6%`。
两者已经解释约98%的实测时间边界。

空资源实验：

```text
512 CTA 空 kernel          2.608 us
+ 224.256 KB/CTA SMEM      4.124 us
CUTLASS Shell-noEpilogue   4.451 us
```

大 SMEM 造成的一 CTA 驻留和多 wave 分发已经把空 kernel 增加约
`1.52 us`；CUTLASS pipeline/TMEM/scheduler 再增加约 `0.33 us`。

Full NCU：

```text
No Eligible       82.11%
eligible/SMSP     0.193
active/SMSP       1.697
Long SB/issue     3.93
L2 hit            94.36%
```

Shell-noEpilogue 中 No Eligible 升至 `88.19%`、Long SB 升至 `8.37`，
SourceCounters 首位仍是 `barrier.h:416` 的 mbarrier `TRYWAIT`
（46 samples，45 not-issued），其后是 warp canonicalization、
load-order barrier、TMEM allocate/free 和 `__syncthreads`。

### Shape11 瓶颈

> shape11 主要是单 K tile、512-column TMEM 独占和大 SMEM下的
> kernel-shell/epilogue 串行成本。输入TMA和UMMA仅有约0.1 us量级暴露；
> 增加stage没有可预取的下一块，也不能产生第二CTA，因此不是有效方向。

## 10. Shape13 深入分析

shape13 同样只有一个 K tile，但输出为64 MiB，网格为1024 CTA。

```text
Full            15.018 us
NO TC           15.001 us
NO TMA          14.295 us
Shell+UTCCP     13.072 us
Shell-noUTCCP   13.105 us
```

UMMA和UTCCP几乎完全隐藏；输入TMA/readiness暴露约 `0.723 us`。主导项是：

```text
固定 shell                         6.606 us
D TMA store                        4.124 us
TMEM load/convert/SMEM staging     2.376 us
```

输出 epilogue 合计 `6.500 us`，占 Full `43.3%`；固定 shell 占
`44.0%`。两者合计约87%。

空资源实验：

```text
1024 CTA 空 kernel         3.608 us
+ 206.848 KB/CTA SMEM      4.132 us
CUTLASS Shell-noEpilogue   6.606 us
```

相对于 shape11，shape13 网格翻倍，shell 还要多支付约2.47 us的
scheduler、pipeline、TMEM和barrier工作。

Full NCU：

```text
No Eligible       85.20%
eligible/SMSP     0.162
active/SMSP       1.711
Long SB/issue     5.35
L2 hit            96.22%
```

去掉所有输入、计算和epilogue后，No Eligible仍为 `89.71%`，
Long SB为 `11.39`。SourceCounters 首位同样是
`barrier.h:416`（118 samples，111 not-issued），其后是
pipeline barrier初始化和同步。由此可见 Long Scoreboard 主要在描述
“当前仅剩执行链等待mbarrier/TMEM状态”，不能归因于HBM或输入TMA。

### Shape13 瓶颈

> shape13 是输出/控制主导的单 K tile kernel。首要限制是64 MiB D的
> TMEM-to-register、转换、SMEM staging与TMA store；固定pipeline/TMEM
> shell同样占约44%。输入TMA是次因，UMMA吞吐不是限制。

## 11. SourceCounters 共性

在三个 `Shell-noEpilogue` 中，输入 TMA、UMMA、UTCCP、TMEM load 和
D store均为0。最高采样位置仍统一落在：

```text
cutlass/arch/barrier.h:416
mbarrier.try_wait.parity.shared::cta.b64
```

| Shape | 该行 samples | Not-issued |
|---:|---:|---:|
| 8 | 117 | 114 |
| 11 | 46 | 45 |
| 13 | 118 | 111 |

其他热点包括：

- `cluster.barrier.wait`；
- `tcgen05.alloc/free/relinquish`；
- load-order/accumulator/persistent-scheduler pipeline；
- warp canonicalization 和 `__syncthreads`。

这形成了完整证据链：

```text
大 SMEM/TMEM资源
-> 实际只有1 CTA/SM
-> 可调度warp组少
-> pipeline/TMEM/cluster状态等待时没有第二CTA填泡
-> eligible warp约0.1-0.2/SMSP/cycle
-> No Eligible和Long Scoreboard偏高
```

## 12. 最终瓶颈表述

### Shape8

未受HBM饱和限制，主要受单CTA/SM下8轮TMA/UMMA readiness交互和
细粒度epilogue staging限制。输入链、计算链各有约6 us暴露，最大的单项
尾部是`TMEM load -> BF16 -> SMEM`。

### Shape11

未受输入payload、Tensor Core吞吐或HBM饱和限制。主要受单K tile的
资源受限CUTLASS shell与输出epilogue串行链限制；512-column TMEM分配
使降低stage也无法得到第二CTA。

### Shape13

未受Tensor Core吞吐或HBM饱和限制。主要受64 MiB D输出epilogue和
固定pipeline/TMEM shell限制；输入TMA readiness是次因。

## 13. 复现资料

新增实验目录：

```text
dyy_test/op1/results/
shape8_11_13_overhead_decomposition_pure_matmul_20260725/
```

关键文件：

- `op1_shape8_11_13_overhead_pure_matmul.cu`
- `include_notc/`、`include_notma/`
- `include_shell_utccp/`
- `include_shell_noutccp/`
- `include_shell_nodstore/`
- `include_shell_noepi/`
- `run_experiment.py`
- `analyze_results.py`
- `event_raw.csv`、`event_summary.csv`
- `decomposition_summary.csv`
- `ncu_key_metrics.csv`
- `theory_summary.csv`
- `source_hotspots.csv`
- `ncu/*.ncu-rep`

原shape5最终分解位于：

```text
shape5_overhead_decomposition_pure_matmul_20260725/
```
