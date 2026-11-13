# M_72a pure matmul Shape13 性能瓶颈分析

## 1. 结论

Shape13 的当前 best 配置为：

```text
Problem  = M8192 x N4096 x K128
Tile     = M128 x N256 x K128
Epilogue = E128 x 128
Cluster  = 1 x 1 x 1
Stage    = Auto = 5
```

主要瓶颈不是 Tensor Core 算力，而是只有一个 K iteration 时无法建立稳态
mainloop，并且单 CTA/SM 下输出尾部和 kernel shell 无法被其他工作隐藏：

1. D TMA store 暴露 `4.102 us`，是最大的单项数据路径开销；
2. `TMEM load -> register -> FP32-to-BF16 -> SMEM staging` 暴露 `2.078 us`；
3. 移除全部输入、计算和输出数据工作后，CUTLASS shell 仍需 `6.191 us`；
4. 输入 TMA 与 UMMA 联合暴露 `2.038 us`，其中输入 TMA 是次要限制，
   UMMA 在完整流水中的边际时延只有 `0.010 us`；
5. `206848 B/CTA` SMEM 将驻留限制为 1 CTA/SM。Full 的 No Eligible 为
   `85.18%`，每 SMSP 只有 `0.163` 个 eligible warp，第二个 CTA 无法填泡。

因此更准确的瓶颈表述是：

> Shape13 只有一个 K tile，5-stage 缓冲不能形成跨 K 的稳定预取。输入
> TMA 和异步 UMMA 共享同一 readiness/control 等待窗口，UMMA 基本被该
> 窗口覆盖；accumulator 就绪后，epilogue 转换与 D TMA store 位于严格的
> 输出依赖链上，合计暴露约 6.18 us。与此同时，大 SMEM 造成 1 CTA/SM，
> scheduler 无法用第二个 CTA 隐藏 mbarrier、pipeline 回收和输出等待。

## 2. 分块、流量和理论下界

每 CTA 负责 `128x256` 输出：

```text
grid CTA = (8192 / 128) * (4096 / 256)
         = 64 * 16
         = 1024

waves = 1024 / 148 = 6.919
七波平均 SM 利用率 = 1024 / (7 * 148) = 98.84%
尾波 CTA = 136，尾波填充率 = 91.89%
```

尾波很满，不是主要瓶颈。

`K=128` 与 `TileK=128` 相同，每 CTA 只有一个真实 K iteration。Auto
虽然分配 5 stages，但只有一个 stage 装入有效 K tile，不能形成 5 轮
TMA-UMMA 稳态流水。

每 CTA 的单 stage 输入为：

```text
A   = 128 * 128 * 0.5 B = 8 KiB
B   = 128 * 256 * 0.5 B = 16 KiB
SFA = 1 KiB
SFB = 2 KiB
合计 = 27 KiB
```

NCU 实测输入 TMA 总量为 `28.311552 MB = 27 MiB`，与
`1024 CTA * 27 KiB` 一致。输出 D 为：

```text
D = 8192 * 4096 * 2 B = 64 MiB
```

SMEM 可分解为：

```text
Mainloop = 5 * 27 KiB = 135 KiB
Epilogue payload = 2 stages * 128 * 128 * 2 B = 64 KiB
Epilogue barrier/padding ≈ 3 KiB
总计 ≈ 202 KiB = 206848 B
```

理论下界：

| 项目 | 理论时延 |
|---|---:|
| `2MNK`，按 9 PFLOP/s | 0.954 us |
| 输入 27 MiB，按 20 TiB/s | 1.287 us |
| 输出 D 64 MiB，按 20 TiB/s | 3.052 us |
| Full NCU L2 133.235 MB，按 20 TiB/s | 6.059 us |
| Full 实测 | 14.411 us |

Full 明显高于计算与 L2 理论下界，不能解释为 Tensor Core 或 L2 峰值带宽
已饱和。

## 3. 高精度消融结果

测试使用 pure matmul 路径：

```text
ElementC = void
D = Identity(accumulator)
无 C load
无 alpha/beta visitor
随机输入数据
40 组随机交错样本
每组 100 次 kernel
```

| 变体 | 移除内容 | Mean | 95% CI | 相对相关变体减少 |
|---|---|---:|---:|---:|
| Full | 无 | 14.4106 us | +/-0.0055 us | - |
| NO-TC | UMMA | 14.4002 us | +/-0.0043 us | 0.0103 us vs Full |
| NO-TMA | A/B/SFA/SFB 输入 TMA | 13.7514 us | +/-0.0186 us | 0.6592 us vs Full |
| Shell+UTCCP | 输入 TMA + UMMA | 12.3725 us | +/-0.0027 us | 2.0381 us vs Full |
| Shell-noUTCCP | 再移除 SFA/SFB SMEM-to-TMEM UTCCP | 12.3707 us | +/-0.0038 us | 0.0018 us |
| Shell-noDStore | 再移除 D TMA store | 8.2687 us | +/-0.0042 us | 4.1020 us |
| Shell-noEpilogue | 再移除 TMEM load、转换和 SMEM staging | 6.1906 us | +/-0.0022 us | 2.0781 us |

按关键路径可近似重构 Full：

```text
Shell/control                       6.191 us  42.96%
TMEM load + convert + SMEM staging  2.078 us  14.42%
D TMA store                         4.102 us  28.47%
输入 TMA + UMMA 联合窗口             2.038 us  14.14%
UTCCP                               0.002 us   0.01%
合计                               14.411 us
```

输出路径合计：

```text
2.078 + 4.102 = 6.180 us
占 Full 的 42.89%
```

D store 的 64 MiB 在暴露的 `4.102 us` 内对应约 `14.88 TiB/s`，低于
20 TiB/s 理论值。差值包含 TMA store transaction、shared fence、
store-pipeline acquire/wait 和尾部 drain，不只是字节传输。

## 4. TMA 与 UMMA 的四象限

| 对照 | 暴露时间 | 95% CI |
|---|---:|---:|
| 无 UMMA 时加入输入 TMA | 2.0277 us | +/-0.0055 us |
| 无输入 TMA 时加入 UMMA | 1.3789 us | +/-0.0189 us |
| 有 UMMA 时加入输入 TMA | 0.6592 us | +/-0.0195 us |
| 有输入 TMA 时加入 UMMA | 0.0103 us | +/-0.0069 us |
| 两者联合：Full - Shell | 2.0381 us | +/-0.0062 us |

若两者完全独立串行，联合增加应为：

```text
2.0277 + 1.3789 = 3.4066 us
```

实测只有 `2.0381 us`，存在 `1.3685 us` 的非加和部分。这里应解释为
**关键路径重叠或共享 readiness wait**，不能解释为同一个 K tile 的输入
TMA 与 UMMA 同时执行：同一 tile 仍有严格的 `TMA -> consumer_wait ->
UTCCP -> UMMA` 数据依赖。

NO-TC overlay 保留 TMA、UTCCP、mainloop/accumulator pipeline 和
epilogue，只移除 `cute::gemm` 对应的 block-scaled UMMA。因此结果表明：

- 没有 TMA 时，UMMA 与其完成等待会暴露约 `1.379 us`；
- 完整 kernel 中，异步 UMMA 大部分落入 producer tail、pipeline
  bookkeeping、barrier/readiness 和其他 warp 工作形成的窗口；
- 输入 TMA 更长，仍有约 `0.659 us` 留在完整 kernel 的关键路径上；
- UMMA 不是零成本，但不是当前独立瓶颈。

## 5. 资源底座

使用 1024 blocks、256 threads/block 的独立控制 kernel：

| 控制 kernel | Mean |
|---|---:|
| 空 kernel，0 SMEM | 4.063 us |
| 空工作，206848B SMEM | 4.136 us |
| 206848B SMEM + `__syncthreads()` | 4.159 us |
| 4B SMEM + 512-column TMEM alloc/free | 6.183 us |
| 206848B SMEM + 512-column TMEM alloc/free | 8.227 us |
| CUTLASS Shell-noEpilogue | 6.191 us |

最匹配的空工作对照为：

```text
CUTLASS Shell-noEpilogue - 同网格同SMEM空工作
= 6.191 - 4.136
= 2.055 us
```

所以 Shell-noEpilogue 的 6.19 us 不能全部写成 launch 或 SMEM 驻留时间：

- 约 4.14 us 是同网格、同 SMEM 的分发底座；
- 另约 2.06 us 来自 CUTLASS scheduler、pipeline 初始化/回收、
  accumulator handshake、barrier 和 warp-specialized 控制。

TMEM alloc/free 微内核只用于证明完整 TMEM 与大 SMEM 的资源生命周期可能
暴露，不能直接代入 Full。CUTLASS 会提前释放 allocation lock，其时序与
朴素微内核不同。

## 6. NCU 与 PC/source 证据

Full 的关键指标：

| 指标 | 值 |
|---|---:|
| NCU duration | 15.008 us |
| SMEM/CTA | 206848 B |
| Registers/thread | 153 |
| Block limit by SMEM | 1 |
| Active warps/SMSP | 1.712 |
| Eligible warps/SMSP | 0.163 |
| No Eligible | 85.18% |
| Long Scoreboard / issue | 5.333 |
| Barrier / issue | 1.627 |
| Wait / issue | 1.471 |
| 输入 TMA load | 4096 inst，27 MiB |
| D TMA store | 4096 inst，64 MiB |
| Tensor-pipe instructions | 2048 |
| TMEM-pipe instructions | 8192 |

换算到每 SM，平均约有 `6.85` 个 active warp，但只有 `0.65` 个 eligible
warp。虽然网格有 6.92 waves，SMEM 仍只允许 1 CTA/SM；大量后续 CTA
只能排队，不能与当前 CTA 同驻留填泡。

三次 SourceCounters 重复中，Full 的 Long Scoreboard 热点稳定落在：

| 位置 | 含义 | Long SB 样本占比 |
|---|---|---:|
| `cutlass/arch/barrier.h:416` | `mbarrier.try_wait` 自旋等待 | 26.9%-36.2% |
| `sm100_epilogue_tma_warpspecialized.hpp:768` | D store 前 async-shared fence | 3.5%-5.2% |
| `sm100_epilogue_tma_warpspecialized.hpp:785` | store pipeline acquire | 约4.1%-4.5% |
| `sm100_pipeline.hpp:1046` | 等 stage 回收 | 约1.2%-2.2% |
| `sm100_epilogue_tma_warpspecialized.hpp:866` | 等 accumulator/TMEM ready | 个别重复约2.2% |

Shell-noEpilogue 已移除输入 TMA、UMMA、UTCCP、TMEM load、转换和 D store，
但 `barrier.h:416` 仍占 Long Scoreboard 样本的 `74.2%-87.7%`。这说明：

- 高 Long Scoreboard 不是输入 TMA 的同义词；
- 数据工作移除后，pipeline/accumulator handshake 和 stage 回收仍会进入
  同一个通用 mbarrier wait loop；
- No Eligible 与 Long Scoreboard 的根因是可运行 warp 不足和 readiness
  等待，而不是单一内存指令吞吐饱和。

## 7. Epilogue 与配置控制

### 7.1 Auto epilogue

高精度配对结果：

```text
E128x128 = 14.4106 +/- 0.0055 us
Eauto    = 14.4162 +/- 0.0091 us
差值     = 0.0056 us
```

NCU 中两者的 SMEM、寄存器、输入/D TMA 指令、Tensor/TMEM 指令和总 SM
指令完全相同。由生成的 kernel 类型也可见 Auto 最终选择
`EpilogueTile<128,128>`。两者性能在误差内相同。

### 7.2 已有标准测试中的 tile/cluster 控制

以下数据来自同一轮标准 top20 独立复测，每项 5 组、每组 10 次；只比较
该表内部相对关系：

| 配置 | Mean |
|---|---:|
| M128_N256_K128_E128x128_C1x1x1 | 14.959 us |
| M128_N256_K128_Eauto_C1x1x1 | 15.014 us |
| M256_N256_K128_E128x128_C2x1x1 | 15.185 us |
| M128_N256_K128_Eauto_C2x1x1 | 15.227 us |
| M128_N256_K128_E128x128_C1x2x1 | 15.255 us |
| M128_N256_K128_E128x128_C2x1x1 | 15.318 us |
| M128_N256_K64_E128x128_C1x1x1 | 17.073 us |

控制结果支持：

- Cluster 扩大没有改善当前关键路径。输入流量不是主项，额外 cluster
  协调也不能隐藏串行输出尾部；
- `TileK=64` 把 mainloop 变为两个 K iteration，但同时把 TMA、UTCCP、
  UMMA 和 barrier transaction 数量加倍。两轮仍不足以形成稳态，反而
  增加约 2.1 us；
- 增大 M 并切换到 2SM 配置没有抵消资源、协调和输出路径代价。

## 8. 与 Shape5/8/11 的区别

- Shape5 有 16 个 K iteration，输入 TMA 已基本被稳态 mainloop 掩盖，
  UMMA 和输出尾部更接近独立暴露；
- Shape8 有 8 个 K iteration，输入 TMA 与 UMMA 都明显暴露，且小
  epilogue 带来较大 consumer 指令开销；
- Shape11 只有 1 个 K iteration，但输出规模较小，输入 TMA 与 UMMA
  的边际时延都接近零，主要是约 4.1 us 的输出尾部和 4.1 us 分发底座；
- Shape13 同样只有 1 个 K iteration，但输出为 64 MiB、CTA 数为 1024。
  D store 暴露增至 4.10 us，CUTLASS shell 增至 6.19 us；输入 TMA 也留下
  0.66 us 的次要暴露。

## 9. 最终瓶颈表述

Shape13 的主要瓶颈可以定稿为：

> `K=128` 使每 CTA 只有一次 TMA-UMMA 迭代，Auto=5 stages 无法建立稳态
> 预取；`206848 B/CTA` 又将驻留限制为 1 CTA/SM，导致 scheduler 在
> mbarrier、accumulator readiness、stage 回收与 D-store pipeline 等待时
> 没有第二条 CTA 工作链填泡。最大的数据路径瓶颈是 accumulator 后的
> epilogue 与 64 MiB D TMA store，合计暴露 6.18 us；CUTLASS
> scheduler/pipeline/分发底座约 6.19 us；输入 TMA 与 UMMA 联合暴露
> 2.04 us，其中输入 TMA 是次要限制，UMMA 大部分被共享等待窗口覆盖。

因此，Shape13 不是 HBM/L2 峰值带宽或 Tensor Core 峰值算力受限，而是
**小 K、单 CTA/SM、输出尾部与 readiness/control latency 共同造成的
scheduler starvation**。

