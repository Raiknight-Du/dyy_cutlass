# M_72a shape8：E64x64 与 E128x64 动态随机输入重测

## 1. 目的

验证以下判断：

> E128x64 在减少 epilogue 指令的同时没有降低 mainloop stage，因此是否应
> 比 E64x64 更快，只是此前收益被其他路径掩盖。

固定配置：

```text
Shape8: M=8192, N=4096, K=2048
MmaTile=256x256x256
Cluster=2x1x1
Stage=5
ElementC=void
Pure matmul: D=convert(accumulator)
```

仅比较：

```text
E64x64
E128x64
```

## 2. 随机数据与测试方法

本轮不读取固定数据集。每个样本启动新进程，并在计时前生成新的随机输入：

- A/B：SplitMix64 生成的随机 packed-FP4 字节；
- SFA/SFB：随机但有效、有限且非零的 UE4M3 编码，范围 `0x18..0x5e`；
- D 初始缓冲区：随机字节；
- 随机生成和 H2D copy 均在 kernel 计时区间外。

每一对测试使用相同 seed 分别生成 E64x64 和 E128x64 的输入，下一对换新
seed。这样既避免固定数据及硬件压缩偏差，又保证同一对配置看到完全相同的
输入。

测试参数：

```text
随机 seed 对数：50
每个进程 warmup：200 次
每个样本计时：100 次 kernel
两配置顺序：每对随机交错
总独立测量：100 个进程
```

## 3. Event 结果

| 配置 | 样本数 | Mean | Median | Stdev | 95% CI 半宽 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| E64x64 | 50 | **30.0427 us** | 30.0442 us | 0.0424 us | 0.0118 us | 29.9450 us | 30.1325 us |
| E128x64 | 50 | **30.6672 us** | 30.6746 us | 0.0421 us | 0.0117 us | 30.5712 us | 30.7645 us |

配对差值定义为：

```text
delta = E128x64 - E64x64
```

| 指标 | 结果 |
|---|---:|
| 平均差值 | **+0.6245 us** |
| 中位差值 | +0.6114 us |
| 去掉两端各 10% 后的均值 | +0.6236 us |
| 差值标准差 | 0.0612 us |
| 差值 95% CI | `+0.6245 +/- 0.0170 us` |
| E128x64 相对变化 | **+2.079%** |
| E128x64 获胜 | 0 / 50 |
| E64x64 获胜 | **50 / 50** |

顺序分组也保持同一方向：

| E128x64 执行顺序 | 对数 | 平均 `E128-E64` |
|---:|---:|---:|
| 先执行 | 15 | +0.6406 us |
| 后执行 | 35 | +0.6176 us |

因此差异不是由执行顺序、单个 seed 或少数异常值造成。

## 4. 稳态随机输入 NCU

使用同一个有效随机 seed、200 次 warmup 后采集一次 NCU：

| 指标 | E64x64 | E128x64 | 变化 |
|---|---:|---:|---:|
| NCU duration | 32.160 us | 32.672 us | +0.512 us |
| Input TMA load inst | 32768 | 32768 | 不变 |
| D TMA store inst | 8192 | 4096 | -50% |
| TMEM pipe inst | 32768 | 16384 | -50% |
| SM inst | 4.192 M | 2.764 M | -34.1% |
| Stage | 5 | 5 | 不变 |
| SMEM/CTA | 214.016 KB | 230.400 KB | +16.384 KB |
| Registers/thread | 108 | 90 | -18 |
| Waves/SM | 6.92 | 6.92 | 不变 |
| No Eligible | **83.97%** | **89.72%** | +5.75 pp |
| Eligible warps/scheduler | **0.171** | **0.109** | -36.1% |
| Long Scoreboard/issue | **4.412** | **10.275** | 2.33x |
| Barrier/issue | 1.561 | 1.707 | +9.3% |

NCU 与 Event 方向一致：E128x64 的指令数明显减少，但 scheduler starvation
和 Long Scoreboard 明显恶化。

## 5. 为什么“没有减少 stage”仍然会变慢

“不影响输入 TMA”只能严格解释为：

```text
输入 TMA 指令数不变
输入搬运字节不变
每 stage payload 不变
Stage=5 不变
```

它不代表 TMA latency 在关键路径中的暴露量不变。TMA 能否被掩盖还取决于
consumer 侧是否持续提供可发射工作。

E64x64 将输出 tile 划分为更多 epilogue 子块：

```text
更多 TMEM load
更多 convert/STSM
更多 D TMA store
更细的 accumulator readiness 和 store pipeline 推进粒度
```

这些指令数量更多，但也提供了更多细粒度、可交错调度的 consumer 工作。
当 mainloop producer、MMA 或下一块输入等待 TMA/mbarrier readiness 时，
scheduler 仍有较大概率找到 epilogue warp 可以发射。

E128x64 把子块合并后：

```text
D TMA store 数减半
TMEM pipe 指令减半
SM 指令减少 34.1%
```

但依赖粒度也变粗，可独立推进的 epilogue 片段减少。它更早耗尽 ready
consumer work，随后直接暴露 mainloop/TMEM readiness 等待：

```text
Eligible/scheduler: 0.171 -> 0.109
No Eligible:        83.97% -> 89.72%
Long Scoreboard:     4.412 -> 10.275
```

所以这不是简单的“删掉了本来被掩盖的无效指令”。Epilogue tile 同时改变了
warp-specialized pipeline 的粒度和 ready-work 分布。E64x64 虽然总指令
更多，但提供了更好的流水交错和延迟掩盖，最终反而快 `2.08%`。

## 6. 对原判断的修正

原判断中有两部分：

1. **E128x64 减少 epilogue 指令且不减少 stage。**
   这一点成立。
2. **因此 E128x64 是完全优化，只是收益被掩盖。**
   这一点不成立。

更准确的结论是：

> E128x64 是“低 epilogue 指令数”配置，但不是当前流水下的低时延配置。
> 它没有改变输入 TMA 的静态工作量，却降低了 consumer 侧可用于延迟掩盖的
> ready-work 密度，使 TMA/TMEM readiness latency 更充分地暴露。

在当前 M_72a shape8 上，应保留：

```text
M256_N256_K256_E64x64_C2x1x1
```

E128x64 可能在进一步降低 TMA/mbarrier latency、增加独立 consumer warp
工作或重构 epilogue/mainloop overlap 后重新体现其低指令数优势，但在当前
实现中实测慢 `2.08%`。

