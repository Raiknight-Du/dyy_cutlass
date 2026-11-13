# M_72a shape8：Epilogue Tile 与 K Tile 对输入 TMA 暴露的影响

## 1. 实验结论

shape8 为：

```text
M=8192, N=4096, K=2048
MmaTile M/N=256x256
Cluster=2x1x1
ElementC=void
pure matmul: D=convert(accumulator)
随机输入数据集固定
```

本轮结论如下。

1. `E64x64` 的 epilogue 确实过细。改成 `E128x64` 后，D TMA store
   指令由 `8192` 降至 `4096`，TMEM pipe 指令由 `32768` 降至 `16384`，
   完整 kernel 的 SM 指令减少 `34.1%`。
2. 但 `E64x64 -> E128x64` 并没有减少 Auto Stage：K256 都是 5 stages，
   K128 都是 10 stages。K256 完整时延也几乎不变：
   `31.660 us -> 31.663 us`。
3. 原因不是 stage 被削减，而是原来较重的 epilogue 指令与输入 TMA/MMA
   有重叠。减少 epilogue 指令后，scheduler 更早进入等待状态，输入 TMA
   readiness latency 暴露得更多。
4. 继续增大到 `E128x128` 才会真正削减 stage：K256 从 5 降到 4，
   K128 从 10 降到 8。K256 因此退化到 `35.208 us`，TMA 独立暴露增加
   `2.290 us`，Long Scoreboard 从 `10.217` 升至 `13.443`。
5. `K256 -> K128` 将每 CTA 每 stage payload 从 `38 KiB` 减为
   `19 KiB`，但 K iteration 从 8 增至 16。总输入 TMA 字节保持不变，
   TMA 指令数却从 `32768` 翻倍至 `65536`。
6. 在 `E128x64` 下，减小 K 后 NO-TMA 和 Shell 几乎不变，但 Full 增加
   `4.108 us`、NO-TC 增加 `4.206 us`、TMA 独立暴露增加 `4.224 us`。
   这直接证明退化来自更多 TMA transaction、地址生成、mbarrier 和循环
   固定开销，而不是 MMA 或 epilogue。

## 2. 实验设计

扫描矩阵：

| K Tile | K iterations | Epilogue Tile |
|---:|---:|---|
| 256 | 8 | 64x64 / 128x64 / 128x128 |
| 128 | 16 | 64x64 / 128x64 / 128x128 |

每个配置测四个严格对应的 pure-matmul 变体：

| 变体 | 输入 TMA | MMA | UTCCP/epilogue/D store |
|---|---|---|---|
| Full | 保留 | 保留 | 保留 |
| NO-TC | 保留 | 移除 | 保留 |
| NO-TMA | 移除 | 保留 | 保留 |
| Shell | 移除 | 移除 | 保留 |

Event 使用 20 组交错随机顺序采样，每组 10 次 kernel。NCU 使用
`cache-control=none`、`clock-control=none`，并核对 TMA、tensor、TMEM
指令及 scheduler 指标。

定义：

```text
TMA 在 MMA 存在时的边际暴露 = Full - NO-TMA
无 MMA 时的 TMA 独立暴露     = NO-TC - Shell
```

这些是消融边际量，不应与其他阶段简单相加。

## 3. Stage、SMEM 与 payload

| K | E tile | Stage | K iter | Mainloop SMEM | Epilogue tensor SMEM | 总 SMEM | 每 CTA/stage TMA |
|---:|---|---:|---:|---:|---:|---:|---:|
| 256 | 64x64 | 5 | 8 | 190 KiB | 17 KiB | 208 KiB | 38 KiB |
| 256 | 128x64 | 5 | 8 | 190 KiB | 33 KiB | 224 KiB | 38 KiB |
| 256 | 128x128 | 4 | 8 | 152 KiB | 65 KiB | 218 KiB | 38 KiB |
| 128 | 64x64 | 10 | 16 | 190 KiB | 17 KiB | 208 KiB | 19 KiB |
| 128 | 128x64 | 10 | 16 | 190 KiB | 33 KiB | 224 KiB | 19 KiB |
| 128 | 128x128 | 8 | 16 | 152 KiB | 65 KiB | 218 KiB | 19 KiB |

CUTLASS 输出的 TMA transaction bytes 是 2SM pair 合计值。K256 为
`76 KiB/pair/stage`，即 `38 KiB/CTA/stage`；K128 为
`38 KiB/pair/stage`，即 `19 KiB/CTA/stage`。因此原先“约 32 KiB”
应修正为当前配置实测的 `38 KiB/CTA/stage`。

`E128x64` 虽然把 epilogue tensor SMEM 从 17 KiB 增到 33 KiB，但剩余
空间仍足以保留原 stage 数。只有 `E128x128` 的 65 KiB epilogue 才跨过
Auto Stage 边界。

## 4. Event 消融结果

单位均为 us。

| K | E tile | Stage | Full | NO-TC | NO-TMA | Shell | TMA 暴露，有 MMA | TMA 独立，无 MMA |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | 64x64 | 5 | 31.660 | 25.555 | 25.608 | 21.444 | 6.052 | 4.111 |
| 256 | 128x64 | 5 | 31.663 | 25.482 | 23.521 | 15.266 | 8.142 | 10.216 |
| 256 | 128x128 | 4 | 35.208 | 27.851 | 25.212 | 15.345 | 9.996 | 12.506 |
| 128 | 64x64 | 10 | 37.004 | 31.647 | 25.680 | 21.482 | 11.324 | 10.165 |
| 128 | 128x64 | 10 | 35.771 | 29.688 | 23.516 | 15.248 | 12.256 | 14.440 |
| 128 | 128x128 | 8 | 35.732 | 29.953 | 23.612 | 15.337 | 12.120 | 14.616 |

Full 的 95% 置信区间半宽依次为：

```text
K256/E64x64:    +/-0.062 us
K256/E128x64:   +/-0.058 us
K256/E128x128:  +/-0.113 us
K128/E64x64:    +/-0.084 us
K128/E128x64:   +/-0.057 us
K128/E128x128:  +/-0.067 us
```

因此 K256 的 `E64x64` 与 `E128x64` 在统计上相同，而 `E128x128`
退化明确。

## 5. NCU 关键指标

| K | E tile | Input TMA inst | Input TMA data | D TMA inst | TMEM pipe inst | SM inst | No Eligible | Eligible/sched | Long SB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 64x64 | 32768 | 318.77 MB | 8192 | 32768 | 4.192 M | 84.02% | 0.170 | 4.344 |
| 256 | 128x64 | 32768 | 318.77 MB | 4096 | 16384 | 2.764 M | 89.75% | 0.109 | 10.217 |
| 256 | 128x128 | 32768 | 318.77 MB | 4096 | 8192 | 2.545 M | 91.50% | 0.089 | 13.443 |
| 128 | 64x64 | 65536 | 318.77 MB | 8192 | 32768 | 4.914 M | 84.70% | 0.163 | 5.122 |
| 128 | 128x64 | 65536 | 318.77 MB | 4096 | 16384 | 3.486 M | 88.71% | 0.119 | 9.242 |
| 128 | 128x128 | 65536 | 318.77 MB | 4096 | 8192 | 3.263 M | 89.32% | 0.114 | 9.967 |

## 6. 增大 Epilogue Tile

### 6.1 `E64x64 -> E128x64`

K256 下：

```text
Stage:              5 -> 5
Shell:         21.444 -> 15.266 us    (-6.178 us)
SM inst:        4.192 -> 2.764 M      (-34.1%)
D TMA inst:      8192 -> 4096
TMEM pipe inst: 32768 -> 16384
Full:          31.660 -> 31.663 us    (无变化)
No Eligible:    84.02 -> 89.75%
Long SB:         4.344 -> 10.217
```

所以 `E128x64` 的确大幅降低了 epilogue 指令开销，而且没有减少 stage。
但原 `E64x64` 的额外 epilogue 工作并非全部串行暴露，它在等待输入 TMA
和 MMA 时占用了原本空闲的 scheduler slot。删掉这些指令后，kernel 更早
抵达 readiness wait，No Eligible 和 Long Scoreboard 上升，节省的指令
没有转化成总时延收益。

### 6.2 `E128x64 -> E128x128`

K256 下：

```text
Stage:                    5 -> 4
SM inst:              -7.94%
Full:          31.663 -> 35.208 us    (+3.545 us)
TMA exposed:    8.142 -> 9.996 us     (+1.853 us)
TMA isolated:  10.216 -> 12.506 us    (+2.290 us)
No Eligible:    89.75 -> 91.50%
Long SB:        10.217 -> 13.443
```

这一组直接验证了“epilogue 继续增大导致 mainloop stage 被削减，进而暴露
更多 TMA latency”。额外减少的 epilogue 指令已经很少，无法补偿少一级
预取带来的损失。

## 7. 减小 K Tile

最干净的对照是固定 `E128x64`：

| 指标 | K256 | K128 | 变化 |
|---|---:|---:|---:|
| Stage | 5 | 10 | x2 |
| K iterations | 8 | 16 | x2 |
| 每 CTA/stage payload | 38 KiB | 19 KiB | x0.5 |
| Input TMA data | 318.77 MB | 318.77 MB | 不变 |
| Input TMA inst | 32768 | 65536 | x2 |
| Shell | 15.266 us | 15.248 us | -0.018 us |
| NO-TMA | 23.521 us | 23.516 us | -0.005 us |
| NO-TC | 25.482 us | 29.688 us | +4.206 us |
| Full | 31.663 us | 35.771 us | +4.108 us |
| TMA 独立暴露 | 10.216 us | 14.440 us | +4.224 us |

NO-TMA 和 Shell 不变，排除了 MMA 与 epilogue 是这 `4.1 us` 退化来源。
总 TMA payload 也不变，排除了搬运字节增加。唯一成倍增加的是：

```text
K iteration
TMA issue 数
descriptor/address generation
mbarrier accounting/wait/release
pipeline acquire/commit/advance
循环与分支控制
```

因此 K128 的实验支持：

> 在 shape8 当前 payload 范围内，增加 stage 数不能抵消 transaction 数翻倍
> 带来的固定控制成本。更小 payload、更多 K iteration 会使输入 TMA 路径
> 更慢，而不是更容易被 mainloop 掩盖。

## 8. 最终判断

原假设需要收紧为：

1. `E64x64` epilogue 的确指令多，但其一部分工作原本填充了 TMA/MMA 等待
   气泡，因此仅减少 epilogue 指令不保证总时延下降。
2. `E128x64` 是不损失 stage 的合法增大点。K256 下它与 `E64x64`
   性能持平；K128 下它比 `E64x64` 快 `1.233 us`。
3. `E128x128` 会削减 stage。K256 只有 8 个 K iteration，对预取距离更敏感，
   因此发生明确退化。
4. 把 K Tile 减为 128 虽然 stage 翻倍，但每 stage payload 减半且 TMA
   transaction 翻倍。固定成本占比上升，使 TMA 暴露增加约 `4.2 us`。
5. 本矩阵最优是 K256 的 `E64x64` 与 `E128x64`，二者统计上持平。
   当前不应继续减小 K；若要从 `E128x64` 的低指令数中获得收益，需要先减少
   四路 TMA/每轮同步固定成本或进一步提高 TMA readiness overlap。

