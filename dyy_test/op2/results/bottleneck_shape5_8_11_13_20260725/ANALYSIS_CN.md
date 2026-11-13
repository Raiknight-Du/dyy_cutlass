# OP2（M_84a）Shape 5 / 8 / 11 / 13 独立性能瓶颈分析

## 1. 测试口径

- 算子：OP2 pure matmul，逻辑权重为 2:4 稀疏矩阵。
- 输入：持久化随机数据；每次测试复用同一份数据。
- 性能基线：标准 CUDA Event quick mean。
- NCU：`cache-control none`、`clock-control none`，跳过稀疏压缩 kernel，只采 1 次 matmul kernel。
- SourceCounters：采集 Long Scoreboard、Short Scoreboard、Wait、Barrier，并映射到 PC/SASS。
- 补充流量计数：L2 read/write sectors、DRAM read/write bytes。

NCU 会改变短 kernel 的运行环境，因此运行时间以 quick mean 为准，NCU duration 只用于确认报告采到的是同一个 kernel。

## 2. 配置与总体指标

| Shape | 逻辑 M,N,K | OP2 实际输入 M,N,K | 最优配置 | Stage | K tile 数 | Quick mean | 有效 TFLOP/s |
|---:|---|---|---|---:|---:|---:|---:|
| 5 | 8192,768,4096 | 768,8192,8192 | M256 N192 K256 E128x32 C2x2x1 | 8 | 32 | 0.017176 ms | 3001 |
| 8 | 8192,4096,2048 | 4096,8192,4096 | M256 N256 K256 E128x64 C2x1x1 | 6 | 16 | 0.042604 ms | 3226 |
| 11 | 16384,768,128 | 768,16384,256 | M256 N192 K256 E128x64 C2x1x1 | 7 | 1 | 0.008990 ms | 358 |
| 13 | 8192,4096,128 | 4096,8192,256 | M256 N256 K256 Eauto C2x1x1 | 6 | 1 | 0.015086 ms | 569 |

四个 kernel 都使用 256 threads/CTA，并占用约 229.4–231.4 KB SMEM/CTA。B200 上每个 SM 只能驻留 1 个 CTA，理论 active warps/SM 为 8；不存在第二 CTA 来填补 TMA、mbarrier、UMMA/TMEM 或 epilogue 等待。

## 3. Scheduler、SOL 与流量

| Shape | Compute SOL | Memory SOL | DRAM | L2 | TC active | No Eligible | Active warps/SM | Long SB / issued inst |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 49.21% | 47.79% | 3.61% | 40.48% | 49.21% | 87.65% | 7.47 | 8.24 / 12.77 cycles |
| 8 | 51.01% | 45.43% | 23.22% | 37.75% | 51.01% | 90.19% | 6.43 | 11.29 / 16.54 cycles |
| 11 | 13.48% | 36.62% | 0.96% | 26.52% | 7.54% | 81.94% | 5.97 | 2.89 / 8.87 cycles |
| 13 | 19.58% | 56.68% | 33.84% | 44.02% | 10.82% | 82.14% | 6.21 | 3.38 / 9.01 cycles |

| Shape | L2 read | L2 write | L2 总流量估算 | L2 带宽估算 | DRAM 带宽估算 |
|---:|---:|---:|---:|---:|---:|
| 5 | 159.6 MB | 18.9 MB | 178.5 MB | 9.45 TiB/s | 1.06 TiB/s |
| 8 | 376.0 MB | 100.7 MB | 476.7 MB | 10.18 TiB/s | 1.60 TiB/s |
| 11 | 11.1 MB | 37.7 MB | 48.9 MB | 4.94 TiB/s | 0.13 TiB/s |
| 13 | 21.4 MB | 100.7 MB | 122.1 MB | 7.36 TiB/s | 2.40 TiB/s |

项目内 TMA/L2 microbenchmark 已测到约 16.9 TiB/s。四个 shape 的 L2 流量都未达到该上限，DRAM 更未饱和。因此不能把这些 kernel 归类为 HBM 带宽瓶颈。

## 4. OP2 每个 K tile 的真实工作

SourceCounters 的动态指令计数给出稳定的 5:1 比例：

| Shape | CTA 数 | K tile 数 | `UTMALDG` | `UTCOMMA` | TMA load / UMMA |
|---:|---:|---:|---:|---:|---:|
| 5 | 264 | 32 | 42,240 | 8,448 | 5 |
| 8 | 1,024 | 16 | 81,920 | 16,384 | 5 |
| 11 | 516 | 1 | 2,580 | 516 | 5 |
| 13 | 1,024 | 1 | 5,120 | 1,024 | 5 |

OP2 不是四路输入搬运，而是每个 K tile 发出五路 TMA：

1. 压缩后的稀疏 A；
2. B；
3. SFA；
4. SFB；
5. 稀疏 metadata E。

五路 TMA 完成并使 mbarrier ready 后，consumer 才能发出对应 UMMA。UMMA 结果写入 TMEM，epilogue 再等待 accumulator ready，从 TMEM 读出，经 STSM 写入 epilogue SMEM，最后由 TMA store 写 D。

## 5. Shape 5

### 结论

Shape 5 的主要瓶颈是：

> **C2x2 cluster 下的设备利用率损失，加上“5 路 TMA → mbarrier → UMMA → accumulator barrier”链路的 readiness latency；单 CTA/SM 无法掩盖这条依赖链。**

它不是 HBM 带宽瓶颈，也不能简单表述为纯 Tensor Core 算力瓶颈。

### 证据

- 32 个 K tile，已经形成 mainloop 稳态流水。
- No Eligible 为 87.65%，每 scheduler 仅 0.13 个 eligible warp。
- Long Scoreboard 占 8.24/12.77 = 64.5% 的 warp cycles/issued instruction。
- 最高热点 PC 为 `0x...52f0`。它是 `SYNCS.PHASECHK.TRANS64.TRYWAIT` 轮询后的分支，随后才执行 `LDTM`，说明 epilogue consumer 正在等待 accumulator/TMEM ready；该 PC 占全部 Long Scoreboard 样本的 65.0%。
- 按 SASS 区域聚合，67.3% 的 Long Scoreboard 样本落在 accumulator-to-epilogue 等待区域；producer TMA 与 MMA consumer 区域合计占 30.2%。
- 每 CTA 使用约 231.4 KB SMEM，只能驻留 1 CTA/SM。
- Grid 为 264 CTA、cluster 为 4 CTA，共 66 个 cluster。硬件最多同时驻留 33 个此类 cluster，因此恰好执行两批；每批只覆盖 132/148 个 SM，约 10.8% SM 因 cluster 打包无法参与。
- L2 read 远大于 write（159.6 MB vs 18.9 MB），说明 mainloop 输入路径占主要流量；但 L2 仅约 9.45 TiB/s、DRAM 仅 3.61% SOL，带宽没有饱和。
- 相同 M256 N192 K256 E128x32 下，C2x2 为 0.01712 ms，C2x1 为 0.01912 ms。C2x2 的 multicast/reuse 收益约 10.4%，足以覆盖其 cluster 打包损失，说明输入复用仍然重要。

### 瓶颈定位

Shape 5 的 epilogue wait 不是“epilogue 自己太慢”，而是 epilogue 消费者在等待上游 accumulator。关键路径在上游 mainloop 的 TMA readiness 与 UMMA accumulator 生产速度；cluster4 的结构性 SM 空置进一步降低整卡利用率。

## 6. Shape 8

### 结论

Shape 8 的主要瓶颈是：

> **mainloop/UMMA accumulator 生产延迟与 scheduler starvation。Grid 足够大，HBM/L2 未饱和，但单 CTA/SM 不能用第二个工作块掩盖五路 TMA、mbarrier 和异步 UMMA/TMEM 依赖。**

### 证据

- 16 个 K tile，存在稳定 mainloop。
- Grid 1,024 CTA、cluster2，共 512 个 cluster；约 6.92 waves，最后一批接近满载，grid tail 不是主要问题。
- No Eligible 达 90.19%，每 scheduler 仅 0.10 个 eligible warp。
- Long Scoreboard 为 11.29/16.54 = 68.2%，是最主要 stall。
- 最高热点 PC `0x...4e10` 同样位于 accumulator barrier 的 `TRYWAIT + branch` 轮询，随后进入 `LDTM`；该 PC 占 Long Scoreboard 样本的 62.3%。
- epilogue 等待区域聚合占 65.7% 的 Long Scoreboard；producer TMA 与 MMA consumer 合计占 33.7%。这是“epilogue 等上游”，不是 D store 饱和。
- TC active 51.0%，shared active 40.6%，但 DRAM 只有 23.2%、L2 只有 37.8% SOL。
- L2 估算约 10.18 TiB/s，仍明显低于项目 microbenchmark 上限。
- K512 变体并未因 TMA 次数减半而更快：K256 为 0.04258 ms，K512 为 0.04368 ms。说明瓶颈不是单独由 TMA transaction 数量决定，K tile、stage、SMEM 与 UMMA 并行度必须一起考虑。

### 瓶颈定位

Shape 8 没有 shape5 的明显 cluster 覆盖损失，因此其证据更纯粹：高 No Eligible、高 Long Scoreboard、accumulator barrier 热点以及未饱和的 L2/DRAM，共同指向 mainloop readiness/UMMA producer 速度不足以持续喂满后续消费者。

## 7. Shape 11

### 结论

Shape 11 的主要瓶颈是：

> **只有一个 K tile，无法建立稳态流水；固定的 TMA/UMMA/TMEM/epilogue 启动与收尾成本、D 输出路径，以及半满的最后一个 cluster wave 共同主导。**

它不是持续性的 TMA 带宽瓶颈，也不是 Tensor Core 算力瓶颈。

### 证据

- `K=256 == TileK`，每 CTA 只执行 1 次 UMMA；多 stage 只提供资源结构，不能产生跨 K 的预取/计算稳态。
- Compute SOL 仅 13.5%，TC active 仅 7.5%；DRAM 仅 1.0%。
- L2 write 37.7 MB，L2 read 11.1 MB，写流量占 77.2%，说明输出 epilogue 比输入 mainloop 更重。
- No Eligible 为 81.94%，但 Long Scoreboard 仅占 2.89/8.87 = 32.5%；Short Scoreboard 和 Wait 分别为 1.69、1.74 cycles/issued instruction。stall 分布明显比 shape5/8 分散。
- 最高 Long Scoreboard PC 仍是 accumulator wait，但只占 Long Scoreboard 样本的 14.2%，不存在单一 dominant wait。
- producer TMA + MMA consumer 区域占 70.0% Long Scoreboard，epilogue 占 24.2%；按全部 stall 样本计，epilogue 本身又占 39.1%。这符合“一次输入、一次计算、立即收尾”的短流水。
- Grid 516 CTA、cluster2，共 258 个 cluster。B200 同时可驻留 74 个此类 cluster，因此执行 3 个完整 wave 加 36/74 的最后 wave；NCU 估计该 tail 最多可占约 25% runtime。
- 最优搜索中 Eauto/N192 为 0.00903 ms，显式 E128x64 为 0.00919 ms，epilogue 选择对该短 shape 可见，但收益受固定下界限制。

### 瓶颈定位

Shape 11 是典型的短 K、窄 M 启停受限 kernel。No Eligible 高不表示存在一个可持续优化的长 TMA pipeline；这里只有一次输入 ready、一次 UMMA、一次 accumulator handoff 和一次输出，再叠加最后 wave 只有约一半 cluster。

## 8. Shape 13

### 结论

Shape 13 的主要瓶颈是：

> **单 K-tile 下无法重叠的输入 TMA readiness，与以 D 写出为主的 epilogue/output path 形成混合瓶颈；其中总体流量和执行样本更偏向 epilogue，Long Scoreboard 则明确指向输入 ready wait。**

### 证据

- 同样只有 1 个 K tile，没有稳态 mainloop。
- Grid 1,024 CTA、cluster2，约 6.92 waves，grid 覆盖充分。
- Memory SOL 56.7% 明显高于 Compute SOL 19.6%；L1/TEX 71.5%、L2 44.0%、DRAM 33.8%。不是 HBM 饱和，但输出侧 memory path 已成为最忙资源。
- L2 write 100.7 MB，L2 read 21.4 MB，写流量占 82.4%；D 输出是流量主体。
- 按全部 SourceCounters stall 样本，epilogue 区域占 44.8%，是最大的单一区域；其中 STSM、epilogue barrier 和 TMA store 周围出现明显 Short Scoreboard/Barrier。
- 最高 Long Scoreboard PC `0x...1c10` 与 shape5/8 不同：它紧跟 5 条 `UTMALDG` 和 mainloop `SYNCS.PHASECHK.TRANS64.TRYWAIT`，单点占 Long Scoreboard 样本的 26.9%。producer TMA 区域合计占 Long Scoreboard 的 50.5%。
- No Eligible 为 82.14%，但 Long Scoreboard 只占 3.38/9.01 = 37.5%；说明它不是 shape8 那种长期 accumulator-starved 模式，而是一次性输入等待与输出阶段串联。
- Eauto 为 0.01501 ms，而 E128x128 为 0.01641 ms，epilogue tile 选择可带来约 9.3% 差异，进一步证明输出实现处在关键路径上。

### 瓶颈定位

Shape 13 的输入 TMA wait 是最明确的 Long Scoreboard 责任点，但整个 kernel 的最大工作区域和绝大部分 L2 流量来自 epilogue/D store。因此正确表述应是“单次输入 readiness + output epilogue 混合受限”，而不是笼统归因于 TMA 或 HBM。

## 9. 四个 shape 的最终归类

| Shape | 主瓶颈 | 次要限制 | 可以排除 |
|---:|---|---|---|
| 5 | mainloop accumulator readiness；单 CTA/SM 无法掩盖 | C2x2 每批仅覆盖 132/148 SM | HBM 饱和、纯 epilogue bound |
| 8 | mainloop/UMMA producer readiness + scheduler starvation | 5 路 TMA/mbarrier 固定链路 | grid tail、HBM/L2 带宽饱和 |
| 11 | 单 K-tile 启停/收尾 + epilogue 输出 + 半满末 wave | 固定同步与 TMEM handoff | Tensor Core 算力、持续 TMA 带宽 |
| 13 | 单次输入 TMA readiness + 输出 epilogue 混合受限 | STSM/barrier/TMA store 控制成本 | grid tail、HBM 饱和 |

## 10. 关键判断

1. 四个 shape 都受 229–231 KB SMEM 限制为 1 CTA/SM，这是高 No Eligible 无法被第二 CTA 掩盖的共同资源原因。
2. OP2 每 K tile 是 5 路 TMA load，而不是 4 路；稀疏 metadata E 是第五路。
3. Shape5/8 有 32/16 个 K tile，瓶颈位于稳定 mainloop 到 accumulator handoff 的 readiness 链。
4. Shape11/13 只有 1 个 K tile，不能用 steady-state pipeline 模型解释；启动、收尾、epilogue 和 wave quantization 更重要。
5. 所有 shape 都未达到 HBM 或项目实测 L2 峰值，因此“带宽未满”与“访存/同步延迟处于关键路径”可以同时成立。

