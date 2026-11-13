# M_72a pure matmul Shape11 SMEM 与增大 Tiling 分析

## 1. 结论

Shape11 当前 best：

```text
Problem  = M16384 x N768 x K128
Tile     = M128 x N192 x K128
Epilogue = E128 x 64
Cluster  = 1 x 1 x 1
Auto stage = 8
```

基线每 CTA 的 SMEM 精确分解为：

```text
Mainloop TensorStorage = 184 KiB
Epilogue TensorStorage =  33 KiB
Kernel pipeline/control =  1 KiB
CUDA driver reserved    =  1 KiB
总计                    = 219 KiB = 224256 B
```

在 shape11 上增大 tiling **没有性能收益**：

| 配置 | 配对 Event | 相对基线 |
|---|---:|---:|
| M128_N192_K128_E128x64_C1x1x1 | 8.2592 us | 基线 |
| M256_N192_K128_E128x64_C2x1x1 | 8.2882 us | +0.35% |
| M256_N256_K128_E128x64_C2x1x1 | 8.3198 us | +0.73% |
| M128_N256_K128_E128x128_C1x1x1 | 9.2601 us | +12.12% |

固定 Stage 同样没有收益：

| Stage | 总 SMEM/CTA | 配对 Event | 相对 Auto |
|---:|---:|---:|---:|
| Auto=8 | 219 KiB | 8.2654 us | 基线 |
| 1 | 58 KiB | 8.8900 us | +7.56% |
| 2 | 81 KiB | 8.4741 us | +2.52% |

原因不是“大 tile 没有减少流量”。2SM 大 tile 确实最多将输入 TMA 降低
38%，但 shape11 的输入 TMA 在先前四象限实验中几乎不在关键路径。大 tile
没有减少 D-store transaction，也没有产生第二个 resident CTA，反而引入
更粗的 accumulator/epilogue readiness、2SM cluster 同步以及更低的
eligible warp。

## 2. SMEM 的真实组成

CUTLASS kernel 的 `SharedStorage` 不是 mainloop 与 epilogue 的复用 union。
源码 `sm100_gemm_tma_warpspecialized.hpp:182-210` 中：

```cpp
struct SharedStorage {
  PipelineStorage pipelines;
  ...
  struct TensorStorage {
    EpilogueTensorStorage epilogue;
    MainloopTensorStorage mainloop;
  } tensors;
};
```

因此 mainloop 和 epilogue TensorStorage 同时存在，SMEM 为两者之和，再加
kernel pipeline/scheduler/barrier 元数据。NCU 还报告 1 KiB
`launch__shared_mem_per_block_driver`。

静态类型探针和 NCU 的结果：

| 配置 | Stage | 每 stage mainloop | Mainloop | Epilogue | Kernel控制 | Driver | 总 SMEM |
|---|---:|---:|---:|---:|---:|---:|---:|
| M128_N192_C1 | 8 | 23 KiB | 184 KiB | 33 KiB | 1 KiB | 1 KiB | 219 KiB |
| M256_N192_C2 | 11 | 17 KiB | 187 KiB | 33 KiB | 1 KiB | 1 KiB | 222 KiB |
| M128_N256_C1 | 5 | 27 KiB | 135 KiB | 65 KiB | 1 KiB | 1 KiB | 202 KiB |
| M256_N256_C2 | 10 | 19 KiB | 190 KiB | 33 KiB | 1 KiB | 1 KiB | 225 KiB |

对应原始字节数：

| 配置 | Mainloop Tensor | Epilogue Tensor | CUTLASS Kernel Shared | NCU 总 SMEM |
|---|---:|---:|---:|---:|
| M128_N192_C1 | 188416 B | 33792 B | 223232 B | 224256 B |
| M256_N192_C2 | 191488 B | 33792 B | 226304 B | 227328 B |
| M128_N256_C1 | 138240 B | 66560 B | 205824 B | 206848 B |
| M256_N256_C2 | 194560 B | 33792 B | 229376 B | 230400 B |

### 2.1 基线 mainloop

每 stage：

```text
A    = 128 * 128 * 0.5 B = 8 KiB
B    = 128 * 192 * 0.5 B = 12 KiB
SFA  = 1 KiB
SFB  = 2 KiB（含布局/padding）
合计 = 23 KiB
```

所以：

```text
8 stages * 23 KiB = 184 KiB
```

### 2.2 基线 epilogue

E128x64 使用双缓冲：

```text
2 stages * 128 * 64 * sizeof(BF16)
= 2 * 128 * 64 * 2
= 32 KiB
```

再加约 1 KiB 的 barrier、对齐和 staging 元数据，Epilogue TensorStorage
为 33 KiB。

### 2.3 为什么 2SM 每 stage 更小

`M256 + Cluster<2,1,1>` 使用 2SM UMMA。逻辑 M tile 是 256，但由两个
CTA/SM 协作，每 CTA 的 accumulator/output 仍对应约 `M128 x N`。A/B/SF
的 SMEM tensor 也按 2SM collective 的分布方式放在两个 CTA 的分布式
SMEM 中，因此每 CTA 的 stage payload 从：

```text
M128_N192 1SM: 23 KiB
M256_N192 2SM: 17 KiB

M128_N256 1SM: 27 KiB
M256_N256 2SM: 19 KiB
```

注意，这是每 CTA TensorStorage 下降，不表示逻辑 tile 的全部输入字节按
同样比例消失；收益来自 2SM 分布和 multicast。

## 3. Auto stage 为什么占满 SMEM

Auto 使用：

```cpp
StageCountAutoCarveout<sizeof(CollectiveEpilogue::SharedStorage)>
```

其目标是在为 epilogue carveout 后，用剩余 SMEM 放尽可能多的 mainloop
stage。因此：

- 2SM M256_N192 每 stage 只有 17 KiB，Auto 增至 11 stages；
- 2SM M256_N256 每 stage 19 KiB，Auto 增至 10 stages；
- 1SM M128_N256 的 epilogue 扩大到 65 KiB，只剩 5 stages。

但 shape11：

```text
K = TileK = 128
k_tile_count = 1
```

无论 Auto 分配 5、8、10 或 11 stages，都只有一个 stage 装入真实 K
数据。额外 stage 不形成更远的 K 预取；但不能据此推导“Stage 越少越
快”，因为 Stage 数还会改变 pipeline ring、barrier phase、静态驻留判断
及生成代码。

按当前 pure-matmul、同一随机输入、40 轮随机交错、每轮 100 iterations
重新配对测试：

| Stage | Event mean | 95% CI | 总 SMEM | NCU SMEM block limit | 平均 Active CTA/SM | Eligible/SMSP | No Eligible | Long SB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Auto=8 | 8.2654 us | +/-0.0048 us | 219 KiB | 1 | 0.680 | 0.191 | 82.26% | 3.867 |
| 1 | 8.8900 us | +/-0.0179 us | 58 KiB | 2 | 0.671 | 0.180 | 83.24% | 4.371 |
| 2 | 8.4741 us | +/-0.0225 us | 81 KiB | 2 | 0.680 | 0.185 | 82.82% | 4.114 |

其中平均 Active CTA/SM 按以下计数器归一化：

```text
sm__ctas_active.avg / sm__cycles_elapsed.avg
```

Stage1/2 虽然让 NCU 的静态 SMEM block limit 从 1 变为 2，实际平均
Active CTA/SM 却与 Auto 相同，没有出现第二条可工作的 CTA 流水线。

根因可以从 CUTLASS 源码直接确认。`sm100_gemm_tma_warpspecialized.hpp`
中的 MMA participant 执行：

```cpp
tmem_allocator.allocate(
    TmemAllocator::Sm100TmemCapacityColumns,
    &shared_storage.tmem_base_ptr);
```

而 `tmem_allocator_sm100.hpp` 定义：

```cpp
Allocator1Sm::Sm100TmemCapacityColumns = 512;
```

也就是每个 CTA 通过
`tcgen05.alloc.cta_group::1.sync.aligned` 申请该 SM 的完整 512 TMEM
columns，而不是按 shape11 实际 accumulator 宽度只申请一部分。只要这个
allocator 策略不变，第二 CTA 就无法建立独立 TMEM accumulator 工作链。
NCU occupancy 页只按 SMEM、register、warp 等传统资源给出静态 block
limit，并没有把这项动态 TMEM 全容量申请反映成一个单独的 block limit。

因此 Stage1/2 释放的 161/138 KiB SMEM 没有转换成计算并发，反而使
Eligible warp 分别下降 5.8%/3.2%，Long Scoreboard 分别增加
13.0%/6.4%，最终更慢。

需要收紧一个因果表述：shape11 只有一个 K iteration，不会发生第二次
真实数据 stage 复用，因而不能笼统地说“Stage 减少导致稳态 pipeline
周转变差”。

Stage1 的额外退化可由 mainloop `load()` 的源码直接解释。每轮取得当前
stage 后，producer 会无条件执行：

```cpp
int write_stage = mainloop_pipe_producer_state.index();
++mainloop_pipe_producer_state;
barrier_token =
    mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);
// 随后才发出当前轮 A/B/SFA/SFB TMA
```

`producer_try_acquire` 会对下一 stage 的 empty barrier 做一次带有限等待的
投机检查。Stage1 中 `++state` 立即绕回同一个 stage、同时 phase 翻转，
于是 producer 在当前轮 TMA 尚未发出前，就开始检查该 buffer 的下一次
复用条件；而该条件必须等当前轮 consumer 完成后才成立。Stage2/Auto
推进到尚未使用的 stage1，该检查可立即返回。因此 Stage1 比 Stage2 多出
一条确定的同槽位 barrier 前视依赖。

Stage2 相对 Auto 仍慢 2.52%，但这部分不能解释为更多 K 预取。补充的
64 KiB 无访问 SMEM padding 控制实验将 Stage1/2 强制回单 block/SM：

| 变体 | 未 padding | +64 KiB padding | 变化 |
|---|---:|---:|---:|
| Stage1 | 8.8900 us | 8.8749 us | -0.17% |
| Stage2 | 8.4741 us | 8.4622 us | -0.14% |

性能没有恢复到 Auto，排除了第二 CTA 在 512-column TMEM allocation
处竞争是 Stage1/2 退化的主要原因。PC-SASS SourceCounters 显示：

- Stage1 新增的 Long Scoreboard 主要落在 tile scheduler 的坐标/divmod
  依赖链以及 mbarrier wait loop；
- Stage2 的额外样本分散在 shared-address/barrier、epilogue loop 和
  scheduler 控制路径；
- 没有证据表明 Stage2 的 0.21 us 差距来自某个新的数据搬运或 UMMA
  工作。

因此当前可以确定：

1. Stage1 存在源码可解释的同 stage 前视等待；
2. Stage1/2 释放 SMEM 都不能换来第二条 TMEM 计算链；
3. Stage2 与 Auto 的较小差距来自 StageCount 改变 SharedStorage、
   barrier tail 和编译后控制路径后的整体短 kernel 调度差异，尚不能归结
   为单一 barrier 指令。

对当前实现，不能为了“去掉未装入真实数据的 stage”而固定为 1 或 2；
同时也不应把 Auto8 的优势解释为它在 shape11 上预取了八个 K tile。

### 3.1 Stage2 到 Stage8 完整扫描

为验证 Stage2 到 Stage8 是否应当等价，使用同一个 pure-matmul 二进制、
同一随机输入，进行 40 轮随机交错配对测试，每轮 100 iterations：

| 变体 | 总 SMEM | Event mean | 95% CI | 相对 Auto8 | SMEM block limit | Eligible/SMSP | No Eligible | Long SB | Barrier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Auto8 | 219 KiB | 8.2605 us | +/-0.0039 us | 基线 | 1 | 0.1928 | 82.11% | 3.933 | 0.851 |
| Stage2 | 81 KiB | 8.4484 us | +/-0.0222 us | +2.28% | 2 | 0.1868 | 82.66% | 4.186 | 0.955 |
| Stage3 | 104 KiB | 8.3453 us | +/-0.0124 us | +1.03% | 2 | 0.1880 | 82.50% | 4.052 | 0.867 |
| Stage4 | 127 KiB | 8.3047 us | +/-0.0123 us | +0.53% | 1 | 0.1891 | 82.43% | 4.050 | 0.857 |
| Stage5 | 150 KiB | 8.2944 us | +/-0.0084 us | +0.41% | 1 | 0.1889 | 82.41% | 3.982 | 0.864 |
| Stage6 | 173 KiB | 8.2647 us | +/-0.0034 us | +0.05% | 1 | 0.1905 | 82.31% | 4.008 | 0.878 |
| Stage7 | 196 KiB | 8.2588 us | +/-0.0038 us | -0.02% | 1 | 0.1913 | 82.25% | 3.942 | 0.875 |
| Stage8 | 219 KiB | 8.2634 us | +/-0.0034 us | +0.04% | 1 | 0.1928 | 82.12% | 3.937 | 0.856 |

结论不是 Stage2 到 Stage8 全部相同，而是：

```text
Stage2 < Stage3 < Stage4/5 < Stage6/7/8 ~= Auto8
```

Stage6、Stage7、显式 Stage8 与 Auto8 的差值均落在联合置信区间内，可视
为同一性能平台；Stage2 到 Stage5 的退化则显著。

这仍然不是更多 K 数据预取造成的。对单 K iteration，producer 在 load
结束后从 stage1 开始执行：

```cpp
for (int count = 0; count < Stages; ++count) {
  empty_barrier_ptr_[state.index()].wait(state.phase());
  ++state;
}
```

只有最后绕回的 stage0 装过真实数据；前面的 `Stages-1` 个 barrier 对应
未使用 stage，通常可以立即通过。StageCount 因此改变了：

- producer 在抵达真正阻塞的 stage0 barrier 前可执行的独立检查数量；
- barrier 数组和 TensorStorage 的 SMEM 地址布局；
- 编译器生成的 tail 展开、分支和地址计算指令；
- producer warp 进入阻塞状态的时间以及 scheduler 可见的 eligible
  指令数量。

NCU 与该控制路径效应一致：从 Stage2 到 Stage8，Eligible/SMSP 总体从
0.1868 增至 0.1928，No Eligible 从 82.66% 降至 82.12%，Long
Scoreboard 从 4.186 降至 3.937；SM 指令数则随每增加一个 stage 约增加
1776 条。新增指令不是有效计算，而主要是在原本容易空闲的短 kernel
窗口中执行的 pipeline/tail 控制工作，因此没有等比例增加关键路径，并
伴随 scheduler starvation 下降。

Stage3 到 Stage4 恰好使静态 SMEM block limit 从2变1，但性能变化是平滑
的，并没有在此处突变；结合前面的 padding 控制，可以再次排除 CTA
occupancy 是该趋势的主要原因。

显式 Stage8 与 Auto8 的 SMEM 都是 219 KiB，SM instructions 都是
1,070,716，Event 仅相差 0.036%，证明当前 Auto policy 解析得到的就是
Stage8，而不是隐藏了其他不同配置。

## 4. 同批 NCU 对照

| 指标 | M128_N192 C1 | M256_N192 C2 | M128_N256 C1 | M256_N256 C2 |
|---|---:|---:|---:|---:|
| Grid CTA | 512 | 512 | 384 | 384 |
| Waves | 3.46 | 3.46 | 2.59 | 2.59 |
| Stage | 8 | 11 | 5 | 10 |
| SMEM/CTA | 224256 B | 227328 B | 206848 B | 230400 B |
| Registers/thread | 105 | 86 | 153 | 90 |
| Block limit by SMEM | 1 | 1 | 1 | 1 |
| Input TMA | 12.059 MB | 8.913 MB | 10.617 MB | 7.471 MB |
| D TMA | 25.166 MB | 25.166 MB | 25.166 MB | 25.166 MB |
| Input TMA inst | 2048 | 2048 | 1536 | 1536 |
| D TMA store inst | 1536 | 1536 | 1536 | 1536 |
| TMEM pipe inst | 6144 | 6144 | 3072 | 6144 |
| SM instructions | 1.071 M | 1.093 M | 0.817 M | 0.930 M |
| Eligible warps/SMSP | 0.193 | 0.206 | 0.160 | 0.145 |
| No Eligible | 82.11% | 80.80% | 85.20% | 86.23% |
| Long SB / issue | 3.957 | 2.789 | 4.908 | 3.817 |
| Barrier / issue | 0.845 | 0.908 | 1.200 | 1.124 |
| Wait / issue | 1.592 | 1.659 | 1.707 | 1.775 |

## 5. 分配置分析

### 5.1 增大 M：M256_N192_C2

逻辑 M 从 128 增至 256，但 2SM cluster 需要两个 CTA 协作：

```text
原来：一个 M128 tile = 1 CTA
现在：一个 M256 tile = 2 CTA
```

所以总 CTA 仍为 512，waves 仍为 3.46，输出/CTA 和 epilogue 也保持
`M128_N192` 与 E128x64。它的实际收益是：

- 输入 TMA 减少 26.1%；
- L2 流量减少 4.1%；
- Long Scoreboard 降低 29.5%；
- Eligible warps 增加 6.4%。

但输入 TMA 在 shape11 四象限实验中只暴露约 `0.004-0.008 us`，不是关键
路径。2SM 协作、cluster barrier 和更多控制指令抵消了流量收益，SM
instructions 反而增加 2.1%，最终慢 0.35%。

这也再次说明：No Eligible 或 Long Scoreboard 改善不等价于 runtime
改善，必须确认被减少的工作原本是否位于关键路径。

### 5.2 增大 N：M128_N256_C1

N tile 从 192 增至 256：

```text
CTA: 512 -> 384，减少25%
waves: 3.46 -> 2.59
```

但平均 wave 填充率没有改善：

```text
512 / (4 * 148) = 86.49%
384 / (3 * 148) = 86.49%
```

它确实减少了：

- 输入 TMA instruction 25%；
- SM instructions 23.7%；
- TMEM pipe instructions 50%。

但出现三个负收益：

1. E128x128 epilogue 把 TensorStorage 从 33 KiB 增至 65 KiB；
2. Registers/thread 从 105 增至 153；
3. D TMA store 仍为 1536 条，输出 transaction 没有减少。

当前底层 D store/TMEM copy atom 仍按 64-column 粒度分解，所以更大的
逻辑 epilogue tile 没有减少 D-store transaction，只让 consumer
readiness 粒度更粗。结果 Eligible warps 下降 17.2%，No Eligible 增加
3.09 个百分点，Long Scoreboard 增加 24%，最终慢 12.12%。

虽然总 SMEM 降到 202 KiB，但仍远高于双 CTA/SM 所需边界，Block Limit
仍为 1，没有换来 occupancy。

### 5.3 同时增大 M/N：M256_N256_C2

这是当前合法 M/N 上界附近的组合：

- 输入 TMA 减少 38.0%；
- SM instructions 减少 13.1%；
- Grid 减少到 384 CTA；
- E128x64 避免了 M128_N256_C1 的寄存器膨胀。

但：

- D store 和 TMEM pipe transaction 与基线完全相同；
- 总 SMEM 增至 225 KiB，仍为 1 CTA/SM；
- Eligible warps 下降 25.1%；
- No Eligible 增至 86.23%；
- Barrier 和 Wait 分别增加 33.0% 与 11.4%。

所以它只做到接近基线，仍慢 0.73%。减少的是非关键输入路径，增加的是
readiness/cluster/consumer 等待。

## 6. K tile 是否值得增大

Shape11 的问题 K 只有 128，当前 TileK 已经是 128。继续增大 TileK：

- 不能把 K iteration 从 1 再减少；
- 会搬入或计算更多 predicated/zero-fill 区域；
- 每 stage payload 增大，Auto stage 下降；
- 不产生新的 A/B reuse。

因此没有理论收益。反方向改为 TileK=64 会把 iteration 从 1 增为 2，
但已有标准测试为 `9.856 us`，也慢于当前配置；两轮仍不足以形成稳态，
同时增加 TMA、UTCCP、UMMA 和 barrier transaction。

## 7. 最终判断

对 shape11，增大 tiling 能带来的潜在收益是：

- 2SM multicast 降低输入 TMA/L2 流量；
- N tile 增大减少 CTA、输入 transaction 和部分 consumer 指令。

但这些收益没有落到当前关键路径。先前消融已经证明输入 TMA 与 UMMA
基本被覆盖，主要时延是：

```text
同网格、1 CTA/SM 的分发/资源底座
+ accumulator 后的 epilogue
+ D TMA store
```

大 tile 不减少 D store transaction，不触发 2 CTA/SM，也不改善平均
wave 填充；更大的 N/cluster 还降低 eligible warp、增加 barrier/readiness
等待。因此：

> 在当前 CUTLASS 2SM UMMA、TMEM allocator 和 TMA epilogue 实现下，
> shape11 的 `M128_N192_K128_E128x64_C1x1x1` 已是更合理的平衡点。
> 单纯增大 M/N/K tiling 没有收益。若后续要改进，应优先减少输出
> epilogue/D-store 串行尾部或改变 TMEM/CTA 驻留条件，而不是继续增大
> 输入 tile。
