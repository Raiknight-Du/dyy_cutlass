# M_72a Shape8 Epilogue 优化未缩短总时延的定位

日期：2026-07-25

## 1. 结论

`E64x64 -> E128x64` 确实降低了 epilogue 自身工作量，但这些工作原本位于
输入 TMA、UMMA 和 accumulator readiness 的重叠窗口内。E128x64 更早结束
上一 work tile 的 epilogue 后，会更早到达：

```cpp
acc_pipeline.consumer_wait(acc_pipe_consumer_state, acc_wait_token);
```

随后等待 MMA warp 将下一份 FP32 accumulator 提交到 TMEM。节省的 epilogue
时间主要转化为更长的 accumulator wait，因此没有缩短完整 kernel 的关键路径。

该 wait 的直接上游是：

```text
输入 TMA 完成
-> UTCCP/SF 可用
-> UMMA 完成全部 K iteration
-> accumulator_pipeline producer_commit
-> epilogue consumer_wait 被唤醒
```

所以它是 TMA-to-MMA-to-TMEM readiness 的汇合等待点，不能只解释为纯 MMA
执行 latency。

## 2. 消融证据

K256、Stage5、Cluster2x1x1：

| 变体 | E64x64 | E128x64 | E128-E64 |
|---|---:|---:|---:|
| Shell，无输入 TMA、无 MMA | 21.444 us | 15.266 us | -6.178 us |
| NO-TMA，保留 MMA | 25.608 us | 23.521 us | -2.087 us |
| Full，旧固定数据 | 31.660 us | 31.663 us | +0.003 us |
| Full，动态随机数据 | 30.0427 us | 30.6672 us | +0.6245 us |

解释：

1. 没有 TMA/MMA 时，epilogue 减少可独立节省 6.178 us。
2. 加入 MMA 后，只剩 2.087 us 可见，约 4.091 us 已与 MMA readiness 重叠。
3. 再加入输入 TMA 后，完整关键路径没有缩短，说明剩余节省继续被
   TMA-to-MMA readiness 窗口吸收。

## 3. PC-SASS 定位

固定随机 seed 和 200 次 warmup，对两配置各重复三次 SourceCounters。

| 配置 | `consumer_wait` Long SB samples | 总 Long SB samples | 总 samples |
|---|---:|---:|---:|
| E64x64 r1/r2/r3 | 84 / 89 / 81 | 592 / 608 / 618 | 1571 / 1588 / 1581 |
| E128x64 r1/r2/r3 | 374 / 393 / 396 | 942 / 993 / 995 | 1558 / 1600 / 1576 |
| 三次均值，E64 | 84.7 | 606.0 | 1580.0 |
| 三次均值，E128 | 387.7 | 976.7 | 1578.0 |

E128x64 的同一源码位置 Long Scoreboard 样本是 E64x64 的约 4.58 倍。

对应源码：

```text
include/cutlass/epilogue/collective/
sm100_epilogue_tma_warpspecialized.hpp:866
```

对应 SASS：

```text
SYNCS.PHASECHK.TRANS64.TRYWAIT
@P1 BRA ...
```

NCU 将 stall 归到分支 PC，但源码映射明确指向：

```cpp
acc_pipeline.consumer_wait(acc_pipe_consumer_state, acc_wait_token);
```

三次采样的 stall 分布：

| Stall | E64x64 | E128x64 |
|---|---:|---:|
| Long Scoreboard | 38.35% | 61.89% |
| Barrier | 20.57% | 13.35% |
| Wait | 14.20% | 8.66% |
| Short Scoreboard | 8.59% | 2.20% |

E128x64 删除了大量 epilogue 指令和同步活动，但空出的调度窗口主要转化为
accumulator consumer wait。按样本占比和约 32 us NCU 时延粗略折算，新增
wait 窗口约 6 us，与 Shell 消融节省的 6.178 us 同量级。该换算只用于判断
量级，PC sampling 比例不应视为精确 wall-clock 分解。

## 4. Wait 前 Sleep 实验

在 `consumer_wait()` 前加入仅用于实验的：

```cpp
__nanosleep(128);
```

只修改 E128x64 epilogue warp 的首次 subtile 路径。使用 15 对动态随机输入，
每进程 warmup 200 次、计时 100 次，顺序交错。

| 配置 | 样本 | Mean | Stdev |
|---|---:|---:|---:|
| E128x64，无 sleep | 15 | 30.66656 us | 0.03692 us |
| E128x64，sleep 128 ns | 15 | 30.66528 us | 0.03709 us |

配对平均变化：

```text
-0.00128 us
```

即 128 ns 的额外延迟完全没有进入 kernel 关键路径。这直接证明 epilogue warp
在到达 accumulator wait 后存在 readiness slack。

## 5. 最终因果关系

```text
E128x64
-> TMEM load、转换、SMEM staging、D-store transaction 减少
-> 上一 work tile 的 epilogue 更早结束
-> epilogue warp 更早请求下一 accumulator
-> MMA 尚未 producer_commit
-> consumer_wait / Long Scoreboard 增大
-> 节省的 epilogue 工作被 readiness wait 替代
-> Full 总时延不下降
```

动态随机测试中 E128x64 还慢约 0.6245 us，说明除了上述关键路径替换外，
`SM100_TMEM_LOAD_32dp32b64x + AutoVectorizing R2S` 的粗粒度依赖链、SMEM
staging 和同步形态也没有优于 E64x64 的
`SM100_TMEM_LOAD_16dp256b8x + SM90_U32x4_STSM_N`。但这部分是次要差异，
本轮已经确认“不降总时延”的主要原因是 accumulator readiness wait。

## 6. 实验完整性

- CUTLASS 头文件已恢复到实验前 SHA256：
  `edf0f073c6685fca1b99a5e224d376f76be71392033fd612fdb94f7f0aed4244`
- 探针头文件、运行脚本、Event CSV 和 PC-SASS CSV 均保存在本实验目录。
- 远程 NCU 报告保存在：
  `dyy_test/op1/results/shape8_epi_random_retest_pure_matmul_20260725/pc_source/`
