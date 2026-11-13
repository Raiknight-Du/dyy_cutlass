# OP1 shape8：MMA 被哪些非输入路径掩盖

## 1. 配置确认

本实验对象是 OP1/M_72a shape8，不是 OP2/M_84a：

```text
M=8192, N=4096, K=2048
MmaTile=256x256x256
Stage=5
EpilogueTile=64x64
ClusterShape=2x1x1
ElementC=void
Pure matmul Identity epilogue
```

`op1_best.cu` 中 shape8 选择 `Config_00`，最终 25 样本配置记录同样是
`M256_N256_K256_E64x64_C2x1x1`。因此 epilogue 是 `E64x64`，不是
`E128x64`。

## 2. 问题与实验原则

需要解释：

```text
NO TMA
    与
NO TMA + NO MMA
```

之间的时延差为什么明显小于 MMA 的理论服务时间。

这两个变体都没有 A/B/SFA/SFB 输入 TMA，因此不能再把 MMA 隐藏归因于输入
TMA。实验固定 NO-TMA 基座，成对改变 MMA 开/关，并逐步移除其余工作：

| 环境 | MMA-on | MMA-off | 两者共同保留 |
|---|---|---|---|
| U+E+D | `notma` | `shell_utccp` | UTCCP、完整 epilogue、D store |
| E+D | `notma_noutccp` | `shell_noutccp` | 完整 epilogue、D store |
| E | `notma_noutccp_nodstore` | `shell_nodstore` | TMEM load、转换、STSM、SMEM staging |
| Shell | `notma_noutccp_noepi` | `shell_noepi` | accumulator handshake 与控制壳层 |

每一行中，两个变体的输入 TMA、UTCCP、epilogue 和 D store 状态完全相同，
唯一关键差异是是否发出 16384 条 UMMA。因此：

```text
MMA exposed(environment) = T(MMA-on) - T(MMA-off)
```

输出数值无正确性意义；NO-TMA 后输入 SMEM 没有被填充。本实验只测相同控制结构
下的指令、依赖和关键路径时间。

## 3. 计时方法

- 持久化随机 shape8 数据；
- 每个变体在进程内预热一次；
- 25 个独立样本；
- 每个样本 10 次 timed iterations；
- 八个变体在每个样本内轮换执行顺序，避免固定热状态偏差；
- 差值按相同样本配对，置信区间由 25 个配对差值计算。

## 4. 配对结果

| 剩余环境 | MMA-on | MMA-off | MMA 暴露时间 |
|---|---:|---:|---:|
| UTCCP + 完整 epilogue + D store | 25.612 us | 21.347 us | **4.265 ± 0.075 us** |
| 完整 epilogue + D store | 25.511 us | 19.767 us | **5.744 ± 0.147 us** |
| TMEM load/转换/STSM，移除 D store | 25.421 us | 17.315 us | **8.106 ± 0.091 us** |
| 仅 accumulator/control shell | 21.268 us | 5.894 us | **15.374 ± 0.064 us** |

没有输入和输出数据路径时，MMA 暴露为 `15.374 us`。shape8 的工作量为：

```text
2*M*N*K = 137.439 TFLOP
137.439 TFLOP / 15.374 us = 8.94 PFLOP/s
```

该结果与约 9 PFLOP/s 对应的 `15.27 us` 理论下界高度一致，说明最底部
MMA-on/off 对成功测到了完整 UMMA 服务与完成等待，而不是只删除了少量控制指令。

## 5. MMA 掩盖量分解

沿“先移除 UTCCP，再移除 D store，最后移除 epilogue 数据路径”的顺序：

| 项目 | 掩盖或暴露的 MMA 时间 | 占完整 MMA 时间 |
|---|---:|---:|
| 最终仍暴露在完整剩余路径外 | 4.265 us | 27.7% |
| 被 SFA/SFB UTCCP 覆盖 | 1.479 ± 0.155 us | 9.6% |
| 被 D TMA store 覆盖 | 2.362 ± 0.161 us | 15.4% |
| 被 TMEM load、转换、STSM、SMEM staging 覆盖 | 7.268 ± 0.079 us | 47.3% |
| 被所有剩余工作覆盖 | **11.109 ± 0.097 us** | **72.3%** |
| 无输入/输出路径时完整 MMA | **15.374 ± 0.064 us** | 100% |

分解是沿固定移除顺序得到的关键路径归因。UTCCP、D store 和 epilogue 之间也有
彼此重叠，所以不同移除顺序可能重新分配少量交互时间；“输出侧是最大掩盖来源”
和“完整 MMA 约为 15.37 us”不受该顺序影响。

## 6. 从另一侧看输出路径

在没有 UTCCP 时：

```text
MMA-off:
  full epilogue - no epilogue
  = 19.767 - 5.894
  = 13.873 us

MMA-on:
  full epilogue - no epilogue
  = 25.511 - 21.268
  = 4.242 us
```

输出路径单独需要约 `13.87 us`，但与 MMA 同时存在时只额外暴露约 `4.24 us`。
因此约 `9.63 us` 的输出工作没有与 MMA 串行相加：

```text
9.63 us = 2.36 us D store overlap
        + 7.27 us TMEM-load/convert/STSM overlap
```

UTCCP 也有相同行为：

```text
MMA-off 时增加 1.581 us
MMA-on  时只增加 0.102 us
差值约 1.479 us
```

说明 UTCCP 的大部分服务时间也位于 UMMA 的执行窗口内。

## 7. NCU 指令闭环

| 变体类别 | Input TMA | Tensor inst | UTCCP | D TMA store | TMEM/LDTM |
|---|---:|---:|---:|---:|---:|
| 所有 MMA-on | 0 | 16384 | 按环境为 3.146 MB 或 0 | 按环境为 8192 或 0 | 按环境存在或为 0 |
| 所有 MMA-off | 0 | 0 | 与配对 MMA-on 相同 | 与配对 MMA-on 相同 | 与配对 MMA-on 相同 |

关键调度指标：

| 变体 | No Eligible | Eligible/sched | Long SB/issue |
|---|---:|---:|---:|
| NO-TMA + MMA + 完整剩余路径 | 84.72% | 0.159 | 4.33 |
| NO-TMA + MMA + NO-UTCCP + NO-EPI | 97.44% | 0.026 | 52.32 |
| NO-TMA + NO-MMA + NO-UTCCP + NO-EPI | 89.90% | 0.108 | 9.77 |

移除 epilogue 后，MMA-on 的 No Eligible 升到约 `97.44%`，Long Scoreboard
升到 `52.32 cycles/issued instruction`。没有 epilogue warp、TMEM load、STSM
和 store pipeline 提供可调度工作时，UMMA 完成等待几乎完整暴露。

这也说明完整 kernel 里的较低 Long Scoreboard 不是 MMA 本身更快，而是其他
warp-specialized 工作填充了部分等待窗口。

## 8. 为什么这种重叠不违反 accumulator 依赖

`tcgen05.mma` 是异步操作。CUTLASS 在 MMA warp 发出 UMMA 后，通过
`PipelineUmmaAsync::producer_commit()` 生成 `tcgen05.commit`，让 mbarrier
跟踪此前 UMMA 的完成；epilogue consumer 只有在 accumulator barrier ready 后
才能执行对应的 TMEM load。

因此：

- 同一个 accumulator 在 UMMA 完成前不会被 epilogue 非法读取；
- “覆盖 11.1 us”是 kernel 关键路径上的并发/流水效果，不表示所有输出指令都在
  同一 accumulator 尚未完成时执行；
- 独立 MMA、epilogue 和 store warp，以及异步 Tensor Core、TMEM 与 TMA-store
  管线，使不同 subtile、cluster 和 wave 的工作可以在整卡时间轴上交错；
- 即使 epilogue consumer 暂时在 accumulator wait，其他输出 warp/store
  pipeline 和其他 SM 上的工作仍可与 UMMA 服务期并行。

NVIDIA PTX 规定 `tcgen05.commit` 跟踪此前异步 `tcgen05.mma` 的完成，只有
mbarrier wait 成功后才能安全访问结果：

- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-commit
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma

## 9. 最终结论

OP1 shape8 中，NO-TMA 基座上的 MMA 理论/实测完整服务时间约为 `15.37 us`。
完整 UTCCP、epilogue 和 D store 存在时，只有 `4.27 us` MMA 时间最终暴露，
约 `11.11 us`、即 `72.3%` 被其余流水覆盖。

掩盖来源按本次累积消融为：

```text
TMEM load + conversion + STSM + SMEM staging   7.27 us
D TMA store                                    2.36 us
SFA/SFB UTCCP                                  1.48 us
```

所以原问题的答案是明确的：MMA 主要被输出 epilogue 数据路径掩盖，其次是
D store，再其次是 scale-factor UTCCP；不是输入 TMA。

## 10. 实验文件

- `run_mma_overlap.py`：构建新增变体并采集 Event/NCU；
- `run_mma_overlap_paired_event.py`：八变体交错配对计时；
- `mma_overlap_paired_event_raw.csv`：200 条配对原始样本；
- `mma_overlap_paired_deltas.csv`：四组 MMA on/off 差值；
- `mma_overlap_decomposition.csv`：掩盖量及置信区间；
- `include_notma_noutccp*`：保留 UMMA、逐步移除剩余路径的可复测源码；
- `ncu/shape8_c0_notma_noutccp*`：新增 NCU 报告与 CSV。
