# SF 搭载主数据 Pipeline 实验

## 1. 实验结论

当前 CUTLASS SM100 block-scaled GEMM 已经实现了本实验设想的核心机制：

- A、B、SFA、SFB 使用同一个 mainloop stage 索引；
- 四路 TMA 记账到同一个 `mbarrier`；
- producer 每个 stage 只执行一次 acquire/commit 状态推进；
- consumer 对该 stage 只执行一次 readiness wait；
- stage ready 后先由 UTCCP 将 SFA/SFB 从 SMEM 搬入 TMEM，再发起 UMMA。

因此，不需要再把 SF “并入”主数据 pipeline。为了验证这种融合是否有效，本实验反向构造了一个拆分版本：A/B 使用主 pipeline，SFA/SFB 使用独立 SF pipeline，并在当前 pure-matmul 基线上进行同输入、同配置对照。

实验没有观察到拆分的性能收益。10 组交替测试中，拆分版本平均慢 `0.004809 ms`，即 `+0.136%`；差值的 95% 置信区间为 `[-0.000095, 0.009713] ms`，小差异仍处于噪声边缘。结合历史同类实验同样呈退化，可判断当前融合方式至少不劣于拆分方式，并避免了明显的额外控制指令。

## 2. 当前源码的真实工作方式

Dense op1 的实现位于：

```text
include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp
```

每个 stage 的 producer 路径可抽象为：

```cpp
mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state);
auto* tma_barrier = mainloop_pipeline.producer_get_barrier(
    mainloop_pipe_producer_state);

copy(tma_load_a.with(*tma_barrier, ...), ...);    // A
copy(tma_load_b.with(*tma_barrier, ...), ...);    // B
copy(tma_load_sfa.with(*tma_barrier, ...), ...);  // SFA
copy(tma_load_sfb.with(*tma_barrier, ...), ...);  // SFB

++mainloop_pipe_producer_state;
```

consumer 路径为：

```text
wait(A + B + SFA + SFB 共用的 stage barrier)
    |
    +-- UTCCP: SFA SMEM -> TMEM
    +-- UTCCP: SFB SMEM -> TMEM
    |
    +-- UMMA: A/B SMEM + SFA/SFB TMEM -> accumulator TMEM
    |
release(stage)
```

Sparse op2 也采用相同组织，只是同一 barrier 下还包含稀疏元数据 E 的 TMA，consumer 侧增加 E 的 UTCCP。

## 3. 对照实验设计

### 3.1 固定条件

| 项目 | 设置 |
|---|---|
| 算子 | op1 pure matmul |
| Shape | `M=170100, N=13824, K=5120` |
| MMA Tile | `256x256x256` |
| Cluster | `2x1x1` |
| Epilogue | `64x64` |
| Stage | Auto，当前配置实际为 5 |
| 输入 | 持久化随机数据，两变体完全相同 |
| 计时 | CUDA Event，10 次迭代/样本 |
| 重复 | 10 组，fused/split 交替运行 |
| 正确性 | 两个变体均通过 |

### 3.2 两个变体

| 变体 | Pipeline 组织 |
|---|---|
| Fused | 当前源码；A/B/SFA/SFB 共用 stage、barrier 和状态机 |
| Split | A/B 使用 main pipeline；SFA/SFB 使用独立 pipeline；consumer 分别 wait/release |

Split 版本通过实验专用 include overlay 编译，未修改仓库中的稳定 CUTLASS 头文件。

## 4. CUDA Event 结果

| 变体 | 样本数 | Mean | Stddev | Min | Max | 相对 Fused |
|---|---:|---:|---:|---:|---:|---:|
| Fused | 10 | 3.547097 ms | 0.005755 ms | 3.536880 ms | 3.554530 ms | 基线 |
| Split | 10 | 3.551906 ms | 0.006240 ms | 3.543000 ms | 3.560890 ms | +0.004809 ms / +0.136% |

成对差值：

| 指标 | 结果 |
|---|---:|
| Split - Fused 平均差值 | +0.004809 ms |
| 差值标准差 | 0.006856 ms |
| 95% CI | [-0.000095, 0.009713] ms |

## 5. NCU 结果

| 指标 | Fused | Split | 变化 |
|---|---:|---:|---:|
| NCU Duration | 3.550656 ms | 3.550752 ms | +0.000096 ms |
| TMA load 指令 | 5.745600 M | 5.745600 M | 0 |
| TMA load bytes | 55.893197 GB | 55.893197 GB | 0 |
| UTCCP TMEM write | 551.577600 MB | 551.577600 MB | 0 |
| SM instructions | 365.499 M | 394.858 M | **+8.03%** |
| Active warps/scheduler | 1.621 | 1.625 | +0.004 |
| Eligible warps/scheduler | 0.115 | 0.123 | +0.009 |
| No Eligible | 89.12% | 88.25% | -0.87 pct |
| Long Scoreboard / issue | 9.684 | 8.902 | -8.07% |
| Barrier / issue | 1.535 | 1.416 | -7.75% |
| MIO Throttle / issue | 0.094 | 0.113 | +20.10% |
| L2 hit rate | 82.70% | 82.22% | -0.47 pct |

## 6. 结果解释

### 6.1 融合的是同步状态，不是物理 TMA 指令

Fused 与 Split 的 TMA 指令数和搬运字节完全一致。当前方案仍然需要分别执行 A、B、SFA、SFB 四条 TMA，因为每条 TMA 使用自己的 tensor map/descriptor。所谓“搭载主数据路径”的收益是共用 stage、barrier 和 pipeline 状态，而不是将四个 descriptor 合成为一条硬件 TMA 指令。

### 6.2 Split 为何局部 stall 指标改善却没有加速

拆分后，consumer 可以先独立确认 SF ready 并执行 UTCCP，再等待 A/B，因而改变了局部 readiness 时序，`Eligible warps/scheduler` 略升，`No Eligible`、Long Scoreboard 和 Barrier stall 略降。

但这不是免费的重叠。Split 同时引入了第二套 producer/consumer state 推进、wait/release 和 barrier 管理，使 SM 指令数增加 `8.03%`，MIO throttle 也上升。局部等待改善被额外控制工作抵消，最终 NCU 时延相同，重复 Event 测试略慢。

### 6.3 对优化方向的约束

当前源码层面再做“SF 与主数据共用 pipeline”没有剩余收益，因为它已经实现。若要进一步减少 TMA 固定开销，需要真正减少物理 TMA transaction，例如：

1. 改变全局内存数据布局，将 A 与 SFA、B 与 SFB 预先打包为可由单个 tensor map 覆盖的存储对象；
2. 设计新的 SMEM 布局和解包路径；
3. 验证额外字节、地址变换、SMEM bank 行为以及 UMMA/UTCCP 接口仍然合法；
4. 对比端到端预处理成本，而不能只比较 kernel 内部时延。

这属于“布局与 descriptor 重构”，不是简单合并 `PipelineTmaAsync` 状态。

## 7. 可复测材料

远程实验目录：

```text
/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op1/results/sf_mainloop_fusion_validation_20260727
```

目录保留：

- `op1_best.cu`：本次 pure-matmul 基线源码；
- `include_split/`：拆分 SF pipeline 的实验头文件；
- `run_experiment.py`：编译、正确性检查和重复 Event 测试；
- `analyze_ncu.py`：NCU 与成对统计汇总；
- `event_raw.csv`、`event_summary.csv`、`paired_summary.csv`；
- `ncu/*.ncu-rep` 和 `ncu_summary.csv`；
- 两个可执行文件及完整日志。

