# 主数据与 SF 打包为同一 TMA 的实验报告

## 1. 结论

本实验测试的不是“共用 pipeline slot、仍分别发起四条 TMA”，而是真正改变数据布局和 TMA descriptor：

```text
Separate:
TMA(A 16 KiB) + TMA(B 16 KiB) + TMA(SFA 2 KiB) + TMA(SFB 2 KiB)

Packed:
TMA(A+SFA 18 KiB) + TMA(B+SFB 18 KiB)
```

在总搬运量、CTA 数、K rounds、Stage、线程数、工作集和 mbarrier transaction bytes 全部相同的条件下，物理 TMA load 数量减半使时延从 `4.835300 ms` 降至 `2.951934 ms`，提升 `38.95%`。

NCU 确认：

- TMA load 指令从 `5.7456 M` 精确减半为 `2.8728 M`；
- 两版 TMA 搬运量均为 `52.95145 GB`；
- SM 指令从 `282.791 M` 降至 `174.702 M`，减少 `38.22%`；
- NCU duration 从 `4.840960 ms` 降至 `2.963552 ms`，减少 `38.78%`。

因此，在本实验的 producer-only 条件下，**四路 TMA 的主要代价确实来自每次 TMA 的 descriptor/坐标生成、issue、TMA 队列与 mbarrier 记账等固定控制成本，而不是 2 KiB SF payload 本身**。将 SF 字节真正打包进主数据 descriptor 是有效方向。

## 2. 为什么上一轮实验没有得到该收益

上一轮验证的是：

```text
A/B/SFA/SFB 共用同一个 stage、mbarrier 和 pipeline state
```

但硬件仍执行：

```text
4 条 cp.async.bulk.tensor / TMA load
```

PTX 的 `cp.async.bulk.tensor` 每条指令只有一个 `tensorMap` 操作数。一个 `tensorMap` 描述一个 tensor 的全局地址、维度、stride、box 和 SMEM swizzle。因此，共用 barrier 不会自动把多个 descriptor 合成一条 TMA。

本次实验为 A+SFA、B+SFB 分别构造了新的连续 tensor 和新的 tensor map，所以硬件层的 TMA 发起数才真正从 4 降为 2。

参考：

- NVIDIA PTX ISA，`cp.async.bulk.tensor`：  
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- CUDA Programming Guide，TMA tensor map 与 SMEM swizzle：  
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

## 3. 控制变量

| 参数 | Separate | Packed |
|---|---:|---:|
| Grid | 35,910 CTA | 35,910 CTA |
| Threads/CTA | 128 | 128 |
| K rounds/CTA | 40 | 40 |
| Stage | 5 | 5 |
| SMEM/CTA | 180.125 KiB | 180.125 KiB |
| 主数据/路/stage | 16 KiB | 包含于 18 KiB |
| SF/路/stage | 2 KiB | 包含于 18 KiB |
| 总 payload/stage | 36 KiB | 36 KiB |
| TMA/stage | 4 | 2 |
| 总 TMA load | 5,745,600 | 2,872,800 |
| 逻辑搬运量 | 52,951,449,600 B | 52,951,449,600 B |
| L2-hot 工作集 | 63 MiB | 63 MiB |
| 数据 | 确定性伪随机 | 相同字节重新打包 |
| mbarrier/stage | 1 | 1 |

每个 CTA 最终读取五个 SMEM stage 中 A、B、SFA、SFB 的首尾数据并生成摘要。两版共 35,910 个 CTA 的摘要逐项一致，排除了 packed TMA 少搬、错搬或被优化掉的可能。

## 4. 重复计时

两变体进行 10 组成对测试，执行顺序交替。

| 变体 | 样本 | Mean | Stddev | Min | Max | Effective TiB/s |
|---|---:|---:|---:|---:|---:|---:|
| Separate 4-TMA | 10 | 4.835300 ms | 0.000082 ms | 4.835200 ms | 4.835430 ms | 9.9599 |
| Packed 2-TMA | 10 | 2.951934 ms | 0.000334 ms | 2.951320 ms | 2.952380 ms | 16.3144 |

| 成对统计 | 结果 |
|---|---:|
| Packed - Separate | -1.883366 ms |
| 相对变化 | **-38.9503%** |
| 差值标准差 | 0.000359 ms |
| 95% CI | [-1.883623, -1.883109] ms |

结果远大于测试误差，且与执行顺序无关。

## 5. NCU 指标

| 指标 | Separate 4-TMA | Packed 2-TMA | 变化 |
|---|---:|---:|---:|
| Duration | 4.840960 ms | 2.963552 ms | -38.78% |
| TMA load 指令 | 5.745600 M | 2.872800 M | **-50.00%** |
| TMA load bytes | 52.951450 GB | 52.951450 GB | 0 |
| SM instructions | 282.791250 M | 174.702150 M | **-38.22%** |
| Active warps/scheduler | 0.9998 | 0.9998 | 基本不变 |
| Eligible warps/scheduler | 0.2021 | 0.2048 | 基本不变 |
| No Eligible | 79.79% | 79.52% | -0.27 pct |
| Long Scoreboard/issue | 0.192 | 0.328 | 指令混合变化 |

`No Eligible` 与 eligible warp 基本不变，说明收益不是由 occupancy 或额外 warp 掩盖带来的。核心变化是同样的 payload 使用更少的 TMA 指令和更少的 SM 控制指令完成。

Packed 的 Long Scoreboard/issue 比例上升不能解释为退化：其分母和指令混合已经发生变化，而绝对 kernel 时延减少近 39%。该指标只表明剩余指令中等待内存完成的相对权重变高。

按 B200 的 148 个 SM 粗略折算：

```text
35,910 CTA / 148 SM = 242.64 waves
每个 CTA 暴露差值约 7.76 us
每 CTA 删除 40 rounds x 2 = 80 条 TMA
```

得到每条被删除 TMA 对当前关键路径的等效贡献约 `97 ns`。这是该调度与工作集下的系统级等效值，包含地址生成、issue、队列和 barrier 推进，不能解释为某条 TMA 硬件指令固定拥有 97 ns latency。

## 6. 接入完整 op1 的限制

本实验已经证明“减少物理 TMA 发起数”有明显潜力，但不能把 microkernel 的 38.95% 直接当作完整 matmul 的预期收益。

当前 CUTLASS op1 中：

- A/B 是 NVFP4 主数据，使用 UMMA 要求的 swizzled SMEM layout；
- SFA/SFB 是 8-bit scale factor，使用 block-scaled 专用 SMEM layout；
- UTCCP 按该 SF layout 将 scale factor 从 SMEM 搬入 TMEM；
- A、B、SFA、SFB 在 kernel 参数中是四个独立的全局 tensor；
- 当前 builder 为它们创建四个 tensor map。

一条 TMA tensor load 只能采用一个 tensor map 和一种 SMEM swizzle。直接把现有 A 与 SFA 指针首尾相接，不能同时生成两种目标 layout。

### 6.1 不推荐：SMEM 中解包

```text
Packed GMEM
    |
一条 TMA -> 临时连续 SMEM
    |
SMEM copy/reorder
    +-> UMMA A/B layout
    +-> UTCCP SF layout
```

该方法需要临时 packed buffer。当前每 stage 为 36 KiB、Stage5；若同时保留原始 A/B/SF buffer，会额外增加约 180 KiB/CTA，超过当前 SMEM 预算。即使减少 stage，SMEM 重排指令也可能吃掉 TMA 减少的收益。

### 6.2 值得继续：GMEM 预打包为物理 SMEM 顺序

可行方向是上游直接生成：

```text
ASFA_packed tile:
[A 的最终 SMEM 物理字节顺序][SFA 的最终 SMEM 物理字节顺序]

BSFB_packed tile:
[B 的最终 SMEM 物理字节顺序][SFB 的最终 SMEM 物理字节顺序]
```

kernel 使用 no-swizzle packed tensor map，把字节原样写入相邻的 A/SFA 或 B/SFB 区域：

```text
GMEM packed physical tile
    |
1 TMA
    |
SMEM [A physical][SFA physical]
       |            |
       UMMA         UTCCP -> TMEM
```

该方案不需要 SMEM 解包，但需要：

1. 改变输入 ABI，由四个 tensor 指针变为两个 packed tensor 指针；
2. 在数据生成或模型权重预处理阶段完成 tile 化和物理 swizzle；
3. 修改 CUTLASS builder、mainloop Params 和 descriptor 创建；
4. 保证 A/SFA、B/SFB 在所有 cluster 配置下具有相同 multicast mask；
5. 保留按 `(M,K)` 与 `(N,K)` tile 的存储索引，避免为了打包而复制跨 N/M 复用的数据；
6. 将预处理成本纳入端到端评估。静态权重较容易摊销，动态 activation 的重排成本可能超过 kernel 收益。

## 7. 对“SF 搭载 KV TMA slot”的准确表述

该短语存在两种实现层级：

| 层级 | 是否减少物理 TMA | 本项目状态 |
|---|---:|---|
| SF 与主数据共用 stage、barrier、acquire/wait/release | 否 | 当前 CUTLASS 已实现 |
| SF 字节与主数据预打包到同一个 tensor map/同一条 TMA | 是 | 本实验已验证纯搬运收益，完整算子尚需输入布局重构 |

如果原 FMHA 方案确实把 SFQ/SFK/SFV 的字节纳入 Q/K/V 的 tensor map，那么它对应第二层，收益来源确实包括减少 TMA 发起数；如果只是“同 slot 管理多个 descriptor”，则只对应第一层。

## 8. 实验材料

远程目录：

```text
/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op1/results/packed_sf_tma_validation_20260727
```

保留：

- `packed_sf_tma_microbenchmark.cu`
- `run_experiment.py`
- `analyze_ncu.py`
- `event_raw.csv`
- `event_summary.csv`
- `paired_summary.csv`
- `ncu_summary.csv`
- `ncu/separate.ncu-rep`
- `ncu/packed.ncu-rep`
- 编译日志和可执行文件

