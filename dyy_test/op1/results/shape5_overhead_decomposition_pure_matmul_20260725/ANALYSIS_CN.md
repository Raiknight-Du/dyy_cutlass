# M_72a shape5：NO-TC / NO-TMA 理论偏差与额外开销分解

## 1. 结论

shape5 的 NO-TC 和 NO-TMA 偏离理论时延，并不是一个遗漏的单项常数造成的。
当前 `M_72a` pure matmul 在该小场景中存在一条约 `5.05 us` 的固定控制/同步底座，
其上还有约 `1.04 us` 的 UTCCP、`1.12 us` 的 D TMA store，以及约 `1.16 us`
的 TMEM 读出、转换和 SMEM staging 暴露时延。

最关键的结论是：

> shape5 不是只由 Tensor Core FLOPS 或 L2 payload 带宽决定。大 SMEM 导致的有限 wave、
> cluster 启动与同步、CUTLASS pipeline 初始化、持久化调度、TMEM 分配/回收、
> accumulator 握手、UTCCP 和 epilogue 共同构成主要时延。

这些开销存在重叠，因此各消融差值表示“从当前关键路径上移除该步骤后减少的暴露时延”，
不能机械地当作彼此独立的硬件 latency 相加。

## 2. 实验口径

固定条件：

| 项目 | 设置 |
|---|---|
| 问题 | `M=8192, N=768, K=4096` |
| Tile | `256x192x256` |
| Cluster | `2x2x1` |
| Mainloop stage | 6 |
| Epilogue | auto，pure matmul，`ElementC=void` |
| 输入 | 同一份持久化随机 A/B/SFA/SFB 数据 |
| 网格 | 256 CTA，256 threads/CTA |
| 资源 | 227.328 KiB SMEM/CTA，1.73 waves/SM |
| 计时 | 5 组独立样本，每组 50 次 CUDA Event |

原 NO-TC 只移除 `cute::gemm`；原 NO-TMA 移除四路输入 TMA，并绕过对应的
mainloop full/empty barrier，但保留 UMMA、UTCCP、accumulator pipeline 和 epilogue。

## 3. Roofline 下界为何不足

工作簿对 dense shape5 给出的下界为：

| 项目 | 理论值 |
|---|---:|
| 计算下界（9 PFLOP/s） | 5.727 us |
| L2 payload 下界（20 TiB/s） | 4.864 us |
| Roofline 下界 `max(compute, L2)` | 5.727 us |

同一时段重测得到：

| 变体 | 实测 | 对应理论下界 | 未解释差额 |
|---|---:|---:|---:|
| Full | 14.518 us | 5.727 us | 8.791 us |
| NO-TC | 11.801 us | 4.864 us | 6.937 us |
| NO-TMA | 14.357 us | 5.727 us | 8.631 us |

理论式只计算稳定状态下的 `FLOPS / peak` 或 `payload / peak_BW`。它没有包含：

1. kernel/cluster 启动与 CTA 分发；
2. 227.328 KiB SMEM 造成的 wave 序列化；
3. 5 套 CUTLASS pipeline 的 barrier 初始化和 cluster 可见性同步；
4. CLC persistent scheduler 的查询、分发和收尾；
5. `tcgen05.alloc/dealloc/relinquish` 及跨 2SM 的 TMEM 协作；
6. accumulator producer/consumer 握手；
7. SFA/SFB 的 SMEM-to-TMEM UTCCP；
8. TMEM-to-register、BF16 转换、SMEM staging 和 D TMA store；
9. prologue/epilogue/tail 在仅 1.73 waves 场景中无法被长循环充分摊薄。

因此 `max(FLOPS/Ppeak, Bytes/Bpeak)` 是硬件峰值下界，不是该短 kernel 的完整时延模型。

## 4. 分层消融结果

| 变体 | 额外移除内容 | Mean | 95% CI | 相对前一相关变体减少 |
|---|---|---:|---:|---:|
| Full | 无 | 14.518 us | ±0.017 us | - |
| NO-TC | UMMA | 11.801 us | ±0.109 us | 2.716 us（相对 Full） |
| NO-TMA | A/B/SFA/SFB TMA | 14.357 us | ±0.044 us | 0.160 us（相对 Full） |
| Shell+UTCCP | 输入 TMA + UMMA | 8.371 us | ±0.022 us | - |
| Shell-noUTCCP | 再移除 UTCCP | 7.334 us | ±0.020 us | 1.038 us |
| Shell-noDStore | 再移除 D TMA store | 6.215 us | ±0.073 us | 1.119 us |
| Shell-noEpilogue | 再移除 TMEM load、转换、SMEM staging | 5.053 us | ±0.042 us | 1.161 us |

`Shell-noEpilogue` 仍保留：

- warp role 分派；
- mainloop/epilogue/CLC/accumulator pipeline 初始化；
- persistent tile scheduler；
- cluster 初始化与可见性同步；
- TMEM allocate、allocation result barrier、release、deallocate；
- accumulator producer/consumer 状态推进；
- producer、scheduler、MMA、epilogue warp 的退出协调。

所以 `5.053 us` 是当前 CUTLASS kernel shell 的实测底座，而不是内存或 Tensor Core
payload 时间。

## 5. 空 cluster 实验

空 kernel 使用相同的 256 CTA、256 threads/CTA 和 `2x2x1` cluster：

| 空 kernel 变体 | Mean | 增量 |
|---|---:|---:|
| cluster，0 KiB SMEM | 2.379 us | kernel/cluster 启动与分发底座 |
| cluster，227.328 KiB SMEM | 3.118 us | +0.739 us |
| 227.328 KiB SMEM + 一次 cluster sync | 4.124 us | +1.007 us |
| CUTLASS Shell-noEpilogue | 5.053 us | +0.929 us |

这组递进实验说明：

1. 单纯启动并完成 256 CTA 的 cluster kernel 已需约 `2.38 us`；
2. 大 SMEM 将执行限制为约 1.73 waves/SM，增加约 `0.74 us`；
3. 一次 cluster 同步再暴露约 `1.01 us`；
4. CUTLASS scheduler、多个 pipeline、TMEM allocate/free 和 accumulator 握手再增加约
   `0.93 us`。

空 kernel 与 CUTLASS shell 并非完全相同的指令流，但控制变量递进清楚地证明：
约 `5 us` 的底座主要来自资源受限下的启动、同步和调度，而不是数据 payload。

## 6. NCU 指令闭环

| 变体 | 输入 TMA inst | D TMA inst | Tensor inst | UTCCP 写 TMEM | TMEM LDT | 总 SM inst |
|---|---:|---:|---:|---:|---:|---:|
| Shell+UTCCP | 0 | 1536 | 0 | 1.573 MB | 98.304 KB | 780.829 K |
| Shell-noUTCCP | 0 | 1536 | 0 | 0 | 98.304 KB | 585.372 K |
| Shell-noDStore | 0 | 0 | 0 | 0 | 98.304 KB | 498.697 K |
| Shell-noEpilogue | 0 | 0 | 0 | 0 | 0 | 206.898 K |

该表确认每一层修改确实命中了目标指令，而不是仅改变数据内容：

- 去 UTCCP 后，UTCCP TMEM 写入从 `1.573 MB` 变为 0；
- 去 D store 后，1536 条 TMA store 和 12.583 MB TMA 写回变为 0；
- 去 epilogue 后，6144 条 TMEM pipe 指令和 98.304 KB TMEM load 变为 0。

即使全部归零，`Shell-noEpilogue` 仍只有约 `0.158 eligible warp/SMSP/cycle`，
Issue Active 约 `14.59%`；NCU 单次测得 Long Scoreboard/issue `3.89`、
Barrier/issue `0.93`、Wait/issue `2.99`。这与“同步/readiness 主导的低 eligible”
一致，而与 HBM/L2 payload 饱和无关。

## 7. SourceCounters 定位

`Shell-noEpilogue` 已无 TMA、UMMA、UTCCP、TMEM load，适合定位纯底座。

SourceCounters 共采到 72 个样本，其中 61 个为 not-issued。最高的五个 PC 分支全部跳入：

```text
SYNCS.PHASECHK.TRANS64.TRYWAIT
NANOSLEEP.SYNCS
SYNCS.PHASECHK.TRANS64
BRA 回到等待循环
```

这五个入口合计 19/72 个全部样本、18/61 个 not-issued 样本。源码映射的最高热点是：

```text
cutlass/arch/barrier.h:416
mbarrier.try_wait.parity.shared::cta.b64
```

该位置得到 20 个映射样本，其中 19 个 not-issued。其他热点包括：

- `sm100_gemm_tma_warpspecialized.hpp:727`：`tmem_allocator.allocate(...)`；
- `tmem_allocator_sm100.hpp:138`：`tcgen05.alloc.cta_group::2.sync`；
- `cluster_sm90.hpp:69`：`barrier.cluster.wait.aligned`；
- `sm90_pipeline.hpp:1069`：多 pipeline barrier 初始化；
- `tmem_allocator_sm100.hpp:176`：`tcgen05.relinquish_alloc_permit`。

因此 `5.05 us` 底座的理论与实测证据一致：它主要是 cluster/pipeline/TMEM
readiness 与调度同步，不是未统计的数据搬运。

## 8. 为什么 NO-TMA 几乎不变

Full 到 NO-TMA 只下降 `0.160 us`，不能解释成 TMA 没有成本。它表示在完整 kernel 中：

1. 输入 TMA 大部分与 UMMA、UTCCP 和 epilogue 重叠；
2. 移除 TMA 后，UMMA/TMEM/accumulator/epilogue 依赖链仍在；
3. 约 `5.05 us` 固定底座和约 `3.32 us` UTCCP+epilogue 路径仍在；
4. scheduler 仍只有很少 eligible warp，第二条独立执行链没有出现。

所以关键路径仍接近原 Full。NO-TMA 的结果证明“输入 payload 不是唯一耗时”，
并不否定完整算子中 TMA readiness 对 UMMA 发起时刻的限制。

## 9. 为什么 NO-TC 仍远高于 L2 理论

NO-TC 保留输入 TMA、四路 transaction、mbarrier、UTCCP、TMEM/accumulator 协议和完整
epilogue。20 TiB/s 公式只计算 payload 穿过 L2 的理想时间；实际还要支付：

- TMA descriptor/地址与 transaction issue；
- stage full/empty barrier；
- SF 到 TMEM 的 UTCCP；
- persistent scheduler；
- D epilogue；
- 约 `5.05 us` 固定底座。

因此 NO-TC 的 `11.80 us` 与 `4.86 us` payload 下界并不矛盾。二者相差的
`6.94 us` 主要是非 payload 控制路径和短 kernel 的启动/尾部。

## 10. 修正后的时延模型

对 shape5 更合适的概念模型为：

```text
Tkernel
≈ Tlaunch/cluster/resource
 + Tpipeline/scheduler/TMEM-handshake
 + max(Tinput-ready path, TUMMA dependency path)
 + Texposed UTCCP
 + Tepilogue tail
```

其中各项仍可能重叠。实测给出的可见边界是：

```text
固定 shell                         ≈ 5.05 us
UTCCP 暴露                         ≈ 1.04 us
D TMA store 暴露                   ≈ 1.12 us
TMEM load + convert + SMEM staging ≈ 1.16 us
```

仅这些非输入-TMA、非-UMMA 路径就达到 `8.37 us`，约为 Full `14.52 us` 的
`57.7%`。所以“其他开销可能是主要开销”的判断成立；更准确地说，shape5 的主要问题是：

> 峰值 roofline 未覆盖的固定 cluster/pipeline/TMEM 同步底座和 epilogue 尾部占比过高，
> 同时低 eligible warp 使这些 readiness 等待无法被其他 CTA/warp 工作充分掩盖。

