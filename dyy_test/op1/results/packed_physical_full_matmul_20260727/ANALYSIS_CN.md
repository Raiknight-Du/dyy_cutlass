# op1 A+SFA / B+SFB 物理合并 TMA 实验

## 1. 实验结论

实验已在 op1 pure matmul 的 13 个最优配置上完成。方案把 GMEM 数据预先整理为最终
SMEM 物理字节顺序，每个 K stage 由：

```text
A TMA + B TMA + SFA TMA + SFB TMA
```

改为：

```text
packed(A+SFA) TMA + packed(B+SFB) TMA
```

搬运字节数、MMA、UTCCP、TMEM accumulator、epilogue 和 D store 均不变，也不在
SMEM 内执行解包。13 个 shape 的完整 BF16 输出 digest 全部逐位一致。

10 次独立配对测试中，12 个 shape 获得 0.81%–7.98% 的性能提升；shape10 的
变化为 +0.08%，95% 置信区间跨过 0，应判定为无显著变化。13-shape 时延之和由
20.4846 ms 降为 20.2404 ms，降低 1.19%。

## 2. 实现方法

预处理器复用原 CUTLASS 四路 TMA descriptor，把每个 atom CTA 实际收到的 A、B、
SFA、SFB SMEM 物理字节写入 packed GMEM。2SM 配置仍由一个
`SM100_TMA_2SM_LOAD` 把 packed tile 分发为两个 CTA-local fragment。

正式 mainloop 为 packed A、packed B 各建立一个 no-swizzle tensor map。TMA 完成后：

1. A/B 区域直接按原 swizzled SMEM layout 生成 UMMA descriptor；
2. SFA/SFB 区域直接按原 SF layout 生成 UTCCP descriptor；
3. UTCCP 仍将 SF 从 SMEM 搬到 TMEM；
4. block-scaled UMMA 和后续 pure-matmul epilogue 保持不变。

因此，本实验减少的是 TMA 发起、地址生成及 transaction accounting 次数，不是
SF 字节，也没有消除 UTCCP。

packed mainloop 的 `TensorStorage` 必须显式 `alignas(1024)`。控制实验中，
N192、E128x16 的 shape6/7 在默认 16 B 实际对齐下输出错误；提高为 1 KiB 后，
shape6/7 及其余 11 个 shape 全部逐位通过。仅使用
`cute::aligned_struct<1024, ...>` 不足，因为实测 `alignof` 仍为 16 B。

## 3. 13-shape 性能

计时标准为随机持久化输入、预处理不计时、每个样本 10 次 kernel iteration、
10 次基线/packed 交替配对。`±` 后为样本标准差。

| ID | 配置 | Baseline ms | Packed ms | 变化 |
|---:|---|---:|---:|---:|
| 1 | M256 N256 K256 E64x64 C2x1x1 | 3.542920 ± 0.004890 | 3.514214 ± 0.004360 | -0.81% |
| 2 | M256 N256 K256 E64x64 C2x1x1 | 11.504600 ± 0.072905 | 11.372130 ± 0.066404 | -1.15% |
| 3 | M256 N256 K256 E64x64 C2x1x1 | 1.299534 ± 0.000700 | 1.281242 ± 0.000363 | -1.41% |
| 4 | M256 N256 K256 E64x64 C2x1x1 | 3.885115 ± 0.016335 | 3.827768 ± 0.016516 | -1.48% |
| 5 | M256 N192 K256 Eauto C2x2x1 | 0.015289 ± 0.000078 | 0.014880 ± 0.000075 | -2.68% |
| 6 | M256 N192 K256 E128x16 C4x1x1 | 0.023836 ± 0.000264 | 0.023032 ± 0.000057 | -3.38% |
| 7 | M256 N192 K256 E128x16 C2x1x1 | 0.042847 ± 0.000231 | 0.039429 ± 0.000071 | -7.98% |
| 8 | M256 N256 K256 E64x64 C2x1x1 | 0.031708 ± 0.000169 | 0.030955 ± 0.000192 | -2.38% |
| 9 | M256 N256 K256 E64x64 C2x1x1 | 0.056600 ± 0.000172 | 0.055532 ± 0.000497 | -1.89% |
| 10 | M128 N256 K128 Eauto C1x2x1 | 0.045079 ± 0.000321 | 0.045116 ± 0.000247 | +0.08% |
| 11 | M128 N192 K128 E128x64 C1x1x1 | 0.009011 ± 0.000143 | 0.008689 ± 0.000078 | -3.58% |
| 12 | M128 N192 K128 E128x64 C1x1x1 | 0.012977 ± 0.000149 | 0.012679 ± 0.000045 | -2.29% |
| 13 | M128 N256 K128 E128x128 C1x1x1 | 0.015090 ± 0.000156 | 0.014728 ± 0.000046 | -2.40% |

除 shape10 外，其余 12 个 shape 的配对 95% 置信区间均完全小于 0。shape10 的
区间为 [-0.234, +0.308] us，说明两版性能等价，TMA 控制路径不是该 shape 的
可见关键路径。

## 4. NCU 证据

shape1 的 NCU 对照如下：

| 指标 | Baseline | Packed | 变化 |
|---|---:|---:|---:|
| Kernel duration | 3.5345 ms | 3.5238 ms | -0.30% |
| TMA load bytes | 55.893 GB | 55.893 GB | 0 |
| TMA load instructions | 5,745,600 | 2,872,800 | -50.00% |
| SM instructions | 365,499,268 | 322,471,984 | -11.77% |
| Shared memory/block | 214.016 KiB | 214.016 KiB | 0 |
| Active warps | 10.156% | 10.155% | 基本不变 |

TMA bytes 完全不变而 TMA load 指令精确减半，证明该实现确实把四次物理 TMA
发起合并为两次；性能收益来自控制/发起路径缩短，而不是少搬数据或提高 occupancy。

## 5. 适用边界

当前计时不包含 GMEM 预打包。该 ABI 适用于权重离线打包、多次复用，或上游算子
直接产生 packed 布局。若每次 GEMM 前都把动态 A/B 重新打包，预处理成本可能超过
1%–8% 的 kernel 收益，必须另做端到端评估。

本实验没有修改稳定源码 `dyy_test/op1/op1_best.cu`。所有实现和结果均保存在独立
实验目录中，可随时复测或撤销。
