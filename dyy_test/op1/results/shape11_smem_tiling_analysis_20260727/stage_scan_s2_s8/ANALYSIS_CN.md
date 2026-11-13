# Shape11 Pure Matmul Stage2-Stage8 扫描

## 实验配置

```text
Problem  = M16384 x N768 x K128
Tile     = M128 x N192 x K128
Epilogue = E128 x 64
Cluster  = 1 x 1 x 1
K iteration = 1
```

所有变体来自同一个 pure-matmul 二进制，使用同一随机输入。计时采用
40轮随机交错配对采样，每次100 iterations。唯一变量为显式 mainloop
StageCount；另保留 Auto8 对照。

## 结果

| 变体 | SMEM/CTA | Mean | 95% CI | 相对Auto8 | Block limit by SMEM | Eligible/SMSP | No Eligible | Long SB | Barrier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Auto8 | 219 KiB | 8.2605 us | +/-0.0039 us | 0 | 1 | 0.1928 | 82.11% | 3.933 | 0.851 |
| Stage2 | 81 KiB | 8.4484 us | +/-0.0222 us | +2.28% | 2 | 0.1868 | 82.66% | 4.186 | 0.955 |
| Stage3 | 104 KiB | 8.3453 us | +/-0.0124 us | +1.03% | 2 | 0.1880 | 82.50% | 4.052 | 0.867 |
| Stage4 | 127 KiB | 8.3047 us | +/-0.0123 us | +0.53% | 1 | 0.1891 | 82.43% | 4.050 | 0.857 |
| Stage5 | 150 KiB | 8.2944 us | +/-0.0084 us | +0.41% | 1 | 0.1889 | 82.41% | 3.982 | 0.864 |
| Stage6 | 173 KiB | 8.2647 us | +/-0.0034 us | +0.05% | 1 | 0.1905 | 82.31% | 4.008 | 0.878 |
| Stage7 | 196 KiB | 8.2588 us | +/-0.0038 us | -0.02% | 1 | 0.1913 | 82.25% | 3.942 | 0.875 |
| Stage8 | 219 KiB | 8.2634 us | +/-0.0034 us | +0.04% | 1 | 0.1928 | 82.12% | 3.937 | 0.856 |

## 结论

Stage2到Stage8并非性能相同。性能随StageCount增加逐步改善，到Stage6
后进入平台：

```text
Stage2 < Stage3 < Stage4/5 < Stage6/7/8 ~= Auto8
```

由于只有一个K iteration，该变化不是更多真实K tile预取产生的。
StageCount仍会改变`producer_tail`遍历的barrier数量、SharedStorage地址
布局和最终生成的控制指令。更多未使用stage让producer在抵达真正依赖
consumer释放的stage0 barrier前执行更多立即完成的检查，改变短kernel中
eligible指令供给和等待分布。

NCU显示No Eligible从Stage2的82.66%逐步下降到Stage8的82.12%，Long
Scoreboard从4.186下降到3.937，与时延趋势一致。Stage3到Stage4虽然让
SMEM block limit从2变为1，但性能没有突变；CTA occupancy不是主要原因。

显式Stage8与Auto8具有相同SMEM和SM指令数，时延仅相差0.036%，说明
Auto policy在当前配置下确实解析为Stage8。
