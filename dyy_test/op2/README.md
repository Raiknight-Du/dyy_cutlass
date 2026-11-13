# op2: M_84a pure matmul baseline 与 best

`op2_baseline.cu` 固化配置：

```text
Sparse mode: 2:4
Tile:        256x128x256
Cluster:     2x1x1
Epilogue:    auto
```

## Pure matmul 语义

- `ElementC = void`，不读取矩阵 C；
- epilogue 为 `Identity(convert(accumulator))`；
- 不含 alpha/beta visitor、scalar broadcast 或 `LinearCombination`。

## 权重稀疏语义

目标运算是逻辑权重 `B[K,N]` 稀疏：

```text
D[M,N] = A_dense[M,K] * B_sparse[K,N]
```

源码通过转置等价式执行：

```text
D^T[N,M] = B_sparse^T[N,2K] * A_dense^T[2K,M]
```

其中稀疏 K 加倍，用于和 M_72a dense 保持相同有效 FLOPs。输出采用 column-major `D^T`，其物理字节顺序与原始 row-major `D` 相同，因此不需要额外输出转置 kernel。

## 固定源码

- `op2_baseline.cu`：CUTLASS 原始 baseline 配置，仅改为上述 pure matmul 语义。
- `op2_best.cu`：13 个 shape 分别使用标准搜索确定的最优配置。

## 搜索与最终测试

- 每个 shape 粗筛 1,080 个合法配置，共 14,040 条结果，失败 0；
- 每个 shape 的 Top20 使用独立进程做 5 次标准复测；
- 近似候选追加 20 次，使用 25 个样本确定 winner；
- 固定后的 `op2_best` 再对每个 shape 独立复测 25 次；
- 所有测试均使用同一组持久化随机输入、CUDA Event、1 次 warmup 和 10 iterations；
- 候选及 shape 顺序按 repeat 轮换/反转，每次启动前确认 GPU 空闲。

最终结果：

```text
13-shape baseline total: 36.931093 ms
13-shape best total:     29.160274 ms
overall improvement:     21.04%
```

详细配置、单 shape 性能和置信区间见：

- `ANALYSIS_CN.md`
- `results/op2_best_configs.csv`
- `results/op2_baseline_vs_best.csv`
- `results/standard_search_20260724`

旧的错误方向、旧 void-C、非标准 10 配置与混合 OP1/OP2 性能数据已删除，记录见 `OLD_DATA_CLEANUP_20260724.md`。
