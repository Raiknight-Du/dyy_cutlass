# op1: M_72a pure matmul

## 当前有效版本

OP1 的性能测试统一遵循远程仓库中的：

```text
dyy_test/PERFORMANCE_TEST_STANDARD_CN.md
```

当前稳定实现是 pure matmul：

- `ElementC = void`，不读取矩阵 C；
- `Identity(convert(accumulator))` epilogue；
- 不包含 alpha/beta、scalar broadcast 或 `LinearCombination`；
- D 为 BF16；
- 使用持久化随机 A/B/SFA/SFB 输入。

最终源码：

```text
/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op1/op1_best.cu
```

## 任意M/N/K测试入口

`op1_any_mnk.cu`保留OP1的pure matmul语义，命令行传入的
`--m/--n/--k`会直接作为问题尺寸，不会被13个内置Shape覆盖。

```bash
dyy_test/op1/bin/op1_any_mnk \
  --m=8192 --n=8192 --k=8192 \
  --shape-id=8192 --config-index=0 --iterations=10 \
  --dataset-root=dyy_test/op1/random_datasets_any_mnk
```

- `shape-id`仅作为`RESULT`记录标签；
- `config-index=0..7`选择已编译的OP1配置；
- 默认`config-index=0`，即`M256_N256_K256_E64x64_C2x1x1`；
- `config-index=-1`依次运行全部8个配置；
- 新Shape首次使用时可加`--generate-dataset --dataset-only`生成持久化随机输入。

## 有效性能数据

唯一正式的配置搜索和性能基线目录：

```text
/home/ubuntu/workspace/dyy/dyy_cutlass/
  dyy_test/op1/results/standard_retest_20260724/
```

测试口径：

- CUDA Event；
- warmup = 1；
- iterations = 10；
- 每个最终 shape 采集 25 个独立样本；
- 每次执行前确认 GPU 空闲；
- 使用固定的持久化随机数据。

13 个 shape 的最优配置 mean 之和为：

```text
20.318088784 ms
```

最终配置表：

```text
dyy_test/op1/results/op1_best_configs.csv
```

最终报告：

```text
dyy_test/op1/results/standard_retest_20260724/
  STANDARD_RETEST_REPORT_CN.md
```

## 有效瓶颈分析

13 个 shape 的 NCU、NSYS 和 SourceCounters 数据位于：

```text
dyy_test/op1/results/bottleneck_13shape_20260724/
```

分析报告：

```text
dyy_test/op1/results/bottleneck_13shape_20260724/ANALYSIS_CN.md
```

## 已清理数据

以下旧数据不再有效，已于 2026-07-24 删除：

- 受外部 GPU 任务影响的旧 Top-20/Top-3 结果；
- 非标准 iterations 口径的 recheck/retest 数据；
- 标准重测前的归档和重复 CSV；
- 旧 `op1_best` 源码副本与 pre-standard 二进制。

历史候选定义与复测脚本仍保留，用于复现搜索空间；它们不作为性能结论。
