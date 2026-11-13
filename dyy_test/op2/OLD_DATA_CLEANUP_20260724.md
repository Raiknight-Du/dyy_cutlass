# OP2 旧数据清理记录

日期：2026-07-24

## 删除范围

- `m84a_pure_matmul_2k_20260722`
  - 输入矩阵方向不符合“逻辑 B[K,N] 为稀疏权重”的最终语义。
- `m84a_voidc_full_legal_scan_20260721`
  - 使用旧 void-C 路径和未转置输入，性能结果不能用于当前 pure matmul OP2 排名。
- `m84a_weight_sparse_transposed_2k_20260722`
  - 矩阵方向正确，但只测试 10 个配置，且计时次数与当前统一性能标准不一致，不能作为最终最优结果。
- `m72a_m84a_voidc_baseline_best_20260722`
  - OP1/OP2 混合旧汇总；OP1 已由 `dyy_test/op1` 的标准重测结果替代，OP2 部分不符合当前测试口径。

## 保留内容

- `dyy_test/op2/op2_baseline.cu` 及其可执行文件。
- 13 个 shape 的持久化随机输入：
  - `dyy_test/op2/random_datasets`
  - `dyy_test/op2/random_datasets_backing`
- 旧扫描得到的合法配置边界与分组元数据：
  - `dyy_test/op2/search_space/legacy_legal_scan`
- 转置权重稀疏参考源码及旧 group/static probe 模板：
  - `dyy_test/op2/references`

## 后续结果口径

OP2 的最终性能结果只写入 `dyy_test/op2/results`，并统一使用：

- pure matmul；
- 逻辑 `B[K,N]` 为稀疏权重，通过转置 GEMM 实现；
- 稀疏 K 为 dense K 的 2 倍，以保持有效 FLOPs 一致；
- CUDA Event；
- warmup 1 次、每次 10 iterations、至少 5 个独立 repeats；
- 每个候选独立进程测试，候选顺序轮换或反转；
- 每次测试前确认 GPU 空闲；
- 所有测试复用上述持久化随机输入。
