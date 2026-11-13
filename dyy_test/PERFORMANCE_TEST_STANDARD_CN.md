# B200 算子性能测试标准

本标准用于 CUTLASS、cuBLASLt、cuSPARSELt 及其变体之间的最终性能比较。

## 标准计时口径

1. 使用 CUDA Event，在执行 kernel 的同一 CUDA stream 上计时。
2. 计时前执行 1 次 warmup，并在 warmup 后同步 stream/device。
3. 正式计时连续执行 10 次算子，报告 `event_elapsed_ms / 10`。
4. 每个 shape/config 至少独立重复 5 轮。
5. 报告 mean、standard deviation、minimum 和 maximum。
6. 最终结果不得取单次最小值，也不得把不同迭代数的数据混合比较。

## 测试环境

1. 每轮开始前确认 GPU 没有其他 compute process。
2. 对照方案采用交替或轮换顺序，抵消动态升频、温度和功耗状态影响。
3. 不主动锁频时记录这一事实；跨日期绝对值只作为参考，配置结论优先采用同场结果。
4. 输入使用持久化随机数据；所有候选使用相同数据文件。
5. 禁止使用全零、全一或其他容易触发硬件压缩的输入。

## 计时边界

计时区间只包含正式算子调用，不包含：

- 内存分配和释放；
- 数据生成、文件读取以及 H2D/D2H；
- descriptor/handle/plan 的首次创建；
- correctness 检查；
- 编译和 profiler 启动。

被比较实现必须具有相同的输入、输出数据类型、矩阵语义和输出写回要求。

## 配置搜索

配置搜索也使用 `1 warmup + 10 iterations`。当候选较多时：

1. 第一阶段测试全部候选，至少 5 repeats，并轮换候选顺序。
2. 第二阶段对每个 shape 的 Top-3 独立执行相同标准。
3. 差异小于测量波动时视为并列，稳定源码保留既有或结构更简单的配置。

更长的 50/100/1000 iterations 可作为持续负载、稳定性或功耗敏感性实验，
但必须明确标记为 stress/sustained 测试，不得写入标准性能基线。

## Profiler 数据

- Nsys kernel duration：用于分析单个 kernel 本体和 kernel 间隙。
- NCU duration/metrics：用于分析硬件行为和性能瓶颈。
- Nsys/NCU 数据不得替代 CUDA Event 标准基线，也不得与旧 CUDA Event 表直接混算。

## 标准标签

标准结果建议标记为：

```text
CUDA Event, warmup=1, iterations=10, repeats>=5, persistent random input, GPU idle
```
