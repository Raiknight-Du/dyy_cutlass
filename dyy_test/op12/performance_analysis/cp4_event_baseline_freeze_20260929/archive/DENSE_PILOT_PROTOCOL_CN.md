# OP12 CP4 Event 唯一基准冻结协议

Baseline ID: `OP12_CP4_EVENT_FP16_B1H16_P8_20260929_V1`

本协议先于正式数据确定。旧 Event / NCU / IKET 数据不覆盖、不筛选为本次结果。

## 测试范围

- FlashInfer 原版固定提交 `e77a4a0d276367895c3b50a642fd8f326c03fb72`，工作树必须 clean。
- NVIDIA B200，148 SM；不锁频，不修改驱动/环境。
- B=1，Hq=Hk=Hv=16，D=128；TND 输入；Q/K/V/O FP16，gate/beta FP32，initial/final state FP32。
- S=8192/16384/32768；公共 API `chunk_gated_delta_rule(use_cp=True)`；源码 host 选择的 C 必须为 1024/2048/4096，P 均为 8。
- Initial state / final state 均启用；无 checkpoint/state pool；kernel 内 QK L2 norm 关闭；K 在计时前 L2 归一化；scale=1/sqrt(128)。
- Q/V 正态分布，K 为归一化正态输入，gate=rand(0,1)，beta=sigmoid(rand(0,1))，initial=0.01*randn。固定每个 shape 的 seed，三个进程使用相同 tensor 内容。

## 唯一计时口径

1. 每次调用必须真实执行 K1→K2→K3→K4；K2 读取当次 K1 的 T，K3 读取当次 K2 的 M/N，K4 读取当次 K3 的 fixed state。禁止孤立循环、HOT prime 或人为 cache flush。
2. 在进程内临时包裹四个内部 Python stage 调用，仅在 CUDA Graph capture 时记录五个边界 Event：K1 前、K1 后、K2 后、K3 后、K4 后。不修改仓库、公开 API 或 GPU kernel。
3. 捕获包含连续 100 次真实 CP 调用的单流图。图中应严格重复四个 kernel 与五个 Event 节点；必须从 DOT 核对 kernel 名、grid/block 和串行依赖。Event 使用 enable_timing=True, external=True。
4. 正式捕获前 JIT 完成，4 组 rotating Q/K/V 输入已就绪，20 次完整调用预热。正式 graph 再运行一次作为图预热、不入样本；随后再做20次原生完整调用，使正式前史回到约定的自然四阶段循环。正式 graph 一次回放，读取100组分段 Event。
5. S 的三个值分别以3个独立进程运行；正式样本每shape为300。正式顺序按 trial 轮转 shape。每个进程完整保留所有样本，不删首项/离群项。
6. 另捕获相同100次调用、仅首尾两个 Event 的原生图，重复5次。其每调用平均值只作为分段 Event 扰动检查，不作为四个kernel的替代 baseline。必须标注 median(stage sum) 与 native batch mean 并非同一种统计量。
7. 正式 baseline：三个进程各自的100样本中位数再取中位数。P5/P95和样本CV另列，CV>5%时追加一轮同协议进程而不删除原轮；追加数据单独注明，冻结采用全部独立进程中位数的中位数。
8. 时间单位 µs。阶段边界 Event 的分辨率与标记开销不从结果中主观扣除。冻结的是“规定标记方案下、真实连续 CP 图中的阶段 Event 区间”。若扰动超过5%，不能宣称无扰动；必须保留并说明，不拟合修正。

## 正确性与归档

- 捕获前后与未包裹的公共 API 比较 O/final state，要求完全一致；每个buffer输出有限；initial 不被改写。
- 另以非CP公共 API 对照，保留差值指标（atol=rtol=0.04 检查）；不将原版算法路径差异误归为 Event 修改。
- 两种图必须核对四kernel路径、配置和顺序，正式图每个stage100次。
- 固定源码、输入tensor内容SHA、每进程环境/时钟/温度/功耗、原始样本、图DOT、日志、正确性记录全部归档。
- frozen_baseline.json 与 FREEZE_MANIFEST.json 构成不可变基准；后续只新增版本，不原地覆盖。

## 消融使用规则

- 主性能差值和加速比引用该 baseline ID 的冻结 Event，不能用旧的 isolated-kernel 或 NCU replay 时间替换。
- 消融须嵌入同样的 K1→K2→K3→K4 图、相同输入buffer内容/轮转/shape/C/dtype/state开关/stream/Event布局。不同上下文不得直接相减；尤其只跑K3的旧消融属于历史隔离实验。
- 可增加同轮未修改 B0 作为环境漂移门禁，不生成替代 baseline。B0 相对冻结基准偏离超过5%时，不给出以该漂移轮为主的因果结论，保留记录并复测。
- 数值不正确的纯性能消融需单独标为 semantics-breaking；不能据其输出认定实现正确。
- 未实测 DRAM/L2 本次不推断实际 cache 命中率；只能冻结自然连续 CP 前史，不声称每个输入来自 HBM。

方法依据：[PyTorch 2.11 CUDA Event](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.Event.html)、[CUDA Graph](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.CUDAGraph.html)。
