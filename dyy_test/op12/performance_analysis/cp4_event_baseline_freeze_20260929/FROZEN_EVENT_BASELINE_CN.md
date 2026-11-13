# OP12：真实 CP 四kernel Event 基准冻结

冻结ID：`OP12_CP4_EVENT_FP16_B1H16_P8_20260929_V1`。唯一数值权威为本目录 `frozen_baseline.json`，SHA256：`f84842965b4aa3245000ded3e91c764344837abe1bee801c8fee154419b6db3f`。旧数据保留，不再作为符合本口径的新消融baseline。

## 1. 冻结结果

单位均为µs。K1–K4为稀疏累计边界 Event 的均值差，再取三个独立进程估计的中位数；不是同一调用内四个直接时间戳。原生CP端到端为无内部计时节点的完整四kernel图时长/调用数，单独实测、单独冻结。

| S | C | K1 T | K2 M/N | K3 Fixup | K4 Final | 阶段之和 | 原生CP端到端 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1024 | 23.345 | 35.033 | 26.498 | 55.362 | 140.238 | 136.851 |
| 16384 | 2048 | 38.445 | 60.604 | 25.671 | 84.366 | 209.086 | 205.461 |
| 32768 | 4096 | 72.501 | 113.281 | 27.166 | 146.159 | 359.107 | 363.336 |

阶段和相对原生端到端分别为 +2.475%、+1.765%、−1.164%；**没有按端到端结果缩放四个stage**。各stage跨进程中位数之和也不一定等于每进程总时长的跨进程中位数。

## 2. 真实执行环境与数据前史

- GPU：NVIDIA B200，148 SM；驱动580.159.03。
- FlashInfer：固定提交 `e77a4a0d276367895c3b50a642fd8f326c03fb72`，从op12归档源码导入；正式运行前后工作树clean。系统同名包的版本号不作为源码版本。
- PyTorch 2.11.0+cu130，CUDA 13.0，CuTe DSL 4.7.0，cuda-python 13.2.0。
- B1/Hq=Hk=Hv=16/D128，TND，Q/K/V/O FP16，gate/beta及initial/final state FP32。
- 初始state和final写回均启用；无checkpoint/state pool；kernel内QK L2归一化关闭。K在计时外L2归一化；scale=1/sqrt(128)。Q/V正态分布，gate=rand(0,1)，beta=sigmoid(rand(0,1))，initial=0.01*randn。
- 三shape均P8；K1 grid=2048/4096/8192，K2=128，K3=32，K4=128；block线程数依次128/384/256/384。
- 4组Q/K/V固定内容轮转。每次执行真实K1→K2→K3→K4，M/N来自当次K2，fixed state来自当次K3；没有孤立K3、HOT prime或人为cache flush。workspace按上游缓存机制复用，没有改变算法数据组织。
- GPU仅一个实验执行者；未修改GPU时钟、功耗限制、驱动或环境。

所有input tensor的shape、dtype、内容SHA均保存。完整tensor保留在B200本证据目录的 `frozen_inputs/S8192|S16384|S32768/inputs.pt`，本地归档保存manifest与存储SHA，后续消融应直接复用它们。

## 3. 为什么不是直接密集五边界计时

预试中，完整CP链每次都放五个边界Event时，图总时长较无内部标记图增加约12.81%。这个版本仅作为方法诊断，不纳入冻结。最早一轮预试因DOT格式解析失败停在图验证阶段，也完整保留，不算性能样本。

正式方法在每4次完整CP调用中只选1次计时，均衡覆盖4个buffer。每种图400次完整CP，1600个kernel，202个Event；每种probe100个Event区间，20次原生完整调用预热。三个shape各3个独立进程，共4500个累计Event观测；每进程另有5个无内部Event的原生batch观测。

| Probe | 时间戳区间 | 其余计算 |
|---|---|---|
| P0 | K1前两个相邻Event的空区间 | 随后仍执行K1、K2、K3、K4 |
| P1 | K1前→K1后 | 随后仍执行K2、K3、K4 |
| P2 | K1前→K2后，前缀内无Event | 随后仍执行K3、K4 |
| P3 | K1前→K3后，前缀内无Event | 随后仍执行K4 |
| P4 | K1前→K4后，前缀内无Event | 一次完整CP |

每进程计算 `Ki=mean(Pi)-mean(Pi-1)`。P0为空Event区间的实际观测，不是按目标kernel拟合的常数。不把来自不同probe的100个样本相减后冒充“100次直接kernel时延”。统计估计、原始区间、P5/P95/CV和差分标准误均保留。

Event使用 `enable_timing=True, external=True`，图捕获前已初始化。仅在进程内临时包裹上游Python stage函数记录Event；未修改上游文件、公开API或GPU kernel。每个图的DOT证明四kernel名称、grid/block/SMEM和严格串行节点链均正确。[PyTorch CUDA Event文档](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.Event.html)、[CUDA Graph文档](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.CUDAGraph.html)。

## 4. 验证结果与局限

- 9/9进程全部通过。图执行后的O/final state与未包裹公共API完全一致；所有输出有限；initial state未被修改。
- CP与非CP均通过 `atol=rtol=0.04`；O相对L2约9.64e−5，final state约3.10e−4～3.17e−4。
- 正式probe batch相对原生batch的最大绝对差，三shape分别1.698%、1.327%、4.001%；非空累计probe的最大样本CV分别1.453%、2.631%、2.486%。这些差异包含标记、缓存和频率/热状态变化，不能全归为Event自身开销。
- K3三个进程的范围：S8192为26.197～26.686µs；S16384为25.598～25.712µs；S32768为25.591～28.011µs。S32768的大前缀相减放大测量与前史差异，**27.166µs不是已证明无偏的kernel本体时延**。
- 未锁频。S32768采样中SM频率随高负载从1965MHz降低，probe前采样曾到1830MHz，结束约1852～1860MHz；功耗接近1000W。采样不是每个Event窗口的积分频率，因此不做“统一乘1.965GHz”的精确cycle换算，也不能把全部差异解释为pipeline瓶颈。
- 本次没有采集DRAM/L2流量：只冻结自然连续CP前史，不承诺数据一定在HBM或一定命中L2。
- 阶段差分之和接近原生端到端，不证明每个stage误差相互独立或各自准确。亚微秒收益需重复观测/配对检查；不能仅凭冻结单点宣称关键路径闭合。
- 这是GPU-only CUDA Graph执行基准，不是包含Python dispatch/JIT/workspace allocation的API wall time。NCU replay、IKET计时不参与数值冻结或时延修正。

### 多进程是否一致，以及是否清空L2

三次进程的源码、tensor内容SHA、S/C/P、dtype、state开关、单stream、buffer轮转、预热和图结构一致；这不等于动态硬件状态相同。S32768各次K3差分分别25.591、27.166、28.011µs。P2前SM频率采样为1875/1860/1867MHz，P3前为1875/1875/1867MHz；它们是采样值而非各计时窗口积分频率。实际L2命中率本次没有观测，不能把三次差异全部归因于L2。

独立进程不是本协议中“L2已清空”的验证步骤。后续需要一致的**缓存前史**，不应在K1/K2/K3/K4之间强制驱逐缓存；那会破坏K3消费刚由K2生成的M/N这一真实场景。当前自然稳态基准维持不驱逐、完整CP预热和轮转。

若分析冷启动敏感性，可以另立ID，在整条CP链之前进行经验证的缓存驱逐，分别测cold-start和完整链预热后的steady-state；不能把cold对照混入V1。简单的大buffer驱逐不是所有L2行都被硬件强制清空的保证。后续NCU也要注明cache-control，不能把默认replay缓存控制的流量直接视作本Event稳态前史。[NCU Cache Control](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control)。

## 5. 后续消融baseline规则

1. 报告中的绝对baseline、节省时间和加速比直接引用该冻结ID，不再从旧表/旧NCU/旧隔离实验任取一个值。
2. 消融嵌入同一真实四kernel链，保持固定输入SHA、轮转、shape/C/dtype/state开关、单stream、五种probe位置与均值差统计一致。
3. 同轮未修改B0只作漂移检查，不能替换冻结数值；任一stage或原生端到端偏离冻结点超过5%时进入review/retest。K3已有范围及差分误差要同时列出；通过5%门禁也不自动证明小收益显著。
4. `runner/compare_to_frozen.py` 已提供只读上下文检查与比较：输入/环境/grid/图节点/Event布局不匹配直接拒绝，stage公式不一致拒绝；收益始终减冻结值，不减同轮B0。
5. 旧K3 isolated、不同private编译路径、HOT/PERT以及其他P配置仅保留历史证据，不能与本基准直接相减。数值错误的性能消融仍需明确标为semantics-breaking。
6. 新轮测量只能创建新版本ID，不覆盖 `frozen_baseline.json`。如后续要冻结锁频/不同缓存模式基准，须另立ID，不能静默混入V1。

## 6. 文件与复现

本地目录：`E:/Users/27194/Documents/Cutlass/op12_flashinfer_gdn_b200_analysis_20260825/cp4_event_baseline_freeze_20260929/`。

B200目录：`/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/performance_analysis/cp4_event_baseline_freeze_20260929/`。

- `frozen_baseline.json`：权威数值与固定上下文；`FREEZE_MANIFEST.json`：SHA清单。
- `outputs/cp4_event_freeze_0929/OP12_CP4_EVENT_FROZEN_BASELINE_20260929.xlsx`：公式驱动summary、9进程差分、累计Event统计和native batch观测。
- `formal/*/prefix_event_samples.csv`：4500个原始累计区间；`native_batch_samples.json`：45个原生batch观测。
- `formal/*/graph_validation.json`、6份DOT、`validation.json`、`environment.json`：路径、配置、正确性、时钟与热状态证据。
- `PROTOCOL_CN.md`、`runner/measure_cp4_event.py`、`runner/run_matrix.py`：计时实现及矩阵运行。
- `runner/compare_to_frozen.py`：后续消融只读比较门禁；本次未重新执行消融。
- `pilot/`和`logs/`：包含方法预试和失败记录，不混入正式样本。

原始正式命令见 `formal_process_records.json`。复现必须使用新的output-root，不能在本冻结目录重新运行 `--mode formal`；脚本对已有运行目录和冻结JSON使用排他创建，防止覆盖。
