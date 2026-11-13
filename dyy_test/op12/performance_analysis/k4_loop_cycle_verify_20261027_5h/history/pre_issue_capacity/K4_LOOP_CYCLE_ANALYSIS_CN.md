# K4 稳态循环周期与shared/LSU依赖核验

## 结论与验收范围

**本轮得到从原生工作量、具名条件速率和依赖传播计算的主体循环预算，三组shape相对同调用返回斜率的偏差为−9.31%至−0.32%，满足本轮±10%的主体目标。** 这不是整阶段实测时间求和，也不是把目标Gap反推成一个资源参数。保留旧文件，正式模型未自动启用。

|S / CP partition C|观测 cy/块|条件模型 cy/块|带符号相对偏差|
|---|---:|---:|---:|
|8192 / 1024|3175.3|2976–3131|-6.28%～-1.40%|
|16384 / 2048|3163.7|2869–3131|-9.31%～-1.03%|
|32768 / 4096|3141.2|2869–3131|-8.67%～-0.32%|

固定B1/H16/D128、FP16、P=8，算法块64token，单CTA递推16/32/64块、grid128、384线程。观测按每次调用中CG1返回点对块号独立OLS拟合，再取3进程×60调用的斜率均值；S8192/S16384/S32768分别使用返回点4..11/8..23/8..55。不拼接不同warp均值、不用NCU replay时延、不用请求1845MHz把µs换成cycle。

理论调度也在上述相同返回点窗口内拟合。S8192只有16块，有限窗口仍存在调度相位效应，不能把长循环的2869cy最小节拍直接当其窗口斜率；同窗口结果为2976–3131。其余两组为2869–3131。外部初始就绪条件仍属模型假设，未用实测INIT时间校准。

比较对象是**插桩返回节拍**。已有插桩局部偏差未完全界定，所以本轮不宣称“原版全kernel性能已解释90%”，不覆盖INIT和首尾。模型/观测比值也不是原始Gap的因果解释比例。机器台账另保留相对旧2296-cycle服务情景的区别，不能把数值预算上移本身视为物理原因已经逐项分离。

![循环预算与观测](figures/K4_LOOP_CYCLE_COMPARISON.png)

历史4-reader流的246.376B/cy仅作独立敏感性，不替换256主情景。它给出2924.25–3160.25cy/块；S32768最大值相对3141.20cy的偏差为**+0.61%**，明确保留超估。此敏感性也仍在±10%内。范围来自明确条件场景，不是统计置信区间或绝对硬件上下界。

## 1. MMA隐式读shared：该新增什么账

**是隐式操作数读取，不是额外生成LDS指令。** tcgen05的SS/TS操作数按descriptor从shared读取；因此要在shared读资源中补齐TC客户端需求，但不能增加shader的LSU指令数，更不能把已含仲裁重试的硬件LSU WF再与同一TC需求双重收费。

|GEMM|M×N×K|atom/块|纯算术cy，8192FLOP/cy|隐式shared B/块|TC客户端WF/块|
|---|---|---:|---:|---:|---:|
|QK|64×64×128|8|128|32,768|256|
|KS|128×64×128|8|256|16,384|128|
|QS|128×64×128|8|256|16,384|128|
|NV|128×64×64|4|128|8,192|64|
|O|128×64×64|4|128|8,192|64|
|dS|128×128×64|4|256|16,384|128|
|合计||36|1152|98,304|768|

shader地址模型另有读616、写260WF/块。TC implicit read为96KiB/128B=768WF/块，其中QK同时读A/B，其他五组主要为shared B；显式TMEM字节另立账。原版的专用`l1tex__data_pipe_tc_wavefronts.sum`=6,291,456=768×8192，与上述分子精确一致。**本轮MMA算术统一8192FLOP/SM-cycle参考，不凭“测到半速”认定固定一半硬件通路**；QK的128算术cy与256操作数读WF是两种不同资源需求。

TMA还涉及每块448 shared-write WF和128 shared-read WF；W11 gate另有4 write WF。它们已列入全量资源账。TMA与所有shared客户端是否完全共用同一服务通路尚无充分匹配数据，不能把四类计数无条件全加成一个“LSU时间”。

## 2. 地址核查、独立实验与原版干预

CG0全部132个shared PC完成逐lane地址/谓词枚举，动态pred-on及Source WF全部匹配，未发现原生地址自身的额外bank conflict。ring stage仅改变已对齐基址，目标三shape均为完整64-token块。独立探针覆盖CG0/CG1代表性的LDS、LDSM、STSM六类序列；不是声称每个PC都测了独立吞吐。

24份NCU显示：只改变是否并行运行SS-QK，目标shader Source WF不变，读指令族的硬件shared读WF增加；例如CG0 LDSM64→320WF/轮、CG1 LDSM128→约384WF/轮，TC读恒为256WF/轮。STSM写WF基本不增加。严格1%扩点数据中，LDSM并行周期近似两种读需求之和；STSM与TC读并行则大部分可覆盖。因此**shared读耦合有实证，不能采用“所有shared读写都同池串行相加”的模型**。

为了定位原版，保持完整冷K1→K2→K3→K4缓存前史，只移动CG1等待位置：KSJOIN在V读取前等KS；QSJOIN先等KS和QS再读V；GATEJOIN把等待移到gate/V读取前。12份有效NCU，输出/state位精确一致，MMA/TMA/TMEM/转换/主要数学动态工作量及Source WF一致；原版自身已有的local指令保留，没有新增spill。

|原版/变体|Source shared WF|TC WF|硬件shared读WF，3份均值|读冲突计数，3份均值|
|---|---:|---:|---:|---:|
|B0|7,178,496|6,291,456|7,050,749|1,980,253|
|KSJOIN|7,178,496|6,291,456|5,634,151|582,457|
|QSJOIN|7,178,496|6,291,456|5,620,946|561,387|
|GATEJOIN|7,178,496|6,291,456|5,430,667|382,326|

![原版等待位置对照](figures/K4_SHARED_ARBITRATION_CONTROL.png)

这是CG1 gate/V读取与未完成MMA的跨客户端仲裁证据；改变了其他工作交错，故不能把全部计数差唯一分配到某条LDS PC，也不能当成该步骤独立时长或实际优化收益。历史Source总数少512WF来自固定控制/atomic统计；本批同CUBIN条件统一为7,178,496，循环仍为876WF/块。

ALU补充对照使用同一CUBIN、32条独立链、4+4warp，保留4096 PACK lane操作和4096三实源IADD3 lane操作/N。修正为只统计活跃warp后，单独PACK约65.375cy/N、IADD3约68.25cy/N，并行约128.625cy/N；对应Event斜率全部通过严格1%扩点门槛，6份NCU将两类目标工作精确记入ALU。它支持本混合条件下合并服务需求，不升级为普适裸硬件速率；IMAD按官方pipe定义排除在此ALU项之外。早期把inactive warp混入中位数的结果已降级，原始数据保留。

## 3. 原生预算如何传播成每块cycle

每条已执行的native指令先由原生工作量和适用速率形成服务需求；PACK按两个标量输入/结果计，FMUL2按两个FP32结果计，ISETP按执行lane而非布尔true数计。保留同warp真实PC顺序、原生控制字中的最小issue间距和scoreboard wait；编译stall不是硬件RAW latency。

```text
ready_i = max(上一个issue允许时刻, RAW/scoreboard就绪, pipeline/buffer可用)
service_start_i = max(ready_i, 对应资源可接收时刻)
service_end_i = service_start_i + work_i / conditional_rate_i
consumer_ready_i = service_end_i + 已独立限定的完成延迟余项
```

控制字、数据RAW、实际异步MMA提交/完成、Ainv三槽、QK两槽、Q/K ring及单槽state/QS反馈均保留。native issuer W10的描述符/寻址/控制也入模型。unknown谓词/选择等容量只保留可证明issue下界，不填一个“Gap税”。

特别核实了`STTM → wait::st → mbarrier.arrive → consumer`。原版PTX在2710、2846、3158、3302、3591行有`tcgen05.wait::st`，不能把发布flag发射就当数据已完成。模型在可证明完成端点加入已有store服务完成的依赖边；MEMBAR/shared发布同样保留。相关边不能再另加一份“store阶段耗时”。

**TMEM吞吐与读结果就绪分开。** 16dp256显式read以及QS write沿用已核条件流256B/cy，其他具名子型保留512假设；本项总显式服务为1120cy/块，仍不是墙钟。新单指令实验仅含LDTM、wait及最小地址RAW链。逐字节替换16个LDTM为NOP，并关闭这16个指令自己的WB/RB，其他native字节/动态计数不变；实际寄存器、输入、SM ID和计时端点保留。

|指令子型|有读Pload cy/单次原语链|NOP对照Pnop cy/链|限定完成范围|
|---|---:|---:|---:|
|32x32b.x32|49.375|25.375|25–50cy|
|16x256b.x8|97.375|25.375|73–98cy|

采用`L ∈ [Pload−Pnop+1, Pload]`，上界向上取整；区间保持控制成本分离的不确定性，不把49.375/97.375当K4阶段权重或裸硬件延迟。用于DAG的额外完成部分为`max(0,L−Bwarp/rate)`，以免重复收取已计的读服务量。原版短burst、TMEM相位/列位置、四warp并发仍属适用条件，不声称全地址通用延迟常数。

独立统计门禁均保存≥4个递增N、20预热、30样本、3进程；native/Event两域相邻尾部三点窗和跨进程斜率变化严格<1%、R²≥0.99、CV及P5–P95波动检查。未过的初轮保留，最多扩到N16384，不用放宽门槛来填平目标Gap。

![条件资源调度](figures/K4_CONDITIONAL_RESOURCE_SCHEDULE.png)

以下只是各native PC服务预算的分步骤汇总；**不把它们直接串行相加或逐阶段取max后称为最终关键路径**。

|步骤|PC范围|资源服务cy（仅非零已记项）|
|---|---|---|
|CG0 T门控变换|0x2330–0x3e40|shared_read:208 / shared_write:64 / ALU_PACK:322 / FP32:120 / FP16:4 / SFU:136 / IMAD:30 / issue_only_floor:69|
|CG0 QK后处理|0x3e50–0x5470|TMEM:64 / shared_read:144 / shared_write:64 / ALU_PACK:224 / FP32:94 / SFU:136 / IMAD:40 / issue_only_floor:34|
|CG1 state读回与A发布|0x6fb0–0x79e0|TMEM:192 / shared_read:4 / ALU_PACK:128 / FP32:128 / IMAD:2 / issue_only_floor:26|
|CG1 state decay发布|0x79f0–0x7a50|TMEM:128 / issue_only_floor:3|
|CG1 gate与V gather|0x7a60–0x7c50|shared_read:260 / IMAD:6 / issue_only_floor:4|
|CG1 KS residual/VKS发布|0x7c60–0x8300|TMEM:160 / ALU_PACK:64 / FP32:64 / FP16:128 / issue_only_floor:7|
|CG1 QS处理与发布|0x8310–0x8820|TMEM:256 / FP32:132 / SFU:32 / issue_only_floor:8|
|CG1 NV/Delta处理与发布|0x8830–0x9080|TMEM:192 / ALU_PACK:128 / FP32:76 / SFU:96 / IMAD:2 / issue_only_floor:13|
|CG1 O暂存与返回|0x9090–0x94d0|TMEM:128 / shared_write:128 / ALU_PACK:64 / IMAD:6 / issue_only_floor:24|

完整1384条PC账单、控制位及公式见工作簿/CSV。CG0和CG1共同的ALU/SMEM/TMEM服务按资源约束传播；FP16/FP32逻辑共享情景、读完成端点的边界分别计算，形成2869–3131而非一个反调出来的点。32dp读完成范围在这些场景中主要被覆盖；16dp短读的消费依赖、shared读耦合和CG1发布/递推是更有影响的组合。改变边会改变FIFO交错，因此情景差值不能当作可加的独立步骤耗时。

## 4. NCU一致性与尚未量化的成本

|指标|原生分子公式|推导|NCU原始观测|
|---|---|---:|---:|
|UTCHMMA warp-inst|36×8192＋0×128|294,912|294,912|
|LDTM warp-inst|52×8192＋0×128|425,984|425,984|
|STTM warp-inst|64×8192＋16×128|526,336|526,336|
|Source shared WF|876×8192＋18×128|7,178,496|7,178,496|
|TC implicit shared WF|768×8192＋0×128|6,291,456|6,291,456|

原版硬件shared读/写WF的额外量与仲裁counter接近，但两者定义不完全相同，保留固定项/atomic及其他服务差异。模型用唯一的操作数读服务和队列依赖解释阻塞，不把retry WF当额外算法字节。没有用stall百分比乘kernel总cycle分摊阶段时间。逻辑HBM、L2请求、Source地址WF、硬件shared/TC WF也没有混为同一物理流量。

原版B0三份NCU的全action均值为DRAM读472,293,973B、写142,104,405B，L2总sector约31,285,207。按8192个完整块仅作工作量摊销，DRAM读写约74,999.8B/块；按用户32B/SM-cycle代理为2343.7cy。L2请求约122,207.8B/块，按64代理为1909.5cy。两者均低于主体条件模型周期，未形成容量矛盾。这里包含INIT/descriptor固定项及缓存前史；不能把全action计数除以内层trace时间冒称实测带宽，也不能把额外DRAM写直接归为K4新计算或已证实的某一脏写回来源。

主要剩余项：稀疏EX2的精确服务粒度、部分谓词/选择和uniform容量、真实队列仲裁、短MMA完成通知、原版local/L1返回，以及TMA/gate/O的晚到暴露。独立共享实验支持读耦合，但未唯一确定每个native warp的实际队列优先级。外部Q/K/V/T/gate在数值主体DAG中按稳态已就绪情景处理；其非零工作量、全冷2312cy/T来自L2时2056cy代理和TMA共享服务另列，并未以未知=0声称已完整建模。

剩余量约10–295cy/块（取决于shape与条件情景），可以由上述完成/通知和调度差异以及插桩偏差共同贡献；现有证据不允许给它们分配独立固定cycle。**本轮停止于主体周期匹配与主要机制证据成立，不进一步开发优化，也不宣称每一个剩余cycle都解释了。**

## 5. 复算与证据

- `K4_LOOP_CYCLE_REVIEW_20261027.xlsx`：循环对照、步骤服务、逐PC公式、NCU对账、独立试验、参数边界。修改速率后须先重跑Python调度，循环周期列为调度快照。
- `recompute_final_model.py` + `native_pc_service_model.py` + `encoded_async_engine.py`：先求条件调度，最后才读取目标观测，禁止反调参数。
- `derived/FINAL_CYCLE_MODEL.json`、`FINAL_SHAPE_COMPARISON.csv`、`STEP_RESOURCE_NATIVE_LEDGER.csv`、`PHASE_RESOURCE_BUDGET.csv`：机器可读推导及每PC工作。
- `cg0_address_audit.py`、`derived/CG0_LANE_ADDRESSES.csv`：CG0地址枚举；`MATCHED_COUNTER_SUMMARY_FINAL.json`及原始24份NCU：指令族隔离/并行验证。
- 原始新证据：`raw_evidence`；远端目录 `/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/performance_analysis/k4_loop_cycle_verify_20261027_5h/`。目录中的失败轮、源码、PTX/SASS/CUBIN、Source/硬件CSV、Event/clock样本全部保留。
- 旧条件速率的独立审阅快照在`inputs/historical_reviews`，原始路径和hash列入`derived/FINAL_DELIVERY_DATA.json`；历史目标trace未覆盖。
- 所有候选均`model_enabled=false`、`effective_model_value=null`，不覆盖`K4理论分析.xlsx`，不修改FlashInfer固定源码。

公开口径：[NVIDIA NCU pipeline/WF定义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#pipelines)、[NVIDIA PTX tcgen05同步与shared访问](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions)、[CUDA CC10.0吞吐表](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-c-best-practices-guide/index.html#arithmetic-instructions-throughput-native-arithmetic-instructions)、[CUTLASS Blackwell相对Hopper吞吐](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md)。
