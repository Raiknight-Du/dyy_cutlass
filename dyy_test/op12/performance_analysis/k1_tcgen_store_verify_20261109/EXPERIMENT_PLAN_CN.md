# K1 tcgen05与地址布局的独立和组合验证

## 范围与事前假设

固定FlashInfer提交e77a4a0d276367895c3b50a642fd8f326c03fb72，上游只读。
主配置B1/Hk=Ho=16/S32768/D128/FP16，原版每CTA一个64-token块，grid8192。
四组：baseline；tcgen05（只重构Wkk及必要结果交接）；layout（只重构T输出线程布局）；both。

- tcgen05：预期去除Wkk显式operand LDSM，增加TMEM及结果交接；不假定驻留仍为8CTA，不预设净提速。
- layout：预期STG sector/WF下降；必须计入重新读取、数据分发或重排的代价。
- both：两项可能相互影响，收益不可直接相加。

## 门禁与测量

1. 先审编译产物，确认目标指令实质改变，保存PTX/CUBIN/SASS、资源和launch信息。binary no-op不重复正式采集。
2. layout须与baseline逐元素一致；tcgen05须通过T输出误差检查及小shape/tail/BF16检查。异常、NaN/Inf、越界或竞态不得进入性能结论。
3. 通过编译和正确性后，以CUDA Event测完整单K1：20次预热、100次样本、3个独立进程，四组交错；分别采用独立驱逐前史和连续重复调用两种缓存条件。
4. GPU仅由一个Runner使用，连续三次间隔30秒空闲准入；监控外部进程、频率和温度。竞争样本作废，不终止外部任务。实验时申请1845MHz，结束恢复自动频率，保留实际频率轨迹。
5. NCU只用于动态计数、资源及指令证据，不使用replay时延计算提速。主配置四组各留原始ncu-rep与CSV；必要时对照重复。
6. correctness和编译失败保留日志。未通过的变体明确标为失败，不能将helper结果当完整K1收益。

## 主要输出

逐组报告：Event中位数/P5/P95/三轮波动、相对baseline加速比、正确性误差、寄存器/SMEM/TMEM/驻留、shared/global/Other/总Data WF、store sectors和指令变化。
原始文件、脚本和简要结论保存在本目录和B200独立目录k1_tcgen_store_verify_20261109，不修改正式模型及旧报告。
