# R6 旧 B0/V0 全 CP 只读复用准备（无 GPU 准入）

旧根 `/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/performance_analysis/k4_init_layout_matched_20261024_2_run01` 的原 `run_matched.py` 可保持不变，以 `--mode profile --profile B0|V0 --out 新目录` 调用。原 `run_ncu.py` 写死旧 `NCU` 并 mkdir(exist_ok=False)，输出已存在，不能原位重跑；仅需新外层 NCU orchestrator 改输出与指标，应用源码、输入、变体和编译路径都不改。

CPU 读取旧14份 replay COMPILE_AUDIT，B0均为 `06ebd78963371a30adac142c06b86950ec300b6cb762a209c5afc67628941814`，V0均为 `6ca13a1d315d7a36ca221e0ec1752dfb1d6c2c4ed2acb6a574db925ed0fce6f4`。各第一replay实际CUBIN重算hash一致，REG168/STACK24。旧脚本新编译后、目标action前还会严格核对原APPROVAL完整hash，不通过即停。

形状 B1/H16/S32768/D128、C4096/P8/L64、grid128×384t，QKV/O FP16、gate/beta/state FP32、初态有值、K4不写final state。V0仅将相同fixed state tensor通过16B alignment/layout标注编译为32个LDG.NA.128，B0为128个LDG.NA；地址布局、数学、后续TMA/MMA与各rotating buffer分配一致。原脚本记录dtype/stride/address binding，重跑会再保存。

每application replay原样构建K1→K2→K3→K4 graph，检查grid/block顺序；每次先4倍查询L2大小的evict fill，再完整CP。20warm轮换4buffers，目标NVTX K4_MATCHED/中回放buffer0完整CP，只有NCU kernel filter选K4。保留gold/四buffer bitwise校验。不同application replay分配地址可不同；每进程B0/V0共享相同captured runtime allocations。

最小候选两次invocation，B0/V0各一次，application replay/cache-control none/clock-control none，沿用旧NVTX/kernel filter/launch-count1。既有SM1845临时请求及独立telemetry/idle guard包裹，finally reset auto。复用当前已查询60指标，重点补LSU Data parent、LGDS read/write、GDS load conflicts、bank reads/writes；保留SourceCounters/InstructionStats、L1请求/sector、L2/DRAM、shared/TC/TMA和spill，用PC确认除fixed scalar/vector外工作是否匹配。

可反证预测仅Data read WF的B0−V0约 `112*(16384−4096)=1376256`，为R3映射外推，不是模型输入或必须闭合的校正。全K4 action差值即便匹配，也不直接分派给INIT时间，不将conflict换cycle，不宣称裸L2速率。

新远端根 `r6_cpu_reuse_audit01/REUSE_AUDIT.json` 包含原脚本hash、旧replay身份、dtype/stride/CP图/输入指纹快照。此次无CUDA import/context，无R6 GPU调用。
