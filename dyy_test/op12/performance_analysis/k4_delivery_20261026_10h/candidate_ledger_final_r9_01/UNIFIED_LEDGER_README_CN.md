# Runner 原始实验与 NCU 账本

`UNIFIED_EXPERIMENT_NCU_LEDGER.json` 是逐调用与逐 host cohort 的总索引；`UNIFIED_NCU_INVOCATIONS.csv` 便于直接查看完整 CLI 命令。`LOCAL_VALID_NCU_REPORTS21.json` 是最终全部可重新导入报告的清单；旧 19 份清单原样保留。

- 有效 NCU 报告 21 份：R1 三份、R3 v2 八份、R4 四份、R5 两份、R6 两份、最后 R9 范围诊断两份，合计 15,972,542 字节。原 19 个路径及 SHA 均与旧独立清单一致。
- 正式 NCU 尝试共 22 次：上述 21 次有效（含两次 R9 计数范围诊断），加 R3 v1 完整采集准备失败一次。另有硬件计数工具诊断失败一次，总 profiler invocation 为 23 次。两次失败均 exit 9，没有生成报告，原始日志保留。
- application replay 的各 pass 和其应用 PID 是同一 profiler invocation 的内部过程，不额外计数。NCU replay 时间不作为正式性能结果。
- Nsight Compute 命令均指向 2026.2.1；额外保存 CLI 2026.2.1.0 build 38283040 的事后只读版本与可执行文件指纹。该事后指纹不冒充每次运行时采集的二进制指纹。

总索引同时保留 25 个 host cohort，涵盖 screen、正式、prepare、图 API 失败和 audit-only。原始 RUN_RECORD/BATCH/STATUS 及命令保持原样。早期 R1/R2 没有记录子进程 PID，账本明确为 `UNKNOWN_NOT_RECORDED`；未记录的精确进程起止时间为 null。遥测 BEFORE/AFTER 或运行期间观测区间只作边界证据，不当作精确进程生命周期。R1/R2 的 samples 表示 ABBA 配对块数，不能直接乘作所有单独调用数。

临时 1845 MHz SM 请求的 finally 恢复证据关联到各批次 guard/clock_lease/restore_automatic.json，并保留可用的独立复查。自然频率计数调用标为无临时租约。请求频率不替代实际频率，也不用于 native cycle 换算。

R4 direct 和 graph 两个正式 cohort 联合门禁均失败，稳定速率分支停止。R5 原七点和独立新进程八点扩展也均未通过联合主门禁，所有条件 FLOP/WF 速率清为 null；通过的单域值只保留描述。两个 R5 cohort 不拼接拟合。旧 derived 与不可变归档保留，当前 `RATE_ANALYSIS.json` 使用 joint-clear 修订，`RATE_ANALYSIS_PRE_JOINT_CLEAR.json` 可对照，统计及原始数据不变。

R7 仅有限对照：20 次采样、2560 个 CTA 端点、1 次 lazy（128 CTA）、4 张严格图证书与每 CTA 全部 32768 元素 gold 通过。C8 相对 C1 的内部周期变化 N128 为 +0.001512%，N256 为 +0.000759%，两点配对变化中位均 0%。按预声明阴性条件停止后续 GPU；不是正式稳定速率。Event 包含共同 16 MiB 输出，输出值不同，不独立解释为 MMA 收益。

R8 首次 prepare 在首个导入缺少 CPU 审计依赖处失败，尚未初始化 CUDA；原始失败记录保留。补齐依赖后 prepare02 完整 CUBIN/text 与静态版本一致、REG56/零 spill、cap1，通过后才获有限 screen 准入。20 次采样、2560 CTA、128 lazy CTA、4 图和全八槽 gold 通过。INTERLEAVED 相对 BLOCKED 的内部周期两个点均多 17 cycle，N128 +0.051325%、N256 +0.025801%，低于预声明 3% 阈值，按阴性条件停止，不追加 NCU 或正式率。两分支输出值相同，Event 仍仅有限描述。

已验收 `candidate_ledger_r7_01` 原样冻结；含 R8 的候选使用新版本目录。新增 CPU 缓存范围核查 `R3_CACHE_SCOPE_CPU_AUDIT.json` 不含任何新增 GPU 调用，也未确定缓存驻留原因。

用户要求提前结束后，只完成已批准 R9 两次，再停止全部 GPU。R9 使用完全相同的 R3 v2 executable/profile/source、S_G128_LOAD N512/20warm/64MiB 槽主序输入；仅缩减为硬件计数，无任何 section、SourceCounters 或 InstructionStats。两次均实际 4 passes，Data read 均为 1073741824 WF，DRAM read 分别为 32778752 和 37426944 字节，明显低于旧完整采集的 382571264 字节。旧完整采集首次为自然频率，新 R9 临时请求 1845，因此这里只支持采集集合/运行条件敏感，不能唯一归因软件插桩、证明 warm 驻留或解释物理 miss；不采纳新稳定率。不同 pass 的 hit/miss/total 不强求相加闭合。

最终只读 `final_readonly_state01/FINAL_READONLY_STATE.json`：HEAD 固定 e77a4a0d276367895c3b50a642fd8f326c03fb72，工作树 clean，7 份 blackwell Python 源文件 SHA 均等于 HEAD blob。R9 的 finally `-rgc` 成功，末三次读取均无 compute app、util 0%、SM120/HBM3996 MHz、GPU idle reason active；当前频率只是读回状态。检查不 import CUDA、不启动 kernel、不改时钟。

`UNIFIED_LEDGER_SUPPORT01.tgz` 补齐 guard、prepare 遥测与失败 CPU 工程证据；R7 完整实际 CUBIN/PTX/SASS、端点与图元数据位于 `R7_FINITE_SCREEN01_EVIDENCE.tgz` 及展开目录。所有有效 `.ncu-rep` 必须与原始 Source/Raw 一并进入最终包。
