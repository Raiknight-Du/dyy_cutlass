# Runner 原始实验与 NCU 账本

`UNIFIED_EXPERIMENT_NCU_LEDGER.json` 是逐调用与逐 host cohort 的总索引；`UNIFIED_NCU_INVOCATIONS.csv` 便于直接查看完整 CLI 命令。`LOCAL_VALID_NCU_REPORTS19.json` 是全部可重新导入报告的独立清单。

- 有效正式 NCU 报告 19 份：R1 三份、R3 v2 八份、R4 四份、R5 两份、R6 两份，合计 15,725,570 字节。19 个路径及 SHA 均与独立清单一致。
- 正式 NCU 尝试共 20 次：上述 19 次有效，加 R3 v1 完整采集准备失败一次。另有硬件计数工具诊断失败一次，总 profiler invocation 为 21 次。两次失败均 exit 9，没有生成报告，原始日志保留。
- application replay 的各 pass 和其应用 PID 是同一 profiler invocation 的内部过程，不额外计数。NCU replay 时间不作为正式性能结果。
- Nsight Compute 命令均指向 2026.2.1；额外保存 CLI 2026.2.1.0 build 38283040 的事后只读版本与可执行文件指纹。该事后指纹不冒充每次运行时采集的二进制指纹。

总索引同时保留 22 个 host cohort，涵盖 screen、正式、prepare、图 API 失败和 audit-only。原始 RUN_RECORD/BATCH/STATUS 及命令保持原样。早期 R1/R2 没有记录子进程 PID，账本明确为 `UNKNOWN_NOT_RECORDED`；未记录的精确进程起止时间为 null。遥测 BEFORE/AFTER 或运行期间观测区间只作边界证据，不当作精确进程生命周期。R1/R2 的 samples 表示 ABBA 配对块数，不能直接乘作所有单独调用数。

临时 1845 MHz SM 请求的 finally 恢复证据关联到各批次 guard/clock_lease/restore_automatic.json，并保留可用的独立复查。自然频率计数调用标为无临时租约。请求频率不替代实际频率，也不用于 native cycle 换算。

R4 direct 和 graph 两个正式 cohort 联合门禁均失败，稳定速率分支停止。R5 原七点和独立新进程八点扩展也均未通过联合主门禁，所有条件 FLOP/WF 速率清为 null；通过的单域值只保留描述。两个 R5 cohort 不拼接拟合。旧 derived 与不可变归档保留，当前 `RATE_ANALYSIS.json` 使用 joint-clear 修订，`RATE_ANALYSIS_PRE_JOINT_CLEAR.json` 可对照，统计及原始数据不变。

R7 仅有限对照：20 次采样、2560 个 CTA 端点、1 次 lazy（128 CTA）、4 张严格图证书与每 CTA 全部 32768 元素 gold 通过。C8 相对 C1 的内部周期变化 N128 为 +0.001512%，N256 为 +0.000759%，两点配对变化中位均 0%。按预声明阴性条件停止后续 GPU；不是正式稳定速率。Event 包含共同 16 MiB 输出，输出值不同，不独立解释为 MMA 收益。

`UNIFIED_LEDGER_SUPPORT01.tgz` 补齐 guard、prepare 遥测与失败 CPU 工程证据；R7 完整实际 CUBIN/PTX/SASS、端点与图元数据位于 `R7_FINITE_SCREEN01_EVIDENCE.tgz` 及展开目录。所有有效 `.ncu-rep` 必须与原始 Source/Raw 一并进入最终包。
