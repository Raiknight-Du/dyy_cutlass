# R3 v1 profiling 工具失败保留

两次尝试均使用 exe SHA256 `bdeabbb53a7a4313d734aa8db7db4f04f567fae39a1fc6880bdf6caf59a30531`、NCU 2026.2.1、application replay、cache-control none、clock-control none、`CUDA_DISABLE_PTX_JIT=1`。有自然空闲三次守卫，无锁频。

1. `r3_ncu_S_G128_LOAD_01`：正式目标 S_G128_LOAD/N512，含 SourceCounters/InstructionStats 及硬件指标；准备阶段失败。
2. `r3_v1_hardware_diagnostic01`：经 Lead 单独批准的唯一硬件诊断，去 SourceCounters/InstructionStats，仅 LaunchStats 与 DRAM read/write、L2 read/write 四指标；仍准备阶段失败。

两次共同日志：`Failed to prepare kernel for profiling`，`Unknown error on device 0`，目标返回 9。无有效报告或计数，不能作为动态工作量、缓存条件或服务速率证据。第二次失败说明问题不限于请求 SourceCounters 的那组 section，但不能确定是哪个工具/驱动/环境环节。

两次 raw command/stdout/stderr/telemetry 与 admission 原样镜像同名目录。v1 GPU 分支已停止。已安装 nvcc/ptxas 有 13.0.88、13.1.115、13.3.73；原可 profiling CuTe PTX 标识 13.3.27，因此不能概括为所有 13.3 产物不支持。
