# R3 v2 CPU 静态门禁（GPU 尚未准入）

唯一算法修订将 CTA-major 改为 slot-major：`((iteration & 7) * gridDim.x + CTA) * 65536 + thread * 512 + column * 4`。相邻 CTA 在同轮跨度 64 KiB；grid1 每 CTA 512 KiB、grid128 总 64 MiB 与 v1 一致。host checksum 分别构造 grid1/grid128 映射，源分配仍为同一 64 MiB。未新增设备计时点或 globaltimer；保留原 clock64、Event、NA load、依赖证书、前后 CTA barrier。

工具链单列为 nvcc/ptxas 13.0.88，应用使用 `-gencode arch=compute_100a,code=sm_100a`。实际可执行文件经 cuobjdump 抽取 CUBIN，`--list-ptx` 无 PTX；另存独立 PTX 审计文件。CPU 门禁审计的是实际可执行文件 SASS，不只审计中间 PTX。该工具链/地址修订不宣称与 v1 完整指令字节相同。

资源为 scalar REG139、vector REG140，二者 STACK/LOCAL/LDL/STL 均零。原生 scalar 128 个 `LDG.E.NA`、vector 32 个 `LDG.E.NA.128`；128 个独立载入值均进入循环后依赖证书。循环内无其他数据访存、同步或 MMA。两 clock/barrier 的前后次序和全部输出在 endclock 后的条件通过。

应用只有 native CUBIN，因此 v2 wrapper 显式取消进程级 `CUDA_DISABLE_PTX_JIT`（v1 为 1）并记录环境差异，避免同时禁止工具内部 JIT。此变更是工具适用性候选，尚无 GPU 证据表明它能修复 v1 profiling 失败。v1 两次失败原样保留，无进一步 v1 GPU 诊断。

Host-only 增加源分配地址/映射 JSON、Event 前后 host 时间戳，均在原设备计时区间外。它们用于与独立频率/节流 sidecar 对齐，不进行不同 SM 时钟相减。正式适用性仍待验收：自然频率下测得的是 SM-cycle 条件服务，L2 时钟域与实际 SM 频率会限制迁移到原版 1845 条件，不能直接宣称原版精确速率。

实际 exe SHA256：`706756dda3c7bd8951361930b1bac35524dbddc2a582aac528851452e04f34f7`。

实际 CUBIN SHA256：`63a3337682d01f6c56f1de294eb72e9b0682edf8cc42ac2d6ac367ed03856d2d`。

下一步只能在 Lead 独立审查与明确 GPU 准入后执行。当前无 v2 GPU launch、无 rate/model 结果。
