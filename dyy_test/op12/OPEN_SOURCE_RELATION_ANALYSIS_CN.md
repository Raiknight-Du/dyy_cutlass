# op12 与 FlashInfer GDN 开源实现关系分析

## 1. 分析对象与判定口径

### 1.1 op12 目标

本次按以下口径拆解 op12：

1. 场景是 LLM prefill，而非单 token decode。
2. 算法是 Gated DeltaNet / gated delta rule。
3. 目标平台是 NVIDIA B200（SM100）。
4. “FP8 类型”约束主计算输入与核心 MMA；仅用 FP8 保存 recurrent state 不算完整达成。
5. BNSD 指四维 `[batch, heads, sequence, dimension]` 输入布局。
6. TND 指 packed-varlen `[total_tokens, heads, dimension]`，由累计序列长度描述各序列边界。
7. “TND 格式 FA 矩阵乘”按 GDN 内部 attention-like 矩阵乘理解，不要求标准 `QK^T -> softmax -> PV` FlashAttention。

### 1.2 开源基线

- 仓库：`https://github.com/flashinfer-ai/flashinfer.git`
- 分支来源：`main`
- 固定提交：`e77a4a0d276367895c3b50a642fd8f326c03fb72`
- 目标源码目录：`flashinfer/gdn_kernels`
- 公共入口：`flashinfer/gdn_prefill.py::chunk_gated_delta_rule`
- 本报告引用的行号均对应上述固定提交。

## 2. 总体结论

FlashInfer 已实现 **B200 上的 GDN prefill + TND packed-varlen + GDN 内部矩阵乘**，但主计算路径是 FP16/BF16，不是 FP8；公共接口也不接受 BNSD 四维 Q/K/V。

因此它与 op12 的关系是：

- **可直接复用的部分**：GDN 数学分解、chunked prefill、TND/`cu_seqlens`、GQA/GVA、SM100 TMA/TMEM/tcgen05 调度、CP 与非 CP 流水。
- **只能作为旁证的部分**：E4M3/E5M2 recurrent state 的加载和写回。
- **尚未实现的部分**：FP8 Q/K/V/O、FP8 tcgen05 MMA、相应 scaling/量化元数据，以及 BNSD Q/K/V 计算入口。

## 3. 要求—实现对照矩阵

| 验收维度 | 判定 | FlashInfer 证据 | 与 op12 的关系 |
|---|---|---|---|
| LLM prefill | 达成 | `flashinfer/gdn_prefill.py:118-142` 将入口定义为 chunked GDN attention for prefill。 | 场景一致。 |
| Gated DeltaNet 语义 | 达成 | `gated_delta_net_chunked.py:29-55` 给出 gate 累积、三角修正、state 更新与输出组合。 | 算法主体一致。 |
| B200 / SM100 | 达成 | `gdn_prefill.py:460-518` 分派到 SM100 kernel；B200 实机正向测试通过。 | 平台一致。 |
| TND packed-varlen | 达成 | `gdn_prefill.py:146-184` 声明三维 Q/K/V 及 `cu_seqlens`；支持变长序列。 | 布局和 prefill 组织方式一致。 |
| GQA / GVA | 达成 | `gdn_prefill.py:140-142,253-256` 明确支持；B200 各测一组。 | 可覆盖不等头数配置。 |
| GDN 内部 7 组 GEMM | 达成 | `gated_delta_net_chunked.py:42-55,572-616`。 | attention-like 矩阵乘结构可复用。 |
| FP8 recurrent state | 部分达成 | `gdn_prefill.py:36-42,165-174` 和 `blackwell/gdn_prefill.py:160-165`。 | 只解决 state 存储；不是 FP8 计算。 |
| FP8 Q/K/V/O | 不达成 | `blackwell/gdn_prefill.py:66-74` 只映射 BF16/FP16；B200 FP8 Q/K/V 负向测试得到相同拒绝。 | 不满足 FP8 主计算输入。 |
| FP8 Tensor Core MMA | 不达成 | `gated_delta_net_chunked.py:559-570` 使用 `tcgen05.MmaF16BF16Op`；`can_implement` 在 380-387 行限制 BF16/FP16 输入、FP32 累加。 | 核心计算精度不满足。 |
| BNSD Q/K/V 入口 | 不达成 | 公共接口仅声明 `[total_seq_len, heads, head_size]`；四维负向测试被拒绝。 | 需要新增布局入口或转换层/专用 kernel。 |
| head size 泛化 | 不达成 | `gdn_prefill.py:468-471` 将 SM100 固定为 128。 | op12 若包含其他 D，需要新增实例。 |

## 4. 开源实现做了什么

### 4.1 公共接口和外部布局

公共入口 `chunk_gated_delta_rule()` 接收：

- `q: [total_seq_len, num_q_heads, head_size]`
- `k: [total_seq_len, num_k_heads, head_size]`
- `v: [total_seq_len, num_v_heads, head_size]`
- `g/beta: [total_seq_len, num_sab_heads]`，FP32
- `cu_seqlens: [num_seqs + 1]`
- 可选 initial/final state、checkpoint 和 state pool index

这正是 packed TND varlen 形态。序列没有在 Q/K/V 上保留独立 batch 维，而由 `cu_seqlens` 将 token 流切分成多条序列。

### 4.2 GDN 数学分解

非 CP SM100 kernel 每个 64-token chunk 实现以下计算：

1. `W_kk = K @ K^T`
2. `W_qk = Q @ K^T`
3. `KS = K @ S_prev`
4. `QS = Q @ S_prev`
5. `NV = A_inv @ V`
6. `O_intra = W_qkv @ NV`
7. `dS = K^T @ delta`

随后组合 intra/inter-chunk 输出并更新 recurrent state。对应源码为 `flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py:29-55`。

这说明“TND 下的 GDN 内部矩阵乘”在算法层面已经存在；但这些 GEMM 都由 FP16/BF16 MMA 实现，不能据此判为 FP8 要求已达成。

### 4.3 SM100 调度

非 CP 路径使用 CuTe DSL、TMA、TMEM 与 `tcgen05`：

- chunk size 固定为 64（`gated_delta_net_chunked.py:413-435`）。
- head size 固定为 128（`gdn_prefill.py:468-471`）。
- accumulator 是 FP32，输入与 inverse 是 FP16/BF16（`gated_delta_net_chunked.py:369-407`）。
- 内部为 persistent kernel，按 compute、MMA、TMA load、epilogue 等角色分工。

CP 路径是多 kernel pipeline，包含 T precompute、M/N precompute、跨 chunk state fixup 和最终 prefill；它适合 batch×head 并行度不足而序列较长的情况。

## 5. 关键不一致

### 5.1 FP8 state 不等于 FP8 矩阵乘

SM100 非 CP 路径允许 state 使用：

- FP32
- BF16
- FP16
- FP8 E4M3FN
- FP8 E5M2

但 Q/K/V/O 的 `_cutlass_io_dtype()` 只接受 BF16 或 FP16，核心 MMA 构造也是 `MmaF16BF16Op`。FP8 state 在加载后参与的是转换后的计算，输出 state 再量化存储；源码没有把 FP8 Q/K/V 直接送入 FP8 Tensor Core MMA。

因此，“支持 FP8 state”只能标为部分相关能力，不能用来宣称 op12 的 FP8 主计算已完成。

### 5.2 没有 BNSD 计算入口

公共接口的 Q/K/V 都是三维 TND。state 虽为 `[N,H,V,K]`，其中：

- `N` 是序列编号，或 state pool slot 数；
- `H` 是 state head；
- `V,K` 是 recurrent matrix 的两个特征维；
- 它不是 `[batch,heads,sequence,dimension]` 的 Q/K/V。

所以不能因为 state 是四维张量，就将开源实现误判为支持 BNSD。B200 四维 Q/K/V 负向测试也没有可用路径。

### 5.3 固定能力边界

| 项目 | 非 CP SM100 | CP SM100 |
|---|---|---|
| Q/K/V/O dtype | FP16、BF16 | FP16、BF16 |
| state dtype | FP32、BF16、FP16、FP8 E4M3/E5M2 | FP32、BF16、FP16；公共 CP 筛选不接受 FP8 state |
| head size | 128 | 128 |
| checkpoint | 支持 64 的整数倍 | 暂不支持 |
| state indices | 非 CP SM100 支持 | 公共分派会回退或拒绝 |
| CUDA | 13+ | 13+ |
| CuTe DSL | 本固定提交声明 4.7.0 | 本固定提交声明 4.7.0 |

## 6. B200 实机验证

### 6.1 环境

- 主机：`metallic-matrix-fin-03`
- GPU：NVIDIA B200，compute capability 10.0
- Python：3.10.12
- PyTorch：2.11.0+cu130
- CUDA runtime：13.0；系统 nvcc：13.3
- 隔离 CuTe DSL：4.7.0
- 隔离 apache-tvm-ffi：0.1.13.post3

隔离 venv 使用 `--system-site-packages` 复用现有 PyTorch，但在 venv 内覆盖 CuTe DSL 与 TVM-FFI；系统 FlashInfer 0.6.12、CuTe DSL 4.5.2 未被修改。首轮仅安装 CuTe DSL 4.7.0 时继承了系统 TVM-FFI 0.1.9，导致 kwargs wrapper 接口不匹配；升级隔离环境中的 TVM-FFI 后重跑通过。首轮日志被保留，用于证明该失败属于依赖闭合问题。

### 6.2 结果

| 用例 | 结果 | 主要观测 |
|---|---|---|
| FP16 TND，GQA，非 CP，对 reference | PASS | output max abs `3.052e-05`；state max abs `7.021e-05` |
| BF16 TND，GVA，`[64,128]` 变长，非 CP，对 reference | PASS | output max abs `2.441e-04`；state max abs `6.132e-04` |
| BF16 TND，2048 token，CP 对非 CP | PASS | output max abs `2.441e-04`；state max abs `7.583e-04` |
| BF16 I/O + FP8 E4M3 state | PASS | output max abs `1.221e-03`；state max abs `7.812e-03` |
| BF16 I/O + FP8 E5M2 state | PASS | output max abs `1.953e-03`；state max abs `1.562e-02` |
| FP8 E4M3 Q/K/V | PASS（预期拒绝） | `ValueError: ... expected bfloat16 or float16` |
| BNSD 四维 Q/K/V | PASS（预期拒绝） | 没有四维布局入口，SM100 shape 检查拒绝 |

最终为 7/7 通过。相对误差在 reference 接近零的位置会放大，因此表中使用 max absolute error；完整 JSONL 位于 `validation/smoke_raw.log`。

## 7. 对 op12 后续开发的含义

若以该实现为基础开发 op12，可复用：

- GDN 的 chunked 数学分解与三角修正逻辑；
- TND + `cu_seqlens` 的变长接口；
- SM100 persistent、TMA/TMEM、warp-role 和 CP 调度框架；
- reference、GQA/GVA、state 与 checkpoint 测试结构。

仍需新增：

1. FP8 Q/K/V/O 类型契约和量化/反量化语义。
2. FP8 scaling metadata、格式选择和数值范围策略。
3. 以 SM100 FP8 Tensor Core 指令替换/扩展 `MmaF16BF16Op` 的核心 GEMM。
4. BNSD 公共入口及其 batch/head/sequence stride 语义，或明确定义 BNSD 到 TND 的转换边界。
5. FP8+BNSD、FP8+TND、GQA/GVA、tail chunk、CP/非 CP 的正确性与性能验证。

本轮没有实施上述适配，也没有执行性能 benchmark 或 NCU 分析。

