# op12：FlashInfer GDN 仅核心 MMA 改 FP8 的 B200 方案

## 1. 唯一改动

本方案以 FlashInfer 固定提交 `e77a4a0d276367895c3b50a642fd8f326c03fb72` 的 SM100 non-CP chunked GDN 为基线。唯一算法实现变量是：七组核心 Tensor Core GEMM 的两个操作数从原版 FP16/BF16 临时转换为普通 E4M3，并以 FP32 accumulator 执行 FP8 MMA。

- 不使用 amax、动态 scale、tile scale 或反标定；MMA ABI 所需的两个 scale 恒为 1。
- E4M3 临时量只存在于 MMA 操作数边界，不改变公共张量或非 MMA 中间量的 dtype。
- 原版算法 `scale=1/sqrt(128)` 保留；它是模型计算系数，不是量化 scale。
- 本轮只定义七组主 GEMM。FlashInfer 分层三角逆内部的 warp MMA、gate/beta、残差、epilogue、state 更新均保持原版路径，不纳入 FP8 替换。

因此，本轮是单变量精度实验，不再采用此前“外部 Q/K/V 固定 E4M3、三角逆统一 FP32、O/state 固定 FP16、内部 tile scaling、BNSD 新入口、2-SM cluster”的多变量方案。

## 2. 保持不变的 FlashInfer 契约

| 项目 | 保持的原版行为 |
|---|---|
| 公共布局 | packed TND：Q/K/V `[total_tokens,H,128]` + `cu_seqlens` |
| Q/K/V/O | FP16 或 BF16，且四者 dtype 一致 |
| gate、beta | FP32 |
| state | 公共布局 `[N,H,V,K]`；SM100 支持 FP32/BF16/FP16/E4M3/E5M2 持久化 |
| head | MHA、GQA、GVA，`Ho=max(Hq,Hv)` |
| head/chunk size | 128 / 64 |
| gate 前缀 | FP32 log/cumulative/transfer 语义 |
| 三角系统 | 保持原版构造、分层求逆与发布 dtype |
| residual/epilogue | 保持原版 FP16/BF16 与 FP32 accumulator 的转换边界 |
| 尾块 | Q/K/V=0、gate=1、beta=0；只写有效 token |

BNSD 不是 FlashInfer 原版公共入口，不能放进“仅改 MMA”的单变量比较。若 op12 最终仍要求 BNSD，它应作为独立的布局适配阶段单独验收，不能与本轮精度结论合并。

## 3. 64-token chunk 数学流程

令 `a_i=prod(alpha_0..alpha_i)`，`G_ij=a_i/a_j`（`i>=j`，否则为 0）。每个 chunk 保持原版公式：

1. `Wkk = K K^T`。
2. `A = I + strict_lower(diag(beta) * G * Wkk)`，构造 `P=A^-1 diag(beta)`。
3. `Wqk = Q K^T`，构造 `Wqkv = causal(G * Wqk * scale)`。
4. `KS=K S0`，`QS=Q S0`。
5. `R=V-diag(a)KS`，`NV=P R`。
6. `Ointra=Wqkv NV`，`O=Ointra+diag(a)QS*scale`。
7. `Delta_i=(a_end/a_i)NV_i`，`dS=K^T Delta`，`Snext=a_end*S0+dS`。

其中 `*` 在门控矩阵处表示逐元素或行/列缩放。七个 GEMM 的数据依赖不变，只有进入这些 GEMM 时的操作数 dtype 发生变化。

## 4. 七组 MMA 精度契约

| # | GEMM | 原版 MMA 输入 | op12 临时 MMA 输入 | 累加 |
|---:|---|---|---|---|
| 1 | `K @ K.T` | FP16/BF16 | E4M3×E4M3 | FP32 |
| 2 | `Q @ K.T` | FP16/BF16 | E4M3×E4M3 | FP32 |
| 3 | `K @ state` | FP16/BF16 | E4M3×E4M3 | FP32 |
| 4 | `Q @ state` | FP16/BF16 | E4M3×E4M3 | FP32 |
| 5 | `P @ residual` | FP16/BF16 | E4M3×E4M3 | FP32 |
| 6 | `Wqkv @ NV` | FP16/BF16 | E4M3×E4M3 | FP32 |
| 7 | `K.T @ Delta` | FP16/BF16 | E4M3×E4M3 | FP32 |

操作数转换为直接饱和 E4M3：`clamp(x,-448,448) -> E4M3`，不做数值重标定。MMA 完成后，FP32 accumulator 按原版位置转换回 FP16/BF16 或进入原版 FP32 epilogue。

## 5. 当前实现边界

`op12_gdn/backend_mma_only.py` 是可执行的 unfused correctness backend，公共输入保持 TND FP16/BF16，只在 `_mma_fp8()` 内创建临时 E4M3 操作数。它用于隔离数值变量和采集 B200 硬件证据，中间 tensor 仍由 PyTorch 管理并可能落 GMEM。

现有 `op12_gdn/cute_fused` 是旧多变量移植：它还改变了主 `io_dtype`、inverse scratch 和输出 dtype，因此不符合新的“仅改 MMA”口径，保留作历史研究但不作为 op12 当前实现。最终融合版本必须从 clean FlashInfer 固定提交重新移植，只修改七组主 MMA 及其必要的临时 E4M3 operand staging。

## 6. B200 影响

- SM100 FP8 `MmaF8F6F4Op` 的 MMA-K 为 32，D=128 与 chunk=64 可整除。
- 公共 Q/K/V 仍是 FP16/BF16，所以不能宣称 GMEM 输入带宽减半。
- 不需要 maxabs 归约、tile-scale SMEM、反标定或 `tcgen05.ld.red.max`；此前为在线 scaling 规划的资源全部删除。
- 新增成本只有 MMA 前的 FP16/BF16→E4M3 转换、临时 operand staging 和相应同步。
- 原版 kernel 是 `cluster_shape=(1,1,1)`、`CtaGroup.ONE`、384 threads、1 CTA/SM；本轮不增加 2-SM cluster、DSM 或 multicast。

原先 2-SM/448-TMEM-column/192-KiB-SMEM 资源表针对另一个多变量设计，不能继续作为本方案的资源承诺。最终 fused port 产出 cubin 后，需重新用编译结果报告 register、SMEM、TMEM 和 spill。

## 7. B200 精度结论

验证数据保持 FlashInfer `gen_qkv` 和 `blockwise_delta_rule`，同时运行原版 SM100 kernel 对照。覆盖 FP16/BF16、`S={1,63,64,65,127,257}`、varlen、GQA 8:2、GVA 2:8、initial state、低/高幅输入和 non-CP 4096-token。

- FlashInfer 原版严格 allclose：15/15。
- MMA-only FP8 的 op12 放宽门槛：13/15。
- MMA-only FP8 的严格 allclose：3/15。
- 失败项：FP16 `S=1` 的 O relative-L2=0.229802；低幅 Q/V×0.25 的 O relative-L2=0.200916、cosine=0.979779。
- 所有 case 有限、最大 FP8 饱和率为 0；失败主要来自无缩放 E4M3 对小操作数的归零和舍入。

因此当前结论仍是：**仅把七组 MMA 改成无缩放 FP8，其他保持 FlashInfer 原版，也没有通过完整精度矩阵。** 详细数据见 `validation/MMA_ONLY_FP8_RESULTS.md`。

## 8. 验收边界

当前完成的是数值和硬件可执行性验证。最终融合实现还必须满足：

- 由 clean upstream 生成的 diff 证明除七组主 MMA 和必要 operand conversion 外无算法、API、layout、state 或 epilogue 变化；
- PTX/SASS 证明七组主 MMA 使用 FP8，三角逆等非目标路径保持原版；
- 原版与改版使用完全相同输入逐 case 比较；
- compute-sanitizer、无 spill、资源实测和性能测试通过。
