# op12：B200 FP8 Gated DeltaNet Prefill 设计与实现说明

## 1. 冻结接口

`gdn_prefill_fp8()` 是 op12 自己的入口，不修改 FlashInfer API。D 固定 128、chunk 固定 64；Q/K/V 只能是普通 `float8_e4m3fn`，外部 scale 恒为 1；gate/beta 为 FP32；O 和 `[B,H,V,K]` state 为 FP16。

两种输入布局均为一等入口：

- BNSD：Q `[B,Hq,S,128]`、K `[B,Hk,S,128]`、V `[B,Hv,S,128]`、gate/beta `[B,Ho,S]`。
- TND：Q `[T,Hq,128]`、K `[T,Hk,128]`、V `[T,Hv,128]`、gate/beta `[T,Ho]`，并要求 CUDA int32 `cu_seqlens[B+1]`。

`Ho=max(Hq,Hv)`。只接受 MHA、GQA `Hq>Hk=Hv` 且整除，或 GVA `Hv>Hq=Hk` 且整除。D stride 必须为 1，基地址至少 16 字节对齐。

## 2. 每个 chunk 的 GDN 流程

对每个 sequence/head，公共 state 在接口上是 `[V,K]`，计算核心用其转置视图 `[K,V]`：

1. FP32 计算 `log(gate)`、chunk 内累计门控 `cumprod` 与下三角 transfer `T[i,j]`。
2. GEMM1 `Wkk=K·Kᵀ`，构造单位下三角 `A=I+tril_strict(beta_row*T*Wkk)`，FP32 求逆，并在列方向合入 beta。
3. GEMM2 `Wqk=Q·Kᵀ`，构造因果 `Wqkv=T*Wqk*attention_scale`。
4. 在线量化 FP16/FP32 state；GEMM3 `KS=K·Sprev`、GEMM4 `QS=Q·Sprev`。
5. FP32 residual `V-cumprod*KS`；在线量化后，GEMM5 `NV=(A⁻¹*beta_col)·residual`。
6. 在线量化 Wqkv/NV；GEMM6 `Ointra=Wqkv·NV`；FP32 合成 `O=Ointra+cumprod*QS*scale`。
7. 构造至 chunk 末端的 decay residual；GEMM7 `dS=Kᵀ·delta`；FP32 更新 `Snext=cumprod_end*Sprev+dS`。
8. 只存有效 token 的 FP16 O；chunk 递推完成后存 FP16 final state。

该式与固定 FlashInfer 源码的七 GEMM 数据依赖一致。尾块按最小 16 行硬件粒度补齐，但无效行 `Q/K/V/beta=0, gate=1`，输出不写回，`Kᵀ·delta` 的无效行严格为零，因此不会把 padding 当作 token 更新 state。

## 3. 精度落点与影响

外部 E4M3 值不重标定；这是接口语义，不把调用方的原值误解为 MXFP8。内部 tile 使用：

`s = max(abs(x))/448`，全零 tile 取 `s=1`；先除 s、夹紧到 `[-448,448]`，再以 CUDA E4M3 round-to-nearest 转换。GEMM epilogue 乘回 `sA*sB`。

gate、beta、累计门控、transfer、三角逆、GEMM accumulator、O/state 最终组合都保持 FP32。持久化边界才转 FP16。这样避免门控指数与递推控制量降至 FP8，但仍存在三类不可恢复误差：外部原始 E4M3 量化、中间 tile 量化、每 chunk state 重新量化。FP32 accumulator 只能控制点积累误差，不能恢复输入量化损失。

当前 backend 为每个 DV=64 半片独立量化 state/NV/delta，这与最终两个 CTA 的所有权边界一致，也避免一个半片的离群值压低另一半片的有效精度。验证输出逐类记录饱和率。

## 4. B200 资源规划

目标融合 CTA 固定 384 threads（12 warps）。寄存器预算为 CG0 224/thread、CG1 240/thread、其余四 warp 32/thread，总计 63,488 registers/CTA。静态 SMEM 明细合计 150,016 B，在 192 KiB 设计门限内保留 46,592 B；TMEM 五段合计 448 columns，达到规划上限。

warp 角色：0–3 负责门控、KK/QK epilogue 和逆；4–7 负责 state、KS/QS、NV/delta、O/state epilogue；8/10 是两组 MMA issuer；9 是 Q/K/V TMA producer；11 负责 O store 与 gate/beta prefetch。

B200 共 148 SM，2-SM cluster 的并发调度基准为 74。最终 cluster 设计中两个 CTA 各持有 DV 的连续 64 列，leader 发布与 DV 无关的 A-inverse/Wqkv FP8 tile 和 FP32 scale 到 DSM；Q/K/gate/beta 用 TMA multicast。每个 CTA 内仍使用 `CtaGroup.ONE` 的 FP8 MMA，规避 GDN M=64 与 `CtaGroup.TWO` 合法 M shape 不匹配的问题。

实测约束沿用 op6 结论：`tcgen05.alloc` 使每 SM 仅驻留 1 CTA；B200 不接受所需的 `tcgen05.ld.red.max`，maxabs 需 TMEM→register、warp shuffle 与 warp-group 共享归约。静态资源脚本只是融合 kernel 的门禁模型，不等于当前 unfused backend 的编译资源报告。

## 5. 当前实现与最终 kernel 的分界

当前 `backend_scaled_mm.py` 已完成算法、FP8 落点、布局/head/tail/state 语义和 B200 硬件正确性验证。每个逻辑 GEMM都进入 E4M3 Tensor-Core kernel；NCU 观察到 SM100 `nvjet` kernel、tensor-pipe 指令和 SASS `UTCQMMA`。

尚未完成的性能工程是把这些 kernel 与 FP32 epilogue 融成单 CTA CuTe kernel，并进一步接入物理 2-CTA cluster。当前 PyTorch backend 会为中间量分配 GMEM tensor，所以不能声称满足“中间量只在片上”或“无中间 GMEM”。

已建立 `op12_gdn/cute_fused` 实验移植：七组 atom 已换成 E4M3 `MmaF8F6F4Op`，并解决了 FP8 TMEM packed layout、16-bit LDSM 和初始 swizzled-SMEM copy verifier 问题；当前在 SM100a NVVM backend 失败，未生成 cubin。该实验路径尚无在线 scale、仍临时使用 FP8 O，绝不接入公共 API。详细状态和复现日志见 `validation/FUSED_PORT_STATUS_CN.md`。

## 6. CP 路径

接口按 `B*Ho<74 && max_seq>=4096` 选择 CP semantic route，并保持 4096-token 宏块边界。当前 route 仍顺序递推，只用于冻结路由和正确性；正式 CP 需要四阶段多 kernel：T 预计算、局部分区 state、FP32 跨分区前缀修正、最终 O fixup。workspace 查询、显存不足回退和并行 prefix 尚未实现，因此实现状态中明确标为未完成。

## 7. 验收口径

FP32 recurrent gold 先把外部 E4M3 精确提升为 FP32。门限：O 相对 L2≤0.20、cosine≥0.98；final state 相对 L2≤0.25；不得出现 NaN/Inf。覆盖 BNSD `S={1,63,64,65,127,257}`、TND varlen 同组长度、GQA 8:2、GVA 2:8、initial state、全零、尾块及 CP semantic route。

最终融合 kernel 还必须增加：编译 PTX 中七组 `tcgen05.mma.kind::f8f6f4` 证据、SMEM/TMEM/register 实测、零 spill、BNSD/TND compute-sanitizer、DSM/multicast 和 CP/非 CP 一致性。静态 atom 构造检查不能替代这些最终门禁。
