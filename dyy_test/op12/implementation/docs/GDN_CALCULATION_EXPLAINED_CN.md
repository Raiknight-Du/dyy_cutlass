# Gated DeltaNet（GDN）入门与详细计算流程

## 1. 一句话理解 GDN

Gated DeltaNet（GDN）是一种带“可控遗忘”和“误差修正写入”的线性注意力/递归记忆机制。它不保存所有历史 token 的 K/V，而是把历史压缩进每个 head 的固定大小矩阵状态 `S`；每到一个新 token，先按 gate 衰减旧记忆，再读取当前 key 对应的旧 value，只把“目标 value 与旧 value 的差”写回状态。

可以把 `S` 想成一张可在线更新的键值关联表：

- `k_t`：指出要访问或修改哪一类记忆；
- `v_t`：希望这个 key 方向最终对应什么内容；
- `q_t`：从更新后的记忆里读取什么；
- `alpha_t`：旧记忆总体保留多少；
- `beta_t`：本 token 的修正写入强度。

GDN 的论文动机是把两种互补机制结合起来：gate 负责快速遗忘，delta rule 负责有针对性地覆盖/修正键值关联。[Gated Delta Networks，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/4904fad153f6434a7bcf04465d4be2cc-Paper-Conference.pdf)

## 2. 它与普通 Attention 的区别

标准 softmax attention 在时刻 `t` 直接拿 `q_t` 与所有历史 key 比较，再对所有历史 value 加权求和。它保留逐 token 的 KV cache，能精细访问历史 token，但 cache 随序列长度增长。

GDN 把历史压缩成固定矩阵：

`S_t in R^(D_v x D_k)`

因此每个 head 的状态大小与序列长度无关。代价是：历史不再逐 token 精确保留，多个 key/value 会在有限状态中产生干扰。GDN 的 delta rule 正是用于降低这种“直接叠加写入”带来的干扰。

| 对比项 | Softmax Attention | GDN |
|---|---|---|
| 历史表示 | 所有历史 K/V | 固定矩阵状态 `S` |
| 解码状态容量 | 随序列增长 | 与序列长度无关 |
| 新信息写入 | 不更新旧 KV | 每 token 更新 `S` |
| 遗忘 | 通常没有显式全局 gate | `alpha_t` 主动衰减旧状态 |
| 检索 | 对历史 token 做显式权重 | 用 `q_t` 乘压缩状态 |

## 3. 逐 token 的精确定义

以下先采用对人最直观、也是 op12 公共接口采用的状态方向：

- `S_(t-1)`：`[D_v, D_k]`
- `k_t, q_t`：`[D_k]`
- `v_t`：`[D_v]`
- `alpha_t, beta_t`：标量

### 步骤 1：遗忘旧记忆

`Sbar_(t-1) = alpha_t * S_(t-1)`

`alpha_t` 是 forget gate。常见取值在 `(0, 1]`：

- `alpha_t = 1`：保留旧状态；
- `alpha_t` 接近 0：快速清空旧记忆；
- 连续多个 token 的历史贡献会乘上多个 `alpha`，所以 gate 的累积乘积非常重要。

注意：不同模型可能先生成 gate logit，再通过指数或 sigmoid 得到 `alpha`。FlashInfer/op12 算子入口接收的是已经可以作为乘法衰减因子的 `alpha`，不是未变换的 logit。

### 步骤 2：用当前 key 读取旧值

`vhat_t = Sbar_(t-1) * k_t`

`vhat_t` 是状态认为“当前 key 已经对应的 value”。

### 步骤 3：计算写入误差

`e_t = v_t - vhat_t`

这一步是 delta rule 的核心。如果旧状态已经能为 `k_t` 给出正确的 `v_t`，误差接近 0，就不必重复写入。如果预测错误，只写入差值。

### 步骤 4：沿 key 方向做秩一更新

`S_t = Sbar_(t-1) + beta_t * e_t * k_t^T`

外积 `e_t * k_t^T` 的形状是 `[D_v, D_k]`。`beta_t` 是 update gate：

- `beta_t = 0`：本 token 不修改状态；
- `beta_t` 越大：越积极地把当前 key 对应内容修正到 `v_t`；
- 若 key 做过 L2 归一化且 `beta_t = 1`，沿 `k_t` 方向读取时可直观看成一次完全纠正。

把步骤 1–4 合并：

`S_t = alpha_t*S_(t-1) + beta_t*(v_t - alpha_t*S_(t-1)*k_t)*k_t^T`

也可以写成：

`S_t = alpha_t*S_(t-1)*(I - beta_t*k_t*k_t^T) + beta_t*v_t*k_t^T`

这清楚展示了两件事：旧状态先整体衰减；随后只在当前 key 指向的子空间中擦除错误关联并写入新 value。

### 步骤 5：用 query 读取更新后的状态

`o_t = scale * S_t * q_t`

op12 默认 `scale = 1/sqrt(128)`。输出使用更新后的 `S_t`，因此当前 token 的写入对当前输出可见；这是因果计算，不会看到未来 token。

### 与 op12 代码中的转置方向对应

op12 参考实现内部保存 `state_kv = S^T`，形状为 `[D_k, D_v]`，因此代码写成：

```text
state_kv = alpha_t * state_kv
residual = beta_t * (v_t - k_t @ state_kv)
state_kv = state_kv + outer(k_t, residual)
o_t = scale * (q_t @ state_kv)
```

它与上面的 `[V,K]` 公式完全等价。对外返回 state 时再转回 `[V,K]`。

## 4. 一个一维小例子

令状态、key、value 都是一维，`S_0=0`、`k=1`、`alpha=1`、`beta=0.5`：

1. 第一个 token 希望记住 `v_1=6`：旧预测为 0，误差为 6，写入后 `S_1=3`。
2. 第二个相同 key 希望改成 `v_2=10`：旧预测为 3，误差为 7，写入后 `S_2=3+0.5*7=6.5`。
3. 继续看到相同目标时，状态会逐步逼近 10，而不是每次无条件把 10 叠加进去。

这就是“只写误差”的含义。

## 5. 为什么 prefill 不能简单逐 token 跑

逐 token 递推非常适合 decode，因为一次本来就只有少量新 token。但 prefill 有成百上千个 token；若完全串行，GPU 的矩阵乘资源无法充分利用。

FlashInfer 与 op12 使用 chunked GDN：把序列切成长度 `C=64` 的块。块与块之间仍传递状态，所以保持严格因果；一个块内部则把 64 次递推改写成矩阵运算和一个下三角线性系统，一次处理多个 token。

令一个块内：

- `Q, K, V`：`[C,D]`，op12 中 `C=64, D=128`；
- `S0`：块入口状态，内部方向 `[D_k,D_v]`；
- `a_i = product_(r=1..i)(alpha_r)`：从块开头到 token `i` 的累计衰减；
- `T_ij = a_i/a_j`：token `j` 的写入传播到 token `i` 时经历的衰减；
- `L`：包含对角线的因果下三角 mask。

### 块内七组矩阵乘

#### 1. `Wkk = K * K^T`

得到块内所有 key 两两相似度，形状 `[C,C]`。它描述较早写入会如何影响后续 token 的 residual。

构造单位下三角矩阵：

`A_ij = 1(i=j) + 1(i>j) * beta_i * T_ij * Wkk_ij`

`A` 编码块内递推依赖。因为它是单位下三角矩阵，可用三角求解/求逆一次解决 64 个 token 间的依赖。

#### 2. `Wqk = Q * K^T`

形状 `[C,C]`，表示每个 query 对块内每个 key 的读取系数。施加因果 mask、累计 gate 和输出 scale 后：

`Wqkv_ij = L_ij * T_ij * Wqk_ij * scale`

#### 3. `KS = K * S0`

形状 `[C,D_v]`。这是只考虑块入口旧状态时，每个 key 会读到什么。

#### 4. `QS = Q * S0`

形状 `[C,D_v]`。这是块入口旧状态对每个 query 输出的贡献。

#### 5. `NV = (A^-1 * diag(beta)) * (V - diag(a) * KS)`

先算相对块入口状态的 residual，再通过下三角系统消除块内 token 相互影响，得到每个 token 真正应写入的修正量 `NV`。

#### 6. `O_intra = Wqkv * NV`

聚合当前块内已经发生的写入对每个 query 的贡献。最终块输出：

`O = O_intra + diag(a) * QS * scale`

前半项来自当前块内写入，后半项来自衰减后的块入口状态。

#### 7. `dS = K^T * diag(a_C/a_i) * NV`

把块内所有修正传播到块末尾，再更新最终状态：

`S_end = a_C * S0 + dS`

这个 `S_end` 作为下一块的 `S0`。因此块内可以并行，块间仍按因果顺序链接。

## 6. 七组矩阵乘与逐 token 公式为什么等价

逐 token 形式中，第 `i` 个 residual 依赖于前 `i-1` 个 token 已经写入的内容。把 64 个 residual 同时展开后，会得到一个下三角线性方程组。`A` 就是这个方程组的系数矩阵；`A^-1` 或三角求解把串行依赖一次性消去。

因此 chunked 算法没有改模型语义，只改变计算次序：

```text
逐 token：写入1 -> 写入2 -> ... -> 写入64
分块：构造全部依赖 A -> 三角求解全部修正量 -> 一次生成输出和块末状态
```

尾块不足 64 时必须用 mask 排除无效 token；否则伪造的 padding token 会错误修改 state。

## 7. op12 中唯一改变的精度落点

当前 op12 严格按单变量实验定义：公共 Q/K/V/O 仍为 FlashInfer 原版 FP16/BF16，gate/beta、state、三角逆、residual 和 epilogue 也保持原版；只有七组核心 GEMM 在 MMA 入口把两个 operands 临时、直接转换为 E4M3，MMA scale 恒为 1，accumulator 为 FP32。MMA 完成后立即回到原版 dtype 边界。

不存在外部 FP8 接口、tile scale、amax 归约或反标定。FP32 accumulator 只能减少点积累加误差，不能恢复操作数转 E4M3 时丢失的小数值；本轮失败也主要来自小操作数归零。

## 8. 布局与 head 映射

- 单变量实验保持 FlashInfer packed TND：`Q=[T,Hq,D] + cu_seqlens`。
- BNSD 不是原版公共入口，若后续需要，应作为独立布局适配阶段验收。

输出 head 数 `Ho=max(Hq,Hv)`。GQA 中多个 query head 共享较少的 K/V head；GVA 中多个 value head 共享较少的 Q/K head。每个输出 head 仍维护自己独立的 `[D_v,D_k]` 状态。

## 9. 最容易混淆的五点

1. GDN 不是 softmax attention：没有 softmax 概率矩阵，`QK^T` 只是在分块改写中出现。
2. state 不是 KV cache：它是压缩后的固定矩阵，不保留每个历史 token。
3. gate 与 beta 作用不同：`alpha` 控制旧状态整体遗忘，`beta` 控制当前误差写入。
4. delta 不是简单的 `v`：它是 `v - 当前状态在 k 上的预测`。
5. `[V,K]` 与 `[K,V]` 只是转置约定：先确认实现采用哪种方向，再判断乘法顺序。

## 10. 与 op12 源码的对应关系

- 逐 token FP32 gold：`op12_gdn/reference.py`
- MMA-only FP8 分块基线：`op12_gdn/backend_mma_only.py`
- 历史多变量 backend：`op12_gdn/backend_scaled_mm.py`
- 可编辑图源：`docs/GDN_CALCULATION_FLOW_CN.drawio`

FlashInfer 的公共入口将该算子描述为 chunked gated delta rule prefill，并规定 Q/K/V、gate、beta、state 与 varlen `cu_seqlens` 的输入语义：[FlashInfer `gdn_prefill.py`](https://github.com/flashinfer-ai/flashinfer/blob/e77a4a0d276367895c3b50a642fd8f326c03fb72/flashinfer/gdn_prefill.py)。
