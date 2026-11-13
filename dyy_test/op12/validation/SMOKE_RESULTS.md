# B200 冒烟验证结果

## 结论

固定提交 `e77a4a0d276367895c3b50a642fd8f326c03fb72` 在 NVIDIA B200 上的 op12 定向冒烟验证最终 **7/7 通过**。

正向结果确认：

- FP16/BF16 TND GDN prefill 可运行，并与上游 blockwise reference 一致。
- GQA、GVA、变长 `cu_seqlens` 可运行。
- SM100 CP pipeline 可启动，结果与非 CP 路径一致。
- BF16 主计算配合 FP8 E4M3/E5M2 state 可运行。

负向结果确认：

- FP8 Q/K/V 被公共 SM100 adapter 明确拒绝。
- BNSD 四维 Q/K/V 没有可用公共入口。

## 环境

| 项目 | 值 |
|---|---|
| 主机 | `metallic-matrix-fin-03` |
| GPU | NVIDIA B200 |
| Compute capability | 10.0 |
| Python | 3.10.12 |
| PyTorch | 2.11.0+cu130 |
| Torch CUDA | 13.0 |
| 系统 nvcc | 13.3 |
| 隔离 CuTe DSL | 4.7.0 |
| 隔离 apache-tvm-ffi | 0.1.13.post3 |
| FlashInfer 源码 | `upstreams/flashinfer` |
| FlashInfer commit | `e77a4a0d276367895c3b50a642fd8f326c03fb72` |

隔离环境位于 `op12/.venv`，没有修改系统现有 FlashInfer 0.6.12 或 CuTe DSL 4.5.2。

## 用例明细

| 用例 | 结果 | Output max abs | State max abs | 判定标准 |
|---|---:|---:|---:|---|
| FP16 TND GQA，seq `[64]`，非 CP vs reference | PASS | `3.0518e-05` | `7.0207e-05` | output `atol=2e-3, rtol=1e-3`; state `atol=1e-3, rtol=1e-4` |
| BF16 TND GVA，seq `[64,128]`，非 CP vs reference | PASS | `2.4414e-04` | `6.1315e-04` | output `atol=1e-2, rtol=1e-2`; state `atol=5e-3, rtol=1e-3` |
| BF16 TND，seq `[2048]`，CP vs non-CP | PASS | `2.4414e-04` | `7.5831e-04` | `atol=4e-2, rtol=4e-2` |
| BF16 I/O + FP8 E4M3 state | PASS | `1.2207e-03` | `7.8125e-03` | `atol=1e-1, rtol=5e-2` |
| BF16 I/O + FP8 E5M2 state | PASS | `1.9531e-03` | `1.5625e-02` | `atol=1e-1, rtol=5e-2` |
| FP8 E4M3 Q/K/V | PASS（预期拒绝） | — | — | 捕获 `ValueError: Unsupported dtype ... expected bfloat16 or float16` |
| BNSD `[1,2,64,128]` Q/K/V | PASS（预期拒绝） | — | — | 四维输入没有布局入口，并被 SM100 shape 检查拒绝 |

原始日志同时记录 max relative error；reference 接近零时该值会被放大，不宜脱离 absolute error 与 `assert_close` 容差单独解读。

## 依赖诊断

首轮仅在隔离 venv 中覆盖 CuTe DSL 4.7.0，仍继承系统 `apache-tvm-ffi 0.1.9`。CuTe DSL 4.7.0 的 kwargs wrapper 使用 `map_dataclass_to_tuple` 参数，旧版 TVM-FFI 不接受该参数，导致 5 个正向用例在 JIT 编译前失败。该轮两个负向接口用例正常通过。

在同一隔离 venv 中升级到 `apache-tvm-ffi 0.1.13.post3` 后，7 个用例全部通过。由此将首轮失败归类为 **隔离环境依赖未闭合**，不是 GDN kernel 功能失败。

对应日志：

- `smoke_raw.log`：最终通过日志。
- `smoke_raw_attempt1_dependency_mismatch.log`：首轮依赖不匹配日志。

## 复现

```bash
base=/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12
repo="$base/upstreams/flashinfer"
export FLASHINFER_REPO="$repo"
export PYTHONPATH="$repo"
"$base/.venv/bin/python" "$base/validation/run_smoke.py"
```

本轮未执行性能 benchmark、NCU 或性能结论外推。

