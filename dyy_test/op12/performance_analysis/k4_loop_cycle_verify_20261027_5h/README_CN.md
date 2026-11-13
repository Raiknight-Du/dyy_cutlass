# K4 循环周期核验交付

先看 `K4_LOOP_CYCLE_ANALYSIS_CN.md` 和 `K4_LOOP_CYCLE_REVIEW_20261027.xlsx`。

本轮范围：原版 FP16、B1/H16/D128、S8192/16384/32768、C1024/2048/4096、完整CP缓存前史；比较每个64-token块的主体返回节拍。

同窗口条件模型为3430.25–3435.25cy/块。对应观测斜率3175.3、3163.7、3141.2；主情景带符号偏差+8.03%至+9.36%，明确属于高估。历史TMEM读流速率敏感性为3436.625–3441.625，仍在三组shape的±10%以内。

旧2869–3131cy近似漏计总发射容量，已撤回并保存在`history/pre_issue_capacity`。最终加入四warp/SM-cycle聚合约束；单LDTM就绪值26/74cy来自独立NOP配对与原生首wait位置q=2，不使用整条原语链时长作节点，不选择有利的宽区间端点。

数字达到本轮±10%目标，主要原生计数与NCU对齐；这是具名条件速率及资源/依赖模型，不是全部硬件常数已识别。观测为插桩CG1返回点，局部偏差仍有限制。没有将完整INIT/UPDATE/FINAL/MMA包的实测时长填入理论节点，也没有按目标Gap回填常数。

主要新增证据：CG0 132个shared PC逐lane枚举；六类shared序列隔离/并行TC的24份计数验证；完整CP中移动CG1等待位置的12份NCU；精确LDTM↔NOP单指令控制的8份NCU及严格收敛完成边界；PACK/IADD3共享条件的6份NCU及修正活跃warp端点后的独立Event/native斜率。

`EVIDENCE_COHORTS.json`明确哪些原始轮次可用、哪些仅作诊断或已被替代。尤其不要从旧`*SENSITIVITY.json`取未经核验的lag；不要复用早期混入inactive warp的ALU周期中位数；不要把旧完整阶段helper的拟合时长当理论参数。

本地复算（CPU，不启动新GPU实验）：

```powershell
python .\recompute_loop_reference.py
python .\derive_single_ldtm_completion.py
python .\recompute_final_model.py
python .\build_delivery_data.py
```

图表脚本另需matplotlib；Excel作者脚本使用Codex配置的`@oai/artifact-tool`。修改资源速率后先重新运行Python调度，再刷新工作簿；工作簿的DAG结果列是调度快照，原生work/rate及NCU对账为公式。四warp锁步/FIFO模型并非逐SMSP硬件模拟；约255–294cy的正偏差仍需保留，不能解释成尚缺的正开销或因果Gap已闭合。

原始实验可在 `raw_evidence` 及证据压缩包复核；NCU版本2026.2.1。远端对应目录为：

`/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/performance_analysis/k4_loop_cycle_verify_20261027_5h/`

所有候选保持`model_enabled=false`、`effective_model_value=null`。没有修改原版上游、正式模型或历史证据。
