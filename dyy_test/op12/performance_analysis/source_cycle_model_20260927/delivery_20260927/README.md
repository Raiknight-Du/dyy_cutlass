# op12 四kernel源码cycle分析交付索引

固定FlashInfer原版提交 `e77a4a0d276367895c3b50a642fd8f326c03fb72`，B200/148SM。三组B1/H16/D128的S8192/16384/32768，C1024/2048/4096，P均为8。FP16/BF16输入，FP32 state；无FP8实验或上游修改。

## 主交付

- [综合中文报告](OP12_SOURCE_DRIVEN_CYCLE_MODEL_CN.md)：四kernel源路径、DAG、角色、依赖/重叠、首稳尾cycle、三shape、假设、审查和事后Event对照。
- [公式工作簿](outputs/source_model_20260927/OP12_FOUR_KERNEL_SOURCE_CYCLE_MODEL_20260927.xlsx)：统一Assumptions，工作/流量，四kernel明细，19个逐节点服务日历，资源容量与依赖span对照，敏感性和Event检查。
- [计算DAG Draw.io源](OP12_SOURCE_CYCLE_DAGS_20260927.drawio)：全局及四kernel、口径六页。
- [pipeline Draw.io源](OP12_SOURCE_PIPELINES_20260927.drawio)：cohort、双branch、预取/release、chunk pair四页。

图源使用Draw.io MCP建立；figures与pipeline_figures为本地离线结构预览，没有向第三方上传源码/图内容。

## 结果与限制

|S|K1 cy|K2 cy|K3 cy|K4 cy|合计 cy|
|---:|---:|---:|---:|---:|---:|
|8192|30879|46619|50203|57560|185261|
|16384|61753|83643|50203|107032|302631|
|32768|123487|157691|50203|205976|537357|

这些是条件化理想服务日历，不是无条件的物理时延下界。共享SMEM/TMEM/TC不按warp组复制；未知响应、完整frontend issue、SFU/barrier、缓存和launch/gap保留符号。特别是每SM1/148的HBM配额仅为假设，欠并行不证明HBM按SM覆盖率下降。

`SOURCE_MODEL_FREEZE.json`在读取目标时延前固定15个模型/脚本/审查hash和5个关键源码hash。独立Sweep重放19个日历，全部一致；review源码计数/phase错误已修正，硬件未知项未强行填数。`event_sanity_check.json`仅在冻结后读取三个旧Event trial，不向理论公式反向馈入数据。

事后检查：K3数量级较接近；K2与K4仍有明显未解释残差。现有stage Event是2warmup/20次、固定重复输入，部分CV>5%；完整CP另有20warmup/100次和rotating buffers。独立stage之和不是端到端Event，也不使用NCU replay时延。

## 源证据

- `hardware_model.md/json`、`environment_snapshot.json`：共同常数、分类、速率单位及设备快照。
- `k1..k4_analysis.md`、`k1..k4_model.json`：完整源码位置、bytes/FLOP/dtype/角色、子步和依赖。
- `approved_model.json`：统一模型；`numerical_sweep.py/json/md`：独立数值复算。
- `reviewer_findings.md/json`：审查问题与修正状态。
- `static_artifacts`：未计时的原版PTX/CUBIN/SASS、指令静态站点、资源元数据及无launch occupancy查询。
- `outputs/source_model_20260927/validation.json`：19个Excel日历公式与模型span完全对账、公式错误扫描和输入修改/恢复检查。
- `delivery_integrity.json`：冻结文件及辅助源码hash复核、最终交付hash。
- `source_archive`：关键固定源码与alpha/varlen helper的只读副本；与冻结文件中的hash一致，供离线核对。
- `event_raw`：事后检查所用三个旧Event trial的相同hash副本，不馈入源模型。

## 复现说明

工作簿使用原本保存的每resource仲裁顺序，改rate后会重新计算服务/start/end及summary，但不会重新优化调度。每packet用ceil(demand/rate)；start=max(源前驱end、同resource前job end)。不得用此操作把参数调到实测时延。

从冻结模型独立复算：运行 `numerical_sweep.py`。源日历重建：`k2_calendar.mjs`、`k4_refresh_calendar.mjs`、`normalize_model.mjs`。已有冻结模型不得在读取Event后覆盖；若未来发现源码错误，应新建版本、独立说明修正原因并重新冻结，保留本版。

原固定clone最后核对HEAD/origin均正确，`git status --porcelain`为空。此目录独立于上游，不修改旧报告或旧工作簿。B200归档路径为 `/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/performance_analysis/source_cycle_model_20260927/`。
