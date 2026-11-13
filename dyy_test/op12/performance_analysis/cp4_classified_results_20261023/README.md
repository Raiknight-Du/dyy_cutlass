# OP12 CP四kernel：性能与理论核验分栏

以用户最新裁决为准，取消完整模型上游补测。两类均为原版完整CP、B1/H16/D128/FP16、FP32 state、P8；S8192/16384/32768对应C1024/2048/4096。

| 分类 | K1前缓存边界 | 频率策略 | 结果用途 |
|---|---|---|---|
| `01_performance_cold_auto` | 每次K1前执行506 MiB驱逐 | 自动、不锁频 | 本项目性能口径，不称完整模型生产实测 |
| `02_theory_cold_locked1845` | 同样的驱逐 | 请求1845 MHz锁频，并核查NVML | 理论分析的受控实验依据；理论cycle本身不改 |

## 当前结果与图表入口

完整CP的独立CUDA Event时延如下，单位µs，驱逐不计时；三个独立进程中位数的中位数：

| S | 性能：冷读、不锁频 | 理论核验：冷读、锁1845 |
|---:|---:|---:|
| 8192 | 143.360 | 153.536 |
| 16384 | 213.024 | 228.336 |
| 32768 | 350.240 | 374.784 |

- 性能类：[K1—K4 GPC轨迹](01_performance_cold_auto/GPC_FREQUENCY_TIMELINE.png)、[独立Event进程NVML轨迹](01_performance_cold_auto/NATIVE_NVML_FREQUENCY_TIMELINE.png)。
- 理论核验类：[K1—K4 GPC轨迹](02_theory_cold_locked1845/GPC_FREQUENCY_TIMELINE.png)、[独立Event进程NVML轨迹](02_theory_cold_locked1845/NATIVE_NVML_FREQUENCY_TIMELINE.png)。
- [两类Event对照图](CP_EVENT_TWO_CATEGORIES.png)；完整统计和原始样本索引见各分类CSV与`EVIDENCE_INDEX.json`。

两类均覆盖三组shape。无分析器计时窗口内，当前复用的不锁频冷读组NVML采样均为1965 MHz，锁频冷读组均为1845 MHz。GPC指标则不等于上述NVML快照，不能据设定频率将Event强行换算成精确cycle。

## 时延与频率如何阅读

- `CP_EVENT_SUMMARY.csv`：无分析器、每进程300次CP Event，中位数；最后取三个独立进程中位数的中位数。只包络K1—K4，驱逐不计时。
- `KERNEL_HW_GPC_SUMMARY.csv`：NSYS进程的分kernel硬件区间和GPC指标，**不是无分析器分kernel Event真值**。
- `GPC_FREQUENCY_TIMELINE.png`：每个shape固定选trial2、batch1、call50，绘制同次调用的真实K1—K4端点和约5µs采样GPC，不拼接阶段均值。
- `NATIVE_NVML_FREQUENCY_TIMELINE.png`：无分析器进程trial2的20ms遥测与计时批次。其采样稀疏，不代表每个kernel的精确频率。
- Event与GPC来自不同进程，禁止将两者汇总中位数相乘冒充精确cycle。即使NVML报告1845，GPC区间均值仍可低于1845；本次不消除这一未解决的时钟语义差异。
- 所有派生统计从原始Event样本和逐kernel CSV重新计算。原始文件位置和SHA见`EVIDENCE_INDEX.json`；具体图中调用见`FIGURE_CALL_IDENTITIES.csv`。

## 缓存条件

每次：506 MiB非零GPU填充 → CP Event开始 → K1→K2→K3→K4 → CP Event结束。两类都保留链内自然缓存，不在K2/3/4前驱逐。既有NCU诊断在S8192上确认K1冷读：K+beta逻辑32.5 MiB，HBM读32.527 MiB；该诊断不等于逐一验证所有shape每次调用的每条缓存线。驱逐在Event外，但仍可能改变功耗、温度和负载节奏。

## 复用与最小补测

复用此前三shape的cold/auto及S32768/cold/locked1845原始数据。只补测S8192、S16384的cold/locked1845：每配置三独立进程，每进程20次图预热、3批×100次完整CP；native与NSYS各一组，共12次进程。复用原测量脚本，不修改上游。GPU空闲准入连续三次、间隔30s；发现外部竞争则作废并停止。结束恢复自动GPU时钟，不改显存频率或功耗上限。

## 清理与恢复

旧版两套混合展示及其原始证据从主目录移至`../_superseded/cp4_frequency_20261023/`，旧运输包一并归档。**未永久删除原始数据**。旧图不再作为当前结论入口；原始证据仍供新结果追溯。源文件内容及既有SHA清单保持不变，归档迁移记录见`CLEANUP_MANIFEST.json`。20260929冻结输入及正式模型文件不移动、不覆盖。

重新生成：在B200本目录执行`../../.venv/bin/python organize_results.py`（实际解释器为op12/.venv/bin/python；脚本自动识别归档路径）。
