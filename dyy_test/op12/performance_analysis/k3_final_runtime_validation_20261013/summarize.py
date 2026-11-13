"""Produce scoped conclusions from independently recomputed formal evidence."""
import hashlib
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent
a=json.loads((root/'analysis_primary_batch_v1/RESULTS.json').read_text())
b=json.loads((root/'analysis_padded_final_v1/RESULTS.json').read_text())
for name in ('analysis_primary_batch_v1','analysis_padded_final_v1'):
    v=json.loads((root/('VALIDATION_'+name+'.json')).read_text())
    assert v['passed'], name

def events(d,style,mode,ref):
    return [r for r in d['Event'] if r['method']=='graph_batch64' and r['target']==[style,mode] and r['reference']==ref]

def gate(d,style,mode):
    return next(r for r in d['slopes']['strict_convergence_gate'] if r['style']==style and r['mode']==mode)

def ncu(d,name):
    return next(r for r in d['NCU'] if r['name']==name+'_source')

def aggregate_event(d,style,mode,ref):
    rs=events(d,style,mode,ref)
    return {'style':style,'mode':mode,'reference':ref,
            'paired_gain_us':[r['paired_median_gain_us'] for r in rs],
            'median_paired_gain_us':statistics.median(r['paired_median_gain_us'] for r in rs),
            'base_median_us':[r['base']['median_us'] for r in rs],
            'variant_median_us':[r['variant']['median_us'] for r in rs],
            'variant_cv':[r['variant']['cv'] for r in rs]}

event_rows=[aggregate_event(a,s,m,r) for s in ('RUNTIME','COALESCED') for m in (1,2,3) for r in ('MODE3','UPSTREAM')]
event_rows += [aggregate_event(b,s,m,r) for s in ('RUNTIME','PADDED') for m in (1,2,3) for r in ('MODE3','UPSTREAM')]
gates=a['slopes']['strict_convergence_gate']+b['slopes']['strict_convergence_gate']
summaries=[]
for d,name in [(a,'profile_RUNTIME_M1'),(a,'profile_RUNTIME_M2'),(a,'profile_RUNTIME_M3'),
               (a,'profile_COALESCED_M3'),(b,'profile_PADDED_M3')]:
    r=ncu(d,name); st=r['FINAL_candidate_store_account']
    actual=st['L2 Theoretical Sectors Global']; ideal=st['L2 Theoretical Sectors Global Ideal']
    wr=r['FINAL_candidate_shared_STS']; rd=r['FINAL_candidate_shared_LDS']
    summaries.append({'name':name,'bounds':r['FINAL_candidate_bounds'],
       'store_account':st,'sector_utilization':ideal/actual if actual else None,
       'FINAL_PC_samples':r['FINAL_candidate_samples'],
       'FINAL_STS':wr,'FINAL_LDS':rd,
       'STS_actual_ideal_ratio':wr['L1 Wavefronts Shared']/wr['L1 Wavefronts Shared Ideal'] if wr['L1 Wavefronts Shared Ideal'] else None,
       'LDS_actual_ideal_ratio':rd['L1 Wavefronts Shared']/rd['L1 Wavefronts Shared Ideal'] if rd['L1 Wavefronts Shared Ideal'] else None,
       'opcode_dynamic_account':r['opcode_dynamic_account'],
       'whole_metrics':r['whole_metrics']})

old=gate(a,'RUNTIME',3)['stable_grid32_repeat_cycles']
new=gate(b,'PADDED',3)['stable_grid32_repeat_cycles']
assert old and new
result={'scope':'B1 H16 S32768 D128 C4096; K3 FP32 states; HOT; grid32; 1845MHz',
        'graph64_event':event_rows,'strict_helper_slopes':gates,'NCU_FINAL':summaries,
        'helper_padding_service_time_reduction':1-new/old,
        'model_enabled':False,'effective_model_value':None,
        'sanitizer_status':'NOT_PASSED; host double-free crash, no valid memcheck conclusion',
        'excluded':['event_v2','slope_v1','slope_v2','layout_event_v1','padded_slope_v1'],
        'no_upstream_edits':True,'no_existing_model_or_report_edits':True}
result['helper_NCU']=[r for d in (a,b) for r in d['NCU'] if r['name'].startswith('micro_profile_')]
(root/'EXPERIMENT_SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))

lines=[
'K3 FINAL 第二轮验证结果（2026-10-13）',
'',
'结论',
'1. 同一 CUBIN 的运行时模式对照确认：第二路 Store 的有效请求与完成开销是 FINAL 成本的重要来源；不是另一遍 TMEM readback，也不是不同编译变体的 spill 差异造成该对照收益。',
'2. 原分布 Store 的逻辑 sector 利用率为50%；重排可提高到100%。未加 padding 的重排产生严重共享内存 wavefront 放大，抵消收益。padding helper 明显改善持续服务，但完整 K3 相对上游没有稳定加速。',
'3. 不把 NCU 采样比例换算成 cycles，不把 helper 斜率当作原 FINAL 的暴露延迟；本轮不能给出3944−2048的精确逐项周期分摊。',
'',
'一、范围和方法',
'固定 FlashInfer e77a4a0d276367895c3b50a642fd8f326c03fb72；B200/148SM；B1/H16/S32768/D128/C4096/P8。K1/K2 数据为 FP16，K3 的 M/N、fixed/final state 为 FP32。K3 grid32/block256；本实验不是单 SM 带宽测试。',
'所有变体在独立实验副本；上游 clean。mode1=fixed-only、mode2=final-only、mode3=both，三种模式调用同一个 CUBIN，只改变运行时 Int32 参数。TMEM 读取位于两路 Store 条件之外。',
'完整 K3 使用真实 K1/K2 前缀 fixture，计时前 prime 原 K3 的 HOT 上下文；这不是自然完整 CP 链的冷缓存环境。20次预热、100次采样、3独立进程、AB/BA交错配对。',
'精确对照用64个相同 K3 调用的 CUDA Graph Event总时长/64，包含图内设备 launch 间隙。不得和单次直接调用的绝对延迟混用。原始单次 Event 数据另保留。',
'临时锁1845MHz，频率轨迹10ms；正式轮均无外部竞争且频率有效，退出恢复默认。NCU application replay/cache-control none，仅分析计数器和PC，NCU时长不用于提速。',
'',
'二、完整 K3：同一编译产物单路/双路对照',
]
for style,mode in [('RUNTIME',1),('RUNTIME',2),('COALESCED',1),('COALESCED',2)]:
    z=aggregate_event(a,style,mode,'MODE3')
    lines.append(f'{style} mode{mode}：mode3−单路，三轮 {z["paired_gain_us"]} µs/调用，中位 {z["median_paired_gain_us"]:.6f} µs。')
for d,style in [(a,'COALESCED'),(b,'PADDED')]:
    z=aggregate_event(d,style,3,'UPSTREAM')
    lines.append(f'{style} mode3相对上游：三轮配对收益 {z["paired_gain_us"]} µs；正数表示变体更快。上游中位 {z["base_median_us"]} µs，变体 {z["variant_median_us"]} µs。')
lines += [
'PADDED 相对原版收益符号不一致、接近计时差异：没有稳定的完整 K3 性能收益。上游与运行时副本不是同一编译产物；二者比较不能当作纯 Store 消融。',
'final-only模式的 fixed workspace 仍由原 UPDATE 循环完整写出，fixed/final 与上游对照通过；fixed-only的 final 故意保持 sentinel。所有正式正确性误差≤1e-6。',
'',
'三、FINAL helper：严格斜率结果',
'仅1次 GMEM→TMEM初始化，循环中使用原 FINAL TMEM readback 与输出路径；无 MMA/TMA/partition递推，固定rotating8输出、R=16…2048。grid32，输出工作集16MiB。',
'后两组三点窗口 [256,512,1024] 与 [512,1024,2048]；同进程及跨三进程斜率变化均严格<1%，R²≥0.99，CV≤5%，P5–P95/median≤10%。每种正式路径均PASS。',
]
for d,style,mode in [(a,'RUNTIME',1),(a,'RUNTIME',2),(a,'RUNTIME',3),(a,'COALESCED',3),(b,'PADDED',3)]:
    g=gate(d,style,mode)
    lines.append(f'{style} mode{mode}: {g["stable_us_per_repeat"]:.9f} µs/轮；Event等效 {g["stable_grid32_repeat_cycles"]:.3f} cycles/轮@1845MHz；跨轮斜率跨度 {g["cross_process_slope_spread"]*100:.6f}%。')
lines += [
f'padding helper 持续服务时间降低 {(1-new/old)*100:.2f}%，但不是完整 K3 提速。原始双路−单路约 {old-gate(a,"RUNTIME",1)["stable_grid32_repeat_cycles"]:.3f} cycles/轮，包含额外有效 Store、背压及其相关依赖，不能叫裸 Store 指令延迟。',
'helper编译REG255、无local spill；完整K3编译REG120且有local访问。helper没有原递推/访存前史，故不能直接映射到FINAL的IKET区间或硬件GMEM/HBM资源速率。',
'',
'四、NCU/SASS证据',
]
for r in summaries:
    st=r['store_account']; s=r['FINAL_PC_samples']
    lines.append(f'{r["name"]}: FINAL候选PC范围 {r["bounds"]}；Store warp opcode执行 {int(st["Instructions Executed"])}，predicate-on thread执行 {int(st["Predicated-On Thread Instructions Executed"])}；theoretical sectors {int(st["L2 Theoretical Sectors Global"])} / ideal {int(st["L2 Theoretical Sectors Global Ideal"])}，sector利用率 {r["sector_utilization"]*100:.1f}%。')
    lines.append(f'  FINAL PC采样 all={int(s["Warp Stall Sampling (All Samples)"])}，lg={int(s["stall_lg"])}，long_sb={int(s["stall_long_sb"])}，mio={int(s["stall_mio"])}，short_sb={int(s["stall_short_sb"])}；新增shared STS实际/ideal wavefront={r["STS_actual_ideal_ratio"]}，LDS={r["LDS_actual_ideal_ratio"]}。')
lines += [
'运行时M1/M2/M3：LDTM18432、UTCHMMA8192、UTMALDG2048、LDL2784、STL1024和STG warp opcode计数22528均相同；FINAL active Store thread计数由65536变为131072。运行时谓词不是消去静态Store位置，而是改变有效请求。不能由相同opcode计数断言发射成本完全相同。',
'原 FINAL 单路逻辑payload1MiB/theoretical sector footprint2MiB；双路2MiB/4MiB。足迹不是实测物理HBM流量。50% sector利用率使有效Store负担约为理想连续布局的2倍，是2048理论预算低估现实服务需求的重要线索；尚不能证明所有gap来自此处。',
'未padding重排的STS实际/ideal wavefront比8倍，PC热点从global Store lg转向SMEM/MIO；padding的STS/LDS实际/ideal均为1倍，确认消除了该放大。但新增SMEM读写和同步并未消失。',
'完整K3的NCU动态SMEM：RUNTIME164864B、COALESCED197632B、PADDED198656B，均另有1024B driver占用，仍为shared-memory限1 CTA/SM。PADDED的local请求4448，RUNTIME为3808；跨布局比较不是资源完全不变的纯Store优化。',
'lg_throttle表示local/global队列压力；long scoreboard表示相关L1TEX结果依赖等待。Store队列压力与TMEM读回/寻址/寄存器复用的相关等待仍需分离，不能把long scoreboard直接命名为HBM等待或TMEM裸延迟。',
'PC采样数量与比例不等于实际stage cycles；本轮只作为热点与变体变化的证据。',
'',
'五、各假设的验证状态',
'额外一路Store：支持。相同CUBIN/TMEM/MMA/TMA/local工作下单路更快，有效Store请求和sector footprint减半。',
'第二路引发Store背压/相关依赖：支持存在Store队列压力；无法拆分精确RAW/WAR或wait cycles。独立helper无local spill及递推，仍能复现双路成本。',
'store_acc(True)额外循环/同步：未发现额外TMEM readback或新增每路同步；运行时副本的两路Store均读取同一rAcc。重排变体另有显式SMEM/compute-group同步，不属于原版。',
'FINAL与UPDATE TMEM读回不同：此前源码与编译产物表明同一读回路径；本轮保留该读回，不支持把差值单独归因于另一遍TMEM读取。',
'理论2048预算：payload/假设速率给出的主导资源下界；即使原预算以max纳入了TMEM读回，也没有反映实际sector分布、队列、寻址/谓词和这些操作的暴露依赖。需要分层模型，不能从实测反向调整硬件速率来贴合。',
'',
'六、限制和异常',
'本轮只测一个shape/缓存条件，不外推所有K3配置。完整K3私有运行时副本STACK32/local访问多于上游STACK16；同一CUBIN模式间该项不变，跨实现比较仍有混杂。',
'compute-sanitizer出现宿主 double free/corruption，未得到有效memcheck结论；ERROR SUMMARY 0不视为通过。正式变体有数值正确性证据，但越界/竞争专项门禁仍未闭合。',
'swizzle变体数值检查出现NaN，已排除；未诊断根因，不归咎于硬件。padded_slope_v1的helper误用64-launch图，发现后停止自有进程，原数据保留并排除；v2已修正为一次helper/图。历史padded Event图确为64次K3，已重建原host并匹配SHA。',
'早期构建/launch错误及不完整实验均保存日志，不混入正式结果。',
'',
'七、复现与证据索引',
'最终汇总：EXPERIMENT_SUMMARY.json；完整详细派生：analysis_primary_batch_v1/RESULTS.json、analysis_padded_final_v1/RESULTS.json。',
'原始Event：runs/event_batch64_v1、runs/padded_event_v1；原始斜率：runs/slope_v3、runs/padded_slope_v2；单次Event：runs/event_v3。',
'NCU八份：runs/ncu_v1（六份）、runs/padded_ncu_v1（两份），含.ncu-rep、raw/source CSV、SASS/PTX/CUBIN、运行日志和频率轨迹。',
'复算门禁：VALIDATION_analysis_primary_batch_v1.json、VALIDATION_analysis_padded_final_v1.json；保留原始239项及后续复算结果，memcheck状态另列。',
'冻结输入与版本：inputs_snapshot、SOURCE_MANIFEST.json、BATCH_MANIFEST.json、PADDED_MANIFEST.json、HOST_HISTORY_VERIFICATION.json。',
'没有修改cycle.xlsx、旧模型、旧报告、固定上游；本轮参数未自动批准或入模。',
'官方指标说明：https://docs.nvidia.com/nsight-compute/NsightCompute/index.html；https://docs.nvidia.com/nsight-compute/ComputeTriage/；https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html。',
]
(root/'K3_FINAL_RUNTIME_VALIDATION_RESULTS_CN.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'summary_created':True,'helper_padding_reduction':1-new/old},indent=2))
