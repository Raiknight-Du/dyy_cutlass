"""Read-only Nsight metric catalogue queries; never supplies an application to ncu."""
from pathlib import Path
import subprocess,json,csv,io,hashlib
O=Path(__file__).resolve().parent;out=O/'r3_hw_only_metric_query01';out.mkdir(exist_ok=False)
ncu='/opt/nvidia/nsight-compute/2026.2.1/ncu'
base=['dram__bytes_read','lts__t_sectors_srcunit_tex_op_read','lts__t_sectors_srcunit_tex_op_read_lookup_hit','lts__t_sectors_srcunit_tex_op_read_lookup_miss','l1tex__data_pipe_lsu_wavefronts_mem_lgds_cmd_read','sm__cycles_elapsed','lts__cycles_elapsed','dram__cycles_elapsed','gpu__time_duration']
exact=[x+'.sum' for x in base[:5]]+[x+'.avg' for x in base[5:8]]+['gpu__time_duration.sum']+[x+'.avg.per_second' for x in base[5:8]]
commands=[('PROFILE_CATALOGUE',[ncu,'--chips','gb100','--query-metrics-mode','all','--metrics',','.join(base),'--csv']),
 ('STATS_CATALOGUE',[ncu,'--chips','gb100','--query-metrics-collection','stats','--query-metrics','--csv']),
 ('SELECTED_LIST',[ncu,'--section','LaunchStats','--metrics',','.join(exact+['profiler__replayer_passes']),'--list-metrics'])]
records=[]
for name,cmd in commands:
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
    (out/(name+'.stdout')).write_text(r.stdout);(out/(name+'.stderr')).write_text(r.stderr)
    records.append({'command':cmd,'returncode':r.returncode,'stdout_SHA256':hashlib.sha256(r.stdout.encode()).hexdigest()})
    assert r.returncode==0,(name,r.stderr)
rows=list(csv.DictReader((out/'PROFILE_CATALOGUE.stdout').open()))
selected={r['Metric name']:r for r in rows if r['Metric name'] in exact}
stats=list(csv.DictReader((out/'STATS_CATALOGUE.stdout').open()))
selected.update({r['Metric name']:r for r in stats if r['Metric name']=='profiler__replayer_passes'})
result={'selected_full_names':exact+['profiler__replayer_passes'],'catalogue_rows':selected,
 'catalogue_missing':[n for n in exact+['profiler__replayer_passes'] if n not in selected],
 'commands':records,'pass_count':'UNDETERMINED_WITHOUT_TARGET_COLLECTION_CONFIGURATION',
 'expectation':'Small direct hardware set, intended to minimize passes; no offline CLI pass-count proof. Actual profiler__replayer_passes/log must decide, never assume one pass.',
 'excluded':['SourceCounters','InstructionStats','warp sampling','PM sampling','new kernel','new rate measurement'],
 'GPU_kernel_launches':0,'model_enabled':False}
(out/'QUERY_SUMMARY.json').write_text(json.dumps(result,indent=2));print(json.dumps({'missing':result['catalogue_missing'],'pass_count':result['pass_count'],'queries':len(records)}))
