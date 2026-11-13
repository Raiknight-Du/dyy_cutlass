"""Two application-replay captures under shared guard, with full CP cache history."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent; out=root/'NCU'; out.mkdir(exist_ok=False)
ncu='/opt/nvidia/nsight-compute/2026.2.1/ncu'
assert os.environ.get('K1_GPU_SESSION')
def run(command,log,timeout=1800):
    record=dict(command=command,start_epoch=time.time())
    with log.open('w') as f: p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    record.update(returncode=p.returncode,end_epoch=time.time()); log.with_suffix(log.suffix+'.command.json').write_text(json.dumps(record,indent=2))
    assert p.returncode==0,log
run([ncu,'--version'],out/'version.txt')
run([ncu,'--query-metrics'],out/'available_metrics.txt')
available=(out/'available_metrics.txt').read_text()
requested=['dram__bytes_read.sum','dram__bytes_write.sum','lts__t_sectors_srcunit_tex_op_read.sum','l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum','l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum','l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum','l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum']
metrics=[x for x in requested if x.split('.')[0] in available]
(out/'METRIC_SELECTION.json').write_text(json.dumps(dict(selected=metrics,UNAVAILABLE=[x for x in requested if x not in metrics]),indent=2))
assert 'dram__bytes_read.sum' in metrics and 'lts__t_sectors_srcunit_tex_op_read.sum' in metrics
for role in ['B0','V0']:
    assert not (Path(os.environ['K1_GUARD_DIR'])/'INVALID_COMPETITION').exists()
    command=[ncu,'--replay-mode','application','--cache-control','none','--clock-control','none','--target-processes','all','--nvtx','--nvtx-include','K4_MATCHED/','--kernel-name','regex:.*CPDeltaRulePrefillTcgen05Sm100.*','--launch-count','1','--export',str(out/role),'--section','LaunchStats','--section','SourceCounters','--section','InstructionStats','--metrics',','.join(metrics),sys.executable,str(root/'run_matched.py'),'--mode','profile','--profile',role,'--out',str(out/(role+'_replays'))]
    run(command,out/(role+'.log'))
    for page,suffix,extra in [('raw','raw',[]),('source','source',['--print-source','sass'])]:
        run([ncu,'--import',str(out/(role+'.ncu-rep')),'--page',page,'--csv','--print-units','base',*extra],out/(role+'_'+suffix+'.csv'))
    print('NCU_CAPTURE_COMPLETE',role,flush=True)
(out/'COMPLETE.json').write_text(json.dumps(dict(capture_invocations=2,status='CAPTURED_PENDING_REVIEW',NCU_time_used_for_performance=False,model_enabled=False),indent=2))
