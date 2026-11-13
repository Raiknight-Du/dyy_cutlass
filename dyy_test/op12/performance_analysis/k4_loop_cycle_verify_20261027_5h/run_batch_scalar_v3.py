from pathlib import Path
import argparse,subprocess,json,os,time,csv,datetime,sys
R=Path(__file__).resolve().parent
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def gpu():
    q=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True).strip()
    return dict(UTC=utc(),gpu=q,compute_apps=apps)
def main():
    a=argparse.ArgumentParser();a.add_argument('--phase',choices=['screen','formal','ncu'],required=True);z=a.parse_args()
    out=R/(z.phase+'_scalar_v3');out.mkdir(exist_ok=True);log=(out/'LEDGER.jsonl').open('a')
    def record(o):log.write(json.dumps(o)+'\n');log.flush();print(json.dumps(o),flush=True)
    quiet=0
    while quiet<3:
        g=gpu();ok=not g['compute_apps'] and int(g['gpu'].split(',')[0])<=10
        quiet=quiet+1 if ok else 0;record(dict(event='admission_quiet_check',sequence=quiet,observation=g))
        if quiet<3:time.sleep(30)
    env=dict(os.environ,K4_5H_ADMITTED='1')
    for kind in [0,1]:
        trials=range(3) if z.phase=='formal' else range(1)
        for tr in trials:
            before=gpu();assert not before['compute_apps'],('BUSY_PAUSE',before)
            if int(before['gpu'].split(',')[0])>10:raise RuntimeError(('GPU_NOT_QUIET',before))
            if z.phase=='ncu':
                choices=[(n,m) for n in [128,512] for m in [1,3]]
            else:choices=[(0,0)]
            for n,m in choices:
                tag=f'kind{kind}_t{tr}' if z.phase!='ncu' else f'kind{kind}_N{n}_mode{m}'
                dest=out/tag
                if (dest/'COMPLETE.json').exists():continue
                cmd=[sys.executable,str(R/'run_matched_shared.py'),'--kind',str(kind),'--out',str(dest),'--trial',str(tr)]
                if z.phase=='screen':cmd+=['--points','128','--samples','1','--warmup','2']
                if z.phase=='formal':cmd+=['--points','128,256,512,1024','--samples','30','--warmup','20']
                if z.phase=='ncu':
                    cmd+=['--points',str(n),'--samples','1','--warmup','20','--modes',str(m),'--profile']
                    metrics=['l1tex__data_pipe_lsu_wavefronts.sum','l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum','l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum','l1tex__data_pipe_lsu_wavefronts_mem_shared_op_misc.sum','l1tex__data_pipe_tc_wavefronts.sum','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum','l1tex__data_bank_conflicts_type_arbitration.sum','sm__cycles_elapsed.avg.per_second','l1tex__cycles_elapsed.avg.per_second']
                    cmd=['/opt/nvidia/nsight-compute/2026.2.1/ncu','--profile-from-start','off','--clock-control','none','--cache-control','none','--section','SourceCounters','--metrics',','.join(metrics),'--force-overwrite','-o',str(out/tag)]+cmd
                record(dict(event='start',tag=tag,before=before,command=cmd))
                start=time.monotonic()
                with (out/(tag+'.log')).open('w') as f:
                    ret=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=600).returncode
                after=gpu();record(dict(event='finish',tag=tag,returncode=ret,wall_s=time.monotonic()-start,after=after))
                assert ret==0,('RUN_FAILED',tag)
                assert not after['compute_apps'],('COMPETING_PROCESS_AFTER',after)
                if z.phase=='ncu':
                    for page,flags in [('raw',['--page','raw','--csv']),('source',['--page','source','--print-source','sass','--csv'])]:
                        with (out/(tag+'_'+page+'.csv')).open('w') as f:
                            subprocess.run(['/opt/nvidia/nsight-compute/2026.2.1/ncu','--import',str(out/(tag+'.ncu-rep'))]+flags,stdout=f,check=True,timeout=180)
    record(dict(event='PHASE_COMPLETE',phase=z.phase));log.close()
if __name__=='__main__':main()
