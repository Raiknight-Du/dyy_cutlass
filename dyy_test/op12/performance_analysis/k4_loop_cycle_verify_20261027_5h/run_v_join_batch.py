from pathlib import Path
import subprocess,sys,json,os,time,datetime,argparse
R=Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument('--phase',choices=['screen','ncu'],required=True);a=a.parse_args()
O=R/('vjoin_'+a.phase);O.mkdir(exist_ok=True)
def state():
 return dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip(),
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True).strip())
def rec(x):
 with (O/'LEDGER.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
 print(json.dumps(x),flush=True)
quiet=0
while quiet<3:
 s=state();quiet=quiet+1 if not s['apps'] and int(s['gpu'].split(',')[0])<=10 else 0
 rec(dict(event='admission',quiet=quiet,state=s))
 if quiet<3:time.sleep(30)
env=dict(os.environ,K4_5H_ADMITTED='1',PYTHONDONTWRITEBYTECODE='1')
for variant in (['QSJOIN','KSJOIN'] if a.phase=='screen' else ['B0','KSJOIN','QSJOIN']):
 for trial in range(1 if a.phase=='screen' else 3):
  tag=f'{variant}_t{trial}';dest=O/tag
  if (dest/('run' if a.phase=='screen' else 'DONE')/'RESULT.json').exists():continue
  before=state();assert not before['apps'] and int(before['gpu'].split(',')[0])<=10,before
  cmd=[sys.executable,str(R/'run_v_join.py'),'--variant',variant,'--out',str(dest),'--samples','3']
  if a.phase=='ncu':
   metrics=['l1tex__data_pipe_lsu_wavefronts.sum','l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum','l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
    'l1tex__data_pipe_tc_wavefronts.sum','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
    'l1tex__data_bank_conflicts_type_arbitration.sum','dram__bytes_read.sum','dram__bytes_write.sum','lts__t_sectors.sum','l1tex__cycles_elapsed.avg.per_second','sm__cycles_elapsed.avg.per_second']
   cmd=['/opt/nvidia/nsight-compute/2026.2.1/ncu','--profile-from-start','off','--replay-mode','application','--cache-control','none','--clock-control','none',
    '--kernel-name','regex:.*(CPDeltaRulePrefillTcgen05Sm100|K4VJoin).*','--section','SourceCounters','--section','LaunchStats','--metrics',','.join(metrics),'-o',str(O/tag)]+cmd+['--profile']
  rec(dict(event='start',tag=tag,before=before,command=cmd))
  with (O/(tag+'.log')).open('w') as f:q=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
  after=state();rec(dict(event='finish',tag=tag,returncode=q.returncode,after=after));assert q.returncode==0,tag;assert not after['apps'],after
  if a.phase=='ncu':
   for page in ['raw','source']:
    flags=['--page',page,'--csv','--print-units','base']+(['--print-source','sass'] if page=='source' else [])
    with (O/(tag+'_'+page+'.csv')).open('w') as f:subprocess.run(['/opt/nvidia/nsight-compute/2026.2.1/ncu','--import',str(O/(tag+'.ncu-rep'))]+flags,stdout=f,check=True)
rec(dict(event='COMPLETE',phase=a.phase))
