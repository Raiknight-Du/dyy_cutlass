from pathlib import Path
import argparse,sys,subprocess,time,os,json,datetime
R=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--phase',choices=['screen','formal','ncu'],required=True);a=p.parse_args();O=R/('alu_mix_'+a.phase);O.mkdir(exist_ok=False)
def gpu():return dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,clocks.sm,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip(),apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True).strip())
def log(v):
 with (O/'LEDGER.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
 print(json.dumps(v),flush=True)
for i in range(3):
 g=gpu();assert not g['apps'] and int(g['gpu'].split(',')[0])<=10;log(dict(event='admission',quiet=i+1,state=g))
 if i<2:time.sleep(30)
for tr in range(3 if a.phase=='formal' else 1):
 for mode in ([1,2,3] if a.phase=='ncu' else [-1]):
  for point in ([256,512] if a.phase=='ncu' else [-1]):
   tag=f't{tr}_m{mode}_p{point}';cmd=[sys.executable,str(R/'run_alu_mix.py'),'--out',str(O/tag),'--trial',str(tr),'--points',('1024,2048,4096,8192' if a.phase=='formal' else str(point if point>0 else 128)),'--samples',('30' if a.phase=='formal' else '1')]
   if a.phase=='ncu':
    cmd+=['--profile','--modes',str(mode)];cmd=['/opt/nvidia/nsight-compute/2026.2.1/ncu','--profile-from-start','off','--clock-control','none','--cache-control','none','--section','SourceCounters','--section','LaunchStats','--metrics','smsp__inst_executed_pipe_alu.sum,smsp__inst_executed_pipe_fmaheavy.sum,smsp__inst_executed_pipe_fmalite.sum,smsp__inst_executed_pipe_xu.sum','-o',str(O/tag)]+cmd
   g=gpu();assert not g['apps'],g;log(dict(event='start',cmd=cmd,state=g))
   with (O/(tag+'.log')).open('w') as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,K4_5H_ADMITTED='1'),timeout=900)
   log(dict(event='finish',tag=tag,returncode=z.returncode,state=gpu()));assert z.returncode==0,tag
   if a.phase=='ncu':
    for page in ['raw','source']:
     flags=['--page',page,'--csv']+(['--print-source','sass'] if page=='source' else [])
     with (O/(tag+'_'+page+'.csv')).open('w') as f:subprocess.run(['/opt/nvidia/nsight-compute/2026.2.1/ncu','--import',str(O/(tag+'.ncu-rep'))]+flags,stdout=f,check=True)
log(dict(event='COMPLETE'))
