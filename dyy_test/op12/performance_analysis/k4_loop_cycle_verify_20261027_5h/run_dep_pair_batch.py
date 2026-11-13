from pathlib import Path
import argparse,subprocess,sys,json,os,time,datetime
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['screen','formal','ncu'],required=True);a=p.parse_args()
O=R/('dep_pair_'+a.phase);O.mkdir(exist_ok=False)
def gpu():
 return dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,clocks.sm,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip(),apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True).strip())
def log(x):
 with (O/'LEDGER.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
 print(json.dumps(x),flush=True)
q=0
while q<3:
 g=gpu();q=q+1 if not g['apps'] and int(g['gpu'].split(',')[0])<=10 else 0
 log(dict(event='admission',quiet=q,state=g))
 if q<3:time.sleep(30)
for kind in [0,1]:
 for warps in [1,4]:
  for trial in range(3 if a.phase=='formal' else 1):
   for mode in ([0,1] if a.phase=='ncu' else [-1]):
    g=gpu();assert not g['apps'] and int(g['gpu'].split(',')[0])<=10,g
    tag=f'k{kind}_w{warps}_t{trial}_m{mode}';dest=O/tag
    cmd=[sys.executable,str(R/'run_tmem_dep_pair.py'),'--kind',str(kind),'--warps',str(warps),'--out',str(dest),'--trial',str(trial),
       '--points',('512,1024,2048,4096' if a.phase=='formal' else '128'),'--samples',('30' if a.phase=='formal' else '1')]
    if a.phase=='ncu':
     cmd+=['--profile','--modes',str(mode)]
     cmd=['/opt/nvidia/nsight-compute/2026.2.1/ncu','--profile-from-start','off','--clock-control','none','--cache-control','none','--section','SourceCounters','--section','LaunchStats','-o',str(O/tag)]+cmd
    log(dict(event='start',command=cmd,state=g))
    with (O/(tag+'.log')).open('w') as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,K4_5H_ADMITTED='1'),timeout=900)
    g=gpu();log(dict(event='finish',tag=tag,returncode=z.returncode,state=g));assert z.returncode==0,tag;assert not g['apps'],g
    if a.phase=='ncu':
     for page in ['raw','source']:
      flags=['--page',page,'--csv']+(['--print-source','sass'] if page=='source' else [])
      with (O/(tag+'_'+page+'.csv')).open('w') as f:subprocess.run(['/opt/nvidia/nsight-compute/2026.2.1/ncu','--import',str(O/(tag+'.ncu-rep'))]+flags,stdout=f,check=True)
log(dict(event='COMPLETE',phase=a.phase))
