from pathlib import Path
import argparse,subprocess,sys,json,os,time,datetime
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['ts_screen','ts_formal','dep_screen','dep_formal'],required=True);a=p.parse_args()
O=R/a.phase;O.mkdir(exist_ok=False)
def gpu():
 return dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,clocks.sm,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip(),apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True).strip())
def log(x):
 with (O/'LEDGER.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
 print(json.dumps(x),flush=True)
q=0
while q<3:
 g=gpu();q=q+1 if not g['apps'] and int(g['gpu'].split(',')[0])<=10 else 0
 log(dict(event='admission',quiet=q,state=g))
 if q<3:time.sleep(30)
formal=a.phase.endswith('formal');ts=a.phase.startswith('ts')
for kind in ([3] if ts else [0,1]):
 for trial in range(3 if formal else 1):
  g=gpu();assert not g['apps'] and int(g['gpu'].split(',')[0])<=10,g
  tag=f'kind{kind}_t{trial}';cmd=[sys.executable,str(R/('run_matched_ts.py' if ts else 'run_tmem_read_dep.py')),
    '--kind',str(kind),'--out',str(O/tag),'--trial',str(trial),'--points',('128,256,512,1024' if formal else '128'),
    '--samples',('30' if formal else '1')]
  if ts:cmd+=['--modes','1,2,3','--warmup',('20' if formal else '2')]
  log(dict(event='start',command=cmd,state=g))
  with (O/(tag+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,K4_5H_ADMITTED='1'),timeout=900)
  g=gpu();log(dict(event='end',tag=tag,returncode=r.returncode,state=g));assert r.returncode==0,tag;assert not g['apps'],g
log(dict(event='COMPLETE',phase=a.phase))
