from pathlib import Path
import subprocess,sys,os,time,json,datetime
R=Path(__file__).resolve().parent;O=R/'dep_nop_extension_formal';O.mkdir(exist_ok=False)
def gpu():return dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,clocks.sm,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip(),apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True).strip())
def log(v):
 with (O/'LEDGER.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
 print(json.dumps(v),flush=True)
for i in range(3):
 g=gpu();assert not g['apps'] and int(g['gpu'].split(',')[0])<=10,g;log(dict(event='admission',quiet=i+1,state=g))
 if i<2:time.sleep(30)
for t in range(3):
 tag=f'k1_w1_t{t}_m-1';g=gpu();assert not g['apps'],g
 cmd=[sys.executable,str(R/'run_tmem_nop.py'),'--kind','1','--warps','1','--out',str(O/tag),'--trial',str(t),'--points','2048,4096,8192,16384','--samples','30']
 log(dict(event='start',command=cmd,state=g))
 with (O/(tag+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,K4_5H_ADMITTED='1'),timeout=900)
 log(dict(event='finish',tag=tag,returncode=p.returncode,state=gpu()));assert p.returncode==0
log(dict(event='COMPLETE'))
