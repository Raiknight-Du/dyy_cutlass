#!/usr/bin/env python3
"""One GPU runner; isolated processes, complete logs, no state-changing setup."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ap=argparse.ArgumentParser()
ap.add_argument('--mode',choices=['pilot','formal','repeat'],required=True)
ap.add_argument('--pilot-trial',type=int,default=0)
args=ap.parse_args()
root=Path(__file__).resolve().parents[1]
op12=Path('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12')
repo=op12/'upstreams/flashinfer'
python=op12/'.venv/bin/python'
cases=[(8192,args.pilot_trial)] if args.mode=='pilot' else (
    [(s,t) for t in range(1,4) for s in [8192,16384,32768]] if args.mode=='formal'
    else [(s,4) for s in [8192,16384,32768]])
logdir=root/'logs';logdir.mkdir(exist_ok=True)
records=[]
for s,t in cases:
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    if apps:
        raise RuntimeError('GPU occupied; do not kill another process: '+apps)
    cmd=[str(python),str(root/'runner/measure_cp4_event.py'),'--repo',str(repo),
         '--output-root',str(root),'--s',str(s),'--trial',str(t)]
    if args.mode=='pilot':cmd+=['--pilot','--samples','20','--warmup','20','--native-repeats','3']
    path=logdir/f'{args.mode}_S{s}_trial{t}.log'
    print(json.dumps({'start':time.time(),'mode':args.mode,'S':s,'trial':t,'command':cmd}),flush=True)
    started=time.time()
    with path.open('x') as log:
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in p.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        code=p.wait()
    records.append({'S':s,'trial':t,'exit_code':code,'elapsed_s':time.time()-started,'command':cmd,'log':str(path)})
    record_name=f'pilot_trial{args.pilot_trial}_process_records.json' if args.mode=='pilot' else f'{args.mode}_process_records.json'
    (root/record_name).write_text(json.dumps(records,indent=2)+'\n')
    if code!=0: raise SystemExit(code)
print(json.dumps({'completed':args.mode,'processes':len(records)}),flush=True)
