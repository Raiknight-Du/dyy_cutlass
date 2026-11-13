"""Three pre-authorized S actions; application replay with complete CP history."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--compile-approved',type=Path,required=True);p.add_argument('--metrics',type=Path,required=True);a=p.parse_args()
assert os.environ.get('K1_GPU_SESSION')
metrics=json.loads(a.metrics.read_text());assert metrics['status']=='READY'
a.out.mkdir(parents=True,exist_ok=False)
ROOT=Path(__file__).resolve().parent
ncu='/opt/nvidia/nsight-compute/2026.2.1/ncu'
records=[]
for case in ['REF','FRAME0','FRAME7']:
    assert not (Path(os.environ['K1_GUARD_DIR'])/'INVALID_COMPETITION').exists()
    cmd=[ncu,'--replay-mode','application','--cache-control','none','--clock-control','none',
         '--nvtx','--nvtx-include','K4_READY_GATE_R1/','--kernel-name-base','function',
         '--kernel-name','regex:.*(CPDeltaRulePrefillTcgen05Sm100|K4READY_GATE).*','--launch-count','1',
         '--export',str(a.out/case),'--section','LaunchStats','--section','SourceCounters','--section','InstructionStats',
         '--metrics',','.join(metrics['metrics']),sys.executable,str(ROOT/'run_round01.py'),'--phase','profile','--profile',case,
         '--layout','S','--compile-approved',str(a.compile_approved),'--out',str(a.out/(case+'_replays'))]
    rec=dict(case=case,command=cmd,start_epoch=time.time());records.append(rec)
    (a.out/'INVOCATIONS.json').write_text(json.dumps(records,indent=2))
    with (a.out/(case+'.log')).open('x') as f:
        q=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
    rec.update(returncode=q.returncode,end_epoch=time.time())
    (a.out/'INVOCATIONS.json').write_text(json.dumps(records,indent=2))
    if q.returncode:sys.exit(q.returncode)
    assert (a.out/(case+'.ncu-rep')).exists(),'No selected target action'
    for page in ['raw','source']:
        exp=[ncu,'--import',str(a.out/(case+'.ncu-rep')),'--page',page,'--csv','--print-units','base']
        if page=='source':exp += ['--print-source','sass']
        with (a.out/(case+'_'+page+'.csv')).open('x') as f:
            subprocess.run(exp,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
    print('NCU_ACTION_COMPLETE',case,flush=True)
(a.out/'COMPLETE.json').write_text(json.dumps(dict(actions=3,scope='S32768 original CP history',timing_used=False,model_enabled=False),indent=2))
