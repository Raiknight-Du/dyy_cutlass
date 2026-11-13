"""Three independent formal processes; invoke only after Lead formal admission."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--layout',choices=['S','V'],required=True);p.add_argument('--compile-approved',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
assert os.environ.get('K1_GPU_SESSION')
a.out.mkdir(parents=True,exist_ok=False)
ROOT=Path(__file__).resolve().parent
records=[]
for trial in range(3):
    assert not (Path(os.environ['K1_GUARD_DIR'])/'INVALID_COMPETITION').exists()
    out=a.out/f'trial{trial}'
    cmd=[sys.executable,str(ROOT/'run_round01.py'),'--phase','formal','--layout',a.layout,'--trial',str(trial),
         '--out',str(out),'--compile-approved',str(a.compile_approved),'--approved',str(a.compile_approved)]
    rec=dict(trial=trial,command=cmd,start_epoch=time.time());records.append(rec)
    (a.out/'BATCH.json').write_text(json.dumps(records,indent=2))
    with (a.out/f'trial{trial}.log').open('x') as f:
        done=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
    rec.update(returncode=done.returncode,end_epoch=time.time())
    (a.out/'BATCH.json').write_text(json.dumps(records,indent=2))
    if done.returncode:sys.exit(done.returncode)
    event=json.loads((out/'EVENT_QC.json').read_text())
    stage=json.loads((out/'PAIRED_STAGE_QC.json').read_text())
    reasons=[]
    for scope in ['CP_EVENT','K4_EVENT']:
        if not event[scope+'_REF__FRAME0']['equivalent']:reasons.append(scope+'_FRAME0_EVENT_EQUIVALENCE_FAILED')
        if a.layout=='S' and not stage[scope+'_PBASE__FRAME0']['relative_stability_pass']:reasons.append(scope+'_PBASE_FRAME0_INIT_RELATIVE_STABILITY_FAILED')
    rec.update(dynamic_gate_reasons=reasons)
    (a.out/'BATCH.json').write_text(json.dumps(records,indent=2))
    subprocess.run([sys.executable,str(ROOT/'analyze_round01.py'),'--run',str(out)],stdout=subprocess.DEVNULL,check=True,timeout=120)
    print('FORMAL_PROCESS_COMPLETE',trial,'gate_reasons',reasons,flush=True)
    if reasons:
        (a.out/'STOPPED_DYNAMIC_GATE.json').write_text(json.dumps(dict(trial=trial,reasons=reasons,model_enabled=False),indent=2))
        sys.exit(5)
(a.out/'COMPLETE.json').write_text(json.dumps(dict(independent_processes=3,warmup=20,paired_blocks=30,model_enabled=False),indent=2))
