"""Pre-approved two-shape small screen, never starts formal or NCU."""
import json
import os
from pathlib import Path
import subprocess
import sys

assert os.environ.get('K1_GPU_SESSION')
ROOT=Path(__file__).resolve().parent
out=ROOT/'round02_screen01';out.mkdir(exist_ok=False)
records=[]
for S in [8192,16384]:
    static=ROOT/'round02_compile01'/f'S{S}'
    gate=json.loads((static/'STATIC_GATE_REVIEW.json').read_text())
    if gate['reasons']:
        records.append(dict(S=S,status='STOPPED_STATIC_GATE',reasons=gate['reasons']));continue
    dst=out/f'S{S}'
    cmd=[sys.executable,str(ROOT/'run_round02.py'),'--phase','screen','--shape',str(S),'--out',str(dst),'--compile-approved',str(static/'COMPILE_AUDIT.json')]
    with (out/f'S{S}.log').open('x') as f:
        q=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
    rec=dict(S=S,command=cmd,returncode=q.returncode);records.append(rec)
    (out/'BATCH.json').write_text(json.dumps(records,indent=2))
    if q.returncode:sys.exit(q.returncode)
    subprocess.run([sys.executable,str(ROOT/'analyze_shape_endpoints.py'),'--run',str(dst),'--shape',str(S)],stdout=subprocess.DEVNULL,check=True,timeout=120)
    qc=json.loads((dst/'EVENT_QC.json').read_text())
    rec.update(event=qc,screen_gate_pass=all(qc[s+'_REF__CLOCK_RETURNS']['equivalent'] for s in ['CP_EVENT','K4_EVENT']))
    (out/'BATCH.json').write_text(json.dumps(records,indent=2))
    print('R2_SCREEN_COMPLETE',S,rec['screen_gate_pass'],qc,flush=True)
(out/'COMPLETE.json').write_text(json.dumps(dict(screen_only=True,shapes=[8192,16384],warmup=3,paired_blocks=3,model_enabled=False),indent=2))
