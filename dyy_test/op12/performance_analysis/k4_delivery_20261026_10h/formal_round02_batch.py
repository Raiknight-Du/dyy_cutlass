"""R2 formal exactly 2 shapes x 3 processes; fail closed at Event CI gates."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

assert os.environ.get('K1_GPU_SESSION')
ROOT=Path(__file__).resolve().parent
out=ROOT/'round02_formal01';out.mkdir(exist_ok=False)
records=[]
for S in [8192,16384]:
    static=ROOT/'round02_compile01'/f'S{S}'
    assert not json.loads((static/'STATIC_GATE_REVIEW.json').read_text())['reasons']
    for trial in range(3):
        assert not (Path(os.environ['K1_GUARD_DIR'])/'INVALID_COMPETITION').exists()
        dst=out/f'S{S}'/f'trial{trial}'
        cmd=[sys.executable,str(ROOT/'run_round02.py'),'--phase','formal','--shape',str(S),'--trial',str(trial),'--out',str(dst),
             '--compile-approved',str(static/'COMPILE_AUDIT.json'),'--approved',str(static/'COMPILE_AUDIT.json')]
        rec=dict(S=S,trial=trial,command=cmd,start_epoch=time.time());records.append(rec)
        (out/'BATCH.json').write_text(json.dumps(records,indent=2))
        with (out/f'S{S}_trial{trial}.log').open('x') as f:
            q=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
        rec.update(returncode=q.returncode,end_epoch=time.time())
        (out/'BATCH.json').write_text(json.dumps(records,indent=2))
        if q.returncode:sys.exit(q.returncode)
        subprocess.run([sys.executable,str(ROOT/'analyze_shape_endpoints.py'),'--run',str(dst),'--shape',str(S)],stdout=subprocess.DEVNULL,check=True,timeout=120)
        qc=json.loads((dst/'EVENT_QC.json').read_text());rec['event']=qc
        rec['gate_pass']=all(qc[s+'_REF__CLOCK_RETURNS']['equivalent'] for s in ['CP_EVENT','K4_EVENT'])
        (out/'BATCH.json').write_text(json.dumps(records,indent=2))
        print('R2_FORMAL_PROCESS',S,trial,rec['gate_pass'],qc,flush=True)
        if not rec['gate_pass']:
            (out/'STOPPED_DYNAMIC_GATE.json').write_text(json.dumps(rec,indent=2));sys.exit(5)
(out/'COMPLETE.json').write_text(json.dumps(dict(shapes=[8192,16384],processes_per_shape=3,warmup=20,paired_blocks=30,model_enabled=False),indent=2))
