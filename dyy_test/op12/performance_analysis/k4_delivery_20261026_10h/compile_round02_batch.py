"""R2 static admission using actual metadata compilation, never launches kernels."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

assert os.environ.get('K1_GPU_SESSION')
ROOT=Path(__file__).resolve().parent
out=ROOT/'round02_compile01';out.mkdir(exist_ok=False)
projection_ops=set(['LDL','STL','UTCHMMA','UTMALDG','UTMASTG','LDTM','STTM','HMMA','LDG','F2FP','BAR','UTCBAR','SYNCS','FENCE','MEMBAR','FFMA','FMUL','FADD','LDSM','STSM','MUFU','PRMT'])
records=[]
for S in [8192,16384]:
    dst=out/f'S{S}'
    cmd=[sys.executable,str(ROOT/'compile_ready_only.py'),'--out',str(dst),'--layout','S','--shape',str(S),'--metadata-device','cuda','--compile-cases','B0,CLOCK']
    q=subprocess.run(cmd,check=True,timeout=300)
    audit=json.loads((dst/'COMPILE_AUDIT.json').read_text());b,p=audit['B0'],audit['CLOCK'];reasons=[]
    for kind in ['REG','STACK','SHARED']:
        if re.findall(kind+r':(\d+)',b['resource'])!=re.findall(kind+r':(\d+)',p['resource']):reasons.append('CHANGED_'+kind)
    for key in ['LDL','STL','LDG','LDTM','STTM','UTCHMMA','UTMALDG','UTMASTG','BAR','SYNCS','FENCE','MEMBAR','UTCBAR']:
        if b['counts'][key]!=p['counts'][key]:reasons.append('CHANGED_'+key)
    project=lambda d:[x['op'] for x in d['instructions'] if x['op'].split('.')[0] in projection_ops]
    if project(b)!=project(p):reasons.append('NATIVE_PROJECTION_ORDER_CHANGED')
    report=dict(S=S,C=S//8,L=S//8//64,P=8,grid=128,block=384,source='Existing PTR_CLOCK_RETURNS, unchanged',
        status='STATIC_PASSED_CONDITIONAL_SCREEN_ADMITTED' if not reasons else 'SHAPE_STOPPED_STATIC_FAILURE',
        reasons=reasons,hashes={k:v['cubin_sha256'] for k,v in audit.items()},resources={k:v['resource'] for k,v in audit.items()},
        counts={k:v['counts'] for k,v in audit.items()},native_projection_ops=len(project(b)),
        actual_JIT_metadata_only=True,gpu_kernel_invoked=False,local_absolute_bias_bound_cycles=None,model_enabled=False)
    (dst/'STATIC_GATE_REVIEW.json').write_text(json.dumps(report,indent=2));records.append(report)
    print('R2_STATIC',S,report['status'],report['hashes'],reasons,flush=True)
(out/'BATCH_STATIC_GATES.json').write_text(json.dumps(records,indent=2))
