from pathlib import Path
import re,json,hashlib
P=Path(__file__).parent
keep=('LDTM','STTM','LDSM','STSM','UTCHMMA','UTMALDG','F2FP','HADD2','FMUL','FFMA','FADD','MUFU','SHFL','FENCE','LDG','LDS','STS')
results=[]
for folder in sorted(P.glob('s*_r?')):
    records=[]
    for bd in sorted((folder/'binary').glob('*_b0')):
        text=(bd/'audit.sass').read_text();seq=[];clocks=[]
        for line in text.splitlines():
            m=re.match(r'\s*/\*([0-9a-f]+)\*/\s+(?:@!?[A-Z0-9]+\s+)?([A-Za-z0-9_.]+)(.*?);',line)
            if not m:continue
            pc,op,body=m.groups()
            if op.startswith(keep):seq.append(op)
            if 'SR_CLOCKLO' in line or 'SR_GLOBALTIMERLO' in line:clocks.append({'PC':pc,'sass':op+body.strip()})
        records.append({'mode':bd.name.split('_')[0],'sequence':seq,'clock_PCs':clocks,'sass_sha256':hashlib.sha256(text.encode()).hexdigest()})
    base=list(next(r for r in records if r['mode']=='B0')['sequence'])
    for r in records:r['core_native_order_equals_B0']=r['sequence']==base;r['core_native_count']=len(r.pop('sequence'))
    results.append({'run':folder.name,'records':records})
(P/'NATIVE_ORDER_AUDIT.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
