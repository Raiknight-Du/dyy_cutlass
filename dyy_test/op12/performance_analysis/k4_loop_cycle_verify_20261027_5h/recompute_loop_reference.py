"""Fit return-cycle slope within each invocation, never splice warp means."""
from pathlib import Path
import json,csv,statistics,math,shutil
from collections import defaultdict
R=Path(__file__).resolve().parent;D=R/'derived'
A=Path(r'E:/Work/Lab/HUAWEI benchmark/op12/K4/分析交付_20261026_10h/01_native_audit/derived')
B=Path(r'E:/Work/Lab/HUAWEI benchmark/op12/K4/理论重算_20261027/inputs')
if (R/'inputs/target_small_shapes').exists():B=R/'inputs/target_small_shapes'
def fit(x,y):
 mx,my=statistics.mean(x),statistics.mean(y)
 b=sum((a-mx)*(c-my) for a,c in zip(x,y))/sum((a-mx)**2 for a in x);a=my-b*mx
 err=sum((v-a-b*k)**2 for k,v in zip(x,y));var=sum((v-my)**2 for v in y)
 return b,(1-err/var)
rows=[]
F=A.parents[1]/'05_frozen_inputs/trace_returns'
if (R/'inputs/target_trace_returns').exists():F=R/'inputs/target_trace_returns'
for trial in range(3):
 p=F/f'trial{trial}/CLOCK_RAW.json';raw=json.loads(p.read_text(encoding='utf-8-sig'));ss=defaultdict(list)
 for r in raw:
  if r['variant']=='CLOCK_RETURNS':ss[r['invocation']].append(r)
 assert len(ss)==60
 for inv,rr in ss.items():
  assert len({(r['CTA'],r['SMID'],r['cubin'],r['timer']) for r in rr})==1
  cr=sorted([x for x in rr if x['stage']=='CG1_CHUNK_RETURN'],key=lambda x:int(x['iteration']))
  assert len(cr)==64 and [x['iteration'] for x in cr]==list(range(64))
  cy=[int(x['begin']) for x in cr];cy=[x-cy[0] for x in cy]
  b,r2=fit(list(range(8,56)),cy[8:56])
  rows.append(dict(S=32768,C=4096,L=64,invocation=f't{trial}_{inv}',window='return8..55',cycle_slope=b,R2=r2,
     endpoint_mean=statistics.mean(cy[i]-cy[i-1] for i in range(8,56)),source=str(p)))
for S in [8192,16384]:
 L=S//512;lo,hi=L//4,3*L//4
 for tr in range(3):
  p=B/f'R2_FORMAL_S{S}_T{tr}_RECOMPUTED.json';x=json.loads(p.read_text(encoding='utf-8'))
  for idx,f in enumerate(v for v in x['features'] if v['series']=='K4_EVENT'):
   it=f['all_return_intervals'];assert len(it)==L
   c=[];acc=0
   for z in it:acc+=z;c.append(acc)
   b,r2=fit(list(range(lo,hi)),c[lo:hi])
   rows.append(dict(S=S,C=S//8,L=L,invocation=f't{tr}_K4_EVENT_{idx}',window=f'return{lo}..{hi-1}',cycle_slope=b,R2=r2,
    endpoint_mean=statistics.mean(it[lo:hi]),source=str(p)))
with (D/'SAME_CALL_CYCLE_SLOPES.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,rows[0]);w.writeheader();w.writerows(rows)
sm=[]
for S in [8192,16384,32768]:
 rr=[x for x in rows if x['S']==S];v=[x['cycle_slope'] for x in rr]
 sm.append(dict(S=S,C=S//8,L=S//512,calls=len(v),mean_cycle_slope=statistics.mean(v),median_cycle_slope=statistics.median(v),min_R2=min(x['R2'] for x in rr),
  mean_return_interval=statistics.mean(x['endpoint_mean'] for x in rr),comparison='Instrumented same-invocation CG1 return regression; local bias not fully bounded; not original-kernel phase ground truth.'))
(D/'LOOP_SLOPE_REFERENCE.json').write_text(json.dumps(sm,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(sm,ensure_ascii=False))
