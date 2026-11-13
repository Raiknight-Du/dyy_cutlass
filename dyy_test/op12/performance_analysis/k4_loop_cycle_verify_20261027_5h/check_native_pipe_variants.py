from pathlib import Path
import json
import native_pc_service_model as m
R=Path(__file__).resolve().parent;orig=m.native_ops
def work(case):
 def adapted(lo,hi,hy):
  ops=orig(lo,hi,hy)
  for o in ops:
   if o['opcode'].startswith('IMAD') and o['resource']=='ALU_PACK':
    o['resource']='FMA_HEAVY';o['reason']+='; IMAD excluded from ALU per official pipe definition.'
   if case=='share_FP_HALF' and o['resource']=='HALF':o['resource']='FP'
  return ops
 m.native_ops=adapted
 ans=[]
 for lag in [{'32dp':17,'16dp':57},{'32dp':41.375,'16dp':81.375}]:
  z,_,_=m.simulate(True,bank_TC=True,qkrate=8192,asynchronous=True,full_issuer=True,tmem_lag=lag)
  ans.append({k:v for k,v in z.items() if k not in ['ret','assumptions']})
 return dict(case=case,scenarios=ans)
x=[work(s) for s in ['map_IMAD_FMA_HEAVY','share_FP_HALF']]
(R/'derived/NATIVE_PIPE_SENSITIVITY.json').write_text(json.dumps(x,indent=2),encoding='utf-8')
print(json.dumps([dict(case=y['case'],cycles=[z['mean_cycle'] for z in y['scenarios']]) for y in x]))
