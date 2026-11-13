from pathlib import Path
import native_pc_service_model as m
import json
R=Path(__file__).resolve().parent
original=m.native_ops
def run(alias):
 def adapted(lo,hi,hybrid):
  ops=original(lo,hi,hybrid)
  for o in ops:
   if alias=='split_CVT_INT' and o['resource']=='ALU_PACK' and not o['opcode'].startswith('F2FP'):o['resource']='INTEGER'
  return ops
 m.native_ops=adapted
 out=[]
 for lag in [{'32dp':17,'16dp':57},{'32dp':41.375,'16dp':81.375}]:
  z,_,_=m.simulate(True,bank_TC=True,qkrate=8192,asynchronous=True,full_issuer=True,tmem_lag=lag)
  out.append({k:v for k,v in z.items() if k not in ['ret','assumptions']})
 return dict(alias=alias,scenarios=out)
out=[run(a) for a in ['shared_CVT_INT','split_CVT_INT']]
(R/'derived/RESOURCE_ALIAS_SENSITIVITY.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps([dict(alias=x['alias'],cycles=[q['mean_cycle'] for q in x['scenarios']]) for x in out]))
