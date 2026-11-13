from pathlib import Path
import json
from native_pc_service_model import simulate
R=Path(__file__).resolve().parent;out=[]
for bank in [False,True]:
 for sl,tl in [(0,0),(16,0),(0,32),(16,32),(32,64)]:
  x,_,_=simulate(True,N=128,bank_TC=bank,qkrate=8192,asynchronous=True,smem_lag=sl,tmem_lag=tl)
  out.append({k:v for k,v in x.items() if k not in ['ret','assumptions']})
(R/'derived/ENCODED_ASYNC_SENSITIVITY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out))
