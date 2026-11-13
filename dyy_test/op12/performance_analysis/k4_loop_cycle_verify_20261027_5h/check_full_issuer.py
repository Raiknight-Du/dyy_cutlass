from pathlib import Path
import json
from native_pc_service_model import simulate
R=Path(__file__).resolve().parent;out=[]
for lag in [dict(LDTM=0),{'32dp':29,'16dp':69},{'32dp':45,'16dp':85}]:
 x,_,_=simulate(True,bank_TC=True,qkrate=8192,asynchronous=True,tmem_lag=lag,full_issuer=True)
 out.append({k:v for k,v in x.items() if k not in ['ret','assumptions']})
(R/'derived/FULL_ISSUER_SENSITIVITY.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out))
