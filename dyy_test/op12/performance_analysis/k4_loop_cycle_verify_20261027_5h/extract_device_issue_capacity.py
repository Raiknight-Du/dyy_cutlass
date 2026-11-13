"""Read existing NCU device metadata, no GPU invocation."""
from pathlib import Path
import csv,json,hashlib
R=Path(__file__).resolve().parent;D=R/'derived';out=[]
keys=['device__attribute_max_ipc_per_multiprocessor','device__attribute_max_ipc_per_scheduler','device__attribute_max_warps_per_scheduler','device__attribute_max_warps_per_multiprocessor']
for i in range(3):
 p=D/f'B0_t{i}_raw.csv'
 z=next(v for v in csv.DictReader(p.open(encoding='utf-8-sig')) if v['ID']=='0')
 values={k:int(z[k]) for k in keys};assert list(values.values())==[4,1,16,64]
 out.append(dict(file=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),values=values))
d=dict(source='NCU2026.2.1, actual originalK4 B200 captures, existing device metadata',records=out,
 aggregate_warp_issue_slots_per_SM_cycle=4,warp_issue_slots_per_scheduler_cycle=1,schedulers_per_SM=4,
 interpretation='Counter/device capacity; not observed sustained throughput. Group/FIFO model uses aggregate four slots; per-SMSP warp skew remains unknown.',model_enabled=False,effective_model_value=None)
(D/'DEVICE_ISSUE_CAPACITY.json').write_text(json.dumps(d,indent=2),encoding='utf-8');print(json.dumps(d))
