"""CPU-only final reconciliation. Does not use target residual as input."""
from pathlib import Path
import json,csv,collections,hashlib
R=Path(__file__).resolve().parent;D=R/'derived'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
model=read(D/'FINAL_CYCLE_MODEL.json');p=read(D/'FINAL_DELIVERY_DATA.json')
assert all(x['all_cases_within10'] for x in model['comparison'])
assert all(.9*x['observed_cycle_slope']<=x['sensitivity_min']<=x['sensitivity_max']<=1.1*x['observed_cycle_slope'] for x in model['comparison'])
assert len(p['native'])==1384 and all(x['predicted']==x['observed'] for x in p['source_checks'])
logs=list(csv.DictReader((D/'REPRESENTATIVE_MODEL_SERVICE.csv').open(encoding='utf-8-sig')))
resources=collections.defaultdict(list)
for i,z in enumerate(logs):
 start,end=float(z['service_start']),float(z['service_end'])
 assert end>=start and start>=float(z['request_cycle'])-1e-7
 for res in z['resource'].split('|'):resources[res].append((start,end,i))
checks=[]
for res,v in resources.items():
 v=sorted(v);overlaps=[(a,b) for a,b in zip(v,v[1:]) if b[0]<a[1]-1e-7]
 assert not overlaps,(res,overlaps[:2])
 checks.append(dict(resource=res,segments=len(v),service_cycles=sum(b-a for a,b,_ in v),overlaps=0))
assert 'ISSUE_SLOTS' in resources
for a,b,_ in resources['ISSUE_SLOTS']:assert abs((b-a)*4-round((b-a)*4))<1e-7
endpoint=read(D/'SINGLE_LDTM_COMPLETION_DERIVATION.json')
assert [x['conditional_completion'] for x in endpoint['records']]==[26,74]
for f in ['ALU_MIX_ACCEPTED_SLOPES.json','MATRIX_STRICT_SLOPE_REVIEW.json']:
 assert all(x['strict_gate'] for x in read(D/f)['summary'])
assert all(x['pass_gate'] for x in read(D/'DEP_NOP_NCU_AUDIT_V2.json'))
assert all(x['pass_gate'] for x in read(D/'V_JOIN_WORK_AUDIT.json'))
out=dict(status='PASS',scope='Conditional grouped/FIFO schedule; aggregate issue-capacity feasibility, not a per-SMSP simulator or unbiased original-phase proof.',
 comparisons=model['comparison'],resource_overlap_checks=checks,native_PCs=1384,counter_checks=p['source_checks'],
 measured_fullphase_time_used_as_node=False,target_residual_used_as_input=False,model_enabled=False,effective_model_value=None)
(D/'FINAL_CPU_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status='PASS',resources=len(checks),native_PCs=1384,shape_checks=3),ensure_ascii=False))
