"""Candidate native-service + native-scoreboard model, never phase fitting.

New independent latency inputs are bounded one-instruction feedback probes.
Complete K4 observations are loaded only AFTER all model cases are solved.
"""
from pathlib import Path
import csv,json,math,statistics,itertools
import native_pc_service_model as m
import encoded_async_engine as ae
from encoded_async_engine_capacity import CapacityEngine
R=Path(__file__).resolve().parent;D=R/'derived';D.mkdir(exist_ok=True)
old=m.native_ops;oldengine=CapacityEngine;engines=[]
class LoggedEngine(oldengine):
 def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.waitlog=[];engines.append(self)
 def spawn(self,name,gen):
  def trace():
   stores={'TMEM':0.,'SMEM':0.}
   for z in gen:
    if z[0]=='op':
     o=z[1];opcode=o['opcode']
     if opcode.startswith(('MEMBAR','FENCE.VIEW.ASYNC')):
      keys=['TMEM'] if opcode.endswith('.T') else ['SMEM']
      until=max(stores[k] for k in keys)
      if until>self.t:
       t=self.t;yield 'delay',until-self.t
       self.waitlog.append(dict(actor=name,kind='store_publication_edge',item=f'{o["chunk"]}:{o["pc_hex"]}:{opcode}',start=t,end=self.t))
    t=self.t;yield z
    if z[0]=='op' and z[1]['opcode'].startswith(('STTM','STSM','STS')):
     o=z[1];label=f'{o["chunk"]}:{o["pc_hex"]}'
     row=next(a for a in reversed(self.log) if a[0]==name and a[1]==label)
     key='TMEM' if o['opcode'].startswith('STTM') else 'SMEM';stores[key]=max(stores[key],row[-1])
    if z[0] in ['wait','delay'] and self.t>t:self.waitlog.append(dict(actor=name,kind=z[0],item=str(z[1]),start=t,end=self.t))
  return super().spawn(name,trace())
ae.EncodedEngine=LoggedEngine
case='separate_FMA_logical';read_rate=256.
def mapped(lo,hi,hybrid):
 ops=old(lo,hi,hybrid)
 for o in ops:
  if o['opcode'].startswith('IMAD') and o['resource']=='ALU_PACK':
   o['resource']='FMA_HEAVY';o['reason']='IMAD excluded from ALU per NVIDIA pipeline definition; generic64 results/cy condition.'
  if case=='shared_FP16_FP32' and o['resource']=='HALF':o['resource']='FP'
  if o['opcode'].startswith('LDTM.16dp256bit'):
   o['rate']=read_rate;o['cycles']=math.ceil(o['work']/read_rate)
  o['model_enabled']=False;o['effective_model_value']=None
 return ops
m.native_ops=mapped
rev=[json.loads((D/'DEP_NOP_SLOPE_REVIEW_ORIGINAL.json').read_text()),json.loads((D/'DEP_NOP_SLOPE_REVIEW_EXTENSION.json').read_text())]
bb=[rev[0]['bounds'][0],rev[1]['bounds'][0]]
assert all(x['strict_gate'] for x in bb)
bounds={key:[math.floor(b['conditional_completion_lower']+1e-8),math.ceil(b['conditional_completion_upper'])] for key,b in zip(['32dp','16dp'],bb)}
loose_bounds=dict(bounds)
endpoint=json.loads((D/'SINGLE_LDTM_COMPLETION_DERIVATION.json').read_text())
assert all(x['strict_gate'] for x in endpoint['records'])
bounds={key:[v['conditional_completion'],v['conditional_completion']] for key,v in zip(['32dp','16dp'],endpoint['records'])}
scenarios=[]
def do(label,hy=True,bank=True,lat=(0,0),capture=False):
 lag={'32dp':max(0,lat[0]-4096/512),'16dp':max(0,lat[1]-4096/read_rate)} if lat!=(0,0) else {'32dp':0,'16dp':0}
 z,g0,g1=m.simulate(hy,N=80,bank_TC=bank,qkrate=8192,asynchronous=True,full_issuer=True,tmem_lag=lag,instruction_issue=True)
 z['resource_work']['ISSUE_SLOTS']=sum(a[-1]-a[-2] for a in engines[-1].log if a[2]=='ISSUE_SLOTS')/80
 z.update(label=label,pipe_case=case,TMEM16_read_rate=read_rate,conditional_single_warp_completion=list(lat),no_target_time_as_input=True)
 if capture:
  e=engines[-1]
  with (D/'REPRESENTATIVE_MODEL_SERVICE.csv').open('w',encoding='utf-8-sig',newline='') as f:
   w=csv.writer(f);w.writerow(['actor','node','resource','request_cycle','service_start','service_end']);w.writerows(e.log)
  (D/'REPRESENTATIVE_MODEL_WAITS.json').write_text(json.dumps(e.waitlog,indent=2),encoding='utf-8')
  for name,rows in [('CG0',g0),('CG1',g1)]:
   plain=[{k:v for k,v in r.items() if k!='control'} for r in rows]
   for r,o in zip(plain,rows):r.update(o['control'])
   with (D/f'FINAL_NATIVE_PC_{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(plain[0]));w.writeheader();w.writerows(plain)
 scenarios.append(z);return z
do('User512_native_service_no_TC_operand_arbitration',False,False)
do('Subtype256_native_service_no_TC_operand_arbitration',True,False)
do('Subtype256_plus_TC_shared_read_no_extra_read_completion')
for case in ['separate_FMA_logical','shared_FP16_FP32']:
 for lat in itertools.product(*[sorted(set(v)) for v in bounds.values()]):
  do('Conditional_native_DAG',lat=lat,capture=case=='separate_FMA_logical' and lat==(bounds['32dp'][1],bounds['16dp'][1]))
# Alternate historical four-reader stream: sensitivity, not replacement
# of the primary256 service assumption or an asserted universal capacity.
read_rate=65536/266
for case in ['separate_FMA_logical','shared_FP16_FP32']:
 for lat in sorted(set([tuple(v[0] for v in bounds.values()),tuple(v[1] for v in bounds.values())])):do('Historical_four_reader_service_sensitivity',lat=lat)
read_rate=256.
conditional=[z for z in scenarios if z['label']=='Conditional_native_DAG']
low=min(z['mean_cycle'] for z in conditional);high=max(z['mean_cycle'] for z in conditional)
windows={8192:(4,12),16384:(8,24),32768:(8,56)}
def window_slope(z,a,b):
 rr={i:t for i,t in z['ret']};xx=list(range(a,b));yy=[rr[i] for i in xx];xm,ym=statistics.mean(xx),statistics.mean(yy)
 return sum((x-xm)*(y-ym) for x,y in zip(xx,yy))/sum((x-xm)**2 for x in xx)
model_windows={S:[window_slope(z,*w) for z in conditional] for S,w in windows.items()}
sens_windows={S:[window_slope(z,*w) for z in scenarios if z['label']=='Historical_four_reader_service_sensitivity'] for S,w in windows.items()}
# Observations are used for validation only, after parameters and schedule
# have been solved. Never use a target residual as a node input.
obs=json.loads((D/'LOOP_SLOPE_REFERENCE.json').read_text());comparison=[]
for o in obs:
 reference=o['mean_cycle_slope']
 lo,hi=min(model_windows[o['S']]),max(model_windows[o['S']]);a,b=windows[o['S']]
 comparison.append(dict(S=o['S'],C=o['C'],L=o['L'],calls=o['calls'],observed_cycle_slope=reference,
   conditional_model_min=lo,conditional_model_max=hi,signed_error_min=(lo-reference)/reference,signed_error_max=(hi-reference)/reference,
   model_window=f'return{a}..{b-1}',sensitivity_min=min(sens_windows[o['S']]),sensitivity_max=max(sens_windows[o['S']]),
   all_cases_within10=lo>=.9*reference and hi<=1.1*reference,
   comparison_scope=o['comparison'],model_enabled=False))
summary={k:v for k,v in scenarios[2].items() if k not in ['ret','assumptions']}
result=dict(scope='One full64-token steady block/CTA, originalFP16 K4, completeCP original trace context, CG1 return cadence comparison; no INIT/tail or whole-kernel90% claim.',
  completion_bounds=bounds,previous_loose_completion_bounds=loose_bounds,single_instruction_endpoint=endpoint,scenarios=scenarios,comparison=comparison,known_service_ledger=summary,
  main_range=[low,high],model_enabled=False,effective_model_value=None,
  unresolved=['Sparse EX2 exact lane-service granularity; some predicate/select/uniform native capacity and completion endpoints.',
   'Actual hardware queue priority, per-warp issue skew and short-MMA notification latency; FIFO/group model is conditional.',
   'TMA/gate/O inputs modeled as steady-ready; their full service constraints are reported separately, not charged zero. Late arrival contribution remains unknown.',
   'Original local/spill traffic is retained in work inventory but not assigned a fabricated latency.',
   'Return-marker trace local bias is not fully bounded; these slopes are instrumented observations, not original phase truth.'])
(D/'FINAL_CYCLE_MODEL.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
with (D/'FINAL_SHAPE_COMPARISON.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(comparison[0]));w.writeheader();w.writerows(comparison)
print(json.dumps(dict(bounds=bounds,models=[dict(label=z['label'],case=z['pipe_case'],lat=z['conditional_single_warp_completion'],period=z['mean_cycle']) for z in scenarios],comparison=comparison),ensure_ascii=False))
