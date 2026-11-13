from pathlib import Path
import csv,json,hashlib,statistics,math,shutil
from collections import defaultdict
R=Path(__file__).resolve().parent;D=R/'derived'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
model=read(D/'FINAL_CYCLE_MODEL.json');counts=read(D/'MATCHED_COUNTER_SUMMARY_FINAL.json');vjoin=read(D/'V_JOIN_COUNTERS_V2.json' if (D/'V_JOIN_COUNTERS_V2.json').exists() else D/'V_JOIN_COUNTERS.json')
cg0=read(D/'CG0_FIRST_PASS.json');assert not cg0['fails'] and len(cg0['PC'])==132
assert counts['all_Source_gates'] and counts['captures']==24
assert all(r['pass_gate'] for r in read(D/'V_JOIN_WORK_AUDIT.json'))
assert all(r['pass_gate'] for r in read(D/'DEP_NOP_NCU_AUDIT_V2.json'))
assert all(r['strict_gate'] for r in read(D/'MATRIX_STRICT_SLOPE_REVIEW.json')['summary'])
mix=read(D/'ALU_MIX_ACCEPTED_SLOPES.json');assert all(s['strict_gate'] for s in mix['summary'])
phases=[('CG0 T门控变换',0x2330,0x3e40),('CG0 QK后处理',0x3e50,0x5470),
 ('CG1 state读回与A发布',0x6fb0,0x79e0),('CG1 state decay发布',0x79f0,0x7a50),
 ('CG1 gate与V gather',0x7a60,0x7c50),('CG1 KS residual/VKS发布',0x7c60,0x8300),
 ('CG1 QS处理与发布',0x8310,0x8820),('CG1 NV/Delta处理与发布',0x8830,0x9080),('CG1 O暂存与返回',0x9090,0x94d0)]
native=[]
for role in ['CG0','CG1']:
 for row in csv.DictReader((D/f'FINAL_NATIVE_PC_{role}.csv').open(encoding='utf-8-sig')):
  row['role']=role;row['phase']=next((p for p,lo,hi in phases if lo<=int(row['pc'])<=hi),'control')
  row['work']=float(row['work']);row['rate']=float(row['rate']);row['cycles']=float(row['cycles']);native.append(row)
with (D/'STEP_RESOURCE_NATIVE_LEDGER.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(native[0]));w.writeheader();w.writerows(native)
phase=[]
for name,lo,hi in phases:
 rs=[r for r in native if r['phase']==name];cy=defaultdict(float)
 for r in rs:cy[r['resource']]+=r['cycles']
 phase.append(dict(phase=name,PC_begin=hex(lo),PC_end=hex(hi),native_PCs=len(rs),TMEM=cy.get('TMEM',0),shared_read=cy.get('LSU_Data|SMEM_Read',0),
   shared_write=cy.get('LSU_Data|SMEM_Write',0),ALU_PACK=cy.get('ALU_PACK',0),FP32=cy.get('FP',0),FP16=cy.get('HALF',0),
   SFU=cy.get('SFU',0),IMAD=cy.get('FMA_HEAVY',0),issue_only_floor=cy.get('ISSUE',0),
   interpretation='Resource service sums per native PC; not observed phase time, not serial additions to loop period. Unknown operation latency is not replaced by issue floor.'))
with (D/'PHASE_RESOURCE_BUDGET.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(phase[0]));w.writeheader();w.writerows(phase)
mma=[]
for name,shape,atoms,a,b in [('QK',(64,64,128),8,16384,16384),('KS',(128,64,128),8,0,16384),('QS',(128,64,128),8,0,16384),('NV',(128,64,64),4,0,8192),('O',(128,64,64),4,0,8192),('dS',(128,128,64),4,0,16384)]:
 M,N,K=shape;mma.append(dict(name=name,M=M,N=N,K=K,atoms=atoms,FLOP=2*M*N*K,TC_rate=8192,shared_A_B=a,shared_B_B=b,shared_WF=(a+b)/128))
assert sum(x['atoms'] for x in mma)==36 and sum(x['FLOP'] for x in mma)==9437184 and sum(x['shared_WF'] for x in mma)==768
source_checks=[
 dict(metric='UTCHMMA warp-inst',per_block=36,fixed_per_CTA=0,blocks=8192,CTAs=128,observed=294912,interpretation='Six GEMMs,8+8+8+4+4+4 atoms'),
 dict(metric='LDTM warp-inst',per_block=52,fixed_per_CTA=0,blocks=8192,CTAs=128,observed=425984,interpretation='Explicit shader TMEM loads only'),
 dict(metric='STTM warp-inst',per_block=64,fixed_per_CTA=16,blocks=8192,CTAs=128,observed=526336,interpretation='64/block plus initial-state publication'),
 dict(metric='Source shared WF',per_block=876,fixed_per_CTA=18,blocks=8192,CTAs=128,observed=7178496,interpretation='616read+260write/block, plus fixed read/write/atomic'),
 dict(metric='TC implicit shared WF',per_block=768,fixed_per_CTA=0,blocks=8192,CTAs=128,observed=6291456,interpretation='Dedicated TC counter, not LSU instruction count'),
]
baseline=next(r for r in vjoin['records'] if r['variant']=='B0' and r['trial']==0)
for x in source_checks:
 if x['metric'].split()[0] in ['UTCHMMA','LDTM','STTM']:
  prefix=x['metric'].split()[0];x['observed']=sum(v for k,v in baseline['op_counts'].items() if k.startswith(prefix))
 elif x['metric']=='Source shared WF':x['observed']=baseline['Source_WF']
 elif x['metric']=='TC implicit shared WF':x['observed']=baseline['TC']
for x in source_checks:x['predicted']=x['per_block']*x['blocks']+x['fixed_per_CTA']*x['CTAs'];assert x['predicted']==x['observed']
b0mean=next(z for z in vjoin['summary'] if z['variant']=='B0')
traffic=[
 dict(metric='DRAM read',total=b0mean['DRAM_read_B'],unit='B/full action',bytes_per_block=b0mean['DRAM_read_B']/8192,rate=None,conditional_cycles=None),
 dict(metric='DRAM write',total=b0mean['DRAM_write_B'],unit='B/full action',bytes_per_block=b0mean['DRAM_write_B']/8192,rate=None,conditional_cycles=None),
 dict(metric='DRAM read+write',total=b0mean['DRAM_read_B']+b0mean['DRAM_write_B'],unit='B/full action',bytes_per_block=(b0mean['DRAM_read_B']+b0mean['DRAM_write_B'])/8192,rate=32,conditional_cycles=(b0mean['DRAM_read_B']+b0mean['DRAM_write_B'])/8192/32),
 dict(metric='L2 total sectors',total=b0mean['L2_sectors'],unit='32B sector/full action',bytes_per_block=b0mean['L2_sectors']*32/8192,rate=64,conditional_cycles=b0mean['L2_sectors']*32/8192/64),
]
with (D/'NCU_FULL_ACTION_TRAFFIC_CROSSCHECK.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(traffic[0]));w.writeheader();w.writerows(traffic)
params=[
 ['TC dense FP16',8192,'FLOP/SM-cycle','Official-derived reference','Hopper dense4096 times Blackwell2x; no universal M64 half claim'],
 ['SMEM read service',1,'address-WF/SM-cycle','Counter-capacity + matched validation','Shader read and TC implicit operand read share read service; writes are separate'],
 ['SMEM write service',1,'address-WF/SM-cycle','Counter-capacity condition','Matched STSM write/TC read overlaps; not common read/write sum'],
 ['TMEM32 explicit service',512,'register-payload B/SM-cycle','User assumption','Not asserted hardware peak; separate completion bounds'],
 ['TMEM16dp256 service',256,'register-payload B/SM-cycle','Historical matched conditional stream','Read and QS write only; independent strict1% review retained'],
 ['TMEM16dp128 service',512,'register-payload B/SM-cycle','User assumption + matched differential','VKS/NV/Delta publication; not same as256-bit subtype'],
 ['PACK conversion',128,'scalar inputs/results/SM-cycle','User assumption + native-unit audit','Two scalar values per native PACK; equivalent64 PACK lane ops'],
 ['Generic integer ALU',64,'native lane results/SM-cycle','Official generic/conditional','ISETP/IADD3 matched hardware ALU counters; IMAD excluded'],
 ['FP32 math',128,'scalar results/SM-cycle','Official generic condition','FMUL2 counted two scalar results'],
 ['FP16 arithmetic/promote',64,'scalar results/SM-cycle','Declared instruction-specific condition','HADD2.F32 promotion one result; packed residual two, never infer from suffix alone'],
 ['EX2/LG2',16,'results/SM-cycle','Official generic condition','Sparse lane issue/service ambiguity retained; W11 one-warp limit separate'],
 ['HBM service proxy',32,'logical B/SM-cycle','User assumption','All-cold2312cy/block orT-in-L2 2056; not measured per-SM HBM peak'],
 ['L2 proxy',64,'request B/SM-cycle','User assumption','Global shared resource; not automatically exclusive per-SM physical peak'],
 ['Aggregate warp issue slots',4,'warp instructions/SM-cycle','Actual NCU device counter capacity','3 originalB0 reports: max_ipc_per_multiprocessor4, per_scheduler1. Fourwarps consume1cy; per-SMSP skew/priority remains unmodeled.'],
]
comp=[]
for x in model['comparison']:
 y=dict(x);y['model_observation_min']=x['conditional_model_min']/x['observed_cycle_slope'];y['model_observation_max']=x['conditional_model_max']/x['observed_cycle_slope'];
 y['old_scenario']=2296;y['scenario_shift_over_old_gap_min']=(x['conditional_model_min']-2296)/(x['observed_cycle_slope']-2296);y['scenario_shift_over_old_gap_max']=(x['conditional_model_max']-2296)/(x['observed_cycle_slope']-2296);comp.append(y)
for y in comp:
 y['sensitivity_signed_error_min']=(y['sensitivity_min']-y['observed_cycle_slope'])/y['observed_cycle_slope']
 y['sensitivity_signed_error_max']=(y['sensitivity_max']-y['observed_cycle_slope'])/y['observed_cycle_slope']
 assert y['sensitivity_min']>=.9*y['observed_cycle_slope'] and y['sensitivity_max']<=1.1*y['observed_cycle_slope']
references=[]
oldroot=Path(r'E:/Users/27194/Documents/Cutlass/op12_flashinfer_gdn_b200_analysis_20260825/k4_resource_cycle_audit_20261019_round2')
oldpaths=['tmem_design/review/DUAL_READ_FORMAL_INDEPENDENT_REVIEW.json','tmem_design/review/QS_DEPTH_FORMAL_INDEPENDENT_REVIEW.json','tmem_design/review/W128_DEPTH_FORMAL_INDEPENDENT_REVIEW.json','native_audit/INTEGER_NCU_REVIEW_V2/INTEGER_NCU_INDEPENDENT_REVIEW.json']
od=R/'inputs/historical_reviews';od.mkdir(exist_ok=True)
for p in oldpaths:
 fp=oldroot/p
 if fp.exists():
  dst=od/fp.name;shutil.copy2(fp,dst);references.append(dict(path=str(fp),sha256=sha(fp),snapshot=str(dst.relative_to(R))))
k1=Path(r'E:/Users/27194/Documents/Cutlass/op12_flashinfer_gdn_b200_analysis_20260825/k1_six_hour_verify_20261018/independent_review/MIX_NCU_AND_UNROLL_STATIC_REVIEW.json')
if k1.exists():
 shutil.copy2(k1,od/k1.name);references.append(dict(path=str(k1),sha256=sha(k1),snapshot=str((od/k1.name).relative_to(R))))
payload=dict(model=model,comparison=comp,native=native,phase=phase,mma=mma,parameters=params,
 source_checks=source_checks,traffic=traffic,matched_counts=counts,vjoin=vjoin,ALU_mix=mix,old_references=references,
 completion_original=read(D/'DEP_NOP_SLOPE_REVIEW_ORIGINAL.json'),completion_extension=read(D/'DEP_NOP_SLOPE_REVIEW_EXTENSION.json'),
 required_disclosures=['Candidate service/queue model, not a universal hardware peak or a measured K4 phase-sum model.',
 'The comparison is instrumented same-call CG1 return slope; local instrumentation bias is not fully bounded.',
 'All candidate inputs remain model_enabled=false and effective_model_value=null.'],model_enabled=False)
(D/'FINAL_DELIVERY_DATA.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(native_rows=len(native),phase_rows=len(phase),range=model['main_range'],checks=len(source_checks),CG0_PCs=len(cg0['PC']),old_refs=len(references)),ensure_ascii=False))
