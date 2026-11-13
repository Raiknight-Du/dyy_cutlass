from pathlib import Path
import csv,json,re,sys,statistics,hashlib
R=Path(__file__).resolve().parent;B=R/'ncu_v2';F=json.loads((R/'FAMILIES.json').read_text(encoding='utf-8'))
def num(v):
    try:return float(v.replace(',',''))
    except:return None
def raw(p):
    a=list(csv.reader(p.open(encoding='utf-8-sig',newline='')));i=next(i for i,r in enumerate(a) if r and r[0]=='ID' and 'Kernel Name' in r)
    return {k:num(v) for k,v in zip(a[i],a[i+2])}
def source(p):
    a=list(csv.reader(p.open(encoding='utf-8-sig',newline='')));i=next(i for i,r in enumerate(a) if 'Address' in r and 'Source' in r)
    return [dict(zip(a[i],r)) for r in a[i+1:] if r and r[0].startswith('0x')]
metrics={'read':'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum','write':'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
  'misc':'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_misc.sum','TC':'l1tex__data_pipe_tc_wavefronts.sum',
  'read_conflict':'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum','write_conflict':'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
  'arbitration':'l1tex__data_bank_conflicts_type_arbitration.sum','total':'l1tex__data_pipe_lsu_wavefronts.sum'}
rs=[];gates=[]
for f in F:
 for N in [128,512]:
  for mode in [1,3]:
   folder=R/('ncu_gate_v4' if f['id']==1 else 'ncu_scalar_v3' if f['id']==0 else 'ncu_v2')
   tag=f"kind{f['id']}_N{N}_mode{mode}";p=folder/(tag+'_raw.csv');sp=folder/(tag+'_source.csv')
   if not p.exists() or not sp.exists():continue
   x=raw(p);src=source(sp)
   target=[s for s in src if re.match(r'\s*(?:@!?\w+\s+)?(?:'+'|'.join(re.escape(op) for op in set(f['opcodes']))+r')\s',s['Source']) and (num(s['Instructions Executed']) or 0)>128*4]
   wf=sum(num(s['L1 Wavefronts Shared']) or 0 for s in target)
   ideal=sum(num(s['L1 Wavefronts Shared Ideal']) or 0 for s in target)
   pred=sum(num(s['Predicated-On Thread Instructions Executed']) or 0 for s in target)
   expected=f['expected_Source_WF_per_round_CTA']*128*N
   # ptxas can unroll the inline PTX loop. Dynamic totals, not a fixed
   # number of static PCs, establish preservation. Store unroll multiplicity.
   ok=(len(target)%len(f['original_PCs'])==0) and wf==ideal==expected
   rec=dict(family=f['name'],kind=f['id'],N=N,mode=mode,target_PCs=len(target),expected_PCs=len(f['original_PCs']),
     Source_WF=wf,Source_ideal=ideal,expected_Source_WF=expected,Source_gate=ok,pred_on=pred,
     native_LSU_HW_clock_ratio=(x.get('l1tex__cycles_elapsed.avg.per_second')/x.get('sm__cycles_elapsed.avg.per_second') if x.get('sm__cycles_elapsed.avg.per_second') else None),
     **{k:x.get(v) for k,v in metrics.items()})
   rs.append(rec)
   gates.append(dict(tag=tag,pass_gate=ok,sha256={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in [p,sp]}))
summary=[]
for f in F:
 for mode in [1,3]:
  rr=sorted([r for r in rs if r['kind']==f['id'] and r['mode']==mode],key=lambda r:r['N'])
  if len(rr)!=2:continue
  a,b=rr;dd=(b['N']-a['N'])*128
  field='write' if f['ptx'].startswith('stmatrix') else 'read'
  delta={k:(b[k]-a[k])/dd for k in metrics}
  summary.append(dict(family=f['name'],mode=mode,expected_Source_WF_per_round=f['expected_Source_WF_per_round_CTA'],
    hardware_target_direction_WF_per_round=delta[field],hardware_conflict_per_round=delta[field+'_conflict'],
    hardware_over_Source_ratio=delta[field]/f['expected_Source_WF_per_round_CTA'],
    Source_gate=a['Source_gate'] and b['Source_gate'],raw_slopes=delta,
    interpretation='Counter-demand slope, not timingservice; no per-stage absoluteattribution or purehardwarecapacity claim'))
out={'status':'COMPLETE' if len(rs)==24 and all(r['Source_gate'] for r in rs) else 'PARTIAL_OR_SOURCE_GATE_FAILED',
     'captures':len(rs),'all_Source_gates':all(r['Source_gate'] for r in rs),'records':rs,'slopes':summary,'gates':gates,'model_enabled':False}
(R/'MATCHED_COUNTER_SUMMARY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8');print(json.dumps({'status':out['status'],'captures':len(rs),'slopes':summary},ensure_ascii=False))
