from pathlib import Path
import csv,json,re,statistics
R=Path(__file__).resolve().parent
d=json.loads((R/'V_JOIN_COUNTERS.json').read_text());base=d['records'][0]['op_counts'];result=[]
native_prefix=('UTCHMMA','UTMALDG','UTMASTG','LDTM','STTM','LDSM','STSM','LDS','STS','F2FP','FMUL2','FADD2','HADD2','HMUL2','HFMA2','MUFU')
for folder,names in [('vjoin_retry2_ncu',['B0','KSJOIN','QSJOIN']),('gatejoin_ncu',['GATEJOIN'])]:
 for name in names:
  for t in range(3):
   tag=R/folder/f'{name}_t{t}';rr=next(x for x in d['records'] if x['variant']==name and x['trial']==t)
   artifacts=[json.loads(p.read_text()) for p in tag.glob('replay_pid*/COMPILE_AUDIT.json')]
   checks=[json.loads(p.read_text()) for p in tag.glob('replay_pid*/RESULT.json')]
   diff={k:[base.get(k,0),rr['op_counts'].get(k,0)] for k in set(base)|set(rr['op_counts']) if k.startswith(native_prefix) and base.get(k,0)!=rr['op_counts'].get(k,0)}
   # Source PC addresses include independent process ASLR. Keep relative PCs.
   aa=list(csv.reader((R/folder/f'{name}_t{t}_source.csv').open(encoding='utf-8-sig')))
   h=next(i for i,x in enumerate(aa) if 'Address' in x and 'Source' in x)
   src=[dict(zip(aa[h],x)) for x in aa[h+1:] if x and x[0].startswith('0x')]
   start=min(int(s['Address'],16) for s in src)
   candidates=[s for s in src if 'LDSM' in s['Source'] and 0x6000<int(s['Address'],16)-start<0xa000]
   # Export the local native sequence around V LDSM for human/source review.
   lo=min(int(s['Address'],16)-start for s in candidates)-160 if candidates else 0
   hi=max(int(s['Address'],16)-start for s in candidates)+80 if candidates else 0
   seq=[dict(pc=hex(int(s['Address'],16)-start),SASS=s['Source'],executed=s['Instructions Executed']) for s in src if lo<=int(s['Address'],16)-start<=hi]
   passgate=not diff and all(q['correctness_bitwise'] and q['full_CP'] and q['cold_before_K1'] for q in checks) and rr['Source_WF']==7178496 and rr['TC']==6291456
   result.append(dict(variant=name,trial=t,pass_gate=passgate,native_work_differences=diff,source_WF=rr['Source_WF'],TC_WF=rr['TC'],
     compilation_records=artifacts,bitwise_fullCP=len(checks)>0 and all(q['correctness_bitwise'] for q in checks),V_LDSM_local_native_sequence=seq,
     note='One once/CTA LDS polling/control instruction differs from historical Source total by 512WF; current cohort equal and loop work unchanged.',model_enabled=False))
(R/'V_JOIN_WORK_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps([dict(variant=x['variant'],trial=x['trial'],gate=x['pass_gate'],diff=x['native_work_differences']) for x in result],ensure_ascii=False))
