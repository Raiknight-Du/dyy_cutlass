from pathlib import Path
import csv,re,json,statistics,sys
R=Path(__file__).resolve().parent
MET={'read':'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum','write':'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
 'read_conflict':'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum','write_conflict':'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
 'TC':'l1tex__data_pipe_tc_wavefronts.sum','arbitration':'l1tex__data_bank_conflicts_type_arbitration.sum','Data':'l1tex__data_pipe_lsu_wavefronts.sum',
 'DRAM_read_B':'dram__bytes_read.sum','DRAM_write_B':'dram__bytes_write.sum','L2_sectors':'lts__t_sectors.sum',
 'f_SM':'sm__cycles_elapsed.avg.per_second','f_L1':'l1tex__cycles_elapsed.avg.per_second'}
def num(v):
 try:return float(v.replace(',',''))
 except:return None
def raw(p):
 a=list(csv.reader(p.open(encoding='utf-8-sig')));i=next(i for i,r in enumerate(a) if r and r[0]=='ID' and 'Kernel Name' in r)
 # Header is followed by the units row, then the single selected action.
 assert len([r for r in a[i+2:] if r and len(r)>10 and r[0].isdigit()])==1,p
 return dict(zip(a[i],a[i+2]))
def source(p):
 a=list(csv.reader(p.open(encoding='utf-8-sig')));i=next(i for i,r in enumerate(a) if 'Address' in r and 'Source' in r)
 return [dict(zip(a[i],r)) for r in a[i+1:] if r and r[0].startswith('0x')]
records=[]
for folder,variants in [('vjoin_retry2_ncu',['B0','KSJOIN','QSJOIN']),('gatejoin_ncu',['GATEJOIN'])]:
 for variant in variants:
  for t in range(3):
   p=R/folder/f'{variant}_t{t}_raw.csv';sp=R/folder/f'{variant}_t{t}_source.csv'
   if not p.exists() or not sp.exists():continue
   x=raw(p);src=source(sp);ops={}
   for s in src:
    m=re.match(r'\s*(?:@!?\w+\s+)?(\w[\w.]*)',s['Source'])
    if m:ops[m[1]]=ops.get(m[1],0)+(num(s['Instructions Executed']) or 0)
   swf=sum(num(s['L1 Wavefronts Shared']) or 0 for s in src)
   iwf=sum(num(s['L1 Wavefronts Shared Ideal']) or 0 for s in src)
   rec=dict(variant=variant,trial=t,Source_WF=swf,Source_ideal=iwf,op_counts=ops,**{k:num(x.get(v,'')) for k,v in MET.items()})
   rec.update(HW_shared_excess=rec['read']+rec['write']-swf,
    Source_address_excess=swf-iwf,TC_gate=rec['TC']==6291456,
    all_Source_same_with_current_B0=swf==7178496,
    note='Full K4 native counters, complete cold-K1 CP history; not per-stage durations. Cross-client arbitration attribution needs native placement and controls.')
   records.append(rec)
summary=[]
for v in ['B0','KSJOIN','QSJOIN','GATEJOIN']:
 rr=[r for r in records if r['variant']==v]
 if not rr:continue
 summary.append(dict(variant=v,n=len(rr),**{k:statistics.mean(r[k] for r in rr) for k in ['read','write','read_conflict','write_conflict','arbitration','TC','HW_shared_excess','Source_WF','DRAM_read_B','DRAM_write_B','L2_sectors']},
  exact_Source_no_address_conflict=all(r['Source_address_excess']==0 for r in rr),all_TC=all(r['TC_gate'] for r in rr),
  opcode_reconciliation='Review per-record original-versus-variant totals, including moved barrier polls separately.'))
out=dict(records=records,summary=summary,model_enabled=False)
(R/'V_JOIN_COUNTERS_V2.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8');print(json.dumps(summary,ensure_ascii=False))
