from pathlib import Path
import csv,json,re,collections,statistics,sys
R=Path(__file__).resolve().parent;B=R/'alu_mix_v2_ncu'
def num(v):
 try:return float(v.replace(',',''))
 except:return 0.
def load(p,source=False):
 a=list(csv.reader(p.open(encoding='utf-8-sig')))
 if source:
  h=next(i for i,r in enumerate(a) if 'Address' in r and 'Source' in r)
  z=[dict(zip(a[h],r)) for r in a[h+1:] if r and r[0].startswith('0x')];base=min(int(r['Address'],16) for r in z)
  return [(int(r['Address'],16)-base,r) for r in z]
 h=next(i for i,r in enumerate(a) if r and r[0]=='ID' and 'Kernel Name' in r);return dict(zip(a[h],a[h+2]))
records=[]
for mode in [1,2,3]:
 for n in [256,512]:
  tag=f't0_m{mode}_p{n}';ss=load(B/(tag+'_source.csv'),True);raw=load(B/(tag+'_raw.csv'))
  pack=[r for pc,r in ss if 'F2FP.F16.F32.PACK_AB' in r['Source']]
  integer=[r for pc,r in ss if re.match(r'\s*IADD3 ',r['Source'])]
  # Dynamic IADD loop counts are >= n/8*4; fixed checksum/control
  # counters are tiny and are preserved separately instead of hidden.
  ints=[r for r in integer if num(r['Instructions Executed'])>=n/8*4 and ' -0x8' not in r['Source'] and '0xfffffff8' not in r['Source']]
  pcount=sum(num(r['Instructions Executed']) for r in pack)
  # The integer loop has256static targetPCs, identifiable by >=n/8*4
  # once enabled; one loop-counter IADD may also qualify. Check static
  # interval by excluding the decrement and export all candidates.
  icount=sum(num(r['Instructions Executed']) for r in ints)
  records.append(dict(mode=mode,N=n,pack_static=len(pack),pack_warp_inst=pcount,
    pack_gate=pcount==n*32*4*bool(mode&1),integer_candidate_static=len(ints),integer_candidate_warp_inst=icount,
    integer_native_rows=[dict(PC=r['Address'],SASS=r['Source'],executed=num(r['Instructions Executed'])) for r in ints],
    pipes={k:num(raw.get(k,'')) for k in ['smsp__inst_executed_pipe_alu.sum','smsp__inst_executed_pipe_fmaheavy.sum','smsp__inst_executed_pipe_fmalite.sum','smsp__inst_executed_pipe_xu.sum']},
    model_enabled=False))
(R/'ALU_MIX_NCU_AUDIT.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps([{k:v for k,v in r.items() if k!='integer_native_rows'} for r in records]))
