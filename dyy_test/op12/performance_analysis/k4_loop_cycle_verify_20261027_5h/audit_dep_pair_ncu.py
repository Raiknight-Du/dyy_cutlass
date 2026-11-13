from pathlib import Path
import csv,json,re,collections,hashlib
R=Path(__file__).resolve().parent;B=R/'dep_pair_ncu'
def no(x):
 try:return int(float(x.replace(',','')))
 except:return 0
def src(p):
 a=list(csv.reader(p.open(encoding='utf-8-sig')));i=next(i for i,r in enumerate(a) if 'Address' in r and 'Source' in r)
 return [dict(zip(a[i],r)) for r in a[i+1:] if r and r[0].startswith('0x')]
out=[]
for kind in [0,1]:
 for w in [1,4]:
  data=[]
  for m in [0,1]:
   tag=f'k{kind}_w{w}_t0_m{m}';p=B/(tag+'_source.csv');ss=src(p)
   ld=[s for s in ss if re.search(r'\bLDTM',s['Source'])]
   counts=collections.Counter()
   for s in ss:
    if no(s['Instructions Executed'])>4:
     op=re.search(r'(?:@!?\w+\s+)?(\w[\w.]*)',s['Source'].strip())[1]
     if op.startswith('LDTM'):continue
     counts[op]+=no(s['Instructions Executed'])
   iden=json.loads((B/tag/'IDENTITY.json').read_text())
   gate=len(ld)==16 and all(no(x['Instructions Executed'])==128*w and no(x['Predicated-On Thread Instructions Executed'])==128*w*32*m for x in ld)
   data.append(dict(mode=m,target_PC_count=len(ld),target_warp_inst=sum(no(x['Instructions Executed']) for x in ld),
      target_pred_on=sum(no(x['Predicated-On Thread Instructions Executed']) for x in ld),target_gate=gate,
      nonload_dynamic_loop_ops=dict(counts),cubin_sha256=iden['cubin_sha256']))
  same=data[0]['nonload_dynamic_loop_ops']==data[1]['nonload_dynamic_loop_ops']
  out.append(dict(kind=kind,warps=w,records=data,same_nonload_work=same,same_cubin=data[0]['cubin_sha256']==data[1]['cubin_sha256'],
    pass_gate=same and all(x['target_gate'] for x in data) and data[0]['cubin_sha256']==data[1]['cubin_sha256'],
    interpretation='Single read/address feedback primitive and no-load protocol, same CUBIN and same nonload operation counts. Conditional completion bounds only; not a GDN stage cost.',model_enabled=False))
(R/'DEP_PAIR_NCU_AUDIT.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps([dict(kind=x['kind'],warps=x['warps'],pass_gate=x['pass_gate'],same_nonload=x['same_nonload_work'],same_cubin=x['same_cubin']) for x in out]))
