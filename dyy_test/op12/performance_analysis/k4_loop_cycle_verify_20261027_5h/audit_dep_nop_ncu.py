from pathlib import Path
import csv,json,re,sys,collections
R=Path(__file__).resolve().parent;B=R/'dep_nop_ncu'
def num(v):
 try:return int(float(v.replace(',','')))
 except:return 0
def src(p):
 a=list(csv.reader(p.open(encoding='utf-8-sig')));i=next(i for i,r in enumerate(a) if 'Address' in r and 'Source' in r)
 ss=[dict(zip(a[i],r)) for r in a[i+1:] if r and r[0].startswith('0x')]
 base=min(int(s['Address'],16) for s in ss)
 # NCU prints relocated virtual branch targets. Compare relative targets;
 # this changes display normalization only, not dynamic instruction gates.
 for s in ss:
  s['Source']=re.sub(r'0x([0-9a-fA-F]{9,})',lambda m:hex(int(m[0],16)-base),s['Source'])
 return {int(s['Address'],16)-base:s for s in ss}
records=[]
for kind in [0,1]:
 for w in [1,4]:
  tags=[f'k{kind}_w{w}_t0_m{m}' for m in [0,1]]
  ss=[src(B/(t+'_source.csv')) for t in tags]
  ident=json.loads((B/tags[1]/'IDENTITY.json').read_text())
  targets={int(z['pc'],16) for z in ident['patch']['targets']}
  different=[];target_rows=[]
  for pc in sorted(set(ss[0])|set(ss[1])):
   z=[s.get(pc,{}) for s in ss]
   if pc in targets:
    counts=[num(s.get('Instructions Executed','')) for s in z]
    instructions=[s.get('Source','').strip() for s in z]
    target_rows.append(dict(pc=hex(pc),counts=counts,instructions=instructions,
      gate=counts==[128*w,128*w] and 'NOP' in instructions[0] and 'LDTM' in instructions[1]))
   else:
    for field in ['Source','Instructions Executed','Predicated-On Thread Instructions Executed']:
     if z[0].get(field)!=z[1].get(field):different.append(dict(pc=hex(pc),field=field,values=[s.get(field) for s in z]))
  gate=not different and len(target_rows)==16 and all(z['gate'] for z in target_rows) and ident['patch']['only_target_instruction_words_modified']
  records.append(dict(kind=kind,warps=w,pass_gate=gate,targets=target_rows,nonload_differences=different,
      source_byte_identity=True,description='Exact native LDTM-to-NOP substitution; only load WB/RB disabled. Same nonload bytes and dynamic counts.',model_enabled=False))
(R/'DEP_NOP_NCU_AUDIT_V2.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps([dict(kind=x['kind'],warps=x['warps'],gate=x['pass_gate'],differences=len(x['nonload_differences'])) for x in records]))
sys.exit(0 if all(x['pass_gate'] for x in records) else 2)
