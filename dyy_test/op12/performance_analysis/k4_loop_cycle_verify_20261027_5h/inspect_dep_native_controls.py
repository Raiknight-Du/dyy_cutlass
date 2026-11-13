from pathlib import Path
import re,struct,json
R=Path(__file__).resolve().parent;out=[]
for kind in [0,1]:
 d=R/f'dep_nop_formal/k{kind}_w1_t0_m-1/build';p=next(x for x in d.glob('*.cubin') if x.name!='NOP.cubin');ss=p.with_name(p.name.replace('.sm_100a.cubin','.sass')).read_text()
 b=p.read_bytes();off=struct.unpack_from('<Q',b,40)[0];sz,n,st=struct.unpack_from('<HHH',b,58);hh=[struct.unpack_from('<IIQQQQIIQQ',b,off+i*sz) for i in range(n)];h=hh[st];names=b[h[4]:h[4]+h[5]]
 text=next(h for h in hh if names[h[0]:].split(b'\0')[0].startswith(b'.text.'))
 rows=[]
 for line in ss.splitlines():
  m=re.search(r'/\*([0-9a-f]+)\*/\s+(.*?);',line)
  if m:
   pc=int(m[1],16);v=int.from_bytes(b[text[4]+pc:text[4]+pc+16],'little');rows.append(dict(pc=pc,SASS=m[2].strip(),stall=(v>>105)&15,WB=(v>>110)&7,RB=(v>>113)&7,waitmask=(v>>116)&63))
 result=[]
 for i,z in enumerate(rows):
  if not z['SASS'].startswith('LDTM'):continue
  q=0;success=None
  for x in rows[i+1:i+8]:
   q+=rows[rows.index(x)-1]['stall'] or 1
   if z['WB']<6 and x['waitmask']&(1<<z['WB']):success=dict(first_wait_PC=hex(x['pc']),q_before_wait=q);break
  result.append(dict(load_PC=hex(z['pc']),WB=z['WB'],following=rows[i:i+6],first_wait=success))
 out.append(dict(kind=kind,loads=result))
(R/'DEP_NOP_CONTROL_WAIT_POSITION.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps([dict(kind=x['kind'],positions=[z['first_wait'] for z in x['loads']]) for x in out]))
