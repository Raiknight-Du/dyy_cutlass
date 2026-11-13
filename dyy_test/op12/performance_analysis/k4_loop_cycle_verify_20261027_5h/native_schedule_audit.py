from pathlib import Path
import struct,csv,re,json,hashlib,sys
from collections import defaultdict,Counter
R=Path(__file__).resolve().parent;D=R/'derived'
P=Path(r'E:/Work/Lab/HUAWEI benchmark/op12/K4/分析交付_20261026_10h/01_native_audit/inputs/original_K4.cubin')
b=P.read_bytes();assert hashlib.sha256(b).hexdigest()=='06ebd78963371a30adac142c06b86950ec300b6cb762a209c5afc67628941814'
off=struct.unpack_from('<Q',b,40)[0];size,num,si=struct.unpack_from('<HHH',b,58)
hh=[struct.unpack_from('<IIQQQQIIQQ',b,off+i*size) for i in range(num)]
s=hh[si];names=b[s[4]:s[4]+s[5]]
tt=[(names[h[0]:].split(b'\0')[0].decode(),h) for h in hh]
sec=next(h for n,h in tt if n.startswith('.text.'))
text=b[sec[4]:sec[4]+sec[5]]
source=list(csv.DictReader((R/'inputs/EXTENDED_NATIVE_PER_PC.csv').open(encoding='utf-8-sig')))
rs=[];agg=defaultdict(lambda:defaultdict(float))
for r in source:
    pc=int(r['relative_pc'],16);word=int.from_bytes(text[pc:pc+16],'little')
    stall=(word>>105)&15;yieldbit=(word>>109)&1;wrbar=(word>>110)&7;rdbar=(word>>113)&7;waitmask=(word>>116)&63
    warp=float(r['dynamic_warp_instructions_full_action']);mathwork=not r['opcode'].startswith(('SYNCS','BAR','BRA','NANOSLEEP','DEPBAR','FENCE','MEMBAR','BSSY','BSYNC','EXIT','NOP'))
    z=dict(pc=hex(pc),opcode=r['opcode'],SASS=r['SASS'],area=r['area'],stall=stall,yieldbit=yieldbit,wrbar=wrbar,rdbar=rdbar,waitmask=waitmask,
        warp_full_action=warp,per_CG1_warp_loop_issued=warp/8192/4,
        nonpolling_or_sync=mathwork,compiled_stall_service_per_CG1_warp=warp/8192/4*max(1,stall))
    rs.append(z)
    if 0x6fb0<=pc<=0x95f0 and mathwork:
        a=agg[r['opcode']];a['per_warp_inst']+=warp/8192/4;a['stall_sum']+=warp/8192/4*max(1,stall)
out={'scope':'OriginalCG1 nonpolling/control instructionstream; controlfield lowerbound, nothardwareRAWlatency; dynamicweights limitedto provenonce/block families',
 'stall_field':'bits108:105, sameB200 priorSHFL controlpatch experiment',
 'sum_nonpolling_compiled_stall_CG1':sum(v['stall_sum'] for v in agg.values()),
 'by_opcode':dict(agg),'model_enabled':False}
(D/'NATIVE_SCHEDULE_SUMMARY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
with (D/'NATIVE_SCHEDULE_PC.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(rs[0]));w.writeheader();w.writerows(rs)
sys.stdout.reconfigure(encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
