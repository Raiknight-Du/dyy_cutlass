"""Offline restricted SASS interpreter for CG0 address/predicate lineage.
Float payload is unknown; never emulates CUDA mathematics or times targetcode.
"""
from pathlib import Path
import re,csv,json,sys
from collections import defaultdict
R=Path(__file__).resolve().parent;D=R/'derived';D.mkdir(exist_ok=True)
sys.stdout.reconfigure(encoding='utf-8')
def rows(n):return list(csv.DictReader((R/'inputs'/n).open(encoding='utf-8-sig',newline='')))
def lut(a,b,c,imm):
    return sum(((imm>>((((a>>i)&1)<<2)|(((b>>i)&1)<<1)|((c>>i)&1)))&1)<<i for i in range(32))
def signed(x):return (x&0x7fffffff)-(x&0x80000000)
def prog():
    pp=[]
    for line in (R/'inputs/original_K4.sass').read_text().splitlines():
        m=re.search(r'/\*([0-9a-f]+)\*/\s+(?:(@!?[A-Z0-9]+)\s+)?(\S+)\s*(.*?)\s*;',line)
        if m:
            pc,p,op,args=m.groups();pc=int(pc,16)
            if 0x1ca0<=pc<=0x5410:pp.append((pc,p,op,[x.strip().replace('.reuse','') for x in args.split(',')]))
    return pp
PROG=prog()
def run(tid,chunk=0,valid=4096,vim_min=True,debug=False):
    rr={'R0':tid,'R4':64,'R6':tid>>1,'R7':64,'R68':0,'R132':tid&31,
        'UR22':0,'UR4':64,'UR37':valid,'UR12':chunk//2,'UR8':64,'UR9':0,'R2':0,'R3':0}
    pp={'PT':True,'UPT':True};rec=[];unknown=[]
    def get(s):
        neg=s.startswith('-');s=s.lstrip('-');s=s.replace('.U32','')
        if s in ['RZ','URZ']:v=0
        elif s in ['PT','UPT']:v=True
        elif re.fullmatch(r'!?U?P\d+',s):v=pred(s)
        elif s in rr:v=rr[s]
        else:
            try:v=int(s,0)
            except ValueError:v=None
        return None if v is None else (-v if neg else v)
    def pred(s):
        inv=s.startswith('!');v=pp.get(s.lstrip('!'))
        return None if v is None else (not v if inv else bool(v))
    def put(k,v):
        if re.fullmatch(r'U?R\d+',k):rr[k]=None if v is None else int(v)&0xffffffff
        elif k not in ['PT','UPT']:pp[k]=None if v is None else bool(v)
    def calc(fn,*ss):
        vv=[get(s) for s in ss]
        return fn(*vv) if all(v is not None for v in vv) else None
    for pc,p,op,a in PROG:
        if debug and pc in [0x34d0,0x34e0,0x3500,0x3560,0x36d0]:print(hex(pc),a,{k:rr.get(k) for k in ['R36','UR4','R88','UR5','R70','R69']},pp)
        enabled=True if not p else pred(p[1:])
        if enabled is False:continue
        if enabled is None:unknown.append((hex(pc),'predicate',p))
        fam=op.split('.')[0]
        if fam in ['BRA','BAR','FENCE','MEMBAR','NOP','USETMAXREG','BSSY','BSYNC']:continue
        if fam=='LDL':put(a[0],64);continue
        if fam in ['LDS','LDSM','STS','STSM']:
            ad=next(q for q in a if '[' in q)
            expr=ad.strip('[]').replace('URZ','0').split('+')
            vv=[get(q) for q in expr];addr=sum(vv) if all(v is not None for v in vv) else None
            rec.append(dict(pc=hex(pc),opcode=op,tid=tid,warp=tid//32,lane=tid%32,
                active=enabled,address_B=addr))
            if addr is None:unknown.append((hex(pc),'address',ad))
            if fam in ['LDS','LDSM']:
                dest=int(a[0][1:]);sz=4 if fam=='LDSM' else 4 if '.128' in op else 2 if '.64' in op else 1
                for n in range(sz):put('R'+str(dest+n),None)
            continue
        if fam=='SYNCS':
            if a and re.fullmatch(r'P\d+',a[0]):put(a[0],True)
            continue
        if fam=='LOP3':
            ix=1 if a[0].startswith('P') else 0
            dest=a[ix];v=calc(lambda q,w,e:lut(q,w,e,int(a[ix+4],0)),*a[ix+1:ix+4])
            put(dest,v)
            if ix:put(a[0],None if v is None else v!=0)
        elif fam=='PLOP3':
            vals=[pred(s) for s in a[2:5]]
            v=None if any(v is None for v in vals) else (int(a[5],0)>>((int(vals[0])<<2)|(int(vals[1])<<1)|int(vals[2])))&1
            put(a[0],v)
        elif fam=='UPLOP3':
            vals=[pred(s) for s in a[2:5]]
            v=None if any(v is None for v in vals) else (int(a[5],0)>>((int(vals[0])<<2)|(int(vals[1])<<1)|int(vals[2])))&1
            put(a[0],v)
        elif fam in ['ISETP','UISETP']:
            q,w=get(a[2]),get(a[3]);cond=None
            if q is not None and w is not None:
                if '.U32' in op:q&=0xffffffff;w&=0xffffffff
                else:q=signed(q);w=signed(w)
                cmp=op.split('.')[1];cond={'EQ':q==w,'NE':q!=w,'LT':q<w,'LE':q<=w,'GT':q>w,'GE':q>=w}[cmp]
            z=pred(a[4]);bop=op.split('.')[-1]
            v=None if cond is None or z is None else (cond and z if bop=='AND' else cond or z if bop=='OR' else cond!=z)
            put(a[0],v)
        elif fam=='VIMNMX':
            c=pred(a[-1]);q,w=get(a[-3]),get(a[-2]);v=None
            if q is not None and w is not None and c is not None:
                use_min=(not c) if vim_min else c;v=min(signed(q),signed(w)) if use_min else max(signed(q),signed(w))
            put(a[0],v)
        elif fam in ['IMAD','UIMAD']:
            if '.MOV' in op:put(a[0],get(a[-1]))
            elif len(a)==4:put(a[0],calc(lambda q,w,e:q*w+e,*a[1:]))
            else:put(a[0],None)
        elif fam in ['IADD3','UIADD3']:
            ss=a[-3:];put(a[0],calc(lambda q,w,e:q+w+e,*ss))
        elif fam in ['LEA','ULEA']:
            if len(a)==4:put(a[0],calc(lambda q,w:(q<<int(a[3],0))+w,*a[1:3]))
            else:put(a[0],None)
        elif fam in ['VIADD','VIADDMNMX']:
            put(a[0],calc(lambda q,w:q+w,*a[1:3]))
        elif fam in ['SEL','USEL']:
            z=pred(a[3]);put(a[0],None if z is None else get(a[1] if z else a[2]))
        elif fam in ['SHF','USHF']:
            if '.R.' in op:v=calc(lambda q,w,e:((e<<32)|(q&0xffffffff))>>int(w)>>32,*a[1:4])
            else:v=calc(lambda q,w,e:((q<<int(w))|(e>>(32-int(w)) if int(w) else e))&0xffffffff,*a[1:4])
            put(a[0],v)
        elif fam in ['MOV','UMOV','R2UR']:put(a[0],get(a[1]))
        elif fam=='CS2R' and a[-1]=='SRZ':
            n=int(a[0][1:]);put(a[0],0);put('R'+str(n+1),0)
        elif fam=='P2R':
            mask=int(a[3],0);vals=[pred('P'+str(i)) for i in range(7) if mask&(1<<i)]
            put(a[0],None if any(z is None for z in vals) else sum(int(pred('P'+str(i)))<<i for i in range(7) if mask&(1<<i)))
        else:
            if a and re.fullmatch(r'R\d+',a[0]):put(a[0],None)
    return rec,unknown
def main():
    target=[r for r in rows('EXPLICIT_SHARED_NATIVE_PER_PC.csv') if r['stage'] in ['T_transform','QK_epilogue']]
    by={};allrr=[];fail=[]
    for tid in range(128):
        rr,un=run(tid)
        for r in rr:
            if r['pc'] in {r['relative_pc'] for r in target}:allrr.append(r)
    for a in target:
        rr=[r for r in allrr if r['pc']==a['relative_pc']]
        actual=sum(r['active'] is True for r in rr)
        expected=float(a['pred_on_lane_instructions_full_action'])/8192
        badaddr=[r for r in rr if r['active'] is not False and r['address_B'] is None]
        by[a['relative_pc']]={'actual_pred_on':actual,'expected_pred_on':expected,'unknown_addresses':len(badaddr)}
        if actual!=expected or badaddr:fail.append((a['relative_pc'],actual,expected,len(badaddr)))
        bankwf=0;maxconf=0
        for warp in range(4):
            ww=[r for r in rr if r['warp']==warp and r['active']]
            if a['opcode']=='LDS':
                banks=defaultdict(set)
                for r in ww:banks[(r['address_B']//4)%32].add(r['address_B']//4)
                conf=max([len(z) for z in banks.values()] or [0]);bankwf+=conf;maxconf=max(maxconf,conf)
            else:
                assert len(ww)==32
                for atom in range(4):
                    banks=defaultdict(set)
                    for r in ww[atom*8:atom*8+8]:
                        assert r['address_B']%16==0
                        for z in range(4):banks[(r['address_B']//4+z)%32].add(r['address_B']//4+z)
                    conf=max(len(z) for z in banks.values());bankwf+=conf;maxconf=max(maxconf,conf)
        by[a['relative_pc']].update(bank_predicted_WF=bankwf,expected_Source_WF=float(a['Source_address_model_wavefronts_full_action'])/8192,max_address_conflict=maxconf)
        assert bankwf==float(a['Source_address_model_wavefronts_full_action'])/8192,(a['relative_pc'],bankwf)
        assert maxconf==1
    (D/'CG0_FIRST_PASS.json').write_text(json.dumps({'fails':fail,'PC':by},indent=2),encoding='utf-8')
    print('Matched',len(target)-len(fail),'of',len(target),'fail first20',fail[:20])
    assert not fail
    with (D/'CG0_LANE_ADDRESSES.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(allrr[0]));w.writeheader();w.writerows(allrr)
if __name__=='__main__':main()
