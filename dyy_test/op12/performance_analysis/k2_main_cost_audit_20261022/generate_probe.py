"""Isolated inherited K2: only wrap compute-group loop in two endpoints."""
import ast,copy,hashlib,json,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
tree=ast.parse(a.source.read_text());base=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CPDeltaRuleMNPrecomputeUtcmma1Sm100')
methods=[n for n in base.body if isinstance(n,ast.FunctionDef) and n.name in ('run_compute_group_0','run_compute_group_1')]
classes=[];mapping=[]
for mode,period in [('B1',0),('P1',128),('P8',16)]:
    mm=[]
    for orig in methods:
        m=copy.deepcopy(orig);loops=[(i,n) for i,n in enumerate(m.body) if isinstance(n,ast.For)]
        assert len(loops)==1
        i,loop=loops[0]
        mapping.append({'mode':mode,'role':m.name,'source_begin':loop.lineno,'source_end':loop.end_lineno,'endpoints':'before first loop wait / after final loop release, before workspace store','period':period})
        if period:
            before=ast.parse(f'_probe_clock_asm(self.probe_address, 0, {period})').body
            after=ast.parse(f'_probe_clock_asm(self.probe_address, 1, {period})').body
            m.body[i:i+1]=before+[loop]+after
        mm.append(m)
    classes.append(ast.ClassDef(name='K2Body_'+mode,bases=[ast.Name(id='_Base',ctx=ast.Load())],keywords=[],body=mm,decorator_list=[]))
header='''import cutlass
import cutlass.cute as cute
from flashinfer.gdn_kernels.blackwell import gated_delta_net_cp as _SOURCE
globals().update({n:getattr(_SOURCE,n) for n in dir(_SOURCE) if not n.startswith("__")})
_Base=_SOURCE.CPDeltaRuleMNPrecomputeUtcmma1Sm100
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
@dsl_user_op
def _probe_clock_asm(address, endpoint, period, *, loc=None, ip=None):
    asm="""{
    .reg .pred sel, p;
    .reg .b32 bx, tid, warp, lane, rem, sid;
    .reg .b64 addr, clk, ns, sm;
    mov.u32 bx, %ctaid.x;
    mov.u32 tid, %tid.x;
    shr.u32 warp, tid, 5;
    and.b32 lane, tid, 31;
    rem.u32 rem, bx, PERIOD;
    setp.eq.u32 sel, rem, 0;
    setp.eq.u32 p, lane, 0; and.pred sel, sel, p;
    setp.lt.u32 p, warp, 8; and.pred sel, sel, p;
    mov.u64 addr, ADDRESS;
    mad.wide.u32 addr, bx, 384, addr;
    mad.wide.u32 addr, warp, 48, addr;
    add.u64 addr, addr, OFFSET;
    @sel mov.u64 clk, %clock64;
    @sel mov.u64 ns, %globaltimer;
    mov.u32 sid, %smid;
    cvt.u64.u32 sm, sid;
    @sel st.global.u64 [addr], clk;
    @sel st.global.u64 [addr+8], ns;
    @sel st.global.u64 [addr+16], sm;
    }"""
    asm=asm.replace("PERIOD",str(period)).replace("ADDRESS",str(address)).replace("OFFSET",str(endpoint*24))
    llvm.inline_asm(res=None,operands_=[],asm_string=asm,constraints="~{memory}",has_side_effects=True,asm_dialect=llvm.AsmDialect.AD_ATT,loc=loc,ip=ip)
'''
a.out.mkdir(parents=True,exist_ok=True)
target=a.out/'k2_body_generated.py';target.write_text(header+ast.unparse(ast.fix_missing_locations(ast.Module(body=classes,type_ignores=[])))+'\n')
(a.out/'probe_manifest.json').write_text(json.dumps({'source':str(a.source),'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'generated_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'mapping':mapping,'local_bias_bound':'UNKNOWN'},indent=2))
print(target)
