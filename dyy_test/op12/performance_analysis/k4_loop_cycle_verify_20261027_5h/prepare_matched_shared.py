from pathlib import Path
import json,csv,sys,importlib.util,shutil
R=Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding='utf-8')
sp=importlib.util.spec_from_file_location('cg0',R/'cg0_address_audit.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
cg0=list(csv.DictReader((R/'derived/CG0_LANE_ADDRESSES.csv').open(encoding='utf-8-sig')))
pcrows=m.rows('EXPLICIT_SHARED_NATIVE_PER_PC.csv')
specs=[('CG0_LDS', [r['relative_pc'] for r in pcrows if r['stage']=='T_transform' and r['opcode']=='LDS'],0x36400,'ld.shared.b32',144),
       ('CG1_GATE', [r['relative_pc'] for r in pcrows if r['stage']=='CG1_gate_shared_reads'],0x36400,'mixed_scalar64',136),
       ('CG0_LDSM', [r['relative_pc'] for r in pcrows if r['stage']=='T_transform' and r['opcode'].startswith('LDSM')],0x20400,'ldmatrix.sync.aligned.m8n8.x4.shared.b16',64),
       ('CG1_LDSM', [r['relative_pc'] for r in pcrows if r['stage']=='CG1.V_gather_for_residual'],0x14400,'ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16',128),
       ('CG0_STSM', [r['relative_pc'] for r in pcrows if r['stage']=='T_transform' and r['opcode'].startswith('STSM')],0x24400,'stmatrix.sync.aligned.m8n8.x4.trans.shared.b16',64),
       ('CG1_STSM', [r['relative_pc'] for r in pcrows if r['stage']=='CG1.O_pack_to_SMEM'],0x2e400,'stmatrix.sync.aligned.m8n8.x4.trans.shared.b16',128)]
def lut(a,b,c,imm):return m.lut(a,b,c,imm)
def vbase(tid):
    r4=tid*64;r7=lut(tid,0x28,0,0xc0);r5=tid*32
    r7=lut(r7,0x1c0,r4,0xf8);r5=lut(r5,0x200,0,0xc0);r6=r7>>3
    r5=lut(r5,0x1000,r4,0xf8);r6=lut(r7,0x38,r6,0x78);r5=lut(r5,r6,0,0xfc)
    return 2*r5,32 if not(r7&0x80) else -32
families=[]
for idx,(name,pcs,base,op,wf) in enumerate(specs):
    table=[];ops=[]
    for pc in pcs:
        r=next(r for r in pcrows if r['relative_pc']==pc)
        ops.append(r['opcode'])
    for tid in range(128):
        rr=[]
        for j,pc in enumerate(pcs):
            if name.startswith('CG0'):
                rec=next((a for a in cg0 if a['pc']==pc and int(a['tid'])==tid),None)
                off=int(rec['address_B'])-base if rec and rec['active']=='True' else -1
            elif name=='CG1_GATE':
                if pc=='0x7050':off=0x5fc
                elif pc=='0x7a70':off=0xfc
                else:
                    s=m.PROG # keep independent decode based original SASS string
                    expr=r'\+0x([0-9a-f]+)\]'
                    import re
                    add=int(re.search(expr,next(r['SASS'] for r in pcrows if r['relative_pc']==pc))[1],16)
                    off=8*(tid%4)+add-base
            else:
                aa,dd=vbase(tid+128)
                off=aa+(dd if j>=4 else 0)+(j%4)*0x800
            assert -1<=off<32768,(name,tid,j,off)
            rr.append(off)
        table.append(rr)
    widths=[4 if op.startswith(('ldmatrix','stmatrix')) else 2 if code=='LDS.64' else 1 for code in ops]
    starts=[];vv=0
    for z in widths:starts.append(vv);vv+=z
    families.append(dict(id=idx,name=name,original_PCs=pcs,opcodes=ops,ptx=op,widths=widths,value_starts=starts,value_count=vv,
       expected_Source_WF_per_round_CTA=wf,participating_warps=4,offsets=table,
       offset_units='B from alignedscratchbase',original_stage_slot=0,
       scope='Nativeinstruction family, same laneaddress/predicate/width and4warps; independentperaccessregisters toavoid syntheticWAW; count mechanism only, notmatchednativeintervalrate; syntheticresident input andQK SS competitor',
       all_instructions_have_active_lanes=all(any(t[j]>=0 for t in table) for j in range(len(pcs)))))
(R/'FAMILIES.json').write_text(json.dumps(families,ensure_ascii=False,indent=2),encoding='utf-8')
# Derive one independent module from archivedfour-mode scaffold, not upstream.
src=(R/'shared_coupling.py').read_text(encoding='utf-8')
start=src.index('def shader_assembly():');end=src.index('class SharedCoupling',start)
pre=r'''FAMILIES = __import__('json').loads(__import__('pathlib').Path(__file__).with_name('FAMILIES.json').read_text())

def matched_assembly(kind):
    fam=FAMILIES[kind];count=len(fam['original_PCs'])
    q=['{','.reg .b32 v<%d>, addr<%d>, i, sum;'%(fam['value_count'],count),'.reg .b64 pair64;','.reg .pred on, active, more;',
       'setp.ne.u32 active, $%d, 0;'%(count+3)]
    q += ['mov.b32 v%d, %d;'%(j,(0x3c003c00+j*0x10001)&0xffffffff) for j in range(fam['value_count'])]
    q += ['add.u32 addr%d, $1, $%d;'%(j,j+2) for j in range(count)]
    q += ['mov.u32 i, 0;','MATCH_LOOP:']
    for j in range(count):
        q += ['setp.ge.s32 on, $%d, 0;'%(j+2),'and.pred on, on, active;']
        op=fam['ptx']
        if op=='mixed_scalar64':op='ld.shared.v2.b32' if fam['opcodes'][j]=='LDS.64' else 'ld.shared.b32'
        start=fam['value_starts'][j];regs='{'+','.join('v%d'%(start+k) for k in range(fam['widths'][j]))+'}'
        if op.startswith('ldmatrix'):inst='%s %s, [addr%d];'%(op,regs,j)
        elif op.startswith('stmatrix'):inst='%s [addr%d], %s;'%(op,j,regs)
        elif 'v2' in op:
            q += ['@on ld.volatile.shared.b64 pair64, [addr%d];'%j,'@on mov.b64 %s, pair64;'%regs]
            continue
        else:inst='%s v%d, [addr%d];'%(op,start,j)
        if op.startswith('ld.shared'):
            # Preserve each resident scalar read across iterations; otherwise
            # ptxas legally hoists/merges repeated invariant loads.
            inst=inst.replace('ld.shared','ld.volatile.shared')
        q += ['@on '+inst]
    q += ['add.u32 i, i, 1;','setp.lt.u32 more, i, $%d;'%(count+2),'@more bra MATCH_LOOP;','mov.b32 sum,v0;']
    q += ['xor.b32 sum, sum, v%d;'%k for k in range(1,fam['value_count'])]
    q += ['mov.b32 $0,sum;','}']
    return '\n'.join(q)

@dsl_user_op
def matched_loop(base,reg_offsets,n,enabled,kind,*,loc=None,ip=None):
    vals=[cutlass.Int32(base)] + [cutlass.Int32(reg_offsets[j]) for j in range(len(FAMILIES[kind]['original_PCs']))] + [cutlass.Int32(n),cutlass.Int32(enabled)]
    value=llvm.inline_asm(cutlass.Uint32.mlir_type,[x.ir_value(loc=loc,ip=ip) for x in vals],
        matched_assembly(kind),'=r,'+','.join(['r']*len(vals))+',~{memory}',has_side_effects=True,
        is_align_stack=False,asm_dialect=llvm.AsmDialect.AD_ATT,loc=loc,ip=ip)
    return cutlass.Uint32(value)

'''
src=src[:start]+pre+src[end:]
src=src.replace('class SharedCoupling(QkResident):','class MatchedShared(QkResident):')
src=src.replace('def __init__(self):\n        super().__init__(instrumented=False,per_round_wait=False)','def __init__(self,kind):\n        super().__init__(instrumented=False,per_round_wait=False)\n        self.kind=kind\n        self.count=len(FAMILIES[kind]["original_PCs"])')
src=src.replace('shader_checks:cute.Tensor,trace:cute.Tensor,','shader_checks:cute.Tensor,trace:cute.Tensor,offsets:cute.Tensor,')
src=src.replace('trace,iterations,mode).launch','trace,offsets,iterations,mode).launch')
src=src.replace('shader_checks,trace,iterations:cutlass.Int32,mode:cutlass.Int32):','shader_checks,trace,offsets,iterations:cutlass.Int32,mode:cutlass.Int32):')
src=src.replace('value=(idx*cutlass.Uint32(0x9e3779b9))^((idx>>3)*cutlass.Uint32(0x85ebca6b))','value=cutlass.Uint32(0x3c003c00)')
src=src.replace('            # First rendezvous completes initialization;', '''            reg_offsets=cute.make_rmem_tensor((self.count,),cutlass.Int32)
            for j in cutlass.range(self.count,unroll_full=True):
                reg_offsets[j]=offsets[tid,j]
            # First rendezvous completes initialization;''')
src=src.replace('check=shader_loop(st.shader.data_ptr().toint(),tid,iterations,mode&1)','check=matched_loop(st.shader.data_ptr().toint(),reg_offsets,iterations,mode&1,self.kind)')
(R/'matched_shared.py').write_text(src,encoding='utf-8')
print('Prepared families',[(f['name'],len(f['original_PCs']),f['expected_Source_WF_per_round_CTA']) for f in families])
