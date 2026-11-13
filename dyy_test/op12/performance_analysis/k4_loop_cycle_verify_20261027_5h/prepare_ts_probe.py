from pathlib import Path
import ast,json,struct
R=Path(__file__).resolve().parent
s=(R/'matched_shared.py').read_text(encoding='utf-8')
s=s.replace('class MatchedShared(QkResident):','class MatchedSharedTS(QkResident):')
old='(64,64,16),self.cta_group,tcgen05.OperandSource.SMEM,OperandMajorMode.K,OperandMajorMode.K))'
new='(128,64,16),self.cta_group,tcgen05.OperandSource.TMEM,OperandMajorMode.K,OperandMajorMode.K))'
assert s.count(old)==1;s=s.replace(old,new)
where='        al=sm100_utils.make_smem_layout_a(mma,(64,64,128),cutlass.Float16,1)'
assert s.count(where)==1
s=s.replace(where,'''        layout_mma=cute.make_tiled_mma(tcgen05.MmaF16BF16Op(cutlass.Float16,cutlass.Float32,
             (64,64,16),self.cta_group,tcgen05.OperandSource.SMEM,OperandMajorMode.K,OperandMajorMode.K))
        al=sm100_utils.make_smem_layout_a(layout_mma,(64,64,128),cutlass.Float16,1)''')
s=s.replace('            self.clear_c_enclosing_region(ptr,zero_mma,tid)','''            self.clear_c_enclosing_region(ptr,zero_mma,tid)
            self.seed_nonzero_a(ptr+128,zero_mma,tid)''')
old='            aa,bb=mma.make_fragment_A(sA),mma.make_fragment_B(sB)'
assert s.count(old)==1
s=s.replace(old,'''            ash=mma.partition_shape_A((128,128))
            afake=mma.make_fragment_A(cute.append(ash,1))
            aa=cute.make_tensor(cute.recast_ptr(ptr+128,dtype=cutlass.Float16),afake.layout)
            bb=mma.make_fragment_B(sB)''')
s=s.replace('for it in cutlass.range(iterations,unroll=1):','for it in cutlass.range(iterations*2,unroll=1):')
# M128 N64 output, preserve a stage dimension; literal helper methods are
# copied rather than modifying the archived scaffold.
t=(R/'old_qk_resident.py').read_text(encoding='utf-8');tr=ast.parse(t)
c=next(n for n in tr.body if isinstance(n,ast.ClassDef))
for name in ['make_c','output']:
 m=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
 meth='\n'.join(t.splitlines()[m.lineno-2:m.end_lineno])+'\n'
 meth=meth.replace('(64, 64)','(128, 64)')
 if name=='make_c':
  meth=meth.replace('fake = mma.make_fragment_C(shape)','fake = mma.make_fragment_C(cute.append(shape,1))')
  meth=meth.replace('staged = cute.flat_product(fake.layout, cute.make_layout((1,), stride=(1,)))','staged = fake.layout')
 if name=='output':
  k=meth.index('        for sub in cutlass.range(')
  meth=meth[:k]+'''        cute.copy(cp,src[None,None,None,0],reg)
        for value in cutlass.range(cute.size(reg),unroll_full=True):
            row,col=coords[value]
            gO[row,col]=reg[value]
        cute.arch.fence_view_async_tmem_load()
'''
 s+='\n'+meth
mtree=ast.parse((R/'matched_shared.py').read_text(encoding='utf-8'))
mc=next(x for x in mtree.body if isinstance(x,ast.ClassDef))
init=next(x for x in mc.body if isinstance(x,ast.FunctionDef) and x.name=='clear_c_enclosing_region')
lines=(R/'matched_shared.py').read_text(encoding='utf-8').splitlines()
seed='\n'.join(lines[init.lineno-2:init.end_lineno])+'\n'
seed=seed.replace('def clear_c_enclosing_region','def seed_nonzero_a').replace('r.fill(0)',f'r.fill(cutlass.Float32({struct.unpack("<f",struct.pack("<I",0x3c003c00))[0]!r}))')
s+='\n'+seed
ast.parse(s);(R/'matched_shared_ts.py').write_text(s,encoding='utf-8')
r=(R/'run_matched_shared.py').read_text(encoding='utf-8')
r=r.replace('from matched_shared import MatchedShared,FAMILIES','from matched_shared_ts import MatchedSharedTS as MatchedShared,FAMILIES')
r=r.replace('go=torch.empty((128,64,64)','go=torch.empty((128,128,64)')
r=r.replace("['matched_shared.py','FAMILIES.json','old_qk_resident.py']","['matched_shared_ts.py','FAMILIES.json','old_qk_resident.py']")
r=r.replace('go==(128.0 if m&2 else 0.0)','go==(128.0 if m&2 else 0.0)')
r=r.replace("profile=a.profile,model_enabled=False)","profile=a.profile,competitor='2x M128N64K128 TS perround, A nonzero residentTMEM, B SMEM',TC_FLOP_per_round=4194304,TC_shared_WF_per_round=256,model_enabled=False)")
ast.parse(r);(R/'run_matched_ts.py').write_text(r,encoding='utf-8')
print('Prepared TS comparator; no upstream change')
