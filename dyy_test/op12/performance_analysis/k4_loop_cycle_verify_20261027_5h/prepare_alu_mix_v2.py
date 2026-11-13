from pathlib import Path
R=Path(__file__).resolve().parent
t=(R/'prepare_alu_mix.py').read_text()
for old,new in [
 ("R/'alu_mix_build'","R/'alu_mix_v2_build'"),
 ('.param .u32 salt)', '.param .u32 salt,.param .u32 salt2,.param .u32 zero)'),
 ('.b32 t,w,rem,md,z,sm,check,bits,p<32>', '.b32 t,w,rem,md,z,z2,sm,check,bits,p<32>'),
 ("'mov.b32 b,0;'", "'ld.param.u32 z2,[salt2];','ld.param.f32 b,[zero];'"),
 ("f'add.s32 p{i},p{i},z;'", "f'add.s32 p{i},p{i},z;',f'add.s32 p{i},p{i},z2;'"),
]:
 assert old in t,old;t=t.replace(old,new)
exec(compile(t,'prepare_alu_mix_v2_generated.py','exec'))
