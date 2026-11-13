from pathlib import Path
R=Path(__file__).resolve().parent
t=(R/'run_alu_mix.py').read_text()
for old,new in [
 ('alu_mix_build/alu_mix.cubin','alu_mix_v2_build/alu_mix.cubin'),
 ('C.c_uint32(m),C.c_uint32(1)', 'C.c_uint32(m),C.c_uint32(1),C.c_uint32(3),C.c_uint32(0)'),
 ('(C.c_void_p*4)', '(C.c_void_p*6)'),
 ('496+32*n', '496+128*n'),
 ('clock_cycles=float(statistics.median(cy))', 'clock_cycles=float(statistics.median(cy[:128] if m==1 else cy[128:] if m==2 else cy))'),
 ('global_ns=float(statistics.median(ns))', 'global_ns=float(statistics.median(ns[:128] if m==1 else ns[128:] if m==2 else ns)),warp_cycles=json.dumps([float(statistics.median(cy[i:i+32])) for i in range(0,256,32)]),warp_global_ns=json.dumps([float(statistics.median(ns[i:i+32])) for i in range(0,256,32)])'),
]:
 assert old in t,old;t=t.replace(old,new)
exec(compile(t,'run_alu_mix_v3_generated.py','exec'))
