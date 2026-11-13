"""Generate auditable plain-NA inline PTX. CPU only; never compiles or runs."""
from pathlib import Path
import hashlib,json
O=Path(__file__).resolve().parent

def assembly(vector):
    q=['{','.reg .u32 d<128>, i, slot, off, sum, grid;',
       '.reg .u64 addr, offset;','.reg .pred loadp, more, bad, zero;',
       'mov.u32 grid, %%nctaid.x;','setp.eq.u32 zero, %8, 0;','setp.ne.u32 loadp, %7, 0;']
    q += [f'mov.u32 d{i}, 0;' for i in range(128)]
    q += ['mov.u64 %1, 0;','mov.u64 %2, 0;','mov.u32 %3, 0;','mov.u32 %4, 0;',
          '@zero mov.u32 %3, %%smid;','@zero mov.u64 %1, %%clock64;',
          'bar.sync 0;','mov.u32 i, 0;','R3_LOOP:',
          'and.b32 slot, i, 7;','mad.lo.u32 off, slot, grid, %9;',
          'mul.lo.u32 off, off, 65536;',
          'mad.lo.u32 off, %8, 512, off;',
          'cvt.u64.u32 offset, off;','add.u64 addr, %5, offset;',
          '@!loadp bra R3_SKIP;']
    if vector:
        q += [f'ld.global.L1::no_allocate.v4.b32 {{d{i},d{i+1},d{i+2},d{i+3}}}, [addr+{4*i}];'
              for i in range(0,128,4)]
    else:
        q += [f'ld.global.L1::no_allocate.b32 d{i}, [addr+{4*i}];' for i in range(128)]
    q += ['R3_SKIP:','add.u32 i, i, 1;','setp.lt.u32 more, i, %6;','@more bra R3_LOOP;',
          'mov.u32 sum, d0;']
    q += [f'xor.b32 sum, sum, d{i};' for i in range(1,128)]
    q += ['setp.eq.u32 bad, sum, 0xdeadbeef;','@bad trap;',
          'bar.sync 0;','@zero mov.u64 %2, %%clock64;',
          '@zero mov.u32 %4, %%smid;','mov.u32 %0, sum;','}']
    return '\n'.join(q)

template=(O/'ldg_service_template.cu.in').read_text(encoding='utf-8')
for name,vector in [('SCALAR',False),('VECTOR',True)]:
    template=template.replace('@ASM_'+name+'@',assembly(vector))
out=O/'ldg_service.cu';out.write_text(template,encoding='utf-8')
(O/'GENERATION_RECORD.json').write_text(json.dumps({'output':out.name,
    'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
    'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'template_sha256':hashlib.sha256((O/'ldg_service_template.cu.in').read_bytes()).hexdigest(),
    'PTX_ld_volatile':False,'CUDA_CXX_asm_volatile':True,'model_enabled':False},indent=2),encoding='utf-8')
print(out)
