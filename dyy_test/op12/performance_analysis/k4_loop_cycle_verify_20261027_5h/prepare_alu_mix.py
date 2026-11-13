from pathlib import Path
import subprocess,json,re,hashlib
R=Path(__file__).resolve().parent;D=R/'alu_mix_build';D.mkdir(exist_ok=False)
s=['.version 9.0','.target sm_100a','.address_size 64',
 '.visible .entry alu_mix(.param .u32 n,.param .u64 output,.param .u32 mode,.param .u32 salt)',
 '.maxntid 256,1,1','{','.reg .b32 t,w,rem,md,z,sm,check,bits,p<32>;','.reg .f32 f<32>,b;',
 '.reg .b64 out,addr,c0,c1,g0,g1,cy,ns;','.reg .pred low,on,loop;',
 'mov.u32 t,%tid.x;','shr.u32 w,t,5;','mov.u32 sm,%smid;',
 'ld.param.u32 rem,[n];','ld.param.u64 out,[output];','ld.param.u32 md,[mode];','ld.param.u32 z,[salt];',
 'mov.b32 b,0;','mov.b32 bits,0x40000000;']
for i in range(32):s += [f'mov.b32 f{i},bits;',f'mov.b32 p{i},{i};']
s += ['bar.sync 0;','mov.u64 g0,%globaltimer;','mov.u64 c0,%clock64;',
 'setp.lt.u32 low,w,4;','@!low bra INT_BEGIN;',
 'and.b32 check,md,1;','setp.ne.u32 on,check,0;','@!on bra DONE;', 'PACK_LOOP:']
for _ in range(8):
 for i in range(32):s += [f'cvt.rn.f16x2.f32 p{i},f{i},b;',f'mov.b32 f{i},p{i};']
s += ['add.u32 rem,rem,-8;','setp.gt.u32 loop,rem,0;','@loop bra PACK_LOOP;','bra DONE;',
 'INT_BEGIN:','and.b32 check,md,2;','setp.ne.u32 on,check,0;','@!on bra DONE;','INT_LOOP:']
for _ in range(8):
 for i in range(32):s += [f'add.s32 p{i},p{i},z;']
s += ['add.u32 rem,rem,-8;','setp.gt.u32 loop,rem,0;','@loop bra INT_LOOP;',
 'DONE:','bar.sync 0;','mov.u64 c1,%clock64;','mov.u64 g1,%globaltimer;','sub.u64 cy,c1,c0;','sub.u64 ns,g1,g0;',
 'mov.u32 check,0;']
for i in range(32):s += [f'add.u32 check,check,p{i};']
s += ['mul.wide.u32 addr,t,32;','add.u64 addr,out,addr;','st.global.u64 [addr],cy;','st.global.u64 [addr+8],ns;',
 'st.global.u32 [addr+16],sm;','st.global.u32 [addr+20],check;','ret;','}']
ptx=D/'alu_mix.ptx';ptx.write_text('\n'.join(s));cub=D/'alu_mix.cubin'
cmd=['/usr/local/cuda/bin/ptxas','-O3','-arch=sm_100a','-v',str(ptx),'-o',str(cub)]
p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(D/'BUILD.log').write_text(p.stdout);assert p.returncode==0,p.stdout
ss=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(cub)],text=True);(D/'alu_mix.sass').write_text(ss)
pack=len(re.findall(r'\bF2FP\.F16\.F32\.PACK_AB\b',ss));assert pack==256,pack
assert not re.search(r'\b(?:LDL|STL)(?:\.|\s)',ss)
meta=dict(cubin_sha256=hashlib.sha256(cub.read_bytes()).hexdigest(),pack_static=pack,
 int_requires_native_review=True,prediction='perN PACK4096 native lane + IADD4096 native lane; shared64lane/cy predicts128cy/N; independent resources predict64cy/N, plus bounded loop control.',
 model_enabled=False)
(D/'MANIFEST.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
