from pathlib import Path
import ctypes as C,struct,re,hashlib,json,subprocess
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def patch(cub,sass,out):
 b=bytearray(cub.read_bytes());off=struct.unpack_from('<Q',b,40)[0];sz,nn,si=struct.unpack_from('<HHH',b,58)
 hh=[struct.unpack_from('<IIQQQQIIQQ',b,off+i*sz) for i in range(nn)];h=hh[si];names=b[h[4]:h[4]+h[5]]
 sections=[(names[z[0]:].split(b'\0')[0].decode(),z) for z in hh]
 textname,t=next((n,z) for n,z in sections if n.startswith('.text.'))
 pp=[]
 for ln in sass.read_text().splitlines():
  m=re.search(r'/\*([0-9a-f]+)\*/\s+(?:(@!?\w+)\s+)?(\w[\w.]*)\s*(.*?)\s*;',ln)
  if m:pp.append((int(m[1],16),m[3],m[4]))
 pcs=[pc for pc,op,_ in pp if op.startswith('LDTM')];assert len(pcs)==16,pcs
 nop_pc=next(pc for pc,op,_ in pp if op=='NOP');nop=int.from_bytes(b[t[4]+nop_pc:t[4]+nop_pc+16],'little')
 mask=((1<<105)-1);manifest=[];old=bytes(b)
 for pc in pcs:
  at=t[4]+pc;v=int.from_bytes(b[at:at+16],'little');n=(v&~mask)|(nop&mask)
  # NOP creates no memory scoreboard producer. Preserve all original
  # stalls, input wait masks and control bits outside WB/RB.
  n=(n&~(7<<110)&~(7<<113))|(7<<110)|(7<<113)
  b[at:at+16]=n.to_bytes(16,'little');manifest.append(dict(pc=hex(pc),original=f'{v:032x}',patched=f'{n:032x}'))
 changed=[i for i,(x,y) in enumerate(zip(old,b)) if x!=y]
 assert all(any(t[4]+pc<=i<t[4]+pc+16 for pc in pcs) for i in changed)
 out.write_bytes(b)
 ss=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(out)],text=True)
 assert not re.search(r'\bLDTM',ss);out.with_suffix('.sass').write_text(ss)
 info=dict(original=sha(cub),patched=sha(out),function=textname[6:],targets=manifest,only_target_instruction_words_modified=True,model_enabled=False)
 out.with_suffix('.patch.json').write_text(json.dumps(info,indent=2));return info

class Driver:
 def __init__(self):
  self.lib=C.CDLL('libcuda.so.1');L=self.lib
  L.cuModuleLoadData.argtypes=[C.POINTER(C.c_void_p),C.c_void_p];L.cuModuleGetFunction.argtypes=[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p]
  L.cuFuncGetParamInfo.argtypes=[C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)]
  L.cuLaunchKernel.argtypes=[C.c_void_p]+[C.c_uint]*7+[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
  self.buffers=[]
 def ok(self,v):
  if v:raise RuntimeError(('CUDA_DRIVER_ERROR',v))
 def load(self,path,symbol):
  buf=C.create_string_buffer(Path(path).read_bytes());self.buffers.append(buf);mod=C.c_void_p();self.ok(self.lib.cuModuleLoadData(C.byref(mod),C.cast(buf,C.c_void_p)))
  fn=C.c_void_p();self.ok(self.lib.cuModuleGetFunction(C.byref(fn),mod,symbol.encode()))
  params=[]
  for i in range(8):
   o,s=C.c_size_t(),C.c_size_t();e=self.lib.cuFuncGetParamInfo(fn,i,C.byref(o),C.byref(s))
   if e:break
   params.append((o.value,s.value))
  assert [s for o,s in params] in [[32,8,8,4],[32,8,8,4,4]],params
  return dict(module=mod,function=fn,params=params)
 def launch(self,k,checks,trace,n,stream):
  vals=[C.create_string_buffer(32),C.c_uint64(checks),C.c_uint64(trace),C.c_int32(n)]
  if len(k['params'])==5:vals.append(C.c_int32(1))
  ptr=(C.c_void_p*len(vals))(*[C.addressof(x) for x in vals])
  self.ok(self.lib.cuLaunchKernel(k['function'],1,1,1,128,1,1,4,C.c_void_p(stream),ptr,None))
