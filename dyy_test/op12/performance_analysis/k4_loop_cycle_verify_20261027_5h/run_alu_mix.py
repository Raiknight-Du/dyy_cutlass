from pathlib import Path
import ctypes as C,argparse,json,csv,statistics,os
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--points',default='1024,2048,4096,8192');p.add_argument('--samples',type=int,default=30);p.add_argument('--trial',type=int,default=0);p.add_argument('--modes',default='1,2,3');p.add_argument('--profile',action='store_true');a=p.parse_args()
assert os.environ.get('K4_5H_ADMITTED')=='1';a.out.mkdir(exist_ok=False,parents=True)
import torch
torch.set_num_threads(1);out=torch.empty(256,8,device='cuda',dtype=torch.int32);stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());torch.cuda.synchronize()
L=C.CDLL('libcuda.so.1');L.cuModuleLoadData.argtypes=[C.POINTER(C.c_void_p),C.c_void_p];L.cuModuleGetFunction.argtypes=[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p];L.cuLaunchKernel.argtypes=[C.c_void_p]+[C.c_uint]*7+[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
def ok(v):
 if v:raise RuntimeError(('CUDA',v))
buf=C.create_string_buffer((R/'alu_mix_build/alu_mix.cubin').read_bytes());mod=C.c_void_p();fn=C.c_void_p();ok(L.cuModuleLoadData(C.byref(mod),C.cast(buf,C.c_void_p)));ok(L.cuModuleGetFunction(C.byref(fn),mod,b'alu_mix'))
def launch(n,m):
 vals=[C.c_uint32(n),C.c_uint64(out.data_ptr()),C.c_uint32(m),C.c_uint32(1)];pp=(C.c_void_p*4)(*[C.addressof(v) for v in vals]);ok(L.cuLaunchKernel(fn,1,1,1,256,1,1,0,C.c_void_p(stream.cuda_stream),pp,None))
rows=[];modes=list(map(int,a.modes.split(',')))
for n in map(int,a.points.split(',')):
 for _ in range(20):
  for m in modes:launch(n,m)
 stream.synchronize()
 for sample in range(a.samples):
  for m in (modes if sample%2==0 else list(reversed(modes))):
   if a.profile:torch.cuda.profiler.start()
   e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   with torch.cuda.stream(stream):e0.record();launch(n,m);e1.record()
   e1.synchronize()
   if a.profile:torch.cuda.profiler.stop()
   q=out.cpu().numpy();cy=q[:,:2].copy().view('uint64').ravel();ns=q[:,2:4].copy().view('uint64').ravel();check=q[:,5].view('uint32')
   assert len(set(q[:,4]))==1
   for tid,v in enumerate(check):
    expected=0 if tid<128 and m&1 else (496+32*n if tid>=128 and m&2 else 496)
    assert int(v)==expected,('OUTPUT',tid,m,n,int(v),expected)
   rows.append(dict(kind=0,mode=m,trial=a.trial,N=n,sample=sample,clock_cycles=float(statistics.median(cy)),event_us=e0.elapsed_time(e1)*1000,global_ns=float(statistics.median(ns)),SMID=int(q[0,4]),valid=1))
 with (a.out/'RAW.csv').open('w',newline='') as f:w=csv.DictWriter(f,rows[0]);w.writeheader();w.writerows(rows)
(a.out/'COMPLETE.json').write_text(json.dumps(dict(rows=len(rows),correctness=True,model_enabled=False),indent=2));print('COMPLETE',a.trial,flush=True)
