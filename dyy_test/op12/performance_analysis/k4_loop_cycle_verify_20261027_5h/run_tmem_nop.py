from pathlib import Path
import argparse,json,csv,re,statistics,hashlib,subprocess,os
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--kind',type=int,required=True);p.add_argument('--warps',type=int,required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--points',default='512,1024,2048,4096');p.add_argument('--samples',type=int,default=30);p.add_argument('--trial',type=int,default=0);p.add_argument('--profile',action='store_true');p.add_argument('--modes',default='0,1')
a=p.parse_args();assert os.environ.get('K4_5H_ADMITTED')=='1';a.out.mkdir(parents=True,exist_ok=False)
import torch,cutlass
import cutlass.cute as cute
from cutlass.base_dsl.compiler import DumpDir
from cuda.bindings import driver as cuda
from tmem_dep_nop import TmemReadDepNop
from cubin_nop_driver import patch,Driver,sha
torch.set_num_threads(1);stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());torch.cuda.synchronize()
ck=torch.empty(128,device='cuda',dtype=torch.int32);tr=torch.empty(6,device='cuda',dtype=torch.int64)
cv=lambda x:cute.runtime.from_dlpack(x,assumed_align=16,enable_tvm_ffi=True)
args=[cv(ck),cv(tr)];cs=cuda.CUstream(stream.cuda_stream);d=a.out/'build';d.mkdir();obj=TmemReadDepNop(a.kind,a.warps)
f=cute.compile[(cute.EnableTVMFFI(True),cute.GPUArch('sm_100a'),cute.KeepPTX(True),cute.KeepCUBIN(True),cute.KeepSASS(True),DumpDir(str(d)))](obj,*args,cutlass.Int32(128),cutlass.Int32(1),cs)
cub=next(d.glob('*.cubin'));ss=next(d.glob('*.sass'));ptx=next(d.glob('*.ptx')).read_text()
assert not re.search(r'ld\.param[^;]*_param_0',ptx),'TiledMMA runtime metadata unexpectedly used; cannot zero driver arg0'
assert not re.search(r'\b(?:LDL|STL)(?:\.|\s)',ss.read_text())
meta=patch(cub,ss,d/'NOP.cubin');driver=Driver();func={1:driver.load(cub,meta['function']),0:driver.load(d/'NOP.cubin',meta['function'])}
# One original-Cute launch proves parameter/initialization identity before
# the exact same CUBIN is dispatched through the explicit driver ABI.
with torch.cuda.stream(stream):f(*args,128,1,cs)
stream.synchronize();assert bool(torch.all(ck==0));before=tr.cpu().tolist()
driver.launch(func[1],ck.data_ptr(),tr.data_ptr(),128,stream.cuda_stream);stream.synchronize();assert bool(torch.all(ck==0));after=tr.cpu().tolist()
assert all(q[0]<q[1] and q[2]<q[3] and q[4]==q[5] for q in [before,after])
iden=dict(kind=a.kind,warps=a.warps,grid=1,threads=128,original_sha256=sha(cub),nop_sha256=sha(d/'NOP.cubin'),
    ABI={str(k):v['params'] for k,v in func.items()},Cute_and_driver_original_verified=True,patch=meta,
    description='Same final machine code, only 16 LDTM instructions replaced by NOP and their WB/RB disabled. Nonload instruction bytes, registers and schedule unchanged. Address data all zero in both modes.',model_enabled=False)
(a.out/'IDENTITY.json').write_text(json.dumps(iden,indent=2));rows=[]
for n in map(int,a.points.split(',')):
 modes=list(map(int,a.modes.split(',')))
 for _ in range(20):
  for m in modes:driver.launch(func[m],ck.data_ptr(),tr.data_ptr(),n,stream.cuda_stream)
 stream.synchronize()
 for i in range(a.samples):
  for m in (modes if i%2==0 else list(reversed(modes))):
   e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   if a.profile:torch.cuda.profiler.start()
   with torch.cuda.stream(stream):
    e0.record();driver.launch(func[m],ck.data_ptr(),tr.data_ptr(),n,stream.cuda_stream);e1.record()
   e1.synchronize()
   if a.profile:torch.cuda.profiler.stop()
   q=tr.cpu().tolist();assert bool(torch.all(ck==0)) and q[0]<q[1] and q[4]==q[5]
   rows.append(dict(kind=a.kind,warps=a.warps,mode=m,trial=a.trial,N=n,sample=i,reads=16*n,
      clock_cycles=q[1]-q[0],global_ns=q[3]-q[2],event_us=e0.elapsed_time(e1)*1000,SMID=q[4]))
 with (a.out/'RAW.csv').open('w',newline='') as z:w=csv.DictWriter(z,rows[0]);w.writeheader();w.writerows(rows)
(a.out/'COMPLETE.json').write_text(json.dumps(dict(correctness=True,rows=len(rows),model_enabled=False),indent=2));print('COMPLETE',a.kind,a.warps,a.trial,flush=True)
