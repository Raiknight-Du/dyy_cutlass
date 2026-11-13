from pathlib import Path
import argparse,sys,os,json,csv,subprocess,re,statistics,hashlib
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--kind',type=int,choices=[0,1],required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--points',default='512,1024,2048,4096');p.add_argument('--trial',type=int,default=0);p.add_argument('--samples',type=int,default=30);p.add_argument('--profile',action='store_true');p.add_argument('--warps',type=int,choices=[1,4],default=4);p.add_argument('--modes',default='0,1')
a=p.parse_args();assert os.environ.get('K4_5H_ADMITTED')=='1';a.out.mkdir(parents=True,exist_ok=False)
import torch,cutlass
import cutlass.cute as cute
from cutlass.base_dsl.compiler import DumpDir
from cuda.bindings import driver as cuda
from tmem_dep_pair import TmemReadDepPair
torch.set_num_threads(1);stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());torch.cuda.synchronize()
ck=torch.empty((128,),device='cuda',dtype=torch.int32);tr=torch.empty((6,),device='cuda',dtype=torch.int64)
cv=lambda t:cute.runtime.from_dlpack(t,assumed_align=16,enable_tvm_ffi=True)
args=[cv(ck),cv(tr)];cs=cuda.CUstream(stream.cuda_stream);d=a.out/'build';d.mkdir()
obj=TmemReadDepPair(a.kind,a.warps)
f=cute.compile[(cute.EnableTVMFFI(True),cute.GPUArch('sm_100a'),cute.KeepPTX(True),cute.KeepCUBIN(True),cute.KeepSASS(True),DumpDir(str(d)))](obj,*args,cutlass.Int32(128),cutlass.Int32(1),cs)
sass=next(d.glob('*.sass'));s=sass.read_text();res=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(next(d.glob('*.cubin')))],text=True)
assert not re.search(r'\b(?:LDL|STL)(?:\.|\s)',s),('SPILL',res)
(a.out/'IDENTITY.json').write_text(json.dumps(dict(kind=a.kind,warps=a.warps,grid=1,threads=128,elementary_reads_per_N=16,
 raw_endpoint='4warp dependentTMEMread + AND + ADD address feedback; control amortized 1/16. Not bare load latency.',
 resource=res,cubin_sha256=hashlib.sha256(next(d.glob('*.cubin')).read_bytes()).hexdigest(),model_enabled=False),indent=2))
def call(n,m):
 with torch.cuda.stream(stream):f(*args,n,m,cs)
rows=[]
for n in map(int,a.points.split(',')):
 modes=list(map(int,a.modes.split(',')))
 for _ in range(20):
  for m in modes:call(n,m)
 stream.synchronize()
 for i in range(a.samples):
  for m in (modes if i%2==0 else list(reversed(modes))):
   e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   if a.profile:torch.cuda.profiler.start()
   with torch.cuda.stream(stream):e0.record();f(*args,n,m,cs);e1.record()
   e1.synchronize()
   if a.profile:torch.cuda.profiler.stop()
   q=tr.cpu().tolist();assert bool(torch.all(ck==0)),ck.cpu().tolist();assert q[0]<q[1] and q[2]<q[3] and q[4]==q[5],q
   rows.append(dict(kind=a.kind,warps=a.warps,mode=m,trial=a.trial,N=n,sample=i,reads=16*n,clock_cycles=q[1]-q[0],global_ns=q[3]-q[2],event_us=e0.elapsed_time(e1)*1000,SMID=q[4]))
 with (a.out/'RAW.csv').open('w',newline='') as z:
  w=csv.DictWriter(z,rows[0]);w.writeheader();w.writerows(rows)
(a.out/'COMPLETE.json').write_text(json.dumps(dict(correctness=True,rows=len(rows),model_enabled=False),indent=2))
print('COMPLETE',a.kind,a.trial,flush=True)
