"""Compile metadata only. Optional authorized CUDA allocation; no invocation."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('--out',type=Path,required=True)
p.add_argument('--layout',choices=['S','V'],default='S')
p.add_argument('--shape',type=int,choices=[8192,16384,32768],default=32768)
p.add_argument('--metadata-device',choices=['cpu','cuda'],default='cpu')
p.add_argument('--backend',choices=['returns','init'],default='returns')
p.add_argument('--compile-cases',default='B0,CLOCK,FRAME')
a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
OP=Path('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12')
REPO=OP/'upstreams/flashinfer'
ROOT=Path(__file__).resolve().parent
assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()=='e77a4a0d276367895c3b50a642fd8f326c03fb72'
sys.path.insert(0,str(REPO))
os.environ['FLASHINFER_WORKSPACE_BASE']=str(ROOT/'cpu_jit_cache')
import torch
import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass.base_dsl.compiler import DumpDir
from flashinfer.gdn_kernels.blackwell.gdn_cp_prefill import _get_prefill_kernel
from flashinfer.gdn_kernels.blackwell.gdn_prefill import _mark_state_layout

def cv(t,align=16,leading=None,dynamic=True):
    if a.metadata_device=='cuda':
        # No value copy or initialization, and never invoke a CUDA kernel.
        t=torch.empty_strided(t.shape,t.stride(),dtype=t.dtype,device='cuda')
        allocations.append(t)
    x=cute.runtime.from_dlpack(t,assumed_align=align,enable_tvm_ffi=True)
    return x.mark_layout_dynamic(leading_dim=leading) if dynamic else x

S=a.shape
allocations=[]
if a.metadata_device=='cuda':
    assert os.environ.get('K1_GPU_SESSION') and not (Path(os.environ['K1_GUARD_DIR'])/'INVALID_COMPETITION').exists()
data=torch.load(OP/f'performance_analysis/cp4_event_baseline_freeze_20260929/frozen_inputs/S{S}/inputs.pt',map_location='cpu',weights_only=True)
b=data['buffers'][0]
q,k,v=[b[n] for n in ['q','k','v']]
gate=torch.empty((S,16),dtype=torch.float32)
t=torch.empty((S//64,16,64,64),dtype=torch.float16).as_strided((64,64,16,S//64),(64,1,4096,16*4096))
output=torch.empty((S,16,128),dtype=torch.float16)
fixed=torch.empty((8,16,128,128),dtype=torch.float32)
initial=torch.empty((1,16,128,128),dtype=torch.float32)
state=torch.empty_like(initial)
tmaps=torch.empty((1*16*8*5*128,),dtype=torch.uint8)
trace=torch.empty((12288,),dtype=torch.int64)
cu=data['shared']['cu_seqlens']
ini,st=cv(initial,dynamic=False),cv(state,dynamic=False)
_mark_state_layout(ini,False,128)
_mark_state_layout(st,False,128)
fx=cv(fixed)
if a.layout=='V':
    fx.mark_compact_shape_dynamic(mode=3,stride_order=(0,1,2,3),divisibility=128)
args=[cv(q),cv(k),cv(v),cv(gate,leading=1),cv(t,leading=1),cv(output),cv(cu,align=8),fx,ini,None,
      cutlass.Int32(S//8),cutlass.Int32(8),cutlass.Int32(8),cutlass.Int32(1),cutlass.Float32(1/(128**.5)),cv(tmaps,align=128),cuda.CUstream(0)]
base=_get_prefill_kernel(cutlass.Float16,cutlass.Float32,148,True,1,True,False,cutlass.Int64 if cu.dtype==torch.int64 else cutlass.Int32)
manifest=dict(status='METADATA_COMPILE_STARTED',layout=a.layout,shape=S,gpu_kernel_invoked=False,metadata_device=a.metadata_device,K4_store_final_state=False,note='Public CP invokes K4 state=None; K3 already writes requested output state',tensors={n:dict(shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),source_device=str(x.device)) for n,x in [('q',q),('k',k),('v',v),('gate',gate),('t',t),('fixed',fixed),('initial',initial),('cu',cu)]})
(a.out/'IDENTITY.json').write_text(json.dumps(manifest,indent=2))

def load(filename,classname):
    spec=importlib.util.spec_from_file_location(classname,ROOT/filename)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    obj=getattr(mod,classname).__new__(getattr(mod,classname));obj.__dict__.update(base.__dict__)
    return obj

clock_file,clock_class,frame_file,frame_class=('PTR_CLOCK_RETURNS.py','K4CLOCK_RETURNS','PTR_READY_GATE.py','K4READY_GATE') if a.backend=='returns' else ('PTR_CLOCK_INIT.py','K4CLOCK_INIT','PTR_READY_GATE_INIT.py','K4READY_GATE_INIT')
cases=[('B0',base,args),('CLOCK',load(clock_file,clock_class),args[:-1]+[cv(trace,dynamic=False),args[-1]]),
       ('FRAME',load(frame_file,frame_class),args[:-1]+[cv(trace,dynamic=False),cutlass.Int32(0),args[-1]])]
requested=a.compile_cases.split(',');assert set(requested)<=set(['B0','CLOCK','FRAME'])
cases=[x for x in cases if x[0] in requested]
audit={}
for name,obj,ca in cases:
    dst=a.out/name;dst.mkdir()
    opts=(cute.EnableTVMFFI(True),cute.GPUArch('sm_100a'),cute.KeepPTX(True),cute.KeepCUBIN(True),cute.KeepSASS(True),DumpDir(str(dst)))
    fn=cute.compile[opts](obj,*ca)
    # Deliberately never call fn.
    cub=next(dst.glob('*.cubin'));sass=next(dst.glob('*.sass')).read_text()
    resource=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(cub)],text=True)
    instructions=[]
    for line in sass.splitlines():
        m=re.match(r'\s*/\*([0-9a-f]+)\*/\s+(.*?);',line)
        if m:
            txt=m[2].strip();op=re.sub(r'^@!?\w+\s+','',txt).split()[0]
            instructions.append(dict(pc=m[1],text=txt,op=op))
    counts={op:sum(x['op'].split('.')[0]==op for x in instructions) for op in ['LDL','STL','LDG','STG','LDTM','STTM','UTCHMMA','UTMALDG','UTMASTG','BAR','SYNCS','FENCE','MEMBAR','UTCBAR']}
    audit[name]=dict(cubin_sha256=hashlib.sha256(cub.read_bytes()).hexdigest(),resource=resource,counts=counts,instructions=instructions)
    (a.out/'COMPILE_AUDIT.json').write_text(json.dumps(audit,indent=2))
    print(name,resource,counts,flush=True)
    if name=='B0' and a.layout=='S' and a.shape==32768:
        assert audit[name]['cubin_sha256']=='06ebd78963371a30adac142c06b86950ec300b6cb762a209c5afc67628941814','Metadata/specialization must first reproduce frozen original binary'
manifest.update(status='CPU_METADATA_COMPILED_PENDING_STATIC_REVIEW',gpu_kernel_invoked=False,model_enabled=False)
(a.out/'IDENTITY.json').write_text(json.dumps(manifest,indent=2))
