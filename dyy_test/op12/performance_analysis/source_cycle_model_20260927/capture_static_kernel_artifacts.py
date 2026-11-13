"""Compile pinned CP path and save PTX/CUBIN. No target-kernel timing."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
root = Path(a.source).resolve()
out = Path(a.output).resolve()
out.mkdir(parents=True, exist_ok=True)
(out / 'binary').mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root))
os.environ['FLASHINFER_WORKSPACE_BASE'] = str(out / 'cache')
import torch
import cutlass.cute as cute
from cutlass.base_dsl.compiler import DumpDir
import flashinfer
from flashinfer.gdn_kernels.blackwell import gdn_cp_prefill as cp
assert Path(flashinfer.__file__).resolve().is_relative_to(root)
orig = cp._blackwell_compile_options
cp._blackwell_compile_options = lambda device: tuple(orig(device)) + (cute.KeepPTX(True), cute.KeepCUBIN(True), cute.KeepSASS(True), DumpDir(str(out / 'binary')))
S,H,D,C=8192,16,128,1024
q=torch.zeros(S,H,D,device='cuda',dtype=torch.float16)
k=torch.zeros_like(q)
v=torch.zeros_like(q)
o=torch.empty_like(q)
gate=torch.full((S,H),0.99,device='cuda',dtype=torch.float32)
beta=torch.full((S,H),0.5,device='cuda',dtype=torch.float32)
cu=torch.tensor([0,S],device='cuda',dtype=torch.int32)
initial=torch.zeros(1,H,D,D,device='cuda',dtype=torch.float32)
final=torch.empty_like(initial)
cp.cp_delta_rule_dsl_sm100(o,final,q,k,v,gate,beta,cu,1/(D**0.5),initial_state=initial,max_seqlen=S,cp_chunk_len=C)
torch.cuda.synchronize()
manifest={
 'import':str(flashinfer.__file__),
 'source_files':{},
 'device':torch.cuda.get_device_name(),
 'sm_count':torch.cuda.get_device_properties(0).multi_processor_count,
 'shape':{'S':S,'H':H,'D':D,'C':C},
 'dtype':'FP16 with FP32 initial/final state',
 'target_kernel_timings_collected':False,
}
for filename in ('gdn_cp_prefill.py','gated_delta_net_cp.py','gated_delta_net_cp_prefill.py'):
 source=root/'flashinfer/gdn_kernels/blackwell'/filename
 manifest['source_files'][filename]=hashlib.sha256(source.read_bytes()).hexdigest()
(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
