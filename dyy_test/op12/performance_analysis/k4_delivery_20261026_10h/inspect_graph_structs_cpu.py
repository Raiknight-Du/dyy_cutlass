import os,json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from cuda.bindings import driver as cuda
assert not torch.cuda.is_initialized()
edge=cuda.CUgraphEdgeData();params=cuda.CUDA_KERNEL_NODE_PARAMS()
out=dict(context_initialized_before=False,graph_handle_roundtrip=int(cuda.CUgraph(1)),
  edge_type_field_available=hasattr(edge,'type'),edge_type_default=int(edge.type),
  kernel_params_fields={n:hasattr(params,n) for n in ['func','gridDimX','gridDimY','gridDimZ','blockDimX','blockDimY','blockDimZ','sharedMemBytes']},
  context_initialized_after=torch.cuda.is_initialized())
assert out['graph_handle_roundtrip']==1 and all(out['kernel_params_fields'].values()) and not out['context_initialized_after']
(Path(__file__).resolve().parent/'graph_api_cpu_audit03/STRUCT_PROBE.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
