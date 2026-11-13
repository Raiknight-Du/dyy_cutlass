"""Only inspect installed host API declarations, with CUDA hidden."""
from pathlib import Path
import inspect,json,os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from cuda.bindings import driver as cuda
O=Path(__file__).resolve().parent;out=O/'graph_api_cpu_audit03';out.mkdir(exist_ok=False)
assert not torch.cuda.is_initialized()
records={'torch':torch.__version__,'torch_cuda':torch.version.cuda,'context_initialized_before':torch.cuda.is_initialized(),'APIs':{}}
records['Event_class_doc']=torch.cuda.Event.__doc__
for name,obj in [('Event.__new__',torch.cuda.Event.__new__),('Event.__init__',torch.cuda.Event.__init__),('CUDAGraph.__init__',torch.cuda.CUDAGraph.__init__),('CUDAGraph.raw_cuda_graph',torch.cuda.CUDAGraph.raw_cuda_graph),('CUDAGraph.debug_dump',torch.cuda.CUDAGraph.debug_dump),('CUDAGraph.instantiate',torch.cuda.CUDAGraph.instantiate)]:
    try:source=inspect.getsource(obj)
    except (OSError,TypeError):source='UNAVAILABLE'
    try:signature=str(inspect.signature(obj))
    except (ValueError,TypeError):signature='UNAVAILABLE'
    records['APIs'][name]={'signature':signature,'doc':obj.__doc__,'source':source}
for name in ['cuGraphGetNodes','cuGraphGetNodeType','cuGraphNodeGetType','cuGraphGetEdges','cuGraphEventRecordNodeGetEvent','cuGraphKernelNodeGetParams','cuFuncGetName','cuFuncGetAttribute']:
    obj=getattr(cuda,name,None)
    records['APIs'][name]={'available':obj is not None,'doc':obj.__doc__ if obj else None}
records['graph_node_types']={name:int(getattr(cuda.CUgraphNodeType,name)) for name in ['CU_GRAPH_NODE_TYPE_KERNEL','CU_GRAPH_NODE_TYPE_EVENT_RECORD']}
event=torch.cuda.Event(enable_timing=True,external=True)
records['lazy_Event_constructor_external_keyword_accepted']=True
records['lazy_Event_device']=str(event.device)
records['context_initialized_after']=torch.cuda.is_initialized();assert not records['context_initialized_after']
(out/'HOST_API_AUDIT.json').write_text(json.dumps(records,indent=2));print(json.dumps({k:v for k,v in records.items() if k!='APIs'}))
