"""Read binding implementation/header and dlsym only. No cuInit or graph calls."""
from pathlib import Path
import ctypes,hashlib,importlib.metadata,json,os,re,subprocess
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from cuda.bindings import driver as cuda
O=Path(__file__).resolve().parent;out=O/'graph_binding_cpu_audit01';out.mkdir(exist_ok=False)
assert not torch.cuda.is_initialized()
class EdgeData(ctypes.Structure):
    _fields_=[('from_port',ctypes.c_ubyte),('to_port',ctypes.c_ubyte),('type',ctypes.c_ubyte),('reserved',ctypes.c_ubyte*5)]
lib=ctypes.CDLL('libcuda.so.1');exports={n:hasattr(lib,n) for n in ['cuGraphGetEdges','cuGraphGetEdges_v2','cuGetProcAddress']}
fn=lib.cuGraphGetEdges_v2;fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(EdgeData),ctypes.POINTER(ctypes.c_size_t)];fn.restype=ctypes.c_int
header=Path('/usr/local/cuda-13.0/include/cuda.h');h=header.read_text();(out/'CUDA_HEADER_GRAPH_EXCERPTS.txt').write_text('\n\n'.join(h[max(0,m.start()-250):m.start()+1000] for m in re.finditer(r'(?:typedef struct CUgraphEdgeData_st|cuGraphGetEdges(?:_v2)?\()',h)))
package=Path(cuda.__file__).parent
paths=list(package.rglob('*driver*.pyx'))+list(package.rglob('*driver*.pxd'))
found=[]
for p in paths:
    text=p.read_text(errors='replace')
    for m in re.finditer(r'cuGraphGetEdges',text):found.append(dict(path=str(p),excerpt=text[max(0,m.start()-300):m.start()+3500]))
(out/'BINDING_IMPLEMENTATION_EXCERPTS.json').write_text(json.dumps(found,indent=2))
info=dict(cuda_bindings_version=importlib.metadata.version('cuda-bindings'),driver_extension=str(cuda.__file__),
 driver_extension_SHA256=hashlib.sha256(Path(cuda.__file__).read_bytes()).hexdigest(),libcuda_exports=exports,
 ctypes_sizeof_EdgeData=ctypes.sizeof(EdgeData),ctypes_field_offsets={n:getattr(EdgeData,n).offset for n,_ in EdgeData._fields_},
 prototype='CUresult cuGraphGetEdges_v2(CUgraph,CUgraphNode*,CUgraphNode*,CUgraphEdgeData*,size_t*)',
 dlsym_only_no_driver_function_called=True,header_path=str(header),header_SHA256=hashlib.sha256(header.read_bytes()).hexdigest(),
 binding_source_files=[str(x) for x in paths],matching_binding_source_excerpts=len(found),API_doc=cuda.cuGraphGetEdges.__doc__,
 context_initialized_after=torch.cuda.is_initialized(),model_enabled=False)
assert info['ctypes_sizeof_EdgeData']==8 and not info['context_initialized_after']
(out/'CPU_ABI_AUDIT.json').write_text(json.dumps(info,indent=2));print(json.dumps({k:v for k,v in info.items() if k not in ['API_doc','binding_source_files']}))
