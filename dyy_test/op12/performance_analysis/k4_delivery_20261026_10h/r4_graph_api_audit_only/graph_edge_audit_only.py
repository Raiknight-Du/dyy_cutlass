"""Only capture/query one graph; intentionally cannot instantiate or replay."""
from pathlib import Path
import json
from graph_edge_native_audit import compare_live_graph

class AuditOnlyCompleted(Exception):pass
def checked(result):
    assert int(result[0])==0,result
    return result[1:]

class AuditOnlyCapture:
    def __init__(self,launch,stream,out,expected_name,expected_grid,expected_block,expected_smem,expected_regs):
        import torch
        from cuda.bindings import driver as cuda
        begin=torch.cuda.Event(enable_timing=True,external=True);end=torch.cuda.Event(enable_timing=True,external=True)
        begin.record(stream);end.record(stream);stream.synchronize()
        graph=torch.cuda.CUDAGraph(keep_graph=True);graph.enable_debug_mode()
        with torch.cuda.graph(graph,stream=stream):begin.record(stream);launch();end.record(stream)
        out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);graph.debug_dump(str(out.with_suffix('.dot')))
        raw=cuda.CUgraph(graph.raw_cuda_graph());_,count=checked(cuda.cuGraphGetNodes(raw));assert count==3
        nodes,count=checked(cuda.cuGraphGetNodes(raw,3));assert count==len(nodes)==3
        types={int(n):int(checked(cuda.cuGraphNodeGetType(n))[0]) for n in nodes}
        kernel_type=int(cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL);event_type=int(cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_EVENT_RECORD)
        assert list(types.values()).count(kernel_type)==1 and list(types.values()).count(event_type)==2
        events={int(checked(cuda.cuGraphEventRecordNodeGetEvent(n))[0]):int(n) for n in nodes if types[int(n)]==event_type}
        assert set(events)=={int(begin.cuda_event),int(end.cuda_event)}
        kernel=next(n for n in nodes if types[int(n)]==kernel_type)
        expected={(events[int(begin.cuda_event)],int(kernel)),(int(kernel),events[int(end.cuda_event)])}
        params=checked(cuda.cuGraphKernelNodeGetParams(kernel))[0]
        name=checked(cuda.cuFuncGetName(params.func))[0];name=name.decode() if isinstance(name,bytes) else str(name)
        grid=[int(params.gridDimX),int(params.gridDimY),int(params.gridDimZ)];block=[int(params.blockDimX),int(params.blockDimY),int(params.blockDimZ)]
        regs=int(checked(cuda.cuFuncGetAttribute(cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,params.func))[0])
        meta=dict(node_types=types,event_node_handles=events,begin_event=int(begin.cuda_event),end_event=int(end.cuda_event),kernel_name=name,grid=grid,block=block,dynamic_SMEM_B=int(params.sharedMemBytes),REG=regs,
           direct_lazy_invocations_in_enclosing_entry=1,graph_instantiates=0,graph_replays=0,graph_warmups=0,timing_samples=0,model_enabled=False)
        out.with_name('PRE_EDGE_METADATA.json').write_text(json.dumps(meta,indent=2))
        compare_live_graph(raw,expected,out.with_suffix('.json'),meta)
        assert expected_name in name and grid==expected_grid and block==expected_block and regs==expected_regs and meta['dynamic_SMEM_B']==expected_smem
        raise AuditOnlyCompleted('One original CONTROL128 lazy + same graph capture/query only; no instantiate/replay/warm/sample')
    def replay(self):raise RuntimeError('FORBIDDEN_IN_AUDIT_ONLY')
    def measure_us(self):raise RuntimeError('FORBIDDEN_IN_AUDIT_ONLY')
