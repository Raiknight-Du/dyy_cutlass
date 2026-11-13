"""Host-only authoring; constructor requires an already admitted CUDA context."""
from pathlib import Path
import json,time
from graph_edge_native_audit import native_query,snapshot_python_tuple

def checked(result):
    assert int(result[0])==0,result
    return result[1:]

class AuditedEventGraph:
    def __init__(self,launch,stream,out,expected_name,expected_grid,expected_block,expected_smem,expected_regs):
        import torch
        from cuda.bindings import driver as cuda
        self.begin=torch.cuda.Event(enable_timing=True,external=True)
        self.end=torch.cuda.Event(enable_timing=True,external=True)
        # Realize lazy event handles outside capture and all timed samples.
        self.begin.record(stream);self.end.record(stream);stream.synchronize()
        self.graph=torch.cuda.CUDAGraph(keep_graph=True);self.graph.enable_debug_mode()
        with torch.cuda.graph(self.graph,stream=stream):
            self.begin.record(stream);launch();self.end.record(stream)
        out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
        self.graph.debug_dump(str(out.with_suffix('.dot')))
        raw=cuda.CUgraph(self.graph.raw_cuda_graph())
        _,count=checked(cuda.cuGraphGetNodes(raw));assert count==3,count
        nodes,count=checked(cuda.cuGraphGetNodes(raw,count));assert count==len(nodes)==3
        kinds={int(node):int(checked(cuda.cuGraphNodeGetType(node))[0]) for node in nodes}
        kernel_type=int(cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL)
        event_type=int(cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_EVENT_RECORD)
        assert list(kinds.values()).count(kernel_type)==1 and list(kinds.values()).count(event_type)==2,kinds
        event_nodes={int(checked(cuda.cuGraphEventRecordNodeGetEvent(node))[0]):int(node) for node in nodes if kinds[int(node)]==event_type}
        begin_handle=int(self.begin.cuda_event);end_handle=int(self.end.cuda_event)
        assert begin_handle!=end_handle and set(event_nodes)=={begin_handle,end_handle},event_nodes
        kernel=next(node for node in nodes if kinds[int(node)]==kernel_type)
        expected_edges={(event_nodes[begin_handle],int(kernel)),(int(kernel),event_nodes[end_handle])}
        py_tuple=cuda.cuGraphGetEdges(raw,2)
        py_diagnostic=snapshot_python_tuple(py_tuple)
        native_edges=native_query(raw)
        pending=dict(status='NATIVE_EDGE_QUERY_CAPTURED_BEFORE_ADMISSION',graph_handle=int(raw),node_types=kinds,
            expected_edges=[list(e) for e in sorted(expected_edges)],Python_edge_data_diagnostic=py_diagnostic,native_owned_v2_edges=native_edges,model_enabled=False)
        out.with_suffix('.json').write_text(json.dumps(pending,indent=2))
        assert native_edges['count_query_returncode']==native_edges.get('data_query_returncode')==native_edges.get('NULL_edgeData_returncode')==0,native_edges
        assert native_edges['edge_count']==native_edges['returned_count']==native_edges['NULL_edgeData_count']==2
        edges={(e['from'],e['to']) for e in native_edges['edges']}
        assert edges==expected_edges and {tuple(e) for e in native_edges['NULL_edgeData_edges']}==expected_edges
        assert py_diagnostic['returncode']==0 and py_diagnostic['count']==2 and {tuple(e) for e in py_diagnostic['edges']}==expected_edges
        assert all(e['from_port']==e['to_port']==e['type']==0 and not any(e['reserved']) for e in native_edges['edges']),native_edges
        edge_types=[e['type'] for e in native_edges['edges']]
        params=checked(cuda.cuGraphKernelNodeGetParams(kernel))[0]
        grid=[int(params.gridDimX),int(params.gridDimY),int(params.gridDimZ)]
        block=[int(params.blockDimX),int(params.blockDimY),int(params.blockDimZ)]
        name=checked(cuda.cuFuncGetName(params.func))[0];name=name.decode() if isinstance(name,bytes) else str(name)
        regs=int(checked(cuda.cuFuncGetAttribute(cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,params.func))[0])
        assert expected_name in name and grid==expected_grid and block==expected_block
        assert int(params.sharedMemBytes)==expected_smem and regs==expected_regs
        self.graph.instantiate()
        record=dict(Python_edge_data_diagnostic=py_diagnostic,native_owned_v2_edges=native_edges,status='THREE_NODE_TWO_EDGE_EVENT_KERNEL_EVENT_PASS',graph_handle=int(raw),nodes=kinds,edges=[list(x) for x in sorted(edges)],edge_types=edge_types,
            begin_event_handle=begin_handle,end_event_handle=end_handle,kernel_name=name,grid=grid,block=block,dynamic_SMEM_B=int(params.sharedMemBytes),REG=regs,
            lazy_event_records_before_capture=2,capture_executes_kernel=False,measurement='one graph.replay containing two external timing EventRecord nodes and the original kernel only',
            audit_host_time_ns=time.time_ns(),model_enabled=False)
        out.with_suffix('.json').write_text(json.dumps(record,indent=2))

    def replay(self):self.graph.replay()
    def measure_us(self):
        self.graph.replay();self.end.synchronize()
        return self.begin.elapsed_time(self.end)*1000
