"""Prepared CPU code. Query function needs a separately admitted live graph.

No graph creation, cuInit, replay, kernel launch, driver reset, or clock change.
"""
import ctypes,json,gc
from pathlib import Path

class EdgeData(ctypes.Structure):
    _fields_=[('from_port',ctypes.c_ubyte),('to_port',ctypes.c_ubyte),('type',ctypes.c_ubyte),('reserved',ctypes.c_ubyte*5)]
assert ctypes.sizeof(EdgeData)==8

def native_function():
    lib=ctypes.CDLL('libcuda.so.1');fn=lib.cuGraphGetEdges_v2
    fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(EdgeData),ctypes.POINTER(ctypes.c_size_t)];fn.restype=ctypes.c_int
    return lib,fn

def native_query(graph):
    lib,fn=native_function();count=ctypes.c_size_t(0)
    rc=fn(ctypes.c_void_p(int(graph)),None,None,None,ctypes.byref(count))
    record={'count_query_returncode':rc,'edge_count':count.value,'native_v2_address':ctypes.cast(fn,ctypes.c_void_p).value}
    if rc!=0:return record
    n=count.value;frm=(ctypes.c_void_p*n)();to=(ctypes.c_void_p*n)();data=(EdgeData*n)()
    record['zero_initialized_bytes_before']=ctypes.string_at(ctypes.addressof(data),ctypes.sizeof(data)).hex()
    rc=fn(ctypes.c_void_p(int(graph)),frm,to,data,ctypes.byref(count))
    record.update(data_query_returncode=rc,returned_count=count.value,
      edges=[{'from':frm[i],'to':to[i],'from_port':int(data[i].from_port),'to_port':int(data[i].to_port),'type':int(data[i].type),'reserved':list(data[i].reserved),'raw8':bytes(data[i]).hex()} for i in range(n)])
    # NULL edgeData is a separate lossless-query gate: non-default edges must fail.
    frm2=(ctypes.c_void_p*n)();to2=(ctypes.c_void_p*n)();count2=ctypes.c_size_t(n)
    rc2=fn(ctypes.c_void_p(int(graph)),frm2,to2,None,ctypes.byref(count2))
    record.update(NULL_edgeData_returncode=rc2,NULL_edgeData_count=count2.value,NULL_edgeData_edges=[[frm2[i],to2[i]] for i in range(n)])
    return record

def snapshot_python_tuple(result):
    record={'tuple_type':str(type(result)),'tuple_length':len(result),'element_types':[str(type(x)) for x in result],'returncode':int(result[0])}
    if int(result[0])!=0:return record
    frm,to,data,count=result[1:]
    record.update(from_type=str(type(frm)),to_type=str(type(to)),edge_data_type=str(type(data)),count_type=str(type(count)),count=int(count),
      edges=[[int(f),int(t)] for f,t in zip(frm,to)],edge_data=[])
    for e in data:
        d={'python_type':str(type(e)),'repr':repr(e),'from_port':int(e.from_port),'to_port':int(e.to_port),'type':int(e.type),'reserved':list(e.reserved)}
        if hasattr(e,'getPtr'):
            d['getPtr']=int(e.getPtr());d['raw8']=ctypes.string_at(d['getPtr'],8).hex()
        record['edge_data'].append(d)
    return record

def compare_live_graph(graph,expected_edges,out):
    from cuda.bindings import driver as cuda
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    first=cuda.cuGraphGetEdges(cuda.CUgraph(int(graph)),2)
    immediate=snapshot_python_tuple(first)
    native=native_query(graph)
    # Retain the original tuple while allocating CPU objects, then resnapshot.
    pressure=[bytearray(64) for _ in range(10000)];gc.collect()
    delayed=snapshot_python_tuple(first);second=snapshot_python_tuple(cuda.cuGraphGetEdges(cuda.CUgraph(int(graph)),2))
    record={'Python_immediate':immediate,'native_v2':native,'Python_retained_after_CPU_allocation':delayed,'Python_repeat':second,'expected_edges':[list(x) for x in sorted(expected_edges)],'replay_performed':False,'model_enabled':False}
    out.write_text(json.dumps(record,indent=2))
    assert native['count_query_returncode']==native.get('data_query_returncode')==native.get('NULL_edgeData_returncode')==0,record
    assert native['edge_count']==native['returned_count']==native['NULL_edgeData_count']==2
    assert {(x['from'],x['to']) for x in native['edges']}==set(expected_edges)
    assert {tuple(x) for x in native['NULL_edgeData_edges']}==set(expected_edges)
    assert all(x['from_port']==x['to_port']==x['type']==0 and not any(x['reserved']) for x in native['edges']),native
    # This diagnostic reports binding discrepancies; it does not silently admit timing.
    return record
