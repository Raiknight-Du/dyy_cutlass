"""Runner-only R4 CUDA metadata/screen/formal/profile entry; never CPU author.

Requires explicit Lead admission JSON and the Runner's external exclusive GPU
guard. Actual CUDA metadata is compiled once per process and native text must
match the offline fake-metadata gate BEFORE any invocation of the helper.
"""
from pathlib import Path
import argparse,collections,csv,hashlib,json,os,re,struct,subprocess,sys
O=Path('/home/ubuntu/workspace/op12/k4_10h_native_r4_20261026')
sys.path.insert(0,str(O))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def native_text(p):
    raw=Path(p).read_bytes();off=struct.unpack_from('<Q',raw,40)[0];sz,num,si=struct.unpack_from('<HHH',raw,58)
    hh=[struct.unpack_from('<IIQQQQIIQQ',raw,off+i*sz) for i in range(num)]
    ss=hh[si];strings=raw[ss[4]:ss[4]+ss[5]];out={}
    for h in hh:
        name=strings[h[0]:].split(b'\0',1)[0].decode()
        if name.startswith('.text.'):out[name]=hashlib.sha256(raw[h[4]:h[4]+h[5]]).hexdigest()
    return out
def quant(v,p):
    v=sorted(v);x=p*(len(v)-1);a=int(x);b=min(a+1,len(v)-1);return v[a]+(v[b]-v[a])*(x-a)
def check_cuda(result):
    assert int(result[0])==0,result
    return result[1:] if len(result)>1 else None
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','screen','formal','profile'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True)
    p.add_argument('--gate',type=Path,default=O/'STATIC_GATE.json');p.add_argument('--trial',type=int,default=0)
    p.add_argument('--config',choices=['all','CONTROL','SHADER_ONLY','TC_ONLY','BOTH'],default='all')
    p.add_argument('--points',default='');a=p.parse_args();assert a.mode in ['screen','formal'],'Graph cohort only; no new NCU/prepare'
    gate=json.loads(a.gate.read_text());auth=json.loads(a.authorization.read_text())
    assert auth['round']=='R4' and ('NCU' if a.mode=='profile' else a.mode) in auth['approved_modes']
    assert auth['source_SHA256']==gate['source_SHA256']['shared_coupling.py']==sha(O/'shared_coupling.py')
    for n,h in gate['source_SHA256'].items():assert sha(O/n)==h
    assert os.environ.get('K1_GPU_SESSION') or os.environ.get('R4_GPU_SESSION'),'External exclusive Runner guard required'
    out=a.out/f'application_{os.getpid()}' if a.mode=='profile' else a.out
    out.mkdir(parents=True,exist_ok=False)
    nn=list(map(int,(a.points or ('128,256' if a.mode=='screen' else '512' if a.mode=='profile' else '128,256,512,1024,2048,4096,8192')).split(',')))
    assert all(n%128==0 and 128<=n<=min(16384,auth['max_N']) for n in nn)
    if a.mode=='screen':assert nn==[128,256]
    if a.mode=='profile':assert nn==[512] and a.config!='all'
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.base_dsl.compiler import DumpDir
    from cuda.bindings import driver as cuda
    from shared_coupling import SharedCoupling
    torch.set_num_threads(1);torch.cuda.set_device(0)
    prop=torch.cuda.get_device_properties(0)
    # Metadata allocation is not a helper launch. Values are copied only after
    # actual native text/resource verification below.
    ga=torch.empty((64,128),dtype=torch.float16,device='cuda')
    gb=torch.empty_like(ga);go=torch.empty((128,64,64),dtype=torch.float32,device='cuda')
    checks=torch.empty((128,128),dtype=torch.int32,device='cuda')
    trace=torch.empty((128,6),dtype=torch.int64,device='cuda')
    cv=lambda x:cute.runtime.from_dlpack(x,assumed_align=16,enable_tvm_ffi=True)
    args=[cv(x) for x in [ga,gb,go,checks,trace]]
    stream=torch.cuda.Stream();cs=cuda.CUstream(stream.cuda_stream)
    build=out/'actual_metadata';build.mkdir()
    obj=SharedCoupling();opts=(cute.EnableTVMFFI(True),cute.GPUArch('sm_100a'),cute.KeepPTX(True),cute.KeepCUBIN(True),cute.KeepSASS(True),DumpDir(str(build)))
    fn=cute.compile[opts](obj,*args,cutlass.Int32(128),cutlass.Int32(0),cs)
    cubin=next(build.glob('*.cubin'));actual_text=native_text(cubin)
    resource=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(cubin)],text=True)
    static_ok=sorted(actual_text.values())==sorted(gate['CPU_native_text_hashes'].values()) and bool(re.search(r'REG:79 STACK:0 SHARED:1024 LOCAL:0',resource))
    actual={'actual_CUBIN_SHA256':sha(cubin),'native_text_hashes':actual_text,'CPU_text_exact_match':static_ok,
       'resource':resource,'python':sys.executable,'torch':torch.__version__,'cutlass':cutlass.__version__,
       'torch_cuda':torch.version.cuda,'device':str(prop),'reserved_dynamic_SMEM_B':225280,
       'used_struct_SMEM_B':obj.shared.size_in_bytes(),'grid':128,'threads':192,'mode_flags_runtime':True,
       'admission':auth,'helper_invoked':False,'model_enabled':False}
    (out/'ACTUAL_METADATA_GATE.json').write_text(json.dumps(actual,indent=2))
    assert static_ok,'Actual CUDA metadata text/resource changed: stop before any helper invocation'
    assert actual['actual_CUBIN_SHA256']==gate['CPU_CUBIN_SHA256'],'Graph cohort must preserve whole actual CUBIN hash'
    # Query occupancy from the same frozen CUBIN, never launch the extra module.
    mod=check_cuda(cuda.cuModuleLoadData(cubin.read_bytes()))[0]
    symbol=next(iter(actual_text)).removeprefix('.text.')
    fun=check_cuda(cuda.cuModuleGetFunction(mod,symbol.encode()))[0]
    check_cuda(cuda.cuFuncSetAttribute(fun,cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,225280))
    occ=check_cuda(cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(fun,192,225280))[0]
    check_cuda(cuda.cuModuleUnload(mod));actual['max_CTA_per_SM']=int(occ)
    (out/'ACTUAL_METADATA_GATE.json').write_text(json.dumps(actual,indent=2));assert int(occ)==1
    if a.mode=='prepare':return
    ga.copy_(torch.ones((64,128),dtype=torch.float16));gb.copy_(torch.ones((64,128),dtype=torch.float16))
    torch.cuda.synchronize()
    names=['CONTROL','SHADER_ONLY','TC_ONLY','BOTH'];modes=[i for i,name in enumerate(names) if a.config in ['all',name]]
    samples=5 if a.mode=='screen' else 1 if a.mode=='profile' else 30
    expected=[]
    for tid in range(128):
        value=0
        for j in range(16):
            for k in range(4):
                index=4*tid+512*j+k
                value^=((index*0x9e3779b9)^((index>>3)*0x85ebca6b))&0xffffffff
        assert value!=0xdeadbeef;expected.append(value)
    ev=(out/'EVENT_RAW.csv').open('w',newline='');tc=(out/'CTA_CLOCK_RAW.csv').open('w',newline='')
    ew=csv.DictWriter(ev,fieldnames=['invocation','trial','config','N','sample','event_us','cta_cycle_median','cta_cycle_p5','cta_cycle_p95','cta_cycle_max','unique_SMs','valid','model_enabled'])
    tw=csv.DictWriter(tc,fieldnames=['invocation','trial','config','N','sample','cta','begin_cycle','end_cycle','smid_begin','smid_end','cycles'])
    ew.writeheader();tw.writeheader();invocation=0
    def launch(n,m):
        with torch.cuda.stream(stream):fn(*args,n,m,cs)
    from audited_event_graph import AuditedEventGraph
    # One explicitly disclosed lazy-module call, outside the 20 graph warmups.
    launch(128,0);stream.synchronize()
    actual['helper_invoked']=True;actual['direct_lazy_invocations']=1
    (out/'ACTUAL_METADATA_GATE.json').write_text(json.dumps(actual,indent=2))
    (out/'GRAPH_TIMING_PROTOCOL.json').write_text(json.dumps(dict(direct_lazy_invocations=1,direct_lazy_mode='CONTROL',direct_lazy_N=128,graph_replays_warmup_per_mode_per_N=20,external_events=True,kernel_changed=False,model_enabled=False),indent=2))
    for n in nn:
        timers={}
        for m in modes:
            timers[m]=AuditedEventGraph(lambda n=n,m=m:launch(n,m),stream,out/'GRAPH_NODES'/f'N{n}_M{m}',
                'kernel_cutlass_kernel_shared_couplingSharedCoupling',[128,1,1],[192,1,1],225280,79)
            for _ in range(20):timers[m].replay()
            stream.synchronize()
        for sample in range(samples):
            order=modes if sample%2==0 else list(reversed(modes))
            for m in order:
                event=timers[m].measure_us()
                hh=trace.cpu().tolist();ss=checks.cpu().tolist();oo=go.cpu()
                assert bool(torch.all(oo==(128.0 if m&2 else 0.0))),('QK output',n,m)
                durations=[];sms=[]
                for c,r in enumerate(hh):
                    assert len(r)==6 and r[0]<r[1] and r[2]==r[3] and r[4:]==[n,m],r
                    assert [x&0xffffffff for x in ss[c]]==(expected if m&1 else [0]*128),('shader output',n,m,c)
                    durations.append(r[1]-r[0]);sms.append(r[2])
                    tw.writerow(dict(invocation=invocation,trial=a.trial,config=names[m],N=n,sample=sample,
                        cta=c,begin_cycle=r[0],end_cycle=r[1],smid_begin=r[2],smid_end=r[3],cycles=r[1]-r[0]))
                ew.writerow(dict(invocation=invocation,trial=a.trial,config=names[m],N=n,sample=sample,
                    event_us=event,cta_cycle_median=quant(durations,.5),cta_cycle_p5=quant(durations,.05),
                    cta_cycle_p95=quant(durations,.95),cta_cycle_max=max(durations),unique_SMs=len(set(sms)),valid=1,model_enabled=False))
                invocation+=1;ev.flush();tc.flush()
    actual['helper_invoked']=True;actual['actual_invocations']=invocation
    (out/'ACTUAL_METADATA_GATE.json').write_text(json.dumps(actual,indent=2))
    (out/'completion.json').write_text(json.dumps({'completed':True,'invocations':invocation,'correctness':True,
        'warmup':20,'samples':samples,'empty_commit_modes_returned':all(x in modes for x in [0,1]),'model_enabled':False},indent=2))
    ev.close();tc.close()
if __name__=='__main__':main()
