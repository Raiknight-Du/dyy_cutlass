"""Sparse K2 body measurement, independent Event, complete rotating CP cache history."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,os,re,statistics,subprocess,sys,threading,time,collections
ROOT=Path('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12');REPO=ROOT/'upstreams/flashinfer';PIN='e77a4a0d276367895c3b50a642fd8f326c03fb72'
def cmd(*a):return subprocess.check_output(a,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def others():return [x for x in cmd('nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader').splitlines() if x and x.split(',')[0].strip()!=str(os.getpid())]
def tele():return {'t':time.monotonic_ns(),'gpu':cmd('nvidia-smi','--query-gpu=utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw','--format=csv,noheader'),'others':others()}
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--generated',type=Path,required=True);p.add_argument('--trial',type=int,required=True);p.add_argument('--S',type=int,default=32768);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
assert cmd('git','-C',str(REPO),'rev-parse','HEAD')==PIN and not cmd('git','-C',str(REPO),'status','--porcelain')
assert not others(),others()
os.environ['FLASHINFER_WORKSPACE_BASE']=str(a.out/'cache');sys.path.insert(0,str(REPO))
import torch,cutlass.cute as cute,flashinfer
from cutlass.base_dsl.compiler import DumpDir
from flashinfer.gdn_kernels.blackwell import gdn_cp_prefill as cp
assert Path(flashinfer.__file__).resolve().is_relative_to(REPO)
torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location('k2_body_private',a.generated);gen=importlib.util.module_from_spec(spec);sys.modules[spec.name]=gen;spec.loader.exec_module(gen)
Original=cp.CPDeltaRuleMNPrecomputeUtcmma1Sm100
classes={'B0':Original,'B1':gen.K2Body_B1,'P1':gen.K2Body_P1,'P8':gen.K2Body_P8};modes=list(classes)
active=[a.out/'binary'/'prefix'];old_options=cp._blackwell_compile_options
def options(device):
    active[0].mkdir(parents=True,exist_ok=True)
    return tuple(old_options(device))+(cute.KeepPTX(True),cute.KeepCUBIN(True),cute.KeepSASS(True),DumpDir(str(active[0])))
cp._blackwell_compile_options=options
def capture(fn):
    seen=[];og,oc=cp.get_cached_compile,cp.cached_compile
    def wrap(f):
        if f is None:return None
        def call(*x,**kw):seen.append((f,x,kw));return f(*x,**kw)
        return call
    cp.get_cached_compile=lambda *x,**kw:wrap(og(*x,**kw));cp.cached_compile=lambda *x,**kw:wrap(oc(*x,**kw))
    try:res=fn()
    finally:cp.get_cached_compile,cp.cached_compile=og,oc
    assert len(seen)==1
    f,x,kw=seen[0]
    return res,lambda:f(*x[:-1],type(x[-1])(torch.cuda.current_stream().cuda_stream),**kw)
fixture=ROOT/f'performance_analysis/cp4_event_baseline_freeze_20260929/frozen_inputs/S{a.S}/inputs.pt'
data=torch.load(fixture,map_location='cpu',weights_only=True)
gate,beta,cu,initial=(data['shared'][n].cuda() for n in ('gate','beta','cu_seqlens','initial'))
record={'trial':a.trial,'S':a.S,'C':a.S//8,'P':8,'L':a.S//512,'grid':128,'dtype':'FP16','pin':PIN,'before':tele(),'fixture':str(fixture),'fixture_sha256':sha(fixture),'runner_sha256':sha(Path(__file__)),'generated_sha256':sha(a.generated),'cache':'full K1->K2->K3->K4; rotating4 frozen inputs','local_bias_bound':'UNKNOWN','model_enabled':False,'effective_model_value':None,'checks':[]}
bundles=[]
for bi,buf in enumerate(data['buffers']):
    q,k,v=(buf[n].cuda() for n in ('q','k','v'));o=torch.empty_like(q);final=torch.empty_like(initial);iw=torch.empty_like(initial)
    active[0]=a.out/'binary'/'prefix'
    t,k1=capture(lambda:cp.cp_delta_rule_t_precompute_dsl_sm100(k,beta,cu,a.S,max_seqlen=a.S))
    funcs={};traces={}
    for mode,cls in classes.items():
        cp.CPDeltaRuleMNPrecomputeUtcmma1Sm100=cls;cp._get_mn_precompute_kernel.cache_clear()
        trace=torch.zeros((128,8,2,3),device='cuda',dtype=torch.int64);traces[mode]=trace
        factory0=cp._get_mn_precompute_kernel
        def factory(*x,**kw):
            obj=factory0(*x,**kw);obj.probe_address=trace.data_ptr();obj._KeyedCompileMixin_compile_key=(obj._get_compile_key(),mode,trace.data_ptr());return obj
        if mode!='B0':cp._get_mn_precompute_kernel=factory
        active[0]=a.out/'binary'/f'{mode}_b{bi}'
        try:(m,n),f=capture(lambda:cp.cp_delta_rule_mn_precompute_dsl_sm100(k,v,t,gate,cu,a.S,cp_chunk_len=a.S//8,max_seqlen=a.S))
        finally:cp._get_mn_precompute_kernel=factory0
        torch.cuda.synchronize()
        if mode=='B0':mr=m.clone();nr=n.clone()
        else:torch.testing.assert_close(m,mr,atol=0,rtol=0);torch.testing.assert_close(n,nr,atol=0,rtol=0)
        funcs[mode]=f;record['checks'].append({'buffer':bi,'mode':mode,'exact':True})
    active[0]=a.out/'binary'/'suffix'
    fixed,k3=capture(lambda:cp.cp_delta_rule_fixup_dsl_sm100(m,n,cu,a.S,cp_chunk_len=a.S//8,initial_state=initial,output_state=final,initial_state_workspace=iw))
    _,k4=capture(lambda:cp.cp_delta_rule_prefill_dsl_sm100(o,None,q,k,v,t,fixed,gate,1/math.sqrt(128),cu,a.S,cp_chunk_len=a.S//8,max_seqlen=a.S,initial_state=iw))
    bundles.append({'k1':k1,'k2':funcs,'k3':k3,'k4':k4,'traces':traces,'refs':[q,k,v,t,m,n,fixed,o,final,iw,mr,nr]})
del data
cp.CPDeltaRuleMNPrecomputeUtcmma1Sm100=Original;cp._get_mn_precompute_kernel.cache_clear()
audits=[]
for bd in sorted((a.out/'binary').glob('*_b*')):
    cubins=list(bd.glob('*.cubin'));assert len(cubins)==1,(bd,cubins)
    c=cubins[0];sass=cmd('/usr/local/cuda/bin/cuobjdump','--dump-sass',str(c));res=cmd('/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(c))
    (bd/'audit.sass').write_text(sass);(bd/'resource.txt').write_text(res)
    ops=collections.Counter()
    for line in sass.splitlines():
        m=re.match(r'\s*/\*([0-9a-f]+)\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9.]*)\s',line)
        if m:ops[m[2]]+=1
    audits.append({'variant':bd.name,'cubin':str(c),'sha256':sha(c),'resource':res,'opcodes':dict(ops)})
(a.out/'binary_audit.json').write_text(json.dumps(audits,indent=2))
base=next(x for x in audits if x['variant']=='B0_b0')
def count(d,prefix):return sum(v for k,v in d['opcodes'].items() if k.startswith(prefix))
for x in audits:
    assert re.findall(r'(?:REG|STACK|SHARED):\d+',x['resource'])==re.findall(r'(?:REG|STACK|SHARED):\d+',base['resource']),(x['variant'],'resource changed')
    for op in ('LDL','STL','LDTM','STTM','UTCHMMA','LDSM','STSM','F2FP','FMUL2'):
        assert count(x,op)==count(base,op),(x['variant'],op,'work changed')
record['compile_gate']='PASS';print('COMPILE GATE PASS',flush=True)
stop=threading.Event();trajectory=[]
def monitor():
    while not stop.is_set():
        try:trajectory.append(tele())
        except Exception as e:trajectory.append({'error':str(e)})
        stop.wait(.2)
thread=threading.Thread(target=monitor,daemon=True);thread.start()
try:
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());graphs={}
    for mode in modes:
        graphs[mode]=[]
        for b in bundles:
            graph=torch.cuda.CUDAGraph();start=torch.cuda.Event(enable_timing=True,external=True);end=torch.cuda.Event(enable_timing=True,external=True)
            with torch.cuda.graph(graph,stream=stream):
                b['k1']();start.record();b['k2'][mode]();end.record();b['k3']();b['k4']()
            graphs[mode].append((graph,start,end))
    samples=[]
    for i in range(120):
        for j in range(4):
            mode=modes[(j+i+a.trial)%4];graph,start,end=graphs[mode][i%4]
            with torch.cuda.stream(stream):graph.replay()
            torch.cuda.synchronize()
            if i>=20:samples.append({'i':i,'mode':mode,'buffer':i%4,'event_us':start.elapsed_time(end)*1000})
    record['samples']=samples
    record['trace_calls']=[]
    for call in range(12):
        for mode in ('P1','P8'):
            bi=call%4;b=bundles[bi];graph,start,end=graphs[mode][bi]
            with torch.cuda.stream(stream):graph.replay()
            torch.cuda.synchronize()
            trace=b['traces'][mode].cpu().tolist()
            record['trace_calls'].append({'call':call,'mode':mode,'buffer':bi,'event_us':start.elapsed_time(end)*1000,'cta_rows':{str(cta):trace[cta] for cta in (range(0,128,16) if mode=='P8' else [0])}})
finally:
    stop.set();thread.join(timeout=5);record['telemetry']=trajectory
record['after']=tele();record['competition']=any(x.get('others') for x in trajectory)
record['event']={m:{'median_us':statistics.median(x['event_us'] for x in record['samples'] if x['mode']==m)} for m in modes}
assert not cmd('git','-C',str(REPO),'status','--porcelain')
(a.out/'result.json').write_text(json.dumps(record,indent=2));print(json.dumps({'trial':a.trial,'S':a.S,'event':record['event'],'competition':record['competition']}),flush=True)
assert not record['competition'],'INVALID: outside compute activity'
