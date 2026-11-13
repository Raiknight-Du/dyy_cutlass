"""Isolated full-K1 experiment. Run only below guard_run.py + clock_lease.py."""
import argparse, collections, ctypes, hashlib, json, math, os, re, subprocess, sys, time, shutil
from pathlib import Path

assert os.environ.get('K1_GPU_SESSION'), 'serialized idle-GPU guard required'
ROOT = Path(__file__).resolve().parents[1]
REPO = Path('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/upstreams/flashinfer')
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['compile','screen','formal','profile','validate'], required=True)
parser.add_argument('--trial', type=int, default=0)
parser.add_argument('--s', type=int, default=32768)
parser.add_argument('--lens', default=None)
parser.add_argument('--kh', type=int, default=16)
parser.add_argument('--ho', type=int, default=16)
parser.add_argument('--dtype', choices=['fp16','bf16'], default='fp16')
parser.add_argument('--variants', default='baseline,tcgen05,layout,both')
parser.add_argument('--profile-variant', default='baseline')
parser.add_argument('--tag', default='v1')
parser.add_argument('--samples', type=int, default=100)
parser.add_argument('--residency', choices=['normal','matched1'], default='normal')
parser.add_argument('--tc-min8', action='store_true')
args = parser.parse_args()
names = args.variants.split(',')
out = ROOT/'results'/f'{args.tag}_{args.mode}_{args.dtype}_s{args.s}_r{args.trial}'
if args.mode=='profile': out=out/('pass_pid'+str(os.getpid()))
out.mkdir(parents=True, exist_ok=False)
(out/'source_snapshot').mkdir()
for source_file in (ROOT/'source').glob('*.py'):
    shutil.copyfile(source_file,out/'source_snapshot'/source_file.name)
os.environ['FLASHINFER_WORKSPACE_BASE'] = str(out/'cache')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path[:0] = [str(REPO), str(ROOT/'source')]
import numpy as np
import torch
import torch.nn.functional as F
import cutlass.cute as cute
from cutlass.base_dsl.compiler import DumpDir
from cuda.bindings import driver as d
from flashinfer.gdn_kernels.blackwell import gdn_cp_prefill as cp

assert Path(cp.__file__).resolve().is_relative_to(REPO)
assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip() == 'e77a4a0d276367895c3b50a642fd8f326c03fb72'
assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
assert torch.cuda.get_device_properties(0).multi_processor_count == 148
(out/'ENVIRONMENT.json').write_text(json.dumps(dict(torch=torch.__version__,cutlass=__import__('cutlass').__version__,gpu=str(torch.cuda.get_device_properties(0)),driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (out/'source_snapshot').glob('*.py')}),indent=2))

def checked(result):
    if int(result[0]): raise RuntimeError(str(result))
    return None if len(result)==1 else result[1] if len(result)==2 else result[1:]

def healthy():
    if (Path(os.environ['K1_GUARD_DIR'])/'INVALID_COMPETITION').exists():
        raise RuntimeError('external GPU process or telemetry failure: cohort invalid')

classes = {'baseline':cp.CPDeltaRuleTPrecomputeSm100}
if any(n in names for n in ('tcgen05','both')):
    if args.tc_min8: from tcgen_k1_min8 import TcgenK1
    else: from tcgen_k1 import TcgenK1
    classes['tcgen05'] = TcgenK1
if any(n in names for n in ('layout','both')):
    from store_layout import CoalescedStoreMixin
    class LayoutK1(CoalescedStoreMixin, cp.CPDeltaRuleTPrecomputeSm100): pass
    classes['layout'] = LayoutK1
if 'both' in names:
    class CombinedK1(CoalescedStoreMixin, TcgenK1): pass
    classes['both'] = CombinedK1

dtype = torch.float16 if args.dtype=='fp16' else torch.bfloat16
torch.manual_seed(20261109)
lengths=[int(x) for x in args.lens.split(',')] if args.lens else [args.s]
S,H,D = sum(lengths),args.ho,128
k = F.normalize(torch.randn(S,args.kh,D,device='cuda',dtype=torch.float32),dim=-1).to(dtype)
beta = torch.rand(S,H,device='cuda',dtype=torch.float32).sigmoid()
cu = torch.tensor([0]+list(np.cumsum(lengths)),device='cuda',dtype=torch.int32)
if args.mode=='validate':
    cp._get_cp_workspace = lambda name,shape,dtype,device: torch.zeros(shape,device=device,dtype=dtype)
stream = torch.cuda.Stream(); st = d.CUstream(stream.cuda_stream)
torch.cuda.synchronize()
original, opt0 = cp.CPDeltaRuleTPrecomputeSm100, cp._blackwell_compile_options
launches, outputs, binaries, infos, graphs = {},{},{},{},{}

def capture(name):
    dump = out/'binary'/name; dump.mkdir(parents=True)
    cp.CPDeltaRuleTPrecomputeSm100 = classes[name]
    cp._get_t_precompute_kernel.cache_clear()
    cp._blackwell_compile_options = lambda dev: tuple(opt0(dev)) + (cute.KeepPTX(True),cute.KeepCUBIN(True),cute.KeepSASS(True),DumpDir(str(dump)))
    g0,c0 = cp.get_cached_compile,cp.cached_compile
    seen = {}
    cp.get_cached_compile = lambda *a,**kw: None
    def defer(*a,**kw):
        seen['fn'] = c0(*a,**kw)
        def record(*aa,**kk): seen.update(args=aa,kwargs=kk)
        return record
    cp.cached_compile = defer
    try:
        output = cp.cp_delta_rule_t_precompute_dsl_sm100(k,beta,cu,S,max_seqlen=max(lengths),_stream=st)
    finally:
        cp.get_cached_compile,cp.cached_compile = g0,c0
        cp.CPDeltaRuleTPrecomputeSm100,cp._blackwell_compile_options = original,opt0
        cp._get_t_precompute_kernel.cache_clear()
    cb = list(dump.rglob('*.cubin'))
    if len(cb)!=1: raise RuntimeError(f'expected one cubin for {name}: {cb}')
    f = cb[0]
    sass = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(f)],text=True)
    resources = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(f)],text=True)
    (dump/'kernel.sass').write_text(sass); (dump/'resources.txt').write_text(resources)
    opcodes = collections.Counter()
    for line in sass.splitlines():
        m = re.match(r'\s*/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)',line)
        if m: opcodes[m.group(1)] += 1
    binaries[name] = dict(cubin=str(f.relative_to(ROOT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),resources=resources,opcodes=dict(opcodes),spill_sass=sum(v for op,v in opcodes.items() if op.startswith(('LDL','STL'))))
    (out/'BINARY_GATE.json').write_text(json.dumps(binaries,indent=2))
    if binaries[name]['spill_sass']: raise RuntimeError(f'no-spill gate failed: {name}')
    return output, lambda: seen['fn'](*seen['args'],**seen['kwargs'])

for name in names:
    healthy(); print('COMPILING',name,flush=True)
    outputs[name],launches[name] = capture(name)
    print('COMPILED',name,binaries[name]['opcodes'],flush=True)
(out/'BINARY_GATE.json').write_text(json.dumps(binaries,indent=2))
if 'baseline' in binaries:
    base=binaries['baseline']
    for n,b in binaries.items():
        if n!='baseline' and b['sha256']==base['sha256']: raise RuntimeError('BINARY_NO_OP: '+n)

def launch_graph(name):
    launch=launches[name]
    begin,end=checked(d.cuEventCreate(0)),checked(d.cuEventCreate(0))
    checked(d.cuStreamBeginCapture(st,d.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL))
    checked(d.cuEventRecordWithFlags(begin,st,d.CUevent_record_flags.CU_EVENT_RECORD_EXTERNAL))
    launch()
    checked(d.cuEventRecordWithFlags(end,st,d.CUevent_record_flags.CU_EVENT_RECORD_EXTERNAL))
    graph=checked(d.cuStreamEndCapture(st))
    _,count=checked(d.cuGraphGetNodes(graph,0)); nodes,count=checked(d.cuGraphGetNodes(graph,count))
    kernel=next(n for n in nodes if 'KERNEL' in str(checked(d.cuGraphNodeGetType(n))))
    cfg=checked(d.cuGraphKernelNodeGetParams(kernel))
    if args.residency=='matched1':
        checked(d.cuFuncSetAttribute(cfg.func,d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,130048))
        checked(d.cuFuncSetAttribute(cfg.func,d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,100))
        cfg.sharedMemBytes=130048
        checked(d.cuGraphKernelNodeSetParams(kernel,cfg))
    info=dict(grid=[int(cfg.gridDimX),int(cfg.gridDimY),int(cfg.gridDimZ)],block=[int(cfg.blockDimX),int(cfg.blockDimY),int(cfg.blockDimZ)],dynamic_smem=int(cfg.sharedMemBytes))
    info['occupancy_api_max_active_ctas_per_sm']=int(checked(d.cuOccupancyMaxActiveBlocksPerMultiprocessor(cfg.func,int(cfg.blockDimX),int(cfg.sharedMemBytes))))
    for key in ('NUM_REGS','SHARED_SIZE_BYTES','LOCAL_SIZE_BYTES'):
        info[key]=int(checked(d.cuFuncGetAttribute(getattr(d.CUfunction_attribute,'CU_FUNC_ATTRIBUTE_'+key),cfg.func)))
    infos[name]=info
    if args.residency=='matched1': assert info['occupancy_api_max_active_ctas_per_sm']==1,info
    graphs[name]=(graph,checked(d.cuGraphInstantiate(graph,0)),begin,end)

correctness={}
if args.mode=='compile':
    # Graph capture checks launch shape without executing the K1 kernel.
    for name in names: launch_graph(name)
else:
    for name in names:
        if args.mode=='validate':
            from validate_cases import poison_used_tiles,check_structural_output
            poison_used_tiles(outputs[name],lengths)
            torch.cuda.synchronize()
        healthy(); launches[name](); checked(d.cuStreamSynchronize(st))
        actual=outputs[name].clone(); torch.cuda.synchronize()
        assert torch.isfinite(actual).all().item(),name
        if args.mode=='validate': check_structural_output(actual,lengths,beta)
        if name=='baseline': expected=actual
        elif 'baseline' in names:
            diff=(actual.float()-expected.float())
            correctness[name]=dict(max_abs=float(diff.abs().max()),relative_l2=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(expected.float()).clamp_min(1e-20)),exact=bool(torch.equal(actual,expected)))
            if name=='layout': torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            else: torch.testing.assert_close(actual,expected,atol=0.002 if args.dtype=='fp16' else 0.004,rtol=0.005 if args.dtype=='fp16' else 0.02)
        if name=='tcgen05': tcgen_expected=actual
        if name=='both' and 'tcgen05' in names:
            torch.testing.assert_close(actual,tcgen_expected,atol=0,rtol=0)
            correctness[name]['exact_vs_tcgen05']=True
        launch_graph(name)
        print('CORRECTNESS',name,correctness.get(name,{'reference':'original'}),infos[name],flush=True)
(out/'LAUNCH_AND_CORRECTNESS.json').write_text(json.dumps(dict(launch=infos,correctness=correctness),indent=2))

samples=[]
if args.mode in ('screen','formal','profile'):
    eviction=torch.randint(0,2**30,(768*1024**2//4,),device='cuda',dtype=torch.int32)
    torch.cuda.synchronize()
    helper=ctypes.CDLL('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/performance_analysis/k1_eightpoint_2sm_cap1_1845_20261020/build/cold_helpers.so')
    helper.evict_l2.argtypes=[ctypes.c_void_p,ctypes.c_uint64,ctypes.c_uint32,ctypes.c_void_p]; helper.evict_l2.restype=ctypes.c_int
    counter=0
    def once(name,cache,it,pos,role):
        global counter
        healthy(); counter+=1
        if cache=='evicted':
            assert helper.evict_l2(eviction.data_ptr(),eviction.numel()*4,counter,int(st))==0
            checked(d.cuStreamSynchronize(st))
        graph,exe,begin,end=graphs[name]
        if args.mode=='profile' and role=='target': torch.cuda.nvtx.range_push('K1_VARIANT_'+name)
        checked(d.cuGraphLaunch(exe,st));checked(d.cuStreamSynchronize(st))
        if args.mode=='profile' and role=='target': torch.cuda.nvtx.range_pop()
        us=float(checked(d.cuEventElapsedTime(begin,end)))*1000
        row=dict(trial=args.trial,variant=name,cache=cache,iteration=it,position=pos,role=role,event_us=us)
        if role=='target': samples.append(row)
        return us
    for cache in (['evicted'] if args.mode=='profile' else ['evicted','repeat_no_evict']):
        warmup=2 if args.mode=='profile' else 20
        for w in range(warmup):
            for name in names: once(name,cache,-1,w,'warmup')
        n=1 if args.mode=='profile' else 12 if args.mode=='screen' else args.samples
        for it in range(n):
            ordered=names[it%len(names):]+names[:it%len(names)]
            if (it//len(names))%2: ordered=list(reversed(ordered))
            for pos,name in enumerate(ordered):
                if args.mode!='profile' or name==args.profile_variant: once(name,cache,it,pos,'target')
        for name in names:
            z=[r['event_us'] for r in samples if r['variant']==name and r['cache']==cache]
            if z: print('EVENT',cache,name,float(np.median(z)),flush=True)
    (out/'EVENT_SAMPLES.json').write_text(json.dumps(samples,indent=2))

record=dict(status='COMPLETED_PENDING_REVIEW',mode=args.mode,trial=args.trial,shape=[len(lengths),H,S,D],seq_lens=lengths,k_heads=args.kh,dtype=args.dtype,residency=args.residency,source_commit='e77a4a0d276367895c3b50a642fd8f326c03fb72',inputs_seed=20261109,K_normalized_outside_timing=True,beta='sigmoid(uniform)',attention_scale='not used in K1',binaries=binaries,launch=infos,correctness=correctness,samples=samples,endpoint='CUDA external Event nodes around one full K1 kernel in a CUDA graph',cache_eviction='768MiB cg eviction helper, not proof of perfect hardware cold reset',workspace='same original T workspace pointer across variants, except zero-initialized per-variant buffers for ragged validation',model_enabled=False)
(out/'RUN_RECORD.json').write_text(json.dumps(record,indent=2))
for graph,exe,b,e in graphs.values():
    checked(d.cuGraphExecDestroy(exe));checked(d.cuGraphDestroy(graph));checked(d.cuEventDestroy(b));checked(d.cuEventDestroy(e))
print('FINISHED',str(out),flush=True)
