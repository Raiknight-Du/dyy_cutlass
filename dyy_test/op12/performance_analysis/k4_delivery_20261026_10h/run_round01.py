"""Bounded original-K4 trace qualification, only under the shared GPU guard."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OP = Path('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12')
REPO = OP / 'upstreams/flashinfer'
PIN = 'e77a4a0d276367895c3b50a642fd8f326c03fb72'
parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=['screen', 'formal', 'profile'], required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--trial', type=int, default=0)
parser.add_argument('--approved', default='')
parser.add_argument('--family', default='ready_gate_round01')
parser.add_argument('--layout', choices=['S','V'], default='S')
parser.add_argument('--profile', choices=['REF','FRAME0','FRAME7'], default='REF')
parser.add_argument('--compile-approved', required=True)
parser.add_argument('--cases', default='EMPTY,CLOCK_COARSE,GLOBAL_COARSE,CLOCK_DETAIL,GLOBAL_DETAIL')
args = parser.parse_args()
out = Path(args.out)
if args.phase == "profile":
    out = out / f"replay_pid{os.getpid()}"
out.mkdir(parents=True, exist_ok=False)


def save(name, value):
    (out/name).write_text(json.dumps(value, indent=2, ensure_ascii=False))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cmd(*parts):
    return subprocess.check_output(parts, text=True).strip()


def guarded():
    assert os.environ.get('K1_GPU_SESSION')
    assert not (Path(os.environ['K1_GUARD_DIR'])/'INVALID_COMPETITION').exists()


guarded()
assert cmd('git', '-C', str(REPO), 'rev-parse', 'HEAD') == PIN
assert not cmd('git', '-C', str(REPO), 'status', '--porcelain')
sys.path.insert(0, str(REPO))
os.environ['FLASHINFER_WORKSPACE_BASE'] = str(ROOT/'jit_cache')
import numpy as np
import torch
import flashinfer
import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.compiler import DumpDir
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.gdn_kernels.blackwell import gdn_cp_prefill as cp

assert Path(flashinfer.__file__).resolve().is_relative_to(REPO)
torch.set_num_threads(1)
prop = torch.cuda.get_device_properties(0)
assert prop.multi_processor_count == 148 and prop.major == 10
save('ENVIRONMENT.json', dict(gpu=str(prop), python=sys.version, torch=torch.__version__, cuda=torch.version.cuda, pin=PIN, smi=cmd('nvidia-smi', '-q')))
inp = OP/'performance_analysis/cp4_event_baseline_freeze_20260929/frozen_inputs/S32768/inputs.pt'
assert sha(inp) == json.loads((inp.parent/'storage_hash.json').read_text())['inputs.pt']
data = torch.load(inp, map_location='cpu', weights_only=True)
buffers = [{**{k: v.cuda() for k, v in b.items()}, 'output': torch.empty(32768,16,128,dtype=torch.float16,device='cuda'), 'state': torch.empty(1,16,128,128,dtype=torch.float32,device='cuda')} for b in data['buffers']]
shared = {k: v.cuda() for k, v in data['shared'].items()}
del data
trace = torch.full((12288,), -1, dtype=torch.int64, device='cuda')
evict = torch.empty(4*int(prop.L2_cache_size),dtype=torch.uint8,device='cuda')
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
torch.cuda.synchronize()
artifacts, compiled, dsl, objects, runtime = {}, {}, {}, {}, {}
buffer_id = 0
native_ops = ['LDL','STL','UTCHMMA','UTMALDG','UTMASTG','LDTM','STTM','HMMA','LDG','STG','F2FP','BAR','UTCBAR','SYNCS','FENCE','MEMBAR','FFMA','FMUL','FADD','LDSM','STSM','LDS','STS','MUFU','PRMT']
projection_ops = set(native_ops) - {'STG', 'LDS', 'STS'}


def audit(dst):
    cub = next(dst.glob('*.cubin'))
    sass = next(dst.glob('*.sass')).read_text()
    resources = cmd('/usr/local/cuda/bin/cuobjdump', '--dump-resource-usage', str(cub))
    instructions = []
    for line in sass.splitlines():
        m = re.match(r'\s*/\*([0-9a-f]+)\*/\s+(.*?);', line)
        if m:
            text = m[2].strip()
            op = re.sub(r'^@!?\w+\s+', '', text).split()[0]
            instructions.append((m[1], text, op))
    counts = {k: sum(op.split('.')[0] == k for pc, text, op in instructions) for k in native_ops}
    projection = [op for pc, text, op in instructions if op.split('.')[0] in projection_ops]
    return dict(cubin=str(cub), cubin_sha256=sha(cub), resources=resources,
                registers=re.findall(r'REG:(\d+)', resources), stack=re.findall(r'STACK:(\d+)', resources),
                counts=counts, projection=projection, clock_PC=[dict(pc=pc,instruction=text) for pc,text,op in instructions if 'SR_CLOCKLO' in text or 'SR_GLOBALTIMER' in text])


def compile_one(name, kernel, ca, options):
    guarded()
    dst = out/'binary'/name
    dst.mkdir(parents=True)
    fn = cute.compile[(*options, cute.KeepPTX(True), cute.KeepCUBIN(True), cute.KeepSASS(True), DumpDir(str(dst)))](kernel, *ca)
    artifacts[name] = audit(dst)
    save('COMPILE_AUDIT.json', artifacts)
    print('COMPILED', name, artifacts[name]['cubin_sha256'], flush=True)
    return fn


def role(kernel):
    return {'CPDeltaRuleTPrecomputeSm100':'K1','CPDeltaRuleMNPrecomputeUtcmma1Sm100':'K2','CPDeltaRuleFixupUtcmmaSm100':'K3','CPDeltaRulePrefillTcgen05Sm100':'K4'}[type(kernel).__name__]


def compiler(kernel, *ca, compile_options):
    r = role(kernel)
    if r == 'K4' and args.layout == 'V':
        va = list(ca)
        cp._mark_state_layout(va[7], False, 128)
        ca = tuple(va)
    objects[r], dsl[r] = kernel, (ca, compile_options)
    fn = compile_one('REF', kernel, ca, compile_options) if r == 'K4' else cute.compile[compile_options](kernel, *ca)
    def wrapped(*values):
        runtime[buffer_id, r] = values
        return fn(*values)
    compiled[r] = (fn, wrapped)
    return wrapped


cp.get_cached_compile = lambda kernel, options: compiled.get(role(kernel), (None,None))[1]
cp.cached_compile = compiler
original_cache = cp._get_cache_buf
cp._get_cache_buf = lambda name, *a, **kw: original_cache(name+f'_trace_review_b{buffer_id}', *a, **kw)
gold = []
for buffer_id in range(4):
    b = buffers[buffer_id]
    with torch.cuda.stream(stream):
        chunk_gated_delta_rule(b['q'], b['k'], b['v'], g=shared['gate'], beta=shared['beta'], scale=1/(128**.5), initial_state=shared['initial'], output_final_state=True, cu_seqlens=shared['cu_seqlens'], use_qk_l2norm_in_kernel=False, output=b['output'], output_state=b['state'], use_cp=True)
    stream.synchronize()
    gold.append((b['output'].clone(), b['state'].clone()))
    torch.cuda.synchronize()
expected = json.loads(Path(args.compile_approved).read_text())
assert artifacts['REF']['cubin_sha256'] == expected['B0']['cubin_sha256']
functions = {'REF': compiled['K4'][0]}
kernel = objects['K4']
ca, options = dsl['K4']
spec = importlib.util.spec_from_file_location('K4READY_GATE', ROOT/'PTR_READY_GATE.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cls = module.K4READY_GATE
target = cls.__new__(cls)
target.__dict__.update(kernel.__dict__)
va = list(ca)
va.insert(-1, cute.runtime.from_dlpack(trace, assumed_align=16, enable_tvm_ffi=True))
va.insert(-1, cutlass.Int32(0))
if args.layout == 'S':
    pspec = importlib.util.spec_from_file_location('K4CLOCK_RETURNS', ROOT/'PTR_CLOCK_RETURNS.py')
    pmodule = importlib.util.module_from_spec(pspec)
    pspec.loader.exec_module(pmodule)
    ptarget = pmodule.K4CLOCK_RETURNS.__new__(pmodule.K4CLOCK_RETURNS)
    ptarget.__dict__.update(kernel.__dict__)
    pva = list(ca)
    pva.insert(-1, cute.runtime.from_dlpack(trace, assumed_align=16, enable_tvm_ffi=True))
    functions['PBASE'] = compile_one('PBASE', ptarget, pva, options)
    assert artifacts['PBASE']['cubin_sha256'] == expected['CLOCK']['cubin_sha256']
functions['FRAME'] = compile_one('FRAME', target, va, options)
assert artifacts['FRAME']['cubin_sha256'] == expected['FRAME']['cubin_sha256']
functions['FRAME0'] = functions['FRAME7'] = functions['FRAME']
qualified = (['PBASE'] if args.layout == 'S' else []) + ['FRAME0','FRAME7']
save('IDENTITY.json', dict(pin=PIN, shape=[1,16,32768,128], C=4096, P=8, grid=128, block=384,
    input_sha256=sha(inp), sample_CTA=16, layout=args.layout, qualified=qualified,
    runtime_mask={'FRAME0':0,'FRAME7':7}, same_cubin=True, local_bias_bound_cycles=None, model_enabled=False))



def call4(name, bi):
    v = runtime[bi,'K4']
    return functions[name](*v) if name == "REF" else functions[name](*v[:-1],trace,v[-1]) if name == "PBASE" else functions[name](*v[:-1],trace,0 if name == "FRAME0" else 7,v[-1])


def chain(name, bi, scope=None, events=None):
    if scope == 'CP_EVENT':
        events[0].record()
    for r in ['K1','K2','K3']:
        compiled[r][0](*runtime[bi,r])
    if scope == 'K4_EVENT':
        events[0].record()
    call4(name, bi)
    if scope in ['CP_EVENT','K4_EVENT']:
        events[1].record()


correctness = {}
for name in qualified:
    correctness[name] = []
    for bi in range(4):
        signal.alarm(30)
        with torch.cuda.stream(stream):
            chain(name, bi)
        stream.synchronize()
        signal.alarm(0)
        ok = torch.equal(buffers[bi]['output'],gold[bi][0]) and torch.equal(buffers[bi]['state'],gold[bi][1])
        correctness[name].append(ok)
        assert ok, name
save('CORRECTNESS.json', correctness)
# Screen includes only 3 warmups/3 paired blocks; it is not formal admission.
approved = expected
if args.phase == 'formal':
    approved = json.loads(Path(args.approved).read_text())
# Literal output address differs across processes; compare each actual compiled gate, retain hashes.
assert artifacts['REF']['cubin_sha256'] == expected['B0']['cubin_sha256']
for name in qualified:
    assert artifacts['FRAME']['cubin_sha256'] == expected['FRAME']['cubin_sha256']

graphs = {}
for name in ['REF',*qualified]:
    for scope in ['CP_EVENT','K4_EVENT'] + (['CLOCK'] if name in qualified else []):
        for bi in range(4):
            ev = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)]
            with torch.cuda.stream(stream):
                for e in ev:
                    e.record()
            stream.synchronize()
            g = torch.cuda.CUDAGraph(keep_graph=True)
            g.enable_debug_mode()
            with torch.cuda.graph(g, stream=stream):
                trace.fill_(-1)
                evict.fill_(90)
                chain(name, bi, scope, ev)
            g.debug_dump(str(out/f'{name}_{scope}_b{bi}.dot'))
            g.instantiate()
            graphs[name,scope,bi] = g,ev


def execute(name, scope, bi):
    guarded()
    g, ev = graphs[name,scope,bi]
    with torch.cuda.stream(stream):
        g.replay()
    stream.synchronize()
    return trace.cpu().tolist() if scope == 'CLOCK' else ev[0].elapsed_time(ev[1])*1000



if args.phase == 'profile':
    for i in range(3 if args.phase == "screen" else 20):
        execute(args.profile,'K4_EVENT',i%4)
    torch.cuda.nvtx.range_push('K4_READY_GATE_R1')
    execute(args.profile,'K4_EVENT',0)
    torch.cuda.nvtx.range_pop()
    save('PROFILE_COMPLETE.json',dict(variant=args.profile,layout=args.layout,model_enabled=False))
    sys.exit(0)

event_rows, clock_rows, paired_clock_rows, summaries = [], [], [], {}
rng = np.random.default_rng([20261025,args.trial])


def pair(scope, A, B):
    for i in range(3 if args.phase == "screen" else 20):
        execute(A, scope, i%4)
        execute(B, scope, i%4)
    ratios, differences, values = [], [], {A:[],B:[]}
    for block in range(3 if args.phase == "screen" else 30):
        order = [A,B,B,A] if rng.integers(2) == 0 else [B,A,A,B]
        local = {A:[],B:[]}
        for member,name in enumerate(order):
            val = execute(name,scope,block%4)
            local[name].append(val)
            values[name].append(val)
            event_rows.append(dict(process=args.trial, series=scope, pair=A+'__'+B, block=block, member=member, buffer=block%4, variant=name, us=val))
            if name != 'REF':
                arr = trace.cpu().numpy().reshape(12,128,8)
                rawdir = out/'PAIRED_RAW'
                rawdir.mkdir(exist_ok=True)
                invocation = f't{args.trial}_{scope}_{A}__{B}_{block}_{member}_{name}'
                np.savez_compressed(rawdir/(invocation+'.npz'), timestamps=arr, buffer=block%4)
                ib,ie = int(arr[4,64,0]),int(arr[4,64,1])
                assert 0 <= ib <= ie
                paired_clock_rows.append(dict(process=args.trial,series=scope,pair=A+'__'+B,
                    block=block,member=member,buffer=block%4,variant=name,invocation=invocation,
                    INIT_begin=ib,INIT_end=ie,INIT_cycles=ie-ib,
                    first_CG1_return=int(arr[4,0,0]),last_CG1_return=int(arr[4,63,0]),
                    INIT_end_to_CG1_first=int(arr[4,0,0])-ie,
                    SMID=int(arr[4,64,7]),cubin=artifacts['PBASE' if name=='PBASE' else 'FRAME']['cubin_sha256'],
                    local_bias_bound=None))
        av,bv = statistics.mean(local[A]),statistics.mean(local[B])
        ratios.append(100*(bv/av-1))
        differences.append(av-bv)
    boot = np.median(np.array(ratios)[rng.integers(0,len(ratios),(10000,len(ratios)))],axis=1)
    ci = np.percentile(boot,[2.5,97.5]).tolist()
    s = dict(paired_change_pct=statistics.median(ratios), CI95_pct=ci, paired_saving_us=statistics.median(differences),
             equivalent=ci[0]>=-2 and ci[1]<=2)
    summaries[scope+'_'+A+'__'+B] = s
    print('PAIR',scope,A,B,s,flush=True)
    return s


for scope in ['CP_EVENT','K4_EVENT']:
    pair(scope,'REF','FRAME0')
    if args.layout == 'S':
        pair(scope,'PBASE','FRAME0')
    pair(scope,'FRAME0','FRAME7')

stage_summaries = {}
for scope in ['CP_EVENT','K4_EVENT']:
    for A,B,limit in ([('PBASE','FRAME0',5),('FRAME0','FRAME7',None)] if args.layout == 'S' else [('FRAME0','FRAME7',None)]):
        rows = [r for r in paired_clock_rows if r['series']==scope and r['pair']==A+'__'+B]
        ratios=[]
        for block in sorted({r['block'] for r in rows}):
            ar=[r['INIT_cycles'] for r in rows if r['block']==block and r['variant']==A]
            br=[r['INIT_cycles'] for r in rows if r['block']==block and r['variant']==B]
            assert len(ar)==len(br)==2
            ratios.append(100*(statistics.mean(br)/statistics.mean(ar)-1))
        ci=np.percentile(np.median(np.array(ratios)[rng.integers(0,len(ratios),(10000,len(ratios)))],axis=1),[2.5,97.5]).tolist()
        stage_summaries[scope+'_'+A+'__'+B]=dict(paired_change_pct=statistics.median(ratios),CI95_pct=ci,raw_paired_block_change_pct=ratios,
            relative_stability_limit_pct=limit,relative_stability_pass=(ci[0]>=-limit and ci[1]<=limit) if limit else None,
            local_absolute_bias_bound_cycles=None,screen_only=args.phase=='screen')
save('PAIRED_STAGE_QC.json',stage_summaries)

usable = qualified if all(summaries[s+'_REF__FRAME0']['equivalent'] for s in ['CP_EVENT','K4_EVENT']) else []
for name in usable:
    for i in range(3 if args.phase == "screen" else 20):
        execute(name,'CLOCK',i%4)
    for sample in range(6 if args.phase == "screen" else 60):
        vals = execute(name,'CLOCK',sample%4)
        invocation = f't{args.trial}_{name}_{sample}'
        array=np.asarray(vals,dtype=np.int64).reshape(12,128,8)
        rawdir=out/'RAW'; rawdir.mkdir(exist_ok=True)
        np.savez_compressed(rawdir/(invocation+'.npz'),timestamps=array,buffer=sample%4)
        timer='clock64_cycles'
        for warp in range(12):
            for it in range(70):
                for b,e,label in [(0,1,'DISPATCH' if it<64 else {64:'INIT',65:'ROLE_LOOP',66:'ENTRY_EXIT',67:'QKV_FIRST_ISSUED',68:'T_FIRST_ISSUED',69:'MMA_FIRST_ISSUED'}[it]),(2,3,'INIT' if it==64 else 'T_LOAD_OR_O_STORE')]:
                    start,end=int(array[warp,it,b]),int(array[warp,it,e])
                    if start<0 and end<0: continue
                    if (it>=67 or (warp in [4,9] and it<64)) and start>=0: end=start
                    if warp==9 and it<64: label='QKV_DISPATCH_RETURN' if b==0 else 'T_DISPATCH_RETURN'
                    if warp==4 and it<64: label='CG1_CHUNK_RETURN'
                    assert 0<=start<=end,(name,warp,it,b,e,start,end)
                    clock_rows.append(dict(process=args.trial,invocation=invocation,variant=name,sample=sample,buffer=sample%4,CTA=16,warp=warp,iteration=it,stage=label,SMID=int(array[warp,65,7] if array[warp,65,7]>=0 else array[warp,it,7]),begin=start,end=end,duration=end-start,timer=timer,cubin=artifacts['PBASE' if name=='PBASE' else 'FRAME']['cubin_sha256'],local_bias_bound=None))
with (out/'EVENT_RAW.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(event_rows[0]));writer.writeheader();writer.writerows(event_rows)
save('CLOCK_RAW.json',clock_rows)
save('PAIRED_CLOCK_RAW.json',paired_clock_rows)
save('EVENT_QC.json',summaries)
save('STATUS.json',dict(status='COLLECTED',usable=usable, event_samples=len(event_rows),clock_rows=len(clock_rows),
                        gpu_executed=True, local_bias_bound_cycles=None, model_enabled=False))
assert not cmd('git','-C',str(REPO),'status','--porcelain')
