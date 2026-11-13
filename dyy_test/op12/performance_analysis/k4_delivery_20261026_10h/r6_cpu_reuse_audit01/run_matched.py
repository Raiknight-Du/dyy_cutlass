"""Six-role, same-address K4 experiment; never edits upstream. GPU guard required."""
import argparse, csv, hashlib, importlib.util, inspect, json, os, re, statistics, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OP=ROOT.parents[1]; REPO=OP/'upstreams/flashinfer'
P=argparse.ArgumentParser(); P.add_argument('--out',required=True); P.add_argument('--mode',choices=['screen','formal','profile'],required=True); P.add_argument('--trial',type=int,default=0); P.add_argument('--profile',choices=['B0','V0'])
a=P.parse_args(); out=Path(a.out)
if a.mode=='profile': out=out/f'replay_pid{os.getpid()}'
out.mkdir(parents=True,exist_ok=False)
def save(name,obj): (out/name).write_text(json.dumps(obj,indent=2))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cmd(*x): return subprocess.check_output(x,text=True).strip()
def guard():
    assert os.environ.get('K1_GPU_SESSION')
    assert not (Path(os.environ['K1_GUARD_DIR'])/'INVALID_COMPETITION').exists()
guard(); PIN='e77a4a0d276367895c3b50a642fd8f326c03fb72'
assert cmd('git','-C',str(REPO),'rev-parse','HEAD')==PIN
assert not cmd('git','-C',str(REPO),'status','--porcelain')
sys.path.insert(0,str(REPO)); os.environ['FLASHINFER_WORKSPACE_BASE']=str(out/'jit_cache')
import numpy as np
import torch, flashinfer, cutlass, cutlass.cute as cute
from cutlass.base_dsl.compiler import DumpDir
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.gdn_kernels.blackwell import gdn_cp_prefill as cp
assert Path(flashinfer.__file__).resolve().is_relative_to(REPO)
torch.set_num_threads(1); prop=torch.cuda.get_device_properties(0)
assert prop.multi_processor_count==148 and prop.major==10 and 'B200' in prop.name
save('ENVIRONMENT.json',dict(host=cmd('hostname'),GPU=str(prop),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cutlass=getattr(cutlass,'__version__','unknown'),nvcc=cmd('/usr/local/cuda/bin/nvcc','--version'),smi=cmd('nvidia-smi','-q'),pin=PIN,import_path=flashinfer.__file__))
(out/'mark_state_layout.py').write_text(inspect.getsource(cp._mark_state_layout))
inp=OP/'performance_analysis/cp4_event_baseline_freeze_20260929/frozen_inputs/S32768/inputs.pt'
assert sha(inp)==json.loads((inp.parent/'storage_hash.json').read_text())['inputs.pt']
data=torch.load(inp,map_location='cpu',weights_only=True); manifest=json.loads((inp.parent/'inputs_manifest.json').read_text())
for i,buf in enumerate(data['buffers']):
    for k,t in buf.items(): assert hashlib.sha256(t.numpy().tobytes()).hexdigest()==manifest['buffers'][i][k]['sha256']
for k,t in data['shared'].items(): assert hashlib.sha256(t.numpy().tobytes()).hexdigest()==manifest['shared'][k]['sha256']
buffers=[{**{k:v.cuda() for k,v in b.items()},'output':torch.empty(32768,16,128,dtype=torch.float16,device='cuda'),'state':torch.empty(1,16,128,128,dtype=torch.float32,device='cuda')} for b in data['buffers']]
shared={k:v.cuda() for k,v in data['shared'].items()}; del data
initial=shared['initial'].clone(); trace=torch.full((16,),-1,dtype=torch.int64,device='cuda'); evict=torch.empty(4*int(prop.L2_cache_size),dtype=torch.uint8,device='cuda')
stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream()); torch.cuda.synchronize()
origclass=cp.CPDeltaRulePrefillTcgen05Sm100
artifacts={}; compiled={}; dsl={}; objects={}; runtime={}; buffer_id=0
def role(kernel):
    n=type(kernel).__name__
    return {'CPDeltaRuleTPrecomputeSm100':'K1','CPDeltaRuleMNPrecomputeUtcmma1Sm100':'K2','CPDeltaRuleFixupUtcmmaSm100':'K3','CPDeltaRulePrefillTcgen05Sm100':'K4'}[n]
def inslines(sass):
    return [(m[1],m[2].strip()) for line in sass.splitlines() if (m:=re.match(r'\s*/\*([0-9a-f]+)\*/\s+(.*?);',line))]
def audit(dst):
    cub=next(dst.glob('*.cubin')); sass=next(dst.glob('*.sass')).read_text(); res=cmd('/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(cub)); ins=inslines(sass)
    assert len(ins)>1000
    normal=[re.sub(r'`\(kernel_cutlass_kernel[^)]+\)','`(<entry>)',s) for pc,s in ins]
    names=['LDL','STL','UTCHMMA','UTMALDG','UTMASTG','LDTM','STTM','HMMA','LDG','STG','F2FP','BAR','UTCBAR','SYNCS','FENCE','MEMBAR','FFMA','FMUL','FADD','LDSM','STSM','LDS','STS','PRMT']
    counts={k:sum(bool(re.match(r'^(?:@!?\w+\s+)?'+k+r'(?:\.|\s)',s)) for pc,s in ins) for k in names}
    return dict(cubin=str(cub),cubin_sha256=sha(cub),resources=res,registers=re.findall(r'REG:(\d+)',res),stack=re.findall(r'STACK:(\d+)',res),counts=counts,normalized_sha256=hashlib.sha256('\n'.join(normal).encode()).hexdigest(),clock_pc=[pc for pc,s in ins if 'SR_CLOCKLO' in s],smid_pc=[pc for pc,s in ins if 'SR_VIRTUALSMID' in s])
def do_compile(name,kernel,args,options):
    guard(); dst=out/'binary'/name; dst.mkdir(parents=True)
    opts=(*options,cute.KeepPTX(True),cute.KeepCUBIN(True),cute.KeepSASS(True),DumpDir(str(dst)))
    fun=cute.compile[opts](kernel,*args); artifacts[name]=audit(dst); save('COMPILE_AUDIT.json',artifacts)
    return fun
def compiler(kernel,*args,compile_options):
    r=role(kernel); objects[r]=kernel; dsl[r]=(args,compile_options)
    if r=='K4': fun=do_compile('B0',kernel,args,compile_options)
    else: fun=cute.compile[compile_options](kernel,*args)
    def wrapped(*values):
        runtime[buffer_id,r]=values
        return fun(*values)
    compiled[r]=(fun,wrapped); return wrapped
cp.get_cached_compile=lambda kernel,options: compiled.get(role(kernel),(None,None))[1]
cp.cached_compile=compiler
# Separate rotating-buffer workspace allocations; all six roles reuse each allocation.
origbuf=cp._get_cache_buf
cp._get_cache_buf=lambda name,*args,**kw: origbuf(name+f'_k4_matched_b{buffer_id}',*args,**kw)
gold=[]
for buffer_id in range(4):
    b=buffers[buffer_id]
    with torch.cuda.stream(stream):
        chunk_gated_delta_rule(b['q'],b['k'],b['v'],g=shared['gate'],beta=shared['beta'],scale=1/(128**.5),initial_state=shared['initial'],output_final_state=True,cu_seqlens=shared['cu_seqlens'],use_qk_l2norm_in_kernel=False,output=b['output'],output_state=b['state'],use_cp=True)
    stream.synchronize(); gold.append((b['output'].clone(),b['state'].clone())); torch.cuda.synchronize()
assert artifacts['B0']['cubin_sha256']=='06ebd78963371a30adac142c06b86950ec300b6cb762a209c5afc67628941814','FROZEN_B0_IDENTITY_MISMATCH'
def info(x):
    if isinstance(x,torch.Tensor): return dict(ptr=x.data_ptr(),shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),bytes=x.numel()*x.element_size())
    return str(x)
bindings={str(bi):{r:[info(x) for x in runtime[bi,r]] for r in ['K1','K2','K3','K4']} for bi in range(4)}
for bi in range(4):
    fixed=runtime[bi,'K4'][7]
    assert fixed.is_contiguous() and fixed.shape[-1]==128 and fixed.stride()[-1]==1 and fixed.data_ptr()%16==0
assert len({runtime[bi,'K4'][7].data_ptr() for bi in range(4)})==4
save('ADDRESS_BINDINGS.json',dict(shared_by_all_variants=True,method='capture original public runtime arguments once per rotating buffer; direct same compiled prefix',buffers=bindings))
kernel=objects['K4']; args,options=dsl['K4']; functions={'B0':compiled['K4'][0]}; variants=['B0','V0'] if a.mode=='profile' else ['B0','V0','B1','V1','P0','PV']
classes={}
for module_name in ['B1','INIT_A_PAIR_W4_C16']:
    spec=importlib.util.spec_from_file_location('v_'+module_name,ROOT/'variants'/module_name/'source.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); classes[module_name]=m.K4Probe
for name in variants[1:]:
    target=kernel
    if name in ['B1','V1','P0','PV']:
        target=classes['B1' if name in ['B1','V1'] else 'INIT_A_PAIR_W4_C16'].__new__(classes['B1' if name in ['B1','V1'] else 'INIT_A_PAIR_W4_C16'])
        target.__dict__.update(kernel.__dict__)
    ca=list(args)
    if name.startswith('V') or name=='PV':
        fixed=cute.runtime.from_dlpack(runtime[0,'K4'][7],assumed_align=16,enable_tvm_ffi=True)
        cp._mark_state_layout(fixed,False,128); ca[7]=fixed
    if name in ['B1','V1','P0','PV']: ca.insert(-1,cute.runtime.from_dlpack(trace,assumed_align=16,enable_tvm_ffi=True))
    functions[name]=do_compile(name,target,ca,options)
    print('COMPILED',name,artifacts[name]['cubin_sha256'],flush=True)
hist={x['sha256'] for x in json.loads((ROOT/'contract_snapshot/EVIDENCE_INDEX.json').read_text())['sources'] if 'cubin' in x['id']}
new={x['cubin_sha256'] for x in artifacts.values()}-hist
save('COMPILE_BUDGET.json',dict(candidates=len(variants),new_unique=len(new),new_hashes=sorted(new),repeated_process_builds_not_new_candidates=True)); assert len(new)<=4
review={}; allowed={}
for name in variants:
    rec=artifacts[name]; base='V0' if name in ['V1','PV'] else 'B0'; br=artifacts[base]; reasons=[]
    for k in ['LDL','STL','UTCHMMA','UTMALDG','UTMASTG','LDTM','STTM','HMMA','F2FP','BAR','UTCBAR','SYNCS','FENCE','MEMBAR']:
        if rec['counts'][k]!=br['counts'][k]: reasons.append('CHANGED_'+k)
    if max(map(int,rec['stack']),default=0)>max(map(int,br['stack']),default=0): reasons.append('ADDED_STACK')
    if max(map(int,rec['registers']),default=0)>max(map(int,br['registers']),default=0): reasons.append('ADDED_REGISTERS')
    if name in ['B1','V1'] and rec['normalized_sha256']!=br['normalized_sha256']: reasons.append('EMPTY_INTERFACE_SCHEDULE_CHANGED')
    if name in ['P0','PV'] and len(rec['clock_pc'])!=2: reasons.append('CLOCK_ENDPOINT_COUNT')
    review[name]=dict(structural_pass=not reasons,reasons=reasons,relative_to=base,local_bias_bound_cycles=None)
    allowed[name]=not reasons
save('COMPILE_REVIEW.json',review)
def call4(name,bi):
    v=runtime[bi,'K4']
    return functions[name](*v) if name in ['B0','V0'] else functions[name](*v[:-1],trace,v[-1])
def chain(name,bi,scope=None,events=None):
    if scope=='CP_EVENT': events[0].record()
    for r in ['K1','K2','K3']: compiled[r][0](*runtime[bi,r])
    if scope=='K4_EVENT': events[0].record()
    call4(name,bi)
    if scope in ['CP_EVENT','K4_EVENT']: events[1].record()
correct={}
for name in variants:
    if not allowed[name]: continue
    correct[name]=[]
    for bi in range(4):
        with torch.cuda.stream(stream): chain(name,bi)
        stream.synchronize(); b=buffers[bi]
        ok=torch.equal(b['output'],gold[bi][0]) and torch.equal(b['state'],gold[bi][1]); correct[name].append(ok); assert ok,'BITWISE_MISMATCH '+name
save('CORRECTNESS.json',correct)
save('IDENTITY.json',dict(pin=PIN,shape=[1,16,32768,128],C=4096,P=8,L=64,input_sha256=sha(inp),content_signature=manifest['content_signature_sha256'],evict_bytes=evict.numel(),trial=a.trial,requested_MHz=1845,model_enabled=False,specialization={k:str(getattr(kernel,k,None)) for k in ['needs_initial_state','store_final_state','num_regs_compute_group_0','num_regs_compute_group_1','num_regs_other','tmem_alloc_cols']}))
if a.mode=='screen': print('SCREEN_DONE',review,flush=True); sys.exit(0)
approval=json.loads((ROOT/'APPROVAL.json').read_text())
for name in variants:
    assert artifacts[name]['cubin_sha256']==approval['hashes'][name]
    allowed[name]=allowed[name] and approval['allowed'][name]
assert allowed['B0'] and allowed['V0']
graphs={}; graph_records=[]
scopes=['CLOCK'] if a.mode=='profile' else ['CP_EVENT','K4_EVENT','CLOCK']
for name in variants:
    if not allowed[name]: continue
    for bi in range(4):
        for scope in scopes:
            if scope=='CLOCK' and a.mode!='profile' and name not in ['P0','PV']: continue
            events=[torch.cuda.Event(enable_timing=True,external=True) for _ in range(2)]
            with torch.cuda.stream(stream):
                for e in events: e.record()
            stream.synchronize(); g=torch.cuda.CUDAGraph(keep_graph=True); g.enable_debug_mode()
            with torch.cuda.graph(g,stream=stream):
                trace.fill_(-1); buffers[bi]['output'].fill_(float('nan')); buffers[bi]['state'].fill_(float('nan')); evict.fill_(90)
                chain(name,bi,scope,events)
            path=out/f'{name}_b{bi}_{scope}.dot'; g.debug_dump(str(path)); g.instantiate(); raw=path.read_text()
            nodes=list(re.finditer(r'"(graph_\d+_node_\d+)"\[(.*?)\];',raw,re.S)); roles=[]; positions=[]; launches=[]
            for ni,node in enumerate(nodes):
                body=node.group(2)
                for r,sym in [('K1','CPDeltaRuleTPrecomputeSm100'),('K2','CPDeltaRuleMNPrecomputeUtcmma1Sm100'),('K3','CPDeltaRuleFixupUtcmmaSm100'),('K4','CPDeltaRulePrefillTcgen05Sm100' if name in ['B0','V0'] else 'K4Probe')]:
                    if 'label="{KERNEL' in body and sym in body:
                        roles.append(r); positions.append(ni); cfg=body.splitlines()[1].rsplit('\\<\\<\\<',1)[1].split('\\>')[0].split(','); launches.append([int(x) for x in cfg[:3]])
            assert roles==['K1','K2','K3','K4']; assert positions[-1]-positions[0]==(4 if scope=='K4_EVENT' else 3)
            assert [x[0] for x in launches]==[8192,128,32,128] and [x[1] for x in launches]==[128,384,256,384]
            graph_records.append(dict(variant=name,buffer=bi,scope=scope,roles=roles,node_positions=positions,launch=launches,sha256=sha(path)))
            graphs[name,bi,scope]=(g,events)
save('GRAPHS.json',graph_records)
def execute(name,bi,scope):
    guard(); g,events=graphs[name,bi,scope]
    with torch.cuda.stream(stream): g.replay()
    stream.synchronize()
    if scope=='CLOCK':
        raw=trace.cpu().tolist()
        if name in ['P0','PV']: assert 0<=raw[0]<=raw[1] and raw[2]>=0
        return raw
    return events[0].elapsed_time(events[1])*1000
if a.mode=='profile':
    for i in range(20): execute(a.profile,i%4,'CLOCK')
    torch.cuda.nvtx.range_push('K4_MATCHED'); execute(a.profile,0,'CLOCK'); torch.cuda.nvtx.range_pop()
    assert torch.equal(buffers[0]['output'],gold[0][0]) and torch.equal(buffers[0]['state'],gold[0][1])
    save('PROFILE_COMPLETE.json',dict(role=a.profile,hash=artifacts[a.profile]['cubin_sha256'],full_CP=True,cold_before_K1=True)); sys.exit(0)
rows=[]; clockrows=[]; qcs={}; rng=np.random.default_rng(20261024)
def stats(vals):
    x=np.array(vals,dtype=float); med=float(np.median(x)); p5,p95=np.percentile(x,[5,95]); cv=float(np.std(x)/np.mean(x)*100)
    return dict(median=med,p5=float(p5),p95=float(p95),cv_pct=cv,spread_pct=float((p95-p5)/med*100),stable=bool(cv<=5 and (p95-p5)/med<=.1))
def pairtest(scope,A,B):
    for i in range(20):
        for n in [A,B]: execute(n,i%4,scope)
    ratios=[]; deltas=[]; vals={A:[],B:[]}
    for block in range(30):
        order=[A,B,B,A] if block%2==0 else [B,A,A,B]; local={A:[],B:[]}
        for member,name in enumerate(order):
            t=execute(name,block%4,scope); local[name].append(t); vals[name].append(t)
            rows.append(dict(process=a.trial,series=scope,pair=A+'__'+B,block=block,member=member,buffer=block%4,variant=name,us=t))
        av,bv=[statistics.mean(local[n]) for n in [A,B]]; ratios.append(100*(bv/av-1)); deltas.append(av-bv)
    ci=np.percentile(np.median(np.array(ratios)[rng.integers(0,30,(10000,30))],axis=1),[2.5,97.5])
    result=dict(A=stats(vals[A]),B=stats(vals[B]),paired_change_percent=float(np.median(ratios)),CI95_percent=ci.tolist(),paired_saving_us=float(np.median(deltas)),equivalent=bool(ci[0]>=-2 and ci[1]<=2),stable=stats(vals[A])['stable'] and stats(vals[B])['stable'])
    qcs[scope+'_'+A+'__'+B]=result
    with (out/'EVENT_RAW.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    save('EVENT_QC.json',qcs); print('PAIR',scope,A,B,result,flush=True)
for scope in ['CP_EVENT','K4_EVENT']:
    pairtest(scope,'B0','V0')
    for base,probe in [('B0','P0'),('V0','PV')]:
        if allowed.get(probe): pairtest(scope,base,probe)
# B1/V1 Event waived only when exact normalized native sequence and resources pass.
qcpass=all(allowed.get(p,False) and all(qcs[s+'_'+b+'__'+p]['equivalent'] and qcs[s+'_'+b+'__'+p]['stable'] for s in ['CP_EVENT','K4_EVENT']) for b,p in [('B0','P0'),('V0','PV')])
if qcpass:
    for i in range(20):
        for n in ['P0','PV']: execute(n,i%4,'CLOCK')
    for block in range(30):
        order=['P0','PV','PV','P0'] if block%2==0 else ['PV','P0','P0','PV']
        for member,name in enumerate(order):
            raw=execute(name,block%4,'CLOCK')
            clockrows.append(dict(process=a.trial,invocation=f'{a.trial}_{block}_{member}',block=block,member=member,buffer=block%4,variant=name,CTA=16,warp=4,lane=0,SMID=raw[2],start_cycle=raw[0],end_cycle=raw[1],cycles=raw[1]-raw[0],cubin=artifacts[name]['cubin_sha256'],backend='native_clock64',local_bias_bound_cycles=None))
    save('CLOCK_RAW.json',clockrows)
for bi in range(4): assert torch.equal(buffers[bi]['output'],gold[bi][0]) and torch.equal(buffers[bi]['state'],gold[bi][1])
assert torch.equal(initial,shared['initial']) and not cmd('git','-C',str(REPO),'status','--porcelain')
save('SUMMARY.json',dict(status='COLLECTED',Event_samples=len(rows),clock_records=len(clockrows),clock_branch='OBSERVATION_ONLY' if qcpass else 'SKIPPED_QC_FAIL',empty_interface_Event_waiver='identical normalized SASS and resource review',correctness_bitwise=True,model_enabled=False))
print('COMPLETE',a.trial,len(rows),len(clockrows),flush=True)
