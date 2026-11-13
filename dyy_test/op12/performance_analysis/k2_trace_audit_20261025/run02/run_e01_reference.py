"""Complete-CP clock64 screening/collection. One target, one CTA, no chain-internal nodes."""
import argparse,csv,hashlib,json,math,os,re,statistics,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;OP=Path('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12');REPO=OP/'upstreams/flashinfer'
p=argparse.ArgumentParser();p.add_argument('--kernel',choices=['K1','K2','K4'],default='K2');p.add_argument('--out',required=True);p.add_argument('--s',type=int,default=32768);p.add_argument('--trial',type=int,default=0);p.add_argument('--warmup',type=int,default=20);p.add_argument('--samples',type=int,default=100);p.add_argument('--variants',default='B0,B1,E');a=p.parse_args()
TARGET=a.kernel;NW=4 if TARGET=='K1' else 12;TI=int(TARGET[1])-1
CLASS={'K1':'CPDeltaRuleTPrecomputeSm100','K2':'CPDeltaRuleMNPrecomputeUtcmma1Sm100','K4':'CPDeltaRulePrefillTcgen05Sm100'}[TARGET]
GETTER={'K1':'_get_t_precompute_kernel','K2':'_get_mn_precompute_kernel','K4':'_get_prefill_kernel'}[TARGET]
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
def save(n,x):(out/n).write_text(json.dumps(x,indent=2))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cmd(*x):return subprocess.check_output(x,text=True).strip()
assert os.environ.get('K1_GPU_SESSION')
assert (R/'PRE_RUN_DECISION.json').is_file()
assert TARGET=='K2' and a.variants=='B0,B1,E', 'E01 scope only'
PIN='e77a4a0d276367895c3b50a642fd8f326c03fb72'
assert cmd('git','-C',str(REPO),'rev-parse','HEAD')==PIN and not cmd('git','-C',str(REPO),'status','--porcelain')
sys.path[:0]=[str(REPO),str(R/(TARGET+'_CLOCK'))]
os.environ['FLASHINFER_WORKSPACE_BASE']=str(out/'jit_cache')
import torch,flashinfer,cutlass,cutlass.cute as cute
from cutlass.base_dsl.compiler import DumpDir
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.gdn_kernels.blackwell import gdn_cp_prefill as cp
import variants
assert Path(flashinfer.__file__).resolve().is_relative_to(REPO)
torch.set_num_threads(1);prop=torch.cuda.get_device_properties(0)
assert prop.multi_processor_count==148 and prop.major==10 and 'B200' in prop.name
inp=OP/'performance_analysis/cp4_event_baseline_freeze_20260929/frozen_inputs'/f'S{a.s}'/'inputs.pt'
assert sha(inp)==json.loads((inp.parent/'storage_hash.json').read_text())['inputs.pt']
data=torch.load(inp,map_location='cpu',weights_only=True);manifest=json.loads((inp.parent/'inputs_manifest.json').read_text())
for i,b in enumerate(data['buffers']):
    for k,t in b.items():assert hashlib.sha256(t.numpy().tobytes()).hexdigest()==manifest['buffers'][i][k]['sha256']
for k,t in data['shared'].items():assert hashlib.sha256(t.numpy().tobytes()).hexdigest()==manifest['shared'][k]['sha256']
buffers=[{**{k:v.cuda() for k,v in b.items()},'output':torch.empty(a.s,16,128,dtype=torch.float16,device='cuda'),'state':torch.empty(1,16,128,128,dtype=torch.float32,device='cuda')} for b in data['buffers']]
shared={k:v.cuda() for k,v in data['shared'].items()};del data
trace=torch.full((12,66,12,2,2),-1,dtype=torch.int64,device='cuda')
evict=torch.empty(4*int(prop.L2_cache_size),dtype=torch.uint8,device='cuda')
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());torch.cuda.synchronize()
initial=shared['initial'].clone()
origclass=getattr(cp,CLASS);orig_get=cp.get_cached_compile;orig_compile=cp.cached_compile
target_cache={};active='B0';artifacts={};target_events=None;last_mn=None
def is_target(kernel):return isinstance(kernel,origclass)
def get(kernel,options):
    if is_target(kernel):return target_cache.get(active)
    return orig_get(kernel,options)
def compile_target(kernel,*args,compile_options):
    if not is_target(kernel):return orig_compile(kernel,*args,compile_options=compile_options)
    dst=out/'binary'/active;dst.mkdir(parents=True)
    opts=(*compile_options,cute.KeepPTX(True),cute.KeepCUBIN(True),cute.KeepSASS(True),DumpDir(str(dst)))
    if active=='B0':
        fun=cute.compile[opts](kernel,*args)
    else:
        tr=cute.runtime.from_dlpack(trace,assumed_align=16,enable_tvm_ffi=True)
        fun=cute.compile[opts](kernel,*args[:-1],tr,args[-1])
    def wrapper(*runtime):
        global last_mn
        last_mn=(runtime[4],runtime[5])
        if target_events is not None:target_events[0].record()
        result=fun(*runtime) if is_base else fun(*runtime[:-1],trace,runtime[-1])
        if target_events is not None:target_events[1].record()
        return result
    is_base=active=='B0'
    target_cache[active]=wrapper
    cub=next(dst.glob('*.cubin'));sass=next(dst.glob('*.sass')).read_text()
    res=cmd('/usr/local/cuda-13.0/bin/cuobjdump','--dump-resource-usage',str(cub))
    record=dict(cubin_sha256=sha(cub),resources=res,stack=[int(x) for x in re.findall(r'STACK:(\d+)',res)],local_ops=len(re.findall(r'\b(?:LDL|STL)(?:\.|\s)',sass)),clock_sites=re.findall(r'/\*([0-9a-f]+)\*/[^\n]*SR_CLOCKLO',sass),smid_sites=re.findall(r'/\*([0-9a-f]+)\*/[^\n]*SR_VIRTUALSMID',sass),mma_static=len(re.findall(r'\bUTCHMMA(?:\.|\s)',sass)),tma_static=len(re.findall(r'\bUTMALDG\.',sass)),tmem_load_static=len(re.findall(r'\bLDTM\.',sass)),tmem_store_static=len(re.findall(r'\bSTTM\.',sass)))
    record['hmma_static']=len(re.findall(r'\bHMMA\.',sass))
    if TARGET=='K1':
        ins=[x for x in sass.splitlines() if re.match(r'\s*/\*[0-9a-f]+\*/',x)];hm=[i for i,x in enumerate(ins) if 'HMMA.' in x]
        record['Wkk_LDSM']=[sum('LDSM.' in x for x in ins[:hm[0]]),sum('LDSM.' in x for x in ins[hm[0]:hm[63]+1])]
    artifacts[active]=record;save('COMPILE_AUDIT.json',artifacts)
    if active!='B0':
        base=artifacts['B0']
        if record['local_ops']>base['local_ops'] or max(record['stack'],default=0)>max(base['stack'],default=0):raise RuntimeError('New local/spill: '+active)
        assert re.findall(r'(?:REG|STACK|SHARED|LOCAL):\d+',res)==re.findall(r'(?:REG|STACK|SHARED|LOCAL):\d+',base['resources']), 'Resource change'
        for key in ('mma_static','tma_static','tmem_load_static','tmem_store_static','hmma_static'):
            if record[key]!=base[key]:raise RuntimeError('Native work changed: '+active+' '+key)
        if TARGET=='K1' and record['Wkk_LDSM']!=base['Wkk_LDSM']:raise RuntimeError('Wkk LDSM/HMMA scheduling changed')
    return wrapper
cp.get_cached_compile=get;cp.cached_compile=compile_target
def select(v):
    global active
    active=v;setattr(cp,CLASS,origclass if v=='B0' else getattr(variants,TARGET+'Clock'+v))
    getattr(cp,GETTER).cache_clear()
def invoke(bi):
    b=buffers[bi]
    return chunk_gated_delta_rule(b['q'],b['k'],b['v'],g=shared['gate'],beta=shared['beta'],scale=1/math.sqrt(128),initial_state=shared['initial'],output_final_state=True,cu_seqlens=shared['cu_seqlens'],use_qk_l2norm_in_kernel=False,output=b['output'],output_state=b['state'],use_cp=True)
graphs={};gold=[];gold_mn=[];modes=a.variants.split(',')
for v in modes:
    select(v)
    with torch.cuda.stream(stream):
        for bi in range(4):
            invoke(bi)
            if v=='B0':gold_mn.append(tuple(t.clone() for t in last_mn))
            else:
                assert all(torch.equal(t,g) for t,g in zip(last_mn,gold_mn[bi])),v+' changes M/N workspace'
    stream.synchronize()
    if v=='B0':
        gold=[(b['output'].clone(),b['state'].clone()) for b in buffers];torch.cuda.synchronize()
    else:
        for bi,b in enumerate(buffers):
            assert torch.equal(b['output'],gold[bi][0]) and torch.equal(b['state'],gold[bi][1]),v+' changes math'
    for layout in ('G0','G1'):
      for bi in range(4):
        start,end,ks,ke=[torch.cuda.Event(enable_timing=True,external=True) for _ in range(4)]
        with torch.cuda.stream(stream):
            for evt in (start,end,ks,ke):evt.record()
        stream.synchronize()
        target_events=(ks,ke) if layout=='G1' else None
        g=torch.cuda.CUDAGraph(keep_graph=True);g.enable_debug_mode()
        with torch.cuda.graph(g,stream=stream):
            trace.fill_(-1);evict.fill_(90);start.record();invoke(bi);end.record()
        target_events=None
        path=out/f'{v}_{layout}_b{bi}.dot';g.debug_dump(str(path));g.instantiate()
        raw=path.read_text();nodes=list(re.finditer(r'"(graph_\d+_node_\d+)"\[(.*?)\];',raw,re.S))
        roles=[];positions=[];launches=[]
        symbols=[('K1','CPDeltaRuleTPrecomputeSm100'),('K2','CPDeltaRuleMNPrecomputeUtcmma1Sm100' if v=='B0' else 'K2Clock'+v),('K3','CPDeltaRuleFixupUtcmmaSm100'),('K4','CPDeltaRulePrefillTcgen05Sm100')]
        for ni,n in enumerate(nodes):
            body=n.group(2)
            for role,sym in symbols:
                if 'label="{KERNEL' in body and sym in body:
                    roles.append(role);positions.append(ni)
                    cfg=body.splitlines()[1].rsplit('\\<\\<\\<',1)[1].split('\\>')[0].split(',')
                    launches.append([int(x) for x in cfg[:3]])
        assert roles==['K1','K2','K3','K4'],(v,layout,roles)
        assert positions[-1]-positions[0]==(3 if layout=='G0' else 5),(layout,positions)
        if layout=='G1':
            assert positions[1]-positions[0]==2 and positions[2]-positions[1]==2
            assert 'EVENT' in nodes[positions[1]-1].group(2) and 'EVENT' in nodes[positions[1]+1].group(2)
        assert [x[0] for x in launches]==[a.s//64*16,128,32,128]
        assert [x[1] for x in launches]==[128,384,256,384]
        if v=='B0' and bi==0 and layout=='G0':base_smem=launches[TI][2]
        assert launches[TI][2]==base_smem
        graphs[(v,layout,bi)]=(g,start,end,ks,ke)
save('IDENTITY.json',dict(kernel=TARGET,pin=PIN,shape=[1,16,a.s,128],C=a.s//8,P=8,grid=[a.s//64*16,128,32,128],input_sha256=sha(inp),content_signature=manifest['content_signature_sha256'],backend='native clock64',sample_CTA=0,evict_bytes=evict.numel(),clock_request_MHz=1845,trial=a.trial,vars=vars(a),model_enabled=False,chain_verified=True))
save('EVENT_LAYOUT_AUDIT.json',dict(chain=['K1','K2','K3','K4'],G0='CP begin/end only',G1='CP begin,K1,K2 begin,K2,K2 end,K3,K4,CP end',graph_dot_count=len(graphs),same_variant_MN_consumed_by_suffix=True,eviction='before CP',copyback='after CP',local_bias_bound='UNKNOWN'))
for rep in range(a.warmup):
    for v in modes:
        for layout in ('G0','G1'):graphs[v,layout,rep%4][0].replay()
    stream.synchronize()
rows=[];records=[]
generation=json.loads((R/'K2_CLOCK/GENERATION.json').read_text())
stages=generation['stages']
pairs=[(( 'B0',layout),(v,layout)) for layout in ('G0','G1') for v in modes if v!='B0']
pairs.append((('B0','G0'),('B0','G1')))
for rep in range(a.samples):
    for pair_id,(aa,bb) in enumerate(pairs):
        ordering=[aa,bb,bb,aa] if (rep+pair_id+a.trial)%2==0 else [bb,aa,aa,bb]
        bi=rep%4
        for position,(v,layout) in enumerate(ordering):
            if (Path(os.environ['K1_GUARD_DIR'])/'INVALID_COMPETITION').exists():raise RuntimeError('Guard invalid')
            g,start,end,ks,ke=graphs[v,layout,bi]
            g.replay();end.synchronize()
            invocation=f'{a.trial}:{pair_id}:{rep}:{position}'
            rows.append(dict(trial=a.trial,pair_id=pair_id,pair_A=':'.join(aa),pair_B=':'.join(bb),rep=rep,position=position,variant=v,layout=layout,buffer=bi,invocation=invocation,CP_Event_us=start.elapsed_time(end)*1000,K2_Event_us=ks.elapsed_time(ke)*1000 if layout=='G1' else ''))
            if v in ('E','S'):
                raw=trace.cpu().tolist()
                for w in range(8):
                    assert raw[w][65][0][0][0]>=0
                    for it in range(66):
                        for st in (0,4,8):
                            b,e=raw[w][it][st]
                            if b[0]<0 and e[0]<0:continue
                            assert 0<=b[0]<=e[0] and b[1]==e[1]
                            records.append(dict(trial=a.trial,invocation=invocation,variant=v,layout=layout,S=a.s,cta=0,warp=w,lane=0,iteration=it,stage=stages[str(st)],smid=b[1],start_cycle=str(b[0]),end_cycle=str(e[0]),cycles=e[0]-b[0],binary_sha256=artifacts[v]['cubin_sha256']))
            for key,gi in [('output',0),('state',1)]:assert torch.equal(buffers[bi][key],gold[bi][gi])
    save('CLOCK_RECORDS.json',records)
    with (out/'EVENT_SAMPLES.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    if rep%20==19:print(json.dumps(dict(trial=a.trial,blocks=rep+1,rows=len(rows))),flush=True)
assert torch.equal(initial,shared['initial'])
save('SUMMARY.json',dict(records=len(records),event_rows=len(rows),correctness_exact=True,MN_workspace_correctness_exact=True,model_enabled=False,local_bias_bound=None))
print(json.dumps(dict(rows=len(rows),records=len(records),correctness=True)),flush=True)
assert not cmd('git','-C',str(REPO),'status','--porcelain')

