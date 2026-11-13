from pathlib import Path
import argparse,csv,hashlib,importlib.util,importlib.metadata,json,math,os,re,statistics as st,subprocess,sys,time
R=Path(__file__).resolve().parent;ROOT=R.parents[1];REPO=ROOT/'upstreams/flashinfer';FROZEN=R.parent/'cp4_event_baseline_freeze_20260929'
p=argparse.ArgumentParser();p.add_argument('--s',type=int,required=True);p.add_argument('--trial',type=int,required=True);p.add_argument('--cache',choices=['cold','warm'],required=True);p.add_argument('--clock',choices=['auto','locked1845'],required=True);p.add_argument('--mode',required=True);p.add_argument('--calls',type=int,default=100);p.add_argument('--batches',type=int,default=3);p.add_argument('--warmup',type=int,default=20);p.add_argument('--hot-k',action='store_true');p.add_argument('--evict-mult',type=int,default=4);p.add_argument('--allow-profiler-replay',action='store_true');a=p.parse_args()
tag=f'S{a.s}_{a.cache}_{a.clock}_{a.mode}_t{a.trial}';out=R/'runs'/tag
if a.allow_profiler_replay:
    ri=0
    while out.exists():ri+=1;out=R/'runs'/f'{tag}_replay{ri}'
out.mkdir(parents=True,exist_ok=False)
def save(n,x):(out/n).write_text(json.dumps(x,indent=2))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def cmd(*x):return subprocess.check_output(x,text=True).strip()
PIN='e77a4a0d276367895c3b50a642fd8f326c03fb72'
assert cmd('git','-C',str(REPO),'rev-parse','HEAD')==PIN and not cmd('git','-C',str(REPO),'status','--porcelain')
sys.path.insert(0,str(REPO));os.environ['FLASHINFER_WORKSPACE_BASE']=str(out/'jit_cache')
import torch,flashinfer
from flashinfer.gdn_prefill import chunk_gated_delta_rule
assert Path(flashinfer.__file__).resolve().is_relative_to(REPO)
torch.set_num_threads(1);props=torch.cuda.get_device_properties(0);assert props.multi_processor_count==148
inp=FROZEN/'frozen_inputs'/f'S{a.s}'/'inputs.pt';assert sha(inp)==json.loads((inp.parent/'storage_hash.json').read_text())['inputs.pt']
data=torch.load(inp,map_location='cpu',weights_only=True);manifest=json.loads((inp.parent/'inputs_manifest.json').read_text())
for i,b in enumerate(data['buffers']):
    for k,t in b.items():assert hashlib.sha256(t.numpy().tobytes()).hexdigest()==manifest['buffers'][i][k]['sha256']
for k,t in data['shared'].items():assert hashlib.sha256(t.numpy().tobytes()).hexdigest()==manifest['shared'][k]['sha256']
buffers=[{**{k:v.cuda() for k,v in b.items()},'output':torch.empty(a.s,16,128,dtype=torch.float16,device='cuda'),'state':torch.empty(1,16,128,128,dtype=torch.float32,device='cuda')} for b in data['buffers']]
shared={k:v.cuda() for k,v in data['shared'].items()};del data
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());torch.cuda.synchronize();initial_copy=shared['initial'].clone()
def invoke(bi,cp=True,output=None,state=None):
    b=buffers[bi]
    return chunk_gated_delta_rule(b['q'],b['k'],b['v'],g=shared['gate'],beta=shared['beta'],scale=1/math.sqrt(128),initial_state=shared['initial'],output_final_state=True,cu_seqlens=shared['cu_seqlens'],use_qk_l2norm_in_kernel=False,output=b['output'] if output is None else output,output_state=b['state'] if state is None else state,use_cp=cp)
with torch.cuda.stream(stream):
    for bi in range(4):invoke(bi)
stream.synchronize();gold=[(b['output'].clone(),b['state'].clone()) for b in buffers]
no=torch.empty_like(buffers[0]['output']);ns=torch.empty_like(buffers[0]['state'])
with torch.cuda.stream(stream):invoke(0,False,no,ns)
stream.synchronize();torch.testing.assert_close(no,gold[0][0],atol=.04,rtol=.04);torch.testing.assert_close(ns,gold[0][1],atol=.04,rtol=.04)
l2=int(props.L2_cache_size);evict=torch.empty(a.evict_mult*l2,dtype=torch.uint8,device='cuda') if a.cache=='cold' else None
batch=(torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True));pairs=[(torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)) for _ in range(a.calls)]
with torch.cuda.stream(stream):
    for e in (*batch,*(e for pair in pairs for e in pair)):e.record()
    if evict is not None:evict.fill_(90)
stream.synchronize();graph=torch.cuda.CUDAGraph(keep_graph=True);graph.enable_debug_mode()
with torch.cuda.graph(graph,stream=stream):
    batch[0].record()
    for j in range(a.calls):
        bi=j%4
        if a.hot_k:hot_sink=buffers[bi]['k'].sum(dim=-1)
        if evict is not None:evict.fill_(90)
        pairs[j][0].record();invoke(bi);pairs[j][1].record()
    batch[1].record()
dot=out/'chain.dot';graph.debug_dump(str(dot));graph.instantiate()
raw=dot.read_text();nodes=list(re.finditer(r'"(graph_\d+_node_\d+)"\[(.*?)\];',raw,re.S));edges=re.findall(r'"(graph_\d+_node_\d+)"\s*->\s*"(graph_\d+_node_\d+)"',raw)
assert set(edges)==set(zip([x.group(1) for x in nodes][:-1],[x.group(1) for x in nodes][1:]))
roles=[];entries=[];evnodes=0;gdnpos=[]
symbols=[('K1','CPDeltaRuleTPrecomputeSm100'),('K2','CPDeltaRuleMNPrecomputeUtcmma1Sm100'),('K3','CPDeltaRuleFixupUtcmmaSm100'),('K4','CPDeltaRulePrefillTcgen05Sm100')]
for ni,m in enumerate(nodes):
    body=m.group(2)
    if 'EVENT_RECORD' in body:evnodes+=1
    if 'label="{KERNEL' not in body:continue
    r=next((r for r,s in symbols if s in body),'evict_or_hot_touch');line=body.splitlines()[1];launch=line.split(' | ',2)[-1].rsplit('\\<\\<\\<',1);launch_text=launch[1].split('\\>')[0];cfg=launch_text.split(',');e={'role':r,'node':m.group(1),'grid':int(cfg[0]) if r!='evict_or_hot_touch' else None,'threads':int(cfg[1]) if r!='evict_or_hot_touch' else None,'smem':int(cfg[2]) if r!='evict_or_hot_touch' else None,'launch_config_raw':launch_text,'name':launch[0].strip()};entries.append(e)
    if r!='evict_or_hot_touch':roles.append(r);gdnpos.append(ni)
assert roles==['K1','K2','K3','K4']*a.calls and evnodes==2*a.calls+2
for j in range(a.calls):
    assert gdnpos[4*j+3]-gdnpos[4*j]==3,'Nodes inserted within complete CP chain'
    four=[e for e in entries if e['role']!='evict_or_hot_touch'][4*j:4*j+4]
    assert [e['threads'] for e in four]==[128,384,256,384] and [e['grid'] for e in four]==[a.s//64*16,128,32,128]
extra=sum(x['role']=='evict_or_hot_touch' for x in entries)
assert extra==a.calls*((1 if a.cache=='cold' else 0)+(1 if a.hot_k else 0)),(extra,a)
save('GRAPH_VALIDATION.json',{'strict_single_chain':True,'complete_CP_calls':a.calls,'kernel_nodes':len(entries),'event_nodes':evnodes,'extra_nodes_before_K1_only':extra,'no_nodes_between_K1_K4':True,'launches':entries,'dot_sha256':sha(dot)})
save('IDENTITY.json',{'pin':PIN,'S':a.s,'C':a.s//8,'P':8,'B':1,'H':16,'D':128,'dtype':'FP16','state_dtype':'FP32','trial':a.trial,'mode':a.mode,'cache':a.cache,'clock_request':a.clock,'calls':a.calls,'batches':a.batches,'hot_K_before_evict':a.hot_k,'L2_bytes':l2,'eviction_bytes':a.evict_mult*l2 if evict is not None else 0,'eviction_method':'uint8 GPU fill_(90) in same stream, outside CP Event' if evict is not None else 'none','flashinfer_import':flashinfer.__file__,'input_sha256':sha(inp),'input_content_signature':manifest['content_signature_sha256'],'runner_sha256':sha(__file__),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':str(props),'nsys':cmd('nsys','--version'),'kernel_modified':False,'model_enabled':False})
# Replaying the actual graph also applies the requested cold boundary in warmup.
for j in range(a.warmup):
    graph.replay();batch[1].synchronize()
rows=[];torch.cuda.cudart().cudaProfilerStart();measurement_start_wall_ns=time.time_ns()
for rep in range(a.batches):
    name=f'CP4C_{tag}_b{rep}';host_begin=time.time_ns();torch.cuda.nvtx.range_push(name);graph.replay();batch[1].synchronize();torch.cuda.nvtx.range_pop();host_end=time.time_ns()
    samples=[x.elapsed_time(y)*1000 for x,y in pairs]
    rows.append({'range':name,'batch':rep,'CP_Event_us':samples,'batch_with_eviction_and_events_us':batch[0].elapsed_time(batch[1])*1000,'calls':a.calls,'host_begin_wall_ns':host_begin,'host_end_wall_ns':host_end})
measurement_end_wall_ns=time.time_ns();torch.cuda.cudart().cudaProfilerStop();save('EVENT_SAMPLES.json',rows);save('MEASUREMENT_WALL_RANGE.json',{'begin_wall_ns':measurement_start_wall_ns,'end_wall_ns':measurement_end_wall_ns,'scope':'three measured graph replays; kernel/Event timing data remain separate'})
valid=[]
for bi,b in enumerate(buffers):
    torch.testing.assert_close(b['output'],gold[bi][0],atol=0,rtol=0);torch.testing.assert_close(b['state'],gold[bi][1],atol=0,rtol=0)
    assert torch.isfinite(b['output']).all().item() and torch.isfinite(b['state']).all().item();valid.append({'buffer':bi,'O_exact':True,'state_exact':True,'finite':True})
assert torch.equal(initial_copy,shared['initial']);save('CORRECTNESS.json',valid)
ss=sorted(x for row in rows for x in row['CP_Event_us']);mean=st.mean(ss)
summary={'S':a.s,'cache':a.cache,'clock':a.clock,'mode':a.mode,'trial':a.trial,'CP_Event_us':{'n':len(ss),'median':st.median(ss),'mean':mean,'p5':ss[int((len(ss)-1)*.05)],'p95':ss[int((len(ss)-1)*.95)],'cv_percent':100*st.pstdev(ss)/mean},'event_scope':'complete K1-K4 only, each eviction excluded; no internal kernel Events','model_enabled':False}
save('SUMMARY.json',summary);print(json.dumps(summary),flush=True)
assert not cmd('git','-C',str(REPO),'status','--porcelain')
