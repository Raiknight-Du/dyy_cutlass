"""Full original CP history + one private, same-work K4 schedule variant.
Counter mechanism verification only. No clock instrumentation in K4.
"""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,inspect,json,os,re,subprocess,sys
R=Path(__file__).resolve().parent;OP=R.parents[1];REPO=OP/'upstreams/flashinfer'
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['B0','KSJOIN','QSJOIN','GATEJOIN'],required=True)
p.add_argument('--out',type=Path,required=True);p.add_argument('--profile',action='store_true');p.add_argument('--samples',type=int,default=30)
a=p.parse_args();assert os.environ.get('K4_5H_ADMITTED')=='1'
out=a.out/(f'replay_pid{os.getpid()}' if a.profile else 'run');out.mkdir(parents=True,exist_ok=False)
def sh(*z):return subprocess.check_output(z,text=True).strip()
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(n,x):(out/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
assert sh('git','-C',str(REPO),'rev-parse','HEAD')=='e77a4a0d276367895c3b50a642fd8f326c03fb72'
assert not sh('git','-C',str(REPO),'status','--porcelain')
sys.path.insert(0,str(REPO));os.environ['FLASHINFER_WORKSPACE_BASE']=str(out/'jit_cache')
import torch,cutlass
import cutlass.cute as cute
from cutlass.base_dsl.compiler import DumpDir
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.gdn_kernels.blackwell import gdn_cp_prefill as cp
torch.set_num_threads(1);torch.cuda.set_device(0);prop=torch.cuda.get_device_properties(0)
inp=OP/'performance_analysis/cp4_event_baseline_freeze_20260929/frozen_inputs/S32768/inputs.pt'
assert sha(inp)==json.loads((inp.parent/'storage_hash.json').read_text())['inputs.pt']
data=torch.load(inp,map_location='cpu',weights_only=True)
buffers=[{**{k:v.cuda() for k,v in b.items()},'output':torch.empty(32768,16,128,dtype=torch.float16,device='cuda'),
 'state':torch.empty(1,16,128,128,dtype=torch.float32,device='cuda')} for b in data['buffers'][:4]]
shared={k:v.cuda() for k,v in data['shared'].items()};del data
evict=torch.empty(4*int(prop.L2_cache_size),dtype=torch.uint8,device='cuda')
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());torch.cuda.synchronize()
compiled={};objects={};dsl={};runtime={};bi=0;artifacts={}
def role(k):
 return {'CPDeltaRuleTPrecomputeSm100':'K1','CPDeltaRuleMNPrecomputeUtcmma1Sm100':'K2',
 'CPDeltaRuleFixupUtcmmaSm100':'K3','CPDeltaRulePrefillTcgen05Sm100':'K4'}[type(k).__name__]
def compile4(name,obj,args,opts):
 d=out/'binary'/name;d.mkdir(parents=True)
 fn=cute.compile[(*opts,cute.KeepPTX(True),cute.KeepCUBIN(True),cute.KeepSASS(True),DumpDir(str(d)))](obj,*args)
 cub=next(d.glob('*.cubin'));sass=next(d.glob('*.sass'));ss=sass.read_text()
 lines=[m[1] for ln in ss.splitlines() if (m:=re.search(r'/\*[0-9a-f]+\*/\s+(.*?);',ln))]
 ops={k:sum(bool(re.match(r'^(?:@!?\w+\s+)?'+k+r'(?:\.|\s)',v.strip())) for v in lines) for k in
 ['LDL','STL','LDSM','LDS','STSM','STS','LDTM','STTM','UTCHMMA','UTMALDG','UTMASTG','F2FP','FMUL2','MUFU','SYNCS']}
 # Frozen upstream B0 itself contains local instructions. Reject *added*
 # spill in a variant, never silently erase baseline local work.
 if name!='B0':
  assert ops['LDL']<=artifacts['B0']['ops']['LDL'] and ops['STL']<=artifacts['B0']['ops']['STL'],('ADDED_SPILL',name,ops)
 artifacts[name]=dict(cubin_sha256=sha(cub),SASS_sha256=sha(sass),source_sha256=sha(R/f'variant_{name}.py') if name!='B0' else sha(REPO/'flashinfer/gdn_kernels/blackwell/gated_delta_net_cp_prefill.py'),
   resource=sh('/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(cub)),ops=ops)
 save('COMPILE_AUDIT.json',artifacts);return fn
def compiler(k,*args,compile_options):
 r=role(k);objects[r]=k;dsl[r]=(args,compile_options)
 fn=compile4('B0',k,args,compile_options) if r=='K4' else cute.compile[compile_options](k,*args)
 def wrapped(*v):runtime[bi,r]=v;return fn(*v)
 compiled[r]=(fn,wrapped);return wrapped
cp.get_cached_compile=lambda k,o:compiled.get(role(k),(None,None))[1]
cp.cached_compile=compiler
oldbuf=cp._get_cache_buf
cp._get_cache_buf=lambda n,*x,**kw:oldbuf(n+f'_vjoin_b{bi}',*x,**kw)
gold=[]
for bi in range(4):
 b=buffers[bi]
 with torch.cuda.stream(stream):
  chunk_gated_delta_rule(b['q'],b['k'],b['v'],g=shared['gate'],beta=shared['beta'],scale=1/(128**.5),
   initial_state=shared['initial'],output_final_state=True,cu_seqlens=shared['cu_seqlens'],use_qk_l2norm_in_kernel=False,
   output=b['output'],output_state=b['state'],use_cp=True)
 stream.synchronize();gold.append((b['output'].clone(),b['state'].clone()))
assert artifacts['B0']['cubin_sha256']=='06ebd78963371a30adac142c06b86950ec300b6cb762a209c5afc67628941814',artifacts['B0']
target=compiled['K4'][0]
if a.variant!='B0':
 sp=importlib.util.spec_from_file_location('vjoin_'+a.variant,R/f'variant_{a.variant}.py');mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
 obj=mod.K4VJoin.__new__(mod.K4VJoin);obj.__dict__.update(objects['K4'].__dict__)
 args,opts=dsl['K4'];target=compile4(a.variant,obj,args,opts)
 def relevant(z):return {k:v for k,v in z.items() if k not in ['SYNCS']}
 assert relevant(artifacts[a.variant]['ops'])==relevant(artifacts['B0']['ops']),('CHANGED_WORK',artifacts)
def chain(bi):
 for r in ['K1','K2','K3']:compiled[r][0](*runtime[bi,r])
 target(*runtime[bi,'K4'])
for bi in range(4):
 with torch.cuda.stream(stream):chain(bi)
 stream.synchronize()
 assert torch.equal(buffers[bi]['output'],gold[bi][0]) and torch.equal(buffers[bi]['state'],gold[bi][1]),('NOT_BITWISE_EQUAL',a.variant,bi)
graphs={}
for bi in range(4):
 e0,e1=torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)
 with torch.cuda.stream(stream):e0.record();e1.record()
 stream.synchronize();g=torch.cuda.CUDAGraph(keep_graph=True);g.enable_debug_mode()
 with torch.cuda.graph(g,stream=stream):
  evict.fill_(90)
  for r in ['K1','K2','K3']:compiled[r][0](*runtime[bi,r])
  e0.record();target(*runtime[bi,'K4']);e1.record()
 d=out/f'chain_b{bi}.dot';g.debug_dump(str(d));g.instantiate();graphs[bi]=(g,e0,e1)
def run(bi):
 g,e0,e1=graphs[bi]
 with torch.cuda.stream(stream):g.replay()
 stream.synchronize();return e0.elapsed_time(e1)*1000
for i in range(20):run(i%4)
vals=[]
if a.profile:
 torch.cuda.profiler.start();torch.cuda.nvtx.range_push('K4_VJOIN_FULL_CP')
 vals.append(run(0));torch.cuda.nvtx.range_pop();torch.cuda.profiler.stop()
else:
 for i in range(a.samples):vals.append(run(i%4))
for bi in range(4):
 assert torch.equal(buffers[bi]['output'],gold[bi][0]) and torch.equal(buffers[bi]['state'],gold[bi][1])
save('RESULT.json',dict(variant=a.variant,K4_Event_us=vals,full_CP=True,cold_before_K1=True,rotating_buffers=4,
  correctness_bitwise=True,scope='Same-work schedule mechanism; stage costs not inferred from Event difference.',
  actual_clock_and_throttle=sh('nvidia-smi','-q','-d','CLOCK,PERFORMANCE,POWER,TEMPERATURE'),model_enabled=False))
assert not sh('git','-C',str(REPO),'status','--porcelain')
print(a.variant,'COMPLETE',flush=True)
