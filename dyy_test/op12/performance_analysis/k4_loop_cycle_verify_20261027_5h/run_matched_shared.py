"""Authorized5h task runner. New independentmicro only; no upstream changes."""
from pathlib import Path
import argparse,os,json,csv,hashlib,re,subprocess,time,statistics,sys
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--kind',type=int,required=True);p.add_argument('--modes',default='1,3')
    p.add_argument('--points',default='128,256,512,1024');p.add_argument('--samples',type=int,default=30)
    p.add_argument('--warmup',type=int,default=20);p.add_argument('--trial',type=int,default=0)
    p.add_argument('--profile',action='store_true');p.add_argument('--screen',action='store_true')
    a=p.parse_args(); assert os.environ.get('K4_5H_ADMITTED')=='1'
    a.out.mkdir(parents=True,exist_ok=False)
    import torch,cutlass
    import cutlass.cute as cute
    from cutlass.base_dsl.compiler import DumpDir
    from cuda.bindings import driver as cuda
    from matched_shared import MatchedShared,FAMILIES
    torch.set_num_threads(1);torch.cuda.set_device(0)
    obj=MatchedShared(a.kind);fam=FAMILIES[a.kind]
    ga=torch.ones((64,128),dtype=torch.float16,device='cuda');gb=torch.ones_like(ga)
    go=torch.empty((128,64,64),dtype=torch.float32,device='cuda')
    checks=torch.empty((128,128),dtype=torch.int32,device='cuda');trace=torch.empty((128,6),dtype=torch.int64,device='cuda')
    offs=torch.tensor(fam['offsets'],dtype=torch.int32,device='cuda').contiguous()
    cv=lambda t:cute.runtime.from_dlpack(t,assumed_align=16,enable_tvm_ffi=True)
    args=[cv(t) for t in [ga,gb,go,checks,trace,offs]]
    stream=torch.cuda.Stream();cs=cuda.CUstream(stream.cuda_stream)
    build=a.out/'build';build.mkdir()
    opts=(cute.EnableTVMFFI(True),cute.GPUArch('sm_100a'),cute.KeepPTX(True),cute.KeepCUBIN(True),cute.KeepSASS(True),DumpDir(str(build)))
    fn=cute.compile[opts](obj,*args,cutlass.Int32(128),cutlass.Int32(1),cs)
    cb=next(build.glob('*.cubin'));ss=next(build.glob('*.sass'))
    resource=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(cb)],text=True)
    sass=ss.read_text();pcs=[]
    for ln in sass.splitlines():
        m=re.search(r'/\*([0-9a-f]+)\*/\s+(?:(@!?\w+)\s+)?(\w[\w.]*)\s+(.*?)\s*;',ln)
        if m:pcs.append(dict(pc='0x'+m[1],pred=m[2],op=m[3],operands=m[4]))
    native_ldl=[r for r in pcs if r['op'].startswith(('LDL','STL'))]
    assert not native_ldl,('UNEXPECTED_SPILL',native_ldl)
    targetops=sorted(set(fam['opcodes']));got={op:sum(r['op']==op for r in pcs) for op in targetops}
    for op in targetops:assert got[op]>=fam['opcodes'].count(op),(op,got)
    ident=dict(source_SHA256={n:sha(R/n) for n in ['matched_shared.py','FAMILIES.json','old_qk_resident.py']},
        CUBIN_SHA256=sha(cb),SASS_SHA256=sha(ss),resource=resource,target_static_op_counts=got,
        native_LDL_STL=len(native_ldl),grid=128,threads=192,SMEM_reserved_B=225280,
        family={k:v for k,v in fam.items() if k!='offsets'},python=sys.executable,
        torch=torch.__version__,cutlass=cutlass.__version__,device=str(torch.cuda.get_device_properties(0)),
        profile=a.profile,model_enabled=False)
    (a.out/'IDENTITY.json').write_text(json.dumps(ident,ensure_ascii=False,indent=2))
    points=list(map(int,a.points.split(',')));modes=list(map(int,a.modes.split(',')))
    assert all(1<=n<=16384 for n in points)
    def call(n,m):
        with torch.cuda.stream(stream):fn(*args,n,m,cs)
    def expect(tid,mode):
        vs=[(0x3c003c00+j*0x10001)&0xffffffff for j in range(fam['value_count'])]
        if mode&1 and not fam['ptx'].startswith('stmatrix'):
            for j,off in enumerate(fam['offsets'][tid]):
                if off>=0:
                    for k in range(fam['widths'][j]):vs[fam['value_starts'][j]+k]=0x3c003c00
        ans=0
        for z in vs:ans^=z
        return ans
    # Uniformlyconstant memory makes bit-check invariant across matrix transposes.
    # Volatileasm and nativecounter gates, rather than nonzero checksum, prove retention.
    f=(a.out/'EVENT_RAW.csv').open('w',newline='');w=csv.DictWriter(f,['trial','kind','family','mode','N','sample','event_us','cta_cycle_median','cta_cycle_p5','cta_cycle_p95','SMs','valid']);w.writeheader()
    cf=(a.out/'CTA_CLOCK_RAW.csv').open('w',newline='');cw=csv.DictWriter(cf,['trial','kind','mode','N','sample','CTA','begin','end','sm_begin','sm_end']);cw.writeheader()
    for n in points:
        for m in modes:
            for _ in range(a.warmup):call(n,m)
            stream.synchronize()
        for s in range(a.samples):
            for m in (modes if s%2==0 else list(reversed(modes))):
                e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                if a.profile:torch.cuda.profiler.start();torch.cuda.nvtx.range_push('K4_MATCHED_SHARED_TARGET')
                with torch.cuda.stream(stream):e0.record();fn(*args,n,m,cs);e1.record()
                e1.synchronize()
                if a.profile:torch.cuda.nvtx.range_pop();torch.cuda.profiler.stop()
                clocks=trace.cpu().tolist();ch=checks.cpu().tolist()
                assert bool(torch.all(go==(128.0 if m&2 else 0.0))),('MMA_output',fam['name'],n,m)
                assert all((v&0xffffffff)==expect(t,m) for row in ch for t,v in enumerate(row)),('SHADER_CHECK',fam['name'],n,m,ch[0][:16])
                dur=[]
                for c,q in enumerate(clocks):
                    assert q[0]<q[1] and q[2]==q[3] and q[4:]==[n,m],q
                    dur.append(q[1]-q[0]);cw.writerow(dict(trial=a.trial,kind=a.kind,mode=m,N=n,sample=s,CTA=c,begin=q[0],end=q[1],sm_begin=q[2],sm_end=q[3]))
                sd=sorted(dur);w.writerow(dict(trial=a.trial,kind=a.kind,family=fam['name'],mode=m,N=n,sample=s,event_us=e0.elapsed_time(e1)*1000,cta_cycle_median=statistics.median(dur),cta_cycle_p5=sd[6],cta_cycle_p95=sd[121],SMs=len(set(q[2] for q in clocks)),valid=1));f.flush();cf.flush()
    f.close();cf.close()
    (a.out/'COMPLETE.json').write_text(json.dumps(dict(complete=True,points=points,modes=modes,samples=a.samples,correctness=True,model_enabled=False),indent=2))
    print(json.dumps(dict(status='COMPLETE',family=fam['name'],profile=a.profile,out=str(a.out))))
if __name__=='__main__':main()
