"""Runner-only GPU entry; requires a separate hash-bound Lead admission JSON.

Authoring/importing this file performs no device operations. Never run it during
the CPU engineering turn. Native executable does all CUDA work in a child.
"""
from pathlib import Path
import argparse,csv,hashlib,json,os,subprocess,sys,threading,time
O=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def admission(args,mode):
    gate=json.loads(Path(args.gate).read_text());auth=json.loads(Path(args.authorization).read_text())
    assert gate['status']=='STATIC_PASS_PENDING_LEAD_GPU_ADMISSION'
    assert gate['application_fatbin_native_only']
    assert sha(args.binary)==gate['actual_executable_sha256']==auth['binary_sha256']
    assert auth['round']=='R3' and mode in auth['approved_modes']
    assert args.processes<=auth['max_processes']
    nn=list(map(int,args.points.split(',')))
    assert all(1<=n<=min(16384,auth['max_N']) for n in nn)
    if mode=='screen':assert nn==[128,256] and args.processes==1
    if mode=='formal':assert len(nn)>=4 and args.processes==3
    return gate,auth
def state():
    r=subprocess.run(['nvidia-smi','--query-gpu=uuid,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw',
                      '--format=csv,noheader,nounits'],capture_output=True,text=True)
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True)
    assert r.returncode==0 and p.returncode==0
    apps=[line for line in p.stdout.splitlines() if line.strip()]
    utilization=[float(line.split(',')[1].strip()) for line in r.stdout.splitlines() if line.strip()]
    return {'time_ns':time.time_ns(),'gpu':r.stdout.strip(),'utilization_pct':utilization,'compute_apps':apps}
def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--gate',type=Path,default=O/'STATIC_GATE.json');p.add_argument('--authorization',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['screen','formal'],required=True)
    p.add_argument('--processes',type=int,required=True);p.add_argument('--points',required=True)
    a=p.parse_args();gate,auth=admission(a,a.mode)
    a.out.mkdir(parents=True,exist_ok=False)
    # No clock/MPS/cache management. Initial guard is three idle observations.
    guards=[]
    for i in range(3):
        s=state();assert not s['compute_apps'] and all(x<=5 for x in s['utilization_pct']),s;guards.append(s)
        if i<2:time.sleep(30)
    (a.out/'INITIAL_GUARD.json').write_text(json.dumps(guards,indent=2))
    env=os.environ.copy();env.pop('CUDA_FORCE_PTX_JIT',None);env.pop('CUDA_DISABLE_PTX_JIT',None)
    results=[]
    for t in range(a.processes):
        out=a.out/f'trial{t}'
        cmd=[str(a.binary.resolve()),'--mode',a.mode,'--out',str(out),'--config','all',
             '--points',a.points,'--trial',str(t),'--warmup','20','--samples','30']
        telemetry=[];stop=threading.Event();foreign=[]
        with (a.out/f'trial{t}.stdout').open('w') as stdout,(a.out/f'trial{t}.stderr').open('w') as stderr:
            proc=subprocess.Popen(cmd,env=env,stdout=stdout,stderr=stderr)
            def monitor():
                while not stop.is_set():
                    s=state();telemetry.append(s)
                    other=[x for x in s['compute_apps'] if int(x.split(',')[0])!=proc.pid]
                    if other:foreign.extend(other);proc.terminate();return
                    stop.wait(1)
            thread=threading.Thread(target=monitor,daemon=True);thread.start()
            code=proc.wait();stop.set();thread.join(timeout=5)
        (a.out/f'trial{t}_TELEMETRY.json').write_text(json.dumps(telemetry,indent=2))
        row={'trial':t,'command':cmd,'pid':proc.pid,'returncode':code,'foreign':foreign,
             'binary_sha256':sha(a.binary),'CUDA_DISABLE_PTX_JIT':'UNSET_NATIVE_ONLY_APPLICATION'};results.append(row)
        (a.out/'RUN_RECORD.json').write_text(json.dumps({'mode':a.mode,'admission':auth,'runs':results,
                 'model_enabled':False},indent=2))
        assert code==0 and not foreign,row
        runtime=json.loads((out/'runtime_metadata.json').read_text())
        resources={x['variant']:x['REG'] for x in gate['records']}
        assert runtime['scalar_REG']==resources['scalar'] and runtime['vector_REG']==resources['vector']
        assert runtime['scalar_local_B']==runtime['vector_local_B']==0
        assert runtime['dynamic_SMEM_B']==225280
        assert runtime['max_CTA_per_SM_scalar']==runtime['max_CTA_per_SM_vector']==1
    print(json.dumps({'completed':True,'independent_processes':len(results),'model_enabled':False}))
if __name__=='__main__':main()
