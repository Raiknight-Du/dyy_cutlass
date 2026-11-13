from pathlib import Path
import argparse,json,os,signal,subprocess,threading,time
import pynvml as nv
R=Path(__file__).resolve().parent;PY=R.parents[1]/'.venv/bin/python';NCU='/opt/nvidia/nsight-compute/2026.2.1/ncu'
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot','cachecheck','formal'],required=True);p.add_argument('--attempt',type=int,default=1);a=p.parse_args();phase_tag=a.phase+(f'_try{a.attempt}' if a.attempt>1 else '');pre=phase_tag.upper()
nv.nvmlInit();h=nv.nvmlDeviceGetHandleByIndex(0)
def stop_requested(signum,frame):raise SystemExit(128+signum)
for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop_requested)
def save(n,x):(R/n).write_text(json.dumps(x,indent=2))
def tele():
    u=nv.nvmlDeviceGetUtilizationRates(h)
    ans={'wall_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),'SM_MHz':nv.nvmlDeviceGetClockInfo(h,nv.NVML_CLOCK_SM),'memory_MHz':nv.nvmlDeviceGetClockInfo(h,nv.NVML_CLOCK_MEM),'power_mW':nv.nvmlDeviceGetPowerUsage(h),'temperature_C':nv.nvmlDeviceGetTemperature(h,nv.NVML_TEMPERATURE_GPU),'throttle':nv.nvmlDeviceGetCurrentClocksThrottleReasons(h),'util':u.gpu,'processes':[x.pid for x in nv.nvmlDeviceGetComputeRunningProcesses(h)]}
    for name,policy in [('power_violation',nv.NVML_PERF_POLICY_POWER),('thermal_violation',nv.NVML_PERF_POLICY_THERMAL)]:
        try:
            z=nv.nvmlDeviceGetViolationStatus(h,policy);ans[name]={'violationTime':z.violationTime,'referenceTime':z.referenceTime,'unit':'NVML raw nanoseconds'}
        except nv.NVMLError as e:ans[name]={'error':str(e)}
    return ans
def smi():return subprocess.check_output(['nvidia-smi','-q','-d','CLOCK,POWER,PERFORMANCE,TEMPERATURE'],text=True)
assert not (R/(pre+'_STATUS.json')).exists(),'Do not overwrite evidence'
idle=[];consecutive=0;deadline=time.monotonic()+4*3600
while consecutive<3:
    t=tele();idle.append(t);save(pre+'_IDLE.json',idle);consecutive=consecutive+1 if not t['processes'] and t['util']<5 else 0
    print('idle',consecutive,flush=True);assert time.monotonic()<deadline
    if consecutive<3:time.sleep(30)
before=tele();save(pre+'_CLOCK_BEFORE.json',{'NVML':before,'smi':smi(),'intended_restore':'pre-existing automatic GPU clocks; no memory or power changes'})
assert before['SM_MHz']<1000,'Unexpected non-idle/high fixed clock; do not assume automatic restoration'
result=[];changes=[];current='auto';locked_touched=False;child=None
if a.phase=='pilot':jobs=[(8192,'cold','auto',0,'pilot')]
elif a.phase=='cachecheck':jobs=[(8192,c,'auto',a.attempt-1,'cachecheck') for c in ('warm','cold')]
else:
    jobs=[]
    for t in (1,2,3):
        # Only fill the two missing cold+locked shapes; reuse the existing
        # three-shape auto evidence and S32768 locked evidence without rerunning.
        groups=[(8192,'cold','locked1845'),(16384,'cold','locked1845')]
        if t%2==0:groups.reverse()
        for s,cache,clock in groups:
            for mode in (('native','metrics') if t%2 else ('metrics','native')):jobs.append((s,cache,clock,t,mode))
try:
    for s,cache,clock,t,mode in jobs:
        while tele()['processes'] or tele()['util']>=5:
            print('WAIT_IDLE',s,cache,clock,t,mode,flush=True);assert time.monotonic()<deadline;time.sleep(30)
        if clock!=current:
            first=tele();assert not first['processes']
            if clock=='locked1845':nv.nvmlDeviceSetGpuLockedClocks(h,1845,1845);locked_touched=True
            else:nv.nvmlDeviceResetGpuLockedClocks(h)
            current=clock;time.sleep(.2);changes.append({'wall_ns':time.time_ns(),'request':clock,'before':first,'after':tele(),'smi':smi()});save(pre+'_CLOCK_CHANGES.json',changes)
        stem=f'{phase_tag}_S{s}_{cache}_{clock}_{mode}_t{t}'
        args=[str(PY),str(R/'measure.py'),'--s',str(s),'--trial',str(t),'--cache',cache,'--clock',clock,'--mode',mode]
        if mode=='pilot':args+=['--calls','8','--batches','1','--warmup','2']
        if mode=='metrics':args=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','-o',str(R/stem),'--gpu-metrics-devices=0','--gpu-metrics-set=gb10x','--gpu-metrics-frequency=200000']+args
        if mode=='cachecheck':
            args+=['--calls','1','--batches','1','--warmup','2','--hot-k','--allow-profiler-replay']
            metrics='dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_read_lookup_hit.sum'
            args=[NCU,'--profile-from-start','off','--target-processes','all','--kernel-name','regex:.*CPDeltaRuleTPrecomputeSm100.*','--launch-count','1','--replay-mode','application','--cache-control','none','--clock-control','none','--metrics',metrics,'-o',str(R/stem)]+args
        print('START',stem,flush=True);samples=[];foreign=[];done=threading.Event();started=time.monotonic()
        with (R/(stem+'.log')).open('x') as log:
            child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            def monitor():
                while not done.is_set():
                    try:
                        x=tele();samples.append(x)
                        for pid in x['processes']:
                            chain=[];cur=pid
                            while cur>1:
                                chain.append(cur)
                                if cur==child.pid:break
                                try:cur=int(Path('/proc/'+str(cur)+'/stat').read_text().split(') ')[1].split()[1])
                                except FileNotFoundError:break
                            if child.pid not in chain:foreign.append({'pid':pid,'chain':chain})
                    except Exception as e:samples.append({'error':str(e)})
                    done.wait(.02)
            th=threading.Thread(target=monitor,daemon=True);th.start()
            try:
                while child.poll() is None:
                    if foreign or time.monotonic()-started>1200:os.killpg(child.pid,signal.SIGTERM);raise RuntimeError('Competition/timeout '+str(foreign))
                    time.sleep(.2)
                assert child.returncode==0,(stem,child.returncode)
            finally:done.set();th.join();save(stem+'_NVML.json',samples)
        result.append({'stem':stem,'S':s,'cache':cache,'clock':clock,'trial':t,'mode':mode,'command':args,'foreign':foreign,'seconds':time.monotonic()-started,'after':tele()});save(pre+'_JOBS.json',result)
        print('DONE',stem,flush=True)
    save(pre+'_STATUS.json',{'done':True,'jobs':result,'memory_clock_unchanged':True,'power_limit_unchanged':True})
finally:
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    if locked_touched:
        nv.nvmlDeviceResetGpuLockedClocks(h);time.sleep(.2)
    save(pre+'_CLOCK_RESTORED.json',{'reset_gpu_clocks_issued':locked_touched,'NVML':tele(),'smi':smi(),'memory_clock_unchanged':nv.nvmlDeviceGetClockInfo(h,nv.NVML_CLOCK_MEM)==before['memory_MHz'],'power_limit_mW':nv.nvmlDeviceGetPowerManagementLimit(h)})
