"""Run only after Lead confirms all GPU experiment work has ended. No CUDA imports or mutations."""
from pathlib import Path
import argparse, subprocess, json, hashlib, time, datetime

P=argparse.ArgumentParser();P.add_argument('--out',required=True);a=P.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
OP=Path('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12')
repo=OP/'upstreams/flashinfer';pin='e77a4a0d276367895c3b50a642fd8f326c03fb72'
def run(argv):
    start=time.time_ns();z=subprocess.run(argv,capture_output=True,text=True,timeout=30)
    return {'command':argv,'start_time_ns':start,'end_time_ns':time.time_ns(),'returncode':z.returncode,'stdout':z.stdout,'stderr':z.stderr}
head=run(['git','-C',str(repo),'rev-parse','HEAD'])
status=run(['git','-C',str(repo),'status','--porcelain'])
sources=[]
for path in sorted((repo/'flashinfer/gdn_kernels/blackwell').glob('*.py')):
    relative=path.relative_to(repo).as_posix()
    z=subprocess.run(['git','-C',str(repo),'show','HEAD:'+relative],capture_output=True,timeout=30)
    working=hashlib.sha256(path.read_bytes()).hexdigest()
    committed=hashlib.sha256(z.stdout).hexdigest() if z.returncode==0 else None
    sources.append({'path':relative,'working_source_SHA256':working,'HEAD_source_SHA256':committed,'same_as_HEAD':working==committed,'git_show_returncode':z.returncode})
snapshots=[]
for index in range(3):
    snapshots.append({'index':index,
      'gpu':run(['nvidia-smi','--query-gpu=uuid,pstate,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active','--format=csv,noheader,nounits']),
      'apps':run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'])})
    if index<2:time.sleep(1)
detail=run(['nvidia-smi','-q','-d','CLOCK,PERFORMANCE,POWER'])
result={'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'purpose':'Final read-only upstream/source and GPU state record; no experiment or context initialization',
 'upstream_HEAD':head,'upstream_status':status,'upstream_pin_match':head['returncode']==0 and head['stdout'].strip()==pin,
 'upstream_clean':status['returncode']==0 and not status['stdout'].strip(),'sources':sources,
 'all_sources_equal_HEAD':bool(sources) and all(x['same_as_HEAD'] for x in sources),
 'GPU_snapshots':snapshots,'GPU_detail':detail,
 'clock_restore_limit':'Read-only current state is paired with prior successful finally -rgc records; current MHz alone cannot prove absence of a requested lock.',
 'CUDA_imported':False,'kernel_launched':False,'clock_changed':False,'system_changed':False,
 'script_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_enabled':False}
(out/'FINAL_READONLY_STATE.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:result[k] for k in ['upstream_pin_match','upstream_clean','all_sources_equal_HEAD','CUDA_imported','kernel_launched','clock_changed']}))
