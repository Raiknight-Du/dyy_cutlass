"""Apply existing authorized 1845 lease to a hash-bound stage command."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
O=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--command-json',type=Path,required=True);a=p.parse_args()
assert os.environ.get('K1_GPU_SESSION')
cmd=json.loads(a.command_json.read_text());assert isinstance(cmd,list) and all(isinstance(x,str) for x in cmd)
lease='/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/performance_analysis/k4_init_l2_competition_20261024_3_run01/clock_lease.py'
code=None
try:code=subprocess.run([sys.executable,lease,*cmd],timeout=5500).returncode
finally:
    reset=Path(os.environ['K1_GUARD_DIR'])/'clock_lease/restore_automatic.json'
    observations=[]
    for _ in range(3):
        q=subprocess.run(['nvidia-smi','--query-gpu=uuid,clocks.sm,clocks.mem,clocks.applications.graphics,clocks.default_applications.graphics,clocks_event_reasons.active','--format=csv,noheader,nounits'],capture_output=True,text=True)
        observations.append(dict(time_ns=time.time_ns(),returncode=q.returncode,stdout=q.stdout,stderr=q.stderr));time.sleep(.25)
    (O/(a.stage+'_AUTO_RESTORE_VERIFICATION.json')).write_text(json.dumps(dict(command_returncode=code,reset_record=json.loads(reset.read_text()) if reset.exists() else 'MISSING',post_reset_observations=observations,model_enabled=False),indent=2))
assert code==0,code
