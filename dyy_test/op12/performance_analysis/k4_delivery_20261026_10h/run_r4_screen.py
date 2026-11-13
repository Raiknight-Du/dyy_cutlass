"""Finite R4 screen; process group termination applies only to this child."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
O=Path(__file__).resolve().parent;N=Path('/home/ubuntu/workspace/op12/k4_10h_native_r4_20261026')
assert os.environ.get('K1_GPU_SESSION')
assert hashlib.sha256((N/'run_probe.py').read_bytes()).hexdigest()=='fd5e01d3f5baab05924df289d20361f24d295ddac479db5609c4eb492b4477ec'
out=O/'r4_screen01';supervision=O/'r4_screen01_supervision';supervision.mkdir(exist_ok=False)
cmd=[sys.executable,str(N/'run_probe.py'),'--mode','screen','--points','128,256','--authorization',str(O/'R4_SCREEN_ADMISSION.json'),'--out',str(out)]
(supervision/'COMMAND.json').write_text(json.dumps(cmd,indent=2))
started=time.time_ns();timed_out=False
with (supervision/'stdout').open('w') as stdout,(supervision/'stderr').open('w') as stderr:
    proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,start_new_session=True)
    try:code=proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(proc.pid,signal.SIGTERM)
        try:code=proc.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait(timeout=5)
(supervision/'STATUS.json').write_text(json.dumps(dict(pid=proc.pid,returncode=code,timed_out=timed_out,start_time_ns=started,end_time_ns=time.time_ns(),clock_policy='automatic',only_own_child_process_group_terminated=True,model_enabled=False),indent=2))
assert not timed_out and code==0,(timed_out,code)
print('R4_SCREEN_CHILD_COMPLETE',flush=True)
