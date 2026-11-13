"""Three independently initialized R4 processes beneath authorized lease."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
O=Path(__file__).resolve().parent;N=Path('/home/ubuntu/workspace/op12/k4_10h_native_r4_20261026')
assert os.environ.get('K1_GPU_SESSION')
assert hashlib.sha256((N/'run_probe.py').read_bytes()).hexdigest()=='fd5e01d3f5baab05924df289d20361f24d295ddac479db5609c4eb492b4477ec'
for case in ['CONTROL','SHADER_ONLY','TC_ONLY']:
    c=json.loads((O/'r4_ncu_remaining01'/case/'COUNTER_AUDIT.json').read_text())
    r=json.loads((O/'r4_ncu_remaining01'/case/'IDENTITY_RESOURCE_AUDIT.json').read_text())
    assert c['status']=='TARGET_PC_COUNTS_PASS' and c['TC_hardware_gate_status']=='CHECKED' and len(c['TC_hardware_checks'])==4
    assert r['status']=='ACTUAL_NATIVE_IDENTITY_RESOURCE_PASS'
out=O/'r4_formal01';out.mkdir(exist_ok=False);records=[]
for trial in range(3):
    cmd=[sys.executable,str(N/'run_probe.py'),'--mode','formal','--trial',str(trial),'--points','128,256,512,1024,2048,4096,8192','--authorization',str(O/'R4_FORMAL_ADMISSION.json'),'--out',str(out/f'trial{trial}')]
    start=time.time_ns();timeout=False
    with (out/f'trial{trial}.stdout').open('w') as stdout,(out/f'trial{trial}.stderr').open('w') as stderr:
        child=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,start_new_session=True)
        try:code=child.wait(timeout=120)
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=5)
    records.append(dict(trial=trial,pid=child.pid,command=cmd,start_time_ns=start,end_time_ns=time.time_ns(),returncode=code,timed_out=timeout,model_enabled=False))
    (out/'RUN_RECORD.json').write_text(json.dumps(records,indent=2))
    assert code==0 and not timeout,records[-1]
subprocess.run([sys.executable,str(N/'analyze_rates.py'),'--input',str(out),'--out',str(out/'RATE_ANALYSIS.json')],check=True,timeout=120)
