"""Three-process formal batch under existing 1845 lease, restored in finally."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
O=Path(__file__).resolve().parent;V=O/'r3_ldg_service_v2'
assert os.environ.get('K1_GPU_SESSION')
auth=O/'R3_V2_FORMAL_ADMISSION.json';a=json.loads(auth.read_text())
binary=V/'compile01_cuda130/ldg_service';assert hashlib.sha256(binary.read_bytes()).hexdigest()==a['binary_sha256']
lease=Path('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/performance_analysis/k4_init_l2_competition_20261024_3_run01/clock_lease.py')
cmd=[sys.executable,str(lease),sys.executable,str(V/'run_batch.py'),'--binary',str(binary),'--gate',str(V/'STATIC_GATE.json'),
 '--authorization',str(auth),'--out',str(O/'r3_v2_formal01'),'--mode','formal','--processes','3','--points','128,256,512,1024,2048,4096,8192']
p=O/'R3_V2_FORMAL_COMMAND.json';p.write_text(json.dumps(cmd,indent=2))
code=None
try:
    code=subprocess.run([sys.executable,str(O/'gpu_telemetry_sidecar.py'),'--out',str(O/'r3_v2_formal01_telemetry'),
       '--command-json',str(p),'--interval-ms','100'],timeout=5400).returncode
finally:
    observations=[]
    for _ in range(3):
        q=subprocess.run(['nvidia-smi','--query-gpu=uuid,clocks.sm,clocks.mem,clocks.applications.graphics,clocks.default_applications.graphics,clocks_event_reasons.active','--format=csv,noheader,nounits'],capture_output=True,text=True)
        observations.append(dict(time_ns=time.time_ns(),returncode=q.returncode,stdout=q.stdout,stderr=q.stderr));time.sleep(.25)
    restore=json.loads((Path(os.environ['K1_GUARD_DIR'])/'clock_lease/restore_automatic.json').read_text())
    (O/'R3_V2_FORMAL_AUTO_RESTORE_VERIFICATION.json').write_text(json.dumps(dict(reset_command=restore,post_reset_observations=observations,command_returncode=code,limitation='Reset success plus subsequent sampled state; requested SM clock never substituted for measured native cycles.',model_enabled=False),indent=2))
assert code==0,code
subprocess.run([sys.executable,str(V/'analyze_rates.py'),'--input',str(O/'r3_v2_formal01'),'--out',str(O/'r3_v2_formal01/RATE_ANALYSIS.json')],check=True,timeout=120)
