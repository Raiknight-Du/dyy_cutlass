"""Authorized serial N512 NCU actions; sections/passes never become rate timing."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
O=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--round',choices=['R3','R4'],required=True);p.add_argument('--authorization',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--configs',required=True);a=p.parse_args()
assert os.environ.get('K1_GPU_SESSION')
auth=json.loads(a.authorization.read_text());configs=a.configs.split(',')
assert auth['round']==a.round and 'NCU' in auth['approved_modes'] and set(configs)<=set(auth['allowed_configurations'])
assert len(configs)<=auth['authorized_NCU_invocations']
if a.round=='R4' and any(c!='BOTH' for c in configs):
    gate=json.loads((O/'r4_ncu_BOTH01/BOTH/COUNTER_AUDIT.json').read_text())
    identity=json.loads((O/'r4_ncu_BOTH01/BOTH/IDENTITY_RESOURCE_AUDIT.json').read_text())
    assert gate['status']=='TARGET_PC_COUNTS_PASS' and gate['TC_hardware_gate_status']=='CHECKED' and len(gate['TC_hardware_checks'])==4
    assert identity['status']=='ACTUAL_NATIVE_IDENTITY_RESOURCE_PASS'
a.out.mkdir(exist_ok=False)
env=os.environ.copy();prior={k:env.pop(k,None) for k in ['CUDA_DISABLE_PTX_JIT','CUDA_FORCE_PTX_JIT']}
(a.out/'JIT_ENVIRONMENT.json').write_text(json.dumps(dict(before=prior,tool_and_application_policy='CUDA_DISABLE_PTX_JIT and CUDA_FORCE_PTX_JIT unset; R3 application native-only fatbin; R4 actual metadata text verified each process'),indent=2))
ncu='/opt/nvidia/nsight-compute/2026.2.1/ncu'
metrics=json.loads((O/'r3_v2_ncu_S_G128_LOAD_01/METRICS.json').read_text())
N=O/'r3_ldg_service_v2' if a.round=='R3' else Path('/home/ubuntu/workspace/op12/k4_10h_native_r4_20261026')
for config in configs:
    out=a.out/config;out.mkdir();report=out/config
    if a.round=='R3':
        binary=N/'compile01_cuda130/ldg_service';assert hashlib.sha256(binary.read_bytes()).hexdigest()==auth['binary_sha256']
        app=[sys.executable,str(N/'profile_one.py'),'--binary',str(binary),'--gate',str(N/'STATIC_GATE.json'),'--authorization',str(a.authorization),'--config',config,'--out-root',str(out/'replays')]
        kernel=[]
    else:
        assert hashlib.sha256((N/'shared_coupling.py').read_bytes()).hexdigest()==auth['source_SHA256']
        app=[sys.executable,str(N/'run_probe.py'),'--mode','profile','--config',config,'--points','512','--authorization',str(a.authorization),'--out',str(out/'replays')]
        kernel=['--kernel-name','regex:.*shared_couplingSharedCoupling.*']
    command=[ncu,'--target-processes','all','--replay-mode','application','--cache-control','none','--clock-control','none','--profile-from-start','off','--launch-count','1',*kernel,
       '--export',str(report),'--section','LaunchStats','--section','SourceCounters','--section','InstructionStats','--metrics',','.join(metrics),*app]
    cp=out/'COMMAND.json';cp.write_text(json.dumps(command,indent=2))
    record=dict(round=a.round,configuration=config,N=512,start_time_ns=time.time_ns(),status='STARTED',clock_policy='temporary1845request_actual_sampled',model_enabled=False)
    with (O/'RUNNER_NCU_LEDGER.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    q=subprocess.run([sys.executable,str(O/'gpu_telemetry_sidecar.py'),'--out',str(out/'telemetry'),'--command-json',str(cp),'--interval-ms','100'],timeout=1800,env=env)
    record.update(status='REPORT_CREATED' if q.returncode==0 and report.with_suffix('.ncu-rep').exists() else 'FAILED',returncode=q.returncode,end_time_ns=time.time_ns())
    (out/'RESULT.json').write_text(json.dumps(record,indent=2))
    with (O/'RUNNER_NCU_LEDGER.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    assert record['status']=='REPORT_CREATED',record
    for page in ['raw','source']:
        cmd=[ncu,'--import',str(report.with_suffix('.ncu-rep')),'--page',page,'--csv','--print-units','base']
        if page=='source':cmd+=['--print-source','sass']
        with (out/(config+'_'+page+'.csv')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
    print(a.round,config,'NCU_COMPLETE',flush=True)
(a.out/'COMPLETE.json').write_text(json.dumps(dict(completed=True,configurations=configs,NCU_invocations=len(configs),model_enabled=False),indent=2))
