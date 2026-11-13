from pathlib import Path
import hashlib,json,os,subprocess,sys
O=Path(__file__).resolve().parent;N=Path('/home/ubuntu/workspace/op12/k4_10h_native_r5_20261026')
assert os.environ.get('K1_GPU_SESSION')
assert hashlib.sha256((N/'run_probe.py').read_bytes()).hexdigest()=='561a3942b7f0793e1e2bdf2bc64d3903ae81386c26f25527c4b03a89cc423d40'
cmd=[sys.executable,str(N/'run_probe.py'),'--mode','prepare','--points','128','--authorization',str(O/'R5_PREPARE_ADMISSION.json'),'--out',str(O/'r5_actual_prepare01')]
p=O/'R5_PREPARE_COMMAND.json';p.write_text(json.dumps(cmd,indent=2))
subprocess.run([sys.executable,str(O/'gpu_telemetry_sidecar.py'),'--out',str(O/'r5_actual_prepare01_telemetry'),'--command-json',str(p),'--interval-ms','100'],check=True,timeout=300)
for source in ['SS','TS']:
    d=json.loads((O/'r5_actual_prepare01'/('ACTUAL_METADATA_GATE_'+source+'.json')).read_text())
    assert d['CPU_text_exact_match'] and d['max_CTA_per_SM']==1 and not d['helper_invoked']
    print(source,json.dumps(d),flush=True)
