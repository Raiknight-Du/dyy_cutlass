from pathlib import Path
import hashlib,json,os,subprocess,sys
O=Path(__file__).resolve().parent;N=Path('/home/ubuntu/workspace/op12/k4_10h_native_r8_20261026');assert os.environ.get('K1_GPU_SESSION')
assert hashlib.sha256((N/'run_probe_graph.py').read_bytes()).hexdigest()=='7c3819611d6ca9f1763db1c50b44c0135c91438757c4f9722acddf18c35aa4eb'
cmd=[sys.executable,str(N/'run_probe_graph.py'),'--mode','prepare','--authorization',str(O/'R8_PREPARE_ADMISSION.json'),'--out',str(O/'r8_actual_prepare01')]
p=O/'R8_PREPARE_COMMAND.json';p.write_text(json.dumps(cmd,indent=2))
subprocess.run([sys.executable,str(O/'gpu_telemetry_sidecar.py'),'--out',str(O/'r8_actual_prepare01_telemetry'),'--command-json',str(p),'--interval-ms','100'],check=True,timeout=300)
d=json.loads((O/'r8_actual_prepare01/ACTUAL_METADATA_GATE.json').read_text())
assert d['actual_CUBIN_SHA256']=='3136891d6594d34f3fc4dffc33bb28bfa0fd5370dc3e156f90e1df06d192afca'
assert d['CPU_text_exact_match'] and d['max_CTA_per_SM']==1 and not d['helper_invoked']
print(json.dumps(d,indent=2),flush=True)
