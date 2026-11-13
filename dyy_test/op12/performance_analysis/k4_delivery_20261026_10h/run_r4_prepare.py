"""Guarded CUDA metadata compilation only. No helper or gold launch."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
O=Path(__file__).resolve().parent;N=Path('/home/ubuntu/workspace/op12/k4_10h_native_r4_20261026')
assert os.environ.get('K1_GPU_SESSION')
assert hashlib.sha256((N/'run_probe.py').read_bytes()).hexdigest()=='fd5e01d3f5baab05924df289d20361f24d295ddac479db5609c4eb492b4477ec'
cmd=[sys.executable,str(N/'run_probe.py'),'--mode','prepare','--points','128','--authorization',str(O/'R4_PREPARE_ADMISSION.json'),'--out',str(O/'r4_actual_prepare01')]
p=O/'R4_PREPARE_COMMAND.json';p.write_text(json.dumps(cmd,indent=2))
subprocess.run([sys.executable,str(O/'gpu_telemetry_sidecar.py'),'--out',str(O/'r4_actual_prepare01_telemetry'),'--command-json',str(p),'--interval-ms','100'],check=True,timeout=300)
g=json.loads((O/'r4_actual_prepare01/ACTUAL_METADATA_GATE.json').read_text())
assert g['CPU_text_exact_match'] and not g['helper_invoked'] and g['max_CTA_per_SM']==1
print(json.dumps(g,indent=2))
