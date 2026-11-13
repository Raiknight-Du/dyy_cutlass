from pathlib import Path
import hashlib,json,os,subprocess,sys
O=Path(__file__).resolve().parent;N=Path('/home/ubuntu/workspace/op12/k4_10h_native_r7_20261026');assert os.environ.get('K1_GPU_SESSION')
assert hashlib.sha256((N/'run_probe_graph.py').read_bytes()).hexdigest()=='3c7127fa511927d28ed2eb06712958ed63ab258c2d8dbb3a4c612a9715abbb96'
cmd=[sys.executable,str(N/'run_probe_graph.py'),'--mode','prepare','--authorization',str(O/'R7_PREPARE_ADMISSION.json'),'--out',str(O/'r7_actual_prepare01')]
p=O/'R7_PREPARE_COMMAND.json';p.write_text(json.dumps(cmd,indent=2))
subprocess.run([sys.executable,str(O/'gpu_telemetry_sidecar.py'),'--out',str(O/'r7_actual_prepare01_telemetry'),'--command-json',str(p),'--interval-ms','100'],check=True,timeout=300)
d=json.loads((O/'r7_actual_prepare01/ACTUAL_METADATA_GATE.json').read_text())
assert d['actual_CUBIN_SHA256']=='5a3461fc33dc34bccde4b45a7b5fc55a70f68abf9cd3b453d02cef623119ba9c'
assert d['CPU_text_exact_match'] and d['max_CTA_per_SM']==1 and not d['helper_invoked']
print(json.dumps(d,indent=2),flush=True)
