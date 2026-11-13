from pathlib import Path
import tarfile,hashlib,json,subprocess
p=Path(__file__).resolve().parent
items=[x for x in p.glob('guard*') if x.is_dir()]
items += [p/x for x in ['r4_actual_prepare01_telemetry','r7_actual_prepare01_telemetry','r4_graph_edge_api_diag01_supervision','r4_screen01_telemetry','r7_graph_screen01_telemetry','compile_S01','compile_S_attempt1','compile_S02','compile_V01','compile_V02'] if (p/x).exists()]
# No GPU launch: profiler version is a read-only CLI operation.
z=subprocess.run(['/opt/nvidia/nsight-compute/2026.2.1/ncu','--version'],capture_output=True,text=True)
f=Path('/opt/nvidia/nsight-compute/2026.2.1/ncu')
(p/'NCU_CLI_POSTRUN_FINGERPRINT.json').write_text(json.dumps({'stdout':z.stdout,'returncode':z.returncode,'path':str(f),'SHA256':hashlib.sha256(f.read_bytes()).hexdigest(),'scope':'postrun read-only observation, not original per-call capture'},indent=2))
items.append(p/'NCU_CLI_POSTRUN_FINGERPRINT.json')
a=p/'UNIFIED_LEDGER_SUPPORT01.tgz'
assert not a.exists()
with tarfile.open(a,'w:gz') as t:
 for q in items:t.add(q,arcname=q.name)
print(json.dumps({'bytes':a.stat().st_size,'SHA256':hashlib.sha256(a.read_bytes()).hexdigest(),'items':len(items)}))
