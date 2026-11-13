"""Snapshot exact dependencies and hash the read-only evidence delivery."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root=Path(__file__).resolve().parent
op12=root.parents[1]
repo=op12/'upstreams/flashinfer'
snapshot=root/'inputs_snapshot'
snapshot.mkdir(exist_ok=True)
files={
 'run_causal_k3.py':root.parent/'k3_causal_ablation_20260928/runner/run_causal_k3.py',
 'run_k3_iket.py':root.parent/'k3_iket_stage_analysis_20260928/implementation/run_k3_iket.py',
 'gated_delta_net_cp.py':repo/'flashinfer/gdn_kernels/blackwell/gated_delta_net_cp.py',
 'gdn_cp_prefill.py':repo/'flashinfer/gdn_kernels/blackwell/gdn_cp_prefill.py',
}
records=[]
for label,path in files.items():
    target=snapshot/label
    if target.exists():
        assert target.read_bytes()==path.read_bytes(), target
    else:
        shutil.copy2(path,target)
    records.append({'source':str(path),'snapshot':str(target.relative_to(root)),
                    'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
assert head=='e77a4a0d276367895c3b50a642fd8f326c03fb72' and not dirty
assert records[2]['sha256']=='cb9a174f6fb02d6a69415e34dbe1bf799bf283e2a46747b39d4c538d83919b8b'
assert records[3]['sha256']=='8dd9577f8b950f03f69ce22fdaf3b84acf8ef016c70d8d40c6cfdc9f2b2493b5'
env={}
for label,cmd in {
 'pip_freeze':['/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/.venv/bin/python','-m','pip','freeze'],
 'ncu_version':['/opt/nvidia/nsight-compute/2026.2.1/ncu','--version'],
 'gpu_after':['nvidia-smi','--query-gpu=name,uuid,driver_version,clocks.sm,clocks.memory,temperature.gpu,power.draw,utilization.gpu','--format=csv'],
 'gpu_processes_after':['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv'],
}.items():
    p=subprocess.run(cmd,capture_output=True,text=True)
    env[label]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
if not (snapshot/'SNAPSHOT_MANIFEST.json').exists():
    (snapshot/'SNAPSHOT_MANIFEST.json').write_text(json.dumps({'inputs':records,'head':head,'dirty':dirty,'environment':env},indent=2))
hashes=[]
for path in sorted(root.rglob('*')):
    rel=path.relative_to(root)
    if not path.is_file() or any(p in ('__pycache__','compile_cache') for p in rel.parts) or path.name=='EVIDENCE_SHA256.json':
        continue
    hashes.append({'path':rel.as_posix(),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(root/'EVIDENCE_SHA256.json').write_text(json.dumps({'files':hashes,'excluded_directory_names':['compile_cache','__pycache__']},indent=2))
print(json.dumps({'frozen_inputs':len(records),'evidence_files':len(hashes),'bytes':sum(r['bytes'] for r in hashes)},indent=2))
