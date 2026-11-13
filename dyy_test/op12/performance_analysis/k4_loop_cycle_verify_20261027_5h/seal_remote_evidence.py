from pathlib import Path
import hashlib,json,subprocess,datetime,tarfile,os,sys
R=Path(__file__).resolve().parent
assert R.name=='k4_loop_cycle_verify_20261027_5h' and R.parent.name=='performance_analysis'
REPO=R.parents[1]/'upstreams/flashinfer'
def sh(*args):return subprocess.check_output(args,text=True).strip()
assert sh('git','-C',str(REPO),'rev-parse','HEAD')=='e77a4a0d276367895c3b50a642fd8f326c03fb72'
assert not sh('git','-C',str(REPO),'status','--porcelain')
apps=sh('nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits')
env=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),upstream_HEAD=sh('git','-C',str(REPO),'rev-parse','HEAD'),upstream_clean=True,
 GPU=sh('nvidia-smi','--query-gpu=name,uuid,driver_version,compute_cap,memory.total,clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv'),
 SM_count=int(sh(sys.executable,'-c','import torch;print(torch.cuda.get_device_properties(0).multi_processor_count)')),
 active_apps=apps,NCU=sh('/opt/nvidia/nsight-compute/2026.2.1/ncu','--version'),
 CUDA=sh('/usr/local/cuda/bin/nvcc','--version'),frequency_policy_modified=False,model_enabled=False)
(R/'ENVIRONMENT_FINAL.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
files=[p for p in sorted(R.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ['REMOTE_MANIFEST.json','REMOTE_ARCHIVE_RECEIPT.json'] and not p.name.endswith('.tar.gz')]
records=[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
manifest=dict(records=records,total_files=len(records),total_bytes=sum(x['bytes'] for x in records),NCU_reports=sum(x['path'].endswith('.ncu-rep') for x in records),model_enabled=False)
(R/'REMOTE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
archive=R.parent/(R.name+'_evidence.tar.gz')
assert not archive.exists(),archive
with tarfile.open(archive,'w:gz',compresslevel=5,dereference=True) as tar:
 for p in files+[R/'REMOTE_MANIFEST.json']:tar.add(p,arcname=str(p.relative_to(R)),recursive=False)
receipt=dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256((R/'REMOTE_MANIFEST.json').read_bytes()).hexdigest(),files=len(records),NCU_reports=manifest['NCU_reports'],unmodified_upstream=True)
(R/'REMOTE_ARCHIVE_RECEIPT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)
