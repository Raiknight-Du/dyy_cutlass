from pathlib import Path
import subprocess,sys,json,hashlib
R=Path(__file__).resolve().parent
assert json.loads((R/'FORMAL_STATUS.json').read_text())['done']
restored=json.loads((R/'FORMAL_CLOCK_RESTORED.json').read_text())
assert restored['reset_gpu_clocks_issued'] and restored['memory_clock_unchanged']
for s in (8192,16384):
    for t in (1,2,3):
        stem=f'formal_S{s}_cold_locked1845_metrics_t{t}'
        folder=R/stem
        if not folder.exists():
            subprocess.run([sys.executable,str(R/'analyze.py'),stem,f'S{s}_cold_locked1845_metrics_t{t}'],check=True)
subprocess.run([sys.executable,str(R/'organize_results.py')],check=True)
repo=R.parents[1]/'upstreams/flashinfer'
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='e77a4a0d276367895c3b50a642fd8f326c03fb72'
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
manifest={p.relative_to(R).as_posix():sha(p) for p in sorted(R.rglob('*')) if p.is_file() and p.name not in ('SHA256SUMS.json','postprocess.log','MIRROR_VERIFICATION.json') and '__pycache__' not in p.parts}
(R/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
print('READY',len(manifest))
