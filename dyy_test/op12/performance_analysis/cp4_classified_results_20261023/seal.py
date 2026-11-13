from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
skip={'SHA256SUMS.json','MIRROR_VERIFICATION.json','postprocess.log'}
files={p.relative_to(R).as_posix():sha(p) for p in sorted(R.rglob('*')) if p.is_file() and p.name not in skip and '__pycache__' not in p.parts}
(R/'SHA256SUMS.json').write_text(json.dumps(files,indent=2))
print(len(files))
