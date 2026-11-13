from pathlib import Path
import tarfile,json,hashlib,datetime
R=Path(__file__).resolve().parent
extensions={'.py','.mjs','.md','.json','.xlsx','.drawio','.ndjson'}
files=[p for p in R.iterdir() if p.is_file() and p.suffix in extensions and p.name!='LOCAL_ANALYSIS_TRANSPORT.json']
for name in ['inputs','derived','figures','history']:
 files += [p for p in (R/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files=sorted(set(files));out=R/'LOCAL_ANALYSIS_PAYLOAD.tar.gz'
with tarfile.open(out,'w:gz',compresslevel=5) as t:
 for p in files:t.add(p,arcname=p.relative_to(R).as_posix(),recursive=False)
receipt=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),file=out.name,bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),files=len(files),excluded=['.plot_env','node_modules','__pycache__','raw_evidence'],model_enabled=False)
(R/'LOCAL_ANALYSIS_TRANSPORT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))
