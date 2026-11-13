from pathlib import Path,PurePosixPath
import json,hashlib,tarfile,datetime
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
receipt=read(R/'REMOTE_ARCHIVE_RECEIPT.json');a=R/'K4_LOOP_EVIDENCE_20261027.tar.gz'
assert hashlib.sha256(a.read_bytes()).hexdigest()==receipt['sha256']
manifest=read(R/'REMOTE_MANIFEST.json')
assert hashlib.sha256((R/'REMOTE_MANIFEST.json').read_bytes()).hexdigest()==receipt['manifest_sha256']
dest=R/'raw_evidence';assert not dest.exists()
with tarfile.open(a) as tf:
 for m in tf.getmembers():
  p=PurePosixPath(m.name);assert not p.is_absolute() and '..' not in p.parts and m.isfile(),m.name
 dest.mkdir();tf.extractall(dest)
errors=[];common=[]
for r in manifest['records']:
 p=dest/r['path'];h=hashlib.sha256(p.read_bytes()).hexdigest()
 if p.stat().st_size!=r['bytes'] or h!=r['sha256']:errors.append(r['path'])
 q=R/r['path']
 if q.is_file():
  h2=hashlib.sha256(q.read_bytes()).hexdigest();common.append(dict(path=r['path'],match=h2==h))
assert not errors,errors
assert all(z['match'] for z in common),[z for z in common if not z['match']]
out=dict(status='PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files_verified=len(manifest['records']),
 NCU_reports=manifest['NCU_reports'],archive_bytes=a.stat().st_size,archive_sha256=receipt['sha256'],
 local_remote_common_files=len(common),common_file_hashes_match=True,model_enabled=False,effective_model_value=None)
(R/'LOCAL_EVIDENCE_VERIFICATION.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))
