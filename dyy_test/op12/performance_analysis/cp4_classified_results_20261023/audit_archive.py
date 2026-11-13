from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;B=R.parent;A=B/'_superseded/cp4_frequency_20261023'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return h.hexdigest()
entries=[]
for name in ('cp4_inchain_frequency_retest_20261023','cp4_cold_l2_lock1845_20261023'):
    p=A/name;assert p.is_dir() and not (B/name).exists()
    raw=json.loads((p/'SHA256SUMS.json').read_text(encoding='utf-8-sig'))
    for key,expected in raw.items():
        q=p/Path(key);assert q.is_file(),str(q)
        assert sha(q)==expected,str(q)
    entries.append({'original':name,'archive':p.relative_to(B).as_posix(),
        'original_manifest_sha256':sha(p/'SHA256SUMS.json'),'verified_manifest_files':len(raw),
        'action':'recoverable directory move; no content changes'})
    tar=A/(name+'_transport.tar.gz');assert tar.is_file() and not (B/tar.name).exists()
    entries.append({'original':tar.name,'archive':tar.relative_to(B).as_posix(),'sha256':sha(tar),'bytes':tar.stat().st_size,'action':'recoverable move'})
(R/'CLEANUP_MANIFEST.json').write_text(json.dumps({'deleted_irrecoverably':False,'entries':entries},indent=2),encoding='utf-8')
print(json.dumps(entries,indent=2))
