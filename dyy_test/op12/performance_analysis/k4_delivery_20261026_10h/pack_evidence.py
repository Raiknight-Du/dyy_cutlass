from pathlib import Path
import argparse,hashlib,json,tarfile
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--items',nargs='+',required=True);a=p.parse_args()
O=Path(__file__).resolve().parent;out=O/(a.name+'.tgz');assert not out.exists()
with tarfile.open(out,'w:gz') as t:
    for x in a.items:
        target=(O/x).resolve();assert target.is_relative_to(O)
        t.add(target,arcname=x)
d=dict(file=out.name,bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest())
(O/(a.name+'_MANIFEST.json')).write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
