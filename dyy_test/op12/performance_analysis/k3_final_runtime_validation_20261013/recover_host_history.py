"""Reconstruct the prior mechanically-generated host script and verify its hash."""
import ast
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parent
tree=ast.parse((root/'run_padded.py').read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='graph_for')
withnode=next(n for n in fn.body if isinstance(n,ast.With))
conditional=withnode.body[-1]
assert isinstance(conditional,ast.If) and ast.unparse(conditional.test)=='len(runtime) == 6'
withnode.body[-1]=conditional.orelse[0]
text=ast.unparse(ast.fix_missing_locations(tree))+'\n'
path=root/'run_padded_graph64_v1.py'
manifests=list(root.glob('runs/padded_event_v1/*/manifest.json'))+list(root.glob('runs/padded_slope_v1/*/manifest.json'))
expected={json.loads(p.read_text())['runner_sha256'] for p in manifests}
assert len(expected)==1, expected
variants={'LF':text.encode(),'CRLF':text.replace('\n','\r\n').encode()}
match=next((k for k,b in variants.items() if hashlib.sha256(b).hexdigest() in expected),None)
assert match is not None, {'expected':sorted(expected),'candidate':{k:hashlib.sha256(b).hexdigest() for k,b in variants.items()}}
path.write_bytes(variants[match])
actual=hashlib.sha256(path.read_bytes()).hexdigest()
checks=[{'manifest':str(p.relative_to(root)),'expected':json.loads(p.read_text())['runner_sha256'],'matches':json.loads(p.read_text())['runner_sha256']==actual} for p in manifests]
assert checks and all(c['matches'] for c in checks)
(root/'HOST_HISTORY_VERIFICATION.json').write_text(json.dumps({'reconstructed_sha256':actual,'newline_encoding':match,'checks':checks,
    'padded_slope_v1':'excluded: micro graph repeated64 times; owned run stopped and rerun with one helper per graph'},indent=2))
print(json.dumps({'checks':len(checks),'all_match':True}))
