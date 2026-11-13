"""Mechanical host-only follow-up: 64 identical target launches per CUDA graph."""
import ast
import json
import hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
tree=ast.parse((root/'run_experiment.py').read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='graph_for')
withnode=next(n for n in function.body if isinstance(n,ast.With))
call=withnode.body[-1]
assert ast.unparse(call)=='fn(*rt, **kwargs)'
withnode.body[-1]=ast.For(target=ast.Name(id='_repeat',ctx=ast.Store()),
    iter=ast.Call(func=ast.Name(id='range',ctx=ast.Load()),args=[ast.Constant(value=64)],keywords=[]),
    body=[call],orelse=[])
for n in ast.walk(tree):
    if isinstance(n,ast.Constant) and n.value=='graph_one_kernel': n.value='graph_batch64'
    if isinstance(n,ast.Dict):
        keys=[k.value if isinstance(k,ast.Constant) else None for k in n.keys]
        if 'comparison' in keys and 'method' in keys and 'us' in keys:
            i=keys.index('us')
            n.values[i]=ast.BinOp(left=n.values[i],op=ast.Div(),
                         right=ast.parse('(64 if method != "direct" else 1)',mode='eval').body)
text=ast.unparse(ast.fix_missing_locations(tree))+'\n'
(root/'run_batch.py').write_text(text)
suite=ast.parse((root/'run_suite.py').read_text())
for n in ast.walk(suite):
    if isinstance(n,ast.Constant) and n.value=='run_experiment.py': n.value='run_batch.py'
(root/'run_batch_suite.py').write_text(ast.unparse(ast.fix_missing_locations(suite))+'\n')
(root/'BATCH_MANIFEST.json').write_text(json.dumps({'host_only_change':True,'graph_target_launches':64,
    'reported_event_us':'elapsed graph Event time /64, includes device launch gaps',
    'source_runner_sha256':hashlib.sha256((root/'run_experiment.py').read_bytes()).hexdigest(),
    'batch_runner_sha256':hashlib.sha256(text.encode()).hexdigest(),
    'device_source_sha256':hashlib.sha256((root/'runtime_k3.py').read_bytes()).hexdigest()},indent=2))
