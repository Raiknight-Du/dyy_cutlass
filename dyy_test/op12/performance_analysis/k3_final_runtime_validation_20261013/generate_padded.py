"""Bank-conflict follow-up using simple vector-aligned 132-FP32 row stride."""
import ast
import copy
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parent
tree=ast.parse((root/'runtime_k3.py').read_text())
template=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='K3FinalCOALESCED')
clone=copy.deepcopy(template); clone.name='K3FinalPADDED'
for n in ast.walk(clone):
    if isinstance(n,ast.Constant) and n.value=='COALESCED':n.value='PADDED'
kernel=next(n for n in clone.body if isinstance(n,ast.FunctionDef) and n.name=='kernel')
allocation=next(n for n in kernel.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='sFinal')
allocation.value.args[1]=ast.parse('cute.make_layout((64,128),stride=(132,1))',mode='eval').body
call=next(n for n in clone.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
for n in ast.walk(call):
    if isinstance(n,ast.keyword) and n.arg=='smem':
        assert isinstance(n.value,ast.BinOp) and n.value.right.value==32768
        n.value.right.value=33792
tree.body.insert(tree.body.index(template)+1,clone)
mapping=next(n for n in tree.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='_KERNEL_CLASSES')
mapping.value.keys.append(ast.parse("('PADDED','event')",mode='eval').body)
mapping.value.values.append(ast.Name(id='K3FinalPADDED',ctx=ast.Load()))
(root/'runtime_k3_padded.py').write_text(ast.unparse(ast.fix_missing_locations(tree))+'\n')
micro=ast.parse((root/'final_helper.py').read_text())
for node in micro.body:
    if isinstance(node,ast.ImportFrom) and node.module=='runtime_k3':
        node.module='runtime_k3_padded'
        node.names=[ast.alias(name='K3FinalRUNTIME'),ast.alias(name='K3FinalPADDED',asname='K3FinalCOALESCED')]
for n in ast.walk(micro):
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='scratch' for t in n.targets):
        n.value.args[1]=ast.parse('cute.make_layout((64,128),stride=(132,1))',mode='eval').body
    if isinstance(n,ast.keyword) and n.arg=='smem':n.value=ast.Constant(value=34816)
(root/'final_helper_padded.py').write_text(ast.unparse(ast.fix_missing_locations(micro))+'\n')
runner=ast.parse((root/'run_batch.py').read_text())
for n in ast.walk(runner):
    if isinstance(n,ast.Constant):
        if n.value=='runtime_k3.py':n.value='runtime_k3_padded.py'
        elif n.value=='final_helper.py':n.value='final_helper_padded.py'
        elif n.value=='COALESCED':n.value='PADDED'
    if isinstance(n,ast.ImportFrom) and n.module=='final_helper':n.module='final_helper_padded'
graph_fn=next(n for n in runner.body if isinstance(n,ast.FunctionDef) and n.name=='graph_for')
graph_with=next(n for n in graph_fn.body if isinstance(n,ast.With))
repeat_loop=graph_with.body[-1]
assert isinstance(repeat_loop,ast.For)
graph_with.body[-1]=ast.If(test=ast.parse('len(runtime) == 6',mode='eval').body,
    body=copy.deepcopy(repeat_loop.body),orelse=[repeat_loop])
(root/'run_padded.py').write_text(ast.unparse(ast.fix_missing_locations(runner))+'\n')
suite=ast.parse((root/'run_suite.py').read_text())
for n in ast.walk(suite):
    if isinstance(n,ast.Constant) and n.value=='run_experiment.py':n.value='run_padded.py'
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='jobs' for t in n.targets):
        n.value=ast.parse("[('profile','PADDED',3),('micro_profile','PADDED',3)]",mode='eval').body
(root/'run_padded_suite.py').write_text(ast.unparse(ast.fix_missing_locations(suite))+'\n')
files=['runtime_k3_padded.py','final_helper_padded.py','run_padded.py','run_padded_suite.py']
(root/'PADDED_MANIFEST.json').write_text(json.dumps({'changes':'scratch row stride132 rather than128 FP32; preserve vector16 alignment; verify wavefronts rather than assume conflict freedom',
    'extra_smem_vs_coalesced_bytes':1024,'global_payload_unchanged':True,'barriers_unchanged':True,
    'k3_graph_target_launches':64,'micro_graph_target_launches':1,
    'files':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}},indent=2))
