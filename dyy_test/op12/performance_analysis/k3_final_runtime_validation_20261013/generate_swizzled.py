"""New files only: remove the experimentally identified scratch bank conflict."""
import ast
import copy
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
tree=ast.parse((root/'runtime_k3.py').read_text())
template=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='K3FinalCOALESCED')
clone=copy.deepcopy(template); clone.name='K3FinalSWIZZLED'
for n in ast.walk(clone):
    if isinstance(n,ast.Constant) and n.value=='COALESCED':n.value='SWIZZLED'
kernel=next(n for n in clone.body if isinstance(n,ast.FunctionDef) and n.name=='kernel')
allocation=next(n for n in kernel.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='sFinal')
allocation.value.args[1]=ast.parse('cute.make_composed_layout(cute.make_swizzle(3,2,5), 0, cute.make_layout((64,128),stride=(128,1)))',mode='eval').body
index=tree.body.index(template); tree.body.insert(index+1,clone)
mapping=next(n for n in tree.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='_KERNEL_CLASSES')
mapping.value.keys.append(ast.parse("('SWIZZLED','event')",mode='eval').body)
mapping.value.values.append(ast.Name(id='K3FinalSWIZZLED',ctx=ast.Load()))
runtime=ast.unparse(ast.fix_missing_locations(tree))+'\n'
(root/'runtime_k3_swizzled.py').write_text(runtime)

micro=ast.parse((root/'final_helper.py').read_text())
for node in micro.body:
    if isinstance(node,ast.ImportFrom) and node.module=='runtime_k3':
        node.module='runtime_k3_swizzled'
        node.names=[ast.alias(name='K3FinalRUNTIME'),ast.alias(name='K3FinalSWIZZLED',asname='K3FinalCOALESCED')]
for n in ast.walk(micro):
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='scratch' for t in n.targets):
        n.value.args[1]=ast.parse('cute.make_composed_layout(cute.make_swizzle(3,2,5), 0, cute.make_layout((64,128),stride=(128,1)))',mode='eval').body
(root/'final_helper_swizzled.py').write_text(ast.unparse(ast.fix_missing_locations(micro))+'\n')

runner=ast.parse((root/'run_experiment.py').read_text())
for n in ast.walk(runner):
    if isinstance(n,ast.Constant):
        if n.value=='runtime_k3.py':n.value='runtime_k3_swizzled.py'
        elif n.value=='final_helper.py':n.value='final_helper_swizzled.py'
        elif n.value=='COALESCED':n.value='SWIZZLED'
    if isinstance(n,ast.ImportFrom) and n.module=='final_helper':n.module='final_helper_swizzled'
function=next(n for n in runner.body if isinstance(n,ast.FunctionDef) and n.name=='graph_for')
withnode=next(n for n in function.body if isinstance(n,ast.With)); call=withnode.body[-1]
withnode.body[-1]=ast.For(target=ast.Name(id='_repeat',ctx=ast.Store()),
    iter=ast.parse('range(64)',mode='eval').body,body=[call],orelse=[])
for n in ast.walk(runner):
    if isinstance(n,ast.Constant) and n.value=='graph_one_kernel':n.value='graph_batch64'
    if isinstance(n,ast.Dict):
        keys=[k.value if isinstance(k,ast.Constant) else None for k in n.keys]
        if 'comparison' in keys and 'method' in keys and 'us' in keys:
            i=keys.index('us'); n.values[i]=ast.BinOp(left=n.values[i],op=ast.Div(),right=ast.parse('(64 if method != "direct" else 1)',mode='eval').body)
(root/'run_layout.py').write_text(ast.unparse(ast.fix_missing_locations(runner))+'\n')

suite=ast.parse((root/'run_suite.py').read_text())
for n in ast.walk(suite):
    if isinstance(n,ast.Constant) and n.value=='run_experiment.py':n.value='run_layout.py'
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='jobs' for t in n.targets):
        n.value=ast.parse("[('profile','SWIZZLED',3),('micro_profile','SWIZZLED',3)]",mode='eval').body
(root/'run_layout_suite.py').write_text(ast.unparse(ast.fix_missing_locations(suite))+'\n')
files=['runtime_k3_swizzled.py','final_helper_swizzled.py','run_layout.py','run_layout_suite.py']
(root/'LAYOUT_FOLLOWUP_MANIFEST.json').write_text(json.dumps({'source_original_runtime_sha256':hashlib.sha256((root/'runtime_k3.py').read_bytes()).hexdigest(),
    'changes':'scratch layout only: Float32 element-offset Swizzle<3,2,5>, preserve low2 bits for 16B vector alignment; verify counters, not assume conflict freedom',
    'stage_count':'same scratch GMEM payload, same barriers, same 32KiB scratch',
    'graph_target_launches':64,'files':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}},indent=2))
