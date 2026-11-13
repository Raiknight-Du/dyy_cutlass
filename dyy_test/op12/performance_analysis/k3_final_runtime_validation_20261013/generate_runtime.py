"""Small AST-only experiment patch over frozen K3; no upstream writes."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT.parent

def load(path):
    s = importlib.util.spec_from_file_location('frozen_runtime_generator', path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

def argument(name, annotation='cutlass.Int32'):
    return ast.arg(arg=name, annotation=ast.parse(annotation, mode='eval').body)

def build():
    gen = load(AUDIT / 'k3_causal_ablation_20260928/implementation/generate_causal_k3.py')
    source = AUDIT / 'ablation_20260908/source_original/gated_delta_net_cp.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == gen.SOURCE_SHA256
    orig = gen.method_map(gen.find_class(ast.parse(source.read_text()), gen.SOURCE_CLASS))
    emitted = []
    parts = [gen.module_header()]
    for variant in ('RUNTIME', 'COALESCED'):
        methods = {n: copy.deepcopy(orig[n]) for n in ('__init__', '__call__', 'kernel', 'run_compute_role', 'store_acc', 'load_tensor_tma', 'run_mma_role', 'copy_tensor_to_acc')}
        methods['__init__'] = gen.edit_constructor(orig['__init__'], variant, 'event')
        # Runtime argument, not a constructor option, hence no mode-dependent cache key.
        methods['__call__'].args.args.insert(-1, argument('final_mode'))
        methods['kernel'].args.args.append(argument('final_mode'))
        methods['run_compute_role'].args.args.append(argument('final_mode'))
        for n in ast.walk(methods['__call__']):
            if isinstance(n, ast.Call) and ast.unparse(n.func) == 'self.kernel':
                n.args.append(ast.Name(id='final_mode', ctx=ast.Load()))
        for n in ast.walk(methods['kernel']):
            if isinstance(n, ast.Call) and ast.unparse(n.func) == 'self.run_compute_role':
                n.args.append(ast.Name(id='final_mode', ctx=ast.Load()))
                if variant == 'COALESCED':
                    n.args.append(ast.Name(id='sFinal', ctx=ast.Load()))
        tail = next(n for n in methods['run_compute_role'].body if isinstance(n, ast.If) and ast.unparse(n.test) == 'cutlass.const_expr(self.store_final_state)')
        final_call = next(n for n in ast.walk(tail) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'self.store_acc')
        final_call.args.append(ast.Name(id='final_mode', ctx=ast.Load()))
        if variant == 'RUNTIME':
            store = methods['store_acc']
            store.args.args.append(argument('final_mode'))
            store.args.defaults.append(ast.parse('cutlass.Int32(3)', mode='eval').body)
            loop = next(n for n in store.body if isinstance(n, ast.For))
            read, fixed, output = copy.deepcopy(loop.body)
            runtime_body = ast.parse('if (final_mode & 1) != 0:\n    pass\nif (final_mode & 2) != 0:\n    pass').body
            runtime_body[0].body = [copy.deepcopy(fixed)]
            runtime_body[1].body = copy.deepcopy(output.body)
            loop.body = [read, ast.If(test=ast.parse('cutlass.const_expr(store_output)', mode='eval').body, body=runtime_body, orelse=[fixed])]
        else:
            for node in ast.walk(methods['__call__']):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'launch':
                    for kw in node.keywords:
                        if kw.arg == 'smem':
                            kw.value = ast.BinOp(left=kw.value, op=ast.Add(), right=ast.Constant(value=32768))
            final_call.func.attr = 'store_final_layout'
            final_call.args.append(ast.Name(id='sFinal', ctx=ast.Load()))
            methods['run_compute_role'].args.args.append(argument('sFinal', 'cute.Tensor'))
            allocator_index = next(i for i,n in enumerate(methods['kernel'].body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'allocator')
            methods['kernel'].body[allocator_index+1:allocator_index+1] = ast.parse('sFinal = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((64,128), stride=(128,1)), byte_alignment=128)').body
            helper = copy.deepcopy(orig['store_acc'])
            helper.name = 'store_final_layout'
            helper.args.args += [argument('final_mode'), argument('sFinal', 'cute.Tensor')]
            helper.body = ast.parse('''
tCtAcc_mn = utils.gemm.sm100.transform_partitioned_tensor_layout(tCtAcc)
atom_t2r = cute.make_copy_atom(tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(16)), self.acc_dtype)
tiled_t2r = tcgen05.make_tmem_copy(atom_t2r, tCtAcc_mn[None,None,0])
thr = tiled_t2r.get_slice(tidx)
tsrc = thr.partition_S(tCtAcc_mn)
tdst = thr.partition_D(sFinal)
reg = cute.make_rmem_tensor_like(tdst, self.acc_dtype)
for sub in cutlass.range(cute.size(reg, mode=[2]), unroll_full=True):
    cute.copy(tiled_t2r, tsrc[None,0,sub,stage], reg[None,0,sub])
    cute.autovec_copy(reg[None,0,sub], tdst[None,0,sub])
pipeline.NamedBarrier(barrier_id=2, num_threads=128).arrive_and_wait()
atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype, num_bits_per_copy=128)
ct = cute.make_tiled_copy_tv(atom, cute.make_layout((4,32), stride=(32,1)), cute.make_layout((1,4), stride=(4,1)))
sl = ct.get_slice(tidx)
sr = sl.partition_S(sFinal)
sr = cute.make_tensor(sr.iterator.align(16), sr.layout)
rf = cute.make_rmem_tensor_like(sr, self.acc_dtype)
rf = cute.make_tensor(rf.iterator.align(16), rf.layout)
cute.copy(ct, sr, rf)
if (final_mode & 1) != 0:
    df = sl.partition_D(gFixedState)
    df = cute.make_tensor(df.iterator.align(16), df.layout)
    cute.copy(ct, rf, df)
if (final_mode & 2) != 0:
    do = sl.partition_D(gOutputState)
    do = cute.make_tensor(do.iterator.align(16), do.layout)
    cute.copy(ct, rf, do)
pipeline.NamedBarrier(barrier_id=2, num_threads=128).arrive_and_wait()
''').body
            methods['store_final_layout'] = helper
        name = 'K3Final' + variant
        cls = gen.emit_class(name, methods, variant, 'event')
        if variant == 'COALESCED':
            cls.body.append(methods['store_final_layout'])
        parts.append(ast.unparse(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])))+'\n')
        emitted.append((variant, 'event', name))
    parts.append(gen.make_runtime_footer(emitted))
    text = ''.join(parts)
    ast.parse(text)
    (ROOT / 'runtime_k3.py').write_text(text, encoding='utf8')
    (ROOT / 'SOURCE_MANIFEST.json').write_text(json.dumps({'source_commit':gen.SOURCE_COMMIT, 'source_sha256':gen.SOURCE_SHA256, 'runtime_module_sha256':hashlib.sha256(text.encode()).hexdigest(), 'mode_is_runtime_Int32':True, 'modes':[1,2,3], 'IKET':False, 'layout_added_smem_bytes':32768},indent=2))

if __name__ == '__main__':
    build()
