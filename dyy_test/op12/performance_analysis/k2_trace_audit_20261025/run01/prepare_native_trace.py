"""Bounded source-pinned coarse/fine clock64 views; never changes upstream."""
import ast, copy, hashlib, json
from pathlib import Path

R = Path(__file__).resolve().parent
SRC = Path('/home/ubuntu/workspace/dyy/dyy_cutlass/dyy_test/op12/upstreams/flashinfer/flashinfer/gdn_kernels/blackwell/gated_delta_net_cp.py')
OUT = R / 'K2_CLOCK'
OUT.mkdir(exist_ok=False)
tree = ast.parse(SRC.read_text())
base = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CPDeltaRuleMNPrecomputeUtcmma1Sm100')
methods = {n.name: n for n in base.body if isinstance(n, ast.FunctionDef)}
ROLES = ['run_compute_group_0', 'run_compute_group_1', 'run_transfer_mma_role', 'run_state_mma_role', 'run_load_alpha_role_sm100']
STAGES = {
    0: ('ROLE_LOOP_BODY', 'mixed role loop; includes waits, excludes init/final store'),
    1: ('ROLE_INIT', 'compute TMEM initialization+publish or issuer init-ready wait'),
    2: ('ROLE_FINAL', 'M/N final store and done handoff; CG1 includes wait for CG0 done'),
    3: ('SELECTED_BLOCK', 'one complete source loop body at runtime selected block'),
    4: ('M_ALPHA_X_READY', 'wait alpha/X plus X producer slot acquire'),
    5: ('X_MATERIALIZE_PUBLISH', 'TMEM load/FP16 pack/SMEM store/fence and publish'),
    6: ('M_OPERAND_PUBLISH', 'TMEM read/FP16 pack/TMEM store/fence and m_in publish'),
    7: ('Z_ACC_WAIT', 'consumer wait call; not MMA latency'),
    8: ('Z_CONVERT_PUBLISH', 'TMEM read/pack/TMEM write/fence and z_ready publish'),
    9: ('M_COEFF_UPDATE_WAIT', 'coefficient load plus m_acc consumer wait'),
    10: ('M_SCALE_RELEASE', 'TMEM scale and releases; no extra completion barrier'),
    11: ('N_V_ALPHA_READY', 'V/alpha consumer waits'),
    12: ('N_OPERAND_PUBLISH', 'N conversion/TMEM publish'),
    13: ('Y_ACC_WAIT', 'Y consumer wait'),
    14: ('Y_PROCESS_SCALE_PUBLISH', 'coefficient+V/Y processing+N scale+fence+publish; joined'),
    15: ('N_UPDATE_WAIT_RELEASE', 'n_acc consumer wait and releases'),
    16: ('K_READY_WAIT', 'W8 K ready wait'),
    17: ('M_OPERAND_WAIT', 'W8 m_in ready wait'),
    18: ('Z_ISSUE_COMMIT', '8 Z MMA atoms+original z_acc acquire/commit; not completion'),
    19: ('Z_READY_WAIT_RELEASE', 'W8 z_ready wait and operand release'),
    20: ('X_READY_M_SLOT', 'W8 X ready wait plus M accumulator slot acquire'),
    21: ('M_UPDATE_ISSUE_RELEASE', 'M-update atoms+commit and operand releases'),
    22: ('KT_READY_X_SLOT', 'W11 K/T waits plus X accumulator slot acquire'),
    23: ('X_ISSUE_COMMIT_RELEASE_T', 'X atoms+commit+T release'),
    24: ('X_N_READY_Y_SLOT', 'W11 X/n_in waits plus Y slot acquire'),
    25: ('Y_ISSUE_COMMIT_RELEASE', 'Y atoms+commit+input releases'),
    26: ('Y_READY_N_SLOT', 'Y ready wait plus N accumulator slot acquire'),
    27: ('N_UPDATE_ISSUE_RELEASE', 'N-update atoms+commit+X release'),
    28: ('K_TMA_CALL', 'K slot acquire/TMA submit; not transfer completion'),
    29: ('V_TMA_CALL', 'V slot acquire/TMA submit; not transfer completion'),
    30: ('T_TMA_CALL', 'T slot acquire/TMA submit; not transfer completion'),
    31: ('ALPHA_PRODUCE_BLOCK', 'alpha slot acquire/global load/scan/SMEM/fence/commit'),
}
header = '''import flashinfer.gdn_kernels.blackwell.gated_delta_net_cp as _source
globals().update({k:v for k,v in vars(_source).items() if not k.startswith('__')})
@cute.jit
def _stamp(g_trace, stage:cutlass.Int32, endpoint:cutlass.Int32):
    if cute.arch.block_idx()[0]==0:
        if cute.arch.lane_idx()==0:
            wi=cute.arch.warp_idx()
            g_trace[wi,stage,endpoint,0]=cute.arch.clock64()
            g_trace[wi,stage,endpoint,1]=cutlass.Int64(cute.arch.smid())
'''

def marker(stage, end, sampled=False):
    code = f'_stamp(g_trace,cutlass.Int32({stage}),cutlass.Int32({end}))'
    if sampled:
        code = 'if blk==sample_block:\n    ' + code
    return ast.parse(code).body[0]

def wrap(body, stage, sampled=False):
    return [marker(stage, 0, sampled), *body, marker(stage, 1, sampled)]

def find(body, text):
    return next(i for i, n in enumerate(body) if text in ast.unparse(n))

def fine_body(name, b):
    if name == 'run_compute_group_0':
        split = next(i for i, n in enumerate(b) if isinstance(n, ast.If) and ast.unparse(n.test) == 'blk > 0')
        start_helper = find(b, 'self._materialize_x(')
        prefix = wrap(b[:start_helper], 4, True) + wrap(b[start_helper:split], 5, True)
        branch = b[split]
        z_wait = find(branch.body, 'z_acc_consumer.wait_and_advance')
        branch.body = wrap(branch.body[:z_wait], 6, True) + wrap([branch.body[z_wait]], 7, True) + wrap(branch.body[z_wait+1:], 8, True)
        tail = b[split+1:]
        scale = find(tail, 'self._scale_tmem_matrix(')
        return prefix + [branch] + wrap(tail[:scale], 9, True) + wrap(tail[scale:], 10, True)
    if name == 'run_compute_group_1':
        conv = find(b, 'self._convert_matrix_to_z_input(')
        yw = find(b, 'y_acc_consumer.wait_and_advance')
        nw = find(b, 'n_acc_consumer.wait_and_advance')
        return wrap(b[:conv], 11, True) + wrap(b[conv:yw], 12, True) + wrap([b[yw]], 13, True) + wrap(b[yw+1:nw], 14, True) + wrap(b[nw:], 15, True)
    if name == 'run_transfer_mma_role':
        split = next(i for i, n in enumerate(b) if isinstance(n, ast.If) and ast.unparse(n.test) == 'blk > 0')
        branch = b[split]
        mma = next(i for i, n in enumerate(branch.body) if isinstance(n, ast.For))
        ready = find(branch.body, 'z_ready_consumer.wait_and_advance')
        branch.body = wrap(branch.body[:mma], 17, True) + wrap(branch.body[mma:ready], 18, True) + wrap(branch.body[ready:], 19, True)
        tail = b[split+1:]
        update = next(i for i, n in enumerate(tail) if isinstance(n, ast.For))
        return wrap(b[:split], 16, True) + [branch] + wrap(tail[:update], 20, True) + wrap(tail[update:], 21, True)
    if name == 'run_state_mma_role':
        loops = [i for i, n in enumerate(b) if isinstance(n, ast.For)]
        assert len(loops) == 3
        x,y,n = loops
        xr = find(b, 'x_ready_handle =')
        yr = find(b, 'y_ready_consumer.wait_and_advance')
        return wrap(b[:x], 22, True) + wrap(b[x:xr], 23, True) + wrap(b[xr:y], 24, True) + wrap(b[y:yr], 25, True) + wrap(b[yr:n], 26, True) + wrap(b[n:], 27, True)
    if name == 'run_load_alpha_role_sm100':
        return wrap(b, 31, True)
    raise ValueError(name)

parts = [header]
maps = {}
for variant in ['B1','COARSE','FINE']:
    names = ['__call__','kernel'] + (ROLES if variant != 'B1' else [])
    mm = {n:copy.deepcopy(methods[n]) for n in names}
    for name, fn in mm.items():
        at = -1 if name == '__call__' else len(fn.args.args)
        for arg, ann in [('g_trace','cute.Tensor'),('sample_block','cutlass.Int32')]:
            fn.args.args.insert(at, ast.arg(arg=arg, annotation=ast.parse(ann, mode='eval').body))
            if at >= 0: at += 1
    for fn in mm.values():
        for call in ast.walk(fn):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                name = call.func.attr
                if name == 'kernel' or (variant != 'B1' and name in ROLES):
                    call.args.extend([ast.Name(id='g_trace',ctx=ast.Load()),ast.Name(id='sample_block',ctx=ast.Load())])
    if variant != 'B1':
        for name in ROLES:
            fn = mm[name]
            pos = next(i for i,n in enumerate(fn.body) if isinstance(n,ast.For) and 'range(num_blocks' in ast.unparse(n.iter))
            loop = fn.body[pos]
            # Anonymous source loop variable has no semantic uses. Record this rename explicitly.
            loop.target = ast.Name(id='blk',ctx=ast.Store())
            loop.body = wrap(fine_body(name,loop.body) if variant=='FINE' else loop.body,3,True)
            before,after = fn.body[:pos],fn.body[pos+1:]
            # Init metadata/tensor construction is not labelled as TMEM initialization.
            if name in ['run_compute_group_0','run_compute_group_1']:
                begin = find(before,'self._initialize_tmem_matrix(')
                before = before[:begin]+wrap(before[begin:],1)
                if after: after = wrap(after,2)
            elif name in ['run_transfer_mma_role','run_state_mma_role']:
                init = next(i for i,n in enumerate(before) if 'init_consumer.wait_and_advance' in ast.unparse(n))
                before = before[:init]+wrap([before[init]],1)+before[init+1:]
            fn.body = before+wrap([loop],0)+after
        # TMA producer is in kernel, not in a separate method.
        kernel = mm['kernel']
        loop = next(n for n in ast.walk(kernel) if isinstance(n,ast.For) and getattr(n.target,'id',None)=='blk')
        if variant=='FINE':
            assert len(loop.body)==3
            loop.body = sum((wrap([n],28+i,True) for i,n in enumerate(loop.body)),[])
        loop.body = wrap(loop.body,3,True)
        class ReplaceLoop(ast.NodeTransformer):
            def visit_For(self,n):
                if n is loop:return wrap([n],0)
                return self.generic_visit(n)
        mm['kernel']=ReplaceLoop().visit(kernel)
    dc=ast.ClassDef(name='K2Trace'+variant,bases=[ast.Name(id=base.name,ctx=ast.Load())],keywords=[],body=list(mm.values()),decorator_list=[])
    parts.append(ast.unparse(ast.fix_missing_locations(dc)))
    maps[variant] = {name: {'source_start':methods[name].lineno,'source_end':methods[name].end_lineno} for name in names}
code='\n\n'.join(parts)+'\n'
ast.parse(code)
(OUT/'variants.py').write_text(code)
(OUT/'GENERATION.json').write_text(json.dumps(dict(source=str(SRC),source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),variant_sha256=hashlib.sha256(code.encode()).hexdigest(),stages={str(k):v[0] for k,v in STAGES.items()},semantics={str(k):v[1] for k,v in STAGES.items()},overrides=maps,trace_shape=[12,48,2,2],sample_CTA=0,anonymous_loop_rename='only _ -> blk; no source arithmetic/pipeline edit',frequency_request_MHz=1845,model_enabled=False,local_bias_bound=None),indent=2))
print('GENERATED_COARSE_FINE',flush=True)
