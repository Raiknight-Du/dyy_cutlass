#!/usr/bin/env python3
"""AST-only generator of the K3 observer-trace variants.  No CUDA import, no upstream write.

Reads the pinned gated_delta_net_cp.py, verifies its SHA256, and emits generated/variants_k3obs.py with one class per
candidate.  Every class copies `__call__` and `kernel` verbatim, threads two tensor arguments (g_obs, g_hdr) through
them, and edits only the idle-warp `else` branch (warps 6-7).  Candidates differ in what is live at the role dispatch
and in the register budget of that branch, because ptxas re-allocates the whole kernel when that changes:

  name          observer method          index   warps  idle-branch budget      note
  K3ObsB1       none (control)           -       -      32 (unchanged)          only the two unused arguments added
  K3Obs         run_observer_role        bx      6,7    64                      full bookkeeping (per-event try counts)
  K3ObsLean32   run_observer_role_lean   smid    6,7    32 (unchanged)          no CTA index live-in, no try counts
  K3ObsLeanND   run_observer_role_lean   smid    6,7    none (no decrease)      idle warps keep the launch count (120)
  K3ObsA32      run_observer_role_a      smid    6      32 (unchanged)          chain A only (READY, MMA_DONE, UPDATE_RELEASED)
  K3ObsAND      run_observer_role_a      smid    6      none (no decrease)
  K3ObsABND     run_observer_role_ab     smid    6      none (no decrease)      both chains on one warp, causal order per partition
  K3ObsI        run_observer_role        bx      6,7    64                      + issue stamps in warps 4/5 (compile-audit gated)

Register pool: the kernel launches with 120 registers per thread (8 warps x 120 = 960); the compute warps' increase
target is 120 = the launch count, so no warp ever waits for registers released by warps 6-7, and every budget above is
pool-consistent (max 4x120 + 2x32 + 2x120 = 784 <= 960).

Recovery proof: stripping the generator's additions (arguments, observer call, budget edit, issue stamps) from every
generated method reproduces the original method AST exactly (ast.dump equality); the generator aborts otherwise.
"""
import argparse, ast, copy, hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CFG = json.loads((ROOT / 'config.json').read_text())
OBS_SRC = HERE / 'observer_role_src.py'
NEW_ARGS = ('g_obs', 'g_hdr')
ORIG_DEC = 'cute.arch.setmaxregister_decrease(self.num_regs_other)'
CANDIDATES = [
    # name, method, index, budget ('32' keep original statement, '64', 'none'), issue stamps
    ('K3ObsB1', None, None, '32', False),
    ('K3Obs', 'run_observer_role', 'bx', '64', False),
    ('K3ObsLean32', 'run_observer_role_lean', 'smid', '32', False),
    ('K3ObsLeanND', 'run_observer_role_lean', 'smid', 'none', False),
    ('K3ObsA32', 'run_observer_role_a', 'smid', '32', False),
    ('K3ObsAND', 'run_observer_role_a', 'smid', 'none', False),
    ('K3ObsABND', 'run_observer_role_ab', 'smid', 'none', False),
    ('K3ObsI', 'run_observer_role', 'bx', '64', True),
]


def _arg(name): return ast.arg(arg=name, annotation=ast.parse('cute.Tensor', mode='eval').body)
def _stmt(src): return ast.parse(src).body[0]


class AppendArgs(ast.NodeTransformer):
    def __init__(self, mapping): self.mapping = mapping
    def visit_Call(self, n):
        self.generic_visit(n); f = n.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == 'self' and f.attr in self.mapping:
            for name in self.mapping[f.attr]: n.args.append(ast.Name(id=name, ctx=ast.Load()))
        return n


def load_source(path: Path):
    text = path.read_bytes(); sha = hashlib.sha256(text).hexdigest()
    if sha != CFG['source_sha256']: raise SystemExit(f'pinned source SHA mismatch: {sha} ({path})')
    tree = ast.parse(text.decode())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CFG['k3_class'])
    return sha, {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def observer_pieces():
    tree = ast.parse(OBS_SRC.read_text()); helpers, methods = [], {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name.startswith('run_observer_role'): methods[n.name] = n
        else: helpers.append(n)
    return ast.unparse(ast.Module(body=helpers, type_ignores=[])), methods


def find_tma_if(kernel_fn):
    for n in ast.walk(kernel_fn):
        if isinstance(n, ast.If) and ast.unparse(n.test) == 'warp_idx == self.tma_warp_id': return n
    raise SystemExit('TMA branch not found in kernel()')


def build_variant(name, method, index, budget, issue, orig, obs_methods):
    mm = {k: copy.deepcopy(orig[k]) for k in ('__call__', 'kernel')}
    if issue:
        for k in ('run_mma_role', 'load_tensor_tma'): mm[k] = copy.deepcopy(orig[k])
    for a in NEW_ARGS:
        mm['__call__'].args.args.insert(-1, _arg(a)); mm['kernel'].args.args.append(_arg(a))
    if issue:
        for k in ('run_mma_role', 'load_tensor_tma'): mm[k].args.args.append(_arg('g_obs'))
    mapping = {'kernel': NEW_ARGS}
    if issue: mapping.update({'run_mma_role': ('g_obs',), 'load_tensor_tma': ('g_obs',)})
    for f in mm.values(): AppendArgs(mapping).visit(f)
    body = list(mm.values())
    if method is not None:
        tma_if = find_tma_if(mm['kernel'])
        assert len(tma_if.orelse) == 1 and ast.unparse(tma_if.orelse[0]) == ORIG_DEC, ast.unparse(tma_if.orelse)
        call = f'self.run_observer_role(warp_idx, {"bx, " if index == "bx" else ""}storage, num_chunks, g_obs, g_hdr)'
        if budget == '32': tma_if.orelse = [tma_if.orelse[0], _stmt(call)]
        elif budget == '64': tma_if.orelse = [_stmt('cute.arch.setmaxregister_decrease(64)'), _stmt(call)]
        else: tma_if.orelse = [_stmt(call)]
        m = copy.deepcopy(obs_methods[method]); m.name = 'run_observer_role'; body.append(m)
    if issue:
        loop = next(n for n in mm['run_mma_role'].body if isinstance(n, ast.For)); b = loop.body
        assert ast.unparse(b[0]).startswith('m_handle = ') and ast.unparse(b[4]) == 'done_handle.commit()', [ast.unparse(x) for x in b]
        st = lambda slot: _stmt(f'_stamp_issue(g_obs, cutlass.Int32(2), cutlass.Int32(_iter_idx), cutlass.Int32({slot}))')
        loop.body = [st(0), b[0], b[1], b[2], st(1), b[3], b[4], st(2), *b[5:]]
        load = mm['load_tensor_tma']; ret = load.body[-1]; assert isinstance(ret, ast.Return)
        load.body = [*load.body[:-1], _stmt('_stamp_issue(g_obs, cutlass.Int32(3), cutlass.Int32(chunk_idx), cutlass.Int32(1) if cutlass.const_expr(is_transfer) else cutlass.Int32(0))'), ret]
    cls = ast.ClassDef(name=name, bases=[ast.Name(id=CFG['k3_class'], ctx=ast.Load())], keywords=[], body=body, decorator_list=[])
    return cls, mm


def strip_for_recovery(fn):
    f = copy.deepcopy(fn); f.args.args = [a for a in f.args.args if a.arg not in NEW_ARGS]
    class Strip(ast.NodeTransformer):
        def visit_Call(self, n):
            self.generic_visit(n); fx = n.func
            if isinstance(fx, ast.Attribute) and isinstance(fx.value, ast.Name) and fx.value.id == 'self':
                n.args = [a for a in n.args if not (isinstance(a, ast.Name) and a.id in NEW_ARGS)]
            return n
        def visit_If(self, n):
            self.generic_visit(n)
            if ast.unparse(n.test) == 'warp_idx == self.tma_warp_id':
                kept = [s for s in n.orelse if not ast.unparse(s).startswith('self.run_observer_role(')]
                kept = [_stmt(ORIG_DEC) if ast.unparse(s) == 'cute.arch.setmaxregister_decrease(64)' else s for s in kept]
                if not kept: kept = [_stmt(ORIG_DEC)]        # budget 'none' removed the decrease; restore it
                n.orelse = kept
            return n
        def visit_Expr(self, n):
            self.generic_visit(n)
            if ast.unparse(n).startswith('_stamp_issue('): return None
            return n
    Strip().visit(f); return ast.fix_missing_locations(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', default=str(Path(CFG['host']['flashinfer_repo']) / CFG['source_relpath']))
    ap.add_argument('--out', default=str(ROOT / 'generated'))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sha, orig = load_source(Path(a.source)); helpers_src, obs_methods = observer_pieces()
    texts = ['from flashinfer.gdn_kernels.blackwell.gated_delta_net_cp import *\n\n# ---- observer helpers (verbatim from k3trace/observer_role_src.py) ----\n' + helpers_src + '\n\n']
    inventory = []
    for name, method, index, budget, issue in CANDIDATES:
        cls, mm = build_variant(name, method, index, budget, issue, orig, obs_methods)
        for k, f in mm.items():
            if ast.dump(strip_for_recovery(f)) != ast.dump(orig[k]): raise SystemExit(f'recovery check failed for {name}.{k}')
        texts.append(ast.unparse(ast.fix_missing_locations(cls)) + '\n\n')
        inventory.append(dict(class_name=name, observer_method=method, index=index, idle_branch_budget=budget, issue_stamps=issue,
                              observer_warps=([] if method is None else ([6] if method.endswith(('_a', '_ab')) else [6, 7])), issue_stamp_warps=[4, 5] if issue else [],
                              modified_methods=list(mm.keys()), compute_warps_touched=False, new_barriers=0, recovery_check='PASS'))
    code = ''.join(texts); ast.parse(code)
    (out / 'variants_k3obs.py').write_text(code)
    (out / 'GENERATION.json').write_text(json.dumps(dict(source=str(a.source), source_sha256=sha, pin_commit=CFG['pin_commit'],
        observer_role_src_sha256=hashlib.sha256(OBS_SRC.read_bytes()).hexdigest(), variants_sha256=hashlib.sha256(code.encode()).hexdigest(),
        variants=inventory, schema=CFG['schema'], register_pool='launch 8x120=960 per-thread registers; compute increase target 120; idle budgets 32/64/none all pool-consistent',
        clock_units='raw clock64 cycles; globaltimer ns; no MHz conversion applied', model_enabled=False), indent=2))
    print(json.dumps(dict(status='AST_ONLY_GENERATED', out=str(out), variants=[v['class_name'] for v in inventory], source_sha256=sha), indent=2))


if __name__ == '__main__':
    main()
