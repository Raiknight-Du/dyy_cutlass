#!/usr/bin/env python3
"""AST generator of the round-6 K3 kernel variant (2026-10-26): a coalesced per-partition fixed-state store.

  K3OptCoalescedStore  the compute warps read the accumulator from TMEM into registers as before (one row per lane), but
                       write it to a padded shared-memory staging tile (rows_per_cta x D fp32, row stride D + 4 to avoid bank
                       conflicts) instead of global memory; after a named barrier of the four compute warps, the tile is
                       written to the fixed-state buffer with row-contiguous 16-byte stores (128 threads = 4 rows x 32 lanes
                       per instruction: every warp instruction writes 512 contiguous bytes).  Same values, same destination:
                       bit-exact (the runner checks).  The final-state store (store_output = True, once per CTA) is unchanged.
                       Round 4 measured the pinned store at 32 sectors per request and 2564 exposed cycles per partition round.
Changes: SharedStorage gains the staging tile (+33 KiB dynamic SMEM), kernel() creates its tensor and passes it to
run_compute_role(), which passes it to store_acc(); store_acc() gets the staged path.  Everything else is the pinned code.
model_enabled=false.
"""
import argparse, ast, copy, hashlib, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; CFG = json.loads((ROOT / 'config.json').read_text())
K3_CLASS = CFG['k3_class']
STORE_ACC = '''
@cute.jit
def store_acc(self, tiled_mma, tCtAcc: cute.Tensor, tidx: cutlass.Int32, gFixedState: cute.Tensor, gOutputState: cute.Tensor, store_output: bool, stage: cutlass.Int32, sStage: cute.Tensor):
    tCtAcc_mn = utils.gemm.sm100.transform_partitioned_tensor_layout(tCtAcc)
    cAcc = cute.make_identity_tensor((self.rows_per_cta, self.D))
    if cutlass.const_expr(self.rows_per_cta == 128):
        atom_t2r = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype)
    else:
        atom_t2r = cute.make_copy_atom(tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(16)), self.acc_dtype)
    tiled_t2r = tcgen05.make_tmem_copy(atom_t2r, tCtAcc_mn[None, None, 0])
    thr_t2r = tiled_t2r.get_slice(tidx)
    tTR_tAcc = thr_t2r.partition_S(tCtAcc_mn)
    tTR_cAcc = thr_t2r.partition_D(cAcc)
    tTR_rAcc = cute.make_rmem_tensor_like(tTR_cAcc, self.acc_dtype)
    if cutlass.const_expr(store_output):
        tTR_gFixedState = thr_t2r.partition_D(gFixedState)
        tTR_gFixedState = cute.make_tensor(tTR_gFixedState.iterator.align(16), tTR_gFixedState.layout)
        tTR_gOutputState = thr_t2r.partition_D(gOutputState)
        tTR_gOutputState = cute.make_tensor(tTR_gOutputState.iterator.align(16), tTR_gOutputState.layout)
        tTR_rOutput = cute.make_rmem_tensor_like(tTR_cAcc, self.state_dtype)
        for sub in cutlass.range(cute.size(tTR_rAcc, mode=[2]), unroll_full=True):
            cute.copy(tiled_t2r, tTR_tAcc[None, 0, sub, stage], tTR_rAcc[None, 0, sub])
            cute.autovec_copy(tTR_rAcc[None, 0, sub], tTR_gFixedState[None, 0, sub])
            tTR_rOutput[None, 0, sub].store(tTR_rAcc[None, 0, sub].load().to(self.state_dtype))
            cute.autovec_copy(tTR_rOutput[None, 0, sub], tTR_gOutputState[None, 0, sub])
    else:
        tTR_sStage = thr_t2r.partition_D(sStage)
        tTR_sStage = cute.make_tensor(tTR_sStage.iterator.align(16), tTR_sStage.layout)
        for sub in cutlass.range(cute.size(tTR_rAcc, mode=[2]), unroll_full=True):
            cute.copy(tiled_t2r, tTR_tAcc[None, 0, sub, stage], tTR_rAcc[None, 0, sub])
            cute.autovec_copy(tTR_rAcc[None, 0, sub], tTR_sStage[None, 0, sub])
        cute.arch.barrier(barrier_id=3, number_of_threads=128)
        atom_s2g = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype, num_bits_per_copy=128)
        tiled_s2g = cute.make_tiled_copy_tv(atom_s2g, cute.make_layout((4, 32), stride=(32, 1)), cute.make_layout((1, 4)))
        thr_s2g = tiled_s2g.get_slice(tidx)
        tSG_s = thr_s2g.partition_S(sStage)
        tSG_s = cute.make_tensor(tSG_s.iterator.align(16), tSG_s.layout)
        tSG_g = thr_s2g.partition_D(gFixedState)
        tSG_g = cute.make_tensor(tSG_g.iterator.align(16), tSG_g.layout)
        cute.copy(tiled_s2g, tSG_s, tSG_g)
        cute.arch.barrier(barrier_id=3, number_of_threads=128)
'''
STORE_ACC_TMA = '''
@cute.jit
def store_acc(self, tiled_mma, tCtAcc: cute.Tensor, tidx: cutlass.Int32, gFixedState: cute.Tensor, gOutputState: cute.Tensor, store_output: bool, stage: cutlass.Int32, sStage: cute.Tensor, tma_atom_fs, tma_tensor_fs, row_cta_idx: cutlass.Int32, head_idx: cutlass.Int32, chunk: cutlass.Int32, use_tma: bool):
    tCtAcc_mn = utils.gemm.sm100.transform_partitioned_tensor_layout(tCtAcc)
    cAcc = cute.make_identity_tensor((self.rows_per_cta, self.D))
    if cutlass.const_expr(self.rows_per_cta == 128):
        atom_t2r = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype)
    else:
        atom_t2r = cute.make_copy_atom(tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(16)), self.acc_dtype)
    tiled_t2r = tcgen05.make_tmem_copy(atom_t2r, tCtAcc_mn[None, None, 0])
    thr_t2r = tiled_t2r.get_slice(tidx)
    tTR_tAcc = thr_t2r.partition_S(tCtAcc_mn)
    tTR_cAcc = thr_t2r.partition_D(cAcc)
    tTR_rAcc = cute.make_rmem_tensor_like(tTR_cAcc, self.acc_dtype)
    if cutlass.const_expr(store_output):
        tTR_gFixedState = thr_t2r.partition_D(gFixedState)
        tTR_gFixedState = cute.make_tensor(tTR_gFixedState.iterator.align(16), tTR_gFixedState.layout)
        tTR_gOutputState = thr_t2r.partition_D(gOutputState)
        tTR_gOutputState = cute.make_tensor(tTR_gOutputState.iterator.align(16), tTR_gOutputState.layout)
        tTR_rOutput = cute.make_rmem_tensor_like(tTR_cAcc, self.state_dtype)
        for sub in cutlass.range(cute.size(tTR_rAcc, mode=[2]), unroll_full=True):
            cute.copy(tiled_t2r, tTR_tAcc[None, 0, sub, stage], tTR_rAcc[None, 0, sub])
            cute.autovec_copy(tTR_rAcc[None, 0, sub], tTR_gFixedState[None, 0, sub])
            tTR_rOutput[None, 0, sub].store(tTR_rAcc[None, 0, sub].load().to(self.state_dtype))
            cute.autovec_copy(tTR_rOutput[None, 0, sub], tTR_gOutputState[None, 0, sub])
    elif cutlass.const_expr(not use_tma):
        tTR_gFixedState = thr_t2r.partition_D(gFixedState)
        tTR_gFixedState = cute.make_tensor(tTR_gFixedState.iterator.align(16), tTR_gFixedState.layout)
        for sub in cutlass.range(cute.size(tTR_rAcc, mode=[2]), unroll_full=True):
            cute.copy(tiled_t2r, tTR_tAcc[None, 0, sub, stage], tTR_rAcc[None, 0, sub])
            cute.autovec_copy(tTR_rAcc[None, 0, sub], tTR_gFixedState[None, 0, sub])
    else:
        cute.arch.cp_async_bulk_wait_group(0, read=True)
        cute.arch.barrier(barrier_id=3, number_of_threads=128)
        tTR_sStage = thr_t2r.partition_D(sStage)
        tTR_sStage = cute.make_tensor(tTR_sStage.iterator.align(16), tTR_sStage.layout)
        for sub in cutlass.range(cute.size(tTR_rAcc, mode=[2]), unroll_full=True):
            cute.copy(tiled_t2r, tTR_tAcc[None, 0, sub, stage], tTR_rAcc[None, 0, sub])
            cute.autovec_copy(tTR_rAcc[None, 0, sub], tTR_sStage[None, 0, sub])
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(barrier_id=3, number_of_threads=128)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == self.compute_warp_group_id:
            mT = tma_tensor_fs[None, None, head_idx, chunk]
            gT = cute.zipped_divide(mT, (self.rows_per_cta, self.D))[((None, None), (row_cta_idx, 0))]
            tSs, tSg = cpasync.tma_partition(tma_atom_fs, 0, cute.make_layout(1), cute.group_modes(sStage, 0, 2), cute.group_modes(gT, 0, 2))
            cute.copy(tma_atom_fs, tSs, tSg)
            cute.arch.cp_async_bulk_commit_group()
'''
def parse_func(text):
    f = ast.parse(text).body[0]; assert isinstance(f, ast.FunctionDef); return f
class AppendArgToSelfCalls(ast.NodeTransformer):
    def __init__(self, meth, arg): self.meth = meth; self.arg = arg; self.n = 0
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self' and node.func.attr == self.meth:
            node.args.append(ast.Name(id=self.arg, ctx=ast.Load())); self.n += 1
        return node
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument('--source', default=str(Path(CFG['host']['flashinfer_repo']) / CFG['source_relpath'])); ap.add_argument('--out', default=str(ROOT / 'generated'))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    text = Path(a.source).read_bytes(); sha = hashlib.sha256(text).hexdigest()
    if sha != CFG['source_sha256']: raise SystemExit('pinned source SHA mismatch')
    tree = ast.parse(text.decode()); cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == K3_CLASS)
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    texts = ['from flashinfer.gdn_kernels.blackwell.gated_delta_net_cp import *\n\n']; inv = []
    for mode in ('coalesced', 'tma'):
        c, det = build(methods, mode); texts.append(ast.unparse(c) + '\n\n')
        inv.append(dict(class_name=c.name, description=('per-partition fixed-state store staged through padded shared memory and written with row-contiguous 16-byte stores' if mode == 'coalesced' else 'per-partition fixed-state store staged through shared memory and written by one asynchronous TMA bulk-tensor store per partition (cp.async.bulk.tensor, commit/wait groups)'), edits=det, handshakes_touched=False, outputs_valid_expected=True, instrument='design', smem_changed=True))
    code = ''.join(texts); ast.parse(code)
    tmp = out / 'variants_k3opt.py.tmp'; tmp.write_text(code); tmp.replace(out / 'variants_k3opt.py')
    (out / 'GENERATION_K3OPT.json').write_text(json.dumps(dict(source=str(a.source), source_sha256=sha, pin_commit=CFG['pin_commit'], variants=inv, variants_sha256=hashlib.sha256(code.encode()).hexdigest(), model_enabled=False), indent=2))
    print(json.dumps(dict(status='AST_GENERATED', variants=[v['class_name'] for v in inv]), indent=1))
def build(methods, mode):
    detail = {}
    # 1. __call__: SharedStorage gains sStage
    call = copy.deepcopy(methods['__call__']); ss = next(n for n in ast.walk(call) if isinstance(n, ast.ClassDef) and n.name == 'SharedStorage')
    field = ast.parse('sStage: cute.struct.Align[cute.struct.MemRange[self.acc_dtype, self.rows_per_cta * (self.D + 4)], 128]' if mode == 'coalesced' else 'sStage: cute.struct.Align[cute.struct.MemRange[self.acc_dtype, self.rows_per_cta * self.D], 1024]').body[0]; assert isinstance(field, ast.AnnAssign)
    ss.body.append(field); detail['__call__'] = dict(shared_storage_field_added='sStage (rows_per_cta x (D + 4) fp32, 128-byte aligned)' if mode == 'coalesced' else 'sStage (rows_per_cta x D fp32, 1024-byte aligned, TMA source)')
    if mode == 'tma':
        # TMA store atom over the fixed-state buffer viewed as (D, D, heads, chunks) like the kernel's out_layout; created before the kernel launch
        i_k = next(i for i, st in enumerate(call.body) if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call) and ast.unparse(st.value.func).endswith('.launch'))
        pre = ast.parse("""tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
g_fs_view = cute.make_tensor(g_fixed_state_t.iterator.align(128), cute.make_layout((self.D, self.D, num_heads, total_cp_chunks), stride=(self.D, 1, self.D * self.D, self.D * self.D * num_heads)))
fs_smem_layout = cute.make_layout((self.rows_per_cta, self.D), stride=(self.D, 1))
tma_atom_fs, tma_tensor_fs = cpasync.make_tiled_tma_atom(tma_store_op, g_fs_view, fs_smem_layout, (self.rows_per_cta, self.D))""").body
        for j, st in enumerate(pre): call.body.insert(i_k + j, st)
        launch = call.body[i_k + len(pre)].value; kcall = launch.func.value; assert isinstance(kcall, ast.Call) and ast.unparse(kcall.func) == 'self.kernel'
        kcall.args += [ast.Name(id='tma_atom_fs', ctx=ast.Load()), ast.Name(id='tma_tensor_fs', ctx=ast.Load())]; detail['__call__']['tma_store_atom'] = 'CopyBulkTensorTileS2GOp over (D, D, heads, chunks), box (rows_per_cta, D), passed to kernel()'
    # 2. kernel: create the tensor after sN and pass it to run_compute_role
    kern = copy.deepcopy(methods['kernel']); body = kern.body
    i_sn = next(i for i, s in enumerate(body) if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name) and s.targets[0].id == 'sN')
    mk = ast.parse('sStage = storage.sStage.get_tensor(cute.make_layout((self.rows_per_cta, self.D), stride=(self.D + 4, 1)))' if mode == 'coalesced' else 'sStage = storage.sStage.get_tensor(cute.make_layout((self.rows_per_cta, self.D), stride=(self.D, 1)))').body[0]; body.insert(i_sn + 1, mk)
    extra = ['sStage'] if mode == 'coalesced' else ['sStage', 'tma_atom_fs', 'tma_tensor_fs', 'row_cta_idx', 'head_idx']
    if mode == 'tma': kern.args.args += [ast.arg(arg='tma_atom_fs', annotation=None), ast.arg(arg='tma_tensor_fs', annotation=None)]
    n_calls = 0
    for nm in extra:
        t = AppendArgToSelfCalls('run_compute_role', nm); t.visit(kern); assert t.n == 1, t.n; n_calls = t.n
    detail['kernel'] = dict(sStage_tensor_created=True, run_compute_role_calls_patched=n_calls, extra_args=extra)
    # 3. run_compute_role: new parameters, forwarded to every store_acc call (+ the chunk index for the TMA store)
    rcr = copy.deepcopy(methods['run_compute_role']); rcr.args.args += [ast.arg(arg=nm, annotation=None) for nm in extra]
    for nm in extra:
        t2 = AppendArgToSelfCalls('store_acc', nm); t2.visit(rcr); assert t2.n >= 2, t2.n
    if mode == 'tma':
        n_tma = 0
        for node in ast.walk(rcr):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'store_acc':
                sub = node.args[3]
                if isinstance(sub, ast.Subscript) and ast.unparse(sub).startswith('gFixedState['):     # fixed-state chunk store: TMA path
                    node.args += [copy.deepcopy(sub.slice.elts[2]), ast.Constant(True)]; n_tma += 1
                else:                                                                                   # other destinations (initial-state workspace): pinned direct path
                    node.args += [ast.parse('cutlass.Int32(0)', mode='eval').body, ast.Constant(False)]
        assert n_tma >= 2, n_tma; detail['store_acc_tma_calls'] = n_tma
        rcr.body.append(ast.parse('cute.arch.cp_async_bulk_wait_group(0)').body[0])   # outstanding TMA stores complete before the CTA exits
    detail['run_compute_role'] = dict(store_acc_calls_patched=t2.n, tail_wait=(mode == 'tma'))
    # 4. store_acc: staged path
    sa = parse_func(STORE_ACC if mode == 'coalesced' else STORE_ACC_TMA)
    detail['store_acc'] = dict(rewritten=True, staging=('padded SMEM tile, named barrier 3 (128 threads), 4x32 threads x 16 B row-contiguous stores' if mode == 'coalesced' else 'SMEM tile, fence.proxy.async, named barrier 3, one TMA bulk-tensor store per partition from compute warp 0, commit group; wait_group.read before the tile is rewritten'))
    for f in (call, kern, rcr, sa): ast.fix_missing_locations(f)
    return ast.ClassDef(name=('K3OptCoalescedStore' if mode == 'coalesced' else 'K3OptTmaStore'), bases=[ast.Name(id=K3_CLASS, ctx=ast.Load())], keywords=[], body=[sa, rcr, kern, call], decorator_list=[]), detail
if __name__ == '__main__': main()
