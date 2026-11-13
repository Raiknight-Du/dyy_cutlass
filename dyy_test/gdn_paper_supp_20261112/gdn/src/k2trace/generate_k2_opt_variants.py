#!/usr/bin/env python3
"""AST generator of the round-6 K2 restructuring variants (2026-10-26): kernel changes that shorten the two recurrence loops
without changing the arithmetic (expected bit-exact; the runner verifies outputs, state, M and N against B0).

  K2OptScaleNEarly   compute group 1 scales N (N *= block coefficient) right after it has published the N operand, i.e. while
                     the Y gemm runs, instead of after the Y processing.  N is not read by the Y gemm (which reads the fp16
                     operand), and the scaling precedes the N update in both orders, so the values are identical; the 391
                     cycles of the scaling leave the N loop's critical path.
  K2OptFusedM        compute group 0 fuses the scaling of M with the conversion of M to the next block's fp16 operand in one
                     TMEM pass at the end of the iteration (read M once, write c*M back and fp16(c*M) to m_in), and publishes
                     m_in for block b+1 there.  Same fp32 product, same fp16 rounding as scale-then-convert; the Z gemm of
                     block b+1 no longer waits for X(b+1) to be materialised first, and one TMEM read pass per block is saved.
  K2OptProcYPre      compute group 1 computes V * alpha for the whole tile before waiting for the Y accumulator and keeps it in
                     registers, so only the TMEM read, the FMA and the store remain after the wait (register pressure: audited).
  K2OptScaleNEarlyFusedM, K2OptAll: combinations.
Every mbarrier protocol is kept; only the order of compute-warp work inside an iteration changes (and, for FusedM, the point
at which the m_in producer commits).  model_enabled=false.
"""
import argparse, ast, copy, hashlib, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; CFG = json.loads((ROOT / 'config.json').read_text())
K2_CLASS = 'CPDeltaRuleMNPrecomputeUtcmma1Sm100'
FUSED_HELPER = '''
@cute.jit
def _scale_and_convert_tmem_matrix(self, tmem_ptr: cutlass.Int64, src_offset: int, dst_offset: int, tiled_mma_update, tiled_mma_z, tidx: cutlass.Int32, scale: cutlass.Float32):
    src_shape = tiled_mma_update.partition_shape_C((self.D, self.D))
    tCtSrc_fake = tiled_mma_update.make_fragment_C(cute.append(src_shape, 1))
    tCtSrc = cute.make_tensor(tmem_ptr + src_offset, tCtSrc_fake.layout)
    tCtSrc_mn = self.transform_partitioned_tensor_layout(tCtSrc)
    cSrc = cute.make_identity_tensor((self.D, self.D))
    src_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype)
    src_copy = tcgen05.make_tmem_copy(src_atom, tCtSrc[(None, None), 0, 0, 0])
    src_thr = src_copy.get_slice(tidx)
    tTR_tSrc = src_thr.partition_S(tCtSrc_mn)
    tTR_cSrc = src_thr.partition_D(cSrc)
    rSrc = cute.make_rmem_tensor_like(tTR_cSrc, self.acc_dtype)
    back_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype)
    back_copy = tcgen05.make_tmem_copy(back_atom, tCtSrc[(None, None), 0, 0, 0])
    back_thr = back_copy.get_slice(tidx)
    tRT_tSrc = back_thr.partition_D(tCtSrc_mn)
    dst_shape = tiled_mma_z.partition_shape_A((self.D, self.D))
    tCtDst_fake = tiled_mma_z.make_fragment_A(cute.append(dst_shape, 1))
    tCtDst = cute.make_tensor(cute.recast_ptr(tmem_ptr + dst_offset, dtype=self.dtype), tCtDst_fake.layout)
    tCtDst_mn = self.transform_partitioned_tensor_layout(tCtDst)
    cDst = cute.make_identity_tensor((self.D, self.D))
    dst_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), self.dtype)
    dst_copy = tcgen05.make_tmem_copy(dst_atom, tCtDst_mn[None, None, 0])
    dst_thr = dst_copy.get_slice(tidx)
    tRT_cDst = dst_thr.partition_S(cDst)
    tRT_tDst = dst_thr.partition_D(tCtDst_mn)
    rDst = cute.make_rmem_tensor_like(tRT_cDst, self.dtype)
    for iter_m in cutlass.range(cute.size(rSrc, mode=[1]), unroll_full=True):
        for iter_n in cutlass.range(cute.size(rSrc, mode=[2]), unroll_full=True):
            cute.copy(src_copy, tTR_tSrc[None, iter_m, iter_n, 0], rSrc[None, iter_m, iter_n])
            for i in cutlass.range(cute.size(rSrc, mode=[0]), vectorize=True):
                rSrc[i, iter_m, iter_n] = rSrc[i, iter_m, iter_n] * scale
            cute.copy(back_copy, rSrc[None, iter_m, iter_n], tRT_tSrc[None, iter_m, iter_n, 0])
            rDst[None, iter_m, iter_n].store(rSrc[None, iter_m, iter_n].load().to(self.dtype))
            cute.copy(dst_copy, rDst[None, iter_m, iter_n], tRT_tDst[None, iter_m, iter_n, 0])
    cute.arch.fence_view_async_tmem_store()
'''
CG0_FUSED = '''
@cute.jit
def run_compute_group_0(self, tidx: cutlass.Int32, tmem_ptr: cutlass.Int64, tiled_mma_x, tiled_mma_z, tiled_mma_update, sX: cute.Tensor, sAlpha: cute.Tensor, alpha_consumer, math_pipelines: tuple, num_blocks: cutlass.Int32, gM: cute.Tensor):
    (m_init_producer, x_acc_consumer, x_ready_producer, m_in_producer, z_acc_consumer, z_ready_producer, m_acc_consumer, done_producer) = math_pipelines
    self._initialize_tmem_matrix(tmem_ptr, self.tmem_m_offset, tiled_mma_update, tidx, True)
    m_init_producer.acquire_and_advance().commit()
    for blk in cutlass.range(num_blocks, unroll=1):
        alpha_handle = alpha_consumer.wait_and_advance()
        x_handle = x_acc_consumer.wait_and_advance()
        x_ready_handle = x_ready_producer.acquire_and_advance()
        self._materialize_x(tmem_ptr, tiled_mma_x, tidx, sX, x_ready_handle.index)
        x_handle.release()
        x_ready_handle.commit()
        if blk > 0:
            z_handle = z_acc_consumer.wait_and_advance()
            self._convert_scratch_to_update_input(tmem_ptr, self.tmem_m_inp_offset, tiled_mma_x, tiled_mma_update, tidx)
            z_handle.release()
            z_ready_producer.acquire_and_advance().commit()
        block_coeff = cutlass.Float32(sAlpha[self.BLK - 1, AlphaProcessor.CUMPROD, alpha_handle.index])
        m_handle = m_acc_consumer.wait_and_advance()
        if blk + 1 < num_blocks:
            m_in_handle = m_in_producer.acquire_and_advance()
            self._scale_and_convert_tmem_matrix(tmem_ptr, self.tmem_m_offset, self.tmem_m_inp_offset, tiled_mma_update, tiled_mma_z, tidx, block_coeff)
            m_in_handle.commit()
        else:
            self._scale_tmem_matrix(tmem_ptr, self.tmem_m_offset, tiled_mma_update, tidx, block_coeff)
        m_handle.release()
        alpha_handle.release()
    self._store_tmem_matrix(tmem_ptr, self.tmem_m_offset, tiled_mma_update, tidx, gM)
    done_producer.acquire_and_advance().commit()
'''
PROCY_SPLIT = '''
@cute.jit
def _process_y_split(self, tmem_ptr: cutlass.Int64, tiled_mma_x, tidx: cutlass.Int32, sV: cute.Tensor, sAlpha: cute.Tensor, v_stage: cutlass.Int32, alpha_stage: cutlass.Int32, block_coeff: cutlass.Float32, tiled_mma_update, y_acc_consumer):
    acc_shape = tiled_mma_x.partition_shape_C((self.D, self.BLK))
    tCtY_fake = tiled_mma_x.make_fragment_C(cute.append(acc_shape, 1))
    tCtY = cute.make_tensor(tmem_ptr + self.tmem_x_y_acc_offset, tCtY_fake.layout)
    tCtY_mn = self.transform_partitioned_tensor_layout(tCtY)
    cY = cute.make_identity_tensor((self.D, self.BLK))
    atom_t2r = cute.make_copy_atom(tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype)
    tiled_t2r = tcgen05.make_tmem_copy(atom_t2r, tCtY[(None, None), 0, 0, 0])
    thr_t2r = tiled_t2r.get_slice(tidx)
    tTR_tY = thr_t2r.partition_S(tCtY_mn)
    tTR_cY = thr_t2r.partition_D(cY)
    rY = cute.make_rmem_tensor_like(tTR_cY, self.acc_dtype)
    rVA = cute.make_rmem_tensor_like(tTR_cY, self.acc_dtype)
    sAlpha_blk = sAlpha[None, None, alpha_stage]
    dst_shape = tiled_mma_update.partition_shape_A((self.D, self.BLK))
    tCtDst_fake = tiled_mma_update.make_fragment_A(cute.append(dst_shape, 1))
    tCtDst = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_n_inp_offset, dtype=self.dtype), tCtDst_fake.layout)
    tCtDst_mn = self.transform_partitioned_tensor_layout(tCtDst)
    cDst = cute.make_identity_tensor((self.D, self.BLK))
    dst_atom = cute.make_copy_atom(tcgen05.copy.St16x128bOp(tcgen05.copy.Repetition(8)), self.dtype)
    dst_copy = tcgen05.make_tmem_copy(dst_atom, tCtDst_mn[None, None, 0])
    dst_thr = dst_copy.get_slice(tidx)
    tRT_cDst = dst_thr.partition_S(cDst)
    tRT_tDst = dst_thr.partition_D(tCtDst_mn)
    rDst = cute.make_rmem_tensor_like(tRT_cDst, self.dtype)
    rV = cute.make_rmem_tensor_like(tRT_cDst, self.dtype)
    atom_v_s2r = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True), self.dtype)
    tiled_v_s2r = cute.make_tiled_copy_S(atom_v_s2r, dst_copy)
    thr_v_s2r = tiled_v_s2r.get_slice(tidx)
    tVsV = thr_v_s2r.partition_S(sV)
    tVrV = tiled_v_s2r.retile(rV)
    for iter_m in cutlass.range(cute.size(rY, mode=[1]), unroll_full=True):
        for iter_n in cutlass.range(cute.size(rY, mode=[2]), unroll_full=True):
            cute.copy(tiled_v_s2r, tVsV[None, iter_m, iter_n, v_stage], tVrV[None, iter_m, iter_n])
            for i in cutlass.range(cute.size(rY, mode=[0]), unroll_full=True):
                _, token = tTR_cY[i, iter_m, iter_n]
                rVA[i, iter_m, iter_n] = cutlass.Float32(rV[i, iter_m, iter_n]) * sAlpha_blk[token, AlphaProcessor.CUMPROD_NEG_END_RCP]
    y_handle = y_acc_consumer.wait_and_advance()
    for iter_m in cutlass.range(cute.size(rY, mode=[1]), unroll_full=True):
        for iter_n in cutlass.range(cute.size(rY, mode=[2]), unroll_full=True):
            cute.copy(tiled_t2r, tTR_tY[None, iter_m, iter_n, 0], rY[None, iter_m, iter_n])
            for i in cutlass.range(cute.size(rY, mode=[0]), unroll_full=True):
                rY[i, iter_m, iter_n] = block_coeff * rY[i, iter_m, iter_n] + rVA[i, iter_m, iter_n]
            rDst[None, iter_m, iter_n].store(rY[None, iter_m, iter_n].load().to(self.dtype))
            cute.copy(dst_copy, rDst[None, iter_m, iter_n], tRT_tDst[None, iter_m, iter_n, 0])
    cute.arch.fence_view_async_tmem_store()
    return y_handle
'''
def parse_func(text):
    f = ast.parse(text).body[0]; assert isinstance(f, ast.FunctionDef); return f
def self_call_name(stmt):
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Attribute) and isinstance(stmt.value.func.value, ast.Name) and stmt.value.func.value.id == 'self':
        return stmt.value.func.attr
    return None
def loop_body(f):
    fors = [n for n in f.body if isinstance(n, ast.For)]; assert len(fors) == 1; return fors[0].body
def scale_n_early(f):
    """run_compute_group_1: move block_coeff and the N scaling before the y_acc wait (right after the n_in commit)."""
    body = loop_body(f)
    i_coeff = next(i for i, s in enumerate(body) if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name) and s.targets[0].id == 'block_coeff')
    i_scale = next(i for i, s in enumerate(body) if self_call_name(s) == '_scale_tmem_matrix')
    i_anchor = next(i for i, s in enumerate(body) if isinstance(s, ast.Expr) and ast.unparse(s).startswith('n_in_producer.acquire_and_advance().commit()'))
    i_ywait = next(i for i, s in enumerate(body) if isinstance(s, ast.Assign) and ast.unparse(s).startswith('y_handle = y_acc_consumer.wait_and_advance()'))
    assert i_anchor < i_ywait < i_coeff < i_scale
    coeff, scale = body[i_coeff], body[i_scale]
    for i in sorted((i_coeff, i_scale), reverse=True): body.pop(i)
    body.insert(i_anchor + 1, scale); body.insert(i_anchor + 1, coeff)
    return dict(moved=['block_coeff', '_scale_tmem_matrix'], before='y_acc_consumer.wait_and_advance()')
def procy_pre(f):
    """run_compute_group_1: replace `y_handle = y_acc_consumer.wait_and_advance()` + `self._process_y(...)` by the split helper."""
    body = loop_body(f)
    i_ywait = next(i for i, s in enumerate(body) if isinstance(s, ast.Assign) and ast.unparse(s).startswith('y_handle = y_acc_consumer.wait_and_advance()'))
    i_py = next(i for i, s in enumerate(body) if self_call_name(s) == '_process_y')
    call = body[i_py].value; call.func.attr = '_process_y_split'; call.args.append(ast.Name(id='y_acc_consumer', ctx=ast.Load()))
    new = ast.Assign(targets=[ast.Name(id='y_handle', ctx=ast.Store())], value=call); ast.copy_location(new, body[i_py])
    # keep everything between the old wait and the process call (block_coeff) in place; the wait moves into the helper
    body.pop(i_py); body.insert(i_py, new); body.pop(i_ywait)
    return dict(replaced=['y_acc wait + _process_y -> _process_y_split (wait inside, after V*alpha)'])
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument('--source', default=str(Path(CFG['host']['flashinfer_repo']) / CFG['source_relpath'])); ap.add_argument('--out', default=str(ROOT / 'generated'))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    text = Path(a.source).read_bytes(); sha = hashlib.sha256(text).hexdigest()
    if sha != CFG['source_sha256']: raise SystemExit('pinned source SHA mismatch')
    tree = ast.parse(text.decode()); cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == K2_CLASS)
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    variants = []
    def cg1(edits):
        f = copy.deepcopy(methods['run_compute_group_1']); d = {}
        for e in edits: d.update(e(f))
        ast.fix_missing_locations(f); return f, d
    # ScaleNEarly
    f1, d1 = cg1([scale_n_early]); variants.append(('K2OptScaleNEarly', [f1], {'run_compute_group_1': d1}, 'N scaled before the y_acc wait (off the N loop critical path)'))
    # FusedM
    variants.append(('K2OptFusedM', [parse_func(FUSED_HELPER), parse_func(CG0_FUSED)], {'run_compute_group_0': dict(rewritten=True, helper='_scale_and_convert_tmem_matrix')}, 'M scaled and converted in one TMEM pass at the end of the iteration; m_in for block b+1 published there'))
    # ProcYPre
    f3, d3 = cg1([procy_pre]); variants.append(('K2OptProcYPre', [parse_func(PROCY_SPLIT), f3], {'run_compute_group_1': d3}, 'V*alpha computed before the y_acc wait'))
    # combinations
    f4, d4 = cg1([scale_n_early]); variants.append(('K2OptScaleNEarlyFusedM', [parse_func(FUSED_HELPER), parse_func(CG0_FUSED), f4], {'run_compute_group_0': dict(rewritten=True), 'run_compute_group_1': d4}, 'ScaleNEarly + FusedM'))
    f5, d5 = cg1([scale_n_early, procy_pre]); variants.append(('K2OptAll', [parse_func(FUSED_HELPER), parse_func(CG0_FUSED), parse_func(PROCY_SPLIT), f5], {'run_compute_group_0': dict(rewritten=True), 'run_compute_group_1': d5}, 'ScaleNEarly + FusedM + ProcYPre'))
    # pairs with ProcYPre (K2OptAll hangs at the first invocation: these localise the conflicting pair)
    f6, d6 = cg1([scale_n_early, procy_pre]); variants.append(('K2OptScaleNEarlyProcYPre', [parse_func(PROCY_SPLIT), f6], {'run_compute_group_1': d6}, 'ScaleNEarly + ProcYPre'))
    f7, d7 = cg1([procy_pre]); variants.append(('K2OptFusedMProcYPre', [parse_func(FUSED_HELPER), parse_func(CG0_FUSED), parse_func(PROCY_SPLIT), f7], {'run_compute_group_0': dict(rewritten=True), 'run_compute_group_1': d7}, 'FusedM + ProcYPre'))
    texts = ['from flashinfer.gdn_kernels.blackwell.gated_delta_net_cp import *\n\n']; inv = []
    for name, body, detail, desc in variants:
        c = ast.ClassDef(name=name, bases=[ast.Name(id=K2_CLASS, ctx=ast.Load())], keywords=[], body=body, decorator_list=[])
        texts.append(ast.unparse(c) + '\n\n'); inv.append(dict(class_name=name, description=desc, edits=detail, handshakes_touched=(name != 'K2OptScaleNEarly' and 'FusedM' in name or name == 'K2OptAll'), outputs_valid_expected=True, instrument='design'))
    code = ''.join(texts); ast.parse(code); tmp = out / 'variants_k2opt.py.tmp'; tmp.write_text(code); tmp.replace(out / 'variants_k2opt.py')
    (out / 'GENERATION_K2OPT.json').write_text(json.dumps(dict(source=str(a.source), source_sha256=sha, pin_commit=CFG['pin_commit'], variants=inv, variants_sha256=hashlib.sha256(code.encode()).hexdigest(), model_enabled=False), indent=2))
    print(json.dumps(dict(status='AST_GENERATED', variants=[v['class_name'] for v in inv]), indent=1))
if __name__ == '__main__': main()
