"""K4 QK-matched FP16 M64N64K128 SS UMMA probe; no target timing claim.

The first atom of EACH GEMM overwrites C, exactly as upstream QK does.  The
resident variant only publishes done after the last GEMM; the wait variant
performs producer/consumer completion handoff after every GEMM.  Their delta
is not pure MMA RAW latency.  GMEM/TMA initialisation and final TMEM readback
are outside the CLOCK envelope, but are fixed overhead inside EVENT timing.
"""
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05, cpasync, OperandMajorMode


def semantic(layout):
    saved = layout
    if isinstance(layout, cute.ComposedLayout):
        layout = layout.outer
    shape, stride = layout.shape, layout.stride
    out = cute.make_layout(
        ((shape[0][0], shape[1]), (shape[0][1], shape[2]), *shape[3:]),
        stride=((stride[0][0], stride[1]), (stride[0][1], stride[2]), *stride[3:]),
    )
    if isinstance(saved, cute.ComposedLayout):
        out = cute.make_composed_layout(saved.inner, saved.offset, out)
    return out


def semantic_tensor(tensor):
    return cute.make_tensor(tensor.iterator, semantic(tensor.layout))


class QkResident:
    def __init__(self, instrumented=True, per_round_wait=False):
        self.instrumented = instrumented
        self.per_round_wait = per_round_wait
        self.cta_group = tcgen05.CtaGroup.ONE
        self.alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=160)

    @cute.jit
    def __call__(self, gA: cute.Tensor, gB: cute.Tensor, gO: cute.Tensor,
                 trace: cute.Tensor, iterations: cutlass.Int32, stream):
        mma = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            cutlass.Float16, cutlass.Float32, (64, 64, 16), self.cta_group,
            tcgen05.OperandSource.SMEM, OperandMajorMode.K, OperandMajorMode.K))
        a_layout = sm100_utils.make_smem_layout_a(mma, (64, 64, 128), cutlass.Float16, 1)
        b_layout = sm100_utils.make_smem_layout_b(mma, (64, 64, 128), cutlass.Float16, 1)
        atom_a, tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(self.cta_group), gA,
            cute.slice_(semantic(a_layout), (None, None, 0)), (64, 128))
        atom_b, tensor_b = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(self.cta_group), gB,
            cute.slice_(semantic(b_layout), (None, None, 0)), (64, 128))

        @cute.struct
        class Shared:
            input_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            done_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding: cutlass.Int32
            sA: cute.struct.Align[cute.struct.MemRange[cutlass.Float16, cute.cosize(a_layout)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[cutlass.Float16, cute.cosize(b_layout)], 1024]

        self.shared = Shared
        self.kernel(mma, atom_a, tensor_a, a_layout, atom_b, tensor_b,
                    b_layout, gO, trace, iterations).launch(
            grid=(1, 1, 1), block=(192, 1, 1), smem=Shared.size_in_bytes(),
            stream=stream, min_blocks_per_mp=1)

    @cute.jit
    def make_c(self, ptr, mma):
        # Same M64 logical TMEM fragment and load layout as K4 CG0 epilogue.
        shape = mma.partition_shape_C((64, 64))
        fake = mma.make_fragment_C(shape)
        # M64 C has a special atom hierarchy. Preserve it, then append stage
        # exactly as upstream CG0 does (do not feed the stage into fragment C).
        staged = cute.flat_product(fake.layout, cute.make_layout((1,), stride=(1,)))
        return cute.make_tensor(ptr, staged)

    @cute.jit
    def output(self, ptr, mma, tid, gO):
        c = self.make_c(ptr, mma)
        cm = semantic_tensor(c)
        cp = tcgen05.make_tmem_copy(cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), cutlass.Float32),
            c[(None, None), 0, 0, 0])
        th = cp.get_slice(tid)
        src = th.partition_S(cm)
        # Follow original CG0's identity-coordinate register view instead of
        # deriving the register hierarchy from a flattened GMEM partition.
        coords = th.partition_D(cute.make_identity_tensor((64, 64)))
        reg = cute.make_rmem_tensor_like(coords, cutlass.Float32)
        for sub in cutlass.range(cute.size(reg, mode=[2]), unroll_full=True):
            cute.copy(cp, src[None, 0, sub, 0], reg[None, 0, sub])
            for value in cutlass.range(32, unroll_full=True):
                row, col = coords[value, 0, sub]
                gO[row, col] = reg[value, 0, sub]
        cute.arch.fence_view_async_tmem_load()

    @cute.kernel
    def kernel(self, mma, atom_a, tensor_a, a_layout, atom_b, tensor_b,
               b_layout, gO, trace, iterations: cutlass.Int32):
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        allocator = utils.SmemAllocator()
        st = allocator.allocate(self.shared)
        sA = st.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        sB = st.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
        cg1 = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        cg128 = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
        ip, ic = pipeline.PipelineTmaUmma.create(
            num_stages=1, producer_group=cg1, consumer_group=cg1,
            tx_count=2 * 64 * 128 * 2, barrier_storage=st.input_mbar.data_ptr(),
            defer_sync=True).make_participants()
        dp, dc = pipeline.PipelineUmmaAsync.create(
            num_stages=1, producer_group=cg1, consumer_group=cg128,
            barrier_storage=st.done_mbar.data_ptr(), defer_sync=True).make_participants()
        pipeline.pipeline_init_arrive(cluster_shape_mn=cute.make_layout((1, 1)), is_relaxed=True)
        pipeline.pipeline_init_wait(cluster_shape_mn=cute.make_layout((1, 1)))
        tm = utils.TmemAllocator(st.tmem_holding.ptr,
            barrier_for_retrieve=self.alloc_barrier, allocator_warp_id=0)
        tm.allocate(512)
        if warp <= 4:
            tm.wait_for_alloc()
        sc, ec, sg, eg = (cutlass.Int64(0),) * 4
        if warp < 4:
            cute.arch.sync_threads()
            if cutlass.const_expr(self.instrumented):
                sc, sg = cute.arch.clock64(), cute.arch.globaltimer()
            if cutlass.const_expr(self.per_round_wait):
                for it in cutlass.range(iterations, unroll=1):
                    handle = dc.wait_and_advance()
                    handle.release()
            else:
                handle = dc.wait_and_advance()
                handle.release()
            if cutlass.const_expr(self.instrumented):
                ec, eg = cute.arch.clock64(), cute.arch.globaltimer()
        elif warp == 4:
            ptr = tm.retrieve_ptr(cutlass.Float32)
            c = self.make_c(ptr, mma)
            aa, bb = mma.make_fragment_A(sA), mma.make_fragment_B(sB)
            ih = ic.wait_and_advance()
            if cutlass.const_expr(not self.per_round_wait):
                done = dp.acquire_and_advance()
            cute.arch.sync_threads()
            if cutlass.const_expr(self.instrumented):
                sc, sg = cute.arch.clock64(), cute.arch.globaltimer()
            for it in cutlass.range(iterations, unroll=1):
                if cutlass.const_expr(self.per_round_wait):
                    round_done = dp.acquire_and_advance()
                for kk in cutlass.range(cute.size(aa, mode=[2]), unroll_full=True):
                    # K4 QK also resets accumulator at kphase=0 of EVERY block.
                    mma.set(tcgen05.Field.ACCUMULATE, kk != 0)
                    cute.gemm(mma, c[None, None, None, 0], aa[None, None, kk, ih.index],
                              bb[None, None, kk, ih.index], c[None, None, None, 0])
                if cutlass.const_expr(self.per_round_wait):
                    round_done.commit()
            if cutlass.const_expr(not self.per_round_wait):
                done.commit()
            ih.release()
            if cutlass.const_expr(self.instrumented):
                ec, eg = cute.arch.clock64(), cute.arch.globaltimer()
        elif warp == 5:
            handle = ip.acquire_and_advance()
            ts_a, tg_a = cpasync.tma_partition(
                atom_a, 0, cute.make_layout(1),
                cute.group_modes(semantic_tensor(sA)[None, None, handle.index], 0, 2),
                cute.group_modes(tensor_a, 0, 2))
            cute.copy(atom_a, tg_a, ts_a, tma_bar_ptr=handle.barrier)
            ts_b, tg_b = cpasync.tma_partition(
                atom_b, 0, cute.make_layout(1),
                cute.group_modes(semantic_tensor(sB)[None, None, handle.index], 0, 2),
                cute.group_modes(tensor_b, 0, 2))
            cute.copy(atom_b, tg_b, ts_b, tma_bar_ptr=handle.barrier)
            cute.arch.sync_threads()
        cute.arch.sync_threads()
        if warp < 4:
            self.output(tm.retrieve_ptr(cutlass.Float32), mma, tid, gO)
        cute.arch.sync_threads()
        if warp <= 4:
            if cutlass.const_expr(self.instrumented):
                if cute.arch.lane_idx() == 0:
                    trace[warp, 0] = sc
                    trace[warp, 1] = ec
                    trace[warp, 2] = sg
                    trace[warp, 3] = eg
                    trace[warp, 4] = cutlass.Int64(cute.arch.smid())
        if warp < 4:
            ptr = tm.retrieve_ptr(cutlass.Float32)
            tm.relinquish_alloc_permit()
            tm.free(ptr)
