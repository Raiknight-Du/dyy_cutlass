"""Generated source-pinned K3 causal-ablation subclasses; do not edit by hand."""
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from typing import Type
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from flashinfer.gdn_kernels.blackwell.gated_delta_net_cp import CPDeltaRuleFixupUtcmmaSm100 as _CPDeltaRuleFixupUtcmmaSm100, CPDeltaRuleMNPrecomputeUtcmma1Sm100, chunks_for_len, varlen_chunk_idx, varlen_chunk_valid_len

class K3FinalRUNTIME(_CPDeltaRuleFixupUtcmmaSm100):
    _k3_default_m_stage = 2
    _k3_default_n_stage = 1

    def __init__(self, rows_per_cta: int=64, needs_initial_state: bool=False, store_final_state: bool=False, initial_state_dtype: Type[cutlass.Numeric]=cutlass.Float32, state_dtype: Type[cutlass.Numeric]=cutlass.Float32, cu_seqlens_dtype: Type[cutlass.Numeric]=cutlass.Int32, use_state_indices: bool=False, store_initial_state: bool=False, m_stage=None, n_stage=None):
        self.trace_level = 'event'
        self.causal_variant = 'RUNTIME'
        if m_stage is None:
            m_stage = self._k3_default_m_stage
        if n_stage is None:
            n_stage = self._k3_default_n_stage
        if m_stage not in (1, 2):
            raise ValueError(f'K3 m_stage must be 1 or 2, got {m_stage}')
        if n_stage not in (1, 2):
            raise ValueError(f'K3 n_stage must be 1 or 2, got {n_stage}')
        self.needs_initial_state = needs_initial_state
        self.store_final_state = store_final_state
        self.initial_state_dtype = initial_state_dtype
        self.state_dtype = state_dtype
        self.cu_seqlens_dtype = cu_seqlens_dtype
        self.use_state_indices = use_state_indices
        self.store_initial_state = store_initial_state
        self.acc_dtype = cutlass.Float32
        self.mma_dtype = cutlass.TFloat32
        self.D = 128
        if rows_per_cta not in (64, 128):
            raise ValueError(f'rows_per_cta must be 64 or 128, got {rows_per_cta}')
        self.rows_per_cta = rows_per_cta
        self.row_ctas = self.D // rows_per_cta
        self.m_stage = m_stage
        self.n_stage = n_stage
        self.threads_per_cta = 256
        self.compute_warp_group_id = 0
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.num_regs_compute = 120 if rows_per_cta == 64 else 256
        self.num_regs_other = 32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tmem_acc_offset = 0
        self.tmem_opd_offset = 128
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=160)
        self.manual_cache_key('rows_per_cta', 'row_ctas', 'needs_initial_state', 'store_final_state', 'initial_state_dtype', 'state_dtype', 'cu_seqlens_dtype', 'use_state_indices', 'store_initial_state', 'm_stage', 'n_stage', 'num_regs_compute', 'num_regs_other', 'causal_variant', 'trace_level')

    @cute.jit
    def __call__(self, g_transfer_t: cute.Tensor, g_local_state_t: cute.Tensor, g_initial_state_t: cute.Tensor, g_initial_state_workspace_t: cute.Tensor, g_fixed_state_t: cute.Tensor, g_output_state_t: cute.Tensor, g_state_indices_t: cute.Tensor, g_cu_seqlens: cute.Tensor, chunk_len: cutlass.Int32, total_cp_chunks: cutlass.Int32, num_seqs: cutlass.Int32, num_heads: cutlass.Int32, final_mode: cutlass.Int32, stream):
        tiled_mma = cute.make_tiled_mma(tcgen05.MmaTF32Op((self.rows_per_cta, 64, 8), self.cta_group, tcgen05.OperandSource.TMEM, OperandMajorMode.K, OperandMajorMode.MN))
        m_layout = sm100_utils.make_smem_layout_b(tiled_mma, (self.rows_per_cta, self.D, self.D), self.acc_dtype, self.m_stage)
        n_layout = sm100_utils.make_smem_layout_a(tiled_mma, (self.rows_per_cta, self.D, self.D), self.acc_dtype, self.n_stage, is_k_major=True)
        m_semantic_layout = CPDeltaRuleMNPrecomputeUtcmma1Sm100.transform_partitioned_layout(m_layout)
        n_semantic_layout = CPDeltaRuleMNPrecomputeUtcmma1Sm100.transform_partitioned_layout(n_layout)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        m_tma_layout = cute.slice_(m_semantic_layout, (None, None, 0))
        n_tma_layout = cute.slice_(n_semantic_layout, (None, None, 0))
        tma_m = cpasync.make_tiled_tma_atom(tma_op, g_transfer_t, m_tma_layout, (self.D, self.D))
        tma_n = cpasync.make_tiled_tma_atom(tma_op, g_local_state_t, n_tma_layout, (self.rows_per_cta, self.D))
        (tma_atom_m, tma_tensor_m) = tma_m
        (tma_atom_n, tma_tensor_n) = tma_n
        self.tma_m_bytes = cute.size(m_tma_layout) * 4
        self.tma_n_bytes = cute.size(n_tma_layout) * 4

        @cute.struct
        class SharedStorage:
            m_mbar: cute.struct.MemRange[cutlass.Int64, self.m_stage * 2]
            n_mbar: cute.struct.MemRange[cutlass.Int64, self.n_stage * 2]
            mma_ready_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            mma_done_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: cutlass.Int32
            sM: cute.struct.Align[cute.struct.MemRange[self.acc_dtype, cute.cosize(m_layout)], 1024]
            sN: cute.struct.Align[cute.struct.MemRange[self.acc_dtype, cute.cosize(n_layout)], 1024]
        self.shared_storage = SharedStorage
        self.kernel(tiled_mma, tma_atom_m, tma_tensor_m, tma_atom_n, tma_tensor_n, g_initial_state_t, g_initial_state_workspace_t, g_fixed_state_t, g_output_state_t, g_state_indices_t, g_cu_seqlens, m_layout, n_layout, chunk_len, total_cp_chunks, num_seqs, num_heads, final_mode).launch(grid=(num_seqs * num_heads * self.row_ctas, 1, 1), block=(self.threads_per_cta, 1, 1), max_number_threads=(self.threads_per_cta, 1, 1), smem=self.shared_storage.size_in_bytes(), stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, tiled_mma, tma_atom_m, tma_tensor_m, tma_atom_n, tma_tensor_n, g_initial_state_t, g_initial_state_workspace_t, g_fixed_state_t, g_output_state_t, g_state_indices_t, g_cu_seqlens, m_layout, n_layout, chunk_len, total_cp_chunks, num_seqs, num_heads, final_mode: cutlass.Int32):
        (tidx, _, _) = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        (bx, _, _) = cute.arch.block_idx()
        row_cta_idx = bx % self.row_ctas
        head_seq_idx = bx // self.row_ctas
        head_idx = head_seq_idx % num_heads
        seq_idx = head_seq_idx // num_heads
        seq_start = cutlass.Int32(g_cu_seqlens[seq_idx])
        seq_end = cutlass.Int32(g_cu_seqlens[seq_idx + 1])
        num_chunks = chunks_for_len(seq_end - seq_start, chunk_len)
        chunk_start = varlen_chunk_idx(seq_idx, seq_start, 0, chunk_len)
        start = cutlass.Int32(0) if cutlass.const_expr(self.needs_initial_state) else cutlass.Int32(1)
        num_iters = num_chunks - start
        allocator = utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)
        sM = storage.sM.get_tensor(m_layout.outer, swizzle=m_layout.inner)
        sN = storage.sN.get_tensor(n_layout.outer, swizzle=n_layout.inner)
        sM_tma = CPDeltaRuleMNPrecomputeUtcmma1Sm100.transform_partitioned_tensor_layout(sM)
        sN_tma = CPDeltaRuleMNPrecomputeUtcmma1Sm100.transform_partitioned_tensor_layout(sN)
        cg_tma = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        cg_mma = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        cg_compute = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
        cg_compute_tma = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
        (m_producer, m_consumer) = pipeline.PipelineTmaUmma.create(num_stages=self.m_stage, producer_group=cg_tma, consumer_group=cg_mma, tx_count=self.tma_m_bytes, barrier_storage=storage.m_mbar.data_ptr(), defer_sync=True).make_participants()
        (n_producer, n_consumer) = pipeline.PipelineTmaAsync.create(num_stages=self.n_stage, producer_group=cg_tma, consumer_group=cg_compute_tma, tx_count=self.tma_n_bytes, barrier_storage=storage.n_mbar.data_ptr(), defer_sync=True, enable_multicast_signaling=False).make_participants()
        (mma_ready_producer, mma_ready_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_compute, consumer_group=cg_mma, barrier_storage=storage.mma_ready_mbar.data_ptr(), defer_sync=True).make_participants()
        (mma_done_producer, mma_done_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=cg_mma, consumer_group=cg_compute, barrier_storage=storage.mma_done_mbar.data_ptr(), defer_sync=True).make_participants()
        pipeline.pipeline_init_arrive(cluster_shape_mn=cute.make_layout((1, 1)), is_relaxed=True)
        pipeline.pipeline_init_wait(cluster_shape_mn=cute.make_layout((1, 1)))
        out_layout = cute.make_layout((self.D, self.D, num_heads, total_cp_chunks), stride=(self.D, 1, self.D * self.D, self.D * self.D * num_heads))
        gFixedState = cute.make_tensor(g_fixed_state_t.iterator.align(128), out_layout)
        gFixedStateCta = cute.local_tile(gFixedState[None, None, head_idx, None], (self.rows_per_cta, self.D, total_cp_chunks), (row_cta_idx, 0, 0))
        state_layout = cute.make_layout((self.D, self.D, num_heads, num_seqs), stride=(self.D, 1, self.D * self.D, self.D * self.D * num_heads))
        state_idx = seq_idx
        if cutlass.const_expr(self.use_state_indices):
            state_idx = cutlass.Int32(g_state_indices_t[seq_idx])
        if cutlass.const_expr(self.needs_initial_state):
            if cutlass.const_expr(self.use_state_indices):
                initial_layout = cute.make_layout((self.D, self.D, num_heads, g_initial_state_t.shape[0]), stride=(g_initial_state_t.stride[2], g_initial_state_t.stride[3], g_initial_state_t.stride[1], g_initial_state_t.stride[0]))
                gInitialStateTensor = cute.make_tensor(g_initial_state_t.iterator, initial_layout)
            else:
                gInitialStateTensor = cute.make_tensor(g_initial_state_t.iterator, state_layout)
            gInitialState = cute.local_tile(gInitialStateTensor[None, None, head_idx, state_idx], (self.rows_per_cta, self.D), (row_cta_idx, 0))
            gInitialStateWorkspaceTensor = cute.make_tensor(g_initial_state_workspace_t.iterator, state_layout)
            gInitialStateWorkspace = cute.local_tile(gInitialStateWorkspaceTensor[None, None, head_idx, seq_idx], (self.rows_per_cta, self.D), (row_cta_idx, 0))
        else:
            gInitialState = gFixedStateCta[None, None, chunk_start]
            gInitialStateWorkspace = gInitialState
        if cutlass.const_expr(self.store_final_state):
            if cutlass.const_expr(self.use_state_indices):
                output_layout = cute.make_layout((self.D, self.D, num_heads, g_output_state_t.shape[0]), stride=(g_output_state_t.stride[2], g_output_state_t.stride[3], g_output_state_t.stride[1], g_output_state_t.stride[0]))
                gOutputStateTensor = cute.make_tensor(g_output_state_t.iterator, output_layout)
            else:
                gOutputStateTensor = cute.make_tensor(g_output_state_t.iterator, state_layout)
            gOutputState = cute.local_tile(gOutputStateTensor[None, None, head_idx, state_idx], (self.rows_per_cta, self.D), (row_cta_idx, 0))
        else:
            gOutputState = gFixedStateCta[None, None, chunk_start]
        tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=self.tmem_alloc_barrier, allocator_warp_id=0)
        tmem.allocate(256)
        if warp_idx <= self.mma_warp_id:
            tmem.wait_for_alloc()
        if num_chunks == 0:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            if warp_idx == self.compute_warp_group_id:
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tmem.relinquish_alloc_permit()
                tmem.free(tmem_ptr)
        elif warp_idx >= self.compute_warp_group_id and warp_idx < self.compute_warp_group_id + 4:
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_compute_role(tiled_mma, tmem_ptr, sN_tma, n_consumer, mma_ready_producer, mma_done_consumer, gInitialState, gInitialStateWorkspace, gFixedStateCta, gOutputState, chunk_start, num_chunks, tidx, final_mode)
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_mma_role(tiled_mma, tmem_ptr, sM, m_consumer, mma_ready_consumer, mma_done_producer, num_iters, tidx)
        elif warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            cpasync.prefetch_descriptor(tma_atom_m)
            cpasync.prefetch_descriptor(tma_atom_n)
            for chunk_idx in cutlass.range(num_chunks, unroll=1):
                n_producer = self.load_tensor_tma(tma_atom_n, tma_tensor_n, sN_tma, n_producer, head_idx, chunk_start + chunk_idx, row_cta_idx, False)
                if chunk_idx >= start:
                    m_producer = self.load_tensor_tma(tma_atom_m, tma_tensor_m, sM_tma, m_producer, head_idx, chunk_start + chunk_idx, row_cta_idx, True)
        else:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

    @cute.jit
    def run_compute_role(self, tiled_mma, tmem_ptr, sN, n_consumer, mma_ready_producer, mma_done_consumer, gInitialState, gInitialStateWorkspace, gFixedState, gOutputState, chunk_start, num_chunks, tidx, final_mode: cutlass.Int32):
        tCtAcc = self.make_tmem_tensor(tmem_ptr, tiled_mma)
        tCtOpd = self.make_tmem_operand(tmem_ptr, tiled_mma)
        start = cutlass.Int32(0)
        state_stage = cutlass.Int32(0)
        if cutlass.const_expr(self.needs_initial_state):
            self.copy_tensor_to_acc(tiled_mma, tCtAcc, gInitialState, tidx, self.initial_state_dtype != self.acc_dtype, state_stage)
            if cutlass.const_expr(self.store_initial_state):
                self.store_acc(tiled_mma, tCtAcc, tidx, gInitialStateWorkspace, gOutputState, False, state_stage)
        else:
            start = cutlass.Int32(1)
            n_init_handle = n_consumer.wait_and_advance()
            self.copy_tensor_to_acc(tiled_mma, tCtAcc, sN[None, None, n_init_handle.index], tidx, False, state_stage)
            self.store_acc(tiled_mma, tCtAcc, tidx, gFixedState[None, None, chunk_start], gOutputState, False, state_stage)
            n_init_handle.release()
        for chunk_idx in cutlass.range(start, num_chunks, unroll=1):
            self.convert_acc_to_opd(tiled_mma, tCtAcc, tCtOpd, tidx)
            n_handle = n_consumer.wait_and_advance()
            ready_handle = mma_ready_producer.acquire_and_advance()
            self.copy_tensor_to_acc(tiled_mma, tCtAcc, sN[None, None, n_handle.index], tidx, False, state_stage)
            ready_handle.commit()
            n_handle.release()
            done_handle = mma_done_consumer.wait_and_advance()
            self.store_acc(tiled_mma, tCtAcc, tidx, gFixedState[None, None, chunk_start + chunk_idx], gOutputState, False, state_stage)
            done_handle.release()
        if cutlass.const_expr(self.store_final_state):
            self.store_acc(tiled_mma, tCtAcc, tidx, gFixedState[None, None, chunk_start + num_chunks - 1], gOutputState, True, state_stage, final_mode)

    @cute.jit
    def load_tensor_tma(self, tma_atom, tma_tensor, sTensor, tensor_pipeline, head_idx, chunk_idx, row_cta_idx: cutlass.Int32, is_transfer: bool):
        handle = tensor_pipeline.acquire_and_advance()
        sTensor_stage = sTensor[None, None, handle.index]
        mTensor = tma_tensor[None, None, head_idx, chunk_idx]
        if cutlass.const_expr(is_transfer):
            gTensor = cute.zipped_divide(mTensor, (self.D, self.D))[(None, None), (0, 0)]
        else:
            gTensor = cute.zipped_divide(mTensor, (self.rows_per_cta, self.D))[(None, None), (row_cta_idx, 0)]
        (tTsT, tTgT) = cpasync.tma_partition(tma_atom, 0, cute.make_layout(1), cute.group_modes(sTensor_stage, 0, 2), cute.group_modes(gTensor, 0, 2))
        cute.copy(tma_atom, tTgT, tTsT, tma_bar_ptr=handle.barrier)
        return tensor_pipeline

    @cute.jit
    def run_mma_role(self, tiled_mma, tmem_ptr, sM, m_consumer, mma_ready_consumer, mma_done_producer, num_iters, tidx):
        tCtAcc = self.make_tmem_tensor(tmem_ptr, tiled_mma)
        tCrS = self.make_tmem_operand(tmem_ptr, tiled_mma)
        tCrM = tiled_mma.make_fragment_B(sM)
        for _iter_idx in cutlass.range(num_iters, unroll=1):
            m_handle = m_consumer.wait_and_advance()
            ready_handle = mma_ready_consumer.wait_and_advance()
            done_handle = mma_done_producer.acquire_and_advance()
            for kphase in cutlass.range(cute.size(tCrS, mode=[2]), unroll_full=True):
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(tiled_mma, tCtAcc[None, None, None, 0], tCrS[None, None, kphase, 0], tCrM[None, None, kphase, m_handle.index], tCtAcc[None, None, None, 0])
            done_handle.commit()
            ready_handle.release()
            m_handle.release()

    @cute.jit
    def store_acc(self, tiled_mma, tCtAcc: cute.Tensor, tidx: cutlass.Int32, gFixedState: cute.Tensor, gOutputState: cute.Tensor, store_output: bool, stage: cutlass.Int32, final_mode: cutlass.Int32=cutlass.Int32(3)):
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
        tTR_gFixedState = thr_t2r.partition_D(gFixedState)
        tTR_gFixedState = cute.make_tensor(tTR_gFixedState.iterator.align(16), tTR_gFixedState.layout)
        tTR_rAcc = cute.make_rmem_tensor_like(tTR_cAcc, self.acc_dtype)
        if cutlass.const_expr(store_output):
            tTR_gOutputState = thr_t2r.partition_D(gOutputState)
            tTR_gOutputState = cute.make_tensor(tTR_gOutputState.iterator.align(16), tTR_gOutputState.layout)
            tTR_rOutput = cute.make_rmem_tensor_like(tTR_cAcc, self.state_dtype)
        for sub in cutlass.range(cute.size(tTR_rAcc, mode=[2]), unroll_full=True):
            cute.copy(tiled_t2r, tTR_tAcc[None, 0, sub, stage], tTR_rAcc[None, 0, sub])
            if cutlass.const_expr(store_output):
                if final_mode & 1 != 0:
                    cute.autovec_copy(tTR_rAcc[None, 0, sub], tTR_gFixedState[None, 0, sub])
                if final_mode & 2 != 0:
                    tTR_rOutput[None, 0, sub].store(tTR_rAcc[None, 0, sub].load().to(self.state_dtype))
                    cute.autovec_copy(tTR_rOutput[None, 0, sub], tTR_gOutputState[None, 0, sub])
            else:
                cute.autovec_copy(tTR_rAcc[None, 0, sub], tTR_gFixedState[None, 0, sub])

    @cute.jit
    def copy_tensor_to_acc(self, tiled_mma, tCtAcc: cute.Tensor, src: cute.Tensor, tidx: cutlass.Int32, convert_src: bool, stage: cutlass.Int32):
        tCtAcc_mn = utils.gemm.sm100.transform_partitioned_tensor_layout(tCtAcc)
        cAcc = cute.make_identity_tensor((self.rows_per_cta, self.D))
        if cutlass.const_expr(self.rows_per_cta == 128):
            atom_r2t = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype)
        else:
            atom_r2t = cute.make_copy_atom(tcgen05.copy.St16x32bx2Op(tcgen05.copy.Repetition(16)), self.acc_dtype)
        tiled_r2t = tcgen05.make_tmem_copy(atom_r2t, tCtAcc_mn[None, None, 0])
        thr_r2t = tiled_r2t.get_slice(tidx)
        tRT_cAcc = thr_r2t.partition_S(cAcc)
        tRT_tAcc = thr_r2t.partition_D(tCtAcc_mn)
        tRT_src = thr_r2t.partition_S(src)
        tRT_rAcc = cute.make_rmem_tensor_like(tRT_cAcc, self.acc_dtype)
        for sub in cutlass.range(cute.size(tRT_rAcc, mode=[2]), unroll_full=True):
            if cutlass.const_expr(convert_src):
                tRT_rAcc[None, 0, sub].store(tRT_src[None, 0, sub].load().to(self.acc_dtype))
            else:
                cute.autovec_copy(tRT_src[None, 0, sub], tRT_rAcc[None, 0, sub])
            cute.copy(tiled_r2t, tRT_rAcc[None, 0, sub], tRT_tAcc[None, 0, sub, stage])
        cute.arch.fence_view_async_tmem_store()

class K3FinalCOALESCED(_CPDeltaRuleFixupUtcmmaSm100):
    _k3_default_m_stage = 2
    _k3_default_n_stage = 1

    def __init__(self, rows_per_cta: int=64, needs_initial_state: bool=False, store_final_state: bool=False, initial_state_dtype: Type[cutlass.Numeric]=cutlass.Float32, state_dtype: Type[cutlass.Numeric]=cutlass.Float32, cu_seqlens_dtype: Type[cutlass.Numeric]=cutlass.Int32, use_state_indices: bool=False, store_initial_state: bool=False, m_stage=None, n_stage=None):
        self.trace_level = 'event'
        self.causal_variant = 'COALESCED'
        if m_stage is None:
            m_stage = self._k3_default_m_stage
        if n_stage is None:
            n_stage = self._k3_default_n_stage
        if m_stage not in (1, 2):
            raise ValueError(f'K3 m_stage must be 1 or 2, got {m_stage}')
        if n_stage not in (1, 2):
            raise ValueError(f'K3 n_stage must be 1 or 2, got {n_stage}')
        self.needs_initial_state = needs_initial_state
        self.store_final_state = store_final_state
        self.initial_state_dtype = initial_state_dtype
        self.state_dtype = state_dtype
        self.cu_seqlens_dtype = cu_seqlens_dtype
        self.use_state_indices = use_state_indices
        self.store_initial_state = store_initial_state
        self.acc_dtype = cutlass.Float32
        self.mma_dtype = cutlass.TFloat32
        self.D = 128
        if rows_per_cta not in (64, 128):
            raise ValueError(f'rows_per_cta must be 64 or 128, got {rows_per_cta}')
        self.rows_per_cta = rows_per_cta
        self.row_ctas = self.D // rows_per_cta
        self.m_stage = m_stage
        self.n_stage = n_stage
        self.threads_per_cta = 256
        self.compute_warp_group_id = 0
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.num_regs_compute = 120 if rows_per_cta == 64 else 256
        self.num_regs_other = 32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tmem_acc_offset = 0
        self.tmem_opd_offset = 128
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=160)
        self.manual_cache_key('rows_per_cta', 'row_ctas', 'needs_initial_state', 'store_final_state', 'initial_state_dtype', 'state_dtype', 'cu_seqlens_dtype', 'use_state_indices', 'store_initial_state', 'm_stage', 'n_stage', 'num_regs_compute', 'num_regs_other', 'causal_variant', 'trace_level')

    @cute.jit
    def __call__(self, g_transfer_t: cute.Tensor, g_local_state_t: cute.Tensor, g_initial_state_t: cute.Tensor, g_initial_state_workspace_t: cute.Tensor, g_fixed_state_t: cute.Tensor, g_output_state_t: cute.Tensor, g_state_indices_t: cute.Tensor, g_cu_seqlens: cute.Tensor, chunk_len: cutlass.Int32, total_cp_chunks: cutlass.Int32, num_seqs: cutlass.Int32, num_heads: cutlass.Int32, final_mode: cutlass.Int32, stream):
        tiled_mma = cute.make_tiled_mma(tcgen05.MmaTF32Op((self.rows_per_cta, 64, 8), self.cta_group, tcgen05.OperandSource.TMEM, OperandMajorMode.K, OperandMajorMode.MN))
        m_layout = sm100_utils.make_smem_layout_b(tiled_mma, (self.rows_per_cta, self.D, self.D), self.acc_dtype, self.m_stage)
        n_layout = sm100_utils.make_smem_layout_a(tiled_mma, (self.rows_per_cta, self.D, self.D), self.acc_dtype, self.n_stage, is_k_major=True)
        m_semantic_layout = CPDeltaRuleMNPrecomputeUtcmma1Sm100.transform_partitioned_layout(m_layout)
        n_semantic_layout = CPDeltaRuleMNPrecomputeUtcmma1Sm100.transform_partitioned_layout(n_layout)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        m_tma_layout = cute.slice_(m_semantic_layout, (None, None, 0))
        n_tma_layout = cute.slice_(n_semantic_layout, (None, None, 0))
        tma_m = cpasync.make_tiled_tma_atom(tma_op, g_transfer_t, m_tma_layout, (self.D, self.D))
        tma_n = cpasync.make_tiled_tma_atom(tma_op, g_local_state_t, n_tma_layout, (self.rows_per_cta, self.D))
        (tma_atom_m, tma_tensor_m) = tma_m
        (tma_atom_n, tma_tensor_n) = tma_n
        self.tma_m_bytes = cute.size(m_tma_layout) * 4
        self.tma_n_bytes = cute.size(n_tma_layout) * 4

        @cute.struct
        class SharedStorage:
            m_mbar: cute.struct.MemRange[cutlass.Int64, self.m_stage * 2]
            n_mbar: cute.struct.MemRange[cutlass.Int64, self.n_stage * 2]
            mma_ready_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            mma_done_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: cutlass.Int32
            sM: cute.struct.Align[cute.struct.MemRange[self.acc_dtype, cute.cosize(m_layout)], 1024]
            sN: cute.struct.Align[cute.struct.MemRange[self.acc_dtype, cute.cosize(n_layout)], 1024]
        self.shared_storage = SharedStorage
        self.kernel(tiled_mma, tma_atom_m, tma_tensor_m, tma_atom_n, tma_tensor_n, g_initial_state_t, g_initial_state_workspace_t, g_fixed_state_t, g_output_state_t, g_state_indices_t, g_cu_seqlens, m_layout, n_layout, chunk_len, total_cp_chunks, num_seqs, num_heads, final_mode).launch(grid=(num_seqs * num_heads * self.row_ctas, 1, 1), block=(self.threads_per_cta, 1, 1), max_number_threads=(self.threads_per_cta, 1, 1), smem=self.shared_storage.size_in_bytes() + 32768, stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, tiled_mma, tma_atom_m, tma_tensor_m, tma_atom_n, tma_tensor_n, g_initial_state_t, g_initial_state_workspace_t, g_fixed_state_t, g_output_state_t, g_state_indices_t, g_cu_seqlens, m_layout, n_layout, chunk_len, total_cp_chunks, num_seqs, num_heads, final_mode: cutlass.Int32):
        (tidx, _, _) = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        (bx, _, _) = cute.arch.block_idx()
        row_cta_idx = bx % self.row_ctas
        head_seq_idx = bx // self.row_ctas
        head_idx = head_seq_idx % num_heads
        seq_idx = head_seq_idx // num_heads
        seq_start = cutlass.Int32(g_cu_seqlens[seq_idx])
        seq_end = cutlass.Int32(g_cu_seqlens[seq_idx + 1])
        num_chunks = chunks_for_len(seq_end - seq_start, chunk_len)
        chunk_start = varlen_chunk_idx(seq_idx, seq_start, 0, chunk_len)
        start = cutlass.Int32(0) if cutlass.const_expr(self.needs_initial_state) else cutlass.Int32(1)
        num_iters = num_chunks - start
        allocator = utils.SmemAllocator()
        sFinal = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((64, 128), stride=(128, 1)), byte_alignment=128)
        storage = allocator.allocate(self.shared_storage)
        sM = storage.sM.get_tensor(m_layout.outer, swizzle=m_layout.inner)
        sN = storage.sN.get_tensor(n_layout.outer, swizzle=n_layout.inner)
        sM_tma = CPDeltaRuleMNPrecomputeUtcmma1Sm100.transform_partitioned_tensor_layout(sM)
        sN_tma = CPDeltaRuleMNPrecomputeUtcmma1Sm100.transform_partitioned_tensor_layout(sN)
        cg_tma = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        cg_mma = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        cg_compute = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
        cg_compute_tma = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
        (m_producer, m_consumer) = pipeline.PipelineTmaUmma.create(num_stages=self.m_stage, producer_group=cg_tma, consumer_group=cg_mma, tx_count=self.tma_m_bytes, barrier_storage=storage.m_mbar.data_ptr(), defer_sync=True).make_participants()
        (n_producer, n_consumer) = pipeline.PipelineTmaAsync.create(num_stages=self.n_stage, producer_group=cg_tma, consumer_group=cg_compute_tma, tx_count=self.tma_n_bytes, barrier_storage=storage.n_mbar.data_ptr(), defer_sync=True, enable_multicast_signaling=False).make_participants()
        (mma_ready_producer, mma_ready_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_compute, consumer_group=cg_mma, barrier_storage=storage.mma_ready_mbar.data_ptr(), defer_sync=True).make_participants()
        (mma_done_producer, mma_done_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=cg_mma, consumer_group=cg_compute, barrier_storage=storage.mma_done_mbar.data_ptr(), defer_sync=True).make_participants()
        pipeline.pipeline_init_arrive(cluster_shape_mn=cute.make_layout((1, 1)), is_relaxed=True)
        pipeline.pipeline_init_wait(cluster_shape_mn=cute.make_layout((1, 1)))
        out_layout = cute.make_layout((self.D, self.D, num_heads, total_cp_chunks), stride=(self.D, 1, self.D * self.D, self.D * self.D * num_heads))
        gFixedState = cute.make_tensor(g_fixed_state_t.iterator.align(128), out_layout)
        gFixedStateCta = cute.local_tile(gFixedState[None, None, head_idx, None], (self.rows_per_cta, self.D, total_cp_chunks), (row_cta_idx, 0, 0))
        state_layout = cute.make_layout((self.D, self.D, num_heads, num_seqs), stride=(self.D, 1, self.D * self.D, self.D * self.D * num_heads))
        state_idx = seq_idx
        if cutlass.const_expr(self.use_state_indices):
            state_idx = cutlass.Int32(g_state_indices_t[seq_idx])
        if cutlass.const_expr(self.needs_initial_state):
            if cutlass.const_expr(self.use_state_indices):
                initial_layout = cute.make_layout((self.D, self.D, num_heads, g_initial_state_t.shape[0]), stride=(g_initial_state_t.stride[2], g_initial_state_t.stride[3], g_initial_state_t.stride[1], g_initial_state_t.stride[0]))
                gInitialStateTensor = cute.make_tensor(g_initial_state_t.iterator, initial_layout)
            else:
                gInitialStateTensor = cute.make_tensor(g_initial_state_t.iterator, state_layout)
            gInitialState = cute.local_tile(gInitialStateTensor[None, None, head_idx, state_idx], (self.rows_per_cta, self.D), (row_cta_idx, 0))
            gInitialStateWorkspaceTensor = cute.make_tensor(g_initial_state_workspace_t.iterator, state_layout)
            gInitialStateWorkspace = cute.local_tile(gInitialStateWorkspaceTensor[None, None, head_idx, seq_idx], (self.rows_per_cta, self.D), (row_cta_idx, 0))
        else:
            gInitialState = gFixedStateCta[None, None, chunk_start]
            gInitialStateWorkspace = gInitialState
        if cutlass.const_expr(self.store_final_state):
            if cutlass.const_expr(self.use_state_indices):
                output_layout = cute.make_layout((self.D, self.D, num_heads, g_output_state_t.shape[0]), stride=(g_output_state_t.stride[2], g_output_state_t.stride[3], g_output_state_t.stride[1], g_output_state_t.stride[0]))
                gOutputStateTensor = cute.make_tensor(g_output_state_t.iterator, output_layout)
            else:
                gOutputStateTensor = cute.make_tensor(g_output_state_t.iterator, state_layout)
            gOutputState = cute.local_tile(gOutputStateTensor[None, None, head_idx, state_idx], (self.rows_per_cta, self.D), (row_cta_idx, 0))
        else:
            gOutputState = gFixedStateCta[None, None, chunk_start]
        tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=self.tmem_alloc_barrier, allocator_warp_id=0)
        tmem.allocate(256)
        if warp_idx <= self.mma_warp_id:
            tmem.wait_for_alloc()
        if num_chunks == 0:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            if warp_idx == self.compute_warp_group_id:
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tmem.relinquish_alloc_permit()
                tmem.free(tmem_ptr)
        elif warp_idx >= self.compute_warp_group_id and warp_idx < self.compute_warp_group_id + 4:
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_compute_role(tiled_mma, tmem_ptr, sN_tma, n_consumer, mma_ready_producer, mma_done_consumer, gInitialState, gInitialStateWorkspace, gFixedStateCta, gOutputState, chunk_start, num_chunks, tidx, final_mode, sFinal)
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_mma_role(tiled_mma, tmem_ptr, sM, m_consumer, mma_ready_consumer, mma_done_producer, num_iters, tidx)
        elif warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            cpasync.prefetch_descriptor(tma_atom_m)
            cpasync.prefetch_descriptor(tma_atom_n)
            for chunk_idx in cutlass.range(num_chunks, unroll=1):
                n_producer = self.load_tensor_tma(tma_atom_n, tma_tensor_n, sN_tma, n_producer, head_idx, chunk_start + chunk_idx, row_cta_idx, False)
                if chunk_idx >= start:
                    m_producer = self.load_tensor_tma(tma_atom_m, tma_tensor_m, sM_tma, m_producer, head_idx, chunk_start + chunk_idx, row_cta_idx, True)
        else:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

    @cute.jit
    def run_compute_role(self, tiled_mma, tmem_ptr, sN, n_consumer, mma_ready_producer, mma_done_consumer, gInitialState, gInitialStateWorkspace, gFixedState, gOutputState, chunk_start, num_chunks, tidx, final_mode: cutlass.Int32, sFinal: cute.Tensor):
        tCtAcc = self.make_tmem_tensor(tmem_ptr, tiled_mma)
        tCtOpd = self.make_tmem_operand(tmem_ptr, tiled_mma)
        start = cutlass.Int32(0)
        state_stage = cutlass.Int32(0)
        if cutlass.const_expr(self.needs_initial_state):
            self.copy_tensor_to_acc(tiled_mma, tCtAcc, gInitialState, tidx, self.initial_state_dtype != self.acc_dtype, state_stage)
            if cutlass.const_expr(self.store_initial_state):
                self.store_acc(tiled_mma, tCtAcc, tidx, gInitialStateWorkspace, gOutputState, False, state_stage)
        else:
            start = cutlass.Int32(1)
            n_init_handle = n_consumer.wait_and_advance()
            self.copy_tensor_to_acc(tiled_mma, tCtAcc, sN[None, None, n_init_handle.index], tidx, False, state_stage)
            self.store_acc(tiled_mma, tCtAcc, tidx, gFixedState[None, None, chunk_start], gOutputState, False, state_stage)
            n_init_handle.release()
        for chunk_idx in cutlass.range(start, num_chunks, unroll=1):
            self.convert_acc_to_opd(tiled_mma, tCtAcc, tCtOpd, tidx)
            n_handle = n_consumer.wait_and_advance()
            ready_handle = mma_ready_producer.acquire_and_advance()
            self.copy_tensor_to_acc(tiled_mma, tCtAcc, sN[None, None, n_handle.index], tidx, False, state_stage)
            ready_handle.commit()
            n_handle.release()
            done_handle = mma_done_consumer.wait_and_advance()
            self.store_acc(tiled_mma, tCtAcc, tidx, gFixedState[None, None, chunk_start + chunk_idx], gOutputState, False, state_stage)
            done_handle.release()
        if cutlass.const_expr(self.store_final_state):
            self.store_final_layout(tiled_mma, tCtAcc, tidx, gFixedState[None, None, chunk_start + num_chunks - 1], gOutputState, True, state_stage, final_mode, sFinal)

    @cute.jit
    def load_tensor_tma(self, tma_atom, tma_tensor, sTensor, tensor_pipeline, head_idx, chunk_idx, row_cta_idx: cutlass.Int32, is_transfer: bool):
        handle = tensor_pipeline.acquire_and_advance()
        sTensor_stage = sTensor[None, None, handle.index]
        mTensor = tma_tensor[None, None, head_idx, chunk_idx]
        if cutlass.const_expr(is_transfer):
            gTensor = cute.zipped_divide(mTensor, (self.D, self.D))[(None, None), (0, 0)]
        else:
            gTensor = cute.zipped_divide(mTensor, (self.rows_per_cta, self.D))[(None, None), (row_cta_idx, 0)]
        (tTsT, tTgT) = cpasync.tma_partition(tma_atom, 0, cute.make_layout(1), cute.group_modes(sTensor_stage, 0, 2), cute.group_modes(gTensor, 0, 2))
        cute.copy(tma_atom, tTgT, tTsT, tma_bar_ptr=handle.barrier)
        return tensor_pipeline

    @cute.jit
    def run_mma_role(self, tiled_mma, tmem_ptr, sM, m_consumer, mma_ready_consumer, mma_done_producer, num_iters, tidx):
        tCtAcc = self.make_tmem_tensor(tmem_ptr, tiled_mma)
        tCrS = self.make_tmem_operand(tmem_ptr, tiled_mma)
        tCrM = tiled_mma.make_fragment_B(sM)
        for _iter_idx in cutlass.range(num_iters, unroll=1):
            m_handle = m_consumer.wait_and_advance()
            ready_handle = mma_ready_consumer.wait_and_advance()
            done_handle = mma_done_producer.acquire_and_advance()
            for kphase in cutlass.range(cute.size(tCrS, mode=[2]), unroll_full=True):
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(tiled_mma, tCtAcc[None, None, None, 0], tCrS[None, None, kphase, 0], tCrM[None, None, kphase, m_handle.index], tCtAcc[None, None, None, 0])
            done_handle.commit()
            ready_handle.release()
            m_handle.release()

    @cute.jit
    def store_acc(self, tiled_mma, tCtAcc: cute.Tensor, tidx: cutlass.Int32, gFixedState: cute.Tensor, gOutputState: cute.Tensor, store_output: bool, stage: cutlass.Int32):
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
        tTR_gFixedState = thr_t2r.partition_D(gFixedState)
        tTR_gFixedState = cute.make_tensor(tTR_gFixedState.iterator.align(16), tTR_gFixedState.layout)
        tTR_rAcc = cute.make_rmem_tensor_like(tTR_cAcc, self.acc_dtype)
        if cutlass.const_expr(store_output):
            tTR_gOutputState = thr_t2r.partition_D(gOutputState)
            tTR_gOutputState = cute.make_tensor(tTR_gOutputState.iterator.align(16), tTR_gOutputState.layout)
            tTR_rOutput = cute.make_rmem_tensor_like(tTR_cAcc, self.state_dtype)
        for sub in cutlass.range(cute.size(tTR_rAcc, mode=[2]), unroll_full=True):
            cute.copy(tiled_t2r, tTR_tAcc[None, 0, sub, stage], tTR_rAcc[None, 0, sub])
            cute.autovec_copy(tTR_rAcc[None, 0, sub], tTR_gFixedState[None, 0, sub])
            if cutlass.const_expr(store_output):
                tTR_rOutput[None, 0, sub].store(tTR_rAcc[None, 0, sub].load().to(self.state_dtype))
                cute.autovec_copy(tTR_rOutput[None, 0, sub], tTR_gOutputState[None, 0, sub])

    @cute.jit
    def copy_tensor_to_acc(self, tiled_mma, tCtAcc: cute.Tensor, src: cute.Tensor, tidx: cutlass.Int32, convert_src: bool, stage: cutlass.Int32):
        tCtAcc_mn = utils.gemm.sm100.transform_partitioned_tensor_layout(tCtAcc)
        cAcc = cute.make_identity_tensor((self.rows_per_cta, self.D))
        if cutlass.const_expr(self.rows_per_cta == 128):
            atom_r2t = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype)
        else:
            atom_r2t = cute.make_copy_atom(tcgen05.copy.St16x32bx2Op(tcgen05.copy.Repetition(16)), self.acc_dtype)
        tiled_r2t = tcgen05.make_tmem_copy(atom_r2t, tCtAcc_mn[None, None, 0])
        thr_r2t = tiled_r2t.get_slice(tidx)
        tRT_cAcc = thr_r2t.partition_S(cAcc)
        tRT_tAcc = thr_r2t.partition_D(tCtAcc_mn)
        tRT_src = thr_r2t.partition_S(src)
        tRT_rAcc = cute.make_rmem_tensor_like(tRT_cAcc, self.acc_dtype)
        for sub in cutlass.range(cute.size(tRT_rAcc, mode=[2]), unroll_full=True):
            if cutlass.const_expr(convert_src):
                tRT_rAcc[None, 0, sub].store(tRT_src[None, 0, sub].load().to(self.acc_dtype))
            else:
                cute.autovec_copy(tRT_src[None, 0, sub], tRT_rAcc[None, 0, sub])
            cute.copy(tiled_r2t, tRT_rAcc[None, 0, sub], tRT_tAcc[None, 0, sub, stage])
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def store_final_layout(self, tiled_mma, tCtAcc: cute.Tensor, tidx: cutlass.Int32, gFixedState: cute.Tensor, gOutputState: cute.Tensor, store_output: bool, stage: cutlass.Int32, final_mode: cutlass.Int32, sFinal: cute.Tensor):
        tCtAcc_mn = utils.gemm.sm100.transform_partitioned_tensor_layout(tCtAcc)
        atom_t2r = cute.make_copy_atom(tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(16)), self.acc_dtype)
        tiled_t2r = tcgen05.make_tmem_copy(atom_t2r, tCtAcc_mn[None, None, 0])
        thr = tiled_t2r.get_slice(tidx)
        tsrc = thr.partition_S(tCtAcc_mn)
        tdst = thr.partition_D(sFinal)
        reg = cute.make_rmem_tensor_like(tdst, self.acc_dtype)
        for sub in cutlass.range(cute.size(reg, mode=[2]), unroll_full=True):
            cute.copy(tiled_t2r, tsrc[None, 0, sub, stage], reg[None, 0, sub])
            cute.autovec_copy(reg[None, 0, sub], tdst[None, 0, sub])
        pipeline.NamedBarrier(barrier_id=2, num_threads=128).arrive_and_wait()
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype, num_bits_per_copy=128)
        ct = cute.make_tiled_copy_tv(atom, cute.make_layout((4, 32), stride=(32, 1)), cute.make_layout((1, 4), stride=(4, 1)))
        sl = ct.get_slice(tidx)
        sr = sl.partition_S(sFinal)
        sr = cute.make_tensor(sr.iterator.align(16), sr.layout)
        rf = cute.make_rmem_tensor_like(sr, self.acc_dtype)
        rf = cute.make_tensor(rf.iterator.align(16), rf.layout)
        cute.copy(ct, sr, rf)
        if final_mode & 1 != 0:
            df = sl.partition_D(gFixedState)
            df = cute.make_tensor(df.iterator.align(16), df.layout)
            cute.copy(ct, rf, df)
        if final_mode & 2 != 0:
            do = sl.partition_D(gOutputState)
            do = cute.make_tensor(do.iterator.align(16), do.layout)
            cute.copy(ct, rf, do)
        pipeline.NamedBarrier(barrier_id=2, num_threads=128).arrive_and_wait()

class K3FinalSWIZZLED(_CPDeltaRuleFixupUtcmmaSm100):
    _k3_default_m_stage = 2
    _k3_default_n_stage = 1

    def __init__(self, rows_per_cta: int=64, needs_initial_state: bool=False, store_final_state: bool=False, initial_state_dtype: Type[cutlass.Numeric]=cutlass.Float32, state_dtype: Type[cutlass.Numeric]=cutlass.Float32, cu_seqlens_dtype: Type[cutlass.Numeric]=cutlass.Int32, use_state_indices: bool=False, store_initial_state: bool=False, m_stage=None, n_stage=None):
        self.trace_level = 'event'
        self.causal_variant = 'SWIZZLED'
        if m_stage is None:
            m_stage = self._k3_default_m_stage
        if n_stage is None:
            n_stage = self._k3_default_n_stage
        if m_stage not in (1, 2):
            raise ValueError(f'K3 m_stage must be 1 or 2, got {m_stage}')
        if n_stage not in (1, 2):
            raise ValueError(f'K3 n_stage must be 1 or 2, got {n_stage}')
        self.needs_initial_state = needs_initial_state
        self.store_final_state = store_final_state
        self.initial_state_dtype = initial_state_dtype
        self.state_dtype = state_dtype
        self.cu_seqlens_dtype = cu_seqlens_dtype
        self.use_state_indices = use_state_indices
        self.store_initial_state = store_initial_state
        self.acc_dtype = cutlass.Float32
        self.mma_dtype = cutlass.TFloat32
        self.D = 128
        if rows_per_cta not in (64, 128):
            raise ValueError(f'rows_per_cta must be 64 or 128, got {rows_per_cta}')
        self.rows_per_cta = rows_per_cta
        self.row_ctas = self.D // rows_per_cta
        self.m_stage = m_stage
        self.n_stage = n_stage
        self.threads_per_cta = 256
        self.compute_warp_group_id = 0
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.num_regs_compute = 120 if rows_per_cta == 64 else 256
        self.num_regs_other = 32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tmem_acc_offset = 0
        self.tmem_opd_offset = 128
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=160)
        self.manual_cache_key('rows_per_cta', 'row_ctas', 'needs_initial_state', 'store_final_state', 'initial_state_dtype', 'state_dtype', 'cu_seqlens_dtype', 'use_state_indices', 'store_initial_state', 'm_stage', 'n_stage', 'num_regs_compute', 'num_regs_other', 'causal_variant', 'trace_level')

    @cute.jit
    def __call__(self, g_transfer_t: cute.Tensor, g_local_state_t: cute.Tensor, g_initial_state_t: cute.Tensor, g_initial_state_workspace_t: cute.Tensor, g_fixed_state_t: cute.Tensor, g_output_state_t: cute.Tensor, g_state_indices_t: cute.Tensor, g_cu_seqlens: cute.Tensor, chunk_len: cutlass.Int32, total_cp_chunks: cutlass.Int32, num_seqs: cutlass.Int32, num_heads: cutlass.Int32, final_mode: cutlass.Int32, stream):
        tiled_mma = cute.make_tiled_mma(tcgen05.MmaTF32Op((self.rows_per_cta, 64, 8), self.cta_group, tcgen05.OperandSource.TMEM, OperandMajorMode.K, OperandMajorMode.MN))
        m_layout = sm100_utils.make_smem_layout_b(tiled_mma, (self.rows_per_cta, self.D, self.D), self.acc_dtype, self.m_stage)
        n_layout = sm100_utils.make_smem_layout_a(tiled_mma, (self.rows_per_cta, self.D, self.D), self.acc_dtype, self.n_stage, is_k_major=True)
        m_semantic_layout = CPDeltaRuleMNPrecomputeUtcmma1Sm100.transform_partitioned_layout(m_layout)
        n_semantic_layout = CPDeltaRuleMNPrecomputeUtcmma1Sm100.transform_partitioned_layout(n_layout)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        m_tma_layout = cute.slice_(m_semantic_layout, (None, None, 0))
        n_tma_layout = cute.slice_(n_semantic_layout, (None, None, 0))
        tma_m = cpasync.make_tiled_tma_atom(tma_op, g_transfer_t, m_tma_layout, (self.D, self.D))
        tma_n = cpasync.make_tiled_tma_atom(tma_op, g_local_state_t, n_tma_layout, (self.rows_per_cta, self.D))
        (tma_atom_m, tma_tensor_m) = tma_m
        (tma_atom_n, tma_tensor_n) = tma_n
        self.tma_m_bytes = cute.size(m_tma_layout) * 4
        self.tma_n_bytes = cute.size(n_tma_layout) * 4

        @cute.struct
        class SharedStorage:
            m_mbar: cute.struct.MemRange[cutlass.Int64, self.m_stage * 2]
            n_mbar: cute.struct.MemRange[cutlass.Int64, self.n_stage * 2]
            mma_ready_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            mma_done_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: cutlass.Int32
            sM: cute.struct.Align[cute.struct.MemRange[self.acc_dtype, cute.cosize(m_layout)], 1024]
            sN: cute.struct.Align[cute.struct.MemRange[self.acc_dtype, cute.cosize(n_layout)], 1024]
        self.shared_storage = SharedStorage
        self.kernel(tiled_mma, tma_atom_m, tma_tensor_m, tma_atom_n, tma_tensor_n, g_initial_state_t, g_initial_state_workspace_t, g_fixed_state_t, g_output_state_t, g_state_indices_t, g_cu_seqlens, m_layout, n_layout, chunk_len, total_cp_chunks, num_seqs, num_heads, final_mode).launch(grid=(num_seqs * num_heads * self.row_ctas, 1, 1), block=(self.threads_per_cta, 1, 1), max_number_threads=(self.threads_per_cta, 1, 1), smem=self.shared_storage.size_in_bytes() + 32768, stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, tiled_mma, tma_atom_m, tma_tensor_m, tma_atom_n, tma_tensor_n, g_initial_state_t, g_initial_state_workspace_t, g_fixed_state_t, g_output_state_t, g_state_indices_t, g_cu_seqlens, m_layout, n_layout, chunk_len, total_cp_chunks, num_seqs, num_heads, final_mode: cutlass.Int32):
        (tidx, _, _) = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        (bx, _, _) = cute.arch.block_idx()
        row_cta_idx = bx % self.row_ctas
        head_seq_idx = bx // self.row_ctas
        head_idx = head_seq_idx % num_heads
        seq_idx = head_seq_idx // num_heads
        seq_start = cutlass.Int32(g_cu_seqlens[seq_idx])
        seq_end = cutlass.Int32(g_cu_seqlens[seq_idx + 1])
        num_chunks = chunks_for_len(seq_end - seq_start, chunk_len)
        chunk_start = varlen_chunk_idx(seq_idx, seq_start, 0, chunk_len)
        start = cutlass.Int32(0) if cutlass.const_expr(self.needs_initial_state) else cutlass.Int32(1)
        num_iters = num_chunks - start
        allocator = utils.SmemAllocator()
        sFinal = allocator.allocate_tensor(cutlass.Float32, cute.make_composed_layout(cute.make_swizzle(3, 2, 5), 0, cute.make_layout((64, 128), stride=(128, 1))), byte_alignment=128)
        storage = allocator.allocate(self.shared_storage)
        sM = storage.sM.get_tensor(m_layout.outer, swizzle=m_layout.inner)
        sN = storage.sN.get_tensor(n_layout.outer, swizzle=n_layout.inner)
        sM_tma = CPDeltaRuleMNPrecomputeUtcmma1Sm100.transform_partitioned_tensor_layout(sM)
        sN_tma = CPDeltaRuleMNPrecomputeUtcmma1Sm100.transform_partitioned_tensor_layout(sN)
        cg_tma = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        cg_mma = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        cg_compute = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
        cg_compute_tma = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
        (m_producer, m_consumer) = pipeline.PipelineTmaUmma.create(num_stages=self.m_stage, producer_group=cg_tma, consumer_group=cg_mma, tx_count=self.tma_m_bytes, barrier_storage=storage.m_mbar.data_ptr(), defer_sync=True).make_participants()
        (n_producer, n_consumer) = pipeline.PipelineTmaAsync.create(num_stages=self.n_stage, producer_group=cg_tma, consumer_group=cg_compute_tma, tx_count=self.tma_n_bytes, barrier_storage=storage.n_mbar.data_ptr(), defer_sync=True, enable_multicast_signaling=False).make_participants()
        (mma_ready_producer, mma_ready_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_compute, consumer_group=cg_mma, barrier_storage=storage.mma_ready_mbar.data_ptr(), defer_sync=True).make_participants()
        (mma_done_producer, mma_done_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=cg_mma, consumer_group=cg_compute, barrier_storage=storage.mma_done_mbar.data_ptr(), defer_sync=True).make_participants()
        pipeline.pipeline_init_arrive(cluster_shape_mn=cute.make_layout((1, 1)), is_relaxed=True)
        pipeline.pipeline_init_wait(cluster_shape_mn=cute.make_layout((1, 1)))
        out_layout = cute.make_layout((self.D, self.D, num_heads, total_cp_chunks), stride=(self.D, 1, self.D * self.D, self.D * self.D * num_heads))
        gFixedState = cute.make_tensor(g_fixed_state_t.iterator.align(128), out_layout)
        gFixedStateCta = cute.local_tile(gFixedState[None, None, head_idx, None], (self.rows_per_cta, self.D, total_cp_chunks), (row_cta_idx, 0, 0))
        state_layout = cute.make_layout((self.D, self.D, num_heads, num_seqs), stride=(self.D, 1, self.D * self.D, self.D * self.D * num_heads))
        state_idx = seq_idx
        if cutlass.const_expr(self.use_state_indices):
            state_idx = cutlass.Int32(g_state_indices_t[seq_idx])
        if cutlass.const_expr(self.needs_initial_state):
            if cutlass.const_expr(self.use_state_indices):
                initial_layout = cute.make_layout((self.D, self.D, num_heads, g_initial_state_t.shape[0]), stride=(g_initial_state_t.stride[2], g_initial_state_t.stride[3], g_initial_state_t.stride[1], g_initial_state_t.stride[0]))
                gInitialStateTensor = cute.make_tensor(g_initial_state_t.iterator, initial_layout)
            else:
                gInitialStateTensor = cute.make_tensor(g_initial_state_t.iterator, state_layout)
            gInitialState = cute.local_tile(gInitialStateTensor[None, None, head_idx, state_idx], (self.rows_per_cta, self.D), (row_cta_idx, 0))
            gInitialStateWorkspaceTensor = cute.make_tensor(g_initial_state_workspace_t.iterator, state_layout)
            gInitialStateWorkspace = cute.local_tile(gInitialStateWorkspaceTensor[None, None, head_idx, seq_idx], (self.rows_per_cta, self.D), (row_cta_idx, 0))
        else:
            gInitialState = gFixedStateCta[None, None, chunk_start]
            gInitialStateWorkspace = gInitialState
        if cutlass.const_expr(self.store_final_state):
            if cutlass.const_expr(self.use_state_indices):
                output_layout = cute.make_layout((self.D, self.D, num_heads, g_output_state_t.shape[0]), stride=(g_output_state_t.stride[2], g_output_state_t.stride[3], g_output_state_t.stride[1], g_output_state_t.stride[0]))
                gOutputStateTensor = cute.make_tensor(g_output_state_t.iterator, output_layout)
            else:
                gOutputStateTensor = cute.make_tensor(g_output_state_t.iterator, state_layout)
            gOutputState = cute.local_tile(gOutputStateTensor[None, None, head_idx, state_idx], (self.rows_per_cta, self.D), (row_cta_idx, 0))
        else:
            gOutputState = gFixedStateCta[None, None, chunk_start]
        tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=self.tmem_alloc_barrier, allocator_warp_id=0)
        tmem.allocate(256)
        if warp_idx <= self.mma_warp_id:
            tmem.wait_for_alloc()
        if num_chunks == 0:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            if warp_idx == self.compute_warp_group_id:
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tmem.relinquish_alloc_permit()
                tmem.free(tmem_ptr)
        elif warp_idx >= self.compute_warp_group_id and warp_idx < self.compute_warp_group_id + 4:
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_compute_role(tiled_mma, tmem_ptr, sN_tma, n_consumer, mma_ready_producer, mma_done_consumer, gInitialState, gInitialStateWorkspace, gFixedStateCta, gOutputState, chunk_start, num_chunks, tidx, final_mode, sFinal)
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_mma_role(tiled_mma, tmem_ptr, sM, m_consumer, mma_ready_consumer, mma_done_producer, num_iters, tidx)
        elif warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            cpasync.prefetch_descriptor(tma_atom_m)
            cpasync.prefetch_descriptor(tma_atom_n)
            for chunk_idx in cutlass.range(num_chunks, unroll=1):
                n_producer = self.load_tensor_tma(tma_atom_n, tma_tensor_n, sN_tma, n_producer, head_idx, chunk_start + chunk_idx, row_cta_idx, False)
                if chunk_idx >= start:
                    m_producer = self.load_tensor_tma(tma_atom_m, tma_tensor_m, sM_tma, m_producer, head_idx, chunk_start + chunk_idx, row_cta_idx, True)
        else:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

    @cute.jit
    def run_compute_role(self, tiled_mma, tmem_ptr, sN, n_consumer, mma_ready_producer, mma_done_consumer, gInitialState, gInitialStateWorkspace, gFixedState, gOutputState, chunk_start, num_chunks, tidx, final_mode: cutlass.Int32, sFinal: cute.Tensor):
        tCtAcc = self.make_tmem_tensor(tmem_ptr, tiled_mma)
        tCtOpd = self.make_tmem_operand(tmem_ptr, tiled_mma)
        start = cutlass.Int32(0)
        state_stage = cutlass.Int32(0)
        if cutlass.const_expr(self.needs_initial_state):
            self.copy_tensor_to_acc(tiled_mma, tCtAcc, gInitialState, tidx, self.initial_state_dtype != self.acc_dtype, state_stage)
            if cutlass.const_expr(self.store_initial_state):
                self.store_acc(tiled_mma, tCtAcc, tidx, gInitialStateWorkspace, gOutputState, False, state_stage)
        else:
            start = cutlass.Int32(1)
            n_init_handle = n_consumer.wait_and_advance()
            self.copy_tensor_to_acc(tiled_mma, tCtAcc, sN[None, None, n_init_handle.index], tidx, False, state_stage)
            self.store_acc(tiled_mma, tCtAcc, tidx, gFixedState[None, None, chunk_start], gOutputState, False, state_stage)
            n_init_handle.release()
        for chunk_idx in cutlass.range(start, num_chunks, unroll=1):
            self.convert_acc_to_opd(tiled_mma, tCtAcc, tCtOpd, tidx)
            n_handle = n_consumer.wait_and_advance()
            ready_handle = mma_ready_producer.acquire_and_advance()
            self.copy_tensor_to_acc(tiled_mma, tCtAcc, sN[None, None, n_handle.index], tidx, False, state_stage)
            ready_handle.commit()
            n_handle.release()
            done_handle = mma_done_consumer.wait_and_advance()
            self.store_acc(tiled_mma, tCtAcc, tidx, gFixedState[None, None, chunk_start + chunk_idx], gOutputState, False, state_stage)
            done_handle.release()
        if cutlass.const_expr(self.store_final_state):
            self.store_final_layout(tiled_mma, tCtAcc, tidx, gFixedState[None, None, chunk_start + num_chunks - 1], gOutputState, True, state_stage, final_mode, sFinal)

    @cute.jit
    def load_tensor_tma(self, tma_atom, tma_tensor, sTensor, tensor_pipeline, head_idx, chunk_idx, row_cta_idx: cutlass.Int32, is_transfer: bool):
        handle = tensor_pipeline.acquire_and_advance()
        sTensor_stage = sTensor[None, None, handle.index]
        mTensor = tma_tensor[None, None, head_idx, chunk_idx]
        if cutlass.const_expr(is_transfer):
            gTensor = cute.zipped_divide(mTensor, (self.D, self.D))[(None, None), (0, 0)]
        else:
            gTensor = cute.zipped_divide(mTensor, (self.rows_per_cta, self.D))[(None, None), (row_cta_idx, 0)]
        (tTsT, tTgT) = cpasync.tma_partition(tma_atom, 0, cute.make_layout(1), cute.group_modes(sTensor_stage, 0, 2), cute.group_modes(gTensor, 0, 2))
        cute.copy(tma_atom, tTgT, tTsT, tma_bar_ptr=handle.barrier)
        return tensor_pipeline

    @cute.jit
    def run_mma_role(self, tiled_mma, tmem_ptr, sM, m_consumer, mma_ready_consumer, mma_done_producer, num_iters, tidx):
        tCtAcc = self.make_tmem_tensor(tmem_ptr, tiled_mma)
        tCrS = self.make_tmem_operand(tmem_ptr, tiled_mma)
        tCrM = tiled_mma.make_fragment_B(sM)
        for _iter_idx in cutlass.range(num_iters, unroll=1):
            m_handle = m_consumer.wait_and_advance()
            ready_handle = mma_ready_consumer.wait_and_advance()
            done_handle = mma_done_producer.acquire_and_advance()
            for kphase in cutlass.range(cute.size(tCrS, mode=[2]), unroll_full=True):
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(tiled_mma, tCtAcc[None, None, None, 0], tCrS[None, None, kphase, 0], tCrM[None, None, kphase, m_handle.index], tCtAcc[None, None, None, 0])
            done_handle.commit()
            ready_handle.release()
            m_handle.release()

    @cute.jit
    def store_acc(self, tiled_mma, tCtAcc: cute.Tensor, tidx: cutlass.Int32, gFixedState: cute.Tensor, gOutputState: cute.Tensor, store_output: bool, stage: cutlass.Int32):
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
        tTR_gFixedState = thr_t2r.partition_D(gFixedState)
        tTR_gFixedState = cute.make_tensor(tTR_gFixedState.iterator.align(16), tTR_gFixedState.layout)
        tTR_rAcc = cute.make_rmem_tensor_like(tTR_cAcc, self.acc_dtype)
        if cutlass.const_expr(store_output):
            tTR_gOutputState = thr_t2r.partition_D(gOutputState)
            tTR_gOutputState = cute.make_tensor(tTR_gOutputState.iterator.align(16), tTR_gOutputState.layout)
            tTR_rOutput = cute.make_rmem_tensor_like(tTR_cAcc, self.state_dtype)
        for sub in cutlass.range(cute.size(tTR_rAcc, mode=[2]), unroll_full=True):
            cute.copy(tiled_t2r, tTR_tAcc[None, 0, sub, stage], tTR_rAcc[None, 0, sub])
            cute.autovec_copy(tTR_rAcc[None, 0, sub], tTR_gFixedState[None, 0, sub])
            if cutlass.const_expr(store_output):
                tTR_rOutput[None, 0, sub].store(tTR_rAcc[None, 0, sub].load().to(self.state_dtype))
                cute.autovec_copy(tTR_rOutput[None, 0, sub], tTR_gOutputState[None, 0, sub])

    @cute.jit
    def copy_tensor_to_acc(self, tiled_mma, tCtAcc: cute.Tensor, src: cute.Tensor, tidx: cutlass.Int32, convert_src: bool, stage: cutlass.Int32):
        tCtAcc_mn = utils.gemm.sm100.transform_partitioned_tensor_layout(tCtAcc)
        cAcc = cute.make_identity_tensor((self.rows_per_cta, self.D))
        if cutlass.const_expr(self.rows_per_cta == 128):
            atom_r2t = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype)
        else:
            atom_r2t = cute.make_copy_atom(tcgen05.copy.St16x32bx2Op(tcgen05.copy.Repetition(16)), self.acc_dtype)
        tiled_r2t = tcgen05.make_tmem_copy(atom_r2t, tCtAcc_mn[None, None, 0])
        thr_r2t = tiled_r2t.get_slice(tidx)
        tRT_cAcc = thr_r2t.partition_S(cAcc)
        tRT_tAcc = thr_r2t.partition_D(tCtAcc_mn)
        tRT_src = thr_r2t.partition_S(src)
        tRT_rAcc = cute.make_rmem_tensor_like(tRT_cAcc, self.acc_dtype)
        for sub in cutlass.range(cute.size(tRT_rAcc, mode=[2]), unroll_full=True):
            if cutlass.const_expr(convert_src):
                tRT_rAcc[None, 0, sub].store(tRT_src[None, 0, sub].load().to(self.acc_dtype))
            else:
                cute.autovec_copy(tRT_src[None, 0, sub], tRT_rAcc[None, 0, sub])
            cute.copy(tiled_r2t, tRT_rAcc[None, 0, sub], tRT_tAcc[None, 0, sub, stage])
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def store_final_layout(self, tiled_mma, tCtAcc: cute.Tensor, tidx: cutlass.Int32, gFixedState: cute.Tensor, gOutputState: cute.Tensor, store_output: bool, stage: cutlass.Int32, final_mode: cutlass.Int32, sFinal: cute.Tensor):
        tCtAcc_mn = utils.gemm.sm100.transform_partitioned_tensor_layout(tCtAcc)
        atom_t2r = cute.make_copy_atom(tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(16)), self.acc_dtype)
        tiled_t2r = tcgen05.make_tmem_copy(atom_t2r, tCtAcc_mn[None, None, 0])
        thr = tiled_t2r.get_slice(tidx)
        tsrc = thr.partition_S(tCtAcc_mn)
        tdst = thr.partition_D(sFinal)
        reg = cute.make_rmem_tensor_like(tdst, self.acc_dtype)
        for sub in cutlass.range(cute.size(reg, mode=[2]), unroll_full=True):
            cute.copy(tiled_t2r, tsrc[None, 0, sub, stage], reg[None, 0, sub])
            cute.autovec_copy(reg[None, 0, sub], tdst[None, 0, sub])
        pipeline.NamedBarrier(barrier_id=2, num_threads=128).arrive_and_wait()
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype, num_bits_per_copy=128)
        ct = cute.make_tiled_copy_tv(atom, cute.make_layout((4, 32), stride=(32, 1)), cute.make_layout((1, 4), stride=(4, 1)))
        sl = ct.get_slice(tidx)
        sr = sl.partition_S(sFinal)
        sr = cute.make_tensor(sr.iterator.align(16), sr.layout)
        rf = cute.make_rmem_tensor_like(sr, self.acc_dtype)
        rf = cute.make_tensor(rf.iterator.align(16), rf.layout)
        cute.copy(ct, sr, rf)
        if final_mode & 1 != 0:
            df = sl.partition_D(gFixedState)
            df = cute.make_tensor(df.iterator.align(16), df.layout)
            cute.copy(ct, rf, df)
        if final_mode & 2 != 0:
            do = sl.partition_D(gOutputState)
            do = cute.make_tensor(do.iterator.align(16), do.layout)
            cute.copy(ct, rf, do)
        pipeline.NamedBarrier(barrier_id=2, num_threads=128).arrive_and_wait()
_KERNEL_CLASSES = {('RUNTIME', 'event'): K3FinalRUNTIME, ('COALESCED', 'event'): K3FinalCOALESCED, ('SWIZZLED', 'event'): K3FinalSWIZZLED}
_CONFIGURED_CLASSES = {}

def get_kernel_class(variant: str, level: str='event', m_stage: int=2, n_stage: int=1):
    """Return a named constructor class for one causal variant and stage pair.

    The returned class accepts the upstream constructor arguments unchanged;
    its optional trailing m_stage/n_stage arguments default to the pair selected
    here.  Stage and trace identity are also included in the compile key.
    """
    if variant not in {item[0] for item in _KERNEL_CLASSES}:
        raise ValueError(f'unknown K3 causal variant {variant!r}')
    if level not in ('event', 'minimal', 'full'):
        raise ValueError(f'unknown K3 source level {level!r}')
    if m_stage not in (1, 2) or n_stage not in (1, 2):
        raise ValueError(f'K3 stages must each be 1 or 2, got M{m_stage}/N{n_stage}')
    key = (variant, level, m_stage, n_stage)
    configured = _CONFIGURED_CLASSES.get(key)
    if configured is None:
        base = _KERNEL_CLASSES[variant, level]
        configured = type(f'{base.__name__}_M{m_stage}_N{n_stage}', (base,), {'__module__': __name__, '_k3_default_m_stage': m_stage, '_k3_default_n_stage': n_stage})
        _CONFIGURED_CLASSES[key] = configured
    return configured
__all__ = ['get_kernel_class']
