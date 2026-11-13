from flashinfer.gdn_kernels.blackwell.gated_delta_net_cp import *

class K3OptCoalescedStore(CPDeltaRuleFixupUtcmmaSm100):

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

    @cute.jit
    def run_compute_role(self, tiled_mma, tmem_ptr, sN, n_consumer, mma_ready_producer, mma_done_consumer, gInitialState, gInitialStateWorkspace, gFixedState, gOutputState, chunk_start, num_chunks, tidx, sStage):
        tCtAcc = self.make_tmem_tensor(tmem_ptr, tiled_mma)
        tCtOpd = self.make_tmem_operand(tmem_ptr, tiled_mma)
        start = cutlass.Int32(0)
        state_stage = cutlass.Int32(0)
        if cutlass.const_expr(self.needs_initial_state):
            self.copy_tensor_to_acc(tiled_mma, tCtAcc, gInitialState, tidx, self.initial_state_dtype != self.acc_dtype, state_stage)
            if cutlass.const_expr(self.store_initial_state):
                self.store_acc(tiled_mma, tCtAcc, tidx, gInitialStateWorkspace, gOutputState, False, state_stage, sStage)
        else:
            start = cutlass.Int32(1)
            n_init_handle = n_consumer.wait_and_advance()
            self.copy_tensor_to_acc(tiled_mma, tCtAcc, sN[None, None, n_init_handle.index], tidx, False, state_stage)
            self.store_acc(tiled_mma, tCtAcc, tidx, gFixedState[None, None, chunk_start], gOutputState, False, state_stage, sStage)
            n_init_handle.release()
        for chunk_idx in cutlass.range(start, num_chunks, unroll=1):
            self.convert_acc_to_opd(tiled_mma, tCtAcc, tCtOpd, tidx)
            n_handle = n_consumer.wait_and_advance()
            ready_handle = mma_ready_producer.acquire_and_advance()
            self.copy_tensor_to_acc(tiled_mma, tCtAcc, sN[None, None, n_handle.index], tidx, False, state_stage)
            ready_handle.commit()
            n_handle.release()
            done_handle = mma_done_consumer.wait_and_advance()
            self.store_acc(tiled_mma, tCtAcc, tidx, gFixedState[None, None, chunk_start + chunk_idx], gOutputState, False, state_stage, sStage)
            done_handle.release()
        if cutlass.const_expr(self.store_final_state):
            self.store_acc(tiled_mma, tCtAcc, tidx, gFixedState[None, None, chunk_start + num_chunks - 1], gOutputState, True, state_stage, sStage)

    @cute.kernel
    def kernel(self, tiled_mma, tma_atom_m, tma_tensor_m, tma_atom_n, tma_tensor_n, g_initial_state_t, g_initial_state_workspace_t, g_fixed_state_t, g_output_state_t, g_state_indices_t, g_cu_seqlens, m_layout, n_layout, chunk_len, total_cp_chunks, num_seqs, num_heads):
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
        sStage = storage.sStage.get_tensor(cute.make_layout((self.rows_per_cta, self.D), stride=(self.D + 4, 1)))
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
            self.run_compute_role(tiled_mma, tmem_ptr, sN_tma, n_consumer, mma_ready_producer, mma_done_consumer, gInitialState, gInitialStateWorkspace, gFixedStateCta, gOutputState, chunk_start, num_chunks, tidx, sStage)
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
    def __call__(self, g_transfer_t: cute.Tensor, g_local_state_t: cute.Tensor, g_initial_state_t: cute.Tensor, g_initial_state_workspace_t: cute.Tensor, g_fixed_state_t: cute.Tensor, g_output_state_t: cute.Tensor, g_state_indices_t: cute.Tensor, g_cu_seqlens: cute.Tensor, chunk_len: cutlass.Int32, total_cp_chunks: cutlass.Int32, num_seqs: cutlass.Int32, num_heads: cutlass.Int32, stream):
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
            sStage: cute.struct.Align[cute.struct.MemRange[self.acc_dtype, self.rows_per_cta * (self.D + 4)], 128]
        self.shared_storage = SharedStorage
        self.kernel(tiled_mma, tma_atom_m, tma_tensor_m, tma_atom_n, tma_tensor_n, g_initial_state_t, g_initial_state_workspace_t, g_fixed_state_t, g_output_state_t, g_state_indices_t, g_cu_seqlens, m_layout, n_layout, chunk_len, total_cp_chunks, num_seqs, num_heads).launch(grid=(num_seqs * num_heads * self.row_ctas, 1, 1), block=(self.threads_per_cta, 1, 1), max_number_threads=(self.threads_per_cta, 1, 1), smem=self.shared_storage.size_in_bytes(), stream=stream, min_blocks_per_mp=1)

class K3OptTmaStore(CPDeltaRuleFixupUtcmmaSm100):

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
                gT = cute.zipped_divide(mT, (self.rows_per_cta, self.D))[(None, None), (row_cta_idx, 0)]
                (tSs, tSg) = cpasync.tma_partition(tma_atom_fs, 0, cute.make_layout(1), cute.group_modes(sStage, 0, 2), cute.group_modes(gT, 0, 2))
                cute.copy(tma_atom_fs, tSs, tSg)
                cute.arch.cp_async_bulk_commit_group()

    @cute.jit
    def run_compute_role(self, tiled_mma, tmem_ptr, sN, n_consumer, mma_ready_producer, mma_done_consumer, gInitialState, gInitialStateWorkspace, gFixedState, gOutputState, chunk_start, num_chunks, tidx, sStage, tma_atom_fs, tma_tensor_fs, row_cta_idx, head_idx):
        tCtAcc = self.make_tmem_tensor(tmem_ptr, tiled_mma)
        tCtOpd = self.make_tmem_operand(tmem_ptr, tiled_mma)
        start = cutlass.Int32(0)
        state_stage = cutlass.Int32(0)
        if cutlass.const_expr(self.needs_initial_state):
            self.copy_tensor_to_acc(tiled_mma, tCtAcc, gInitialState, tidx, self.initial_state_dtype != self.acc_dtype, state_stage)
            if cutlass.const_expr(self.store_initial_state):
                self.store_acc(tiled_mma, tCtAcc, tidx, gInitialStateWorkspace, gOutputState, False, state_stage, sStage, tma_atom_fs, tma_tensor_fs, row_cta_idx, head_idx, cutlass.Int32(0), False)
        else:
            start = cutlass.Int32(1)
            n_init_handle = n_consumer.wait_and_advance()
            self.copy_tensor_to_acc(tiled_mma, tCtAcc, sN[None, None, n_init_handle.index], tidx, False, state_stage)
            self.store_acc(tiled_mma, tCtAcc, tidx, gFixedState[None, None, chunk_start], gOutputState, False, state_stage, sStage, tma_atom_fs, tma_tensor_fs, row_cta_idx, head_idx, chunk_start, True)
            n_init_handle.release()
        for chunk_idx in cutlass.range(start, num_chunks, unroll=1):
            self.convert_acc_to_opd(tiled_mma, tCtAcc, tCtOpd, tidx)
            n_handle = n_consumer.wait_and_advance()
            ready_handle = mma_ready_producer.acquire_and_advance()
            self.copy_tensor_to_acc(tiled_mma, tCtAcc, sN[None, None, n_handle.index], tidx, False, state_stage)
            ready_handle.commit()
            n_handle.release()
            done_handle = mma_done_consumer.wait_and_advance()
            self.store_acc(tiled_mma, tCtAcc, tidx, gFixedState[None, None, chunk_start + chunk_idx], gOutputState, False, state_stage, sStage, tma_atom_fs, tma_tensor_fs, row_cta_idx, head_idx, chunk_start + chunk_idx, True)
            done_handle.release()
        if cutlass.const_expr(self.store_final_state):
            self.store_acc(tiled_mma, tCtAcc, tidx, gFixedState[None, None, chunk_start + num_chunks - 1], gOutputState, True, state_stage, sStage, tma_atom_fs, tma_tensor_fs, row_cta_idx, head_idx, chunk_start + num_chunks - 1, True)
        cute.arch.cp_async_bulk_wait_group(0)

    @cute.kernel
    def kernel(self, tiled_mma, tma_atom_m, tma_tensor_m, tma_atom_n, tma_tensor_n, g_initial_state_t, g_initial_state_workspace_t, g_fixed_state_t, g_output_state_t, g_state_indices_t, g_cu_seqlens, m_layout, n_layout, chunk_len, total_cp_chunks, num_seqs, num_heads, tma_atom_fs, tma_tensor_fs):
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
        sStage = storage.sStage.get_tensor(cute.make_layout((self.rows_per_cta, self.D), stride=(self.D, 1)))
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
            self.run_compute_role(tiled_mma, tmem_ptr, sN_tma, n_consumer, mma_ready_producer, mma_done_consumer, gInitialState, gInitialStateWorkspace, gFixedStateCta, gOutputState, chunk_start, num_chunks, tidx, sStage, tma_atom_fs, tma_tensor_fs, row_cta_idx, head_idx)
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
    def __call__(self, g_transfer_t: cute.Tensor, g_local_state_t: cute.Tensor, g_initial_state_t: cute.Tensor, g_initial_state_workspace_t: cute.Tensor, g_fixed_state_t: cute.Tensor, g_output_state_t: cute.Tensor, g_state_indices_t: cute.Tensor, g_cu_seqlens: cute.Tensor, chunk_len: cutlass.Int32, total_cp_chunks: cutlass.Int32, num_seqs: cutlass.Int32, num_heads: cutlass.Int32, stream):
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
            sStage: cute.struct.Align[cute.struct.MemRange[self.acc_dtype, self.rows_per_cta * self.D], 1024]
        self.shared_storage = SharedStorage
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        g_fs_view = cute.make_tensor(g_fixed_state_t.iterator.align(128), cute.make_layout((self.D, self.D, num_heads, total_cp_chunks), stride=(self.D, 1, self.D * self.D, self.D * self.D * num_heads)))
        fs_smem_layout = cute.make_layout((self.rows_per_cta, self.D), stride=(self.D, 1))
        (tma_atom_fs, tma_tensor_fs) = cpasync.make_tiled_tma_atom(tma_store_op, g_fs_view, fs_smem_layout, (self.rows_per_cta, self.D))
        self.kernel(tiled_mma, tma_atom_m, tma_tensor_m, tma_atom_n, tma_tensor_n, g_initial_state_t, g_initial_state_workspace_t, g_fixed_state_t, g_output_state_t, g_state_indices_t, g_cu_seqlens, m_layout, n_layout, chunk_len, total_cp_chunks, num_seqs, num_heads, tma_atom_fs, tma_tensor_fs).launch(grid=(num_seqs * num_heads * self.row_ctas, 1, 1), block=(self.threads_per_cta, 1, 1), max_number_threads=(self.threads_per_cta, 1, 1), smem=self.shared_storage.size_in_bytes(), stream=stream, min_blocks_per_mp=1)

