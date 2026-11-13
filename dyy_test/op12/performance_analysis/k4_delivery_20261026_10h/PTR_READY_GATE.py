import cutlass
import cutlass.cute as cute
from flashinfer.gdn_kernels.blackwell import gated_delta_net_cp_prefill as _source
globals().update({k: v for (k, v) in vars(_source).items() if not k.startswith('__')})
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm

@dsl_user_op
def _globaltimer(*, loc=None, ip=None):
    result = llvm.inline_asm(cutlass.Int64.mlir_type, [], 'mov.u64 $0, %clock64;', '=l', has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Int64(result)

class K4READY_GATE(CPDeltaRulePrefillTcgen05Sm100):

    @cute.jit
    def __call__(self, q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, gate: cute.Tensor, t: cute.Tensor, o: cute.Tensor, cu_seqlens: cute.Tensor, fixed_state: cute.Tensor, initial_state: Optional[cute.Tensor], state_out: cute.Tensor, cp_chunk_len: cutlass.Int32, total_cp_chunks: cutlass.Int32, max_cp_chunks_per_seq: cutlass.Int32, num_seqs: cutlass.Int32, scale: cutlass.Float32, tensormap_workspace: cute.Tensor, g_trace: cute.Tensor, ready_gate_mask: cutlass.Int32, stream: cuda.CUstream):
        self.b_t = 64
        h_q = q.shape[1]
        h_v = v.shape[1]
        self._setup_attributes()
        if cutlass.const_expr(self.is_GQA):
            h_r = h_q // h_v
            h_qv = h_v
            q = cute.make_tensor(q.iterator, cute.make_layout((q.shape[0], q.shape[2], (h_r, h_v)), stride=(q.stride[0], q.stride[2], (q.stride[1], h_r * q.stride[1]))))
            k = cute.make_tensor(k.iterator, cute.make_layout((k.shape[0], k.shape[2], (h_r, h_v)), stride=(k.stride[0], k.stride[2], (0, k.stride[1]))))
            v = cute.make_tensor(v.iterator, cute.make_layout((v.shape[2], v.shape[0], (h_r, h_v)), stride=(v.stride[2], v.stride[0], (0, v.stride[1]))))
        else:
            h_r = h_v // h_q
            h_qv = h_q
            q = cute.make_tensor(q.iterator, cute.make_layout((q.shape[0], q.shape[2], (h_r, h_q)), stride=(q.stride[0], q.stride[2], (0, q.stride[1]))))
            k = cute.make_tensor(k.iterator, cute.make_layout((k.shape[0], k.shape[2], (h_r, h_q)), stride=(k.stride[0], k.stride[2], (0, k.stride[1]))))
            v = cute.make_tensor(v.iterator, cute.make_layout((v.shape[2], v.shape[0], (h_r, h_q)), stride=(v.stride[2], v.stride[0], (v.stride[1], h_r * v.stride[1]))))
        gate = cute.make_tensor(gate.iterator, cute.make_layout((gate.shape[0], (h_r, h_qv)), stride=(gate.stride[0], (gate.stride[1], h_r * gate.stride[1]))))
        t = cute.make_tensor(t.iterator, cute.make_layout((t.shape[0], t.shape[1], (h_r, h_qv), t.shape[3]), stride=(t.stride[0], t.stride[1], (t.stride[2], h_r * t.stride[2]), t.stride[3])))
        o = cute.make_tensor(o.iterator, cute.make_layout((o.shape[2], o.shape[0], (h_r, h_qv)), stride=(o.stride[2], o.stride[0], (o.stride[1], h_r * o.stride[1]))))
        fixed_state = cute.make_tensor(fixed_state.iterator, cute.make_layout((fixed_state.shape[2], fixed_state.shape[3], (h_r, h_qv), fixed_state.shape[0]), stride=(fixed_state.stride[2], fixed_state.stride[3], (fixed_state.stride[1], h_r * fixed_state.stride[1]), fixed_state.stride[0])))
        if cutlass.const_expr(initial_state is not None):
            initial_state = cute.make_tensor(initial_state.iterator, cute.make_layout((initial_state.shape[2], initial_state.shape[3], (h_r, h_qv), initial_state.shape[0]), stride=(initial_state.stride[2], initial_state.stride[3], (initial_state.stride[1], h_r * initial_state.stride[1]), initial_state.stride[0])))
        if cutlass.const_expr(self.store_final_state):
            state_out = cute.make_tensor(state_out.iterator, cute.make_layout((state_out.shape[2], state_out.shape[3], (h_r, h_qv), state_out.shape[0]), stride=(state_out.stride[2], state_out.stride[3], (state_out.stride[1], h_r * state_out.stride[1]), state_out.stride[0])))
        else:
            state_out = fixed_state

        def _mma_op(mma_tiler, a_major, b_major, OperandSourceA):
            return tcgen05.MmaF16BF16Op(self.io_dtype, self.acc_dtype, (mma_tiler[0], mma_tiler[1], 16), self.cta_group, OperandSourceA, a_major, b_major)
        tiled_mma_qk = cute.make_tiled_mma(_mma_op(self.mma_tiler_qk, OperandMajorMode.K, OperandMajorMode.K, tcgen05.OperandSource.SMEM))
        tiled_mma_qs = cute.make_tiled_mma(_mma_op(self.mma_tiler_qs, OperandMajorMode.K, OperandMajorMode.K, tcgen05.OperandSource.TMEM))
        tiled_mma_qkv = cute.make_tiled_mma(_mma_op(self.mma_tiler_qkv, OperandMajorMode.K, OperandMajorMode.K, tcgen05.OperandSource.TMEM))
        tiled_mma_qkv_ss = cute.make_tiled_mma(_mma_op(self.mma_tiler_qkv, OperandMajorMode.MN, OperandMajorMode.K, tcgen05.OperandSource.SMEM))
        tiled_mma_kv = cute.make_tiled_mma(_mma_op(self.mma_tiler_kv, OperandMajorMode.K, OperandMajorMode.MN, tcgen05.OperandSource.TMEM))
        self.cluster_layout_vmnk = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,))
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        q_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma_qk, self.mma_tiler_qk, self.io_dtype, self.smem_q_stages)
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma_qk, self.mma_tiler_qk, self.io_dtype, self.smem_k_stages)
        k_trans_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma_kv, self.mma_tiler_kv, self.io_dtype, self.smem_k_stages)
        v_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma_qkv_ss, self.mma_tiler_qkv, self.io_dtype, self.smem_v_stages)
        ainv_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma_qkv, self.mma_tiler_qkv, self.io_dtype, self.smem_ainv_stages)
        t_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma_qkv, self.mma_tiler_qkv, self.io_dtype, self.smem_t_stages)
        qk_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma_qkv, self.mma_tiler_qkv, self.io_dtype, self.smem_qk_stages)
        o_smem_layout_staged = sm100_utils.make_smem_layout_epi(self.io_dtype, cutlass.utils.LayoutEnum.from_tensor(o), self.mma_tiler_qkv[:2], self.smem_o_stages)
        cumsumlog_smem_layout_staged = cute.make_layout((self.b_t, 1, self.smem_gate_stages))

        @cute.struct
        class SharedStorage:
            load_k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_k_stages * 2]
            load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_q_stages * 2]
            load_v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_v_stages * 2]
            load_gate_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_gate_stages * 2]
            load_t_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_t_stages * 2]
            q_state_acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.tmem_q_state_acc_stages * 2]
            kv_acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.tmem_kv_acc_stages * 2]
            cg0_shared_acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.tmem_cg0_shared_acc_stages * 2]
            cg1_shared_acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.tmem_cg1_shared_acc_stages * 2]
            ainv_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_ainv_stages * 2]
            qk_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_qk_stages * 2]
            state_inp_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.tmem_state_inp_stages * 2]
            vks_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            nv_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            decay_v_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            o_store_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_o_stages * 2]
            tmem_holding_buf: cutlass.Int32
            sQ: cute.struct.Align[cute.struct.MemRange[self.io_dtype, cute.cosize(q_smem_layout_staged)], self.buffer_align_bytes]
            sK: cute.struct.Align[cute.struct.MemRange[self.io_dtype, cute.cosize(k_smem_layout_staged)], self.buffer_align_bytes]
            sV: cute.struct.Align[cute.struct.MemRange[self.io_dtype, cute.cosize(v_smem_layout_staged)], self.buffer_align_bytes]
            sT: cute.struct.Align[cute.struct.MemRange[self.io_dtype, cute.cosize(t_smem_layout_staged)], self.buffer_align_bytes]
            sAinv: cute.struct.Align[cute.struct.MemRange[self.io_dtype, cute.cosize(ainv_smem_layout_staged)], self.buffer_align_bytes]
            sQk: cute.struct.Align[cute.struct.MemRange[self.io_dtype, cute.cosize(qk_smem_layout_staged)], self.buffer_align_bytes]
            sO: cute.struct.Align[cute.struct.MemRange[self.io_dtype, cute.cosize(o_smem_layout_staged)], self.buffer_align_bytes]
            cumsumlog: cute.struct.MemRange[cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged)]
            cumprod: cute.struct.MemRange[cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged)]
        self.shared_storage = SharedStorage
        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        t_smem_layout = cute.select(t_smem_layout_staged, mode=[0, 1, 2])
        tma_q = _wrap_tma(cute.nvgpu.make_tiled_tma_atom_A(tma_load_op, q, q_smem_layout, self.mma_tiler_qk, tiled_mma_qk, self.cluster_layout_vmnk.shape))
        tma_k = _wrap_tma(cute.nvgpu.make_tiled_tma_atom_B(tma_load_op, k, k_smem_layout, self.mma_tiler_qk, tiled_mma_qk, self.cluster_layout_vmnk.shape))
        tma_v = _wrap_tma(cute.nvgpu.make_tiled_tma_atom_A(tma_load_op, v, v_smem_layout, self.mma_tiler_qkv, tiled_mma_qkv_ss, self.cluster_layout_vmnk.shape))
        tma_t = _wrap_tma(cute.nvgpu.make_tiled_tma_atom_B(tma_load_op, t, t_smem_layout, self.mma_tiler_qkv, tiled_mma_qkv, self.cluster_layout_vmnk.shape))
        o_smem_layout = cute.select(o_smem_layout_staged, mode=[0, 1])
        tma_o = _wrap_tma(cpasync.make_tiled_tma_atom(tma_store_op, o, o_smem_layout, self.mma_tiler_qkv[:2]))
        self.tma_q_bytes = cute.size_in_bytes(self.io_dtype, q_smem_layout)
        self.tma_k_bytes = cute.size_in_bytes(self.io_dtype, k_smem_layout)
        self.tma_v_bytes = cute.size_in_bytes(self.io_dtype, v_smem_layout)
        self.tma_t_bytes = cute.size_in_bytes(self.io_dtype, t_smem_layout)
        self.tma_o_bytes = cute.size_in_bytes(self.io_dtype, o_smem_layout)
        grid_shape = (h_r * h_qv * max_cp_chunks_per_seq, num_seqs, 1)
        self.kernel(tiled_mma_qk, tiled_mma_qs, tiled_mma_qkv, tiled_mma_qkv_ss, tiled_mma_kv, tma_q, tma_k, tma_v, tma_t, gate, tma_o, cu_seqlens, fixed_state, initial_state, state_out, cp_chunk_len, h_r * h_qv, scale, q_smem_layout_staged, k_smem_layout_staged, k_trans_smem_layout_staged, v_smem_layout_staged, cumsumlog_smem_layout_staged, t_smem_layout_staged, ainv_smem_layout_staged, qk_smem_layout_staged, o_smem_layout_staged, q, k, v, t, o, tensormap_workspace, g_trace, ready_gate_mask).launch(grid=grid_shape, block=(self.threads_per_cta, 1, 1), cluster=self.cluster_shape_mnk, smem=self.shared_storage.size_in_bytes(), stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, tiled_mma_qk: cute.TiledMma, tiled_mma_qs: cute.TiledMma, tiled_mma_qkv: cute.TiledMma, tiled_mma_qkv_ss: cute.TiledMma, tiled_mma_kv: cute.TiledMma, tma_q: TmaInfo, tma_k: TmaInfo, tma_v: TmaInfo, tma_t: TmaInfo, mGate: cute.Tensor, tma_o: TmaInfo, cu_seqlens: cute.Tensor, mFixedState: cute.Tensor, mInitialState: Optional[cute.Tensor], mStateOut: cute.Tensor, cp_chunk_len: cutlass.Int32, num_sab_heads: cutlass.Int32, scale: cutlass.Float32, q_smem_layout_staged: cute.ComposedLayout, k_smem_layout_staged: cute.ComposedLayout, k_trans_smem_layout_staged: cute.ComposedLayout, v_smem_layout_staged: cute.ComposedLayout, cumsumlog_smem_layout_staged: cute.Layout, t_smem_layout_staged: cute.ComposedLayout, ainv_smem_layout_staged: cute.ComposedLayout, qk_smem_layout_staged: cute.ComposedLayout, o_smem_layout_staged: cute.ComposedLayout, mQ, mK, mV, mT, mO, tensormap_workspace: cute.Tensor, g_trace: cute.Tensor, ready_gate_mask: cutlass.Int32):
        """
        Process one explicitly mapped CP chunk for one sequence and SAB head.
        """
        (tidx, _, _) = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        (bidx, bidy, bidz) = cute.arch.block_idx()
        grid_dim = cute.arch.grid_dim()
        if cutlass.const_expr(self.needs_initial_state):
            assert mInitialState is not None, 'mInitialState must be provided if needs_initial_state is True'
        else:
            assert mInitialState is None, 'mInitialState must be None if needs_initial_state is False'
        (head_idx, cp_chunk_idx_in_seq, chunk_start, chunk_len, num_cp_chunks, cp_chunk_idx, t_block_start) = self.get_cp_work(cu_seqlens, bidy, bidx, num_sab_heads, cp_chunk_len)
        cta_linear_idx = bidz * grid_dim[1] * grid_dim[0] + bidy * grid_dim[0] + bidx
        tensormap_manager = TensorMapManager(TensorMapUpdateMode.GMEM, self.bytes_per_tensormap)
        tensormap_workspace = self.initialize_workspace(tensormap_workspace, grid_dim)
        tensormap_q_ptr = tensormap_manager.get_tensormap_ptr(tensormap_workspace[cta_linear_idx, 0, None].iterator)
        tensormap_k_ptr = tensormap_manager.get_tensormap_ptr(tensormap_workspace[cta_linear_idx, 1, None].iterator)
        tensormap_v_ptr = tensormap_manager.get_tensormap_ptr(tensormap_workspace[cta_linear_idx, 2, None].iterator)
        tensormap_o_ptr = tensormap_manager.get_tensormap_ptr(tensormap_workspace[cta_linear_idx, 4, None].iterator)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sQ = storage.sQ.get_tensor(q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner)
        sK = storage.sK.get_tensor(k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner)
        sK_trans = storage.sK.get_tensor(k_trans_smem_layout_staged.outer, swizzle=k_trans_smem_layout_staged.inner)
        sV = storage.sV.get_tensor(v_smem_layout_staged.outer, swizzle=v_smem_layout_staged.inner)
        sT = storage.sT.get_tensor(t_smem_layout_staged.outer, swizzle=t_smem_layout_staged.inner)
        sAinv = storage.sAinv.get_tensor(ainv_smem_layout_staged.outer, swizzle=ainv_smem_layout_staged.inner)
        sQk = storage.sQk.get_tensor(qk_smem_layout_staged.outer, swizzle=qk_smem_layout_staged.inner)
        sO = storage.sO.get_tensor(o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner)
        sCumsumlog = storage.cumsumlog.get_tensor(cumsumlog_smem_layout_staged)
        sCumprod = storage.cumprod.get_tensor(cumsumlog_smem_layout_staged)
        if warp_idx == self.mma_warp_id:
            if ready_gate_mask & 4 == 0:
                cpasync.prefetch_descriptor(tma_q.atom)
                cpasync.prefetch_descriptor(tma_k.atom)
                cpasync.prefetch_descriptor(tma_v.atom)
                cpasync.prefetch_descriptor(tma_t.atom)
                cpasync.prefetch_descriptor(tma_o.atom)
        tmem = cutlass.utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=self.tmem_alloc_barrier, allocator_warp_id=self.compute_group_1_warp_ids[0])

        def _cg(num_threads):
            return pipeline.CooperativeGroup(pipeline.Agent.Thread, num_threads)
        cg_tma = _cg(len([self.tma_qkv_warp_id]))
        cg_gate = _cg(self.threads_per_warp * len([self.gate_warp_id]))
        cg_mma = _cg(len([self.mma_warp_id]))
        cg_mma_both = _cg(len([self.mma_warp_id, self.mma_cg1_warp_id]))
        cg_cg0 = _cg(self.threads_per_warp * len(self.compute_group_0_warp_ids))
        cg_cg0_t = _cg(len(self.compute_group_0_warp_ids))
        cg_cg1 = _cg(self.threads_per_warp * len(self.compute_group_1_warp_ids))
        cg_cg1_v = _cg(len(self.compute_group_1_warp_ids))
        cg_both = _cg(self.threads_per_warp * len(self.compute_group_0_warp_ids) + self.threads_per_warp * len(self.compute_group_1_warp_ids))
        cg_epi = _cg(self.threads_per_warp * len([self.epilogue_warp_id]))
        (load_k_producer, load_k_consumer) = pipeline.PipelineTmaUmma.create(num_stages=self.smem_k_stages, producer_group=cg_tma, consumer_group=cg_mma_both, tx_count=self.tma_k_bytes, barrier_storage=storage.load_k_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
        (load_q_producer, load_q_consumer) = pipeline.PipelineTmaUmma.create(num_stages=self.smem_q_stages, producer_group=cg_tma, consumer_group=cg_mma_both, tx_count=self.tma_q_bytes, barrier_storage=storage.load_q_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
        (load_v_producer, load_v_consumer) = pipeline.PipelineTmaAsync.create(num_stages=self.smem_v_stages, producer_group=cg_tma, consumer_group=cg_cg1_v, tx_count=self.tma_v_bytes, barrier_storage=storage.load_v_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
        (load_gate_producer, load_gate_consumer) = pipeline.PipelineAsync.create(num_stages=self.smem_gate_stages, producer_group=cg_gate, consumer_group=cg_both, barrier_storage=storage.load_gate_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
        (load_t_producer, load_t_consumer) = pipeline.PipelineTmaAsync.create(num_stages=self.smem_t_stages, producer_group=cg_tma, consumer_group=cg_cg0_t, tx_count=self.tma_t_bytes, barrier_storage=storage.load_t_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
        (kv_acc_producer, kv_acc_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=self.tmem_kv_acc_stages, producer_group=cg_mma, consumer_group=cg_cg1, barrier_storage=storage.kv_acc_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
        (q_state_acc_producer, q_state_acc_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=self.tmem_q_state_acc_stages, producer_group=cg_mma, consumer_group=cg_cg1, barrier_storage=storage.q_state_acc_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
        (cg0_shared_acc_producer, cg0_shared_acc_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=self.tmem_cg0_shared_acc_stages, producer_group=cg_mma, consumer_group=cg_cg0, barrier_storage=storage.cg0_shared_acc_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
        (cg1_shared_acc_producer, cg1_shared_acc_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=self.tmem_cg1_shared_acc_stages, producer_group=cg_mma, consumer_group=cg_cg1, barrier_storage=storage.cg1_shared_acc_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
        (a_inv_ready_producer, a_inv_ready_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=self.smem_ainv_stages, producer_group=cg_cg0, consumer_group=cg_mma, barrier_storage=storage.ainv_ready_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
        (qk_ready_producer, qk_ready_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=self.smem_qk_stages, producer_group=cg_cg0, consumer_group=cg_mma, barrier_storage=storage.qk_ready_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
        (state_inp_ready_producer, state_inp_ready_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=self.tmem_state_inp_stages, producer_group=cg_cg1, consumer_group=cg_mma, barrier_storage=storage.state_inp_ready_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
        (vks_ready_producer, vks_ready_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_cg1, consumer_group=cg_mma, barrier_storage=storage.vks_ready_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
        (nv_ready_producer, nv_ready_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_cg1, consumer_group=cg_mma, barrier_storage=storage.nv_ready_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
        (decay_v_ready_producer, decay_v_ready_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_cg1, consumer_group=cg_mma, barrier_storage=storage.decay_v_ready_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
        (o_store_producer, o_store_consumer) = pipeline.PipelineAsync.create(num_stages=self.smem_o_stages, producer_group=cg_cg1, consumer_group=cg_epi, barrier_storage=storage.o_store_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
        pipeline_init_arrive(is_relaxed=True)
        pipeline_init_wait()
        num_pairs = 2
        num_pairs_b = cute.ceil_div(chunk_len, self.b_t * num_pairs)
        num_chunks_b = num_pairs_b * num_pairs
        num_valid_chunks_b = cute.ceil_div(chunk_len, self.b_t)
        chunk_end = chunk_start + chunk_len
        if warp_idx >= self.compute_group_0_warp_ids[0] and warp_idx <= self.compute_group_0_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_compute_group_0)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            sQk_pisl = self._transform_to_position_independent_layout(sQk, qk_smem_layout_staged.inner)
            sAinv_pisl = self._transform_to_position_independent_layout(sAinv, ainv_smem_layout_staged.inner)
            for pair_idx in cutlass.range(num_pairs_b):
                (load_gate_consumer, load_t_consumer, cg0_shared_acc_consumer, a_inv_ready_producer, qk_ready_producer) = self.compute_group_0_pair_cp(tidx, tmem_ptr, scale, (tiled_mma_qk,), (sCumsumlog, sT, sAinv_pisl, sQk_pisl), (load_gate_consumer, load_t_consumer, cg0_shared_acc_consumer, a_inv_ready_producer, qk_ready_producer), (pair_idx, num_pairs_b, num_valid_chunks_b, chunk_len))
            a_inv_ready_producer.tail()
            qk_ready_producer.tail()
        if warp_idx >= self.compute_group_1_warp_ids[0] and warp_idx <= self.compute_group_1_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_compute_group_1)
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            if chunk_len > 0:
                if cute.arch.block_idx()[0] == 16 and cute.arch.lane_idx() == 0 and (warp_idx == 4):
                    g_trace[4 * 1024 + 64 * 8 + 0] = _globaltimer()
                    g_trace[4 * 1024 + 64 * 8 + 7] = cutlass.Int64(cute.arch.smid())
                kv_acc_producer = self._load_cp_initial_state(tidx, mFixedState, mInitialState, head_idx, bidy, cp_chunk_idx_in_seq, cp_chunk_idx, tmem_ptr, tiled_mma_kv, kv_acc_producer)
                if cute.arch.block_idx()[0] == 16 and cute.arch.lane_idx() == 0 and (warp_idx == 4):
                    g_trace[4 * 1024 + 64 * 8 + 1] = _globaltimer()
                sV_pisl = self._transform_to_position_independent_layout(sV, v_smem_layout_staged.inner)
                sO_pisl = self._transform_to_position_independent_layout(sO, o_smem_layout_staged.inner)
                checkpoint_offset = 0
                is_first_chunk = True
                for chunk_idx in cutlass.range(num_chunks_b):
                    (load_v_consumer, load_gate_consumer, cg1_shared_acc_consumer, kv_acc_consumer, q_state_acc_consumer, kv_acc_producer, state_inp_ready_producer, vks_ready_producer, nv_ready_producer, decay_v_ready_producer, o_store_producer, checkpoint_offset) = self.compute_group_1_chunk(tidx, tmem_ptr, scale, (tiled_mma_kv, tiled_mma_qs, tiled_mma_qkv), (sV_pisl, sCumsumlog, sCumprod, sCumprod, sO_pisl), (mStateOut, checkpoint_offset, cutlass.Int32(0)), (load_v_consumer, load_gate_consumer, cg1_shared_acc_consumer, kv_acc_consumer, q_state_acc_consumer, kv_acc_producer, state_inp_ready_producer, vks_ready_producer, nv_ready_producer, decay_v_ready_producer, o_store_producer), (chunk_idx, num_pairs_b, head_idx, chunk_len, is_first_chunk))
                    if cute.arch.block_idx()[0] == 16 and cute.arch.lane_idx() == 0 and (warp_idx == 4):
                        g_trace[4 * 1024 + chunk_idx * 8] = _globaltimer()
                        g_trace[4 * 1024 + chunk_idx * 8 + 7] = cutlass.Int64(cute.arch.smid())
                    is_first_chunk = False
                if cp_chunk_idx_in_seq == num_cp_chunks - 1:
                    kv_acc_consumer = self._store_final_state(tidx, mStateOut, None, head_idx, bidy, tmem_ptr, tiled_mma_kv, kv_acc_consumer, chunk_len, mStateOut, cutlass.Int32(0), cutlass.Int32(0))
                else:
                    kv_acc_handle = kv_acc_consumer.wait_and_advance()
                    kv_acc_handle.release()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
            o_store_producer.tail()
            state_inp_ready_producer.tail()
        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            if (chunk_len > 0) & (ready_gate_mask & 4 != 0):
                cute.arch.mbarrier_wait(storage.kv_acc_mbar_ptr.data_ptr(), 0)
                cpasync.prefetch_descriptor(tma_q.atom)
                cpasync.prefetch_descriptor(tma_k.atom)
                cpasync.prefetch_descriptor(tma_v.atom)
                cpasync.prefetch_descriptor(tma_t.atom)
                cpasync.prefetch_descriptor(tma_o.atom)
            for _pair_idx in cutlass.range(num_pairs_b):
                (cg0_shared_acc_producer, load_k_consumer, load_q_consumer) = self.mma_cg0_pair_cp(tmem_ptr, (tiled_mma_qk, tiled_mma_qkv), (sQ, sK), (cg0_shared_acc_producer, load_k_consumer, load_q_consumer))
            cg0_shared_acc_producer.tail()
        elif warp_idx == self.mma_cg1_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            for chunk_idx in cutlass.range(num_chunks_b):
                (cg1_shared_acc_producer, q_state_acc_producer, kv_acc_producer, load_k_consumer, load_q_consumer, a_inv_ready_consumer, qk_ready_consumer, state_inp_ready_consumer, vks_ready_consumer, nv_ready_consumer, decay_v_ready_consumer) = self.mma_cg1_chunk(tmem_ptr, (tiled_mma_qs, tiled_mma_qkv, tiled_mma_qkv_ss, tiled_mma_kv), (sQ, sK, sK_trans, sV, sAinv, sQk), (cg1_shared_acc_producer, q_state_acc_producer, kv_acc_producer, load_k_consumer, load_q_consumer, a_inv_ready_consumer, qk_ready_consumer, state_inp_ready_consumer, vks_ready_consumer, nv_ready_consumer, decay_v_ready_consumer), chunk_idx == 0)
            cg1_shared_acc_producer.tail()
            q_state_acc_producer.tail()
            kv_acc_producer.tail()
        elif warp_idx == self.tma_qkv_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            if (chunk_len > 0) & (ready_gate_mask & 1 != 0):
                cute.arch.mbarrier_wait(storage.kv_acc_mbar_ptr.data_ptr(), 0)
            tensormap_manager.init_tensormap_from_atom(tma_q.atom, tensormap_q_ptr, self.tma_qkv_warp_id)
            tensormap_manager.init_tensormap_from_atom(tma_k.atom, tensormap_k_ptr, self.tma_qkv_warp_id)
            tensormap_manager.init_tensormap_from_atom(tma_v.atom, tensormap_v_ptr, self.tma_qkv_warp_id)
            tensormap_manager.fence_tensormap_initialization()
            bounded_q = cute.make_tensor(mQ.iterator, cute.make_layout((chunk_end, mQ.shape[1], mQ.shape[2]), stride=(mQ.stride[0], mQ.stride[1], mQ.stride[2])))
            bounded_k = cute.make_tensor(mK.iterator, cute.make_layout((chunk_end, mK.shape[1], mK.shape[2]), stride=(mK.stride[0], mK.stride[1], mK.stride[2])))
            bounded_v = cute.make_tensor(mV.iterator, cute.make_layout((mV.shape[0], chunk_end, mV.shape[2]), stride=(mV.stride[0], mV.stride[1], mV.stride[2])))
            tensormap_manager.update_tensormap((bounded_q, bounded_k, bounded_v), (tma_q.atom, tma_k.atom, tma_v.atom), (tensormap_q_ptr, tensormap_k_ptr, tensormap_v_ptr), self.tma_qkv_warp_id, (None, None, None))
            for chunk_idx in cutlass.range(num_chunks_b):
                chunk_offset = chunk_start + chunk_idx * self.b_t
                (load_q_producer, load_k_producer, load_v_producer) = self.tma_qkv_warp((tiled_mma_qk, tiled_mma_qkv_ss, tiled_mma_kv), (tma_q, tma_k, tma_v), (sQ, sK, sV), (load_q_producer, load_k_producer, load_v_producer), (chunk_offset, chunk_idx, bidy, head_idx), (tensormap_manager, tensormap_q_ptr, tensormap_k_ptr, tensormap_v_ptr))
                if cute.arch.block_idx()[0] == 16 and cute.arch.lane_idx() == 0:
                    g_trace[9 * 1024 + chunk_idx * 8] = _globaltimer()
                    g_trace[9 * 1024 + chunk_idx * 8 + 7] = cutlass.Int64(cute.arch.smid())
                t_chunk_idx = chunk_idx
                if chunk_idx >= num_valid_chunks_b:
                    t_chunk_idx = num_valid_chunks_b - 1
                load_t_producer = self.tma_t_warp(tiled_mma_qkv, tma_t, sT, load_t_producer, t_block_start + t_chunk_idx, head_idx)
                if cute.arch.block_idx()[0] == 16 and cute.arch.lane_idx() == 0:
                    g_trace[9 * 1024 + chunk_idx * 8 + 2] = _globaltimer()
                    g_trace[9 * 1024 + chunk_idx * 8 + 7] = cutlass.Int64(cute.arch.smid())
            load_q_producer.tail()
            load_k_producer.tail()
            load_v_producer.tail()
            load_t_producer.tail()
        if warp_idx == self.epilogue_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            if (chunk_len > 0) & (ready_gate_mask & 2 != 0):
                cute.arch.mbarrier_wait(storage.kv_acc_mbar_ptr.data_ptr(), 0)
            tensormap_manager.init_tensormap_from_atom(tma_o.atom, tensormap_o_ptr, self.epilogue_warp_id)
            tensormap_manager.fence_tensormap_initialization()
            bounded_o = cute.make_tensor(mO.iterator, cute.make_layout((mO.shape[0], chunk_end, mO.shape[2]), stride=(mO.stride[0], mO.stride[1], mO.stride[2])))
            tensormap_manager.update_tensormap((bounded_o,), (tma_o.atom,), (tensormap_o_ptr,), self.epilogue_warp_id, (None,))
            tensormap_manager.fence_tensormap_update(tensormap_o_ptr)
            if chunk_len > 0:
                num_gate_prefetch = 4
                for prefetch_idx in range(2):
                    load_gate_producer = self.load_gate_warp(tidx, mGate, (sCumsumlog, sCumprod), load_gate_producer, (chunk_start + prefetch_idx * self.b_t, head_idx, prefetch_idx >= num_valid_chunks_b - 1, chunk_end))
                if num_chunks_b > 2:
                    for prefetch_idx in range(2, num_gate_prefetch):
                        load_gate_producer = self.load_gate_warp(tidx, mGate, (sCumsumlog, sCumprod), load_gate_producer, (chunk_start + prefetch_idx * self.b_t, head_idx, prefetch_idx >= num_valid_chunks_b - 1, chunk_end))
                for chunk_idx in cutlass.range(num_chunks_b):
                    prefetch_idx = chunk_idx + num_gate_prefetch
                    if prefetch_idx < num_chunks_b:
                        load_gate_producer = self.load_gate_warp(tidx, mGate, (sCumsumlog, sCumprod), load_gate_producer, (chunk_start + prefetch_idx * self.b_t, head_idx, prefetch_idx >= num_valid_chunks_b - 1, chunk_end))
                    o_store_consumer = self.epilogue_warp((sO,), (tma_o,), (o_store_consumer,), (head_idx, chunk_start + chunk_idx * self.b_t), (tensormap_manager, tensormap_o_ptr))
            load_gate_producer.tail()
