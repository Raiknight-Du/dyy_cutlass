import flashinfer.gdn_kernels.blackwell.gated_delta_net_cp as _source
globals().update({k:v for k,v in vars(_source).items() if not k.startswith('__')})
@cute.jit
def _stamp(g_trace, stage:cutlass.Int32, endpoint:cutlass.Int32):
    if cute.arch.block_idx()[0]==0:
        if cute.arch.lane_idx()==0:
            wi=cute.arch.warp_idx()
            g_trace[wi,stage,endpoint,0]=cute.arch.clock64()
            g_trace[wi,stage,endpoint,1]=cutlass.Int64(cute.arch.smid())


class K2TraceB1(CPDeltaRuleMNPrecomputeUtcmma1Sm100):

    @cute.jit
    def __call__(self, g_k: cute.Tensor, g_v: cute.Tensor, g_t: cute.Tensor, g_alpha: cute.Tensor, g_transfer_t: cute.Tensor, g_state_t: cute.Tensor, g_cu_seqlens: cute.Tensor, chunk_len: cutlass.Int32, num_k_heads: cutlass.Int32, num_v_heads: cutlass.Int32, num_sab_heads: cutlass.Int32, total_cp_chunks: cutlass.Int32, max_cp_chunks_per_seq: cutlass.Int32, num_seqs: cutlass.Int32, g_trace: cute.Tensor, sample_block: cutlass.Int32, stream):

        def make_mma(mnk, source_a, a_major, b_major):
            op = tcgen05.MmaF16BF16Op(self.dtype, self.acc_dtype, (mnk[0], mnk[1], 16), self.cta_group, source_a, a_major, b_major)
            return cute.make_tiled_mma(op)
        tiled_mma_x = make_mma((self.D, self.BLK, self.BLK), tcgen05.OperandSource.SMEM, OperandMajorMode.MN, OperandMajorMode.K)
        tiled_mma_z = make_mma((self.D, self.BLK, self.D), tcgen05.OperandSource.TMEM, OperandMajorMode.K, OperandMajorMode.K)
        tiled_mma_y = make_mma((self.D, self.BLK, self.D), tcgen05.OperandSource.TMEM, OperandMajorMode.K, OperandMajorMode.K)
        tiled_mma_m_update = make_mma((self.D, self.D, self.BLK), tcgen05.OperandSource.TMEM, OperandMajorMode.K, OperandMajorMode.MN)
        tiled_mma_n_update = make_mma((self.D, self.D, self.BLK), tcgen05.OperandSource.TMEM, OperandMajorMode.K, OperandMajorMode.MN)
        tiled_mma_m_update_ss = make_mma((self.D, self.D, self.BLK), tcgen05.OperandSource.SMEM, OperandMajorMode.MN, OperandMajorMode.MN)
        k_layout = sm100_utils.make_smem_layout_a(tiled_mma_x, (self.D, self.BLK, self.BLK), self.dtype, self.k_stage)
        k_trans_layout = sm100_utils.make_smem_layout_b(tiled_mma_z, (self.D, self.BLK, self.D), self.dtype, self.k_stage)
        v_layout = sm100_utils.make_smem_layout_a(tiled_mma_x, (self.D, self.BLK, self.BLK), self.dtype, self.v_stage)
        t_layout = sm100_utils.make_smem_layout_b(tiled_mma_x, (self.D, self.BLK, self.BLK), self.dtype, self.t_stage)
        x_layout = sm100_utils.make_smem_layout_b(tiled_mma_m_update, (self.D, self.D, self.BLK), self.dtype, self.x_stage)
        k_semantic_layout = self.transform_partitioned_layout(k_layout)
        v_semantic_layout = self.transform_partitioned_layout(v_layout)
        t_semantic_layout = self.transform_partitioned_layout(t_layout)
        alpha_layout = cute.make_layout((self.BLK, AlphaProcessor.NUM_CHANNELS, self.alpha_stage))
        cluster_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        tma_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        k_tma_layout = cute.slice_(k_semantic_layout, (None, None, 0))
        v_tma_layout = cute.slice_(v_semantic_layout, (None, None, 0))
        t_tma_layout = cute.slice_(t_semantic_layout, (None, None, 0))
        tma_k = cpasync.make_tiled_tma_atom(tma_op, g_k, k_tma_layout, (self.D, self.BLK))
        tma_v = cpasync.make_tiled_tma_atom(tma_op, g_v, v_tma_layout, (self.D, self.BLK))
        tma_t = cpasync.make_tiled_tma_atom(tma_op, g_t, t_tma_layout, (self.BLK, self.BLK))
        (tma_atom_k, tma_tensor_k) = tma_k
        (tma_atom_v, tma_tensor_v) = tma_v
        (tma_atom_t, tma_tensor_t) = tma_t
        dtype_bytes = self.dtype.width // 8
        self.tma_k_bytes = cute.size(k_tma_layout) * dtype_bytes
        self.tma_v_bytes = cute.size(v_tma_layout) * dtype_bytes
        self.tma_t_bytes = cute.size(t_tma_layout) * dtype_bytes

        @cute.struct
        class SharedStorage:
            load_k_mbar: cute.struct.MemRange[cutlass.Int64, self.k_stage * 2]
            load_v_mbar: cute.struct.MemRange[cutlass.Int64, self.v_stage * 2]
            load_t_mbar: cute.struct.MemRange[cutlass.Int64, self.t_stage * 2]
            alpha_mbar: cute.struct.MemRange[cutlass.Int64, self.alpha_stage * 2]
            m_init_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            n_init_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            x_acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            x_ready_mbar: cute.struct.MemRange[cutlass.Int64, self.x_stage * 2]
            m_in_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            n_in_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            z_acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            z_ready_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            m_acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            y_acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            y_ready_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            n_acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            done_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: cutlass.Int32
            sK: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(k_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(v_layout)], 1024]
            sT: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(t_layout)], 1024]
            sX: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(x_layout)], 1024]
            sAlpha: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, cute.cosize(alpha_layout)], 16]
        self.shared_storage = SharedStorage
        self.kernel(tiled_mma_x, tiled_mma_z, tiled_mma_y, tiled_mma_m_update, tiled_mma_n_update, tiled_mma_m_update_ss, tma_atom_k, tma_tensor_k, tma_atom_v, tma_tensor_v, tma_atom_t, tma_tensor_t, g_alpha, g_transfer_t, g_state_t, g_cu_seqlens, k_layout, k_trans_layout, v_layout, t_layout, x_layout, alpha_layout, cluster_layout_vmnk, chunk_len, num_k_heads, num_v_heads, num_sab_heads, total_cp_chunks, num_seqs, g_trace, sample_block).launch(grid=(num_sab_heads * max_cp_chunks_per_seq, num_seqs, 1), block=(self.threads_per_cta, 1, 1), cluster=self.cluster_shape_mnk, smem=self.shared_storage.size_in_bytes(), stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, tiled_mma_x, tiled_mma_z, tiled_mma_y, tiled_mma_m_update, tiled_mma_n_update, tiled_mma_m_update_ss, tma_atom_k, tma_tensor_k, tma_atom_v, tma_tensor_v, tma_atom_t, tma_tensor_t, g_alpha, g_transfer_t, g_state_t, g_cu_seqlens, k_layout, k_trans_layout, v_layout, t_layout, x_layout, alpha_layout, cluster_layout_vmnk, chunk_len, num_k_heads, num_v_heads, num_sab_heads, total_cp_chunks, num_seqs, g_trace: cute.Tensor, sample_block: cutlass.Int32):
        (tidx, _, _) = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        (bx, seq_idx, _) = cute.arch.block_idx()
        sab_head_idx = bx % num_sab_heads
        chunk_idx_in_seq = bx // num_sab_heads
        k_head_idx = sab_head_idx * num_k_heads // num_sab_heads
        v_head_idx = sab_head_idx * num_v_heads // num_sab_heads
        seq_start = cutlass.Int32(g_cu_seqlens[seq_idx])
        seq_end = cutlass.Int32(g_cu_seqlens[seq_idx + 1])
        seq_len = seq_end - seq_start
        num_cp_chunks = chunks_for_len(seq_len, chunk_len)
        is_valid_chunk = chunk_idx_in_seq < num_cp_chunks
        tok_offset = seq_start + chunk_idx_in_seq * chunk_len
        cp_chunk_idx = varlen_chunk_idx(seq_idx, seq_start, chunk_idx_in_seq, chunk_len)
        valid_chunk_len = varlen_chunk_valid_len(seq_len, chunk_idx_in_seq, chunk_len)
        num_blocks = cute.ceil_div(valid_chunk_len, self.BLK)
        t_blocks_per_cp_chunk = cute.ceil_div(chunk_len, self.BLK)
        t_block_start = varlen_chunk_idx(seq_idx, seq_start, chunk_idx_in_seq * t_blocks_per_cp_chunk, self.BLK)
        allocator = utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)
        sK = storage.sK.get_tensor(k_layout.outer, swizzle=k_layout.inner)
        sK_trans = storage.sK.get_tensor(k_trans_layout.outer, swizzle=k_trans_layout.inner)
        sV = storage.sV.get_tensor(v_layout.outer, swizzle=v_layout.inner)
        sT = storage.sT.get_tensor(t_layout.outer, swizzle=t_layout.inner)
        sX = storage.sX.get_tensor(x_layout.outer, swizzle=x_layout.inner)
        sK_tma = self.transform_partitioned_tensor_layout(sK)
        sV_tma = self.transform_partitioned_tensor_layout(sV)
        sT_tma = self.transform_partitioned_tensor_layout(sT)
        sAlpha = storage.sAlpha.get_tensor(alpha_layout)

        def cg(num_threads):
            return pipeline.CooperativeGroup(pipeline.Agent.Thread, num_threads)
        cg_tma = cg(1)
        cg_mma = cg(1)
        cg_mma_both = cg(2)
        cg_alpha = cg(32)
        cg_cg0 = cg(128)
        cg_cg1 = cg(128)
        cg_cg1_v = cg(4)
        cg_both = cg(256)
        load_k_pipeline = pipeline.PipelineTmaUmma.create(num_stages=self.k_stage, producer_group=cg_tma, consumer_group=cg_mma_both, tx_count=self.tma_k_bytes, barrier_storage=storage.load_k_mbar.data_ptr(), defer_sync=True, cta_layout_vmnk=cluster_layout_vmnk, mcast_mode_mn=(1, 0))
        load_k_producer = load_k_pipeline.make_producer()
        load_k_transfer_consumer = load_k_pipeline.make_consumer()
        load_k_state_consumer = load_k_pipeline.make_consumer()
        (load_t_producer, load_t_consumer) = pipeline.PipelineTmaUmma.create(num_stages=self.t_stage, producer_group=cg_tma, consumer_group=cg_mma, tx_count=self.tma_t_bytes, barrier_storage=storage.load_t_mbar.data_ptr(), defer_sync=True, cta_layout_vmnk=cluster_layout_vmnk, mcast_mode_mn=(1, 0)).make_participants()
        (load_v_producer, load_v_consumer) = pipeline.PipelineTmaAsync.create(num_stages=self.v_stage, producer_group=cg_tma, consumer_group=cg_cg1_v, tx_count=self.tma_v_bytes, barrier_storage=storage.load_v_mbar.data_ptr(), defer_sync=True, cta_layout_vmnk=cluster_layout_vmnk, mcast_mode_mn=(1, 0), enable_multicast_signaling=False).make_participants()
        alpha_pipeline = pipeline.PipelineAsync.create(num_stages=self.alpha_stage, producer_group=cg_alpha, consumer_group=cg_both, barrier_storage=storage.alpha_mbar.data_ptr(), defer_sync=True)
        alpha_producer = alpha_pipeline.make_producer()
        alpha_m_consumer = alpha_pipeline.make_consumer()
        alpha_n_consumer = alpha_pipeline.make_consumer()
        (m_init_producer, m_init_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_cg0, consumer_group=cg_mma, barrier_storage=storage.m_init_mbar.data_ptr(), defer_sync=True).make_participants()
        (n_init_producer, n_init_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_cg1, consumer_group=cg_mma, barrier_storage=storage.n_init_mbar.data_ptr(), defer_sync=True).make_participants()
        (x_acc_producer, x_acc_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=cg_mma, consumer_group=cg_cg0, barrier_storage=storage.x_acc_mbar.data_ptr(), defer_sync=True).make_participants()
        x_ready_pipeline = pipeline.PipelineAsyncUmma.create(num_stages=self.x_stage, producer_group=cg_cg0, consumer_group=cg_mma_both, barrier_storage=storage.x_ready_mbar.data_ptr(), defer_sync=True)
        x_ready_producer = x_ready_pipeline.make_producer()
        x_ready_m_consumer = x_ready_pipeline.make_consumer()
        x_ready_n_consumer = x_ready_pipeline.make_consumer()
        (m_in_producer, m_in_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_cg0, consumer_group=cg_mma, barrier_storage=storage.m_in_mbar.data_ptr(), defer_sync=True).make_participants()
        (n_in_producer, n_in_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_cg1, consumer_group=cg_mma, barrier_storage=storage.n_in_mbar.data_ptr(), defer_sync=True).make_participants()
        (z_acc_producer, z_acc_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=cg_mma, consumer_group=cg_cg0, barrier_storage=storage.z_acc_mbar.data_ptr(), defer_sync=True).make_participants()
        (z_ready_producer, z_ready_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_cg0, consumer_group=cg_mma, barrier_storage=storage.z_ready_mbar.data_ptr(), defer_sync=True).make_participants()
        (m_acc_producer, m_acc_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=cg_mma, consumer_group=cg_cg0, barrier_storage=storage.m_acc_mbar.data_ptr(), defer_sync=True).make_participants()
        (y_acc_producer, y_acc_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=cg_mma, consumer_group=cg_cg1, barrier_storage=storage.y_acc_mbar.data_ptr(), defer_sync=True).make_participants()
        (y_ready_producer, y_ready_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_cg1, consumer_group=cg_mma, barrier_storage=storage.y_ready_mbar.data_ptr(), defer_sync=True).make_participants()
        (n_acc_producer, n_acc_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=cg_mma, consumer_group=cg_cg1, barrier_storage=storage.n_acc_mbar.data_ptr(), defer_sync=True).make_participants()
        (done_producer, done_consumer) = pipeline.PipelineAsync.create(num_stages=1, producer_group=cg_cg0, consumer_group=cg_cg1, barrier_storage=storage.done_mbar.data_ptr(), defer_sync=True).make_participants()
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)
        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
        tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=self.tmem_alloc_barrier, allocator_warp_id=self.compute_group_1_warp_ids[0])
        tmem.allocate(cute.arch.get_max_tmem_alloc_cols('sm_100'))
        out_layout = cute.make_layout((self.D, self.D, num_sab_heads, total_cp_chunks), stride=(self.D, 1, self.D * self.D, self.D * self.D * num_sab_heads))
        mM = cute.make_tensor(g_transfer_t.iterator, out_layout)
        mN = cute.make_tensor(g_state_t.iterator, out_layout)
        gM = mM[None, None, sab_head_idx, cp_chunk_idx]
        gN = mN[None, None, sab_head_idx, cp_chunk_idx]
        tensor_cta_coord = 0
        tensor_cta_layout = cute.make_layout(1)
        participates_in_tmem = warp_idx <= self.transfer_mma_warp_id or warp_idx == self.state_mma_warp_id
        if not is_valid_chunk:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            if participates_in_tmem:
                tmem.wait_for_alloc()
            if warp_idx == self.compute_group_1_warp_ids[0]:
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tmem.relinquish_alloc_permit()
                tmem.free(tmem_ptr)
        elif is_valid_chunk and warp_idx >= 0 and (warp_idx <= 3):
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_compute_group_0(tidx, tmem_ptr, tiled_mma_x, tiled_mma_z, tiled_mma_m_update, sX, sAlpha, alpha_m_consumer, (m_init_producer, x_acc_consumer, x_ready_producer, m_in_producer, z_acc_consumer, z_ready_producer, m_acc_consumer, done_producer), num_blocks, gM)
        elif is_valid_chunk and warp_idx >= 4 and (warp_idx <= 7):
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_compute_group_1(tidx % 128, tmem_ptr, tiled_mma_x, tiled_mma_y, tiled_mma_n_update, sV_tma, sAlpha, load_v_consumer, alpha_n_consumer, (n_init_producer, n_in_producer, y_acc_consumer, y_ready_producer, n_acc_consumer, done_consumer), num_blocks, gN)
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
        elif is_valid_chunk and warp_idx == self.transfer_mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)
            cpasync.prefetch_descriptor(tma_atom_t)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_transfer_mma_role(tmem_ptr, (tiled_mma_z, tiled_mma_m_update, tiled_mma_m_update_ss), (sK, sK_trans, sX), load_k_transfer_consumer, (m_init_consumer, x_ready_m_consumer, m_in_consumer, z_acc_producer, z_ready_consumer, m_acc_producer), num_blocks)
        elif is_valid_chunk and warp_idx == self.state_mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_state_mma_role(tmem_ptr, (tiled_mma_x, tiled_mma_y, tiled_mma_n_update), (sK, sK_trans, sT, sX), (load_k_state_consumer, load_t_consumer), (n_init_consumer, x_acc_producer, x_ready_n_consumer, n_in_consumer, y_acc_producer, y_ready_consumer, n_acc_producer), num_blocks)
        elif is_valid_chunk and warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            for blk in cutlass.range(num_blocks, unroll=1):
                load_k_producer = self.load_tensor_block_tma_sm100(tma_atom_k, tma_tensor_k, sK_tma, load_k_producer, tok_offset, t_block_start, k_head_idx, blk, False, tensor_cta_coord, tensor_cta_layout)
                load_v_producer = self.load_tensor_block_tma_sm100(tma_atom_v, tma_tensor_v, sV_tma, load_v_producer, tok_offset, t_block_start, v_head_idx, blk, False, tensor_cta_coord, tensor_cta_layout)
                load_t_producer = self.load_tensor_block_tma_sm100(tma_atom_t, tma_tensor_t, sT_tma, load_t_producer, tok_offset, t_block_start, sab_head_idx, blk, True, tensor_cta_coord, tensor_cta_layout)
        elif is_valid_chunk and warp_idx == self.alpha_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            alpha_producer = self.run_load_alpha_role_sm100(g_alpha, sAlpha, alpha_producer, tok_offset, sab_head_idx, valid_chunk_len, num_blocks, num_sab_heads, tidx)
        else:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

class K2TraceCOARSE(CPDeltaRuleMNPrecomputeUtcmma1Sm100):

    @cute.jit
    def __call__(self, g_k: cute.Tensor, g_v: cute.Tensor, g_t: cute.Tensor, g_alpha: cute.Tensor, g_transfer_t: cute.Tensor, g_state_t: cute.Tensor, g_cu_seqlens: cute.Tensor, chunk_len: cutlass.Int32, num_k_heads: cutlass.Int32, num_v_heads: cutlass.Int32, num_sab_heads: cutlass.Int32, total_cp_chunks: cutlass.Int32, max_cp_chunks_per_seq: cutlass.Int32, num_seqs: cutlass.Int32, g_trace: cute.Tensor, sample_block: cutlass.Int32, stream):

        def make_mma(mnk, source_a, a_major, b_major):
            op = tcgen05.MmaF16BF16Op(self.dtype, self.acc_dtype, (mnk[0], mnk[1], 16), self.cta_group, source_a, a_major, b_major)
            return cute.make_tiled_mma(op)
        tiled_mma_x = make_mma((self.D, self.BLK, self.BLK), tcgen05.OperandSource.SMEM, OperandMajorMode.MN, OperandMajorMode.K)
        tiled_mma_z = make_mma((self.D, self.BLK, self.D), tcgen05.OperandSource.TMEM, OperandMajorMode.K, OperandMajorMode.K)
        tiled_mma_y = make_mma((self.D, self.BLK, self.D), tcgen05.OperandSource.TMEM, OperandMajorMode.K, OperandMajorMode.K)
        tiled_mma_m_update = make_mma((self.D, self.D, self.BLK), tcgen05.OperandSource.TMEM, OperandMajorMode.K, OperandMajorMode.MN)
        tiled_mma_n_update = make_mma((self.D, self.D, self.BLK), tcgen05.OperandSource.TMEM, OperandMajorMode.K, OperandMajorMode.MN)
        tiled_mma_m_update_ss = make_mma((self.D, self.D, self.BLK), tcgen05.OperandSource.SMEM, OperandMajorMode.MN, OperandMajorMode.MN)
        k_layout = sm100_utils.make_smem_layout_a(tiled_mma_x, (self.D, self.BLK, self.BLK), self.dtype, self.k_stage)
        k_trans_layout = sm100_utils.make_smem_layout_b(tiled_mma_z, (self.D, self.BLK, self.D), self.dtype, self.k_stage)
        v_layout = sm100_utils.make_smem_layout_a(tiled_mma_x, (self.D, self.BLK, self.BLK), self.dtype, self.v_stage)
        t_layout = sm100_utils.make_smem_layout_b(tiled_mma_x, (self.D, self.BLK, self.BLK), self.dtype, self.t_stage)
        x_layout = sm100_utils.make_smem_layout_b(tiled_mma_m_update, (self.D, self.D, self.BLK), self.dtype, self.x_stage)
        k_semantic_layout = self.transform_partitioned_layout(k_layout)
        v_semantic_layout = self.transform_partitioned_layout(v_layout)
        t_semantic_layout = self.transform_partitioned_layout(t_layout)
        alpha_layout = cute.make_layout((self.BLK, AlphaProcessor.NUM_CHANNELS, self.alpha_stage))
        cluster_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        tma_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        k_tma_layout = cute.slice_(k_semantic_layout, (None, None, 0))
        v_tma_layout = cute.slice_(v_semantic_layout, (None, None, 0))
        t_tma_layout = cute.slice_(t_semantic_layout, (None, None, 0))
        tma_k = cpasync.make_tiled_tma_atom(tma_op, g_k, k_tma_layout, (self.D, self.BLK))
        tma_v = cpasync.make_tiled_tma_atom(tma_op, g_v, v_tma_layout, (self.D, self.BLK))
        tma_t = cpasync.make_tiled_tma_atom(tma_op, g_t, t_tma_layout, (self.BLK, self.BLK))
        (tma_atom_k, tma_tensor_k) = tma_k
        (tma_atom_v, tma_tensor_v) = tma_v
        (tma_atom_t, tma_tensor_t) = tma_t
        dtype_bytes = self.dtype.width // 8
        self.tma_k_bytes = cute.size(k_tma_layout) * dtype_bytes
        self.tma_v_bytes = cute.size(v_tma_layout) * dtype_bytes
        self.tma_t_bytes = cute.size(t_tma_layout) * dtype_bytes

        @cute.struct
        class SharedStorage:
            load_k_mbar: cute.struct.MemRange[cutlass.Int64, self.k_stage * 2]
            load_v_mbar: cute.struct.MemRange[cutlass.Int64, self.v_stage * 2]
            load_t_mbar: cute.struct.MemRange[cutlass.Int64, self.t_stage * 2]
            alpha_mbar: cute.struct.MemRange[cutlass.Int64, self.alpha_stage * 2]
            m_init_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            n_init_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            x_acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            x_ready_mbar: cute.struct.MemRange[cutlass.Int64, self.x_stage * 2]
            m_in_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            n_in_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            z_acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            z_ready_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            m_acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            y_acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            y_ready_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            n_acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            done_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: cutlass.Int32
            sK: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(k_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(v_layout)], 1024]
            sT: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(t_layout)], 1024]
            sX: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(x_layout)], 1024]
            sAlpha: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, cute.cosize(alpha_layout)], 16]
        self.shared_storage = SharedStorage
        self.kernel(tiled_mma_x, tiled_mma_z, tiled_mma_y, tiled_mma_m_update, tiled_mma_n_update, tiled_mma_m_update_ss, tma_atom_k, tma_tensor_k, tma_atom_v, tma_tensor_v, tma_atom_t, tma_tensor_t, g_alpha, g_transfer_t, g_state_t, g_cu_seqlens, k_layout, k_trans_layout, v_layout, t_layout, x_layout, alpha_layout, cluster_layout_vmnk, chunk_len, num_k_heads, num_v_heads, num_sab_heads, total_cp_chunks, num_seqs, g_trace, sample_block).launch(grid=(num_sab_heads * max_cp_chunks_per_seq, num_seqs, 1), block=(self.threads_per_cta, 1, 1), cluster=self.cluster_shape_mnk, smem=self.shared_storage.size_in_bytes(), stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, tiled_mma_x, tiled_mma_z, tiled_mma_y, tiled_mma_m_update, tiled_mma_n_update, tiled_mma_m_update_ss, tma_atom_k, tma_tensor_k, tma_atom_v, tma_tensor_v, tma_atom_t, tma_tensor_t, g_alpha, g_transfer_t, g_state_t, g_cu_seqlens, k_layout, k_trans_layout, v_layout, t_layout, x_layout, alpha_layout, cluster_layout_vmnk, chunk_len, num_k_heads, num_v_heads, num_sab_heads, total_cp_chunks, num_seqs, g_trace: cute.Tensor, sample_block: cutlass.Int32):
        (tidx, _, _) = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        (bx, seq_idx, _) = cute.arch.block_idx()
        sab_head_idx = bx % num_sab_heads
        chunk_idx_in_seq = bx // num_sab_heads
        k_head_idx = sab_head_idx * num_k_heads // num_sab_heads
        v_head_idx = sab_head_idx * num_v_heads // num_sab_heads
        seq_start = cutlass.Int32(g_cu_seqlens[seq_idx])
        seq_end = cutlass.Int32(g_cu_seqlens[seq_idx + 1])
        seq_len = seq_end - seq_start
        num_cp_chunks = chunks_for_len(seq_len, chunk_len)
        is_valid_chunk = chunk_idx_in_seq < num_cp_chunks
        tok_offset = seq_start + chunk_idx_in_seq * chunk_len
        cp_chunk_idx = varlen_chunk_idx(seq_idx, seq_start, chunk_idx_in_seq, chunk_len)
        valid_chunk_len = varlen_chunk_valid_len(seq_len, chunk_idx_in_seq, chunk_len)
        num_blocks = cute.ceil_div(valid_chunk_len, self.BLK)
        t_blocks_per_cp_chunk = cute.ceil_div(chunk_len, self.BLK)
        t_block_start = varlen_chunk_idx(seq_idx, seq_start, chunk_idx_in_seq * t_blocks_per_cp_chunk, self.BLK)
        allocator = utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)
        sK = storage.sK.get_tensor(k_layout.outer, swizzle=k_layout.inner)
        sK_trans = storage.sK.get_tensor(k_trans_layout.outer, swizzle=k_trans_layout.inner)
        sV = storage.sV.get_tensor(v_layout.outer, swizzle=v_layout.inner)
        sT = storage.sT.get_tensor(t_layout.outer, swizzle=t_layout.inner)
        sX = storage.sX.get_tensor(x_layout.outer, swizzle=x_layout.inner)
        sK_tma = self.transform_partitioned_tensor_layout(sK)
        sV_tma = self.transform_partitioned_tensor_layout(sV)
        sT_tma = self.transform_partitioned_tensor_layout(sT)
        sAlpha = storage.sAlpha.get_tensor(alpha_layout)

        def cg(num_threads):
            return pipeline.CooperativeGroup(pipeline.Agent.Thread, num_threads)
        cg_tma = cg(1)
        cg_mma = cg(1)
        cg_mma_both = cg(2)
        cg_alpha = cg(32)
        cg_cg0 = cg(128)
        cg_cg1 = cg(128)
        cg_cg1_v = cg(4)
        cg_both = cg(256)
        load_k_pipeline = pipeline.PipelineTmaUmma.create(num_stages=self.k_stage, producer_group=cg_tma, consumer_group=cg_mma_both, tx_count=self.tma_k_bytes, barrier_storage=storage.load_k_mbar.data_ptr(), defer_sync=True, cta_layout_vmnk=cluster_layout_vmnk, mcast_mode_mn=(1, 0))
        load_k_producer = load_k_pipeline.make_producer()
        load_k_transfer_consumer = load_k_pipeline.make_consumer()
        load_k_state_consumer = load_k_pipeline.make_consumer()
        (load_t_producer, load_t_consumer) = pipeline.PipelineTmaUmma.create(num_stages=self.t_stage, producer_group=cg_tma, consumer_group=cg_mma, tx_count=self.tma_t_bytes, barrier_storage=storage.load_t_mbar.data_ptr(), defer_sync=True, cta_layout_vmnk=cluster_layout_vmnk, mcast_mode_mn=(1, 0)).make_participants()
        (load_v_producer, load_v_consumer) = pipeline.PipelineTmaAsync.create(num_stages=self.v_stage, producer_group=cg_tma, consumer_group=cg_cg1_v, tx_count=self.tma_v_bytes, barrier_storage=storage.load_v_mbar.data_ptr(), defer_sync=True, cta_layout_vmnk=cluster_layout_vmnk, mcast_mode_mn=(1, 0), enable_multicast_signaling=False).make_participants()
        alpha_pipeline = pipeline.PipelineAsync.create(num_stages=self.alpha_stage, producer_group=cg_alpha, consumer_group=cg_both, barrier_storage=storage.alpha_mbar.data_ptr(), defer_sync=True)
        alpha_producer = alpha_pipeline.make_producer()
        alpha_m_consumer = alpha_pipeline.make_consumer()
        alpha_n_consumer = alpha_pipeline.make_consumer()
        (m_init_producer, m_init_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_cg0, consumer_group=cg_mma, barrier_storage=storage.m_init_mbar.data_ptr(), defer_sync=True).make_participants()
        (n_init_producer, n_init_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_cg1, consumer_group=cg_mma, barrier_storage=storage.n_init_mbar.data_ptr(), defer_sync=True).make_participants()
        (x_acc_producer, x_acc_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=cg_mma, consumer_group=cg_cg0, barrier_storage=storage.x_acc_mbar.data_ptr(), defer_sync=True).make_participants()
        x_ready_pipeline = pipeline.PipelineAsyncUmma.create(num_stages=self.x_stage, producer_group=cg_cg0, consumer_group=cg_mma_both, barrier_storage=storage.x_ready_mbar.data_ptr(), defer_sync=True)
        x_ready_producer = x_ready_pipeline.make_producer()
        x_ready_m_consumer = x_ready_pipeline.make_consumer()
        x_ready_n_consumer = x_ready_pipeline.make_consumer()
        (m_in_producer, m_in_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_cg0, consumer_group=cg_mma, barrier_storage=storage.m_in_mbar.data_ptr(), defer_sync=True).make_participants()
        (n_in_producer, n_in_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_cg1, consumer_group=cg_mma, barrier_storage=storage.n_in_mbar.data_ptr(), defer_sync=True).make_participants()
        (z_acc_producer, z_acc_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=cg_mma, consumer_group=cg_cg0, barrier_storage=storage.z_acc_mbar.data_ptr(), defer_sync=True).make_participants()
        (z_ready_producer, z_ready_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_cg0, consumer_group=cg_mma, barrier_storage=storage.z_ready_mbar.data_ptr(), defer_sync=True).make_participants()
        (m_acc_producer, m_acc_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=cg_mma, consumer_group=cg_cg0, barrier_storage=storage.m_acc_mbar.data_ptr(), defer_sync=True).make_participants()
        (y_acc_producer, y_acc_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=cg_mma, consumer_group=cg_cg1, barrier_storage=storage.y_acc_mbar.data_ptr(), defer_sync=True).make_participants()
        (y_ready_producer, y_ready_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_cg1, consumer_group=cg_mma, barrier_storage=storage.y_ready_mbar.data_ptr(), defer_sync=True).make_participants()
        (n_acc_producer, n_acc_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=cg_mma, consumer_group=cg_cg1, barrier_storage=storage.n_acc_mbar.data_ptr(), defer_sync=True).make_participants()
        (done_producer, done_consumer) = pipeline.PipelineAsync.create(num_stages=1, producer_group=cg_cg0, consumer_group=cg_cg1, barrier_storage=storage.done_mbar.data_ptr(), defer_sync=True).make_participants()
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)
        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
        tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=self.tmem_alloc_barrier, allocator_warp_id=self.compute_group_1_warp_ids[0])
        tmem.allocate(cute.arch.get_max_tmem_alloc_cols('sm_100'))
        out_layout = cute.make_layout((self.D, self.D, num_sab_heads, total_cp_chunks), stride=(self.D, 1, self.D * self.D, self.D * self.D * num_sab_heads))
        mM = cute.make_tensor(g_transfer_t.iterator, out_layout)
        mN = cute.make_tensor(g_state_t.iterator, out_layout)
        gM = mM[None, None, sab_head_idx, cp_chunk_idx]
        gN = mN[None, None, sab_head_idx, cp_chunk_idx]
        tensor_cta_coord = 0
        tensor_cta_layout = cute.make_layout(1)
        participates_in_tmem = warp_idx <= self.transfer_mma_warp_id or warp_idx == self.state_mma_warp_id
        if not is_valid_chunk:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            if participates_in_tmem:
                tmem.wait_for_alloc()
            if warp_idx == self.compute_group_1_warp_ids[0]:
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tmem.relinquish_alloc_permit()
                tmem.free(tmem_ptr)
        elif is_valid_chunk and warp_idx >= 0 and (warp_idx <= 3):
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_compute_group_0(tidx, tmem_ptr, tiled_mma_x, tiled_mma_z, tiled_mma_m_update, sX, sAlpha, alpha_m_consumer, (m_init_producer, x_acc_consumer, x_ready_producer, m_in_producer, z_acc_consumer, z_ready_producer, m_acc_consumer, done_producer), num_blocks, gM, g_trace, sample_block)
        elif is_valid_chunk and warp_idx >= 4 and (warp_idx <= 7):
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_compute_group_1(tidx % 128, tmem_ptr, tiled_mma_x, tiled_mma_y, tiled_mma_n_update, sV_tma, sAlpha, load_v_consumer, alpha_n_consumer, (n_init_producer, n_in_producer, y_acc_consumer, y_ready_producer, n_acc_consumer, done_consumer), num_blocks, gN, g_trace, sample_block)
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
        elif is_valid_chunk and warp_idx == self.transfer_mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)
            cpasync.prefetch_descriptor(tma_atom_t)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_transfer_mma_role(tmem_ptr, (tiled_mma_z, tiled_mma_m_update, tiled_mma_m_update_ss), (sK, sK_trans, sX), load_k_transfer_consumer, (m_init_consumer, x_ready_m_consumer, m_in_consumer, z_acc_producer, z_ready_consumer, m_acc_producer), num_blocks, g_trace, sample_block)
        elif is_valid_chunk and warp_idx == self.state_mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_state_mma_role(tmem_ptr, (tiled_mma_x, tiled_mma_y, tiled_mma_n_update), (sK, sK_trans, sT, sX), (load_k_state_consumer, load_t_consumer), (n_init_consumer, x_acc_producer, x_ready_n_consumer, n_in_consumer, y_acc_producer, y_ready_consumer, n_acc_producer), num_blocks, g_trace, sample_block)
        elif is_valid_chunk and warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(0))
            for blk in cutlass.range(num_blocks, unroll=1):
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(0))
                load_k_producer = self.load_tensor_block_tma_sm100(tma_atom_k, tma_tensor_k, sK_tma, load_k_producer, tok_offset, t_block_start, k_head_idx, blk, False, tensor_cta_coord, tensor_cta_layout)
                load_v_producer = self.load_tensor_block_tma_sm100(tma_atom_v, tma_tensor_v, sV_tma, load_v_producer, tok_offset, t_block_start, v_head_idx, blk, False, tensor_cta_coord, tensor_cta_layout)
                load_t_producer = self.load_tensor_block_tma_sm100(tma_atom_t, tma_tensor_t, sT_tma, load_t_producer, tok_offset, t_block_start, sab_head_idx, blk, True, tensor_cta_coord, tensor_cta_layout)
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(1))
            _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(1))
        elif is_valid_chunk and warp_idx == self.alpha_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            alpha_producer = self.run_load_alpha_role_sm100(g_alpha, sAlpha, alpha_producer, tok_offset, sab_head_idx, valid_chunk_len, num_blocks, num_sab_heads, tidx, g_trace, sample_block)
        else:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

    @cute.jit
    def run_compute_group_0(self, tidx: cutlass.Int32, tmem_ptr: cutlass.Int64, tiled_mma_x, tiled_mma_z, tiled_mma_update, sX: cute.Tensor, sAlpha: cute.Tensor, alpha_consumer, math_pipelines: tuple, num_blocks: cutlass.Int32, gM: cute.Tensor, g_trace: cute.Tensor, sample_block: cutlass.Int32):
        (m_init_producer, x_acc_consumer, x_ready_producer, m_in_producer, z_acc_consumer, z_ready_producer, m_acc_consumer, done_producer) = math_pipelines
        _stamp(g_trace, cutlass.Int32(1), cutlass.Int32(0))
        self._initialize_tmem_matrix(tmem_ptr, self.tmem_m_offset, tiled_mma_update, tidx, True)
        m_init_producer.acquire_and_advance().commit()
        _stamp(g_trace, cutlass.Int32(1), cutlass.Int32(1))
        _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(0))
        for blk in cutlass.range(num_blocks, unroll=1):
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(0))
            alpha_handle = alpha_consumer.wait_and_advance()
            x_handle = x_acc_consumer.wait_and_advance()
            x_ready_handle = x_ready_producer.acquire_and_advance()
            self._materialize_x(tmem_ptr, tiled_mma_x, tidx, sX, x_ready_handle.index)
            x_handle.release()
            x_ready_handle.commit()
            if blk > 0:
                self._convert_matrix_to_z_input(tmem_ptr, self.tmem_m_offset, self.tmem_m_inp_offset, tiled_mma_update, tiled_mma_z, tidx)
                m_in_producer.acquire_and_advance().commit()
                z_handle = z_acc_consumer.wait_and_advance()
                self._convert_scratch_to_update_input(tmem_ptr, self.tmem_m_inp_offset, tiled_mma_x, tiled_mma_update, tidx)
                z_handle.release()
                z_ready_producer.acquire_and_advance().commit()
            block_coeff = cutlass.Float32(sAlpha[self.BLK - 1, AlphaProcessor.CUMPROD, alpha_handle.index])
            m_handle = m_acc_consumer.wait_and_advance()
            self._scale_tmem_matrix(tmem_ptr, self.tmem_m_offset, tiled_mma_update, tidx, block_coeff)
            m_handle.release()
            alpha_handle.release()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(1))
        _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(1))
        _stamp(g_trace, cutlass.Int32(2), cutlass.Int32(0))
        self._store_tmem_matrix(tmem_ptr, self.tmem_m_offset, tiled_mma_update, tidx, gM)
        done_producer.acquire_and_advance().commit()
        _stamp(g_trace, cutlass.Int32(2), cutlass.Int32(1))

    @cute.jit
    def run_compute_group_1(self, tidx: cutlass.Int32, tmem_ptr: cutlass.Int64, tiled_mma_x, tiled_mma_z, tiled_mma_update, sV: cute.Tensor, sAlpha: cute.Tensor, v_consumer, alpha_consumer, math_pipelines: tuple, num_blocks: cutlass.Int32, gN: cute.Tensor, g_trace: cute.Tensor, sample_block: cutlass.Int32):
        (n_init_producer, n_in_producer, y_acc_consumer, y_ready_producer, n_acc_consumer, done_consumer) = math_pipelines
        _stamp(g_trace, cutlass.Int32(1), cutlass.Int32(0))
        self._initialize_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, False)
        n_init_producer.acquire_and_advance().commit()
        _stamp(g_trace, cutlass.Int32(1), cutlass.Int32(1))
        _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(0))
        for blk in cutlass.range(num_blocks, unroll=1):
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(0))
            v_handle = v_consumer.wait_and_advance()
            alpha_handle = alpha_consumer.wait_and_advance()
            self._convert_matrix_to_z_input(tmem_ptr, self.tmem_n_offset, self.tmem_n_inp_offset, tiled_mma_update, tiled_mma_z, tidx)
            n_in_producer.acquire_and_advance().commit()
            y_handle = y_acc_consumer.wait_and_advance()
            block_coeff = cutlass.Float32(sAlpha[self.BLK - 1, AlphaProcessor.CUMPROD, alpha_handle.index])
            self._process_y(tmem_ptr, tiled_mma_x, tidx, sV, sAlpha, v_handle.index, alpha_handle.index, block_coeff, tiled_mma_update)
            y_handle.release()
            self._scale_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, block_coeff)
            cute.arch.fence_view_async_tmem_store()
            y_ready_producer.acquire_and_advance().commit()
            n_acc_consumer.wait_and_advance().release()
            v_handle.release()
            alpha_handle.release()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(1))
        _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(1))
        _stamp(g_trace, cutlass.Int32(2), cutlass.Int32(0))
        self._store_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, gN)
        done_consumer.wait_and_advance().release()
        _stamp(g_trace, cutlass.Int32(2), cutlass.Int32(1))

    @cute.jit
    def run_transfer_mma_role(self, tmem_ptr: cutlass.Int64, mma_args: tuple, smem_args: tuple, k_consumer, math_pipelines: tuple, num_blocks: cutlass.Int32, g_trace: cute.Tensor, sample_block: cutlass.Int32):
        (tiled_mma_z, tiled_mma_update, tiled_mma_update_ss) = mma_args
        (sK, sK_trans, sX) = smem_args
        (m_init_consumer, x_ready_consumer, m_in_consumer, z_acc_producer, z_ready_consumer, m_acc_producer) = math_pipelines
        _stamp(g_trace, cutlass.Int32(1), cutlass.Int32(0))
        m_init_consumer.wait_and_advance().release()
        _stamp(g_trace, cutlass.Int32(1), cutlass.Int32(1))
        z_shape = tiled_mma_z.partition_shape_C((self.D, self.BLK))
        tCtScratch_fake = tiled_mma_z.make_fragment_C(cute.append(z_shape, 1))
        tCtScratch = cute.make_tensor(tmem_ptr + self.tmem_scratch_offset, tCtScratch_fake.layout)
        update_shape = tiled_mma_update.partition_shape_C((self.D, self.D))
        tCtUpdate_fake = tiled_mma_update.make_fragment_C(cute.append(update_shape, 1))
        tCtM = cute.make_tensor(tmem_ptr + self.tmem_m_offset, tCtUpdate_fake.layout)
        z_inp_shape = tiled_mma_z.partition_shape_A((self.D, self.D))
        tCtZInp_fake = tiled_mma_z.make_fragment_A(cute.append(z_inp_shape, 1))
        tCtMInp = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_m_inp_offset, dtype=self.dtype), tCtZInp_fake.layout)
        update_inp_shape = tiled_mma_update.partition_shape_A((self.D, self.BLK))
        tCtUpdateInp_fake = tiled_mma_update.make_fragment_A(cute.append(update_inp_shape, 1))
        tCtZInp = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_m_inp_offset, dtype=self.dtype), tCtUpdateInp_fake.layout)
        tCrM_Z = tCtMInp
        tCrK_ZY = tiled_mma_z.make_fragment_B(sK_trans)
        tCrZ_Update = tCtZInp
        tCrX_Update = tiled_mma_update.make_fragment_B(sX)
        tCrK_Update = tiled_mma_update_ss.make_fragment_A(sK)
        _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(0))
        for blk in cutlass.range(num_blocks, unroll=1):
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(0))
            k_handle = k_consumer.wait_and_advance()
            if blk > 0:
                m_in_handle = m_in_consumer.wait_and_advance()
                for kphase in cutlass.range(cute.size(tCrM_Z, mode=[2]), unroll_full=True):
                    tiled_mma_z.set(tcgen05.Field.ACCUMULATE, kphase != 0)
                    cute.gemm(tiled_mma_z, tCtScratch[None, None, None, 0], tCrM_Z[None, None, kphase, 0], tCrK_ZY[None, None, kphase, k_handle.index], tCtScratch[None, None, None, 0])
                z_handle = z_acc_producer.acquire_and_advance()
                z_handle.commit()
                z_ready_consumer.wait_and_advance().release()
                m_in_handle.release()
            x_ready_handle = x_ready_consumer.wait_and_advance()
            m_handle = m_acc_producer.acquire_and_advance()
            for kphase in cutlass.range(cute.size(tCrX_Update, mode=[2]), unroll_full=True):
                tiled_mma_update.set(tcgen05.Field.ACCUMULATE, True)
                if blk == 0:
                    tiled_mma_update_ss.set(tcgen05.Field.ACCUMULATE, True)
                    cute.gemm(tiled_mma_update_ss, tCtM[None, None, None, m_handle.index], tCrK_Update[None, None, kphase, k_handle.index], tCrX_Update[None, None, kphase, x_ready_handle.index], tCtM[None, None, None, m_handle.index])
                else:
                    cute.gemm(tiled_mma_update, tCtM[None, None, None, m_handle.index], tCrZ_Update[None, None, kphase, 0], tCrX_Update[None, None, kphase, x_ready_handle.index], tCtM[None, None, None, m_handle.index])
            m_handle.commit()
            x_ready_handle.release()
            k_handle.release()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(1))
        _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(1))

    @cute.jit
    def run_state_mma_role(self, tmem_ptr: cutlass.Int64, mma_args: tuple, smem_args: tuple, load_pipelines: tuple, math_pipelines: tuple, num_blocks: cutlass.Int32, g_trace: cute.Tensor, sample_block: cutlass.Int32):
        (tiled_mma_x, tiled_mma_y, tiled_mma_update) = mma_args
        (sK, sK_trans, sT, sX) = smem_args
        (k_consumer, t_consumer) = load_pipelines
        (n_init_consumer, x_acc_producer, x_ready_consumer, n_in_consumer, y_acc_producer, y_ready_consumer, n_acc_producer) = math_pipelines
        _stamp(g_trace, cutlass.Int32(1), cutlass.Int32(0))
        n_init_consumer.wait_and_advance().release()
        _stamp(g_trace, cutlass.Int32(1), cutlass.Int32(1))
        x_shape = tiled_mma_x.partition_shape_C((self.D, self.BLK))
        tCtX_fake = tiled_mma_x.make_fragment_C(cute.append(x_shape, 1))
        tCtX = cute.make_tensor(tmem_ptr + self.tmem_x_y_acc_offset, tCtX_fake.layout)
        y_shape = tiled_mma_y.partition_shape_C((self.D, self.BLK))
        tCtY_fake = tiled_mma_y.make_fragment_C(cute.append(y_shape, 1))
        tCtY = cute.make_tensor(tmem_ptr + self.tmem_x_y_acc_offset, tCtY_fake.layout)
        update_shape = tiled_mma_update.partition_shape_C((self.D, self.D))
        tCtUpdate_fake = tiled_mma_update.make_fragment_C(cute.append(update_shape, 1))
        tCtN = cute.make_tensor(tmem_ptr + self.tmem_n_offset, tCtUpdate_fake.layout)
        y_inp_shape = tiled_mma_y.partition_shape_A((self.D, self.D))
        tCtYInp_fake = tiled_mma_y.make_fragment_A(cute.append(y_inp_shape, 1))
        tCtNInp = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_n_inp_offset, dtype=self.dtype), tCtYInp_fake.layout)
        update_inp_shape = tiled_mma_update.partition_shape_A((self.D, self.BLK))
        tCtUpdateInp_fake = tiled_mma_update.make_fragment_A(cute.append(update_inp_shape, 1))
        tCtYInp = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_n_inp_offset, dtype=self.dtype), tCtUpdateInp_fake.layout)
        tCrK_X = tiled_mma_x.make_fragment_A(sK)
        tCrT_X = tiled_mma_x.make_fragment_B(sT)
        tCrN_Y = tCtNInp
        tCrK_Y = tiled_mma_y.make_fragment_B(sK_trans)
        tCrY_Update = tCtYInp
        tCrX_Update = tiled_mma_update.make_fragment_B(sX)
        _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(0))
        for blk in cutlass.range(num_blocks, unroll=1):
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(0))
            k_handle = k_consumer.wait_and_advance()
            t_handle = t_consumer.wait_and_advance()
            x_handle = x_acc_producer.acquire_and_advance()
            for kphase in cutlass.range(cute.size(tCrK_X, mode=[2]), unroll_full=True):
                tiled_mma_x.set(tcgen05.Field.ACCUMULATE, kphase != 0)
                cute.gemm(tiled_mma_x, tCtX[None, None, None, x_handle.index], tCrK_X[None, None, kphase, k_handle.index], tCrT_X[None, None, kphase, t_handle.index], tCtX[None, None, None, x_handle.index])
            x_handle.commit()
            t_handle.release()
            x_ready_handle = x_ready_consumer.wait_and_advance()
            n_in_handle = n_in_consumer.wait_and_advance()
            y_handle = y_acc_producer.acquire_and_advance()
            for kphase in cutlass.range(cute.size(tCrN_Y, mode=[2]), unroll_full=True):
                tiled_mma_y.set(tcgen05.Field.ACCUMULATE, kphase != 0)
                cute.gemm(tiled_mma_y, tCtY[None, None, None, y_handle.index], tCrN_Y[None, None, kphase, 0], tCrK_Y[None, None, kphase, k_handle.index], tCtY[None, None, None, y_handle.index])
            y_handle.commit()
            n_in_handle.release()
            k_handle.release()
            y_ready_consumer.wait_and_advance().release()
            n_handle = n_acc_producer.acquire_and_advance()
            for kphase in cutlass.range(cute.size(tCrX_Update, mode=[2]), unroll_full=True):
                tiled_mma_update.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(tiled_mma_update, tCtN[None, None, None, n_handle.index], tCrY_Update[None, None, kphase, 0], tCrX_Update[None, None, kphase, x_ready_handle.index], tCtN[None, None, None, n_handle.index])
            n_handle.commit()
            x_ready_handle.release()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(1))
        _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(1))

    @cute.jit
    def run_load_alpha_role_sm100(self, g_alpha, sAlpha, alpha_producer, tok_offset, head_idx, chunk_len, num_blocks, num_heads, tidx, g_trace: cute.Tensor, sample_block: cutlass.Int32):
        _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(0))
        for blk in cutlass.range(num_blocks, unroll=1):
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(0))
            handle = alpha_producer.acquire_and_advance()
            cute.arch.fence_view_async_shared()
            self.load_alpha_block(g_alpha, sAlpha, tok_offset, head_idx, blk, chunk_len, num_heads, handle.index, tidx)
            AlphaProcessor().run(sAlpha[None, None, handle.index], cutlass.Float32(1.0), True)
            self.mask_alpha_oob_block(sAlpha, blk, chunk_len, handle.index, tidx)
            cute.arch.fence_view_async_shared()
            handle.commit()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(1))
        _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(1))
        return alpha_producer

class K2TraceFINE(CPDeltaRuleMNPrecomputeUtcmma1Sm100):

    @cute.jit
    def __call__(self, g_k: cute.Tensor, g_v: cute.Tensor, g_t: cute.Tensor, g_alpha: cute.Tensor, g_transfer_t: cute.Tensor, g_state_t: cute.Tensor, g_cu_seqlens: cute.Tensor, chunk_len: cutlass.Int32, num_k_heads: cutlass.Int32, num_v_heads: cutlass.Int32, num_sab_heads: cutlass.Int32, total_cp_chunks: cutlass.Int32, max_cp_chunks_per_seq: cutlass.Int32, num_seqs: cutlass.Int32, g_trace: cute.Tensor, sample_block: cutlass.Int32, stream):

        def make_mma(mnk, source_a, a_major, b_major):
            op = tcgen05.MmaF16BF16Op(self.dtype, self.acc_dtype, (mnk[0], mnk[1], 16), self.cta_group, source_a, a_major, b_major)
            return cute.make_tiled_mma(op)
        tiled_mma_x = make_mma((self.D, self.BLK, self.BLK), tcgen05.OperandSource.SMEM, OperandMajorMode.MN, OperandMajorMode.K)
        tiled_mma_z = make_mma((self.D, self.BLK, self.D), tcgen05.OperandSource.TMEM, OperandMajorMode.K, OperandMajorMode.K)
        tiled_mma_y = make_mma((self.D, self.BLK, self.D), tcgen05.OperandSource.TMEM, OperandMajorMode.K, OperandMajorMode.K)
        tiled_mma_m_update = make_mma((self.D, self.D, self.BLK), tcgen05.OperandSource.TMEM, OperandMajorMode.K, OperandMajorMode.MN)
        tiled_mma_n_update = make_mma((self.D, self.D, self.BLK), tcgen05.OperandSource.TMEM, OperandMajorMode.K, OperandMajorMode.MN)
        tiled_mma_m_update_ss = make_mma((self.D, self.D, self.BLK), tcgen05.OperandSource.SMEM, OperandMajorMode.MN, OperandMajorMode.MN)
        k_layout = sm100_utils.make_smem_layout_a(tiled_mma_x, (self.D, self.BLK, self.BLK), self.dtype, self.k_stage)
        k_trans_layout = sm100_utils.make_smem_layout_b(tiled_mma_z, (self.D, self.BLK, self.D), self.dtype, self.k_stage)
        v_layout = sm100_utils.make_smem_layout_a(tiled_mma_x, (self.D, self.BLK, self.BLK), self.dtype, self.v_stage)
        t_layout = sm100_utils.make_smem_layout_b(tiled_mma_x, (self.D, self.BLK, self.BLK), self.dtype, self.t_stage)
        x_layout = sm100_utils.make_smem_layout_b(tiled_mma_m_update, (self.D, self.D, self.BLK), self.dtype, self.x_stage)
        k_semantic_layout = self.transform_partitioned_layout(k_layout)
        v_semantic_layout = self.transform_partitioned_layout(v_layout)
        t_semantic_layout = self.transform_partitioned_layout(t_layout)
        alpha_layout = cute.make_layout((self.BLK, AlphaProcessor.NUM_CHANNELS, self.alpha_stage))
        cluster_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        tma_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        k_tma_layout = cute.slice_(k_semantic_layout, (None, None, 0))
        v_tma_layout = cute.slice_(v_semantic_layout, (None, None, 0))
        t_tma_layout = cute.slice_(t_semantic_layout, (None, None, 0))
        tma_k = cpasync.make_tiled_tma_atom(tma_op, g_k, k_tma_layout, (self.D, self.BLK))
        tma_v = cpasync.make_tiled_tma_atom(tma_op, g_v, v_tma_layout, (self.D, self.BLK))
        tma_t = cpasync.make_tiled_tma_atom(tma_op, g_t, t_tma_layout, (self.BLK, self.BLK))
        (tma_atom_k, tma_tensor_k) = tma_k
        (tma_atom_v, tma_tensor_v) = tma_v
        (tma_atom_t, tma_tensor_t) = tma_t
        dtype_bytes = self.dtype.width // 8
        self.tma_k_bytes = cute.size(k_tma_layout) * dtype_bytes
        self.tma_v_bytes = cute.size(v_tma_layout) * dtype_bytes
        self.tma_t_bytes = cute.size(t_tma_layout) * dtype_bytes

        @cute.struct
        class SharedStorage:
            load_k_mbar: cute.struct.MemRange[cutlass.Int64, self.k_stage * 2]
            load_v_mbar: cute.struct.MemRange[cutlass.Int64, self.v_stage * 2]
            load_t_mbar: cute.struct.MemRange[cutlass.Int64, self.t_stage * 2]
            alpha_mbar: cute.struct.MemRange[cutlass.Int64, self.alpha_stage * 2]
            m_init_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            n_init_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            x_acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            x_ready_mbar: cute.struct.MemRange[cutlass.Int64, self.x_stage * 2]
            m_in_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            n_in_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            z_acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            z_ready_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            m_acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            y_acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            y_ready_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            n_acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            done_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: cutlass.Int32
            sK: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(k_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(v_layout)], 1024]
            sT: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(t_layout)], 1024]
            sX: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(x_layout)], 1024]
            sAlpha: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, cute.cosize(alpha_layout)], 16]
        self.shared_storage = SharedStorage
        self.kernel(tiled_mma_x, tiled_mma_z, tiled_mma_y, tiled_mma_m_update, tiled_mma_n_update, tiled_mma_m_update_ss, tma_atom_k, tma_tensor_k, tma_atom_v, tma_tensor_v, tma_atom_t, tma_tensor_t, g_alpha, g_transfer_t, g_state_t, g_cu_seqlens, k_layout, k_trans_layout, v_layout, t_layout, x_layout, alpha_layout, cluster_layout_vmnk, chunk_len, num_k_heads, num_v_heads, num_sab_heads, total_cp_chunks, num_seqs, g_trace, sample_block).launch(grid=(num_sab_heads * max_cp_chunks_per_seq, num_seqs, 1), block=(self.threads_per_cta, 1, 1), cluster=self.cluster_shape_mnk, smem=self.shared_storage.size_in_bytes(), stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, tiled_mma_x, tiled_mma_z, tiled_mma_y, tiled_mma_m_update, tiled_mma_n_update, tiled_mma_m_update_ss, tma_atom_k, tma_tensor_k, tma_atom_v, tma_tensor_v, tma_atom_t, tma_tensor_t, g_alpha, g_transfer_t, g_state_t, g_cu_seqlens, k_layout, k_trans_layout, v_layout, t_layout, x_layout, alpha_layout, cluster_layout_vmnk, chunk_len, num_k_heads, num_v_heads, num_sab_heads, total_cp_chunks, num_seqs, g_trace: cute.Tensor, sample_block: cutlass.Int32):
        (tidx, _, _) = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        (bx, seq_idx, _) = cute.arch.block_idx()
        sab_head_idx = bx % num_sab_heads
        chunk_idx_in_seq = bx // num_sab_heads
        k_head_idx = sab_head_idx * num_k_heads // num_sab_heads
        v_head_idx = sab_head_idx * num_v_heads // num_sab_heads
        seq_start = cutlass.Int32(g_cu_seqlens[seq_idx])
        seq_end = cutlass.Int32(g_cu_seqlens[seq_idx + 1])
        seq_len = seq_end - seq_start
        num_cp_chunks = chunks_for_len(seq_len, chunk_len)
        is_valid_chunk = chunk_idx_in_seq < num_cp_chunks
        tok_offset = seq_start + chunk_idx_in_seq * chunk_len
        cp_chunk_idx = varlen_chunk_idx(seq_idx, seq_start, chunk_idx_in_seq, chunk_len)
        valid_chunk_len = varlen_chunk_valid_len(seq_len, chunk_idx_in_seq, chunk_len)
        num_blocks = cute.ceil_div(valid_chunk_len, self.BLK)
        t_blocks_per_cp_chunk = cute.ceil_div(chunk_len, self.BLK)
        t_block_start = varlen_chunk_idx(seq_idx, seq_start, chunk_idx_in_seq * t_blocks_per_cp_chunk, self.BLK)
        allocator = utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)
        sK = storage.sK.get_tensor(k_layout.outer, swizzle=k_layout.inner)
        sK_trans = storage.sK.get_tensor(k_trans_layout.outer, swizzle=k_trans_layout.inner)
        sV = storage.sV.get_tensor(v_layout.outer, swizzle=v_layout.inner)
        sT = storage.sT.get_tensor(t_layout.outer, swizzle=t_layout.inner)
        sX = storage.sX.get_tensor(x_layout.outer, swizzle=x_layout.inner)
        sK_tma = self.transform_partitioned_tensor_layout(sK)
        sV_tma = self.transform_partitioned_tensor_layout(sV)
        sT_tma = self.transform_partitioned_tensor_layout(sT)
        sAlpha = storage.sAlpha.get_tensor(alpha_layout)

        def cg(num_threads):
            return pipeline.CooperativeGroup(pipeline.Agent.Thread, num_threads)
        cg_tma = cg(1)
        cg_mma = cg(1)
        cg_mma_both = cg(2)
        cg_alpha = cg(32)
        cg_cg0 = cg(128)
        cg_cg1 = cg(128)
        cg_cg1_v = cg(4)
        cg_both = cg(256)
        load_k_pipeline = pipeline.PipelineTmaUmma.create(num_stages=self.k_stage, producer_group=cg_tma, consumer_group=cg_mma_both, tx_count=self.tma_k_bytes, barrier_storage=storage.load_k_mbar.data_ptr(), defer_sync=True, cta_layout_vmnk=cluster_layout_vmnk, mcast_mode_mn=(1, 0))
        load_k_producer = load_k_pipeline.make_producer()
        load_k_transfer_consumer = load_k_pipeline.make_consumer()
        load_k_state_consumer = load_k_pipeline.make_consumer()
        (load_t_producer, load_t_consumer) = pipeline.PipelineTmaUmma.create(num_stages=self.t_stage, producer_group=cg_tma, consumer_group=cg_mma, tx_count=self.tma_t_bytes, barrier_storage=storage.load_t_mbar.data_ptr(), defer_sync=True, cta_layout_vmnk=cluster_layout_vmnk, mcast_mode_mn=(1, 0)).make_participants()
        (load_v_producer, load_v_consumer) = pipeline.PipelineTmaAsync.create(num_stages=self.v_stage, producer_group=cg_tma, consumer_group=cg_cg1_v, tx_count=self.tma_v_bytes, barrier_storage=storage.load_v_mbar.data_ptr(), defer_sync=True, cta_layout_vmnk=cluster_layout_vmnk, mcast_mode_mn=(1, 0), enable_multicast_signaling=False).make_participants()
        alpha_pipeline = pipeline.PipelineAsync.create(num_stages=self.alpha_stage, producer_group=cg_alpha, consumer_group=cg_both, barrier_storage=storage.alpha_mbar.data_ptr(), defer_sync=True)
        alpha_producer = alpha_pipeline.make_producer()
        alpha_m_consumer = alpha_pipeline.make_consumer()
        alpha_n_consumer = alpha_pipeline.make_consumer()
        (m_init_producer, m_init_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_cg0, consumer_group=cg_mma, barrier_storage=storage.m_init_mbar.data_ptr(), defer_sync=True).make_participants()
        (n_init_producer, n_init_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_cg1, consumer_group=cg_mma, barrier_storage=storage.n_init_mbar.data_ptr(), defer_sync=True).make_participants()
        (x_acc_producer, x_acc_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=cg_mma, consumer_group=cg_cg0, barrier_storage=storage.x_acc_mbar.data_ptr(), defer_sync=True).make_participants()
        x_ready_pipeline = pipeline.PipelineAsyncUmma.create(num_stages=self.x_stage, producer_group=cg_cg0, consumer_group=cg_mma_both, barrier_storage=storage.x_ready_mbar.data_ptr(), defer_sync=True)
        x_ready_producer = x_ready_pipeline.make_producer()
        x_ready_m_consumer = x_ready_pipeline.make_consumer()
        x_ready_n_consumer = x_ready_pipeline.make_consumer()
        (m_in_producer, m_in_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_cg0, consumer_group=cg_mma, barrier_storage=storage.m_in_mbar.data_ptr(), defer_sync=True).make_participants()
        (n_in_producer, n_in_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_cg1, consumer_group=cg_mma, barrier_storage=storage.n_in_mbar.data_ptr(), defer_sync=True).make_participants()
        (z_acc_producer, z_acc_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=cg_mma, consumer_group=cg_cg0, barrier_storage=storage.z_acc_mbar.data_ptr(), defer_sync=True).make_participants()
        (z_ready_producer, z_ready_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_cg0, consumer_group=cg_mma, barrier_storage=storage.z_ready_mbar.data_ptr(), defer_sync=True).make_participants()
        (m_acc_producer, m_acc_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=cg_mma, consumer_group=cg_cg0, barrier_storage=storage.m_acc_mbar.data_ptr(), defer_sync=True).make_participants()
        (y_acc_producer, y_acc_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=cg_mma, consumer_group=cg_cg1, barrier_storage=storage.y_acc_mbar.data_ptr(), defer_sync=True).make_participants()
        (y_ready_producer, y_ready_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=cg_cg1, consumer_group=cg_mma, barrier_storage=storage.y_ready_mbar.data_ptr(), defer_sync=True).make_participants()
        (n_acc_producer, n_acc_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=cg_mma, consumer_group=cg_cg1, barrier_storage=storage.n_acc_mbar.data_ptr(), defer_sync=True).make_participants()
        (done_producer, done_consumer) = pipeline.PipelineAsync.create(num_stages=1, producer_group=cg_cg0, consumer_group=cg_cg1, barrier_storage=storage.done_mbar.data_ptr(), defer_sync=True).make_participants()
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)
        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
        tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=self.tmem_alloc_barrier, allocator_warp_id=self.compute_group_1_warp_ids[0])
        tmem.allocate(cute.arch.get_max_tmem_alloc_cols('sm_100'))
        out_layout = cute.make_layout((self.D, self.D, num_sab_heads, total_cp_chunks), stride=(self.D, 1, self.D * self.D, self.D * self.D * num_sab_heads))
        mM = cute.make_tensor(g_transfer_t.iterator, out_layout)
        mN = cute.make_tensor(g_state_t.iterator, out_layout)
        gM = mM[None, None, sab_head_idx, cp_chunk_idx]
        gN = mN[None, None, sab_head_idx, cp_chunk_idx]
        tensor_cta_coord = 0
        tensor_cta_layout = cute.make_layout(1)
        participates_in_tmem = warp_idx <= self.transfer_mma_warp_id or warp_idx == self.state_mma_warp_id
        if not is_valid_chunk:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            if participates_in_tmem:
                tmem.wait_for_alloc()
            if warp_idx == self.compute_group_1_warp_ids[0]:
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tmem.relinquish_alloc_permit()
                tmem.free(tmem_ptr)
        elif is_valid_chunk and warp_idx >= 0 and (warp_idx <= 3):
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_compute_group_0(tidx, tmem_ptr, tiled_mma_x, tiled_mma_z, tiled_mma_m_update, sX, sAlpha, alpha_m_consumer, (m_init_producer, x_acc_consumer, x_ready_producer, m_in_producer, z_acc_consumer, z_ready_producer, m_acc_consumer, done_producer), num_blocks, gM, g_trace, sample_block)
        elif is_valid_chunk and warp_idx >= 4 and (warp_idx <= 7):
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_compute_group_1(tidx % 128, tmem_ptr, tiled_mma_x, tiled_mma_y, tiled_mma_n_update, sV_tma, sAlpha, load_v_consumer, alpha_n_consumer, (n_init_producer, n_in_producer, y_acc_consumer, y_ready_producer, n_acc_consumer, done_consumer), num_blocks, gN, g_trace, sample_block)
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
        elif is_valid_chunk and warp_idx == self.transfer_mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)
            cpasync.prefetch_descriptor(tma_atom_t)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_transfer_mma_role(tmem_ptr, (tiled_mma_z, tiled_mma_m_update, tiled_mma_m_update_ss), (sK, sK_trans, sX), load_k_transfer_consumer, (m_init_consumer, x_ready_m_consumer, m_in_consumer, z_acc_producer, z_ready_consumer, m_acc_producer), num_blocks, g_trace, sample_block)
        elif is_valid_chunk and warp_idx == self.state_mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_state_mma_role(tmem_ptr, (tiled_mma_x, tiled_mma_y, tiled_mma_n_update), (sK, sK_trans, sT, sX), (load_k_state_consumer, load_t_consumer), (n_init_consumer, x_acc_producer, x_ready_n_consumer, n_in_consumer, y_acc_producer, y_ready_consumer, n_acc_producer), num_blocks, g_trace, sample_block)
        elif is_valid_chunk and warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(0))
            for blk in cutlass.range(num_blocks, unroll=1):
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(0))
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(28), cutlass.Int32(0))
                load_k_producer = self.load_tensor_block_tma_sm100(tma_atom_k, tma_tensor_k, sK_tma, load_k_producer, tok_offset, t_block_start, k_head_idx, blk, False, tensor_cta_coord, tensor_cta_layout)
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(28), cutlass.Int32(1))
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(29), cutlass.Int32(0))
                load_v_producer = self.load_tensor_block_tma_sm100(tma_atom_v, tma_tensor_v, sV_tma, load_v_producer, tok_offset, t_block_start, v_head_idx, blk, False, tensor_cta_coord, tensor_cta_layout)
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(29), cutlass.Int32(1))
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(30), cutlass.Int32(0))
                load_t_producer = self.load_tensor_block_tma_sm100(tma_atom_t, tma_tensor_t, sT_tma, load_t_producer, tok_offset, t_block_start, sab_head_idx, blk, True, tensor_cta_coord, tensor_cta_layout)
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(30), cutlass.Int32(1))
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(1))
            _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(1))
        elif is_valid_chunk and warp_idx == self.alpha_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            alpha_producer = self.run_load_alpha_role_sm100(g_alpha, sAlpha, alpha_producer, tok_offset, sab_head_idx, valid_chunk_len, num_blocks, num_sab_heads, tidx, g_trace, sample_block)
        else:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

    @cute.jit
    def run_compute_group_0(self, tidx: cutlass.Int32, tmem_ptr: cutlass.Int64, tiled_mma_x, tiled_mma_z, tiled_mma_update, sX: cute.Tensor, sAlpha: cute.Tensor, alpha_consumer, math_pipelines: tuple, num_blocks: cutlass.Int32, gM: cute.Tensor, g_trace: cute.Tensor, sample_block: cutlass.Int32):
        (m_init_producer, x_acc_consumer, x_ready_producer, m_in_producer, z_acc_consumer, z_ready_producer, m_acc_consumer, done_producer) = math_pipelines
        _stamp(g_trace, cutlass.Int32(1), cutlass.Int32(0))
        self._initialize_tmem_matrix(tmem_ptr, self.tmem_m_offset, tiled_mma_update, tidx, True)
        m_init_producer.acquire_and_advance().commit()
        _stamp(g_trace, cutlass.Int32(1), cutlass.Int32(1))
        _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(0))
        for blk in cutlass.range(num_blocks, unroll=1):
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(0))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(4), cutlass.Int32(0))
            alpha_handle = alpha_consumer.wait_and_advance()
            x_handle = x_acc_consumer.wait_and_advance()
            x_ready_handle = x_ready_producer.acquire_and_advance()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(4), cutlass.Int32(1))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(5), cutlass.Int32(0))
            self._materialize_x(tmem_ptr, tiled_mma_x, tidx, sX, x_ready_handle.index)
            x_handle.release()
            x_ready_handle.commit()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(5), cutlass.Int32(1))
            if blk > 0:
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(6), cutlass.Int32(0))
                self._convert_matrix_to_z_input(tmem_ptr, self.tmem_m_offset, self.tmem_m_inp_offset, tiled_mma_update, tiled_mma_z, tidx)
                m_in_producer.acquire_and_advance().commit()
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(6), cutlass.Int32(1))
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(7), cutlass.Int32(0))
                z_handle = z_acc_consumer.wait_and_advance()
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(7), cutlass.Int32(1))
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(8), cutlass.Int32(0))
                self._convert_scratch_to_update_input(tmem_ptr, self.tmem_m_inp_offset, tiled_mma_x, tiled_mma_update, tidx)
                z_handle.release()
                z_ready_producer.acquire_and_advance().commit()
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(8), cutlass.Int32(1))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(9), cutlass.Int32(0))
            block_coeff = cutlass.Float32(sAlpha[self.BLK - 1, AlphaProcessor.CUMPROD, alpha_handle.index])
            m_handle = m_acc_consumer.wait_and_advance()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(9), cutlass.Int32(1))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(10), cutlass.Int32(0))
            self._scale_tmem_matrix(tmem_ptr, self.tmem_m_offset, tiled_mma_update, tidx, block_coeff)
            m_handle.release()
            alpha_handle.release()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(10), cutlass.Int32(1))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(1))
        _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(1))
        _stamp(g_trace, cutlass.Int32(2), cutlass.Int32(0))
        self._store_tmem_matrix(tmem_ptr, self.tmem_m_offset, tiled_mma_update, tidx, gM)
        done_producer.acquire_and_advance().commit()
        _stamp(g_trace, cutlass.Int32(2), cutlass.Int32(1))

    @cute.jit
    def run_compute_group_1(self, tidx: cutlass.Int32, tmem_ptr: cutlass.Int64, tiled_mma_x, tiled_mma_z, tiled_mma_update, sV: cute.Tensor, sAlpha: cute.Tensor, v_consumer, alpha_consumer, math_pipelines: tuple, num_blocks: cutlass.Int32, gN: cute.Tensor, g_trace: cute.Tensor, sample_block: cutlass.Int32):
        (n_init_producer, n_in_producer, y_acc_consumer, y_ready_producer, n_acc_consumer, done_consumer) = math_pipelines
        _stamp(g_trace, cutlass.Int32(1), cutlass.Int32(0))
        self._initialize_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, False)
        n_init_producer.acquire_and_advance().commit()
        _stamp(g_trace, cutlass.Int32(1), cutlass.Int32(1))
        _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(0))
        for blk in cutlass.range(num_blocks, unroll=1):
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(0))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(11), cutlass.Int32(0))
            v_handle = v_consumer.wait_and_advance()
            alpha_handle = alpha_consumer.wait_and_advance()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(11), cutlass.Int32(1))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(12), cutlass.Int32(0))
            self._convert_matrix_to_z_input(tmem_ptr, self.tmem_n_offset, self.tmem_n_inp_offset, tiled_mma_update, tiled_mma_z, tidx)
            n_in_producer.acquire_and_advance().commit()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(12), cutlass.Int32(1))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(13), cutlass.Int32(0))
            y_handle = y_acc_consumer.wait_and_advance()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(13), cutlass.Int32(1))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(14), cutlass.Int32(0))
            block_coeff = cutlass.Float32(sAlpha[self.BLK - 1, AlphaProcessor.CUMPROD, alpha_handle.index])
            self._process_y(tmem_ptr, tiled_mma_x, tidx, sV, sAlpha, v_handle.index, alpha_handle.index, block_coeff, tiled_mma_update)
            y_handle.release()
            self._scale_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, block_coeff)
            cute.arch.fence_view_async_tmem_store()
            y_ready_producer.acquire_and_advance().commit()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(14), cutlass.Int32(1))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(15), cutlass.Int32(0))
            n_acc_consumer.wait_and_advance().release()
            v_handle.release()
            alpha_handle.release()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(15), cutlass.Int32(1))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(1))
        _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(1))
        _stamp(g_trace, cutlass.Int32(2), cutlass.Int32(0))
        self._store_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, gN)
        done_consumer.wait_and_advance().release()
        _stamp(g_trace, cutlass.Int32(2), cutlass.Int32(1))

    @cute.jit
    def run_transfer_mma_role(self, tmem_ptr: cutlass.Int64, mma_args: tuple, smem_args: tuple, k_consumer, math_pipelines: tuple, num_blocks: cutlass.Int32, g_trace: cute.Tensor, sample_block: cutlass.Int32):
        (tiled_mma_z, tiled_mma_update, tiled_mma_update_ss) = mma_args
        (sK, sK_trans, sX) = smem_args
        (m_init_consumer, x_ready_consumer, m_in_consumer, z_acc_producer, z_ready_consumer, m_acc_producer) = math_pipelines
        _stamp(g_trace, cutlass.Int32(1), cutlass.Int32(0))
        m_init_consumer.wait_and_advance().release()
        _stamp(g_trace, cutlass.Int32(1), cutlass.Int32(1))
        z_shape = tiled_mma_z.partition_shape_C((self.D, self.BLK))
        tCtScratch_fake = tiled_mma_z.make_fragment_C(cute.append(z_shape, 1))
        tCtScratch = cute.make_tensor(tmem_ptr + self.tmem_scratch_offset, tCtScratch_fake.layout)
        update_shape = tiled_mma_update.partition_shape_C((self.D, self.D))
        tCtUpdate_fake = tiled_mma_update.make_fragment_C(cute.append(update_shape, 1))
        tCtM = cute.make_tensor(tmem_ptr + self.tmem_m_offset, tCtUpdate_fake.layout)
        z_inp_shape = tiled_mma_z.partition_shape_A((self.D, self.D))
        tCtZInp_fake = tiled_mma_z.make_fragment_A(cute.append(z_inp_shape, 1))
        tCtMInp = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_m_inp_offset, dtype=self.dtype), tCtZInp_fake.layout)
        update_inp_shape = tiled_mma_update.partition_shape_A((self.D, self.BLK))
        tCtUpdateInp_fake = tiled_mma_update.make_fragment_A(cute.append(update_inp_shape, 1))
        tCtZInp = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_m_inp_offset, dtype=self.dtype), tCtUpdateInp_fake.layout)
        tCrM_Z = tCtMInp
        tCrK_ZY = tiled_mma_z.make_fragment_B(sK_trans)
        tCrZ_Update = tCtZInp
        tCrX_Update = tiled_mma_update.make_fragment_B(sX)
        tCrK_Update = tiled_mma_update_ss.make_fragment_A(sK)
        _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(0))
        for blk in cutlass.range(num_blocks, unroll=1):
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(0))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(16), cutlass.Int32(0))
            k_handle = k_consumer.wait_and_advance()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(16), cutlass.Int32(1))
            if blk > 0:
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(17), cutlass.Int32(0))
                m_in_handle = m_in_consumer.wait_and_advance()
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(17), cutlass.Int32(1))
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(18), cutlass.Int32(0))
                for kphase in cutlass.range(cute.size(tCrM_Z, mode=[2]), unroll_full=True):
                    tiled_mma_z.set(tcgen05.Field.ACCUMULATE, kphase != 0)
                    cute.gemm(tiled_mma_z, tCtScratch[None, None, None, 0], tCrM_Z[None, None, kphase, 0], tCrK_ZY[None, None, kphase, k_handle.index], tCtScratch[None, None, None, 0])
                z_handle = z_acc_producer.acquire_and_advance()
                z_handle.commit()
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(18), cutlass.Int32(1))
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(19), cutlass.Int32(0))
                z_ready_consumer.wait_and_advance().release()
                m_in_handle.release()
                if blk == sample_block:
                    _stamp(g_trace, cutlass.Int32(19), cutlass.Int32(1))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(20), cutlass.Int32(0))
            x_ready_handle = x_ready_consumer.wait_and_advance()
            m_handle = m_acc_producer.acquire_and_advance()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(20), cutlass.Int32(1))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(21), cutlass.Int32(0))
            for kphase in cutlass.range(cute.size(tCrX_Update, mode=[2]), unroll_full=True):
                tiled_mma_update.set(tcgen05.Field.ACCUMULATE, True)
                if blk == 0:
                    tiled_mma_update_ss.set(tcgen05.Field.ACCUMULATE, True)
                    cute.gemm(tiled_mma_update_ss, tCtM[None, None, None, m_handle.index], tCrK_Update[None, None, kphase, k_handle.index], tCrX_Update[None, None, kphase, x_ready_handle.index], tCtM[None, None, None, m_handle.index])
                else:
                    cute.gemm(tiled_mma_update, tCtM[None, None, None, m_handle.index], tCrZ_Update[None, None, kphase, 0], tCrX_Update[None, None, kphase, x_ready_handle.index], tCtM[None, None, None, m_handle.index])
            m_handle.commit()
            x_ready_handle.release()
            k_handle.release()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(21), cutlass.Int32(1))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(1))
        _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(1))

    @cute.jit
    def run_state_mma_role(self, tmem_ptr: cutlass.Int64, mma_args: tuple, smem_args: tuple, load_pipelines: tuple, math_pipelines: tuple, num_blocks: cutlass.Int32, g_trace: cute.Tensor, sample_block: cutlass.Int32):
        (tiled_mma_x, tiled_mma_y, tiled_mma_update) = mma_args
        (sK, sK_trans, sT, sX) = smem_args
        (k_consumer, t_consumer) = load_pipelines
        (n_init_consumer, x_acc_producer, x_ready_consumer, n_in_consumer, y_acc_producer, y_ready_consumer, n_acc_producer) = math_pipelines
        _stamp(g_trace, cutlass.Int32(1), cutlass.Int32(0))
        n_init_consumer.wait_and_advance().release()
        _stamp(g_trace, cutlass.Int32(1), cutlass.Int32(1))
        x_shape = tiled_mma_x.partition_shape_C((self.D, self.BLK))
        tCtX_fake = tiled_mma_x.make_fragment_C(cute.append(x_shape, 1))
        tCtX = cute.make_tensor(tmem_ptr + self.tmem_x_y_acc_offset, tCtX_fake.layout)
        y_shape = tiled_mma_y.partition_shape_C((self.D, self.BLK))
        tCtY_fake = tiled_mma_y.make_fragment_C(cute.append(y_shape, 1))
        tCtY = cute.make_tensor(tmem_ptr + self.tmem_x_y_acc_offset, tCtY_fake.layout)
        update_shape = tiled_mma_update.partition_shape_C((self.D, self.D))
        tCtUpdate_fake = tiled_mma_update.make_fragment_C(cute.append(update_shape, 1))
        tCtN = cute.make_tensor(tmem_ptr + self.tmem_n_offset, tCtUpdate_fake.layout)
        y_inp_shape = tiled_mma_y.partition_shape_A((self.D, self.D))
        tCtYInp_fake = tiled_mma_y.make_fragment_A(cute.append(y_inp_shape, 1))
        tCtNInp = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_n_inp_offset, dtype=self.dtype), tCtYInp_fake.layout)
        update_inp_shape = tiled_mma_update.partition_shape_A((self.D, self.BLK))
        tCtUpdateInp_fake = tiled_mma_update.make_fragment_A(cute.append(update_inp_shape, 1))
        tCtYInp = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_n_inp_offset, dtype=self.dtype), tCtUpdateInp_fake.layout)
        tCrK_X = tiled_mma_x.make_fragment_A(sK)
        tCrT_X = tiled_mma_x.make_fragment_B(sT)
        tCrN_Y = tCtNInp
        tCrK_Y = tiled_mma_y.make_fragment_B(sK_trans)
        tCrY_Update = tCtYInp
        tCrX_Update = tiled_mma_update.make_fragment_B(sX)
        _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(0))
        for blk in cutlass.range(num_blocks, unroll=1):
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(0))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(22), cutlass.Int32(0))
            k_handle = k_consumer.wait_and_advance()
            t_handle = t_consumer.wait_and_advance()
            x_handle = x_acc_producer.acquire_and_advance()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(22), cutlass.Int32(1))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(23), cutlass.Int32(0))
            for kphase in cutlass.range(cute.size(tCrK_X, mode=[2]), unroll_full=True):
                tiled_mma_x.set(tcgen05.Field.ACCUMULATE, kphase != 0)
                cute.gemm(tiled_mma_x, tCtX[None, None, None, x_handle.index], tCrK_X[None, None, kphase, k_handle.index], tCrT_X[None, None, kphase, t_handle.index], tCtX[None, None, None, x_handle.index])
            x_handle.commit()
            t_handle.release()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(23), cutlass.Int32(1))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(24), cutlass.Int32(0))
            x_ready_handle = x_ready_consumer.wait_and_advance()
            n_in_handle = n_in_consumer.wait_and_advance()
            y_handle = y_acc_producer.acquire_and_advance()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(24), cutlass.Int32(1))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(25), cutlass.Int32(0))
            for kphase in cutlass.range(cute.size(tCrN_Y, mode=[2]), unroll_full=True):
                tiled_mma_y.set(tcgen05.Field.ACCUMULATE, kphase != 0)
                cute.gemm(tiled_mma_y, tCtY[None, None, None, y_handle.index], tCrN_Y[None, None, kphase, 0], tCrK_Y[None, None, kphase, k_handle.index], tCtY[None, None, None, y_handle.index])
            y_handle.commit()
            n_in_handle.release()
            k_handle.release()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(25), cutlass.Int32(1))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(26), cutlass.Int32(0))
            y_ready_consumer.wait_and_advance().release()
            n_handle = n_acc_producer.acquire_and_advance()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(26), cutlass.Int32(1))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(27), cutlass.Int32(0))
            for kphase in cutlass.range(cute.size(tCrX_Update, mode=[2]), unroll_full=True):
                tiled_mma_update.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(tiled_mma_update, tCtN[None, None, None, n_handle.index], tCrY_Update[None, None, kphase, 0], tCrX_Update[None, None, kphase, x_ready_handle.index], tCtN[None, None, None, n_handle.index])
            n_handle.commit()
            x_ready_handle.release()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(27), cutlass.Int32(1))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(1))
        _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(1))

    @cute.jit
    def run_load_alpha_role_sm100(self, g_alpha, sAlpha, alpha_producer, tok_offset, head_idx, chunk_len, num_blocks, num_heads, tidx, g_trace: cute.Tensor, sample_block: cutlass.Int32):
        _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(0))
        for blk in cutlass.range(num_blocks, unroll=1):
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(0))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(31), cutlass.Int32(0))
            handle = alpha_producer.acquire_and_advance()
            cute.arch.fence_view_async_shared()
            self.load_alpha_block(g_alpha, sAlpha, tok_offset, head_idx, blk, chunk_len, num_heads, handle.index, tidx)
            AlphaProcessor().run(sAlpha[None, None, handle.index], cutlass.Float32(1.0), True)
            self.mask_alpha_oob_block(sAlpha, blk, chunk_len, handle.index, tidx)
            cute.arch.fence_view_async_shared()
            handle.commit()
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(31), cutlass.Int32(1))
            if blk == sample_block:
                _stamp(g_trace, cutlass.Int32(3), cutlass.Int32(1))
        _stamp(g_trace, cutlass.Int32(0), cutlass.Int32(1))
        return alpha_producer
