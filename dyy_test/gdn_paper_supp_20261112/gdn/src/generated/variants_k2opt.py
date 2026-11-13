from flashinfer.gdn_kernels.blackwell.gated_delta_net_cp import *

class K2OptScaleNEarly(CPDeltaRuleMNPrecomputeUtcmma1Sm100):

    @cute.jit
    def run_compute_group_1(self, tidx: cutlass.Int32, tmem_ptr: cutlass.Int64, tiled_mma_x, tiled_mma_z, tiled_mma_update, sV: cute.Tensor, sAlpha: cute.Tensor, v_consumer, alpha_consumer, math_pipelines: tuple, num_blocks: cutlass.Int32, gN: cute.Tensor):
        (n_init_producer, n_in_producer, y_acc_consumer, y_ready_producer, n_acc_consumer, done_consumer) = math_pipelines
        self._initialize_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, False)
        n_init_producer.acquire_and_advance().commit()
        for _ in cutlass.range(num_blocks, unroll=1):
            v_handle = v_consumer.wait_and_advance()
            alpha_handle = alpha_consumer.wait_and_advance()
            self._convert_matrix_to_z_input(tmem_ptr, self.tmem_n_offset, self.tmem_n_inp_offset, tiled_mma_update, tiled_mma_z, tidx)
            n_in_producer.acquire_and_advance().commit()
            block_coeff = cutlass.Float32(sAlpha[self.BLK - 1, AlphaProcessor.CUMPROD, alpha_handle.index])
            self._scale_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, block_coeff)
            y_handle = y_acc_consumer.wait_and_advance()
            self._process_y(tmem_ptr, tiled_mma_x, tidx, sV, sAlpha, v_handle.index, alpha_handle.index, block_coeff, tiled_mma_update)
            y_handle.release()
            cute.arch.fence_view_async_tmem_store()
            y_ready_producer.acquire_and_advance().commit()
            n_acc_consumer.wait_and_advance().release()
            v_handle.release()
            alpha_handle.release()
        self._store_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, gN)
        done_consumer.wait_and_advance().release()

class K2OptFusedM(CPDeltaRuleMNPrecomputeUtcmma1Sm100):

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

class K2OptProcYPre(CPDeltaRuleMNPrecomputeUtcmma1Sm100):

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
                    (_, token) = tTR_cY[i, iter_m, iter_n]
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

    @cute.jit
    def run_compute_group_1(self, tidx: cutlass.Int32, tmem_ptr: cutlass.Int64, tiled_mma_x, tiled_mma_z, tiled_mma_update, sV: cute.Tensor, sAlpha: cute.Tensor, v_consumer, alpha_consumer, math_pipelines: tuple, num_blocks: cutlass.Int32, gN: cute.Tensor):
        (n_init_producer, n_in_producer, y_acc_consumer, y_ready_producer, n_acc_consumer, done_consumer) = math_pipelines
        self._initialize_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, False)
        n_init_producer.acquire_and_advance().commit()
        for _ in cutlass.range(num_blocks, unroll=1):
            v_handle = v_consumer.wait_and_advance()
            alpha_handle = alpha_consumer.wait_and_advance()
            self._convert_matrix_to_z_input(tmem_ptr, self.tmem_n_offset, self.tmem_n_inp_offset, tiled_mma_update, tiled_mma_z, tidx)
            n_in_producer.acquire_and_advance().commit()
            block_coeff = cutlass.Float32(sAlpha[self.BLK - 1, AlphaProcessor.CUMPROD, alpha_handle.index])
            y_handle = self._process_y_split(tmem_ptr, tiled_mma_x, tidx, sV, sAlpha, v_handle.index, alpha_handle.index, block_coeff, tiled_mma_update, y_acc_consumer)
            y_handle.release()
            self._scale_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, block_coeff)
            cute.arch.fence_view_async_tmem_store()
            y_ready_producer.acquire_and_advance().commit()
            n_acc_consumer.wait_and_advance().release()
            v_handle.release()
            alpha_handle.release()
        self._store_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, gN)
        done_consumer.wait_and_advance().release()

class K2OptScaleNEarlyFusedM(CPDeltaRuleMNPrecomputeUtcmma1Sm100):

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

    @cute.jit
    def run_compute_group_1(self, tidx: cutlass.Int32, tmem_ptr: cutlass.Int64, tiled_mma_x, tiled_mma_z, tiled_mma_update, sV: cute.Tensor, sAlpha: cute.Tensor, v_consumer, alpha_consumer, math_pipelines: tuple, num_blocks: cutlass.Int32, gN: cute.Tensor):
        (n_init_producer, n_in_producer, y_acc_consumer, y_ready_producer, n_acc_consumer, done_consumer) = math_pipelines
        self._initialize_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, False)
        n_init_producer.acquire_and_advance().commit()
        for _ in cutlass.range(num_blocks, unroll=1):
            v_handle = v_consumer.wait_and_advance()
            alpha_handle = alpha_consumer.wait_and_advance()
            self._convert_matrix_to_z_input(tmem_ptr, self.tmem_n_offset, self.tmem_n_inp_offset, tiled_mma_update, tiled_mma_z, tidx)
            n_in_producer.acquire_and_advance().commit()
            block_coeff = cutlass.Float32(sAlpha[self.BLK - 1, AlphaProcessor.CUMPROD, alpha_handle.index])
            self._scale_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, block_coeff)
            y_handle = y_acc_consumer.wait_and_advance()
            self._process_y(tmem_ptr, tiled_mma_x, tidx, sV, sAlpha, v_handle.index, alpha_handle.index, block_coeff, tiled_mma_update)
            y_handle.release()
            cute.arch.fence_view_async_tmem_store()
            y_ready_producer.acquire_and_advance().commit()
            n_acc_consumer.wait_and_advance().release()
            v_handle.release()
            alpha_handle.release()
        self._store_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, gN)
        done_consumer.wait_and_advance().release()

class K2OptAll(CPDeltaRuleMNPrecomputeUtcmma1Sm100):

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
                    (_, token) = tTR_cY[i, iter_m, iter_n]
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

    @cute.jit
    def run_compute_group_1(self, tidx: cutlass.Int32, tmem_ptr: cutlass.Int64, tiled_mma_x, tiled_mma_z, tiled_mma_update, sV: cute.Tensor, sAlpha: cute.Tensor, v_consumer, alpha_consumer, math_pipelines: tuple, num_blocks: cutlass.Int32, gN: cute.Tensor):
        (n_init_producer, n_in_producer, y_acc_consumer, y_ready_producer, n_acc_consumer, done_consumer) = math_pipelines
        self._initialize_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, False)
        n_init_producer.acquire_and_advance().commit()
        for _ in cutlass.range(num_blocks, unroll=1):
            v_handle = v_consumer.wait_and_advance()
            alpha_handle = alpha_consumer.wait_and_advance()
            self._convert_matrix_to_z_input(tmem_ptr, self.tmem_n_offset, self.tmem_n_inp_offset, tiled_mma_update, tiled_mma_z, tidx)
            n_in_producer.acquire_and_advance().commit()
            block_coeff = cutlass.Float32(sAlpha[self.BLK - 1, AlphaProcessor.CUMPROD, alpha_handle.index])
            self._scale_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, block_coeff)
            y_handle = self._process_y_split(tmem_ptr, tiled_mma_x, tidx, sV, sAlpha, v_handle.index, alpha_handle.index, block_coeff, tiled_mma_update, y_acc_consumer)
            y_handle.release()
            cute.arch.fence_view_async_tmem_store()
            y_ready_producer.acquire_and_advance().commit()
            n_acc_consumer.wait_and_advance().release()
            v_handle.release()
            alpha_handle.release()
        self._store_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, gN)
        done_consumer.wait_and_advance().release()

class K2OptScaleNEarlyProcYPre(CPDeltaRuleMNPrecomputeUtcmma1Sm100):

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
                    (_, token) = tTR_cY[i, iter_m, iter_n]
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

    @cute.jit
    def run_compute_group_1(self, tidx: cutlass.Int32, tmem_ptr: cutlass.Int64, tiled_mma_x, tiled_mma_z, tiled_mma_update, sV: cute.Tensor, sAlpha: cute.Tensor, v_consumer, alpha_consumer, math_pipelines: tuple, num_blocks: cutlass.Int32, gN: cute.Tensor):
        (n_init_producer, n_in_producer, y_acc_consumer, y_ready_producer, n_acc_consumer, done_consumer) = math_pipelines
        self._initialize_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, False)
        n_init_producer.acquire_and_advance().commit()
        for _ in cutlass.range(num_blocks, unroll=1):
            v_handle = v_consumer.wait_and_advance()
            alpha_handle = alpha_consumer.wait_and_advance()
            self._convert_matrix_to_z_input(tmem_ptr, self.tmem_n_offset, self.tmem_n_inp_offset, tiled_mma_update, tiled_mma_z, tidx)
            n_in_producer.acquire_and_advance().commit()
            block_coeff = cutlass.Float32(sAlpha[self.BLK - 1, AlphaProcessor.CUMPROD, alpha_handle.index])
            self._scale_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, block_coeff)
            y_handle = self._process_y_split(tmem_ptr, tiled_mma_x, tidx, sV, sAlpha, v_handle.index, alpha_handle.index, block_coeff, tiled_mma_update, y_acc_consumer)
            y_handle.release()
            cute.arch.fence_view_async_tmem_store()
            y_ready_producer.acquire_and_advance().commit()
            n_acc_consumer.wait_and_advance().release()
            v_handle.release()
            alpha_handle.release()
        self._store_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, gN)
        done_consumer.wait_and_advance().release()

class K2OptFusedMProcYPre(CPDeltaRuleMNPrecomputeUtcmma1Sm100):

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
                    (_, token) = tTR_cY[i, iter_m, iter_n]
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

    @cute.jit
    def run_compute_group_1(self, tidx: cutlass.Int32, tmem_ptr: cutlass.Int64, tiled_mma_x, tiled_mma_z, tiled_mma_update, sV: cute.Tensor, sAlpha: cute.Tensor, v_consumer, alpha_consumer, math_pipelines: tuple, num_blocks: cutlass.Int32, gN: cute.Tensor):
        (n_init_producer, n_in_producer, y_acc_consumer, y_ready_producer, n_acc_consumer, done_consumer) = math_pipelines
        self._initialize_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, False)
        n_init_producer.acquire_and_advance().commit()
        for _ in cutlass.range(num_blocks, unroll=1):
            v_handle = v_consumer.wait_and_advance()
            alpha_handle = alpha_consumer.wait_and_advance()
            self._convert_matrix_to_z_input(tmem_ptr, self.tmem_n_offset, self.tmem_n_inp_offset, tiled_mma_update, tiled_mma_z, tidx)
            n_in_producer.acquire_and_advance().commit()
            block_coeff = cutlass.Float32(sAlpha[self.BLK - 1, AlphaProcessor.CUMPROD, alpha_handle.index])
            y_handle = self._process_y_split(tmem_ptr, tiled_mma_x, tidx, sV, sAlpha, v_handle.index, alpha_handle.index, block_coeff, tiled_mma_update, y_acc_consumer)
            y_handle.release()
            self._scale_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, block_coeff)
            cute.arch.fence_view_async_tmem_store()
            y_ready_producer.acquire_and_advance().commit()
            n_acc_consumer.wait_and_advance().release()
            v_handle.release()
            alpha_handle.release()
        self._store_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, gN)
        done_consumer.wait_and_advance().release()

