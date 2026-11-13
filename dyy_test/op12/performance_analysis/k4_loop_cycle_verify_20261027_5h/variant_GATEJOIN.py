# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Private mechanism test: V gather after selected completed-MMA waits.
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import tcgen05
from flashinfer.gdn_kernels.blackwell.gated_delta_net_cp_prefill import CPDeltaRulePrefillTcgen05Sm100
class K4VJoin(CPDeltaRulePrefillTcgen05Sm100):
    @cute.jit
    def compute_group_1_chunk(
        self,
        tidx: cutlass.Int32,
        tmem_ptr: cutlass.Int64,
        scale: cutlass.Float32,
        mma_args: tuple,
        smem_args: tuple,
        checkpoint_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
    ) -> tuple[
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        cutlass.Int32,
    ]:
        """Warps 4-7: process one chunk from the caller-owned chunk stream."""
        sV, sCumsumlog, sCumprod, sBeta, sO = smem_args
        mS_checkpoints, checkpoint_offset, checkpoint_every_n_tokens = checkpoint_args
        (
            load_v_consumer,
            load_gate_consumer,
            cg1_shared_acc_consumer,
            kv_acc_consumer,
            q_state_acc_consumer,
            kv_acc_producer,
            state_inp_ready_producer,
            vks_ready_producer,
            nv_ready_producer,
            decay_v_ready_producer,
            o_store_producer,
        ) = pipeline_args
        tiled_mma_kv, tiled_mma_qs, tiled_mma_qkv = mma_args
        chunk_iter, num_pairs_b, head_idx, seqlen_b, is_first_chunk = work_args

        # ------------------------------------------------------------------
        # Preamble (identical to compute_group_1)
        # ------------------------------------------------------------------
        num_threads_cg1 = self.threads_per_warp * len(self.compute_group_1_warp_ids)
        cg1_tidx = tidx % num_threads_cg1

        state_acc_shape = tiled_mma_kv.partition_shape_C(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        tCtState_fake = tiled_mma_kv.make_fragment_C(
            cute.append(state_acc_shape, self.tmem_kv_acc_stages)
        )
        tCtState = cute.make_tensor(
            tmem_ptr + self.tmem_state_offset, tCtState_fake.layout
        )
        tCtState_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            tCtState
        )
        tCcState = cute.make_identity_tensor(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        atom_state_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        atom_state_r2t = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tCtState_for_t2r = tCtState[(None, None), 0, 0, 0]
        tiled_state_t2r = tcgen05.make_tmem_copy(atom_state_t2r, tCtState_for_t2r)
        tiled_state_r2t = tcgen05.make_tmem_copy(atom_state_r2t, tCtState_for_t2r)
        thr_state_t2r = tiled_state_t2r.get_slice(cg1_tidx)
        thr_state_r2t = tiled_state_r2t.get_slice(cg1_tidx)
        tTR_tCtState = thr_state_t2r.partition_S(tCtState_mn_view)
        tTR_tCcState = thr_state_t2r.partition_D(tCcState)
        tRT_tCtState = thr_state_r2t.partition_D(tCtState_mn_view)
        tTR_rState = cute.make_rmem_tensor_like(tTR_tCcState, self.acc_dtype)
        tRG_rState = cute.make_rmem_tensor_like(tTR_tCcState, self.state_dtype)

        state_inp_shape = tiled_mma_qs.partition_shape_A(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[2])
        )
        tCtState_inp_fake = tiled_mma_qs.make_fragment_A(
            cute.append(state_inp_shape, self.tmem_state_inp_stages)
        )
        tCtState_inp = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_state_inp_offset, dtype=self.io_dtype),
            tCtState_inp_fake.layout,
        )
        tCtState_inp_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            tCtState_inp
        )
        tCcState_inp = cute.make_identity_tensor(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[2])
        )
        atom_state_inp_r2t = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), self.io_dtype
        )
        tCtState_inp_for_r2t = tCtState_inp_mn_view[None, None, 0]
        tiled_state_inp_r2t = tcgen05.make_tmem_copy(
            atom_state_inp_r2t, tCtState_inp_for_r2t
        )
        thr_state_inp_r2t = tiled_state_inp_r2t.get_slice(cg1_tidx)
        tRT_tCcState_inp = thr_state_inp_r2t.partition_S(tCcState_inp)
        tRT_tCtState_inp = thr_state_inp_r2t.partition_D(tCtState_inp_mn_view)
        tRT_rState_inp = cute.make_rmem_tensor_like(tRT_tCcState_inp, self.io_dtype)

        qkv_acc_shape = tiled_mma_qkv.partition_shape_C(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[1])
        )
        tCtShared_fake = tiled_mma_qkv.make_fragment_C(qkv_acc_shape)
        tCtShared = cute.make_tensor(
            tmem_ptr + self.tmem_cg1_shared_acc_offset,
            cute.flat_product(
                tCtShared_fake.layout,
                cute.make_layout((self.tmem_cg1_shared_acc_stages,)),
            ),
        )
        tCtShared_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            tCtShared
        )
        tCcShared = cute.make_identity_tensor(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[1])
        )
        atom_shared_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tCtShared_for_t2r = tCtShared[(None, None), 0, 0, 0]
        tiled_shared_t2r = tcgen05.make_tmem_copy(atom_shared_t2r, tCtShared_for_t2r)
        thr_shared_t2r = tiled_shared_t2r.get_slice(cg1_tidx)
        tTR_tCtShared = thr_shared_t2r.partition_S(tCtShared_mn_view)
        tTR_tCcShared = thr_shared_t2r.partition_D(tCcShared)

        # Dedicated full-tile KS t2r (separate atom so it can be tuned
        # independently of atom_shared_t2r used by NV/decay_v reads).
        atom_ks_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tiled_ks_t2r = tcgen05.make_tmem_copy(atom_ks_t2r, tCtShared_for_t2r)
        thr_ks_t2r = tiled_ks_t2r.get_slice(cg1_tidx)
        tTR_tCtKS = thr_ks_t2r.partition_S(tCtShared_mn_view)
        tTR_tCcKS = thr_ks_t2r.partition_D(tCcShared)

        qkv_inp_shape = tiled_mma_qkv.partition_shape_A(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[2])
        )
        tCtShared_inp_fake = tiled_mma_qkv.make_fragment_A(
            cute.append(qkv_inp_shape, self.tmem_shared_inp_stages)
        )
        tCtShared_inp = cute.make_tensor(
            cute.recast_ptr(
                tmem_ptr + self.tmem_shared_inp_offset, dtype=self.io_dtype
            ),
            tCtShared_inp_fake.layout,
        )
        tCtShared_inp_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            tCtShared_inp
        )
        tCcShared_inp = cute.make_identity_tensor(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[2])
        )
        atom_shared_inp_r2t = cute.make_copy_atom(
            tcgen05.copy.St16x128bOp(tcgen05.copy.Repetition(8)), self.io_dtype
        )
        tCtShared_inp_for_r2t = tCtShared_inp_mn_view[None, None, 0]
        tiled_shared_inp_r2t = tcgen05.make_tmem_copy(
            atom_shared_inp_r2t, tCtShared_inp_for_r2t
        )
        thr_shared_inp_r2t = tiled_shared_inp_r2t.get_slice(cg1_tidx)
        tRT_tCtShared_inp = thr_shared_inp_r2t.partition_D(tCtShared_inp_mn_view)

        # Dedicated full-tile VKS r2t (separate atom so it can be tuned
        # independently of atom_shared_inp_r2t used by NV/decay_v writes).
        atom_vks_r2t = cute.make_copy_atom(
            tcgen05.copy.St16x128bOp(tcgen05.copy.Repetition(8)), self.io_dtype
        )
        tiled_vks_r2t = tcgen05.make_tmem_copy(atom_vks_r2t, tCtShared_inp_for_r2t)
        thr_vks_r2t = tiled_vks_r2t.get_slice(cg1_tidx)
        tRT_tCtVKS_inp = thr_vks_r2t.partition_D(tCtShared_inp_mn_view)

        qs_acc_shape = tiled_mma_qs.partition_shape_C(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[1])
        )
        tCtQState_fake = tiled_mma_qs.make_fragment_C(
            cute.append(qs_acc_shape, self.tmem_q_state_acc_stages)
        )
        tCtQState = cute.make_tensor(
            tmem_ptr + self.tmem_q_state_offset, tCtQState_fake.layout
        )
        tCtQState_mn_view = utils.gemm.sm100.transform_partitioned_tensor_layout(
            tCtQState
        )
        tCcQState = cute.make_identity_tensor(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[1])
        )
        atom_qs_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        atom_qs_r2t = cute.make_copy_atom(
            tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tCtQState_for_t2r = tCtQState[(None, None), 0, 0, 0]
        tiled_qs_t2r = tcgen05.make_tmem_copy(atom_qs_t2r, tCtQState_for_t2r)
        tCtQState_for_r2t = tCtQState[(None, None), 0, 0, 0]
        tiled_qs_r2t = tcgen05.make_tmem_copy(atom_qs_r2t, tCtQState_for_r2t)
        thr_qs_t2r = tiled_qs_t2r.get_slice(cg1_tidx)
        thr_qs_r2t = tiled_qs_r2t.get_slice(cg1_tidx)
        tTR_tCtQS = thr_qs_t2r.partition_S(tCtQState_mn_view)
        tTR_tCcQS = thr_qs_t2r.partition_D(tCcQState)
        tRT_tCtQS = thr_qs_r2t.partition_D(tCtQState_mn_view)
        tTR_rQS = cute.make_rmem_tensor_like(tTR_tCcQS, self.acc_dtype)

        tRT_tCcV = thr_shared_inp_r2t.partition_S(tCcShared_inp)
        atom_v_s2r = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True),
            self.io_dtype,
        )
        tiled_v_s2r = cute.make_tiled_copy_S(
            atom_v_s2r,
            tiled_shared_inp_r2t,
        )
        thr_v_s2r = tiled_v_s2r.get_slice(cg1_tidx)

        atom_o_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tiled_o_t2r = tcgen05.make_tmem_copy(atom_o_t2r, tCtQState_for_t2r)
        thr_o_t2r = tiled_o_t2r.get_slice(cg1_tidx)
        tTR_tOtO = thr_o_t2r.partition_S(tCtQState_mn_view)
        tTR_tOcO = thr_o_t2r.partition_D(tCcQState)
        atom_o_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=True),
            self.io_dtype,
        )
        tiled_o_r2s = cute.make_tiled_copy_D(atom_o_r2s, tiled_o_t2r)
        thr_o_r2s = tiled_o_r2s.get_slice(cg1_tidx)
        tCsO = thr_o_r2s.partition_D(sO)

        sub_tile_size = 32
        sV_vt_view = utils.gemm.sm100.transform_partitioned_tensor_layout(sV)
        tCsV = thr_v_s2r.partition_S(sV_vt_view)

        rCumprod = cute.make_rmem_tensor((1, cute.size(tTR_tCcShared)), self.acc_dtype)
        tGrCumprod = thr_shared_t2r.partition_D(rCumprod)
        tGrDecayScale = cute.make_rmem_tensor_like(tTR_tCcShared, self.acc_dtype)
        # Pair membership is derived from the caller-provided chunk index.
        is_pair_first = (chunk_iter & 1) == 0

        valid_state = is_first_chunk == False  # noqa: E712
        if cutlass.const_expr(self.use_initial_state):
            valid_state = True
            if is_pair_first:
                kv_acc_producer.advance()
                kv_acc_producer.advance()

        # Only the total decay is needed to finish the state update.  Defer
        # the per-row gate work until the previous state has been published
        # and decayed, keeping its register fragment in one contiguous region.
        gate_handle = load_gate_consumer.wait_and_advance()
        # valid_len = max(0, min(self.b_t, seqlen_b - chunk_iter * self.b_t))
        # gamma_end = 1.0 if valid_len == 0 else sCumprod[valid_len - 1, 0, gate_handle.index]
        # OOB alpha is padded with 1 before the prefix scan, so the last physical slot equals gamma_end.
        cumprod_total = sCumprod[self.b_t - 1, 0, gate_handle.index]

        kv_prev_handle = kv_acc_consumer.current_handle()
        if valid_state:
            kv_prev_handle = kv_acc_consumer.wait_and_advance()
            # No empty-stage wait is needed before reusing state_inp.  For the
            # first publication the stage is initialized empty.  Thereafter,
            # CG1 reaches this point only after waiting for the preceding NewV;
            # the MMA warp releases state_inp before publishing that NewV.
            state_inp_handle = state_inp_ready_producer.current_handle()
            state_inp_ready_producer.advance()
            cute.copy(
                tiled_state_t2r,
                tTR_tCtState[None, 0, None, kv_prev_handle.index],
                tTR_rState[None, 0, None],
            )
            tRT_rState_inp[None, 0, None].store(
                tTR_rState[None, 0, None].load().to(self.io_dtype)
            )
            cute.copy(
                tiled_state_inp_r2t,
                tRT_rState_inp[None, 0, None],
                tRT_tCtState_inp[None, 0, None, state_inp_handle.index],
            )
            cute.arch.fence_view_async_tmem_store()
            state_inp_handle.commit()
            checkpoint_token = self.b_t * chunk_iter
            if cutlass.const_expr(self.enable_checkpoints):
                if checkpoint_token > 0:
                    if checkpoint_token <= seqlen_b:
                        if checkpoint_token % checkpoint_every_n_tokens == 0:
                            gS_checkpoints = cute.flat_divide(
                                mS_checkpoints[
                                    None,
                                    None,
                                    head_idx,
                                    checkpoint_offset,
                                ],
                                (
                                    self.mma_tiler_kv[0],
                                    self.mma_tiler_kv[1],
                                ),
                            )[None, None, 0, 0]
                            tSgCheckpoints = thr_state_t2r.partition_D(gS_checkpoints)
                            if cutlass.const_expr(self.state_dtype != self.acc_dtype):
                                tRG_rState[None, 0, None].store(
                                    tTR_rState[None, 0, None]
                                    .load()
                                    .to(self.state_dtype)
                                )
                            else:
                                tRG_rState = tTR_rState
                            cute.autovec_copy(
                                tRG_rState[None, 0, None],
                                tSgCheckpoints[None, 0, None],
                                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                            )
            for k in cutlass.range(cute.size(tTR_rState), vectorize=True):
                tTR_rState[k] = tTR_rState[k] * cumprod_total
            cute.copy(
                tiled_state_r2t,
                tTR_rState[None, 0, None],
                tRT_tCtState[None, 0, None, kv_prev_handle.index],
            )
            if cutlass.const_expr(self.enable_checkpoints):
                if checkpoint_token > 0:
                    if checkpoint_token <= seqlen_b:
                        if checkpoint_token % checkpoint_every_n_tokens == 0:
                            checkpoint_offset += 1
            cute.arch.fence_view_async_tmem_store()
            kv_prev_handle.release()
        ks_handle = cg1_shared_acc_consumer.current_handle()
        qs_handle = q_state_acc_consumer.current_handle()
        if valid_state:
            ks_handle = cg1_shared_acc_consumer.wait_and_advance()
            qs_handle = q_state_acc_consumer.wait_and_advance()
        for k in cutlass.range_constexpr(cute.size(tTR_tCcShared)):
            coord = tTR_tCcShared[k]
            tGrCumprod[k] = sCumprod[coord[1], 0, gate_handle.index]
        last_cumsumlog = sCumsumlog[self.b_t - 1, 0, gate_handle.index]
        for k in cutlass.range_constexpr(0, cute.size(tTR_tCcShared), 2):
            coord0 = tTR_tCcShared[k]
            coord1 = tTR_tCcShared[k + 1]
            decay_diff = cute.arch.add_packed_f32x2(
                (last_cumsumlog, last_cumsumlog),
                (
                    -sCumsumlog[coord0[1], 0, gate_handle.index],
                    -sCumsumlog[coord1[1], 0, gate_handle.index],
                ),
                ftz=False,
                rnd="rn",
            )
            tGrDecayScale[k] = cute.math.exp2(decay_diff[0], fastmath=True)
            tGrDecayScale[k + 1] = cute.math.exp2(decay_diff[1], fastmath=True)
        gate_handle.release()

        vks_handle = vks_ready_producer.current_handle()
        vks_ready_producer.advance()
        v_handle = load_v_consumer.wait_and_advance()
        tRT_rV = cute.make_rmem_tensor_like(tRT_tCcV, self.io_dtype)
        tCrV = tiled_v_s2r.retile(tRT_rV)
        # Always publish V through fixed TMEM slot 0. The SMEM V ring cursor
        # survives persistent work boundaries, so a new work item cannot
        # assume that its first V tile resides in SMEM stage 0.
        cute.copy(
            tiled_v_s2r,
            tCsV[None, None, None, v_handle.index],
            tCrV,
        )

        if valid_state:
            tTR_rKS = cute.make_rmem_tensor_like(tTR_tCcKS, self.acc_dtype)
            cute.copy(
                tiled_ks_t2r,
                tTR_tCtKS[None, None, None, ks_handle.index],
                tTR_rKS,
            )
            for k in cutlass.range(cute.size(tTR_rKS), vectorize=True):
                tTR_rKS[k] = tTR_rKS[k] * tGrCumprod[k]
            ks_handle.release()
            for k in cutlass.range(cute.size(tTR_rKS), vectorize=True):
                tRT_rV[k] = tRT_rV[k] - tTR_rKS[k].to(self.io_dtype)
        cute.copy(
            tiled_vks_r2t,
            tRT_rV,
            tRT_tCtVKS_inp[None, None, None, 0],
        )
        cute.arch.fence_view_async_tmem_store()
        vks_handle.commit()

        if valid_state:
            cute.copy(
                tiled_qs_t2r,
                tTR_tCtQS[None, None, 0, qs_handle.index],
                tTR_rQS[None, None, 0],
            )
            for k in cutlass.range(cute.size(tTR_rQS), vectorize=True):
                tTR_rQS[k] = tTR_rQS[k] * tGrCumprod[k] * scale
            cute.copy(
                tiled_qs_r2t,
                tTR_rQS[None, None, 0],
                tRT_tCtQS[None, None, 0, qs_handle.index],
            )
            cute.arch.fence_view_async_tmem_store()
            qs_handle.release()

        nv_handle = cg1_shared_acc_consumer.wait_and_advance()
        v_handle.release()
        tTR_rNv = cute.make_rmem_tensor_like(tTR_tCcShared, self.acc_dtype)
        tTR_rNv_inp = cute.make_rmem_tensor_like(tTR_rNv, self.io_dtype)
        for sub in cutlass.range(tTR_rNv.shape[1]):
            cute.copy(
                tiled_shared_t2r,
                tTR_tCtShared[None, sub, 0, nv_handle.index],
                tTR_rNv[None, sub, 0],
            )
            tTR_rNv_inp[None, sub, 0].store(
                tTR_rNv[None, sub, 0].load().to(self.io_dtype)
            )
        nv_handle.release()

        tTR_rDv = tTR_rNv
        for sub in cutlass.range(tTR_rDv.shape[1]):
            for k in cutlass.range(sub_tile_size, vectorize=True):
                tTR_rDv[k, sub, 0] = tTR_rDv[k, sub, 0] * tGrDecayScale[k, sub, 0]

        nv_ready_handle = nv_ready_producer.current_handle()
        nv_ready_producer.advance()
        decay_v_ready_handle = decay_v_ready_producer.current_handle()
        decay_v_ready_producer.advance()
        # Both chunks publish NV for GEMM6/QKV before decayV for GEMM7/KV.
        for sub in cutlass.range(tTR_rDv.shape[1]):
            cute.copy(
                tiled_shared_inp_r2t,
                tTR_rNv_inp[None, sub, 0],
                tRT_tCtShared_inp[None, sub, 0, 0],
            )
            tTR_rNv_inp[None, sub, 0].store(
                tTR_rDv[None, sub, 0].load().to(self.io_dtype)
            )
            cute.copy(
                tiled_shared_inp_r2t,
                tTR_rNv_inp[None, sub, 0],
                tRT_tCtShared_inp[None, sub, 0, 1],
            )
        cute.arch.fence_view_async_tmem_store()
        nv_ready_handle.commit()
        decay_v_ready_handle.commit()

        # Drain this chunk's output at the end of the same chunk.  This keeps
        # O ownership uniform and removes the first/last pending-output cases.
        o_handle = o_store_producer.acquire_and_advance()
        o_qs_handle = q_state_acc_consumer.wait_and_advance()
        tTR_tOrO = cute.make_rmem_tensor_like(tTR_tOcO, self.acc_dtype)
        tTR_rO_out = cute.make_rmem_tensor_like(tTR_tOrO, self.io_dtype)
        tRS_tOrO = tiled_o_r2s.retile(tTR_rO_out)
        cute.copy(
            tiled_o_t2r,
            tTR_tOtO[None, None, None, o_qs_handle.index],
            tTR_tOrO,
        )
        tTR_rO_out.store(tTR_tOrO.load().to(self.io_dtype))
        cute.copy(
            tiled_o_r2s,
            tRS_tOrO,
            tCsO[None, None, None, o_handle.index],
        )
        cute.arch.fence_view_async_shared()
        o_qs_handle.release()
        o_handle.commit()

        return (
            load_v_consumer,
            load_gate_consumer,
            cg1_shared_acc_consumer,
            kv_acc_consumer,
            q_state_acc_consumer,
            kv_acc_producer,
            state_inp_ready_producer,
            vks_ready_producer,
            nv_ready_producer,
            decay_v_ready_producer,
            o_store_producer,
            checkpoint_offset,
        )
