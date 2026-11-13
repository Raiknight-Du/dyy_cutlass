"""Experiment-only K1: tcgen05 SS Wkk; unchanged upstream FP16 inverse/store.

Upstream baseline: e77a4a0d276367895c3b50a642fd8f326c03fb72.
This module deliberately does not patch or write the upstream checkout.
TMEM: 64 columns/CTA; launch min_blocks_per_mp=1 (baseline uses 8).
The actual residency and any launch-bound effect must be measured and reported.
"""
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.experimental.primitives.nvvm_wrapper import tcgen05_fence
from cutlass.cute.nvgpu import cpasync, warp, warpgroup, tcgen05, OperandMajorMode
from flashinfer.gdn_kernels.blackwell.gdn_cp_prefill import CPDeltaRuleTPrecomputeSm100
from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm120 import NamedBarrier
from flashinfer.gdn_kernels.delta_rule_dsl.collective_inverse_hmma import CollectiveInverse
from flashinfer.gdn_kernels.delta_rule_dsl.helpers import select_tensor_10
from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import (
    chunks_for_len, varlen_chunk_idx, varlen_chunk_valid_len,
)


class CPDeltaRuleTPrecomputeTcgen05(CPDeltaRuleTPrecomputeSm100):
    """Compatible full K1 kernel; store_t_to_gmem remains overrideable by mixin."""

    @cute.jit
    def __call__(
        self,
        g_k_tma: cute.Tensor,
        g_beta: cute.Tensor,
        g_t: cute.Tensor,
        cu_seqlens: cute.Tensor,
        num_k_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        total_t_blocks: cutlass.Int32,
        max_t_blocks_per_seq: cutlass.Int32,
        num_seqs: cutlass.Int32,
        stream,
    ):
        qkv_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128,
            self.dtype,
        )
        k_storage_layout_sd = cute.tile_to_shape(
            qkv_smem_layout_atom, (self.BLK, self.D), order=(0, 1)
        )
        k_storage_layout_ds = cute.select(k_storage_layout_sd, [1, 0])
        kk_layout_atom = cute.make_layout((8, 8), stride=(8, 1))
        kk_layout = cute.tile_to_shape(
            kk_layout_atom, (self.BLK, self.BLK), order=(0, 1)
        )
        beta_layout = cute.make_layout(self.BLK)

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            tma_load_op,
            g_k_tma,
            k_storage_layout_ds,
            (self.D, self.BLK),
        )
        self.tma_load_k_bytes = cute.size(k_storage_layout_ds) * (self.dtype.width // 8)

        @cute.struct
        class SharedStorage:
            k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            beta_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cute.struct.MemRange[cutlass.Int32, 1]
            smem_k: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(k_storage_layout_sd)], 128
            ]
            smem_kk: cute.struct.Align[
                cute.struct.MemRange[self.inverse_dtype, cute.cosize(kk_layout)], 16
            ]
            smem_beta: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(beta_layout)], 16
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tma_atom_k,
            tma_tensor_k,
            g_beta,
            g_t,
            cu_seqlens,
            num_k_heads,
            num_sab_heads,
            total_t_blocks,
            num_seqs,
        ).launch(
            grid=(num_sab_heads * max_t_blocks_per_seq, num_seqs, 1),
            block=(128, 1, 1),
            max_number_threads=(128, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        g_beta: cute.Tensor,
        g_t: cute.Tensor,
        cu_seqlens: cute.Tensor,
        num_k_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        total_t_blocks: cutlass.Int32,
        num_seqs: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bx, seq_idx, _ = cute.arch.block_idx()
        sab_head_idx = bx % num_sab_heads
        k_head_idx = sab_head_idx * num_k_heads // num_sab_heads
        block_idx_in_seq = bx // num_sab_heads
        seq_start = cutlass.Int32(cu_seqlens[seq_idx])
        seq_end = cutlass.Int32(cu_seqlens[seq_idx + 1])
        seq_len = seq_end - seq_start
        num_blocks = chunks_for_len(seq_len, self.BLK)
        if block_idx_in_seq < num_blocks:
            tok_offset = seq_start + block_idx_in_seq * self.BLK
            t_block_idx = varlen_chunk_idx(
                seq_idx, seq_start, block_idx_in_seq, self.BLK
            )
            valid_len = varlen_chunk_valid_len(seq_len, block_idx_in_seq, self.BLK)
            mT = cute.make_tensor(
                g_t.iterator,
                cute.make_ordered_layout(
                    (self.BLK, self.BLK, num_sab_heads, total_t_blocks),
                    order=(0, 1, 2, 3),
                ),
            )
            gT = mT[None, None, sab_head_idx, t_block_idx]

            qkv_smem_layout_atom = warpgroup.make_smem_layout_atom(
                warpgroup.SmemLayoutAtomKind.K_SW128,
                self.dtype,
            )
            k_storage_layout_sd = cute.tile_to_shape(
                qkv_smem_layout_atom, (self.BLK, self.D), order=(0, 1)
            )
            k_storage_layout_ds = cute.select(k_storage_layout_sd, [1, 0])
            kk_layout_atom = cute.make_layout((8, 8), stride=(8, 1))
            kk_layout = cute.tile_to_shape(
                kk_layout_atom, (self.BLK, self.BLK), order=(0, 1)
            )
            beta_layout = cute.make_layout(self.BLK)

            allocator = cutlass.utils.SmemAllocator()
            storage = allocator.allocate(self.shared_storage)
            sK_DS = storage.smem_k.get_tensor(
                k_storage_layout_ds.outer, swizzle=k_storage_layout_ds.inner
            )
            sK_SD = select_tensor_10(sK_DS)
            sKK_inv = storage.smem_kk.get_tensor(kk_layout)
            sBeta = storage.smem_beta.get_tensor(beta_layout)

            load_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
            tma_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
            vector_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
            vector_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 128
            )
            k_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.k_mbar_ptr.data_ptr(),
                num_stages=1,
                producer_group=load_producer_group,
                consumer_group=tma_consumer_group,
                tx_count=self.tma_load_k_bytes,
                cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            )
            beta_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=storage.beta_mbar_ptr.data_ptr(),
                num_stages=1,
                producer_group=vector_producer_group,
                consumer_group=vector_consumer_group,
            )
            k_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            beta_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            k_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )
            beta_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

            if warp_idx == 1:
                cpasync.prefetch_descriptor(tma_atom_k)
                self.load_k_tile_tma(
                    tma_atom_k,
                    tma_tensor_k,
                    sK_DS,
                    k_pipeline,
                    k_producer_state,
                    tok_offset,
                    k_head_idx,
                )
            elif warp_idx == 2:
                self.load_beta_tile(
                    g_beta,
                    sBeta,
                    beta_pipeline,
                    beta_producer_state,
                    tok_offset,
                    sab_head_idx,
                    valid_len,
                    num_sab_heads,
                    tidx,
                )

            k_pipeline.consumer_wait(k_consumer_state)
            beta_pipeline.consumer_wait(beta_consumer_state)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()

            kk_tiled_mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16)),
                cute.make_layout((4, 1, 1)),
                permutation_mnk=(self.BLK, self.BLK, self.D),
            )
            kk_thr_mma = kk_tiled_mma.get_slice(tidx)
            # Keep this original warp MMA object exclusively for the inherited
            # inverse output load/store interface. Wkk uses tcgen05 below.
            tc_mma = cute.make_tiled_mma(
                tcgen05.MmaF16BF16Op(
                    self.dtype, self.acc_dtype, (64, 64, 16),
                    tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                    OperandMajorMode.K, OperandMajorMode.K,
                )
            )
            tc_thr = tc_mma.get_slice(0)
            tc_a = tc_mma.make_fragment_A(tc_thr.partition_A(sK_SD))
            tc_b = tc_mma.make_fragment_B(tc_thr.partition_B(sK_SD))
            tc_fake_c = tc_mma.make_fragment_C(tc_mma.partition_shape_C((64, 64)))
            tmem = utils.TmemAllocator(
                storage.tmem_holding_buf.data_ptr(),
                barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=128),
                allocator_warp_id=0,
            )
            tmem.allocate(64)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tc_c = cute.make_tensor(tmem_ptr, tc_fake_c.layout)
            if warp_idx == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(storage.mma_mbar_ptr.data_ptr(), 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            tcgen05_fence("after_thread_sync")
            # TMA completion + original shared async fence precede the SS MMA.
            if warp_idx == 0:
                for kphase in cutlass.range(cute.size(tc_a, mode=[2]), unroll_full=True):
                    tc_mma.set(tcgen05.Field.ACCUMULATE, kphase != 0)
                    cute.gemm(tc_mma, tc_c, tc_a[None, None, kphase],
                              tc_b[None, None, kphase], tc_c)
                with cute.arch.elect_one():
                    tcgen05.commit(storage.mma_mbar_ptr.data_ptr())
            cute.arch.mbarrier_wait(storage.mma_mbar_ptr.data_ptr(), 0)
            tcgen05_fence("after_thread_sync")
            k_pipeline.consumer_release(k_consumer_state)

            # This is the M=64 TMEM load used by upstream Blackwell chunked GDN.
            # Coordinates accompany the load so the original inverse shared
            # layout receives exactly FP16(beta[row] * Wkk[row,col]) below the
            # diagonal, and FP16 zero everywhere else.
            tc_c_mn = utils.gemm.sm100.transform_partitioned_tensor_layout(tc_c)
            t2r_atom = cute.make_copy_atom(
                tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(16)), self.acc_dtype
            )
            t2r = tcgen05.make_tmem_copy(t2r_atom, tc_c[(None, None), 0, 0])
            thr_t2r = t2r.get_slice(tidx)
            src_c = thr_t2r.partition_S(tc_c_mn)
            coords = thr_t2r.partition_D(cute.make_identity_tensor((64, 64)))
            reg_c = cute.make_rmem_tensor_like(coords, self.acc_dtype)
            cute.copy(t2r, src_c, reg_c)
            cute.arch.fence_view_async_tmem_load()
            for i in cutlass.range_constexpr(cute.size(reg_c)):
                row, col = coords[i]
                value = cutlass.Float32(0.0)
                if row > col:
                    value = cutlass.Float32(reg_c[i]) * cutlass.Float32(sBeta[row])
                sKK_inv[row, col] = self.inverse_dtype(value)
            # All TMEM reads complete before allocator warp deallocates. All
            # inverse inputs are visible before the unchanged CollectiveInverse.
            tcgen05_fence("before_thread_sync")
            cute.arch.sync_threads()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
            cute.arch.barrier(barrier_id=NamedBarrier.MATH_SYNC, number_of_threads=128)
            CollectiveInverse().run(sKK_inv, NamedBarrier.MATH_SYNC)
            cute.arch.barrier(barrier_id=NamedBarrier.MATH_SYNC, number_of_threads=128)
            self.store_t_to_gmem(
                sKK_inv,
                sBeta,
                gT,
                valid_len,
                kk_tiled_mma,
                tidx,
            )
            beta_pipeline.consumer_release(beta_consumer_state)


TcgenK1 = CPDeltaRuleTPrecomputeTcgen05

