"""Separate full-K1 candidate: in-place shared epilogue, then bulk S2G."""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp


class BulkStoreMixin:
    @cute.jit
    def store_t_to_gmem(
        self,
        sKK_inv: cute.Tensor,
        sBeta: cute.Tensor,
        gT_CR: cute.Tensor,
        valid_len: cutlass.Int32,
        kk_tiled_mma,
        tidx: cutlass.Int32,
    ):
        # Preserve the original inverse reload and its MMA-C coordinates.
        ldsm_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.inverse_dtype,
        )
        tiled_load = cute.make_tiled_copy_C(ldsm_atom, kk_tiled_mma)
        thr_load = tiled_load.get_slice(tidx)
        rInv = cute.make_rmem_tensor(
            kk_tiled_mma.partition_shape_C((self.BLK, self.BLK)),
            self.inverse_dtype,
        )
        cute.copy(tiled_load, thr_load.partition_S(sKK_inv), thr_load.retile(rInv))

        # Reuse exactly the inverse's 8192-byte storage, with the final
        # workspace's physical linear order. BF16 needs a pointer recast.
        sT = cute.make_tensor(
            cute.recast_ptr(sKK_inv.iterator, dtype=self.dtype),
            cute.make_layout((self.BLK, self.BLK), stride=(1, self.BLK)),
        )
        stsm_atom = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )
        tiled_store = cute.make_tiled_copy_C(stsm_atom, kk_tiled_mma)
        thr_store = tiled_store.get_slice(tidx)
        rInv_store = thr_store.retile(rInv)
        coords = thr_store.partition_D(
            cute.make_identity_tensor((self.BLK, self.BLK))
        )
        rT = cute.make_rmem_tensor(
            kk_tiled_mma.partition_shape_C((self.BLK, self.BLK)), self.dtype
        )
        rT_store = thr_store.retile(rT)
        for i in cutlass.range_constexpr(cute.size(rT_store)):
            col, row = coords[i]
            value = cutlass.Float32(0.0)
            if row < valid_len and col < valid_len:
                beta = cutlass.Float32(sBeta[row])
                value = -beta * cutlass.Float32(rInv_store[i])
            rT_store[i] = self.dtype(value)

        # All four warps must finish reading the inverse before any warp
        # overwrites its storage. Neither tensor is used after this hook.
        cute.arch.sync_threads()
        cute.copy(tiled_store, rT_store, thr_store.partition_D(sT))
        # Publish ordinary/STSM writes to the bulk copy's async proxy.
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        # CopyBulkS2GOp performs its own warp election. All lanes of a
        # converged warp must enter cute.copy, commit and wait together.
        if tidx // 32 == 0:
            sFlat = cute.make_tensor(sT.iterator, cute.make_layout(self.BLK * self.BLK))
            gFlat = cute.make_tensor(gT_CR.iterator, cute.make_layout(self.BLK * self.BLK))
            bulk_atom = cute.make_copy_atom(
                cpasync.CopyBulkS2GOp(),
                self.dtype,
                num_bits_per_copy=self.BLK * self.BLK * self.dtype.width,
            )
            cute.copy(bulk_atom, sFlat, gFlat)
            cute.arch.cp_async_bulk_commit_group()
            # Conservative full completion, including the global writes.
            cute.arch.cp_async_bulk_wait_group(0)
        cute.arch.sync_threads()
