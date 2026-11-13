"""Contiguous-warp output mapping for the *full* upstream K1 kernel.

Place this mixin before CPDeltaRuleTPrecomputeSm120 (or its tcgen05
subclass) in the MRO.  It replaces only the final store_t_to_gmem method;
the shared inverse and every preceding computation remain in place.
"""

import cutlass
import cutlass.cute as cute


class CoalescedStoreMixin:
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
        # The caller launches 128 threads and BLK=64.  Unlike the original
        # MMA-C partition, this mapping is independent of kk_tiled_mma.
        lane = tidx % 32
        warp_idx = tidx // 32

        # Keep the row loop rolled: full unrolling lets ptxas hoist many
        # address/value calculations across the inverse and caused a
        # pointer spill with the unchanged min_blocks_per_mp=8 launch.
        for row_group in cutlass.range(self.BLK // 4, unroll=1):
            row = warp_idx + 4 * row_group
            beta = cutlass.Float32(0.0)
            if row < valid_len:
                beta = cutlass.Float32(sBeta[row])

            for col_half in cutlass.range_constexpr(self.BLK // 32):
                col = lane + 32 * col_half
                # Physical shared coordinates are intentionally identical
                # to the original MMA-C load/global-store coordinates.
                # sKK_inv has already been populated by CollectiveInverse.
                inv = cutlass.Float32(sKK_inv[col, row])
                value = cutlass.Float32(0.0)
                if row < valid_len and col < valid_len:
                    value = -beta * inv
                # gT_CR has strides (1, BLK). Each warp instruction covers
                # 32 adjacent 16-bit outputs, i.e. two aligned 32B sectors.
                # Explicit writes also zero every padded output element.
                gT_CR[col, row] = self.dtype(value)
