"""Standalone matched FINAL helper; initialized TMEM, fixed rotating8 outputs."""
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import tcgen05, OperandMajorMode
from runtime_k3_swizzled import K3FinalRUNTIME, K3FinalSWIZZLED as K3FinalCOALESCED

class FinalHelper(K3FinalRUNTIME):
    store_final_layout = K3FinalCOALESCED.store_final_layout

    def __init__(self, coalesced=False):
        super().__init__(rows_per_cta=64, needs_initial_state=True, store_final_state=True, store_initial_state=False)
        self.coalesced = coalesced

    @cute.jit
    def __call__(self, initial: cute.Tensor, fixed: cute.Tensor, final: cute.Tensor, repeat: cutlass.Int32, mode: cutlass.Int32, stream):
        mma = cute.make_tiled_mma(tcgen05.MmaTF32Op((64, 64, 8), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM, OperandMajorMode.K, OperandMajorMode.MN))
        self.micro(mma, initial, fixed, final, repeat, mode).launch(grid=(initial.shape[2], 1, 1), block=(256, 1, 1), smem=33792, stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def micro(self, mma, initial, fixed, final, repeat, mode):
        (tid, _, _) = cute.arch.thread_idx()
        wid = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        (bx, _, _) = cute.arch.block_idx()
        allocator = utils.SmemAllocator()
        holding = allocator.allocate_array(cutlass.Int32, 1)
        scratch = allocator.allocate_tensor(cutlass.Float32, cute.make_composed_layout(cute.make_swizzle(3, 2, 5), 0, cute.make_layout((64, 128), stride=(128, 1))), byte_alignment=128)
        tmem = utils.TmemAllocator(holding, barrier_for_retrieve=self.tmem_alloc_barrier, allocator_warp_id=0)
        tmem.allocate(256)
        if wid <= 4:
            tmem.wait_for_alloc()
        if wid < 4:
            ptr = tmem.retrieve_ptr(cutlass.Float32)
            acc = self.make_tmem_tensor(ptr, mma)
            self.copy_tensor_to_acc(mma, acc, initial[None, None, bx], tid, False, cutlass.Int32(0))
            pipeline.NamedBarrier(barrier_id=2, num_threads=128).arrive_and_wait()
            for r in cutlass.range(repeat, unroll=1):
                slot = r % 8
                if cutlass.const_expr(self.coalesced):
                    self.store_final_layout(mma, acc, tid, fixed[None, None, slot, bx], final[None, None, slot, bx], True, cutlass.Int32(0), mode, scratch)
                else:
                    self.store_acc(mma, acc, tid, fixed[None, None, slot, bx], final[None, None, slot, bx], True, cutlass.Int32(0), mode)
            tmem.relinquish_alloc_permit()
            tmem.free(ptr)
        else:
            cute.arch.setmaxregister_decrease(32)
