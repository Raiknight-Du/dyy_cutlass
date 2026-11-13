import cutlass
import cutlass.cute as cute
from flashinfer.gdn_kernels.blackwell import gated_delta_net_cp as _SOURCE
globals().update({n:getattr(_SOURCE,n) for n in dir(_SOURCE) if not n.startswith("__")})
_Base=_SOURCE.CPDeltaRuleMNPrecomputeUtcmma1Sm100
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
@dsl_user_op
def _probe_clock_asm(address, endpoint, period, *, loc=None, ip=None):
    asm="""{
    .reg .pred sel, p;
    .reg .b32 bx, tid, warp, lane, rem, sid;
    .reg .b64 addr, clk, ns, sm;
    mov.u32 bx, %ctaid.x;
    mov.u32 tid, %tid.x;
    shr.u32 warp, tid, 5;
    and.b32 lane, tid, 31;
    rem.u32 rem, bx, PERIOD;
    setp.eq.u32 sel, rem, 0;
    setp.eq.u32 p, lane, 0; and.pred sel, sel, p;
    setp.lt.u32 p, warp, 8; and.pred sel, sel, p;
    mov.u64 addr, ADDRESS;
    mad.wide.u32 addr, bx, 384, addr;
    mad.wide.u32 addr, warp, 48, addr;
    add.u64 addr, addr, OFFSET;
    @sel mov.u64 clk, %clock64;
    @sel mov.u64 ns, %globaltimer;
    mov.u32 sid, %smid;
    cvt.u64.u32 sm, sid;
    @sel st.global.u64 [addr], clk;
    @sel st.global.u64 [addr+8], ns;
    @sel st.global.u64 [addr+16], sm;
    }"""
    asm=asm.replace("PERIOD",str(period)).replace("ADDRESS",str(address)).replace("OFFSET",str(endpoint*24))
    llvm.inline_asm(res=None,operands_=[],asm_string=asm,constraints="~{memory}",has_side_effects=True,asm_dialect=llvm.AsmDialect.AD_ATT,loc=loc,ip=ip)
class K2Body_B1(_Base):

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
        self._store_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, gN)
        done_consumer.wait_and_advance().release()

class K2Body_P1(_Base):

    @cute.jit
    def run_compute_group_0(self, tidx: cutlass.Int32, tmem_ptr: cutlass.Int64, tiled_mma_x, tiled_mma_z, tiled_mma_update, sX: cute.Tensor, sAlpha: cute.Tensor, alpha_consumer, math_pipelines: tuple, num_blocks: cutlass.Int32, gM: cute.Tensor):
        (m_init_producer, x_acc_consumer, x_ready_producer, m_in_producer, z_acc_consumer, z_ready_producer, m_acc_consumer, done_producer) = math_pipelines
        self._initialize_tmem_matrix(tmem_ptr, self.tmem_m_offset, tiled_mma_update, tidx, True)
        m_init_producer.acquire_and_advance().commit()
        _probe_clock_asm(self.probe_address, 0, 128)
        for blk in cutlass.range(num_blocks, unroll=1):
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
        _probe_clock_asm(self.probe_address, 1, 128)
        self._store_tmem_matrix(tmem_ptr, self.tmem_m_offset, tiled_mma_update, tidx, gM)
        done_producer.acquire_and_advance().commit()

    @cute.jit
    def run_compute_group_1(self, tidx: cutlass.Int32, tmem_ptr: cutlass.Int64, tiled_mma_x, tiled_mma_z, tiled_mma_update, sV: cute.Tensor, sAlpha: cute.Tensor, v_consumer, alpha_consumer, math_pipelines: tuple, num_blocks: cutlass.Int32, gN: cute.Tensor):
        (n_init_producer, n_in_producer, y_acc_consumer, y_ready_producer, n_acc_consumer, done_consumer) = math_pipelines
        self._initialize_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, False)
        n_init_producer.acquire_and_advance().commit()
        _probe_clock_asm(self.probe_address, 0, 128)
        for _ in cutlass.range(num_blocks, unroll=1):
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
        _probe_clock_asm(self.probe_address, 1, 128)
        self._store_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, gN)
        done_consumer.wait_and_advance().release()

class K2Body_P8(_Base):

    @cute.jit
    def run_compute_group_0(self, tidx: cutlass.Int32, tmem_ptr: cutlass.Int64, tiled_mma_x, tiled_mma_z, tiled_mma_update, sX: cute.Tensor, sAlpha: cute.Tensor, alpha_consumer, math_pipelines: tuple, num_blocks: cutlass.Int32, gM: cute.Tensor):
        (m_init_producer, x_acc_consumer, x_ready_producer, m_in_producer, z_acc_consumer, z_ready_producer, m_acc_consumer, done_producer) = math_pipelines
        self._initialize_tmem_matrix(tmem_ptr, self.tmem_m_offset, tiled_mma_update, tidx, True)
        m_init_producer.acquire_and_advance().commit()
        _probe_clock_asm(self.probe_address, 0, 16)
        for blk in cutlass.range(num_blocks, unroll=1):
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
        _probe_clock_asm(self.probe_address, 1, 16)
        self._store_tmem_matrix(tmem_ptr, self.tmem_m_offset, tiled_mma_update, tidx, gM)
        done_producer.acquire_and_advance().commit()

    @cute.jit
    def run_compute_group_1(self, tidx: cutlass.Int32, tmem_ptr: cutlass.Int64, tiled_mma_x, tiled_mma_z, tiled_mma_update, sV: cute.Tensor, sAlpha: cute.Tensor, v_consumer, alpha_consumer, math_pipelines: tuple, num_blocks: cutlass.Int32, gN: cute.Tensor):
        (n_init_producer, n_in_producer, y_acc_consumer, y_ready_producer, n_acc_consumer, done_consumer) = math_pipelines
        self._initialize_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, False)
        n_init_producer.acquire_and_advance().commit()
        _probe_clock_asm(self.probe_address, 0, 16)
        for _ in cutlass.range(num_blocks, unroll=1):
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
        _probe_clock_asm(self.probe_address, 1, 16)
        self._store_tmem_matrix(tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, gN)
        done_consumer.wait_and_advance().release()
