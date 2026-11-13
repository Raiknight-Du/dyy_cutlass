"""Independent one-instruction TMEM read/address RAW chain.
Not a K4 phase helper. No global/shared access in timed body.
Each elementary chain: tcgen05.ld + wait::ld + AND + ADD(next address).
The address operations stay explicit; total slope is not a bare ld latency.
"""
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import tcgen05,OperandMajorMode
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
from matched_shared import MatchedShared

def assembly(kind):
 shape='32x32b.x32' if kind==0 else '16x256b.x8'
 q=['{','.reg .b32 d<32>, a, dep, i;','.reg .pred more,active;',
    'mov.b32 a,$1;','mov.u32 i,0;','mov.b32 d0,0;','mov.b32 d31,0;','DEP_LOOP:']
 regs='{'+','.join('d%d'%i for i in range(32))+'}'
 for _ in range(16):
  q+=['tcgen05.ld.sync.aligned.%s.b32 %s,[a];'%(shape,regs),'tcgen05.wait::ld.sync.aligned;',
      'and.b32 dep,d0,32;','add.u32 a,$1,dep;','mov.b32 d0,a;']
 q+=['add.u32 i,i,1;','setp.lt.u32 more,i,$2;','@more bra DEP_LOOP;',
     'xor.b32 dep,d0,a;','xor.b32 $0,dep,d31;','}']
 return '\n'.join(q)

@dsl_user_op
def dep_loop(base,n,enabled,kind,*,loc=None,ip=None):
 v=llvm.inline_asm(cutlass.Uint32.mlir_type,[cutlass.Int32(x).ir_value(loc=loc,ip=ip) for x in [base,n,enabled]],
  assembly(kind),'=r,r,r,r,~{memory}',has_side_effects=True,is_align_stack=False,asm_dialect=llvm.AsmDialect.AD_ATT,loc=loc,ip=ip)
 return cutlass.Uint32(v)

class TmemReadDepNop(MatchedShared):
 def __init__(self,kind,warps=4):
  super().__init__(3);self.dep_kind=kind;self.active_warps=warps
  self.alloc_barrier=pipeline.NamedBarrier(barrier_id=1,num_threads=128)
 @cute.jit
 def __call__(self,checks:cute.Tensor,trace:cute.Tensor,n:cutlass.Int32,enabled:cutlass.Int32,stream):
  zmma=cute.make_tiled_mma(tcgen05.MmaF16BF16Op(cutlass.Float16,cutlass.Float32,(128,64,16),tcgen05.CtaGroup.ONE,
   tcgen05.OperandSource.SMEM,OperandMajorMode.K,OperandMajorMode.K))
  @cute.struct
  class Shared:
   holding:cutlass.Int32
  self.shared=Shared
  self.kernel(zmma,checks,trace,n,enabled).launch(grid=(1,1,1),block=(128,1,1),smem=Shared.size_in_bytes(),stream=stream,min_blocks_per_mp=1)
 @cute.kernel
 def kernel(self,zmma,checks,trace,n:cutlass.Int32,enabled:cutlass.Int32):
  tid,_,_=cute.arch.thread_idx();warp=cute.arch.make_warp_uniform(cute.arch.warp_idx())
  st=utils.SmemAllocator().allocate(self.shared)
  tm=utils.TmemAllocator(st.holding.ptr,barrier_for_retrieve=self.alloc_barrier,allocator_warp_id=0)
  tm.allocate(512);tm.wait_for_alloc();ptr=tm.retrieve_ptr(cutlass.Float32)
  self.clear_c_enclosing_region(ptr,zmma,tid)
  cute.arch.sync_threads()
  c0,c1,g0,g1=cutlass.Int64(0),cutlass.Int64(0),cutlass.Int64(0),cutlass.Int64(0)
  s0,s1=cutlass.Int32(0),cutlass.Int32(0)
  if tid==0:
   g0=cute.arch.globaltimer();c0=cute.arch.clock64();s0=cute.arch.smid()
  cute.arch.sync_threads()
  # Native TMEM row subpartition is selected by warp position. Match K4's
  # four CG1 warps; base local row 0, columns [0,63] are zero initialized.
  val=cutlass.Uint32(0)
  if warp<self.active_warps:val=dep_loop(ptr.toint(),n,enabled,self.dep_kind)
  cute.arch.sync_threads()
  if tid==0:
   c1=cute.arch.clock64();g1=cute.arch.globaltimer();s1=cute.arch.smid()
  cute.arch.sync_threads()
  checks[tid]=val
  if tid==0:
   trace[0]=c0;trace[1]=c1;trace[2]=g0;trace[3]=g1;trace[4]=cutlass.Int64(s0);trace[5]=cutlass.Int64(s1)
  cute.arch.sync_threads();tm.relinquish_alloc_permit();tm.free(ptr)
