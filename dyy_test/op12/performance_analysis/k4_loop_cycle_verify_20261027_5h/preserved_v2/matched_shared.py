"""R4 independent shader-LDS/QK-SS coupling candidate; GPU admission separate.

Reuses frozen QkResident fragment/output and resident completion protocol.
All four modes are runtime flags in one kernel/CUBIN. No per-round handshake.
"""
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05,cpasync,OperandMajorMode
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
from old_qk_resident import QkResident,semantic,semantic_tensor

FAMILIES = __import__('json').loads(__import__('pathlib').Path(__file__).with_name('FAMILIES.json').read_text())

def matched_assembly(kind):
    fam=FAMILIES[kind];count=len(fam['original_PCs'])
    q=['{','.reg .b32 v<%d>, addr<%d>, i, sum;'%(fam['value_count'],count),'.reg .pred on, active, more;',
       'setp.ne.u32 active, $%d, 0;'%(count+3)]
    q += ['mov.b32 v%d, %d;'%(j,(0x3c003c00+j*0x10001)&0xffffffff) for j in range(fam['value_count'])]
    q += ['add.u32 addr%d, $1, $%d;'%(j,j+2) for j in range(count)]
    q += ['mov.u32 i, 0;','MATCH_LOOP:']
    for j in range(count):
        q += ['setp.ge.s32 on, $%d, 0;'%(j+2),'and.pred on, on, active;']
        op=fam['ptx']
        if op=='mixed_scalar64':op='ld.shared.v2.b32' if fam['opcodes'][j]=='LDS.64' else 'ld.shared.b32'
        start=fam['value_starts'][j];regs='{'+','.join('v%d'%(start+k) for k in range(fam['widths'][j]))+'}'
        if op.startswith('ldmatrix'):inst='%s %s, [addr%d];'%(op,regs,j)
        elif op.startswith('stmatrix'):inst='%s [addr%d], %s;'%(op,j,regs)
        elif 'v2' in op:inst='%s %s, [addr%d];'%(op,regs,j)
        else:inst='%s v%d, [addr%d];'%(op,start,j)
        q += ['@on '+inst]
    q += ['add.u32 i, i, 1;','setp.lt.u32 more, i, $%d;'%(count+2),'@more bra MATCH_LOOP;','mov.b32 sum,v0;']
    q += ['xor.b32 sum, sum, v%d;'%k for k in range(1,fam['value_count'])]
    q += ['mov.b32 $0,sum;','}']
    return '\n'.join(q)

@dsl_user_op
def matched_loop(base,reg_offsets,n,enabled,kind,*,loc=None,ip=None):
    vals=[cutlass.Int32(base)] + [cutlass.Int32(reg_offsets[j]) for j in range(len(FAMILIES[kind]['original_PCs']))] + [cutlass.Int32(n),cutlass.Int32(enabled)]
    value=llvm.inline_asm(cutlass.Uint32.mlir_type,[x.ir_value(loc=loc,ip=ip) for x in vals],
        matched_assembly(kind),'=r,'+','.join(['r']*len(vals))+',~{memory}',has_side_effects=True,
        is_align_stack=False,asm_dialect=llvm.AsmDialect.AD_ATT,loc=loc,ip=ip)
    return cutlass.Uint32(value)

class MatchedShared(QkResident):
    def __init__(self,kind):
        super().__init__(instrumented=False,per_round_wait=False)
        self.kind=kind
        self.count=len(FAMILIES[kind]["original_PCs"])

    @cute.jit
    def __call__(self,gA:cute.Tensor,gB:cute.Tensor,gO:cute.Tensor,
                 shader_checks:cute.Tensor,trace:cute.Tensor,offsets:cute.Tensor,
                 iterations:cutlass.Int32,mode:cutlass.Int32,stream):
        mma=cute.make_tiled_mma(tcgen05.MmaF16BF16Op(cutlass.Float16,cutlass.Float32,
             (64,64,16),self.cta_group,tcgen05.OperandSource.SMEM,OperandMajorMode.K,OperandMajorMode.K))
        # Used only to zero the enclosing physical C region before timing.
        zero_mma=cute.make_tiled_mma(tcgen05.MmaF16BF16Op(cutlass.Float16,cutlass.Float32,
             (128,64,16),self.cta_group,tcgen05.OperandSource.SMEM,OperandMajorMode.K,OperandMajorMode.K))
        al=sm100_utils.make_smem_layout_a(mma,(64,64,128),cutlass.Float16,1)
        bl=sm100_utils.make_smem_layout_b(mma,(64,64,128),cutlass.Float16,1)
        atom_a,tensor_a=cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(self.cta_group),
            gA,cute.slice_(semantic(al),(None,None,0)),(64,128))
        atom_b,tensor_b=cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(self.cta_group),
            gB,cute.slice_(semantic(bl),(None,None,0)),(64,128))
        @cute.struct
        class Shared:
            input_mbar:cute.struct.MemRange[cutlass.Int64,2]
            done_mbar:cute.struct.MemRange[cutlass.Int64,2]
            tmem_holding:cutlass.Int32
            sA:cute.struct.Align[cute.struct.MemRange[cutlass.Float16,cute.cosize(al)],1024]
            sB:cute.struct.Align[cute.struct.MemRange[cutlass.Float16,cute.cosize(bl)],1024]
            shader:cute.struct.Align[cute.struct.MemRange[cutlass.Uint32,8192],1024]
        self.shared=Shared
        self.kernel(mma,zero_mma,atom_a,tensor_a,al,atom_b,tensor_b,bl,gO,shader_checks,
            trace,offsets,iterations,mode).launch(grid=(128,1,1),block=(192,1,1),smem=225280,
                stream=stream,min_blocks_per_mp=1)

    @cute.jit
    def clear_c_enclosing_region(self,ptr,zero_mma,tid):
        shape=zero_mma.partition_shape_C((128,64))
        fake=zero_mma.make_fragment_C(cute.append(shape,1))
        c=cute.make_tensor(ptr,fake.layout);cm=semantic_tensor(c)
        cp=tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
                cutlass.Float32),c[(None,None),0,0,0])
        th=cp.get_slice(tid);dst=th.partition_D(cm)
        r=cute.make_rmem_tensor_like(th.partition_S(cute.make_identity_tensor((128,64))),cutlass.Float32)
        r.fill(0)
        for sub in cutlass.range(cute.size(r,mode=[2]),unroll_full=True):
            cute.copy(cp,r[None,0,sub],dst[None,0,sub,0])
        cute.arch.fence_view_async_tmem_store()

    @cute.kernel
    def kernel(self,mma,zero_mma,atom_a,tensor_a,al,atom_b,tensor_b,bl,gO,
               shader_checks,trace,offsets,iterations:cutlass.Int32,mode:cutlass.Int32):
        tid,_,_=cute.arch.thread_idx();bid,_,_=cute.arch.block_idx()
        warp=cute.arch.make_warp_uniform(cute.arch.warp_idx())
        alloc=utils.SmemAllocator();st=alloc.allocate(self.shared)
        sA=st.sA.get_tensor(al.outer,swizzle=al.inner);sB=st.sB.get_tensor(bl.outer,swizzle=bl.inner)
        ss=st.shader.get_tensor(cute.make_layout((8192,),stride=(1,)))
        cg1=pipeline.CooperativeGroup(pipeline.Agent.Thread,1)
        cg128=pipeline.CooperativeGroup(pipeline.Agent.Thread,128)
        iprod,icons=pipeline.PipelineTmaUmma.create(num_stages=1,producer_group=cg1,consumer_group=cg1,
            tx_count=32768,barrier_storage=st.input_mbar.data_ptr(),defer_sync=True).make_participants()
        dp,dc=pipeline.PipelineUmmaAsync.create(num_stages=1,producer_group=cg1,consumer_group=cg128,
            barrier_storage=st.done_mbar.data_ptr(),defer_sync=True).make_participants()
        pipeline.pipeline_init_arrive(cluster_shape_mn=cute.make_layout((1,1)),is_relaxed=True)
        pipeline.pipeline_init_wait(cluster_shape_mn=cute.make_layout((1,1)))
        tm=utils.TmemAllocator(st.tmem_holding.ptr,barrier_for_retrieve=self.alloc_barrier,allocator_warp_id=0)
        tm.allocate(512)
        if warp<=4:tm.wait_for_alloc()
        sc,ec=cutlass.Int64(0),cutlass.Int64(0)
        sm0,sm1=cutlass.Int32(0),cutlass.Int32(0)
        if warp<4:
            ptr=tm.retrieve_ptr(cutlass.Float32)
            self.clear_c_enclosing_region(ptr,zero_mma,tid)
            for j in cutlass.range(64,unroll_full=True):
                idx=cutlass.Uint32(tid+128*j)
                value=cutlass.Uint32(0x3c003c00)
                ss[tid+128*j]=value
            reg_offsets=cute.make_rmem_tensor((self.count,),cutlass.Int32)
            for j in cutlass.range(self.count,unroll_full=True):
                reg_offsets[j]=offsets[tid,j]
            # First rendezvous completes initialization; thread0 clock precedes
            # the second rendezvous, which releases both independent N loops.
            cute.arch.sync_threads()
            if tid==0:sm0=cute.arch.smid();sc=cute.arch.clock64()
            cute.arch.sync_threads()
            check=matched_loop(st.shader.data_ptr().toint(),reg_offsets,iterations,mode&1,self.kind)
            done=dc.wait_and_advance();done.release()
            cute.arch.sync_threads()
            if tid==0:ec=cute.arch.clock64();sm1=cute.arch.smid()
            # Fixed OUTSIDE-clock rendezvous prevents other warp stores before
            # thread0 has taken its end timestamp.
            cute.arch.sync_threads()
            shader_checks[bid,tid]=cutlass.Int32(check)
            self.output(ptr,mma,tid,gO[bid,None,None])
            if tid==0:
                trace[bid,0]=sc;trace[bid,1]=ec;trace[bid,2]=cutlass.Int64(sm0)
                trace[bid,3]=cutlass.Int64(sm1);trace[bid,4]=cutlass.Int64(iterations);trace[bid,5]=cutlass.Int64(mode)
        elif warp==4:
            ptr=tm.retrieve_ptr(cutlass.Float32);c=self.make_c(ptr,mma)
            aa,bb=mma.make_fragment_A(sA),mma.make_fragment_B(sB)
            ih=icons.wait_and_advance();done=dp.acquire_and_advance()
            cute.arch.sync_threads();cute.arch.sync_threads()
            if (mode&2)!=0:
                for it in cutlass.range(iterations,unroll=1):
                    for kk in cutlass.range(cute.size(aa,mode=[2]),unroll_full=True):
                        mma.set(tcgen05.Field.ACCUMULATE,kk!=0)
                        cute.gemm(mma,c[None,None,None,0],aa[None,None,kk,ih.index],
                                  bb[None,None,kk,ih.index],c[None,None,None,0])
            # One completion event in ALL modes, including an empty MMA prefix.
            done.commit();ih.release()
            cute.arch.sync_threads();cute.arch.sync_threads()
        elif warp==5:
            h=iprod.acquire_and_advance()
            ta,ga=cpasync.tma_partition(atom_a,0,cute.make_layout(1),
                cute.group_modes(semantic_tensor(sA)[None,None,h.index],0,2),cute.group_modes(tensor_a,0,2))
            cute.copy(atom_a,ga,ta,tma_bar_ptr=h.barrier)
            tb,gb=cpasync.tma_partition(atom_b,0,cute.make_layout(1),
                cute.group_modes(semantic_tensor(sB)[None,None,h.index],0,2),cute.group_modes(tensor_b,0,2))
            cute.copy(atom_b,gb,tb,tma_bar_ptr=h.barrier)
            cute.arch.sync_threads();cute.arch.sync_threads()
            cute.arch.sync_threads();cute.arch.sync_threads()
        cute.arch.sync_threads()
        if warp<4:
            ptr=tm.retrieve_ptr(cutlass.Float32);tm.relinquish_alloc_permit();tm.free(ptr)
