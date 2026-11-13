import fs from "node:fs";
import { schedule, validate } from "./calendar.mjs";

const F = 1965e6, SMS = 148, BGPU = 7.7e12 / F;
const R = { HBM: BGPU / 148, SMEM: 128, TMEM: 512, CVT: 64,
  SIMT: 128, TC16: 2.25e15 / (148 * F), control: 1 };
const S = {
  setup: "gated_delta_net_cp.py:1331-1340",
  seq: "gated_delta_net_cp.py:1152-1153",
  tma: "gated_delta_net_cp.py:153-189,1460-1500",
  alphaLoad: "gated_delta_net_cp.py:109-127,192-224",
  alphaProc: "alpha.py:18-63",
  init: "gated_delta_net_cp.py:227-258",
  scale: "gated_delta_net_cp.py:260-293",
  store: "gated_delta_net_cp.py:295-324",
  x: "gated_delta_net_cp.py:326-374",
  convert: "gated_delta_net_cp.py:376-431",
  zConvert: "gated_delta_net_cp.py:433-488",
  y: "gated_delta_net_cp.py:491-570",
  transfer: "gated_delta_net_cp.py:573-671",
  state: "gated_delta_net_cp.py:673-776",
  cg0: "gated_delta_net_cp.py:777-849",
  cg1: "gated_delta_net_cp.py:851-918",
  storage: "gated_delta_net_cp.py:1032-1068,1181-1329",
  caller: "gdn_cp_prefill.py:283-289",
  kernel: "gated_delta_net_cp.py:1342-1517"
};
function add(jobs,id,op,resource,demand,unit,deps,src,extra={}) {
  jobs.push({id,op,resource:resource||"control",demand,unit,dependency:[...new Set((deps||[]).filter(Boolean))],
    priority:0,source_location:src,
    cycle_formula:resource?"ceil(demand / rate[resource]); zero demand -> 0":"0 numeric cycles; unknown tail remains symbolic",
    symbolic_tail:"",...extra});
  return id;
}
function ctrl(jobs,id,op,deps,src,tail="",extra={}) {
  return add(jobs,id,op,"control",0,"events",deps,src,{symbolic_tail:tail,...extra});
}
function work(jobs,id,op,res,demand,unit,deps,src,extra={}) {
  return add(jobs,id,op,res,demand,unit,deps,src,extra);
}
function phases(jobs,prefix,op,list,deps,src,shape,extra={}) {
  let prev;
  list.forEach((p,i)=>{
    const id=prefix+"_"+i;
    prev=work(jobs,id,op,p[0],p[1],p[2],i?[prev]:deps,src,
      {shape,operation_phase:p[3],...extra});
  });
  return prev;
}
function orderedTiles(jobs,prefix,op,list,deps,src,tileShape,tileCount,extra={}) {
  let priorTile=null;
  for(let tile=0;tile<tileCount;tile++) {
    let priorPhase=null;
    for(const [suffix,res,demand,unit,phase] of list) {
      const nodeId=tile===0?`${prefix}_${suffix}`:`${prefix}_tile${tile}_${suffix}`;
      const pred=priorPhase?[priorPhase]:(priorTile?[priorTile]:deps);
      priorPhase=work(jobs,nodeId,op,res,demand,unit,pred,src,
        {tile_index:tile,tile_count:tileCount,inferred_register_tile_shape:tileShape,
         layout_evidence:"source partition-loop count and payload inference; not compile/PTX-verified",
         operation_phase:phase,...extra});
    }
    priorTile=priorPhase;
  }
  return priorTile;
}

function makeJobs(L,conservative=true) {
  const j=[],id={aReady:[],aRelease:[],mZRelease:[],kRelease:[],vRelease:[],tRelease:[],
    xRelease:[],xMat:[],xMma:[],zMma:[],zReady:[],mUpdate:[],mScale:[],
    nInput:[],yMma:[],yProc:[],nScale:[],nUpdate:[],tmaReady:{k:[],v:[],t:[]}};
  const setup=ctrl(j,"cta_setup","pipeline setup and TMEM allocation",[],S.setup,
    "tau_CTA_dispatch + tau_pipeline_init + tau_TMEM_alloc");
  const seq=work(j,"cu_seqlens_load","load sequence bounds","HBM",8,"B",[setup],S.seq,
    {shape:[2],element_type:"int32",logical_global_bytes:8});
  const inputs=ctrl(j,"cta_indices_ready","chunk coordinates and offsets ready",[seq],S.seq,"tau_index_issue");
  work(j,"m_init_tmem","initialize M identity in TMEM","TMEM",65536,"B",[setup],S.init,
    {shape:[128,128],element_type:"fp32",scalar_fill:"identity"});
  work(j,"n_init_tmem","initialize N zero in TMEM","TMEM",65536,"B",[setup],S.init,
    {shape:[128,128],element_type:"fp32",scalar_fill:"zero"});
  const mi=ctrl(j,"m_init_ready","M initialization published",["m_init_tmem"],S.cg0,
    "tau_TMEM_store_visibility + tau_named_barrier");
  const ni=ctrl(j,"n_init_ready","N initialization published",["n_init_tmem"],S.cg1,
    "tau_TMEM_store_visibility + tau_named_barrier");

  const ttypes=[["k",16384,3,[128,64]],["v",16384,3,[128,64]],["t",8192,3,[64,64]]];
  let lastIssue=inputs,lastPacket=null;
  for(let b=0;b<L;b++) for(const [type,bytes,stages,shape] of ttypes) {
    const p="tma_"+type+"_"+b;
    const issueDeps=[lastIssue];
    const issue=ctrl(j,p+"_issue","issue TMA "+type.toUpperCase(),issueDeps,S.tma,
      "tau_TMA_issue",{block:b,pipe_stage:b%stages});
    lastIssue=issue;
    const hd=[issue];
    if(conservative&&lastPacket)hd.push(lastPacket);
    const hb=work(j,p+"_hbm","TMA "+type.toUpperCase()+" HBM read","HBM",bytes,"B",hd,S.tma,
      {block:b,pipe_stage:b%stages,shape,dtype:"fp16/bf16",logical_global_bytes:bytes,
       packet_policy:conservative?"prior packet SMEM complete":"HBM/SMEM pipelined"});
    const sm=work(j,p+"_smem","TMA "+type.toUpperCase()+" shared write","SMEM",bytes,"B",[hb],S.tma,
      {block:b,pipe_stage:b%stages,shape,dtype:"fp16/bf16",
       smem_note:type==="k"?"K-transpose is an aliasing view of K stage":"staged tensor"});
    const ready=ctrl(j,p+"_ready","TMA "+type.toUpperCase()+" ready",[sm],S.tma,
      "tau_TMA_response + tau_mbarrier",{block:b,pipe_stage:b%stages});
    id.tmaReady[type][b]=ready;lastPacket=sm;
  }

  // Alpha producer and four-stage alpha storage.
  for(let b=0;b<L;b++) {
    const deps=[b===0?inputs:id.aReady[b-1]];if(b>=4)deps.push(id.aRelease[b-4]);
    const issue=ctrl(j,"alpha_"+b+"_issue","alpha load and stage acquire",deps,S.alphaLoad,
      "tau_alpha_issue",{block:b,pipe_stage:b%4});
    const raw=work(j,"alpha_"+b+"_gmem","alpha global read","HBM",256,"B",[issue],S.alphaLoad,
      {block:b,pipe_stage:b%4,shape:[64],element_type:"fp32",logical_global_bytes:256});
    const wr=work(j,"alpha_"+b+"_raw_smem","raw alpha channel stage write","SMEM",256,"B",[raw],S.alphaLoad,
      {block:b,shape:[64,1],element_type:"fp32"});
    const rd=work(j,"alpha_"+b+"_proc_read","AlphaProcessor channel-0 read","SMEM",256,"B",[wr],S.alphaProc,
      {block:b,shape:[64],element_type:"fp32",logical_smem_request_bytes:256});
    let alphaPhase=work(j,"alpha_"+b+"_epsilon","AlphaProcessor raw + epsilon","SIMT",64,"results",[rd],S.alphaProc,
      {block:b,shape:[64],fp32_results:64,fp32_flops:64,operation:"raw + 1e-10"});
    alphaPhase=ctrl(j,"alpha_"+b+"_log2","AlphaProcessor log2 SFU",[alphaPhase],S.alphaProc,
      "64*tau_log2_SFU",{block:b,sfu_calls:{log2:64}});
    for(const off of [1,2,4,8,16]) for(let i=0;i<2;i++) {
      const sh=ctrl(j,`alpha_${b}_scan_off${off}_i${i}_shuffle`,`AlphaProcessor scan shuffle off=${off}, column=${i}`,
        [alphaPhase],S.alphaProc,"tau_shuffle",{block:b,shuffle_calls:1,offset:off,column:i});
      alphaPhase=work(j,`alpha_${b}_scan_off${off}_i${i}_add`,`AlphaProcessor active-lane scan additions off=${off}, column=${i}`,
        "SIMT",32-off,"results",[sh],S.alphaProc,
        {block:b,column:i,offset:off,active_lanes:32-off,fp32_results:32-off,fp32_flops:32-off});
    }
    const carrySh=ctrl(j,"alpha_"+b+"_carry_shuffle","AlphaProcessor inter-column carry shuffle",[alphaPhase],S.alphaProc,
      "tau_shuffle",{block:b,shuffle_calls:1,carry_for_column:1});
    alphaPhase=work(j,"alpha_"+b+"_carry_add","AlphaProcessor inter-column carry additions","SIMT",32,"results",[carrySh],S.alphaProc,
      {block:b,shape:[32],fp32_results:32,fp32_flops:32});
    alphaPhase=ctrl(j,"alpha_"+b+"_end_broadcast","AlphaProcessor terminal-lane end-log shuffle",[alphaPhase],S.alphaProc,
      "tau_shuffle",{block:b,shuffle_calls:1});
    for(let i=0;i<2;i++) {
      const cs=work(j,`alpha_${b}_cumsum_store_i${i}`,"AlphaProcessor cumulative-log channel write","SMEM",128,"B",[alphaPhase],S.alphaProc,
        {block:b,column:i,element_count:32,logical_smem_request_bytes:128});
      const cp=ctrl(j,`alpha_${b}_cumprod_exp2_i${i}`,"AlphaProcessor cumulative-product exp2 SFU",[cs],S.alphaProc,
        "32*tau_exp2_SFU",{block:b,column:i,sfu_calls:{exp2:32}});
      const cpStore=work(j,`alpha_${b}_cumprod_store_i${i}`,"AlphaProcessor cumulative-product channel write","SMEM",128,"B",[cp],S.alphaProc,
        {block:b,column:i,element_count:32,logical_smem_request_bytes:128});
      const sub=work(j,`alpha_${b}_end_sub_i${i}`,"AlphaProcessor end-log minus prefix","SIMT",32,"results",[cpStore],S.alphaProc,
        {block:b,column:i,fp32_results:32,fp32_flops:32,operation:"end_log - frag"});
      const exp=ctrl(j,`alpha_${b}_neg_end_exp2_i${i}`,"AlphaProcessor negative-end reciprocal exp2 SFU",[sub],S.alphaProc,
        "32*tau_exp2_SFU",{block:b,column:i,sfu_calls:{exp2:32}});
      const neg=work(j,`alpha_${b}_neg_i${i}`,"AlphaProcessor reciprocal unary negation","SIMT",32,"results",[exp],S.alphaProc,
        {block:b,column:i,fp32_results:32,fp32_flops:0,unary_negates:32});
      alphaPhase=work(j,`alpha_${b}_neg_store_i${i}`,"AlphaProcessor negative-end channel write","SMEM",128,"B",[neg],S.alphaProc,
        {block:b,column:i,element_count:32,logical_smem_request_bytes:128});
    }
    id.aReady[b]=ctrl(j,"alpha_"+b+"_ready","alpha stage committed",[alphaPhase],S.alphaLoad,
      "tau_shared_fence + tau_barrier_commit",{block:b,pipe_stage:b%4});
    id.aRelease[b]="alpha_"+b+"_release";
  }

  for(let b=0;b<L;b++) {
    const k=id.tmaReady.k[b],v=id.tmaReady.v[b],t=id.tmaReady.t[b],a=id.aReady[b];
    const xReuse=b>=2?id.xRelease[b-2]:null;

    // State issuer computes X, which is then materialized before Y can reuse the
    // same TMEM accumulator slot. Following X waits for prior N update.
    const xi=ctrl(j,"x_mma_"+b+"_issue","X=K*T tcgen05 issue",
      [k,t,ni,...(b?[id.nUpdate[b-1]]:[])],S.state,"tau_MMA_issue",
      {block:b,mma_role:"warp 11",shape:{M:128,N:64,K:64},k_phases:4});
    const xt=work(j,"x_mma_"+b+"_tc","X=K*T tcgen05 service","TC16",1048576,"FLOP",[xi],S.state,
      {block:b,mma_role:"warp 11",shape:{M:128,N:64,K:64},logical_flops:1048576,k_phases:4,
       operand_source:"SMEM K,T",accumulator:"TMEM X/Y shared slot"});
    id.xMma[b]=ctrl(j,"x_mma_"+b+"_done","X MMA complete",[xt],S.state,
      "tau_MMA_response + tau_mbarrier",{block:b});
    const xdeps=[id.xMma[b],a,...(b===0?[mi]:[]),...(xReuse?[xReuse]:[]),...(b?[id.mScale[b-1]]:[])];
    const xs=orderedTiles(j,"x_mat_"+b,"materialize X subtile",
      [["tmem_read","TMEM",16384,"B","fp32 accumulator read"],
       ["cvt","CVT",4096,"elements","fp32-to-input conversion"],
       ["smem","SMEM",8192,"B","low-precision shared write"]],
      xdeps,S.x,[64,64],2,{block:b,element_count_per_tile:4096,pipe_stage:b%2,
       element_type:"fp32->fp16/bf16"});
    id.xMat[b]=ctrl(j,"x_mat_"+b+"_ready","X shared stage ready",[xs],S.cg0,
      "tau_shared_fence + tau_x_ready_barrier",{block:b,pipe_stage:b%2});

    const mc=work(j,"m_coeff_"+b,"CG0 block-coefficient logical loads","SMEM",512,"B",[a],S.cg0,
      {block:b,shape:[128,1],element_type:"fp32",logical_smem_request_bytes:512,
       physical_broadcast_estimate_B:16,physical_estimate:"4 CG0 warps x 4B warp broadcast"});
    const nc=work(j,"n_coeff_"+b,"CG1 block-coefficient logical loads","SMEM",512,"B",[a],S.cg1,
      {block:b,shape:[128,1],element_type:"fp32",logical_smem_request_bytes:512,
       physical_broadcast_estimate_B:16,physical_estimate:"4 CG1 warps x 4B warp broadcast"});

    let zReady=null;
    if(b>0) {
      const mw=orderedTiles(j,"m_convert_"+b,"M state-to-Z-input subtile",
        [["tmem_read","TMEM",16384,"B","fp32 accumulator read"],
         ["cvt","CVT",4096,"elements","fp32-to-input conversion"],
         ["tmem_write","TMEM",8192,"B","low-precision TMEM write"]],
        [id.xMat[b],id.mScale[b-1]],S.convert,[128,32],4,
        {block:b,element_count_per_tile:4096,destination:"TMEM offset 320",element_type:"fp32->fp16/bf16"});
      const mReady=ctrl(j,"m_convert_"+b+"_ready","M Z operand ready",[mw],S.cg0,
        "tau_TMEM_store_visibility + tau_mbarrier",{block:b});
      const zi=ctrl(j,"z_mma_"+b+"_issue","Z=M*K^T tcgen05 issue",[mReady,k],S.transfer,
        "tau_MMA_issue",{block:b,mma_role:"warp 8",shape:{M:128,N:64,K:128},k_phases:8});
      const zt=work(j,"z_mma_"+b+"_tc","Z=M*K^T tcgen05 service","TC16",2097152,"FLOP",[zi],S.transfer,
        {block:b,mma_role:"warp 8",shape:{M:128,N:64,K:128},logical_flops:2097152,k_phases:8,
         operand_source:"TMEM M-input, shared K-transpose",accumulator:"TMEM Z scratch"});
      id.zMma[b]=ctrl(j,"z_mma_"+b+"_done","Z MMA complete",[zt],S.transfer,
        "tau_MMA_response + tau_mbarrier",{block:b});
      const zw=orderedTiles(j,"z_convert_"+b,"Z scratch-to-update-input subtile",
        [["tmem_read","TMEM",16384,"B","fp32 scratch read"],
         ["cvt","CVT",4096,"elements","fp32-to-input conversion"],
         ["tmem_write","TMEM",8192,"B","low-precision TMEM write"]],
        [id.zMma[b],...(b>=2?[id.zReady[b-1]]:[])],S.zConvert,[64,64],2,
        {block:b,element_count_per_tile:4096,destination:"TMEM offset 320",element_type:"fp32->fp16/bf16"});
      zReady=ctrl(j,"z_convert_"+b+"_ready","Z converted update operand ready",[zw],S.cg0,
        "tau_TMEM_store_visibility + tau_z_ready_barrier",{block:b});
      id.zReady[b]=zReady;
      id.mZRelease[b]=ctrl(j,"m_z_input_"+b+"_release","M input released after Z-ready consumer wait",[zReady],S.transfer,
        "tau_pipeline_release",{block:b});
    }

    j.find(x=>x.id===mc).dependency.push(b===0?id.xMat[b]:zReady);

    const md=[k,id.xMat[b],b===0?mi:zReady,...(b?[id.zMma[b],id.mZRelease[b]]:[])];
    const mIssue=ctrl(j,"m_update_"+b+"_issue","M recurrence tcgen05 issue",md,S.transfer,
      "tau_MMA_issue",{block:b,mma_role:"warp 8",shape:{M:128,N:128,K:64},k_phases:4,
       first_block_input:b===0?"K in SMEM":undefined});
    const mt=work(j,"m_update_"+b+"_tc","M recurrence tcgen05 service","TC16",2097152,"FLOP",[mIssue],S.transfer,
      {block:b,mma_role:"warp 8",shape:{M:128,N:128,K:64},logical_flops:2097152,k_phases:4,
       operand_source:b===0?"SMEM K,X":"TMEM Z-input, SMEM X"});
    id.mUpdate[b]=ctrl(j,"m_update_"+b+"_done","M recurrence update complete",[mt],S.transfer,
      "tau_MMA_response + tau_mbarrier",{block:b});
    const msw=orderedTiles(j,"m_scale_"+b,"M scale subtile",
      [["tmem_read","TMEM",16384,"B","fp32 state read"],
       ["mul","SIMT",4096,"results","scalar scale multiply"],
       ["tmem_write","TMEM",16384,"B","fp32 scaled-state write"]],
      [id.mUpdate[b],mc],S.scale,[128,32],4,
      {block:b,element_count_per_tile:4096,fp32_results_per_tile:4096,fp32_flops_per_tile:4096,operation:"multiply"});
    id.mScale[b]=ctrl(j,"m_scale_"+b+"_done","M scaled state ready",[msw],S.cg0,
      "tau_TMEM_store_visibility + tau_mbarrier",{block:b});

    const nw=orderedTiles(j,"n_convert_"+b,"N state-to-Y-input subtile",
      [["tmem_read","TMEM",16384,"B","fp32 accumulator read"],
       ["cvt","CVT",4096,"elements","fp32-to-input conversion"],
       ["tmem_write","TMEM",8192,"B","low-precision TMEM write"]],
      [b===0?ni:id.nUpdate[b-1],v,a],S.convert,[128,32],4,
      {block:b,element_count_per_tile:4096,destination:"TMEM offset 384",element_type:"fp32->fp16/bf16"});
    id.nInput[b]=ctrl(j,"n_convert_"+b+"_ready","N low-precision Y operand ready",[nw],S.cg1,
      "tau_TMEM_store_visibility + tau_mbarrier",{block:b});
    const yi=ctrl(j,"y_mma_"+b+"_issue","Y=N*K^T tcgen05 issue",[id.nInput[b],k,id.xMat[b]],S.state,
      "tau_MMA_issue",{block:b,mma_role:"warp 11",shape:{M:128,N:64,K:128},k_phases:8});
    const yt=work(j,"y_mma_"+b+"_tc","Y=N*K^T tcgen05 service","TC16",2097152,"FLOP",[yi],S.state,
      {block:b,mma_role:"warp 11",shape:{M:128,N:64,K:128},logical_flops:2097152,k_phases:8,
       operand_source:"TMEM N-input, shared K-transpose",accumulator:"TMEM X/Y shared slot"});
    id.yMma[b]=ctrl(j,"y_mma_"+b+"_done","Y MMA complete",[yt],S.state,
      "tau_MMA_response + tau_mbarrier",{block:b});
    j.find(x=>x.id===nc).dependency.push(id.nInput[b],id.yMma[b]);
    let yw=null;
    for(let tile=0;tile<2;tile++) {
      const p=`process_y_${b}${tile===0?"":`_tile${tile}`}`;
      const base=tile===0?[id.yMma[b],id.nInput[b],v,a,nc]:[yw];
      const info={block:b,tile_index:tile,tile_count:2,inferred_register_tile_shape:[64,64],
        element_count_per_tile:4096,layout_evidence:"source partition-loop/copy-atom payload inference; not compile/PTX verified"};
      const yr=work(j,p+"_tmem_read","processY Y accumulator read","TMEM",16384,"B",base,S.y,
        {...info,element_type:"fp32"});
      const vl=work(j,p+"_v_ldsm","processY V LDSM","SMEM",8192,"B",
        [yr,id.yMma[b],id.nInput[b],v,a,nc],S.y,
        {...info,element_type:"fp16/bf16",smem_semantics:"explicit LDSM"});
      const vc=work(j,p+"_v_cvt","processY V half-to-fp32 conversion","CVT",4096,"elements",[vl],S.y,
        {...info,conversion:"fp16/bf16->fp32"});
      const ar=work(j,p+"_alpha_smem","processY per-token alpha logical reads","SMEM",16384,"B",
        [vl,id.yMma[b],id.nInput[b],v,a,nc],S.y,
        {...info,logical_smem_request_bytes:16384,element_type:"fp32",
         physical_note:"lane-varying coefficient; no broadcast credit"});
      const ym=work(j,p+"_fp32_math","processY gamma*Y + V*coef","SIMT",8192,"results",
        [yr,vc,ar,nc],S.y,{...info,fp32_results:8192,fp32_flops:12288,
         result_breakdown:"4096 multiply + 4096 FMA results",flop_breakdown:"4096 multiply + 4096 FMA*2 FLOP"});
      const yo=work(j,p+"_output_cvt","processY fp32-to-input conversion","CVT",4096,"elements",[ym],S.y,
        {...info,conversion:"fp32->fp16/bf16"});
      yw=work(j,p+"_tmem_write","processY N-update input TMEM write","TMEM",8192,"B",[yo],S.y,
        {...info,destination:"TMEM offset 384",element_type:"fp16/bf16",
         within_expression_overlap_assumption:"V conversion and lane-alpha read packets are independent after V LDSM and may overlap; FP32 math joins both. Compiler's exact same-warp issue/interleave tail remains unknown."});
    }
    id.yProc[b]=ctrl(j,"process_y_"+b+"_done","processY complete and Y accumulator released",[yw],S.cg1,
      "tau_TMEM_store_visibility + tau_y_acc_release",{block:b});
    const nsw=orderedTiles(j,"n_scale_"+b,"N scale subtile",
      [["tmem_read","TMEM",16384,"B","fp32 state read"],
       ["mul","SIMT",4096,"results","scalar scale multiply"],
       ["tmem_write","TMEM",16384,"B","fp32 scaled-state write"]],
      [id.yProc[b],id.nInput[b],nc],S.scale,[128,32],4,
      {block:b,element_count_per_tile:4096,fp32_results_per_tile:4096,fp32_flops_per_tile:4096,operation:"multiply"});
    id.nScale[b]=ctrl(j,"n_scale_"+b+"_done","N scaled state ready",[nsw],S.cg1,
      "tau_TMEM_store_visibility + tau_y_ready_barrier",{block:b});
    const yReady=ctrl(j,"y_ready_"+b,"CG1 publishes processed Y",[id.nScale[b]],S.cg1,
      "tau_y_ready_barrier",{block:b});
    const ni2=ctrl(j,"n_update_"+b+"_issue","N recurrence tcgen05 issue",
      [yReady,id.nScale[b],id.xMat[b]],S.state,"tau_MMA_issue",
      {block:b,mma_role:"warp 11",shape:{M:128,N:128,K:64},k_phases:4});
    const nt=work(j,"n_update_"+b+"_tc","N recurrence tcgen05 service","TC16",2097152,"FLOP",[ni2],S.state,
      {block:b,mma_role:"warp 11",shape:{M:128,N:128,K:64},logical_flops:2097152,k_phases:4,
       operand_source:"TMEM processed-Y input, SMEM X"});
    id.nUpdate[b]=ctrl(j,"n_update_"+b+"_done","N recurrence update complete",[nt],S.state,
      "tau_MMA_response + tau_n_acc_barrier",{block:b});

    id.kRelease[b]=ctrl(j,"k_"+b+"_release","K release after transfer update and state Y",
      [id.mUpdate[b],id.yMma[b]],S.transfer+" / "+S.state,"tau_load_k_release",{block:b,pipe_stage:b%3});
    id.vRelease[b]=ctrl(j,"v_"+b+"_release","V release after N recurrence",[id.nUpdate[b]],S.cg1,
      "tau_load_v_release",{block:b,pipe_stage:b%3});
    id.tRelease[b]=ctrl(j,"t_"+b+"_release","T release after X MMA",[id.xMma[b]],S.state,
      "tau_load_t_release",{block:b,pipe_stage:b%3});
    id.xRelease[b]=ctrl(j,"x_"+b+"_release","X shared stage release after both updates",
      [id.mUpdate[b],id.nUpdate[b]],S.transfer+" / "+S.state,"tau_x_ready_release",
      {block:b,pipe_stage:b%2});
    id.aRelease[b]=ctrl(j,"alpha_"+b+"_release","alpha stage release after both compute groups",
      [id.mScale[b],id.nUpdate[b]],S.cg0+" / "+S.cg1,"tau_alpha_stage_release",
      {block:b,pipe_stage:b%4});
  }

  // Stage acquisitions gate overwrites, not merely consumers. Add to the TMA
  // issue and shared-write nodes after all consumer release events exist.
  for(let b=3;b<L;b++) for(const key of ["k","v","t"]) {
    const rel=id[key+"Release"][b-3];
    for(const suffix of ["_issue","_smem"]) {
      const node=j.find(x=>x.id==="tma_"+key+"_"+b+suffix);
      node.dependency.push(rel);
    }
  }
  for(let b=0;b<L;b++) {
    if(b>=2) {
      const xr=j.find(x=>x.id==="x_mat_"+b+"_tmem_read");
      xr.dependency.push(id.xRelease[b-2]);
    }
    if(b>=4) {
      const ai=j.find(x=>x.id==="alpha_"+b+"_issue");
      ai.dependency.push(id.aRelease[b-4]);
    }
  }

  const last=L-1;
  const mw=orderedTiles(j,"m_output","M output subtile",
    [["tmem_read","TMEM",16384,"B","fp32 output read"],
     ["gmem","HBM",16384,"B","global output store"]],
    [id.mScale[last]],S.store,[128,32],4,
    {shape:[128,32],element_count_per_tile:4096,element_type:"fp32",
     logical_global_bytes_per_tile:16384,physical_hbm_scenario:"all logical bytes assumed to reach HBM"});
  const done=ctrl(j,"m_done_token","CG0 done token",[mw],S.cg0,
    "tau_done_barrier + tau_global_store_visibility");
  const nw=orderedTiles(j,"n_output","N output subtile",
    [["tmem_read","TMEM",16384,"B","fp32 output read"],
     ["gmem","HBM",16384,"B","global output store"]],
    [id.nUpdate[last]],S.store,[128,32],4,
    {shape:[128,32],element_count_per_tile:4096,element_type:"fp32",
     logical_global_bytes_per_tile:16384,physical_hbm_scenario:"all logical bytes assumed to reach HBM"});
  ctrl(j,"cta_finish","CTA completion",[done,nw],S.cg1,"tau_CTA_tail + tau_final_store_visibility");
  const by=new Map(j.map(x=>[x.id,x]));if(by.size!==j.length)throw Error("duplicate job");
  for(const x of j)for(const d of x.dependency)if(!by.has(d))throw Error("missing "+d+" for "+x.id);
  return j;
}

function workTotals(L) {
  return {
    L,U_FLOP:1048576,TC_FLOP_per_CTA:(9*L-2)*1048576,
    first_block_TC_FLOP:7*1048576,later_block_TC_FLOP:9*1048576,
    source_gemm_kphase_calls:28*L-8,
    gmem_logical_B_per_CTA:41216*L+131072,
    gmem_total_including_cu_seqlens_B_per_CTA:41216*L+131080,
    gmem_per_block_B:{K:16384,V:16384,T:8192,alpha:256},
    gmem_final_outputs_B:{M_f32:65536,N_f32:65536},
    gmem_small_metadata_load_B:{cu_seqlens_two_int32:8},
    smem_logical_request_B_per_CTA:108800*L,
    smem_per_block_B:{TMA_KVT_write:40960,X_materialize_write:16384,V_LDSM_read:16384,
      processY_alpha_lane_reads:32768,CG0_CG1_coeff_logical:1024,
      alpha_raw_stage_write:256,AlphaProcessor_read:256,AlphaProcessor_three_channel_write:768},
    alpha_smem_request_split_B:{lane_varying_logical:32768,block_coeff_logical:1024,
      block_coeff_broadcast_physical_estimate:32,AlphaProcessor_requests:1280},
    tmem_explicit_B_per_CTA:589824*L+114688,
    tmem_components_B:{M_N_init:131072,X_materialize_read:32768*L,
      M_N_convert_read_write:98304*(2*L-1),Z_convert_read_write:49152*(L-1),
      processY_read_write:49152*L,M_N_scale_read_write:262144*L,final_store_read:131072},
    CVT_elements_per_CTA:(64*L-24)*1024,
    CVT_per_block_elements:{X_f32_to_input:8192,M_N_f32_to_input_each:16384,
      Z_f32_to_input:8192,processY_V_to_f32:8192,processY_output_to_input:8192},
    SIMT_result_slots_per_CTA:49634*L,
    SIMT_FP32_add_sub_mul_FLOP_per_CTA:57762*L,
    SIMT_unary_negate_results:64*L,
    SIMT_per_block:{processY_results:16384,processY_FLOP:24576,
      M_N_scale_results:32768,M_N_scale_FLOP:32768,AlphaProcessor_results:482,
      AlphaProcessor_add_sub_FLOP:418,AlphaProcessor_negates:64},
    SFU_calls:{log2:64*L,exp2:128*L,total:192*L},
    shuffle_calls:{scan_and_carry:11*L,terminal_lane_broadcast:L,total:12*L},
    tmem_offsets_columns:{M:0,N:128,Z_scratch:256,M_input:320,N_input:384,X_Y_accumulator:448},
    tmem_columns_allocated:512,
    pipeline_stages:{K:3,V:3,T:3,alpha:4,X_shared:2,M_acc:1,N_acc:1,Z_acc:1,Y_acc:1,M_input:1,N_input:1},
    smem_payload_lower_bound_B:158720,barrier_storage_source_bytes_before_alignment:432
  };
}
function resourceSummary(nodes) {
  const out={};for(const n of nodes)if(n.resource!=="control") {
    const x=out[n.resource]||(out[n.resource]={demand:0,cycles:0,nodes:0,unit:n.unit});
    x.demand+=n.demand||0;x.cycles+=n.cycles||0;x.nodes++;
  }return out;
}
function blocksSummary(nodes,L) {
  const by=new Map(nodes.map(n=>[n.id,n])),out=[];
  const names=["tma_k_","tma_v_","tma_t_","alpha_"];
  for(let b=0;b<L;b++) {
    const events={};
    for(const p of names) {
      const x=by.get(p+b+"_ready");events[p.replaceAll("_","")+"ready"]=x?{start:x.start,end:x.end}:null;
    }
    for(const x of ["x_mma_","x_mat_","z_mma_","z_convert_","m_update_","m_scale_",
      "n_convert_","y_mma_","process_y_","n_scale_","n_update_"]) {
      const y=by.get(x+b+(x==="z_convert_"?"_ready":"_done"));
      const fallback=by.get(x+b+"_ready");
      const z=y||fallback;events[x.replaceAll("_","")]=z?{start:z.start,end:z.end}:null;
    }
    out.push({block:b,phase:b===0?"first/warmup":(b===L-1?"last/drain":"later/steady-body"),events});
  }return out;
}
function scenario(L,conservative=true,hbmRate=R.HBM) {
  const jobs=makeJobs(L,conservative),rates={...R,HBM:hbmRate};
  const cal=schedule(jobs,rates);validate(cal.nodes);return {jobs,cal,rates};
}

const cases=[];
for(const w of [{S:8192,C:1024,L:16},{S:16384,C:2048,L:32},{S:32768,C:4096,L:64}]) {
  const primary=scenario(w.L,false),serialTma=scenario(w.L,true),active128=scenario(w.L,false,BGPU/128);
  const W=workTotals(w.L),grid=128,gridBytes=W.gmem_total_including_cu_seqlens_B_per_CTA*grid;
  const primaryRows=primary.cal.nodes;
  const pSummary={
    model:"source_causal_async_TMA_pipeline",
    HBM_rate_B_per_SM_cycle:R.HBM,
    makespan_cycles:primary.cal.makespan_cycles,
    conditional_us_at_1965MHz:primary.cal.makespan_cycles/(F/1e6),
    task_count:primaryRows.length,
    resource_totals:resourceSummary(primaryRows),
    block_event_summary:blocksSummary(primaryRows,w.L),
    events:primaryRows
  };
  cases.push({
    workload:{B:1,H:16,D:128,S:w.S,C:w.C,L:w.L,P:8,grid_ctas:grid,active_sms_assumption:128,idle_sms:20},
    work:W,
    first_vs_later_block_DAG:{
      first:{TC_FLOP:7*1048576,ops:["X=K*T (1U)","Y=N*K^T (2U; N is zero but source MMA still runs)","M update (2U)","N update (2U)"],
        source_role_chains:{
          CG0:["wait alpha and X","materialize X subtiles (first tile also waits M initialization)","read M coefficient after X materialization","wait M update","scale M subtiles"],
          CG1:["convert zero-initialized N to Y input","wait Y MMA and N conversion; read N coefficient","processY subtiles: Y read -> V LDSM -> V convert -> alpha SMEM -> FP32 math -> output convert -> N-input write","scale N subtiles","publish Y-ready","wait N update; release V and alpha"],
          transfer_warp_8:["wait M initialization and K","wait X-ready","M update from K and X"],
          state_warp_11:["wait N initialization, K and T","X=K*T","wait X-ready and N input","Y=N*K^T","wait Y-ready","N update"],
          producer_warp_9:["issue K, then V, then T; each copy is asynchronous"]}},
      later:{TC_FLOP:9*1048576,ops:["X=K*T (1U)","Z=M*K^T (2U)","Y=N*K^T (2U)","M update (2U)","N update (2U)"],
        source_role_chains:{
          CG0:["wait alpha and X","materialize X subtiles","convert prior M to Z input subtiles","wait Z; convert Z to M-update input subtiles","read M coefficient after Z-ready (independent of transfer M-update completion)","wait M update","scale M subtiles"],
          CG1:["convert N to Y input subtiles","wait Y MMA and N conversion; read N coefficient","processY subtiles: Y read -> V LDSM -> V convert -> alpha SMEM -> FP32 math -> output convert -> N-input write","scale N subtiles","publish Y-ready","wait N update; release V and alpha"],
          transfer_warp_8:["wait K and M input","Z=M*K^T","wait Z-to-update-input ready and X-ready","M update"],
          state_warp_11:["wait prior N update (after block 0)","wait K and T","X=K*T","wait X-ready and N input","Y=N*K^T","wait Y-ready","N update"],
          producer_warp_9:["issue K, then V, then T; each copy is asynchronous"]}},
      shared_XY_TMEM_accumulator:"Y cannot overwrite shared accumulator until X is materialized; following-block X waits preceding M scale in CG0 and preceding N update in the state issuer",
      completion_order:"CG1 N output store is independent of the M done-token wait; CTA finish waits for both CG0 M done token and final CG1 N store"
    },
    hbm_sensitivity:{
      all_logical_bytes_GPU:gridBytes,
      whole_GPU_capacity_floor_cycles:Math.ceil(gridBytes/BGPU),
      representative_CTA_floor_at_Bglobal_over_148:Math.ceil(W.gmem_total_including_cu_seqlens_B_per_CTA/(BGPU/148)),
      representative_CTA_floor_at_Bglobal_over_128:Math.ceil(W.gmem_total_including_cu_seqlens_B_per_CTA/(BGPU/128)),
      cache_policy:"all logical GMEM bytes assumed to reach HBM; assumption only"
    },
    source_ordered_primary:pSummary,
    serial_TMA_packet_sensitivity:{
      description:"Alternative source-ordered packet scenario: each later TMA HBM packet waits for the prior packet SMEM write. This extra serialization can alter greedy resource arbitration and is not guaranteed to increase the reported span.",
      HBM_rate_B_per_SM_cycle:R.HBM,makespan_cycles:serialTma.cal.makespan_cycles,
      conditional_us_at_1965MHz:serialTma.cal.makespan_cycles/(F/1e6),
      task_count:serialTma.cal.nodes.length,
      resource_totals:resourceSummary(serialTma.cal.nodes)
    },
    HBM_Bglobal_over_128_sensitivity:{
      description:"Source-ordered primary calendar with only the HBM allocation changed to Bglobal/128 active CTAs.",
      HBM_rate_B_per_SM_cycle:BGPU/128,makespan_cycles:active128.cal.makespan_cycles,
      conditional_us_at_1965MHz:active128.cal.makespan_cycles/(F/1e6)
    },
    validation:{all_primary_dependencies_and_resource_exclusion_checked:true,
      all_serial_sensitivity_dependencies_and_resource_exclusion_checked:true,timing_evidence_used:false,
      unknown_tails:"symbolic in job rows; numeric ideal-extra cost is zero"}
  });
}

const model={
  schema_version:1,model_id:"B200_148SM_SOURCE_A_20260927_K2",
  status:"SOURCE_ONLY_CONDITIONAL_RESOURCE_CALENDAR",timing_evidence_inspected:false,
  strict_architectural_lower_bound:false,
  source_identity:{commit:"e77a4a0d276367895c3b50a642fd8f326c03fb72",
    source:"../ablation_20260908/source_original/gated_delta_net_cp.py",
    source_sha256:"CB9A174F6FB02D6A69415E34DBE1BF799BF283E2A46747B39D4C538D83919B8B",
    alpha_sha256:"7F40522B7C7B49F4FFED29D3998BCCEA56F2ACC22224ECD0B4429E3F0CE0BEFB",
    varlen_sha256:"C9CDA29F3E9D0105D91326C4C670FEACE92F988ED9FDE61DCEA3035FD6E704AB"},
  static_clean_cubin_resources:{evidence:"static_artifacts/resource_usage.json; untimed clean compilation",
    threads:384,registers_per_thread:168,registers_per_cta:64512,cta_per_sm:1,static_shared_B:1024,tmem_columns:512},
  hardware:{sms:148,reference_clock_hz:F,grid_ctas:128,balanced_one_wave_assumption:true},
  approved_common_constants:{model_id:"B200_148SM_SOURCE_A_20260927",
    TC16_FLOP_per_SM_cycle:R.TC16,SMEM_B_per_SM_cycle:R.SMEM,TMEM_B_per_SM_cycle:R.TMEM,
    CVT_elements_per_SM_cycle:R.CVT,SIMT_scalar_results_per_SM_cycle:R.SIMT,
    HBM_B_per_GPU_cycle:BGPU,primary_HBM_B_per_representative_SM_cycle:R.HBM,
    HBM_B_per_active_SM_cycle_sensitivity:BGPU/128,
    unknown_tails:"TMA/MMA issue and response, SFU/shuffle, barrier, memory response, dispatch and final-store tails are symbolic; ideal numeric extra is zero"},
  source_roles:[
    {warps:"0-3",role:"CG0 M recurrence, M scale/conversion/output",source:S.cg0},
    {warps:"4-7",role:"CG1 N recurrence, processY, N scale/conversion/output",source:S.cg1},
    {warps:"8",role:"transfer MMA: later-block Z then M update",source:S.transfer},
    {warps:"9",role:"TMA producer issues K, V, T in that order",source:S.tma},
    {warps:"10",role:"alpha load and AlphaProcessor",source:S.alphaLoad+"; "+S.alphaProc},
    {warps:"11",role:"state MMA: X, Y, N update",source:S.state}],
  buffers_pipeline_and_barriers:{
    tmem_offsets_columns:{M:0,N:128,Z_scratch:256,M_input:320,N_input:384,X_Y_accumulator:448},
    tmem_allocated_columns:512,pipeline_stages:{K:3,V:3,T:3,alpha:4,X_shared:2,
      M_acc:1,N_acc:1,Z_acc:1,Y_acc:1,M_input:1,N_input:1},
    source_release_edges:{
      K:"wait for transfer M update and state Y MMA",
      V:"wait for N recurrence update",
      T:"wait for X MMA",
      alpha:"wait for CG0 M scale and CG1 N update",
      X_shared:"wait for both M and N update",
      XY_accumulator:"materialize X before Y overwrites; following X waits previous N update",
      Z_scratch:"next Z waits previous Z-to-M-input conversion",
      M_N_inputs_and_accumulators:"single-stage dependencies retained"},
    stage_payload_lower_bound_B:158720,barrier_storage_bytes_before_struct_alignment:432,
    shared_layout_caveat:"exact layout padding is not asserted; K-transpose aliases K shared storage"},
  source_locations:S,scheduling:{
    cycle_formula:"integer ceil(demand/rate) for each positive-demand node",
    primary:"source-causal asynchronous TMA issue order; per-packet HBM then SMEM readiness, later TMA HBM may overlap earlier packet SMEM; deterministic earliest-ready list scheduling with declaration-order ties (not makespan optimal)",
    serial_packet_sensitivity:"same source DAG plus each next TMA HBM waiting for prior packet SMEM; ordering can alter greedy arbitration and need not monotonically increase span",
    representative_CTA:"Bglobal/148 HBM allocation; one balanced CTA on each of 128 active SMs, do not multiply CTA span by 128",
    global_floor:"ceil(total_grid_logical_HBM_bytes / Bglobal) shown separately",
    logical_smem:"primary charges logical alpha and coefficient requests; broadcast estimate reported separately, not deducted",
    tensor_feed:"do not double charge tensor-internal MMA operand feed against generic SMEM/TMEM",
    no_cache_credit:"logical GMEM equals modeled HBM only under explicit all-bytes-HBM scenario"},
  helper_subtile_model:{
    fp32_values_per_subtile:4096,fp32_bytes_per_subtile:16384,
    inferred_register_tile_shapes:{materialize_X:[64,64],convert_M_N:[128,32],convert_Z:[64,64],
      processY:[64,64],scale_M_N:[128,32],store_M_N:[128,32]},
    source_layout_basis:"helper copy-atom repetition and partition-loop counts from pinned source; logical packet geometry inferred, not clean-CUBIN/PTX verified",
    copy_atom_basis:{Ld16x256b_Repetition8:"16 KiB FP32 packet; inferred 64x64 logical view",
      Ld32x32b_Repetition32:"16 KiB FP32 packet; inferred CTA view 128x32 per column subtile"},
    helper_tile_counts:{materialize_X_128x64:2,convert_M_N_128x128:4,convert_Z_128x64:2,
      processY_128x64:2,scale_M_N_128x128:4,output_M_N_128x128:4},
    ordering:"within each inferred subtile, helper source operation phases are chained; the next subtile begins after the prior subtile write/store",
    processY_phase_model:"per subtile: 16 KiB Y TMEM read -> 8 KiB V LDSM -> 4096 V conversions and 16 KiB logical alpha reads (independent packet nodes after V read) -> 8192 SIMT result slots (12288 FLOP) joining both inputs -> 4096 output conversions -> 8 KiB N-input TMEM write",
    streaming_caveat:"These are aggregate per-subtile service packets, not instruction-level issue order. V conversion and lane-wise alpha accesses are allowed to overlap after V LDSM as a modeling assumption; same-warp compiler issue/interleave tails remain unknown. No cross-subtile streaming is modeled.",
    alpha_phase_model:"raw SMEM read -> raw+epsilon SIMT -> 64 log2 SFU -> 5 offsets x 2 columns each [shuffle event -> (32-offset) active-lane adds] -> carry shuffle -> 32 carry adds -> terminal-lane shuffle -> for each of 2 columns: cumsum store -> cumulative exp2 -> cumprod store -> 32 end-log subtraction results -> end exp2 -> 32 unary negates -> negative-end store -> alpha stage commit",
    alpha_phase_limits:"All SFU and shuffle service tails remain symbolic and contribute zero numeric cycles; SIMT is charged using the approved 128 scalar-results/SM-cycle ideal rate, with per-phase integer ceiling."
  },
  cases
};

const md=[];
md.push("# K2 M/N source-only cycle model","");
md.push("Pinned source e77a4a0d276367895c3b50a642fd8f326c03fb72; B200 148 SM. Untimed, source-derived conditional resource calendar. No target kernel timing or NCU readings were inspected or fitted.","");
md.push("## Roles and block graph","");
md.push("The CTA has 12 warps: 0-3 CG0; 4-7 CG1; 8 transfer MMA; 9 TMA; 10 alpha; 11 state MMA. It processes L=C/64 tiles. The grid has 128 CTAs and is modeled as one balanced wave on 128 SMs (20 idle). Clean static metadata is 384 threads, 168 registers/thread (64,512/CTA), one CTA/SM and 512 TMEM columns.","");
md.push("Each block executes X=K*T (U=1,048,576 TC FLOP), Y=N*K^T (2U), M update (2U), and N update (2U). The first-block Y executes even though N was zero-initialized, so first block is 7U. Later blocks add Z=M*K^T (2U), hence 9U. CTA total is (9L-2)U. GMEM tensor payload is exactly 41,216L+131,072 B; two cu_seqlens int32 reads add 8 B separately in the HBM calendar. X and Y alias one TMEM accumulator; X materialization precedes Y overwrite, and next X waits prior N update.","");
md.push("The source helpers are packetized at 4,096 FP32 values / 16 KiB per packet: Ld16x256b/Repetition8 helpers use an inferred 64x64 logical view (X materialization, Z conversion, processY), while Ld32x32b/Repetition32 helpers use an inferred 128x32 view (M/N conversion, M/N scale, output store). Counts follow pinned helper partition loops; geometries are source-layout inferences, not clean-CUBIN/PTX proof. Packet counts are X/Z/processY 2 each and M/N conversion, scale, and output 4 each. In processY, V conversion and alpha accesses are separate resource packets after V LDSM and may overlap; FP32 math joins both. This allows a possible within-expression overlap without claiming same-warp issue timing; no cross-subtile streaming is modeled.","");
md.push("| L | TC FLOP/CTA | CVT elements | SIMT result slots | FP32 add/sub/mul FLOP | SMEM logical B | TMEM explicit B | GMEM logical B | Primary cycles | Serial-packet sensitivity cycles | Bglobal/128 sensitivity cycles | HBM GPU floor |");
md.push("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
for(const c of cases){const w=c.work;md.push("| "+w.L+" | "+w.TC_FLOP_per_CTA+" | "+w.CVT_elements_per_CTA+" | "+w.SIMT_result_slots_per_CTA+" | "+w.SIMT_FP32_add_sub_mul_FLOP_per_CTA+" | "+w.smem_logical_request_B_per_CTA+" | "+w.tmem_explicit_B_per_CTA+" | "+w.gmem_logical_B_per_CTA+" | "+c.source_ordered_primary.makespan_cycles+" | "+c.serial_TMA_packet_sensitivity.makespan_cycles+" | "+c.HBM_Bglobal_over_128_sensitivity.makespan_cycles+" | "+c.hbm_sensitivity.whole_GPU_capacity_floor_cycles+" |");}
md.push("","processY has one multiply plus one FMA per output: 16,384 result slots but 24,576 semantic FLOP per 8,192 outputs, plus 8,192 V half-to-FP32 and 8,192 FP32-to-half conversions. Alpha has 64 log2 and 128 exp2 SFU calls; 10 scan shuffles, one carry shuffle, and one terminal-lane shuffle. Its 482 scalar result slots are 64 raw-plus-epsilon adds, 290 prefix/carry adds, 64 end-log subtractions, and 64 unary negations; ordinary FP32 add/sub FLOP is 418 and unary negation is reported separately. SFU/shuffle rates remain unknown and their numeric service is zero in the ideal calendar.","");
md.push("SMEM counts are logical source requests: per block K/V/T TMA writes 40,960 B; X shared write 16,384 B; V LDSM 16,384 B; per-token alpha 32,768 B; two coefficient vectors 1,024 B logical (32 B physical broadcast estimate); and AlphaProcessor 256 B raw write +256 B read +768 B writes. The primary does not subtract the broadcast estimate. TC operand feed is not charged again to generic SMEM/TMEM.","");
md.push("Corrected source edges: each block's X materialization waits the prior block's M scale; M coefficient reads wait on X materialization in block 0 and Z conversion-ready later; N coefficient reads wait on N conversion and Y MMA; processY's V/alpha inputs occur only after those prerequisites. N global output is not gated by the M done token; CTA completion waits for both outputs.","");
md.push("## Cycle interpretation","");
md.push("Primary calendar uses HBM=Bglobal/148 per representative CTA and integer ceil service per operation packet. TMA issue order is K,V,T per block, while independent staged packets may overlap: each packet has HBM then SMEM readiness, but the next packet HBM does not wait for prior SMEM completion. The serial-packet sensitivity adds that wait. Because the deterministic earliest-ready scheduler is a list heuristic, not a makespan optimizer, the extra ordering can change arbitration and can produce a numerically shorter span; the two values are distinct conditional scenarios, not monotone bounds. Unknown issue/response/barrier/SFU/shuffle/memory tails remain symbolic; their ideal numeric extra is zero. No scenario is a measured prediction or rigorous hardware lower bound.","");
md.push("The exact unrolled async-primary job and event calendars for L=16/32/64, with per-node shape, bytes/FLOP/result demand, dependencies, priority, start/end and source locations, are in k2_model.json. That file also contains serial-packet and HBM-share sensitivities and a separate whole-GPU capacity floor.","");
md.push("Source map: "+Object.entries(S).map(([k,v])=>k+"="+v).join("; ")+".","");
fs.writeFileSync(new URL("./k2_model.json",import.meta.url),JSON.stringify(model,null,2)+"\n");
fs.writeFileSync(new URL("./k2_analysis.md",import.meta.url),md.join("\n"));
console.log(JSON.stringify(cases.map(c=>({L:c.work.L,primary:c.source_ordered_primary.makespan_cycles,
  primary_us:c.source_ordered_primary.conditional_us_at_1965MHz,serial_tma:c.serial_TMA_packet_sensitivity.makespan_cycles,
  hbm128:c.HBM_Bglobal_over_128_sensitivity.makespan_cycles,gpu_floor:c.hbm_sensitivity.whole_GPU_capacity_floor_cycles,
  jobs:c.source_ordered_primary.task_count})),null,2));
