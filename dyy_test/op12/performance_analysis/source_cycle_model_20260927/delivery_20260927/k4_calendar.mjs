// Source-only K4 conditional resource calendar; no measured timings.
// Exported builder returns a complete DAG and resource schedule. Run with Node.
export function buildK4Calendar(L,options={}) {
 const f=1965e6,SM=148,R16=2.25e15/(SM*f),BHG=7.7e12/f;
 const cfg={share:148,gateDemand:"source_lane_requests",eta64:1,includeTensorFeed:false,terminalRead:false,policy:"earliest_service_start_then_earliest_ready_then_declaration_order",...options};
 const rates={HBM:BHG/cfg.share,SMEM:128,TMEM:512,CVT:64,SIMT:128,TC:R16};
 const tasks=[],byId=new Map(); let declaration=0;
 const refs={}; const ends={};
 function add(id,stage,resource,work,deps=[],block=null,source=[],extra={}) {
  deps=[...new Set(deps.filter(Boolean))];
  if(byId.has(id))throw Error("duplicate "+id);
  const rate=resource?rates[resource]:null;
  const duration=resource?Math.ceil(work/rate):0;
  const t={id,stage,resource,work,rate,duration,predecessors:deps,block,source_lines:source,declaration:declaration++,symbolic_extra:"",...extra};
  tasks.push(t);byId.set(id,t); return id;
 }
 function chain(prefix,stage,phases,deps,block,source=[]) {
   let last; phases.forEach((ph,j)=>last=add(prefix+"_"+j,stage,ph[0],ph[1],j?[last]:deps,block,source,ph[2]||{})); return last;
 }
 const reuse=cfg.gateDemand==="per_warp_reuse_hypothesis";
 const gt=reuse?1024:16640,gq=gt,gend=reuse?16:512,gpost=reuse?2064:66048;
 add("setup","CTA setup",null,0,[],null,[958,1237],{symbolic_extra:"tau_setup"});
 let initPrev="setup";
 for(let s=0;s<4;s++){
  const hb=add("init_hb_"+s,"Initial state G2R", "HBM",16384,[initPrev],null,[2332,2338]);
  initPrev=add("init_tm_"+s,"Initial state R2T","TMEM",16384,[hb],null,[2346,2355]);
 }
 refs.init=initPrev;
 for(let i=0;i<L;i++) {
  const gp=[i?`g_store_${i-1}`:"setup"];
  if(i>=5)gp.push(`o_store_done_${i-5}`);
  if(i>=5)gp.push(`qk_epi_done_${i-5}`,`cg1_gate_done_${i-5}`);
  add(`g_load_${i}`,"Gate scalar load","HBM",256,gp,i,[1678,1701]);
  add(`g_add_${i}`,"Gate scan arithmetic","SIMT",354,[`g_load_${i}`],i,[1703,1721],{symbolic_extra:"64*tau_log2+64*tau_exp2+11*tau_shuffle+tau_single_warp_issue"});
  add(`g_store_${i}`,"Gate channels publish","SMEM",512,[`g_add_${i}`],i,[1724,1732],{symbolic_extra:"tau_gate_commit"});
  const types=[["k",16384],["q",16384],["v",16384],["t",8192]];
  for(let j=0;j<types.length;j++){
   const [type,b]=types[j];
   const dep=[j?`load_${types[j-1][0]}_issue_${i}`:i?`load_t_issue_${i-1}`:"setup"];
   if(type==="k"&&i>=3)dep.push(`qk_done_${i-3}`,`ds_done_${i-3}`);
   if(type==="q"&&i>=2)dep.push(`qk_done_${i-2}`,`qs_done_${i-2}`);
   if(type==="v"&&i>=3)dep.push(`nv_wait_release_v_${i-3}`);
   if(type==="t"&&i>=2)dep.push(`t_transform_done_${i-2}`);
   add(`load_${type}_issue_${i}`,`TMA ${type.toUpperCase()} issue`,null,0,dep,i,[1484,1509],{symbolic_extra:"tau_tma_issue"});
   add(`load_${type}_hb_${i}`,`TMA ${type.toUpperCase()} HBM service`,"HBM",b,[`load_${type}_issue_${i}`],i,[1484,1509]);
   add(`load_${type}_ready_${i}`,`TMA ${type.toUpperCase()} SMEM write / ready`,"SMEM",b,[`load_${type}_hb_${i}`],i,[1484,1509],{symbolic_extra:"tau_tma_response"});
  }
  const qkPred=[`load_k_ready_${i}`,`load_q_ready_${i}`,i?`qk_issue_${i-1}`:"setup"];
  if(i>=2)qkPred.push(`qk_epi_done_${i-2}`);
  add(`qk_issue_${i}`,"QK issue",null,0,qkPred,i,[1767,1780],{pair:Math.floor(i/2),pair_member:i%2,symbolic_extra:"tau_mma_issue"});
  let qkFeed=`qk_issue_${i}`;
  if(cfg.includeTensorFeed)qkFeed=add(`qk_feed_${i}`,"QK SMEM feed","SMEM",32768,[qkFeed],i,[1773,1779]);
  add(`qk_done_${i}`,"QK tensor service","TC",1048576/cfg.eta64,[qkFeed,i?`qk_done_${i-1}`:"setup"],i,[1773,1780],{logical_flops:1048576,eta_shape:cfg.eta64,atom_shape:[64,64,16],atoms:8,symbolic_extra:"tau_mma_response"});
  const tp=[`g_store_${i}`,`load_t_ready_${i}`,i?`qk_epi_done_${i-1}`:"setup"];
  if(i>=3)tp.push(`nv_done_${i-3}`);
  let tt=chain(`t_transform_${i}`,"T gate transform",[
   ["SMEM",8192],["SMEM",gt],["SIMT",6176],["CVT",8192],["SMEM",8192]
  ],tp,i,[1836,1900]);
  add(`t_transform_done_${i}`,"Ainv ready / T released",null,0,[tt],i,[1897,1900],{symbolic_extra:"2080*tau_exp2+tau_t_predicate+tau_named_barrier+tau_shared_fence"});
  const qe=[`t_transform_done_${i}`,`qk_done_${i}`];
  if(i>=2)qe.push(`o_done_${i-2}`);
  let qt=chain(`qk_epi_${i}`,"QK epilogue",[
   ["TMEM",16384],["SMEM",gq],["SIMT",10272],["CVT",4096],["SMEM",8192]
  ],qe,i,[1902,1942]);
  add(`qk_epi_done_${i}`,"QK operand ready / CG0 gate released",null,0,[qt],i,[1939,1942],{symbolic_extra:"2080*tau_exp2+tau_qk_predicate+tau_shared_fence+tau_commit_release"});
  const sp=[`g_store_${i}`,i?`ds_done_${i-1}`:refs.init,i?`o_stage_done_${i-1}`:"setup"];
  add(`state_endgate_${i}`,"Load end gate before state publication","SMEM",gend,sp,i,[2911,2919]);
  let st=chain(`state_publish_${i}`,"State publication",[
   ["TMEM",65536],["CVT",16384],["TMEM",32768]
  ],[`state_endgate_${i}`],i,[2926,2940]);
  add(`state_io_ready_${i}`,"Undecayed state input ready",null,0,[st],i,[2939,2940],{symbolic_extra:"tau_tmem_fence_commit"});
  let sd=chain(`state_decay_${i}`,"State decay",[
   ["SIMT",16384],["TMEM",65536]
  ],[`state_io_ready_${i}`],i,[2972,2985]);
  add(`state_decay_done_${i}`,"Decayed state accumulator released",null,0,[sd],i,[2984,2985],{symbolic_extra:"tau_tmem_fence_release"});
  let gg=chain(`cg1_gate_${i}`,"CG1 gate expansion",[
   ["SMEM",gpost],["SIMT",8192]
  ],[`state_decay_done_${i}`],i,[2986,3004]);
  add(`cg1_gate_done_${i}`,"CG1 gate arrays ready / gate released",null,0,[gg],i,[3002,3004],{symbolic_extra:`${reuse?256:8192}*tau_exp2+tau_gate_release`});
  function mma(kind,flop,smemFeed,tmemFeed,issueDeps,tcPrev,source) {
    add(`${kind}_issue_${i}`,`${kind.toUpperCase()} issue`,null,0,issueDeps,i,source,{symbolic_extra:"tau_mma_issue"});
    let last=`${kind}_issue_${i}`;
    if(cfg.includeTensorFeed){
      if(smemFeed)last=add(`${kind}_smem_feed_${i}`,`${kind.toUpperCase()} SMEM feed`,"SMEM",smemFeed,[last],i,source);
      if(tmemFeed)last=add(`${kind}_tmem_feed_${i}`,`${kind.toUpperCase()} TMEM feed`,"TMEM",tmemFeed,[last],i,source);
    }
    return add(`${kind}_done_${i}`,`${kind.toUpperCase()} tensor service`,"TC",flop,[last,tcPrev],i,source,{logical_flops:flop,atom_shape:kind==="ds"?[128,128,16]:[128,64,16],atoms:(kind==="ks"||kind==="qs")?8:4,symbolic_extra:"tau_mma_response"});
  }
  const ksp=[`load_k_ready_${i}`,`load_q_ready_${i}`,`state_io_ready_${i}`,i?`ds_issue_${i-1}`:"setup"];
  if(i)ksp.push(`nv_acc_release_${i-1}`);
  mma("ks",2097152,16384,32768,ksp,i?`ds_done_${i-1}`:"setup",[2552,2570]);
  mma("qs",2097152,16384,32768,[`ks_issue_${i}`,`state_io_ready_${i}`,`load_q_ready_${i}`,i?`o_stage_done_${i-1}`:"setup"],`ks_done_${i}`,[2572,2585]);
  add(`resid_vread_${i}`,"Residual V SMEM read","SMEM",16384,[`cg1_gate_done_${i}`,`load_v_ready_${i}`],i,[3008,3018]);
  add(`resid_ksread_${i}`,"Residual KS TMEM read","TMEM",32768,[`resid_vread_${i}`,`ks_done_${i}`],i,[3021,3027]);
  add(`resid_ksmul_${i}`,"Residual gamma*KS / KS accumulator release","SIMT",8192,[`resid_ksread_${i}`],i,[3028,3030]);
  add(`resid_cast_${i}`,"Residual KS narrowing","CVT",8192,[`resid_ksmul_${i}`],i,[3031,3032]);
  add(`resid_sub_${i}`,"Residual V-KS","SIMT",8192,[`resid_cast_${i}`],i,[3031,3032]);
  add(`resid_ready_${i}`,"Residual VKS TMEM publication","TMEM",16384,[`resid_sub_${i}`],i,[3033,3039],{symbolic_extra:"tau_tmem_fence_commit"});
  let qs=chain(`qs_scale_${i}`,"QS in-place scale",[
   ["TMEM",32768],["SIMT",16384],["TMEM",32768]
  ],[`resid_ready_${i}`,`qs_done_${i}`],i,[3041,3056]);
  add(`qs_scaled_${i}`,"QS scaled / accumulator released",null,0,[qs],i,[3055,3056],{symbolic_extra:"tau_tmem_fence_release"});
  mma("nv",1048576,8192,16384,[`qs_issue_${i}`,`resid_ready_${i}`,`t_transform_done_${i}`,`resid_ksmul_${i}`],`qs_done_${i}`,[2587,2600]);
  add(`nv_wait_release_v_${i}`,"Wait NV then release V ring",null,0,[`qs_scaled_${i}`,`nv_done_${i}`],i,[3058,3059],{symbolic_extra:"tau_wait_release"});
  let nr=`nv_wait_release_v_${i}`;
  for(let s=0;s<2;s++)nr=chain(`nv_readcast_${i}_${s}`,"NV read/cast subtile",[
    ["TMEM",16384],["CVT",4096]
  ],[nr],i,[3062,3071]);
  add(`nv_acc_release_${i}`,"NV accumulator released",null,0,[nr],i,[3071],{symbolic_extra:"tau_release"});
  let dd=`nv_acc_release_${i}`;
  for(let s=0;s<2;s++)dd=add(`delta_mul_${i}_${s}`,"Delta scale subtile","SIMT",4096,[dd],i,[3073,3076]);
  let np=dd;
  for(let s=0;s<2;s++)np=chain(`nv_delta_publish_${i}_${s}`,"NV/Delta fixed-input publication subtile",[
    ["TMEM",8192],["CVT",4096],["TMEM",8192]
  ],[np],i,[3083,3099]);
  add(`nv_delta_ready_${i}`,"NV and Delta full-tile ready",null,0,[np],i,[3097,3099],{symbolic_extra:"tau_tmem_fence_two_commits"});
  mma("o",1048576,8192,16384,[`nv_issue_${i}`,`nv_delta_ready_${i}`,`qk_epi_done_${i}`,`qs_scaled_${i}`],`nv_done_${i}`,[2602,2617]);
  mma("ds",2097152,16384,16384,[`o_issue_${i}`,`nv_delta_ready_${i}`,`state_decay_done_${i}`,`load_k_ready_${i}`],`o_done_${i}`,[2619,2634]);
  const op=[`o_done_${i}`,`nv_delta_ready_${i}`];
  if(i>=2)op.push(`o_store_done_${i-2}`);
  let os=chain(`o_stage_${i}`,"O staging",[
    ["TMEM",32768],["CVT",8192],["SMEM",16384]
  ],op,i,[3103,3121]);
  add(`o_stage_done_${i}`,"O SMEM ready / QS-O accumulator released",null,0,[os],i,[3119,3121],{symbolic_extra:"tau_shared_fence_commit_release"});
  const osp=[`o_stage_done_${i}`,i?`o_store_done_${i-1}`:"setup"];
  if(i+4<L)osp.push(`g_store_${i+4}`);
  add(`o_store_issue_${i}`,"O TMA store issue",null,0,osp,i,[1567,1588,3163,3194],{symbolic_extra:"tau_tma_issue"});
  add(`o_store_smem_${i}`,"O TMA SMEM read service","SMEM",16384,[`o_store_issue_${i}`],i,[3187,3198]);
  add(`o_store_hb_${i}`,"O TMA HBM write service","HBM",16384,[`o_store_smem_${i}`],i,[3187,3198]);
  add(`o_store_done_${i}`,"O TMA store wait and O slot release",null,0,[`o_store_hb_${i}`],i,[3196,3200],{symbolic_extra:"tau_tma_response_commit_wait0_release"});
 }
 const terminalPred=[`ds_done_${L-1}`,`o_stage_done_${L-1}`];
 const terminal=add("terminal","Final state wait/release",cfg.terminalRead?"TMEM":null,cfg.terminalRead?65536:0,terminalPred,null,[1352,1369,2417,2460],{symbolic_extra:"tau_state_tail_dealloc"});
 add("finish","CTA completion",null,0,[terminal,`o_store_done_${L-1}`,`qk_epi_done_${L-1}`],null,[1281,1282,1370,1374,1510,1513,1589],{symbolic_extra:"tau_pipeline_tails"});
 for(const t of tasks)for(const dep of t.predecessors)if(!byId.has(dep))throw Error(`missing ${dep} for ${t.id}`);
 const pending=new Set(tasks.map(t=>t.id)),done=new Map(),resources=new Map(),resourceLast=new Map();
 const execution=[];
 while(pending.size) {
   let best=null,key=null;
   for(const id of pending){
     const t=byId.get(id);
     if(t.predecessors.some(d=>!done.has(d)))continue;
     const ready=t.predecessors.reduce((a,d)=>Math.max(a,done.get(d).finish),0);
     const start=Math.max(ready,t.resource?(resources.get(t.resource)||0):0);
     const k=[start,t.duration===0?0:1,ready,t.declaration];
     if(!key||k.some((v,i)=>v!==key[i]&&k.slice(0,i).every((x,j)=>x===key[j])&&v<key[i])){best=t;key=k;}
   }
   if(!best)throw Error("DAG cycle: "+[...pending].slice(0,15));
   const ready=best.predecessors.reduce((a,d)=>Math.max(a,done.get(d).finish),0);
   const start=Math.max(ready,best.resource?(resources.get(best.resource)||0):0),finish=start+best.duration;
   const previous_resource_job=best.resource?(resourceLast.get(best.resource)||null):null;
   const critical_candidates=[...best.predecessors.map(id=>({id,finish:done.get(id).finish,kind:"DAG"})),...(previous_resource_job?[{id:previous_resource_job,finish:done.get(previous_resource_job).finish,kind:"resource"}]:[])];
   const dominant=critical_candidates.filter(p=>p.finish===start).map(p=>p.id);
   const row={...best,serial:execution.length+1,ready,start,finish,previous_resource_job,dominant_predecessors:dominant,start_formula:"MAX(predecessor_finish,previous_resource_job_finish)",finish_formula:"start+CEILING(work/rate,1)",cycles:best.duration};
   execution.push(row);done.set(best.id,row);pending.delete(best.id);
   if(best.resource){resources.set(best.resource,finish);resourceLast.set(best.resource,best.id);}
 }
 const sums={}; for(const t of execution)if(t.resource)sums[t.resource]=(sums[t.resource]||0)+t.duration;
 let criticalId="finish";const crit=[];
 while(criticalId){const t=done.get(criticalId);crit.push({id:t.id,stage:t.stage,resource:t.resource,start:t.start,finish:t.finish,duration:t.duration});criticalId=t.dominant_predecessors[0]||null;}
 crit.reverse();
 const blocks=Array.from({length:L},(_,i)=>({
   block:i,pair:Math.floor(i/2),pair_member:i%2,
   state_start:done.get(`state_endgate_${i}`).start,state_io_ready:done.get(`state_io_ready_${i}`).finish,
   qk_done:done.get(`qk_done_${i}`).finish,Ainv_ready:done.get(`t_transform_done_${i}`).finish,QK_operand_ready:done.get(`qk_epi_done_${i}`).finish,
   KS_done:done.get(`ks_done_${i}`).finish,QS_done:done.get(`qs_done_${i}`).finish,
   residual_ready:done.get(`resid_ready_${i}`).finish,NV_done:done.get(`nv_done_${i}`).finish,NV_Delta_ready:done.get(`nv_delta_ready_${i}`).finish,
   Ointra_done:done.get(`o_done_${i}`).finish,dS_done:done.get(`ds_done_${i}`).finish,O_staged:done.get(`o_stage_done_${i}`).finish,O_stored:done.get(`o_store_done_${i}`).finish,
   state_start_interval:i?done.get(`state_endgate_${i}`).start-done.get(`state_endgate_${i-1}`).start:null,
   store_interval:i?done.get(`o_store_done_${i}`).finish-done.get(`o_store_done_${i-1}`).finish:null
 }));
 const pairs=Array.from({length:L/2},(_,j)=>({pair:j,blocks:[2*j,2*j+1],second_block_stored:blocks[2*j+1].O_stored,pair_store_interval:j?blocks[2*j+1].O_stored-blocks[2*j-1].O_stored:null}));
 let checked=0;for(const t of execution){
  for(const dep of t.predecessors){if(t.start<done.get(dep).finish)throw Error("causality");checked++;}
  if(t.previous_resource_job&&t.start<done.get(t.previous_resource_job).finish)throw Error("resource overlap");
  if(t.finish-t.start!==t.duration)throw Error("duration");
 }
 return {schema:"K4 causal packet calendar v2 (S2G direction corrected)",L,C:L*64,config:cfg,rates,total_cycles:done.get("finish").finish,total_us:done.get("finish").finish/(f/1e6),task_count:execution.length,
  resource_busy_cycles:sums,resource_last_finish:Object.fromEntries(resources),validation:{all_predecessors_satisfied:true,no_same_resource_overlap:true,checked_dependency_edges:checked},
  initial_state_ready:done.get(refs.init).finish,gate0_ready:done.get("g_store_0").finish,gate4_ready:done.get("g_store_4").finish,
  first_output_stored:blocks[0].O_stored,first_pair_stored:pairs[0].second_block_stored,last_pair_store_interval:pairs.at(-1).pair_store_interval,
  blocks,pairs,critical_path:crit,execution
 };
}

export const COMMON_SCENARIO = Object.freeze({f_ref:1965e6,SM:148,R16:2.25e15/(148*1965e6),B_HBM_gpu:7.7e12/1965e6});

if (typeof process !== 'undefined' && process.argv[1] && process.argv[1].endsWith('k4_calendar.mjs')) {
  for (const L of [16,32,64]) {
    const r=buildK4Calendar(L);
    console.log(JSON.stringify({L,total_cycles:r.total_cycles,total_us:r.total_us,nodes:r.task_count,validated:r.validation,resource_busy:r.resource_busy_cycles}));
  }
}
