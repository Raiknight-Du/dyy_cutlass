import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {schedule} from './calendar.mjs';
import {buildK4Calendar} from './k4_calendar.mjs';
const root=path.dirname(fileURLToPath(import.meta.url));
const read=n=>JSON.parse(fs.readFileSync(path.join(root,n),'utf8'));
const out={id:'B200_148SM_SOURCE_A_20260927',timing_used_for_model:false,calendars:[],stage_tables:{K1:[],K2:[],K3:[],K4:[]},k1_excel_formulas:[],sensitivity:[],review:[]};
const k1=read('k1_model.json'),k3=read('k3_model.json'),k4=read('k4_model.json');
const rateMap={B_share_byte_per_sm_cycle_optional:'HBM',B_smem_byte_per_sm_cycle:'SMEM',R16_flop_per_sm_cycle_times_rho_HMMA:'HMMA',R_scalar_results_per_sm_cycle:'SIMT',R_cvt_elements_per_sm_cycle:'CVT',R_warp_issue_instructions_per_sm_cycle:'ISSUE'};
const rates={HBM:7700000000000/1965000000/148,SMEM:128,TMEM:512,CVT:64,SIMT:128,HMMA:2250000000000000/148/1965000000,ISSUE:4,TC16:2250000000000000/148/1965000000,TF32:1100000000000000/148/1965000000,control:1};
function topo(list){
 const map=new Map(list.map(n=>[n.id,n])),done=new Set(),sorted=[];
 while(sorted.length<list.length){let moved=false;for(const n of list){if(done.has(n.id))continue;for(const p of n.dependency)if(!map.has(p))throw Error('Missing dependency '+p+' in '+n.id);if(n.dependency.every(p=>done.has(p))){sorted.push(n);done.add(n.id);moved=true;}}if(!moved)throw Error('Dependency/resource-order cycle');}return sorted;
}
function verify(c,expected){const end=new Map(),last={};for(const n of c.nodes){let start=0;for(const p of n.dependency)start=Math.max(start,end.get(p));start=Math.max(start,last[n.resource]??0);const rate=n.resource==='HBM'?7700000000000/1965000000/c.HBM_divisor:n.resource==='TC16'&&n.eta64?rates.TC16*c.eta_FP16:n.resource==='TF32'?rates.TF32*c.eta_TF32:rates[n.resource]??1;const finish=start+Math.ceil(n.demand/rate);end.set(n.id,finish);if(n.demand>0)last[n.resource]=finish;}const span=Math.max(...end.values());c.expected_cycles=span;if(expected!==undefined&&span!==expected)throw Error(c.key+' formula reconciliation '+span+' != '+expected);return c;}
const sourceOf=stage=>k1.dag_nodes.find(n=>n.id===stage)?.source??'delta_rule_cp_sm120.py:321-477; collective_inverse_hmma.py:14-404';
for(const spec of k1.synchronized_cohort_scenario.substep_specs){for(const ss of spec.substeps){out.stage_tables.K1.push({stage:spec.stage,operation:ss.id,shape:'one 64-token block/head',demand:rateMap[ss.rate_key]?ss.per_cta_demand:0,resource:rateMap[ss.rate_key]??'control',dependency:spec.combine,parallel:spec.combine.includes('max')?'element-stream ideal overlap; source RAW retained symbolically':'independent CTA payload only',overlap:spec.combine,source:sourceOf(spec.source_node??spec.source_nodes?.[0]),confidence:rateMap[ss.rate_key]?'source demand; rate/streaming assumption':'symbolic unpriced operation: '+ss.per_cta_demand});}}
for(let cohort=1;cohort<=8;cohort++){
 const nodes=[];let previous=[];
 const add=(spec,ss,deps)=>{const id=spec.stage+'.'+ss.id+'.'+nodes.length;nodes.push({id,resource:rateMap[ss.rate_key]??'control',demand:rateMap[ss.rate_key]?cohort*ss.per_cta_demand:0,unit:ss.unit,dependency:[...deps],source_location:sourceOf(spec.source_node??spec.source_nodes?.[0]),overlap_rule:spec.combine+'; synchronized '+cohort+' CTA cohort, shared SM rate'});return id;};
 for(const spec of k1.synchronized_cohort_scenario.substep_specs){
  for(let rep=0;rep<(spec.repetitions??1);rep++){
   if(spec.combine==='max'){const next=spec.substeps.map(ss=>add(spec,ss,previous));previous=next;}
   else if(spec.combine==='sum_with_max_element_loop'){previous=[add(spec,spec.substeps[0],previous)];const next=spec.substeps.slice(1,-1).map(ss=>add(spec,ss,previous));previous=[add(spec,spec.substeps.at(-1),next)];}
   else for(const ss of spec.substeps)previous=[add(spec,ss,previous)];
  }
 }
 const c={key:'K1_batch_'+cohort,sheet:'K1 Cohort '+cohort,HBM_divisor:148,eta_FP16:1,eta_TF32:1,nodes};out.calendars.push(verify(c,k1.synchronized_cohort_scenario.batch_cases.find(b=>b.k===cohort).C_batch));
}
// K1 cannot be solo-CTA cycles multiplied by grid: each SM cohort shares rates.
for(let i=0;i<3;i++){const q=[14,28,56][i],full=Math.floor(q/8),tail=q%8;out.k1_excel_formulas.push(`=MAX(${full}*'K1 Cohort 8'!F${out.calendars.find(c=>c.key==='K1_batch_8').nodes.length+8}${tail?`+'K1 Cohort ${tail}'!F${out.calendars.find(c=>c.key==='K1_batch_'+tail).nodes.length+8}`:''},ROUNDUP('Work and Traffic'!E${i+6}*24840/Assumptions!$B$26,0))`);}
const k3jobs=k3.nodes.map(n=>({...n,demand:n.resource==='CVT'?n.data_volume.elements:n.resource==='TF32'?n.data_volume.FLOP:n.data_volume.bytes}));
for(const [div,eta,key,sheet]of[[148,1,'K3_8','K3 P8'],[32,1,'K3_8_div32','K3 HBM share32'],[148,.5,'K3_8_etaHalf','K3 TF32 etaHalf']]){
 const z=schedule(k3jobs,{...rates,HBM:7700000000000/1965000000/div},{etas:{TF32:eta}});
 const c={key,sheet,HBM_divisor:div,eta_FP16:1,eta_TF32:eta,nodes:z.nodes.map(n=>({id:n.id,resource:n.resource,demand:n.demand,unit:n.resource==='CVT'?'elements':n.resource==='TF32'?'FLOP':'B',dependency:n.dependency,source_location:n.source_location,overlap_rule:n.overlap_rule}))};
 out.calendars.push(verify(c,z.makespan_cycles));
}
const seen=new Set();for(const n of k3jobs){const cat=n.id.replace(/\.[0-9]+/g,'.p');if(seen.has(cat))continue;seen.add(cat);out.stage_tables.K3.push({stage:n.serial_stage.replace(/\.[0-9]+/g,'.p'),operation:cat,shape:JSON.stringify(n.parallel_work?.shape??''),demand:n.demand,resource:n.resource,dependency:n.dependency.join(';'),parallel:'N/M next partition prefetch when slot released',overlap:n.overlap_rule,source:n.source_location,confidence:n.confidence});}
function fromK4(j,key,sheet,{fixedOrder=null}={}){
 const by=new Map(j.execution.map(n=>[n.id,n]));const seq=fixedOrder??j.execution;const last={};const nodes=[];
 for(const order of seq){const n=by.get(order.id);const resource=n.resource==='TC'?'TC16':n.resource??'control';const dep=[...n.predecessors];const prior=fixedOrder?last[resource]:n.previous_resource_job;if(prior&&!dep.includes(prior))dep.push(prior);if(n.work>0)last[resource]=n.id;nodes.push({id:n.id,resource,demand:resource==='TC16'?(n.logical_flops??n.work):n.work,unit:resource==='TC16'?'FLOP':(n.datavolume?.unit??'B'),dependency:dep,eta64:n.id.startsWith('qk_done_'),source_location:'gated_delta_net_cp_prefill.py:'+n.source_lines.join('-'),overlap_rule:n.overlap??'source dependencies + fixed resource order; unknown response tails symbolic'});}
 return {key,sheet,HBM_divisor:j.config.share,eta_FP16:j.config.eta64,eta_TF32:1,nodes:topo(nodes)};
}
for(const L of [16,32,64]){const j=read(`k4_calendar_L${L}.json`);out.calendars.push(verify(fromK4(j,'K4_'+L,'K4 L'+L),j.total_cycles));}
const base4=read('k4_calendar_L16.json');for(const [opts,key,sheet]of[[{share:128},'K4_16_div128','K4 L16 HBM128'],[{eta64:.5},'K4_16_etaHalf','K4 L16 etaHalf']]){const j=buildK4Calendar(16,opts);out.calendars.push(verify(fromK4(j,key,sheet,{fixedOrder:base4.execution})));}
for(const n of k4.nodes){for(const ss of n.substepcycles??[]){if(typeof ss==='number')continue;const resource=({TMEM_copy:'TMEM',tensor16:'TC16',conversion:'CVT',SMEM_lane_request:'SMEM'})[ss.resource]??ss.resource;out.stage_tables.K4.push({stage:n.stage,operation:ss.name??n.id,shape:JSON.stringify(n.shape),demand:rates[resource]&&ss.work!==null?ss.work:0,resource:rates[resource]?resource:'control',dependency:n.dependency.join(';'),parallel:n.parallelwork,overlap:n.overlap,source:path.basename(n.source.file)+':'+n.source.lines.join('-'),confidence:n.confidence+(rates[resource]?'':' | unpriced '+ss.resource)});}}
out.sensitivity.push({kernel:'K3',L:8,variable:'HBM divisor148 →32',base_key:'K3_8',alternative_key:'K3_8_div32',note:'Active SM count does not prove HBM scales with SM coverage; this isolates that strong assumption.'},{kernel:'K3',L:8,variable:'M64 TF32 eta1 →0.5',base_key:'K3_8',alternative_key:'K3_8_etaHalf',note:'Half-datapath statement is not evidence of an exact 0.5 throughput penalty.'},{kernel:'K4',L:16,variable:'HBM divisor148 →128',base_key:'K4_16',alternative_key:'K4_16_div128',note:'Same source and fixed resource order; bandwidth allocation only.'},{kernel:'K4',L:16,variable:'QK M64 eta1 →0.5',base_key:'K4_16',alternative_key:'K4_16_etaHalf',note:'Only QK receives eta; other M128 MMAs unchanged.'});
// K2 specialist model is normalized here once the source calendar is saved.
const k2=read('k2_model.json');
if(!k2.scheduling.primary.includes('async')&&!k2.scheduling.primary.includes('overlap'))throw Error('K2 async source-primary promotion pending');
for(const c2 of k2.cases){const L=c2.work.L,nodes=c2.source_ordered_primary.events.map(n=>({id:n.id,resource:n.resource,demand:n.demand,unit:n.unit,dependency:n.dependency,source_location:n.source_location,overlap_rule:'source RAW/warp program order/stage reuse + shared resource calendar; async TMA issue; unknown tails symbolic'}));out.calendars.push(verify({key:'K2_'+L,sheet:'K2 L'+L,HBM_divisor:148,eta_FP16:1,eta_TF32:1,nodes},c2.source_ordered_primary.makespan_cycles));}
const sourceJobs=k2.cases[0].source_ordered_primary.events;const seen2=new Set();for(const n of sourceJobs){if(n.block!==undefined&&n.block>1)continue;const cat=n.id;if(seen2.has(cat))continue;seen2.add(cat);out.stage_tables.K2.push({stage:n.op,operation:n.id,shape:JSON.stringify(n.shape??n.inferred_register_tile_shape??''),demand:n.demand,resource:n.resource,dependency:n.dependency.join(';'),parallel:'CG0/CG1, alpha/TMA producer subject to shared-resource/reuse edges',overlap:'Serial within warp helper; async producer/MMA overlap only after commit/release.',source:n.source_location,confidence:n.demand===0?'source control edge; unpriced tail '+(n.symbolic_tail??''):'source work high; service assumption; '+(n.layout_evidence??'')});}
const leadCorrections=[
 {severity:'Major',issue:'K2 next TMA was made to wait previous SMEM completion',correction:'Remove false completion edge; source async producer graph is primary. Preserve serial packets as sensitivity.',evidence:'gated_delta_net_cp.py TMA producer; k2_calendar.mjs',status:'corrected, no timing input'},
 {severity:'Major',issue:'K4 S2G packet initially had reversed memory direction',correction:'SMEM read precedes HBM write; regenerate all L and sensitivities.',evidence:'K4 TMA O store; superseded_k4_v1',status:'corrected, no timing input'},
 {severity:'Major',issue:'HBM bandwidth cannot be inferred from SM coverage',correction:'Divisor148 is classified as assumption; show divisor32/128 alternatives.',evidence:'hardware_model.json; Sensitivity sheet',status:'uncertainty retained'},
 {severity:'Major',issue:'Source-level service schedule is not a unique hardware lower bound',correction:'Unknown latency/issue/SFU/barrier/cache remain symbolic; list scheduling is not optimal.',evidence:'common calendar and resource arbitration policy',status:'explicit limitation'},
 {severity:'Minor',issue:'Static SASS sites were liable to be mistaken for dynamic instructions',correction:'Keep separate static audit and source dynamic count; do not multiply all static sites by warp count.',evidence:'static_artifacts/sass_audit.json',status:'corrected'},
 {severity:'Minor',issue:'K1 negate proxy was missing in substep decomposition',correction:'Count negate result slots consistently, classification remains modeling assumption.',evidence:'k1_model.json work_per_cta and cohort substeps',status:'corrected'}
];
const reviewed=read('reviewer_findings.json');
if(reviewed.additional_primary_numeric_change_required || !reviewed.final_verdict?.startsWith('Accepted'))throw Error('Final source audit is not accepted');
out.review=reviewed.findings.map(x=>({severity:x.severity,issue:x.id+': '+x.finding,correction:x.correction_or_uncertainty,evidence:x.code_evidence.join('; '),status:x.status}));
out.review.push(...leadCorrections.filter(x=>x.issue.includes('K4 S2G')||x.issue.includes('Static SASS')||x.issue.includes('unique hardware')));
out.review_verdict=reviewed.final_verdict;
fs.writeFileSync(path.join(root,'approved_model.json'),JSON.stringify(out,null,2));
console.log(out.calendars.map(c=>({key:c.key,nodes:c.nodes.length,cycles:c.expected_cycles})));
