import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
const root=path.dirname(fileURLToPath(import.meta.url));
const read=n=>JSON.parse(fs.readFileSync(path.join(root,n),'utf8'));
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const freeze=read('SOURCE_MODEL_FREEZE.json');
for(const entry of freeze.files)if(sha(path.join(root,entry.path))!==entry.sha256)throw Error('Model changed after freeze: '+entry.path);
const model=read('approved_model.json');
const median=x=>{const a=[...x].sort((a,b)=>a-b);return a[Math.floor(a.length/2)];};
const trials=[],sources=[];
for(let trial=1;trial<=3;trial++){
 const relative='../evidence_gap_ablation_20260920/cp_stage_same_round/trial'+trial+'.json';
 const original=path.join(root,relative),p=fs.existsSync(original)?original:path.join(root,'event_raw','trial'+trial+'.json');sources.push({path:relative,actual_read_path:path.relative(root,p),sha256:sha(p)});
 for(const record of JSON.parse(fs.readFileSync(p,'utf8'))){
  if(record.dtype!=='float16'||record.initial!==true||record.final!==true||record.batch!==1||record.output_heads!==16)throw Error('Event scope mismatch');
  trials.push({...record,source:relative});
 }
}
const f=1965000000, cases=[];
for(const [i,S]of [8192,16384,32768].entries()){
 const C=[1024,2048,4096][i],L=C/64;
 const rows=trials.filter(x=>x.total_tokens===S&&x.cp_chunk_len===C);
 if(rows.length!==3)throw Error('Missing three trials');
 const cal=key=>model.calendars.find(c=>c.key===key).expected_cycles;
 const q=Math.ceil(16*S/64/148),tail=q%8;
 const k1=Math.floor(q/8)*cal('K1_batch_8')+(tail?cal('K1_batch_'+tail):0);
 const stages=[];
 for(const [kernel,stage,cycles]of [['K1','t',k1],['K2','mn',cal('K2_'+L)],['K3','fixup',cal('K3_8')],['K4','final',cal('K4_'+L)]]){
  const us=rows.map(x=>x.stage_timings[stage].median_ms*1000),event=median(us),theory=cycles/f*1e6;
  stages.push({kernel,source_calendar_cycles:cycles,theory_us_at_reference_clock:theory,event_median_of_three_trial_medians_us:event,event_reference_cycles:event*f/1e6,residual_us:event-theory,residual_reference_cycles:event*f/1e6-cycles,event_to_theory_ratio:event/theory,trial_medians_us:us,trial_cv_percent:rows.map(x=>x.stage_timings[stage].cv_pct)});
 }
 cases.push({S,C,L,P:8,stages,end_to_end_event_median_of_three_trial_medians_us:median(rows.map(x=>x.timing.median_ms*1000)),source_four_stage_sum_cycles:stages.reduce((s,x)=>s+x.source_calendar_cycles,0),independent_stage_event_median_sum_us:stages.reduce((s,x)=>s+x.event_median_of_three_trial_medians_us,0),finite_output_and_state:rows.every(x=>x.finite_output&&x.finite_state),clock_snapshots_MHz:rows.map(x=>[x.telemetry_before.sm_clock_mhz,x.telemetry_after.sm_clock_mhz])});
}
const result={checked_at_utc:new Date().toISOString(),model_frozen_at_utc:freeze.frozen_at_utc,model_freeze_sha256:sha(path.join(root,'SOURCE_MODEL_FREEZE.json')),model_hashes_verified_before_target_timing_read:true,fit_model_to_measurements:false,reference_clock_Hz:f,measurement_method:'Archived CUDA Event; independent stage Python-wrapper calls use two warmup/twenty events on repeated fixed tensors; end-to-end uses twenty warmup/hundred events and rotating Q/K/V. Three trial medians. No NCU duration used.',limitations:['Several archived independent-stage CVs exceed5%; only trend/order-of-magnitude sanity check, not tight validation.','Clock snapshots1965MHz are not continuous clock traces.','Independent-stage median sums are not true end-to-end latency.','Repeated inputs/workspace permit cache reuse; model all-request-HBM scenario does not measure this.','No target ablation or profiling counter is used to retune constants.'],sources,cases};
fs.writeFileSync(path.join(root,'event_sanity_check.json'),JSON.stringify(result,null,2));
console.log(JSON.stringify(cases.map(c=>({S:c.S,Event:c.stages.map(x=>Number(x.event_median_of_three_trial_medians_us.toFixed(3))),residual_us:c.stages.map(x=>Number(x.residual_us.toFixed(3))),E2E:c.end_to_end_event_median_of_three_trial_medians_us})),null,2));
