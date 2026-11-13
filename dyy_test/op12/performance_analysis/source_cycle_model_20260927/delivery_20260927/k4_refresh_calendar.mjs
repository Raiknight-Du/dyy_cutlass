import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildK4Calendar } from './k4_calendar.mjs';

// Mechanically regenerate only these task-owned calendar artifacts from the corrected model.
const base = path.dirname(fileURLToPath(import.meta.url));
const primary = [16,32,64].map(L => buildK4Calendar(L));
const fixed = [];
const raw = [];
const options = [{gateDemand:'per_warp_reuse_hypothesis'}, {share:128}, {gateDemand:'per_warp_reuse_hypothesis',share:128}, {eta64:0.5}, {includeTensorFeed:true}, {gateDemand:'per_warp_reuse_hypothesis',includeTensorFeed:true}];
for (const p of primary) {
  const destination = path.join(base,'k4_calendar_L'+p.L+'.json');
  const previous = JSON.parse(fs.readFileSync(destination,'utf8'));
  const metadata = {...previous.metadata};
  metadata.common_schema_defaults = {...metadata.common_schema_defaults, hardwareassumption:'Lead scenario A; perCTA HBM/148; integer ceil service; G2S packetHBMread→SMEMwrite; S2G packetSMEMread→HBMwrite; unknown latency0 onlynumeric; gate bytes source-request conservative; tensor operandfeed covered by tensor peak; terminal state reads removed in clean SASS.'};
  metadata.correction = {version:2,reason:'Output TMA S2G must read SMEM before writing GMEM; direction correction independent of timing.',superseded_v1:'superseded_k4_v1',measured_timings_used:false};
  const execution = p.execution.map(t => {
    const {declaration,dominant_predecessors,...keep}=t;
    const unit = t.resource===null?'control':({HBM:'bytes',SMEM:'bytes',TMEM:'bytes',CVT:'elements',SIMT:'scalar_operations',TC:'flops/eta'})[t.resource];
    return {...keep,substage:t.id,dependency:t.predecessors,datavolume:{work:t.work,unit},throughput:t.rate,cycleformula:t.resource?'CEILING(work/rate,1) + symbolic_extra':t.symbolic_extra||'0',substepcycles:t.duration,overlap:'Resource calendars enforce nonoverlap; source order represented in dependency; packet conservatism explicit.'};
  });
  fs.writeFileSync(destination,JSON.stringify({...p,metadata,execution},null,2)+'\n');
  for (const opt of options) {
    const v=buildK4Calendar(p.L,opt);
    const {execution:ve,critical_path,...summary}=v;
    raw.push({...summary,first_output_stored:v.first_output_stored,first_pair_stored:v.first_pair_stored,last_pair_store_interval:v.last_pair_store_interval});
    if (!opt.includeTensorFeed) {
      const vm=new Map(ve.map(t=>[t.id,t])),done=new Map(),last={},rs={};
      for (const bt of p.execution) {
        const t=vm.get(bt.id);
        const ready=t.predecessors.reduce((s,id)=>Math.max(s,done.get(id).finish),0);
        const prev=t.resource?last[t.resource]:null;
        const start=Math.max(ready,prev?done.get(prev).finish:0),finish=start+t.duration;
        done.set(t.id,{...t,start,finish});
        if(t.resource){last[t.resource]=t.id;rs[t.resource]=finish;}
      }
      fixed.push({L:p.L,config:{...v.config,policy:'freeze primary v2 per-resource job order; recompute max-plus timestamps'},total_cycles:done.get('finish').finish,total_us:done.get('finish').finish/1965,first_output_stored:done.get('o_store_done_0').finish,first_pair_stored:done.get('o_store_done_1').finish,last_pair_store_interval:done.get('o_store_done_'+(p.L-1)).finish-done.get('o_store_done_'+(p.L-3)).finish,resource_busy_cycles:v.resource_busy_cycles,resource_last_finish:rs});
    }
  }
}
const summary={schema:'K4 calendar summary v2',primary:primary.map(({execution,critical_path,...p})=>p),fixed_primary_resource_order_sensitivities:fixed,gate_redispatch_sensitivities:raw.filter(r=>r.config.gateDemand==='per_warp_reuse_hypothesis'&&!r.config.includeTensorFeed),tensor_feed_redispatch_sensitivities:raw.filter(r=>r.config.includeTensorFeed),direction_correction:{reason:'S2G outputSMEMread precedesHBMwrite; v1 reversed that dependency. Corrected independently of timing.',superseded_v1_primary:[{L:16,cycles:57274},{L:32,cycles:106594},{L:64,cycles:205234}],archive:'superseded_k4_v1',work_budgets_unchanged:true},warning:'Fixed-order rate sensitivities isolate throughput/work changes; re-dispatch variants use different queue arbitration and are not optimal runtime predictions.'};
fs.writeFileSync(path.join(base,'k4_calendar_summary.json'),JSON.stringify(summary,null,2)+'\n');
for (const p of primary) console.log(JSON.stringify({L:p.L,cycles:p.total_cycles,nodes:p.task_count,validation:p.validation}));

