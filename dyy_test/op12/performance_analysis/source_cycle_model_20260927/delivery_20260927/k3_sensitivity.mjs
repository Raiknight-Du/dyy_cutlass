import fs from 'node:fs';
import path from 'node:path';
import {schedule} from './calendar.mjs';
const root=path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/,'$1'));
const x=JSON.parse(fs.readFileSync(path.join(root,'k3_model.json'),'utf8'));
const rates={HBM:x.hardware.global_HBM_bytes_per_cycle/148,SMEM:128,TMEM:512,CVT:64,TF32:x.hardware.RTF32_FLOP_per_SM_cycle,control:1};
const jobs=x.nodes.map(n=>({...n,demand:n.resource==='CVT'?n.data_volume.elements:n.resource==='TF32'?n.data_volume.FLOP:n.data_volume.bytes}));
const cases=[];
for(const divisor of [148,32])for(const eta of [1,.5]){
  const r={...rates,HBM:x.hardware.global_HBM_bytes_per_cycle/divisor};
  const z=schedule(jobs,r,{etas:{TF32:eta}});
  cases.push({HBM_divisor:divisor,eta64:eta,makespan_cycles:z.makespan_cycles,resource_work:z.resource_work,nodes:z.nodes});
  console.log(JSON.stringify({HBM_divisor:divisor,eta64:eta,makespan_cycles:z.makespan_cycles}));
}
if(cases[0].makespan_cycles!==x.numeric_scenario.makespan_cycles)throw Error('Independent scheduler mismatch');
const out={status:'Independent lead sensitivity; no measured timing input',policy:'Same source DAG, same packet ordering and resource arbitration; only allocation changed',cases};
fs.writeFileSync(path.join(root,'k3_sensitivity.json'),JSON.stringify(out,null,2));
