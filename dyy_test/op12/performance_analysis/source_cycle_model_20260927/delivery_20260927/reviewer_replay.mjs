import fs from 'node:fs';
import {schedule} from './calendar.mjs';
const m=JSON.parse(fs.readFileSync(new URL('./k2_model.json',import.meta.url),'utf8'));
const f=1965e6;
const rates={HBM:7.7e12/f/148,SMEM:128,TMEM:512,CVT:64,SIMT:128,TC16:2.25e15/(148*f),control:1};
for(const c of m.cases){
 const jobs=c.source_ordered_primary.events.map(n=>({...n,started:false,dependency:[...n.dependency]}));
 const by=new Map(jobs.map(n=>[n.id,n]));
 const add=(id,...deps)=>by.get(id).dependency.push(...deps.filter(Boolean));
 for(let b=0;b<c.work.L;b++){
  if(b){add(`alpha_${b}_issue`,`alpha_${b-1}_ready`);add(`x_mat_${b}_tmem_read`,`m_scale_${b-1}_done`);}
  add(`m_coeff_${b}`,b?`z_convert_${b}_ready`:`x_mat_${b}_ready`);
  add(`n_coeff_${b}`,`y_mma_${b}_done`);
  add(`process_y_${b}_v_ldsm`,`y_mma_${b}_done`,`n_coeff_${b}`);
  add(`process_y_${b}_alpha_smem`,`y_mma_${b}_done`,`n_coeff_${b}`);
  add(`process_y_${b}_tmem_read`,`n_coeff_${b}`);
 }
 const z=schedule(jobs,rates);
 console.log(JSON.stringify({L:c.work.L,original:c.source_ordered_primary.makespan_cycles,corrected_phase_edges:z.makespan_cycles,numeric_change:z.makespan_cycles-c.source_ordered_primary.makespan_cycles}));
}
